import random
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sre_solvers import (
    PathCBimatrixSreSolver,
    PathMcpNPlayerSreSolver,
    robust_exploitability,
    robust_policy_values,
)
from sre_solvers.nfg_transformer.torch_utils import robust_policy_values_torch


def _linear_schedule(start, end, step, total_steps):
    if total_steps <= 1:
        return float(end)
    fraction = min(max(step, 0) / float(total_steps - 1), 1.0)
    return float(start + fraction * (end - start))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        joint_actions,
        joint_rewards,
        next_state,
        done,
        action_masks=None,
        next_action_masks=None,
    ):
        self.buffer.append(
            (
                state,
                joint_actions,
                joint_rewards,
                next_state,
                done,
                action_masks,
                next_action_masks,
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        normalized = []
        for transition in batch:
            if len(transition) == 5:
                state, actions, rewards, next_state, done = transition
                normalized.append((state, actions, rewards, next_state, done, None, None))
            else:
                normalized.append(transition)
        states, actions, rewards, next_states, dones, action_masks, next_action_masks = zip(
            *normalized
        )
        return states, actions, rewards, next_states, dones, action_masks, next_action_masks

    def __len__(self):
        return len(self.buffer)


@dataclass
class DuelingDoubleDqnSreAgentConfig:
    # Index of the controlled agent when using single-agent action helpers.
    agent_id: int = 0
    # Flattened observation dimension expected by the Q network.
    obs_dim: int = 4 # 2 agents, 2 coordinates each
    # Number of agents represented in the joint-action game.
    num_agents: int = 2
    # Number of discrete actions available to each agent.
    num_actions: int = 4
    # Filesystem path to the PATH wrapper shared library.
    pathwrap_path: str = "pathwrap.so"
    # Current strategic robustness radius used by SRE solves.
    epsilon_robust: float = 1.0
    # Current epsilon-greedy exploration probability.
    epsilon_explore: float = 1.0
    # Learning rate for the Q-network optimizer.
    lr: float = 3e-4
    # Discount factor for Bellman targets.
    gamma: float = 0.9
    # Exponential decay factor for robust epsilon scheduling.
    decay_rate: float = 0.999
    # Maximum number of transitions stored in replay.
    buffer_size: int = 10000
    # Default replay minibatch size.
    batch_size: int = 16
    # Minimum replay size before gradient updates begin.
    learning_starts: int = 1000
    # Maximum gradient norm used for clipping.
    grad_clip_norm: float = 10.0
    # Whether to use CUDA when it is available.
    use_gpu: bool = True
    # N-player solvers are called inside the DQN training loop; keep the
    # default approximate budget small. Set these back to 20/True for
    # exhaustive offline solver comparisons.
    # Number of random solver restarts for SRE stage games.
    sre_num_repeats: int = 4
    # Whether solver starts should include pure-strategy profiles.
    sre_include_pure_starts: bool = False
    # Number of environment updates between gradient updates.
    train_every: int = 1
    # Optional injected SRE solver instance.
    sre_solver: Any = None
    # Q-network architecture variant.
    network_type: str = "joint_output"
    # Soft target-network update rate; None uses hard updates.
    target_tau: Optional[float] = None
    # Number of gradient updates between hard target-network syncs.
    target_update_steps: int = 100
    # Number of gradient updates between fresh target-equilibrium solves.
    target_equilibrium_update_steps: int = 4
    # Initial epsilon-greedy exploration probability.
    action_epsilon_start: float = 1.0
    # Final epsilon-greedy exploration probability.
    action_epsilon_end: float = 0.05
    # Fraction of training episodes used for exploration decay.
    action_epsilon_decay_fraction: float = 0.5
    # Stage-game solver name used by factory-driven callers.
    sre_solver_name: str = "path_c_pool"
    # Worker count for pooled SRE solvers.
    sre_solver_workers: int = 8
    # Multiprocessing start method for pooled SRE solvers.
    sre_solver_start_method: Optional[str] = None
    # Hidden-layer widths for Q-network MLPs.
    q_hidden_dims: tuple = (128, 128)
    # Initial robust epsilon value used by schedules.
    epsilon_robust_initial: float = 1.0
    # Robust epsilon schedule type.
    epsilon_schedule: str = "exponential"
    # Whether to cache SRE policies for repeated stage games.
    sre_policy_cache_enabled: bool = True
    # Maximum number of cached SRE policy entries.
    sre_policy_cache_size: int = 4096
    # Decimal precision used when keying Q-tensor policy cache entries.
    sre_policy_cache_round_digits: int = 6
    # Decimal precision used when keying state-level policy cache entries.
    sre_state_cache_round_digits: int = 4
    # Whether approximate cache reuse is allowed.
    sre_approx_cache_enabled: bool = True
    # Exploitability tolerance for accepting cached policies.
    sre_cache_exploitability_tol: float = 1e-3
    # Exploitability tolerance passed to compatible SRE solvers.
    sre_solver_exploitability_tol: float = 1e-4
    # Exploitability tolerance for accepting approximate solver candidates.
    sre_approx_accept_tol: float = 1e-2
    # Whether compatible SRE solvers may stop after a good candidate.
    sre_solver_early_exit: bool = True
    # Criterion used to choose among multiple SRE candidates.
    sre_candidate_selection: str = "robust_exploitability"
    # Whether approximate candidates are filtered by exploitability.
    sre_exploitability_filter_enabled: bool = False
    # Value mode used when computing SRE Bellman targets.
    sre_target_value_mode: str = "robust"
    # Whether to fall back to uniform policies when SRE solving fails.
    sre_uniform_fallback_enabled: bool = False


_CHECKPOINT_CONFIG_EXCLUDE = {"sre_solver"}


def _checkpoint_config_dict(config):
    return {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.name not in _CHECKPOINT_CONFIG_EXCLUDE
    }


def _restore_checkpoint_config(config, payload):
    for field in fields(config):
        if field.name in _CHECKPOINT_CONFIG_EXCLUDE:
            continue
        if field.name in payload:
            setattr(config, field.name, payload[field.name])


def _make_feature_mlp(input_dim, hidden_dims):
    hidden_dims = tuple(int(dim) for dim in hidden_dims)
    if any(dim <= 0 for dim in hidden_dims):
        raise ValueError(f"Hidden dimensions must be positive, got {hidden_dims}.")
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
        prev_dim = hidden_dim
    return nn.Sequential(*layers), prev_dim


class DuelingJointQNetwork(nn.Module):
    """
    Dueling network over joint actions for N-player games.
    Output shape: [batch, num_actions, ..., num_actions, num_agents].
    """

    def __init__(self, obs_dim, num_actions, num_agents, hidden_dims=(128, 128)):
        super().__init__()

        self.num_actions = num_actions
        self.num_agents = num_agents
        self.joint_action_count = num_actions ** num_agents
        self.output_shape = [num_actions] * num_agents + [num_agents]

        self.feature, feature_dim = _make_feature_mlp(obs_dim, hidden_dims)
        self.value_head = nn.Linear(feature_dim, num_agents)
        self.adv_head = nn.Linear(feature_dim, self.joint_action_count * num_agents)

    def forward(self, state):
        features = self.feature(state)
        value = self.value_head(features)  # [B, N]
        advantage = self.adv_head(features).view(
            -1, self.joint_action_count, self.num_agents
        )  # [B, |A_joint|, N]

        q_joint = value.unsqueeze(1) + (
            advantage - advantage.mean(dim=1, keepdim=True)
        )
        return q_joint.view(-1, *self.output_shape)


class _DuelingPayoffHead(nn.Module):
    def __init__(self, feature_dim, joint_action_count):
        super().__init__()
        self.value_head = nn.Linear(feature_dim, 1)
        self.adv_head = nn.Linear(feature_dim, joint_action_count)

    def forward(self, features):
        value = self.value_head(features)  # [B, 1]
        advantage = self.adv_head(features)  # [B, |A_joint|]
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DuelingPerAgentJointQNetwork(nn.Module):
    """
    Per-agent critics over joint actions.

    Output shape is kept identical to DuelingJointQNetwork:
    [batch, num_actions, ..., num_actions, num_agents].
    """

    def __init__(self, obs_dim, num_actions, num_agents, hidden_dims=(128, 128)):
        super().__init__()

        self.num_actions = num_actions
        self.num_agents = num_agents
        self.joint_action_count = num_actions ** num_agents
        self.output_shape = [num_actions] * num_agents + [num_agents]
        self.critics = nn.ModuleList(
            [self._make_critic(obs_dim, hidden_dims) for _ in range(num_agents)]
        )

    def _make_critic(self, obs_dim, hidden_dims):
        feature, feature_dim = _make_feature_mlp(obs_dim, hidden_dims)
        return nn.Sequential(
            feature,
            _DuelingPayoffHead(feature_dim, self.joint_action_count),
        )

    def forward(self, state):
        q_by_agent = [critic(state) for critic in self.critics]
        q_joint = torch.stack(q_by_agent, dim=-1)  # [B, |A_joint|, N]
        return q_joint.view(-1, *self.output_shape)


class DuelingSharedTrunkPerAgentJointQNetwork(nn.Module):
    """
    Shared state encoder with separate per-agent dueling payoff heads.

    This shares feature extraction while retaining agent-specific payoff heads.
    """

    def __init__(self, obs_dim, num_actions, num_agents, hidden_dims=(128, 128)):
        super().__init__()

        self.num_actions = num_actions
        self.num_agents = num_agents
        self.joint_action_count = num_actions ** num_agents
        self.output_shape = [num_actions] * num_agents + [num_agents]
        self.feature, feature_dim = _make_feature_mlp(obs_dim, hidden_dims)
        self.heads = nn.ModuleList(
            [
                _DuelingPayoffHead(feature_dim, self.joint_action_count)
                for _ in range(num_agents)
            ]
        )

    def forward(self, state):
        features = self.feature(state)
        q_by_agent = [head(features) for head in self.heads]
        q_joint = torch.stack(q_by_agent, dim=-1)  # [B, |A_joint|, N]
        return q_joint.view(-1, *self.output_shape)


def make_q_network(
    obs_dim,
    num_actions,
    num_agents,
    network_type="joint_output",
    hidden_dims=(128, 128),
):
    if network_type == "joint_output":
        return DuelingJointQNetwork(obs_dim, num_actions, num_agents, hidden_dims)
    if network_type == "per_agent_independent":
        return DuelingPerAgentJointQNetwork(
            obs_dim, num_actions, num_agents, hidden_dims
        )
    if network_type == "shared_trunk_separate_heads":
        return DuelingSharedTrunkPerAgentJointQNetwork(
            obs_dim, num_actions, num_agents, hidden_dims
        )
    raise ValueError(f"Unknown network_type: {network_type}")


class DuelingDoubleDqnSreAgent:
    """
    Dueling Double DQN agent that uses SRE policies for action selection and targets.
    """

    def __init__(self, config: DuelingDoubleDqnSreAgentConfig):
        self.config = config
        self.q_tensor_shape = tuple(
            [config.num_actions] * config.num_agents + [config.num_agents]
        )

        self.initial_epsilon_robust = float(config.epsilon_robust)
        self.initial_epsilon_explore = float(config.epsilon_explore)
        self.config.sre_include_pure_starts = bool(config.sre_include_pure_starts)
        self.config.train_every = max(1, int(config.train_every))
        self._update_calls = 0

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.q_net = make_q_network(
            config.obs_dim,
            config.num_actions,
            config.num_agents,
            network_type=config.network_type,
            hidden_dims=config.q_hidden_dims,
        ).to(self.device)
        self.target_net = make_q_network(
            config.obs_dim,
            config.num_actions,
            config.num_agents,
            network_type=config.network_type,
            hidden_dims=config.q_hidden_dims,
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.update_times = []
        self.sre_solve_time_count = 0
        self.sre_solve_time_sum = 0.0
        self.sre_solve_time_sumsq = 0.0
        self.sre_solve_time_min = None
        self.sre_solve_time_max = None
        self._sre_policy_cache = OrderedDict()
        self._sre_state_policy_keys = {}
        self._last_sre_cache_key = None
        self.sre_cache_exact_hits = 0
        self.sre_cache_approx_hits = 0
        self.sre_cache_misses = 0
        self.sre_cache_evictions = 0
        self.sre_uniform_fallback_count = 0
        self.sre_candidate_return_count = 0
        self.sre_solver_failure_warm_start_reuses = 0

        if config.sre_solver is None:
            if config.num_agents == 2:
                sre_solver = PathCBimatrixSreSolver(pathwrap_path=config.pathwrap_path)
            else:
                sre_solver = PathMcpNPlayerSreSolver(pathwrap_path=config.pathwrap_path)
            self.sre_solver = sre_solver
        else:
            self.sre_solver = config.sre_solver

    def _record_sre_solve_time(self, duration, count=1):
        count = max(1, int(count))
        per_solve_duration = duration / count
        self.sre_solve_time_count += count
        self.sre_solve_time_sum += duration
        self.sre_solve_time_sumsq += count * per_solve_duration * per_solve_duration
        if (
            self.sre_solve_time_min is None
            or per_solve_duration < self.sre_solve_time_min
        ):
            self.sre_solve_time_min = per_solve_duration
        if (
            self.sre_solve_time_max is None
            or per_solve_duration > self.sre_solve_time_max
        ):
            self.sre_solve_time_max = per_solve_duration

    def get_sre_solve_time_summary(self):
        count = self.sre_solve_time_count
        if count == 0:
            return {
                "count": 0,
                "mean_seconds": None,
                "min_seconds": None,
                "max_seconds": None,
                "std_seconds": None,
                "mean_microseconds": None,
                "min_microseconds": None,
                "max_microseconds": None,
                "std_microseconds": None,
            }
        mean = self.sre_solve_time_sum / count
        variance = max(self.sre_solve_time_sumsq / count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        return {
            "count": int(count),
            "mean_seconds": float(mean),
            "min_seconds": float(self.sre_solve_time_min),
            "max_seconds": float(self.sre_solve_time_max),
            "std_seconds": std,
            "mean_microseconds": float(mean * 1_000_000.0),
            "min_microseconds": float(self.sre_solve_time_min * 1_000_000.0),
            "max_microseconds": float(self.sre_solve_time_max * 1_000_000.0),
            "std_microseconds": float(std * 1_000_000.0),
        }

    def get_sre_cache_summary(self):
        exact_hits = int(self.sre_cache_exact_hits)
        approx_hits = int(self.sre_cache_approx_hits)
        misses = int(self.sre_cache_misses)
        requests = exact_hits + approx_hits + misses
        path_avoided = exact_hits + approx_hits
        return {
            "enabled": bool(self._sre_policy_cache_active()),
            "config_enabled": bool(self.config.sre_policy_cache_enabled),
            "disabled_by_solver": bool(self._sre_solver_bypasses_policy_cache()),
            "approx_enabled": bool(self.config.sre_approx_cache_enabled),
            "entries": int(len(self._sre_policy_cache)),
            "max_entries": int(self.config.sre_policy_cache_size),
            "requests": int(requests),
            "exact_hits": exact_hits,
            "approx_hits": approx_hits,
            "misses": misses,
            "hit_rate": None if requests == 0 else float(path_avoided / requests),
            "path_solves_avoided": int(path_avoided),
            "evictions": int(self.sre_cache_evictions),
            "uniform_fallbacks": int(self.sre_uniform_fallback_count),
            "candidate_returned": int(self.sre_candidate_return_count),
            "solver_failure_warm_start_reuses": int(
                self.sre_solver_failure_warm_start_reuses
            ),
            "cache_round_digits": int(self.config.sre_policy_cache_round_digits),
            "state_round_digits": int(self.config.sre_state_cache_round_digits),
            "approx_exploitability_tol": float(self.config.sre_cache_exploitability_tol),
            "solver_exploitability_tol": float(self.config.sre_solver_exploitability_tol),
            "solver_approx_accept_tol": float(self.config.sre_approx_accept_tol),
            "candidate_selection": str(self.config.sre_candidate_selection),
            "exploitability_filter_enabled": bool(self.config.sre_exploitability_filter_enabled),
            "target_value_mode": str(self.config.sre_target_value_mode),
            "uniform_fallback_enabled": bool(self.config.sre_uniform_fallback_enabled),
            "target_equilibrium_update_steps": int(self.config.target_equilibrium_update_steps),
        }

    def _sre_solver_bypasses_policy_cache(self):
        return bool(getattr(self.sre_solver, "bypass_deep_srq_policy_cache", False))

    def _sre_solver_supports_torch_action_solve(self):
        return bool(
            self._sre_solver_bypasses_policy_cache()
            and hasattr(self.sre_solver, "solve_batch_torch")
        )

    def _sre_solver_supports_torch_policy_solve(self):
        return bool(
            self._sre_solver_bypasses_policy_cache()
            and hasattr(self.sre_solver, "solve_policy_batch_torch")
        )

    def _sre_policy_cache_active(self):
        return bool(
            self.config.sre_policy_cache_enabled
            and not self._sre_solver_bypasses_policy_cache()
        )

    def _state_to_vector(self, state):
        vector = np.asarray(state, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.config.obs_dim:
            raise ValueError(
                f"Expected state vector length {self.config.obs_dim}, got {vector.shape[0]}."
            )
        return vector

    def _normalize_policy(self, policy):
        p = np.asarray(policy, dtype=np.float32)
        p = np.clip(p, 0.0, None)
        s = float(p.sum())
        if s <= 0.0:
            size = int(p.shape[0]) if p.ndim == 1 and p.shape[0] > 0 else self.config.num_actions
            return np.full(size, 1.0 / size, dtype=np.float32)
        return p / s

    def _uniform_policies(self):
        u = np.full(self.config.num_actions, 1.0 / self.config.num_actions, dtype=np.float32)
        return [u.copy() for _ in range(self.config.num_agents)]

    def _normalize_action_masks(self, action_masks):
        if action_masks is None:
            return None
        masks = [np.asarray(mask, dtype=bool).reshape(-1) for mask in action_masks]
        if len(masks) != self.config.num_agents:
            raise ValueError(
                f"Expected {self.config.num_agents} action masks, got {len(masks)}."
            )
        normalized = []
        for agent_id, mask in enumerate(masks):
            if mask.shape[0] != self.config.num_actions:
                raise ValueError(
                    f"Expected action mask length {self.config.num_actions} for agent "
                    f"{agent_id}, got {mask.shape[0]}."
                )
            mask = mask.copy()
            if not np.any(mask):
                mask[:] = True
            normalized.append(mask)
        return normalized

    def _normalize_action_masks_batch(self, action_masks_batch, batch_size):
        if action_masks_batch is None:
            return [None] * int(batch_size)
        if isinstance(action_masks_batch, np.ndarray):
            if action_masks_batch.ndim != 3:
                raise ValueError(
                    "action_masks_batch ndarray must have shape [B, N, A], "
                    f"got {action_masks_batch.shape}."
                )
            action_masks_batch = list(action_masks_batch)
        masks_list = list(action_masks_batch)
        if len(masks_list) != int(batch_size):
            raise ValueError(
                f"Expected {batch_size} action-mask entries, got {len(masks_list)}."
            )
        return [self._normalize_action_masks(masks) for masks in masks_list]

    def _sample_action_with_mask(self, mask):
        if mask is None:
            return int(np.random.choice(self.config.num_actions))
        choices = np.flatnonzero(mask)
        if choices.size == 0:
            return int(np.random.choice(self.config.num_actions))
        return int(np.random.choice(choices))

    def _masked_uniform_policies(self, action_masks):
        if not bool(self.config.sre_uniform_fallback_enabled):
            raise RuntimeError(
                "SRE solver failed and uniform fallback is disabled for a masked action set."
            )
        self.sre_uniform_fallback_count += 1
        policies = []
        for mask in action_masks:
            policy = np.zeros(self.config.num_actions, dtype=np.float32)
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                policy[:] = 1.0 / self.config.num_actions
            else:
                policy[valid] = 1.0 / valid.size
            policies.append(policy)
        return policies

    def _slice_q_tensor_for_masks(self, q_tensor, action_masks):
        q_tensor = np.asarray(q_tensor, dtype=np.float32)
        indices = [np.flatnonzero(mask) for mask in action_masks]
        for agent_id, valid in enumerate(indices):
            if valid.size == 0:
                raise ValueError(f"Action mask for agent {agent_id} has no valid actions.")
        return q_tensor[np.ix_(*indices)], indices

    def _expand_reduced_policies(self, reduced_policies, action_masks, action_indices):
        expanded = []
        for agent_id, (policy, mask, valid_indices) in enumerate(
            zip(reduced_policies, action_masks, action_indices)
        ):
            policy = self._normalize_policy(policy)
            if policy.shape[0] != valid_indices.size:
                raise ValueError(
                    "Solver returned malformed masked policy for agent "
                    f"{agent_id}: expected length {valid_indices.size}, got {policy.shape[0]}."
                )
            full_policy = np.zeros(self.config.num_actions, dtype=np.float32)
            full_policy[valid_indices] = policy
            if not np.any(full_policy):
                full_policy[np.flatnonzero(mask)] = 1.0 / max(1, int(np.sum(mask)))
            expanded.append(full_policy)
        return expanded

    def _fallback_policies(self, reason):
        if not bool(self.config.sre_uniform_fallback_enabled):
            raise RuntimeError(
                "SRE solver failed and uniform fallback is disabled. "
                f"{reason}"
            )
        self.sre_uniform_fallback_count += 1
        return self._uniform_policies()

    def _warm_start_policies_or_fallback(self, warm_policies, reason):
        if self._policies_valid(warm_policies):
            self.sre_solver_failure_warm_start_reuses += 1
            return [self._normalize_policy(policy) for policy in warm_policies], False
        return self._fallback_policies(reason), False

    def _sre_batch_key(self, q_tensor):
        q_key = np.ascontiguousarray(
            np.round(q_tensor, int(self.config.sre_policy_cache_round_digits)),
            dtype=np.float32,
        )
        solver_name = getattr(self.sre_solver, "name", type(self.sre_solver).__name__)
        return (
            solver_name,
            round(float(self.config.epsilon_robust), 6),
            int(self.config.sre_num_repeats),
            bool(self.config.sre_include_pure_starts),
            q_key.tobytes(),
        )

    def _sre_cache_key(self, q_tensor):
        # Backward-compatible alias retained for older tests/notebooks.
        return self._sre_batch_key(q_tensor)

    def _sre_state_key(self, state):
        if state is None:
            return None
        state_vec = self._state_to_vector(state)
        state_key = np.ascontiguousarray(
            np.round(state_vec, int(self.config.sre_state_cache_round_digits)),
            dtype=np.float32,
        )
        return state_key.tobytes()

    @staticmethod
    def _copy_policies(policies):
        return [np.asarray(policy, dtype=np.float32).copy() for policy in policies]

    def _policies_valid(self, policies):
        return (
            policies is not None
            and len(policies) == self.config.num_agents
            and all(
                np.asarray(policy).shape[0] == self.config.num_actions
                for policy in policies
            )
        )

    def _store_sre_policy_cache(self, cache_key, policies, *, state_key=None, metadata=None):
        if not self._sre_policy_cache_active():
            return
        max_entries = max(0, int(self.config.sre_policy_cache_size))
        if max_entries <= 0 or not self._policies_valid(policies):
            return
        entry = {
            "policies": self._copy_policies(policies),
            "state_key": state_key,
            "metadata": dict(metadata or {}),
            "stored_at_update": int(self._update_calls),
            "uses": 0,
        }
        if cache_key in self._sre_policy_cache:
            self._sre_policy_cache.move_to_end(cache_key)
        self._sre_policy_cache[cache_key] = entry
        self._last_sre_cache_key = cache_key
        if state_key is not None:
            self._sre_state_policy_keys[state_key] = cache_key
        while len(self._sre_policy_cache) > max_entries:
            evicted_key, evicted_entry = self._sre_policy_cache.popitem(last=False)
            self.sre_cache_evictions += 1
            evicted_state_key = evicted_entry.get("state_key")
            if (
                evicted_state_key is not None
                and self._sre_state_policy_keys.get(evicted_state_key) == evicted_key
            ):
                self._sre_state_policy_keys.pop(evicted_state_key, None)
            if self._last_sre_cache_key == evicted_key:
                self._last_sre_cache_key = next(
                    reversed(self._sre_policy_cache), None
                )

    def _cache_candidate_keys(self, cache_key, state_key):
        candidates = []
        if state_key is not None:
            state_cache_key = self._sre_state_policy_keys.get(state_key)
            if state_cache_key is not None and state_cache_key != cache_key:
                candidates.append(state_cache_key)
        if self._last_sre_cache_key is not None and self._last_sre_cache_key != cache_key:
            candidates.append(self._last_sre_cache_key)
        seen = set()
        ordered = []
        for candidate in candidates:
            if candidate not in seen and candidate in self._sre_policy_cache:
                ordered.append(candidate)
                seen.add(candidate)
        return ordered

    def _lookup_sre_policy_cache(self, q_tensor, cache_key, state_key=None, *, allow_reuse=True):
        if not self._sre_policy_cache_active():
            return None, None

        entry = self._sre_policy_cache.get(cache_key)
        if entry is not None:
            self._sre_policy_cache.move_to_end(cache_key)
            entry["uses"] = int(entry.get("uses", 0)) + 1
            self._last_sre_cache_key = cache_key
            policies = self._copy_policies(entry["policies"])
            if allow_reuse:
                self.sre_cache_exact_hits += 1
                return policies, self._copy_policies(policies)
            return None, policies

        warm_policies = None
        best_gap = None
        for candidate_key in self._cache_candidate_keys(cache_key, state_key):
            candidate = self._sre_policy_cache[candidate_key]
            candidate_policies = self._copy_policies(candidate["policies"])
            if warm_policies is None:
                warm_policies = candidate_policies
            if allow_reuse and self.config.sre_approx_cache_enabled:
                gap, _, _ = robust_exploitability(
                    q_tensor,
                    candidate_policies,
                    self.config.epsilon_robust,
                )
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    warm_policies = candidate_policies
                if gap <= float(self.config.sre_cache_exploitability_tol):
                    candidate["uses"] = int(candidate.get("uses", 0)) + 1
                    self._sre_policy_cache.move_to_end(candidate_key)
                    self.sre_cache_approx_hits += 1
                    self._store_sre_policy_cache(
                        cache_key,
                        candidate_policies,
                        state_key=state_key,
                        metadata={
                            "source": "verified_cached_policy",
                            "robust_exploitability": float(gap),
                        },
                    )
                    return self._copy_policies(candidate_policies), self._copy_policies(candidate_policies)

        self.sre_cache_misses += 1
        return None, warm_policies

    def _call_sre_solver(self, q_tensor, *, initial_policies=None):
        kwargs = {
            "epsilon": self.config.epsilon_robust,
            "num_repeats": self.config.sre_num_repeats,
            "include_pure_starts": self.config.sre_include_pure_starts,
            "exploitability_tol": self.config.sre_solver_exploitability_tol,
            "early_exit": self.config.sre_solver_early_exit,
            "candidate_selection": self.config.sre_candidate_selection,
        }
        if initial_policies is not None:
            kwargs["initial_policies"] = initial_policies
        while True:
            try:
                return self.sre_solver.solve(q_tensor, **kwargs)
            except TypeError as exc:
                message = str(exc)
                removed = False
                for key in (
                    "initial_policies",
                    "exploitability_tol",
                    "early_exit",
                    "include_pure_starts",
                    "candidate_selection",
                ):
                    if key in kwargs and key in message:
                        kwargs.pop(key, None)
                        removed = True
                if not removed:
                    raise

    def _call_sre_solver_batch(self, q_tensors, *, initial_policies_batch=None):
        kwargs = {
            "epsilon": self.config.epsilon_robust,
            "num_repeats": self.config.sre_num_repeats,
            "include_pure_starts": self.config.sre_include_pure_starts,
            "exploitability_tol": self.config.sre_solver_exploitability_tol,
            "early_exit": self.config.sre_solver_early_exit,
            "candidate_selection": self.config.sre_candidate_selection,
        }
        if initial_policies_batch is not None:
            kwargs["initial_policies_batch"] = initial_policies_batch
        while True:
            try:
                return self.sre_solver.solve_batch(q_tensors, **kwargs)
            except TypeError as exc:
                message = str(exc)
                removed = False
                for key in (
                    "initial_policies_batch",
                    "exploitability_tol",
                    "early_exit",
                    "include_pure_starts",
                    "candidate_selection",
                ):
                    if key in kwargs and key in message:
                        kwargs.pop(key, None)
                        removed = True
                if not removed:
                    raise

    def _call_sre_solver_batch_torch(self, q_tensors, *, initial_policies_batch=None):
        kwargs = {
            "epsilon": self.config.epsilon_robust,
            "num_repeats": self.config.sre_num_repeats,
            "include_pure_starts": self.config.sre_include_pure_starts,
            "exploitability_tol": self.config.sre_solver_exploitability_tol,
            "early_exit": self.config.sre_solver_early_exit,
            "candidate_selection": self.config.sre_candidate_selection,
        }
        if initial_policies_batch is not None:
            kwargs["initial_policies_batch"] = initial_policies_batch
        while True:
            try:
                return self.sre_solver.solve_batch_torch(q_tensors, **kwargs)
            except TypeError as exc:
                message = str(exc)
                removed = False
                for key in (
                    "initial_policies_batch",
                    "exploitability_tol",
                    "early_exit",
                    "include_pure_starts",
                    "candidate_selection",
                ):
                    if key in kwargs and key in message:
                        kwargs.pop(key, None)
                        removed = True
                if not removed:
                    raise

    def _call_sre_solver_policy_batch_torch(self, q_tensors):
        return self.sre_solver.solve_policy_batch_torch(
            q_tensors,
            self.config.epsilon_robust,
        )

    def _policies_from_sre_result(self, result, *, warm_policies=None):
        if result is None:
            return self._warm_start_policies_or_fallback(
                warm_policies,
                "Solver returned no result.",
        )

        metadata = dict(getattr(result, "metadata", None) or {})
        if not result.policies or metadata.get("path_failed", False):
            message = getattr(result, "message", "") or "Solver returned no policies."
            return self._warm_start_policies_or_fallback(
                warm_policies,
                f"{message} Metadata: {metadata}",
            )

        self.sre_candidate_return_count += 1
        policies = [self._normalize_policy(policy) for policy in result.policies]
        if (
            len(policies) != self.config.num_agents
            or any(policy.shape[0] != self.config.num_actions for policy in policies)
        ):
            return self._warm_start_policies_or_fallback(
                warm_policies,
                "Solver returned malformed policies. "
                f"Expected {self.config.num_agents} policies of length "
                f"{self.config.num_actions}; got {[policy.shape for policy in policies]}.",
            )

        if result.success:
            return policies, True

        if bool(getattr(self.sre_solver, "trust_approximate_policies", False)):
            return policies, True

        if not self.config.sre_exploitability_filter_enabled:
            return policies, True

        gap = metadata.get("robust_exploitability")
        if gap is not None and float(gap) <= float(self.config.sre_approx_accept_tol):
            return policies, True

        return self._warm_start_policies_or_fallback(
            warm_policies,
            "Rejected approximate SRE candidate because robust exploitability "
            f"{gap!r} exceeded tolerance {self.config.sre_approx_accept_tol}.",
        )

    def _expanded_policies_from_masked_sre_result(
        self,
        result,
        *,
        action_masks,
        action_indices,
    ):
        if result is None:
            return self._masked_uniform_policies(action_masks)

        metadata = dict(getattr(result, "metadata", None) or {})
        if not result.policies or metadata.get("path_failed", False):
            return self._masked_uniform_policies(action_masks)

        self.sre_candidate_return_count += 1
        try:
            policies = self._expand_reduced_policies(
                result.policies,
                action_masks,
                action_indices,
            )
        except ValueError:
            return self._masked_uniform_policies(action_masks)

        if result.success:
            return policies

        if bool(getattr(self.sre_solver, "trust_approximate_policies", False)):
            return policies

        if not self.config.sre_exploitability_filter_enabled:
            return policies

        gap = metadata.get("robust_exploitability")
        if gap is not None and float(gap) <= float(self.config.sre_approx_accept_tol):
            return policies

        return self._masked_uniform_policies(action_masks)

    def _solve_sre(self, q_tensor, state_key=None):
        q_tensor = np.asarray(q_tensor, dtype=np.float32)
        if not self._sre_policy_cache_active():
            solve_start = time.perf_counter()
            try:
                result = self._call_sre_solver(q_tensor)
            except Exception as exc:
                self._record_sre_solve_time(time.perf_counter() - solve_start)
                return self._fallback_policies(f"Solver raised {exc!r}.")
            self._record_sre_solve_time(time.perf_counter() - solve_start)
            policies, _ = self._policies_from_sre_result(result)
            return policies

        cache_key = self._sre_batch_key(q_tensor)
        cached_policies, warm_policies = self._lookup_sre_policy_cache(
            q_tensor, cache_key, state_key=state_key
        )
        if cached_policies is not None:
            return cached_policies

        solve_start = time.perf_counter()
        try:
            result = self._call_sre_solver(q_tensor, initial_policies=warm_policies)
        except Exception as exc:
            self._record_sre_solve_time(time.perf_counter() - solve_start)
            return self._fallback_policies(f"Solver raised {exc!r}.")
        self._record_sre_solve_time(time.perf_counter() - solve_start)

        policies, cacheable = self._policies_from_sre_result(
            result,
            warm_policies=warm_policies,
        )
        if cacheable:
            self._store_sre_policy_cache(
                cache_key,
                policies,
                state_key=state_key,
                metadata=getattr(result, "metadata", None),
            )
        return policies

    def _solve_sre_batch_uncached(self, q_tensors, states=None):
        del states
        q_tensors_torch = q_tensors if isinstance(q_tensors, torch.Tensor) else None
        expected_ndim = self.config.num_agents + 2
        if q_tensors_torch is not None:
            if q_tensors_torch.ndim != expected_ndim:
                raise ValueError(
                    "Expected q_tensors with shape [B, A1, ..., AN, N], "
                    f"got {tuple(q_tensors_torch.shape)}."
                )
            if tuple(q_tensors_torch.shape[1:]) != self.q_tensor_shape:
                raise ValueError(
                    f"Expected per-sample Q tensor shape {self.q_tensor_shape}, "
                    f"got {tuple(q_tensors_torch.shape[1:])}."
                )
            pending_q_tensors = q_tensors_torch
            batch_size = int(q_tensors_torch.shape[0])
        else:
            pending_q_tensors = np.asarray(q_tensors, dtype=np.float32)
            if pending_q_tensors.ndim != expected_ndim:
                raise ValueError(
                    "Expected q_tensors with shape [B, A1, ..., AN, N], "
                    f"got {pending_q_tensors.shape}."
                )
            if tuple(pending_q_tensors.shape[1:]) != self.q_tensor_shape:
                raise ValueError(
                    f"Expected per-sample Q tensor shape {self.q_tensor_shape}, "
                    f"got {pending_q_tensors.shape[1:]}."
                )
            batch_size = int(pending_q_tensors.shape[0])

        solve_start = time.perf_counter()
        try:
            if q_tensors_torch is not None and hasattr(self.sre_solver, "solve_batch_torch"):
                results = self._call_sre_solver_batch_torch(pending_q_tensors)
            elif hasattr(self.sre_solver, "solve_batch"):
                pending_for_solver = (
                    pending_q_tensors.detach().cpu().numpy().astype(np.float32, copy=False)
                    if q_tensors_torch is not None
                    else pending_q_tensors
                )
                results = self._call_sre_solver_batch(pending_for_solver)
            else:
                iterable = (
                    pending_q_tensors.detach().cpu().numpy()
                    if q_tensors_torch is not None
                    else pending_q_tensors
                )
                results = [self._solve_sre_result(q_tensor) for q_tensor in iterable]
        except Exception as exc:
            elapsed = time.perf_counter() - solve_start
            self._record_sre_solve_time(elapsed, count=batch_size)
            if not bool(self.config.sre_uniform_fallback_enabled):
                raise RuntimeError(
                    "SRE batch solver failed and uniform fallback is disabled. "
                    f"Solver raised {exc!r}."
                ) from exc
            results = [None] * batch_size
        else:
            elapsed = time.perf_counter() - solve_start
            self._record_sre_solve_time(elapsed, count=batch_size)

        policies_batch = []
        for result in results:
            policies, _ = self._policies_from_sre_result(result)
            policies_batch.append([policy.copy() for policy in policies])
        return policies_batch

    def _solve_sre_batch_masked(
        self,
        q_tensors,
        action_masks_batch,
        states=None,
        *,
        allow_solver=True,
        allow_cache_reuse=True,
    ):
        del states, allow_solver, allow_cache_reuse
        q_tensors_torch = q_tensors if isinstance(q_tensors, torch.Tensor) else None
        if q_tensors_torch is not None:
            q_tensors_np = q_tensors_torch.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
        else:
            q_tensors_np = np.asarray(q_tensors, dtype=np.float32)

        expected_ndim = self.config.num_agents + 2
        if q_tensors_np.ndim != expected_ndim:
            raise ValueError(
                "Expected q_tensors with shape [B, A1, ..., AN, N], "
                f"got {q_tensors_np.shape}."
            )
        if tuple(q_tensors_np.shape[1:]) != self.q_tensor_shape:
            raise ValueError(
                f"Expected per-sample Q tensor shape {self.q_tensor_shape}, "
                f"got {q_tensors_np.shape[1:]}."
            )

        batch_size = int(q_tensors_np.shape[0])
        masks_batch = self._normalize_action_masks_batch(action_masks_batch, batch_size)
        if all(masks is None for masks in masks_batch):
            return self._solve_sre_batch(q_tensors, states=None)

        full_mask_profile = [
            np.ones(self.config.num_actions, dtype=bool)
            for _ in range(self.config.num_agents)
        ]
        reduced_q_tensors = []
        reduced_indices = []
        normalized_masks_batch = []
        for q_tensor, masks in zip(q_tensors_np, masks_batch):
            if masks is None:
                masks = [mask.copy() for mask in full_mask_profile]
            reduced_q, action_indices = self._slice_q_tensor_for_masks(q_tensor, masks)
            reduced_q_tensors.append(reduced_q.astype(np.float32, copy=False))
            reduced_indices.append(action_indices)
            normalized_masks_batch.append(masks)

        solve_start = time.perf_counter()
        try:
            if hasattr(self.sre_solver, "solve_batch"):
                results = self._call_sre_solver_batch(reduced_q_tensors)
            else:
                results = [self._solve_sre_result(q_tensor) for q_tensor in reduced_q_tensors]
        except Exception as exc:
            elapsed = time.perf_counter() - solve_start
            self._record_sre_solve_time(elapsed, count=batch_size)
            if not bool(self.config.sre_uniform_fallback_enabled):
                raise RuntimeError(
                    "Masked SRE batch solver failed and uniform fallback is disabled. "
                    f"Solver raised {exc!r}."
                ) from exc
            results = [None] * batch_size
        else:
            elapsed = time.perf_counter() - solve_start
            self._record_sre_solve_time(elapsed, count=batch_size)

        policies_batch = []
        for result, masks, action_indices in zip(
            results,
            normalized_masks_batch,
            reduced_indices,
        ):
            policies_batch.append(
                self._expanded_policies_from_masked_sre_result(
                    result,
                    action_masks=masks,
                    action_indices=action_indices,
                )
            )
        return policies_batch

    def _solve_sre_batch(
        self,
        q_tensors,
        states=None,
        *,
        allow_solver=True,
        allow_cache_reuse=True,
    ):
        if not self._sre_policy_cache_active():
            return self._solve_sre_batch_uncached(q_tensors, states=states)

        q_tensors_torch = q_tensors if isinstance(q_tensors, torch.Tensor) else None
        if q_tensors_torch is not None:
            q_tensors_np = q_tensors_torch.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
        else:
            q_tensors_np = np.asarray(q_tensors, dtype=np.float32)
        expected_ndim = self.config.num_agents + 2
        if q_tensors_np.ndim != expected_ndim:
            raise ValueError(
                "Expected q_tensors with shape [B, A1, ..., AN, N], "
                f"got {q_tensors_np.shape}."
            )
        if tuple(q_tensors_np.shape[1:]) != self.q_tensor_shape:
            raise ValueError(
                f"Expected per-sample Q tensor shape {self.q_tensor_shape}, "
                f"got {q_tensors_np.shape[1:]}."
            )

        if states is not None:
            states = np.asarray(states, dtype=np.float32)
            if states.shape[0] != q_tensors_np.shape[0]:
                raise ValueError(
                    "states must be None or have the same leading dimension as q_tensors."
                )

        policies_by_index = [None] * q_tensors_np.shape[0]
        unique_q_tensors = []
        unique_q_tensors_torch = []
        unique_keys = []
        unique_state_keys = []
        key_to_unique_index = {}

        for batch_index, q_tensor in enumerate(q_tensors_np):
            batch_key = self._sre_batch_key(q_tensor)
            unique_index = key_to_unique_index.get(batch_key)
            if unique_index is None:
                unique_index = len(unique_q_tensors)
                key_to_unique_index[batch_key] = unique_index
                unique_q_tensors.append(q_tensor)
                if q_tensors_torch is not None:
                    unique_q_tensors_torch.append(q_tensors_torch[batch_index])
                unique_keys.append(batch_key)
                state_key = None
                if states is not None:
                    state_key = self._sre_state_key(states[batch_index])
                unique_state_keys.append(state_key)
            policies_by_index[batch_index] = unique_index

        unique_policies = [None] * len(unique_q_tensors)
        pending_q_tensors = []
        pending_indices = []
        pending_warm_policies = []
        if unique_q_tensors:
            for unique_index, (q_tensor, batch_key, state_key) in enumerate(
                zip(unique_q_tensors, unique_keys, unique_state_keys)
            ):
                cached_policies, warm_policies = self._lookup_sre_policy_cache(
                    q_tensor,
                    batch_key,
                    state_key=state_key,
                    allow_reuse=allow_cache_reuse,
                )
                if cached_policies is not None:
                    unique_policies[unique_index] = cached_policies
                else:
                    if q_tensors_torch is not None:
                        pending_q_tensors.append(unique_q_tensors_torch[unique_index])
                    else:
                        pending_q_tensors.append(q_tensor)
                    pending_indices.append(unique_index)
                    pending_warm_policies.append(warm_policies)

        if pending_q_tensors and not allow_solver:
            for unique_index, warm_policies in zip(pending_indices, pending_warm_policies):
                if self._policies_valid(warm_policies):
                    unique_policies[unique_index] = self._copy_policies(warm_policies)
                else:
                    unique_policies[unique_index] = self._fallback_policies(
                        "Target-equilibrium cache-only step had no cached or warm-start "
                        "policy and solver refresh was disabled."
                    )
            pending_q_tensors = []
            pending_indices = []
            pending_warm_policies = []

        if pending_q_tensors:
            solve_start = time.perf_counter()
            try:
                if q_tensors_torch is not None and hasattr(self.sre_solver, "solve_batch_torch"):
                    results = self._call_sre_solver_batch_torch(
                        torch.stack(pending_q_tensors, dim=0),
                        initial_policies_batch=pending_warm_policies,
                    )
                elif hasattr(self.sre_solver, "solve_batch"):
                    pending_for_solver = (
                        [
                            q_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                            for q_tensor in pending_q_tensors
                        ]
                        if q_tensors_torch is not None
                        else pending_q_tensors
                    )
                    results = self._call_sre_solver_batch(
                        pending_for_solver,
                        initial_policies_batch=pending_warm_policies,
                    )
                else:
                    pending_for_solver = (
                        [
                            q_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                            for q_tensor in pending_q_tensors
                        ]
                        if q_tensors_torch is not None
                        else pending_q_tensors
                    )
                    results = [
                        self._solve_sre_result(q_tensor, initial_policies=warm_policies)
                        for q_tensor, warm_policies in zip(
                            pending_for_solver, pending_warm_policies
                        )
                    ]
            except Exception as exc:
                elapsed = time.perf_counter() - solve_start
                self._record_sre_solve_time(elapsed, count=len(pending_q_tensors))
                if not bool(self.config.sre_uniform_fallback_enabled):
                    raise RuntimeError(
                        "SRE batch solver failed and uniform fallback is disabled. "
                        f"Solver raised {exc!r}."
                    ) from exc
                results = [None] * len(pending_q_tensors)
            else:
                elapsed = time.perf_counter() - solve_start
                self._record_sre_solve_time(elapsed, count=len(pending_q_tensors))

            for unique_index, result, warm_policies in zip(
                pending_indices,
                results,
                pending_warm_policies,
            ):
                policies, cacheable = self._policies_from_sre_result(
                    result,
                    warm_policies=warm_policies,
                )
                unique_policies[unique_index] = policies
                if cacheable:
                    self._store_sre_policy_cache(
                        unique_keys[unique_index],
                        policies,
                        state_key=unique_state_keys[unique_index],
                        metadata=getattr(result, "metadata", None),
                    )

        for batch_index, entry in enumerate(policies_by_index):
            if isinstance(entry, int):
                policies_by_index[batch_index] = [
                    policy.copy() for policy in unique_policies[entry]
                ]

        return policies_by_index

    def _solve_sre_result(self, q_tensor, initial_policies=None):
        return self._call_sre_solver(q_tensor, initial_policies=initial_policies)

    def _sre_expected_values(self, q_tensor, policies):
        expected = q_tensor
        for policy in policies:
            expected = np.tensordot(policy, expected, axes=([0], [0]))
        return np.asarray(expected, dtype=np.float32)

    def _sre_expected_values_batch(self, q_tensors, policies_batch):
        values = []
        for q_tensor, policies in zip(q_tensors, policies_batch):
            values.append(self._sre_expected_values(q_tensor, policies))
        return np.stack(values, axis=0).astype(np.float32)

    def _sre_robust_values(self, q_tensor, policies):
        q_tensor = np.asarray(q_tensor, dtype=np.float32)
        return np.asarray(
            robust_policy_values(q_tensor, policies, self.config.epsilon_robust),
            dtype=np.float32,
        )

    def _sre_robust_values_batch(self, q_tensors, policies_batch):
        values = []
        for q_tensor, policies in zip(q_tensors, policies_batch):
            values.append(self._sre_robust_values(q_tensor, policies))
        return np.stack(values, axis=0).astype(np.float32)

    def _sre_target_values_batch(self, q_tensors, policies_batch):
        mode = str(self.config.sre_target_value_mode).lower()
        if mode == "robust":
            return self._sre_robust_values_batch(q_tensors, policies_batch)
        if mode == "nominal":
            return self._sre_expected_values_batch(q_tensors, policies_batch)
        raise ValueError(
            "sre_target_value_mode must be 'robust' or 'nominal', "
            f"got {self.config.sre_target_value_mode!r}."
        )

    def act(self, state, agent_id=None, action_masks=None):
        if agent_id is None:
            agent_id = self.config.agent_id
        if not 0 <= agent_id < self.config.num_agents:
            raise ValueError(f"Expected agent_id in [0, {self.config.num_agents}), got {agent_id}.")
        masks = None
        if action_masks is not None:
            masks_arr = np.asarray(action_masks, dtype=bool)
            if masks_arr.ndim == 1:
                masks = [np.ones(self.config.num_actions, dtype=bool) for _ in range(self.config.num_agents)]
                masks[agent_id] = masks_arr.reshape(-1)
                masks = self._normalize_action_masks(masks)
            else:
                masks = self._normalize_action_masks(action_masks)

        if np.random.rand() < self.config.epsilon_explore:
            return self._sample_action_with_mask(None if masks is None else masks[agent_id])

        state_vec = self._state_to_vector(state)
        state_t = torch.as_tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_batch = self.q_net(state_t).detach()

        if masks is not None:
            policies = self._solve_sre_batch_masked(q_batch, [masks])[0]
        elif self._sre_solver_supports_torch_policy_solve():
            solve_start = time.perf_counter()
            policy_tensors = self._call_sre_solver_policy_batch_torch(q_batch)
            self._record_sre_solve_time(time.perf_counter() - solve_start, count=1)
            policies = [
                policy[0].detach().cpu().numpy().astype(np.float32, copy=True)
                for policy in policy_tensors
            ]
        elif self._sre_solver_supports_torch_action_solve():
            policies = self._solve_sre_batch_uncached(q_batch)[0]
        else:
            q_tensor = q_batch.squeeze(0).detach().cpu().numpy()
            policies = self._solve_sre(q_tensor, state_key=self._sre_state_key(state_vec))
        my_policy = self._normalize_policy(policies[agent_id])
        return int(np.random.choice(self.config.num_actions, p=my_policy))

    def act_joint(self, state, action_masks=None):
        """Sample one joint action, solving the SRE policy profile at most once."""
        state_vec = self._state_to_vector(state)
        action_masks = self._normalize_action_masks(action_masks)
        explore_mask = (
            np.random.rand(self.config.num_agents) < self.config.epsilon_explore
        )
        actions = np.empty(self.config.num_agents, dtype=np.int64)
        for agent_id, explore in enumerate(explore_mask):
            if explore:
                mask = None if action_masks is None else action_masks[agent_id]
                actions[agent_id] = self._sample_action_with_mask(mask)

        if np.all(explore_mask):
            return actions.astype(int).tolist()

        state_t = torch.as_tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_batch = self.q_net(state_t).detach()

        if action_masks is not None:
            policies = self._solve_sre_batch_masked(q_batch, [action_masks])[0]
        elif self._sre_solver_supports_torch_policy_solve():
            solve_start = time.perf_counter()
            policy_tensors = self._call_sre_solver_policy_batch_torch(q_batch)
            self._record_sre_solve_time(time.perf_counter() - solve_start, count=1)
            policies = [
                policy[0].detach().cpu().numpy().astype(np.float32, copy=True)
                for policy in policy_tensors
            ]
        elif self._sre_solver_supports_torch_action_solve():
            policies = self._solve_sre_batch_uncached(q_batch)[0]
        else:
            q_tensor = q_batch.squeeze(0).detach().cpu().numpy()
            policies = self._solve_sre(
                q_tensor, state_key=self._sre_state_key(state_vec)
            )
        for agent_id, explore in enumerate(explore_mask):
            if not explore:
                policy = self._normalize_policy(policies[agent_id])
                actions[agent_id] = np.random.choice(self.config.num_actions, p=policy)
        return actions.astype(int).tolist()

    def act_joint_batch(self, states, action_masks_batch=None):
        """Sample one joint action per state with batched Q and SRE solves."""
        states_list = list(states)
        if not states_list:
            return []

        states_arr = np.stack(
            [self._state_to_vector(state) for state in states_list],
            axis=0,
        ).astype(np.float32, copy=False)
        batch_size = int(states_arr.shape[0])
        action_masks_batch = self._normalize_action_masks_batch(
            action_masks_batch,
            batch_size,
        )
        explore_mask = (
            np.random.rand(batch_size, self.config.num_agents)
            < self.config.epsilon_explore
        )
        actions = np.empty((batch_size, self.config.num_agents), dtype=np.int64)
        for batch_idx, agent_id in np.argwhere(explore_mask):
            masks = action_masks_batch[int(batch_idx)]
            mask = None if masks is None else masks[int(agent_id)]
            actions[int(batch_idx), int(agent_id)] = np.random.choice(
                np.flatnonzero(mask) if mask is not None else self.config.num_actions
            )

        pending_indices = np.flatnonzero(~np.all(explore_mask, axis=1))
        if pending_indices.size:
            states_t = torch.as_tensor(
                states_arr[pending_indices],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                q_batch = self.q_net(states_t).detach()

            pending_masks = [action_masks_batch[int(idx)] for idx in pending_indices]
            has_masks = any(masks is not None for masks in pending_masks)
            if has_masks:
                policies_batch = self._solve_sre_batch_masked(q_batch, pending_masks)
            elif self._sre_solver_supports_torch_policy_solve():
                solve_start = time.perf_counter()
                policy_tensors = self._call_sre_solver_policy_batch_torch(q_batch)
                self._record_sre_solve_time(
                    time.perf_counter() - solve_start,
                    count=int(pending_indices.size),
                )
                policies_batch = []
                for local_idx in range(int(pending_indices.size)):
                    policies_batch.append(
                        [
                            policy[local_idx]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=True)
                            for policy in policy_tensors
                        ]
                    )
            elif self._sre_solver_supports_torch_action_solve():
                policies_batch = self._solve_sre_batch_uncached(q_batch)
            else:
                policies_batch = self._solve_sre_batch(
                    q_batch,
                    states=states_arr[pending_indices],
                )

            for local_idx, batch_idx in enumerate(pending_indices):
                policies = policies_batch[local_idx]
                for agent_id in range(self.config.num_agents):
                    if not explore_mask[batch_idx, agent_id]:
                        policy = self._normalize_policy(policies[agent_id])
                        actions[batch_idx, agent_id] = np.random.choice(
                            self.config.num_actions,
                            p=policy,
                        )
        return actions.astype(int).tolist()

    def update(
        self,
        state,
        joint_actions,
        joint_rewards,
        next_state,
        done=False,
        batch_size=64,
        action_masks=None,
        next_action_masks=None,
    ):
        state_vec = self._state_to_vector(state)
        next_state_vec = self._state_to_vector(next_state)
        actions_arr = np.asarray(joint_actions, dtype=np.int64).reshape(-1)
        rewards_arr = np.asarray(joint_rewards, dtype=np.float32).reshape(-1)
        action_masks = self._normalize_action_masks(action_masks)
        next_action_masks = self._normalize_action_masks(next_action_masks)

        if actions_arr.shape[0] != self.config.num_agents:
            raise ValueError(
                f"Expected joint action length {self.config.num_agents}, got {actions_arr.shape[0]}."
            )
        if rewards_arr.shape[0] != self.config.num_agents:
            raise ValueError(
                f"Expected joint reward length {self.config.num_agents}, got {rewards_arr.shape[0]}."
            )

        self._update_calls += 1
        self.replay_buffer.push(
            state_vec,
            actions_arr,
            rewards_arr,
            next_state_vec,
            done,
            action_masks,
            next_action_masks,
        )
        if self._update_calls % self.config.train_every != 0:
            return None
        return self.train_step(batch_size=batch_size)

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < max(batch_size, self.config.learning_starts):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            action_masks,
            next_action_masks,
        ) = self.replay_buffer.sample(batch_size)
        states_arr = np.stack([self._state_to_vector(s) for s in states], axis=0)
        next_states_arr = np.stack([self._state_to_vector(s) for s in next_states], axis=0)
        actions_arr = np.stack(actions, axis=0)
        rewards_arr = np.stack(rewards, axis=0)
        next_action_masks_batch = self._normalize_action_masks_batch(
            next_action_masks,
            len(next_states),
        )

        states_t = torch.as_tensor(states_arr, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            next_states_arr, dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(actions_arr, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards_arr, dtype=torch.float32, device=self.device)
        dones_arr = np.asarray(dones, dtype=np.float32)
        dones_t = torch.as_tensor(dones_arr, dtype=torch.float32, device=self.device)
        if dones_t.ndim == 1:
            dones_t = dones_t.unsqueeze(1)

        update_start = time.perf_counter()
        q_tensor = self.q_net(states_t)  # [B, A1, ..., AN, N]
        batch_idx = torch.arange(states_t.shape[0], device=self.device)
        action_indices = [actions_t[:, agent_id] for agent_id in range(self.config.num_agents)]
        current_q = q_tensor[(batch_idx, *action_indices, slice(None))]  # [B, N]

        with torch.no_grad():
            # Double-DQN style: choose policy from online net, evaluate with target net.
            next_online = self.q_net(next_states_t).detach()
            next_target_t = self.target_net(next_states_t).detach()

            gradient_step_index = len(self.update_times) + 1
            target_equilibrium_update_steps = max(
                1, int(self.config.target_equilibrium_update_steps)
            )
            use_policy_cache = self._sre_policy_cache_active()
            refresh_target_equilibria = (
                not use_policy_cache
                or (gradient_step_index - 1) % target_equilibrium_update_steps == 0
            )

            next_values_t = torch.zeros(
                (states_t.shape[0], self.config.num_agents),
                dtype=torch.float32,
                device=self.device,
            )
            if dones_arr.ndim == 1:
                nonterminal_mask = dones_arr < 1.0
            else:
                nonterminal_mask = np.any(dones_arr < 1.0, axis=1)
            nonterminal_indices = np.flatnonzero(nonterminal_mask)
            if nonterminal_indices.size:
                nonterminal_t = torch.as_tensor(
                    nonterminal_indices, dtype=torch.long, device=self.device
                )
                next_masks_nonterminal = [
                    next_action_masks_batch[int(index)] for index in nonterminal_indices
                ]
                has_next_masks = any(masks is not None for masks in next_masks_nonterminal)
                if (
                    not has_next_masks
                    and
                    self._sre_solver_supports_torch_policy_solve()
                    and str(self.config.sre_target_value_mode).lower() == "robust"
                ):
                    solve_start = time.perf_counter()
                    policy_tensors = self._call_sre_solver_policy_batch_torch(
                        next_online[nonterminal_t]
                    )
                    self._record_sre_solve_time(
                        time.perf_counter() - solve_start,
                        count=int(nonterminal_indices.size),
                    )
                    policy_device = policy_tensors[0].device
                    value_tensors = robust_policy_values_torch(
                        next_target_t[nonterminal_t].to(policy_device),
                        policy_tensors,
                        self.config.epsilon_robust,
                    )
                    next_values_t[nonterminal_t] = torch.stack(
                        value_tensors, dim=-1
                    ).to(device=self.device, dtype=torch.float32)
                else:
                    next_target = next_target_t.detach().cpu().numpy()
                    if has_next_masks:
                        policies_batch = self._solve_sre_batch_masked(
                            next_online[nonterminal_t],
                            next_masks_nonterminal,
                            states=next_states_arr[nonterminal_indices],
                            allow_solver=refresh_target_equilibria,
                            allow_cache_reuse=not refresh_target_equilibria,
                        )
                    else:
                        policies_batch = self._solve_sre_batch(
                            next_online[nonterminal_t],
                            states=next_states_arr[nonterminal_indices],
                            allow_solver=refresh_target_equilibria,
                            allow_cache_reuse=not refresh_target_equilibria,
                        )
                    next_values_t[nonterminal_t] = torch.as_tensor(
                        self._sre_target_values_batch(
                            next_target[nonterminal_indices],
                            policies_batch,
                        ),
                        device=self.device,
                        dtype=torch.float32,
                    )
            target_q = rewards_t + (1.0 - dones_t) * self.config.gamma * next_values_t

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self.update_times.append(time.perf_counter() - update_start)
        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def soft_update_target_network(self, tau=0.005):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def decay_parameters(self, episode_idx, n_episodes):
        cfg = self.config
        if cfg.epsilon_schedule == "constant":
            cfg.epsilon_robust = float(cfg.epsilon_robust_initial)
        elif cfg.epsilon_schedule == "linear":
            cfg.epsilon_robust = _linear_schedule(
                cfg.epsilon_robust_initial, 0.0, episode_idx, n_episodes
            )
        elif cfg.epsilon_schedule == "exponential":
            cfg.epsilon_robust = float(cfg.epsilon_robust_initial) * (cfg.decay_rate ** int(episode_idx))
        else:
            raise ValueError(f"Unsupported epsilon schedule: {cfg.epsilon_schedule}")

        decay_episodes = max(1, int(n_episodes * cfg.action_epsilon_decay_fraction))
        cfg.epsilon_explore = _linear_schedule(
            cfg.action_epsilon_start,
            cfg.action_epsilon_end,
            min(int(episode_idx), decay_episodes - 1),
            decay_episodes,
        )

    def save_checkpoint(self, path, include_replay_buffer=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": _checkpoint_config_dict(self.config),
            "update_calls": self._update_calls,
            "buffer_size": self.replay_buffer.buffer.maxlen,
            "update_times": list(self.update_times),
            "sre_solve_time_count": self.sre_solve_time_count,
            "sre_solve_time_sum": self.sre_solve_time_sum,
            "sre_solve_time_sumsq": self.sre_solve_time_sumsq,
            "sre_solve_time_min": self.sre_solve_time_min,
            "sre_solve_time_max": self.sre_solve_time_max,
            "sre_cache_summary": self.get_sre_cache_summary(),
            "sre_solver_name": getattr(
                self.sre_solver, "name", type(self.sre_solver).__name__
            ),
        }
        if include_replay_buffer:
            payload["replay_buffer"] = list(self.replay_buffer.buffer)
        torch.save(payload, path)

    def load_checkpoint(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        try:
            checkpoint = torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)

        if (s := checkpoint.get("q_net")) is not None:
            self.q_net.load_state_dict(s)
        if (s := checkpoint.get("target_net")) is not None:
            self.target_net.load_state_dict(s)
        if (s := checkpoint.get("optimizer")) is not None:
            self.optimizer.load_state_dict(s)

        cfg = self.config
        _restore_checkpoint_config(cfg, checkpoint["config"])
        self.q_tensor_shape = tuple([cfg.num_actions] * cfg.num_agents + [cfg.num_agents])
        self._update_calls = int(checkpoint.get("update_calls", self._update_calls))

        replay_buffer_items = checkpoint.get("replay_buffer")
        if replay_buffer_items is not None:
            self.replay_buffer = ReplayBuffer(checkpoint.get("buffer_size", len(replay_buffer_items)))
            self.replay_buffer.buffer.extend(replay_buffer_items)

        self.update_times = list(checkpoint.get("update_times", self.update_times))
        self.sre_solve_time_count = checkpoint.get("sre_solve_time_count", self.sre_solve_time_count)
        self.sre_solve_time_sum = checkpoint.get("sre_solve_time_sum", self.sre_solve_time_sum)
        self.sre_solve_time_sumsq = checkpoint.get("sre_solve_time_sumsq", self.sre_solve_time_sumsq)
        self.sre_solve_time_min = checkpoint.get("sre_solve_time_min", self.sre_solve_time_min)
        self.sre_solve_time_max = checkpoint.get("sre_solve_time_max", self.sre_solve_time_max)

    def close(self):
        if hasattr(self, "sre_solver") and self.sre_solver is not None:
            self.sre_solver.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def flatten_gridworld_obs(obs):
    return np.asarray([coord for pos in obs for coord in pos], dtype=np.float32)
