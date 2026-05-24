import random
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
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


def _linear_schedule(start, end, step, total_steps):
    if total_steps <= 1:
        return float(end)
    fraction = min(max(step, 0) / float(total_steps - 1), 1.0)
    return float(start + fraction * (end - start))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, joint_actions, joint_rewards, next_state, done):
        self.buffer.append((state, joint_actions, joint_rewards, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


@dataclass
class DuelingDoubleDqnSreAgentConfig:
    agent_id: int = 0
    obs_dim: int = 4 # 2 agents, 2 coordinates each
    num_agents: int = 2
    num_actions: int = 4
    pathwrap_path: str = "pathwrap.so"
    epsilon_robust: float = 1.0
    epsilon_explore: float = 1.0
    lr: float = 3e-4
    gamma: float = 0.9
    decay_rate: float = 0.999
    buffer_size: int = 10000
    batch_size: int = 16
    learning_starts: int = 1000
    grad_clip_norm: float = 10.0
    use_gpu: bool = True
    # N-player solvers are called inside the DQN training loop; keep the
    # default approximate budget small. Set these back to 20/True for
    # exhaustive offline solver comparisons.
    sre_num_repeats: int = 4
    sre_include_pure_starts: bool = False
    train_every: int = 1
    sre_solver: Any = None
    network_type: str = "joint_output"
    target_tau: Optional[float] = None
    target_update_steps: int = 100
    target_equilibrium_update_steps: int = 1
    action_epsilon_start: float = 1.0
    action_epsilon_end: float = 0.05
    action_epsilon_decay_fraction: float = 0.5
    sre_solver_name: str = "path_c_pool"
    sre_solver_workers: int = 8
    sre_solver_start_method: Optional[str] = None
    q_hidden_dims: tuple = (128, 128)
    epsilon_robust_initial: float = 1.0
    epsilon_schedule: str = "exponential"
    sre_policy_cache_enabled: bool = False
    sre_policy_cache_size: int = 4096
    sre_policy_cache_round_digits: int = 6
    sre_state_cache_round_digits: int = 4
    sre_approx_cache_enabled: bool = True
    sre_cache_exploitability_tol: float = 1e-3
    sre_solver_exploitability_tol: float = 1e-4
    sre_approx_accept_tol: float = 1e-2
    sre_solver_early_exit: bool = True
    sre_candidate_selection: str = "robust_exploitability"
    sre_exploitability_filter_enabled: bool = False
    sre_target_value_mode: str = "robust"
    sre_uniform_fallback_enabled: bool = False


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
        self.sre_cache_stores = 0
        self.sre_cache_evictions = 0
        self.sre_cache_warm_start_uses = 0
        self.sre_cache_validation_count = 0
        self.sre_cache_validation_time_sum = 0.0
        self.sre_cache_lookup_count = 0
        self.sre_cache_lookup_time_sum = 0.0
        self.sre_cache_lookup_time_sumsq = 0.0
        self.sre_cache_lookup_time_min = None
        self.sre_cache_lookup_time_max = None
        self.sre_uniform_fallback_count = 0
        self.sre_solver_result_count = 0
        self.sre_candidate_return_count = 0
        self.sre_certified_candidate_count = 0
        self.sre_approx_candidate_count = 0
        self.sre_rejected_candidate_count = 0
        self.sre_malformed_candidate_count = 0
        self.sre_unfiltered_candidate_count = 0
        self.sre_candidate_gap_values = []
        self.sre_solver_starts_attempted_count = 0
        self.sre_solver_starts_attempted_sum = 0
        self.sre_solver_early_exit_count = 0
        self.sre_cache_forced_refreshes = 0
        self.target_equilibrium_refresh_count = 0
        self.target_equilibrium_cache_only_count = 0
        self.target_equilibrium_cache_only_misses = 0
        self.target_equilibrium_stale_policy_reuses = 0

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

    def _record_cache_lookup_time(self, duration):
        self.sre_cache_lookup_count += 1
        self.sre_cache_lookup_time_sum += duration
        self.sre_cache_lookup_time_sumsq += duration * duration
        if (
            self.sre_cache_lookup_time_min is None
            or duration < self.sre_cache_lookup_time_min
        ):
            self.sre_cache_lookup_time_min = duration
        if (
            self.sre_cache_lookup_time_max is None
            or duration > self.sre_cache_lookup_time_max
        ):
            self.sre_cache_lookup_time_max = duration

    def _cache_lookup_duration_summary(self):
        count = self.sre_cache_lookup_count
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
        mean = self.sre_cache_lookup_time_sum / count
        variance = max(self.sre_cache_lookup_time_sumsq / count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        return {
            "count": int(count),
            "mean_seconds": float(mean),
            "min_seconds": float(self.sre_cache_lookup_time_min),
            "max_seconds": float(self.sre_cache_lookup_time_max),
            "std_seconds": std,
            "mean_microseconds": float(mean * 1_000_000.0),
            "min_microseconds": float(self.sre_cache_lookup_time_min * 1_000_000.0),
            "max_microseconds": float(self.sre_cache_lookup_time_max * 1_000_000.0),
            "std_microseconds": float(std * 1_000_000.0),
        }

    def get_sre_cache_summary(self):
        exact_hits = int(self.sre_cache_exact_hits)
        approx_hits = int(self.sre_cache_approx_hits)
        misses = int(self.sre_cache_misses)
        requests = exact_hits + approx_hits + misses
        path_avoided = exact_hits + approx_hits
        validation_mean = None
        if self.sre_cache_validation_count:
            validation_mean = (
                self.sre_cache_validation_time_sum / self.sre_cache_validation_count
            )
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
            "stores": int(self.sre_cache_stores),
            "evictions": int(self.sre_cache_evictions),
            "warm_start_uses": int(self.sre_cache_warm_start_uses),
            "forced_refreshes": int(self.sre_cache_forced_refreshes),
            "validation_count": int(self.sre_cache_validation_count),
            "validation_mean_seconds": (
                None if validation_mean is None else float(validation_mean)
            ),
            "validation_mean_microseconds": (
                None if validation_mean is None else float(validation_mean * 1_000_000.0)
            ),
            "lookup_time": self._cache_lookup_duration_summary(),
            "uniform_fallbacks": int(self.sre_uniform_fallback_count),
            "solver_results": int(self.sre_solver_result_count),
            "candidate_returned": int(self.sre_candidate_return_count),
            "candidate_return_rate": (
                None
                if self.sre_solver_result_count == 0
                else float(self.sre_candidate_return_count / self.sre_solver_result_count)
            ),
            "certified_candidates": int(self.sre_certified_candidate_count),
            "approx_candidates": int(self.sre_approx_candidate_count),
            "unfiltered_candidates": int(self.sre_unfiltered_candidate_count),
            "rejected_candidates": int(self.sre_rejected_candidate_count),
            "malformed_candidates": int(self.sre_malformed_candidate_count),
            "candidate_robust_exploitability": self._candidate_gap_summary(),
            "solver_starts_attempted": {
                "count": int(self.sre_solver_starts_attempted_count),
                "mean": (
                    None
                    if self.sre_solver_starts_attempted_count == 0
                    else float(
                        self.sre_solver_starts_attempted_sum
                        / self.sre_solver_starts_attempted_count
                    )
                ),
            },
            "solver_early_exits": int(self.sre_solver_early_exit_count),
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
            "target_equilibrium_refreshes": int(self.target_equilibrium_refresh_count),
            "target_equilibrium_cache_only_steps": int(self.target_equilibrium_cache_only_count),
            "target_equilibrium_cache_only_misses": int(self.target_equilibrium_cache_only_misses),
            "target_equilibrium_stale_policy_reuses": int(self.target_equilibrium_stale_policy_reuses),
        }

    def _candidate_gap_summary(self):
        gaps = np.asarray(self.sre_candidate_gap_values, dtype=np.float64)
        if gaps.size == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "p50": None,
                "p90": None,
                "p95": None,
            }
        return {
            "count": int(gaps.size),
            "mean": float(np.mean(gaps)),
            "std": float(np.std(gaps)),
            "min": float(np.min(gaps)),
            "max": float(np.max(gaps)),
            "p50": float(np.percentile(gaps, 50)),
            "p90": float(np.percentile(gaps, 90)),
            "p95": float(np.percentile(gaps, 95)),
        }

    def _sre_solver_bypasses_policy_cache(self):
        return bool(getattr(self.sre_solver, "bypass_deep_srq_policy_cache", False))

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
            return np.full(self.config.num_actions, 1.0 / self.config.num_actions, dtype=np.float32)
        return p / s

    def _uniform_policies(self):
        u = np.full(self.config.num_actions, 1.0 / self.config.num_actions, dtype=np.float32)
        return [u.copy() for _ in range(self.config.num_agents)]

    def _fallback_policies(self, reason):
        if not bool(self.config.sre_uniform_fallback_enabled):
            raise RuntimeError(
                "SRE solver failed and uniform fallback is disabled. "
                f"{reason}"
            )
        self.sre_uniform_fallback_count += 1
        return self._uniform_policies()

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
        self.sre_cache_stores += 1
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
        start = time.perf_counter()
        try:
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
                self.sre_cache_forced_refreshes += 1
                self.sre_cache_warm_start_uses += 1
                return None, policies

            warm_policies = None
            best_gap = None
            for candidate_key in self._cache_candidate_keys(cache_key, state_key):
                candidate = self._sre_policy_cache[candidate_key]
                candidate_policies = self._copy_policies(candidate["policies"])
                if warm_policies is None:
                    warm_policies = candidate_policies
                if allow_reuse and self.config.sre_approx_cache_enabled:
                    validation_start = time.perf_counter()
                    try:
                        gap, _, _ = robust_exploitability(
                            q_tensor,
                            candidate_policies,
                            self.config.epsilon_robust,
                        )
                    finally:
                        self.sre_cache_validation_time_sum += (
                            time.perf_counter() - validation_start
                        )
                        self.sre_cache_validation_count += 1
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
            if warm_policies is not None:
                self.sre_cache_warm_start_uses += 1
            return None, warm_policies
        finally:
            self._record_cache_lookup_time(time.perf_counter() - start)

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

    def _record_sre_result_quality(self, result, metadata):
        self.sre_solver_result_count += 1
        if metadata.get("early_exit"):
            self.sre_solver_early_exit_count += 1
        starts_attempted = metadata.get("num_starts_attempted")
        if starts_attempted is not None:
            self.sre_solver_starts_attempted_count += 1
            self.sre_solver_starts_attempted_sum += int(starts_attempted)
        gap = metadata.get("robust_exploitability")
        if gap is not None and not metadata.get("path_failed", False):
            self.sre_candidate_gap_values.append(float(gap))

    def _policies_from_sre_result(self, result):
        if result is None:
            return self._fallback_policies("Solver returned no result."), False

        metadata = dict(getattr(result, "metadata", None) or {})
        self._record_sre_result_quality(result, metadata)
        if not result.policies or metadata.get("path_failed", False):
            message = getattr(result, "message", "") or "Solver returned no policies."
            return self._fallback_policies(f"{message} Metadata: {metadata}"), False

        self.sre_candidate_return_count += 1
        policies = [self._normalize_policy(policy) for policy in result.policies]
        if (
            len(policies) != self.config.num_agents
            or any(policy.shape[0] != self.config.num_actions for policy in policies)
        ):
            self.sre_malformed_candidate_count += 1
            return self._fallback_policies(
                "Solver returned malformed policies. "
                f"Expected {self.config.num_agents} policies of length "
                f"{self.config.num_actions}; got {[policy.shape for policy in policies]}."
            ), False

        if result.success:
            self.sre_certified_candidate_count += 1
            return policies, True

        if not self.config.sre_exploitability_filter_enabled:
            self.sre_unfiltered_candidate_count += 1
            return policies, True

        gap = metadata.get("robust_exploitability")
        if gap is not None and float(gap) <= float(self.config.sre_approx_accept_tol):
            self.sre_approx_candidate_count += 1
            return policies, True

        self.sre_rejected_candidate_count += 1
        return self._fallback_policies(
            "Rejected approximate SRE candidate because robust exploitability "
            f"{gap!r} exceeded tolerance {self.config.sre_approx_accept_tol}."
        ), False

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

        policies, cacheable = self._policies_from_sre_result(result)
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
            self.target_equilibrium_cache_only_misses += len(pending_q_tensors)
            for unique_index, warm_policies in zip(pending_indices, pending_warm_policies):
                if self._policies_valid(warm_policies):
                    unique_policies[unique_index] = self._copy_policies(warm_policies)
                    self.target_equilibrium_stale_policy_reuses += 1
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

            for unique_index, result in zip(pending_indices, results):
                policies, cacheable = self._policies_from_sre_result(result)
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

    def act(self, state, agent_id=None):
        if agent_id is None:
            agent_id = self.config.agent_id
        if not 0 <= agent_id < self.config.num_agents:
            raise ValueError(f"Expected agent_id in [0, {self.config.num_agents}), got {agent_id}.")

        if np.random.rand() < self.config.epsilon_explore:
            return int(np.random.choice(self.config.num_actions))

        state_vec = self._state_to_vector(state)
        state_t = torch.as_tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_tensor = self.q_net(state_t).squeeze(0).detach().cpu().numpy()

        policies = self._solve_sre(q_tensor, state_key=self._sre_state_key(state_vec))
        my_policy = self._normalize_policy(policies[agent_id])
        return int(np.random.choice(self.config.num_actions, p=my_policy))

    def act_joint(self, state):
        """Sample one joint action, solving the SRE policy profile at most once."""
        state_vec = self._state_to_vector(state)
        explore_mask = (
            np.random.rand(self.config.num_agents) < self.config.epsilon_explore
        )
        actions = np.empty(self.config.num_agents, dtype=np.int64)
        for agent_id, explore in enumerate(explore_mask):
            if explore:
                actions[agent_id] = np.random.choice(self.config.num_actions)

        if np.all(explore_mask):
            return actions.astype(int).tolist()

        state_t = torch.as_tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_tensor = self.q_net(state_t).squeeze(0).detach().cpu().numpy()

        policies = self._solve_sre(q_tensor, state_key=self._sre_state_key(state_vec))
        for agent_id, explore in enumerate(explore_mask):
            if not explore:
                policy = self._normalize_policy(policies[agent_id])
                actions[agent_id] = np.random.choice(self.config.num_actions, p=policy)
        return actions.astype(int).tolist()

    def update(self, state, joint_actions, joint_rewards, next_state, done=False, batch_size=64):
        state_vec = self._state_to_vector(state)
        next_state_vec = self._state_to_vector(next_state)
        actions_arr = np.asarray(joint_actions, dtype=np.int64).reshape(-1)
        rewards_arr = np.asarray(joint_rewards, dtype=np.float32).reshape(-1)

        if actions_arr.shape[0] != self.config.num_agents:
            raise ValueError(
                f"Expected joint action length {self.config.num_agents}, got {actions_arr.shape[0]}."
            )
        if rewards_arr.shape[0] != self.config.num_agents:
            raise ValueError(
                f"Expected joint reward length {self.config.num_agents}, got {rewards_arr.shape[0]}."
            )

        self._update_calls += 1
        self.replay_buffer.push(state_vec, actions_arr, rewards_arr, next_state_vec, done)
        if self._update_calls % self.config.train_every != 0:
            return None
        return self.train_step(batch_size=batch_size)

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < max(batch_size, self.config.learning_starts):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states_arr = np.stack([self._state_to_vector(s) for s in states], axis=0)
        next_states_arr = np.stack([self._state_to_vector(s) for s in next_states], axis=0)
        actions_arr = np.stack(actions, axis=0)
        rewards_arr = np.stack(rewards, axis=0)

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
            next_target = self.target_net(next_states_t).detach().cpu().numpy()

            gradient_step_index = len(self.update_times) + 1
            target_equilibrium_update_steps = max(
                1, int(self.config.target_equilibrium_update_steps)
            )
            use_policy_cache = self._sre_policy_cache_active()
            refresh_target_equilibria = (
                not use_policy_cache
                or (gradient_step_index - 1) % target_equilibrium_update_steps == 0
            )
            if refresh_target_equilibria:
                self.target_equilibrium_refresh_count += 1
            else:
                self.target_equilibrium_cache_only_count += 1

            next_values = np.zeros(
                (states_t.shape[0], self.config.num_agents), dtype=np.float32
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
                policies_batch = self._solve_sre_batch(
                    next_online[nonterminal_t],
                    states=next_states_arr[nonterminal_indices],
                    allow_solver=refresh_target_equilibria,
                    allow_cache_reuse=not refresh_target_equilibria,
                )
                next_values[nonterminal_indices] = self._sre_target_values_batch(
                    next_target[nonterminal_indices],
                    policies_batch,
                )

            next_values_t = torch.as_tensor(
                next_values, device=self.device
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
            "epsilon_robust": self.config.epsilon_robust,
            "epsilon_explore": self.config.epsilon_explore,
            "gamma": self.config.gamma,
            "decay_rate": self.config.decay_rate,
            "learning_starts": self.config.learning_starts,
            "grad_clip_norm": self.config.grad_clip_norm,
            "obs_dim": self.config.obs_dim,
            "num_actions": self.config.num_actions,
            "num_agents": self.config.num_agents,
            "agent_id": self.config.agent_id,
            "sre_num_repeats": self.config.sre_num_repeats,
            "sre_include_pure_starts": self.config.sre_include_pure_starts,
            "train_every": self.config.train_every,
            "network_type": self.config.network_type,
            "q_hidden_dims": tuple(self.config.q_hidden_dims),
            "sre_policy_cache_enabled": self.config.sre_policy_cache_enabled,
            "sre_policy_cache_size": self.config.sre_policy_cache_size,
            "sre_policy_cache_round_digits": self.config.sre_policy_cache_round_digits,
            "sre_state_cache_round_digits": self.config.sre_state_cache_round_digits,
            "sre_approx_cache_enabled": self.config.sre_approx_cache_enabled,
            "sre_cache_exploitability_tol": self.config.sre_cache_exploitability_tol,
            "sre_solver_exploitability_tol": self.config.sre_solver_exploitability_tol,
            "sre_approx_accept_tol": self.config.sre_approx_accept_tol,
            "sre_solver_early_exit": self.config.sre_solver_early_exit,
            "sre_candidate_selection": self.config.sre_candidate_selection,
            "sre_exploitability_filter_enabled": self.config.sre_exploitability_filter_enabled,
            "sre_target_value_mode": self.config.sre_target_value_mode,
            "sre_uniform_fallback_enabled": self.config.sre_uniform_fallback_enabled,
            "target_equilibrium_update_steps": self.config.target_equilibrium_update_steps,
            "update_calls": self._update_calls,
            "buffer_size": self.replay_buffer.buffer.maxlen,
            "update_times": list(self.update_times),
            "sre_solve_time_count": self.sre_solve_time_count,
            "sre_solve_time_sum": self.sre_solve_time_sum,
            "sre_solve_time_sumsq": self.sre_solve_time_sumsq,
            "sre_solve_time_min": self.sre_solve_time_min,
            "sre_solve_time_max": self.sre_solve_time_max,
            "target_equilibrium_refresh_count": self.target_equilibrium_refresh_count,
            "target_equilibrium_cache_only_count": self.target_equilibrium_cache_only_count,
            "target_equilibrium_cache_only_misses": self.target_equilibrium_cache_only_misses,
            "target_equilibrium_stale_policy_reuses": self.target_equilibrium_stale_policy_reuses,
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
        cfg.epsilon_robust = checkpoint.get("epsilon_robust", cfg.epsilon_robust)
        cfg.epsilon_explore = checkpoint.get("epsilon_explore", cfg.epsilon_explore)
        cfg.gamma = checkpoint.get("gamma", cfg.gamma)
        cfg.decay_rate = checkpoint.get("decay_rate", cfg.decay_rate)
        cfg.learning_starts = checkpoint.get("learning_starts", cfg.learning_starts)
        cfg.grad_clip_norm = checkpoint.get("grad_clip_norm", cfg.grad_clip_norm)
        cfg.num_agents = int(checkpoint.get("num_agents", cfg.num_agents))
        cfg.sre_num_repeats = checkpoint.get("sre_num_repeats", cfg.sre_num_repeats)
        cfg.sre_include_pure_starts = bool(checkpoint.get("sre_include_pure_starts", cfg.sre_include_pure_starts))
        cfg.train_every = max(1, int(checkpoint.get("train_every", cfg.train_every)))
        cfg.network_type = checkpoint.get("network_type", cfg.network_type)
        cfg.q_hidden_dims = tuple(checkpoint.get("q_hidden_dims", cfg.q_hidden_dims))
        cfg.sre_policy_cache_enabled = bool(checkpoint.get("sre_policy_cache_enabled", cfg.sre_policy_cache_enabled))
        cfg.sre_policy_cache_size = int(checkpoint.get("sre_policy_cache_size", cfg.sre_policy_cache_size))
        cfg.sre_policy_cache_round_digits = int(checkpoint.get("sre_policy_cache_round_digits", cfg.sre_policy_cache_round_digits))
        cfg.sre_state_cache_round_digits = int(checkpoint.get("sre_state_cache_round_digits", cfg.sre_state_cache_round_digits))
        cfg.sre_approx_cache_enabled = bool(checkpoint.get("sre_approx_cache_enabled", cfg.sre_approx_cache_enabled))
        cfg.sre_cache_exploitability_tol = float(checkpoint.get("sre_cache_exploitability_tol", cfg.sre_cache_exploitability_tol))
        cfg.sre_solver_exploitability_tol = float(checkpoint.get("sre_solver_exploitability_tol", cfg.sre_solver_exploitability_tol))
        cfg.sre_approx_accept_tol = float(checkpoint.get("sre_approx_accept_tol", cfg.sre_approx_accept_tol))
        cfg.sre_solver_early_exit = bool(checkpoint.get("sre_solver_early_exit", cfg.sre_solver_early_exit))
        cfg.sre_candidate_selection = checkpoint.get("sre_candidate_selection", cfg.sre_candidate_selection)
        cfg.sre_exploitability_filter_enabled = bool(checkpoint.get("sre_exploitability_filter_enabled", cfg.sre_exploitability_filter_enabled))
        cfg.sre_target_value_mode = checkpoint.get("sre_target_value_mode", cfg.sre_target_value_mode)
        cfg.sre_uniform_fallback_enabled = bool(checkpoint.get("sre_uniform_fallback_enabled", cfg.sre_uniform_fallback_enabled))
        cfg.target_equilibrium_update_steps = max(1, int(checkpoint.get("target_equilibrium_update_steps", cfg.target_equilibrium_update_steps)))
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
        self.target_equilibrium_refresh_count = int(checkpoint.get("target_equilibrium_refresh_count", self.target_equilibrium_refresh_count))
        self.target_equilibrium_cache_only_count = int(checkpoint.get("target_equilibrium_cache_only_count", self.target_equilibrium_cache_only_count))
        self.target_equilibrium_cache_only_misses = int(checkpoint.get("target_equilibrium_cache_only_misses", self.target_equilibrium_cache_only_misses))
        self.target_equilibrium_stale_policy_reuses = int(checkpoint.get("target_equilibrium_stale_policy_reuses", self.target_equilibrium_stale_policy_reuses))

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
