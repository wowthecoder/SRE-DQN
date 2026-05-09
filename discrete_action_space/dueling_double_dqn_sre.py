import random
import sys
import time
from collections import deque
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

from sre_solvers import IterativeNPlayerSreSolver, PathCBimatrixSreSolver


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
    sre_num_repeats: int = 20
    sre_include_pure_starts: bool = True
    train_every: int = 1
    sre_solver: Any = None
    network_type: str = "joint_output"
    target_tau: Optional[float] = None
    target_update_steps: int = 100
    action_epsilon_start: float = 1.0
    action_epsilon_end: float = 0.05
    action_epsilon_decay_fraction: float = 0.5
    sre_solver_name: str = "path_c_pool"
    sre_solver_workers: int = 8
    sre_solver_start_method: Optional[str] = None
    epsilon_robust_initial: float = 1.0
    epsilon_schedule: str = "exponential"


class DuelingJointQNetwork(nn.Module):
    """
    Dueling network over joint actions for N-player games.
    Output shape: [batch, num_actions, ..., num_actions, num_agents].
    """

    def __init__(self, obs_dim, num_actions, num_agents):
        super().__init__()

        self.num_actions = num_actions
        self.num_agents = num_agents
        self.joint_action_count = num_actions ** num_agents
        self.output_shape = [num_actions] * num_agents + [num_agents]

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(128, num_agents)
        self.adv_head = nn.Linear(128, self.joint_action_count * num_agents)

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

    def __init__(self, obs_dim, num_actions, num_agents):
        super().__init__()

        self.num_actions = num_actions
        self.num_agents = num_agents
        self.joint_action_count = num_actions ** num_agents
        self.output_shape = [num_actions] * num_agents + [num_agents]
        self.critics = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    _DuelingPayoffHead(128, self.joint_action_count),
                )
                for _ in range(num_agents)
            ]
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

    def __init__(self, obs_dim, num_actions, num_agents):
        super().__init__()

        self.num_actions = num_actions
        self.num_agents = num_agents
        self.joint_action_count = num_actions ** num_agents
        self.output_shape = [num_actions] * num_agents + [num_agents]
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [
                _DuelingPayoffHead(128, self.joint_action_count)
                for _ in range(num_agents)
            ]
        )

    def forward(self, state):
        features = self.feature(state)
        q_by_agent = [head(features) for head in self.heads]
        q_joint = torch.stack(q_by_agent, dim=-1)  # [B, |A_joint|, N]
        return q_joint.view(-1, *self.output_shape)


def make_q_network(obs_dim, num_actions, num_agents, network_type="joint_output"):
    if network_type == "joint_output":
        return DuelingJointQNetwork(obs_dim, num_actions, num_agents)
    if network_type == "per_agent_independent":
        return DuelingPerAgentJointQNetwork(obs_dim, num_actions, num_agents)
    if network_type == "shared_trunk_separate_heads":
        return DuelingSharedTrunkPerAgentJointQNetwork(
            obs_dim, num_actions, num_agents
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
        ).to(self.device)
        self.target_net = make_q_network(
            config.obs_dim,
            config.num_actions,
            config.num_agents,
            network_type=config.network_type,
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

        if config.sre_solver is None:
            if config.num_agents == 2:
                sre_solver = PathCBimatrixSreSolver(pathwrap_path=config.pathwrap_path)
            else:
                sre_solver = IterativeNPlayerSreSolver()
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

    def _sre_batch_key(self, q_tensor):
        q_key = np.ascontiguousarray(np.round(q_tensor, 6), dtype=np.float32)
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

    def _solve_sre(self, q_tensor):
        solve_start = time.perf_counter()
        try:
            try:
                result = self.sre_solver.solve(
                    q_tensor,
                    epsilon=self.config.epsilon_robust,
                    num_repeats=self.config.sre_num_repeats,
                    include_pure_starts=self.config.sre_include_pure_starts,
                )
            except TypeError as exc:
                if "include_pure_starts" not in str(exc):
                    raise
                result = self.sre_solver.solve(
                    q_tensor,
                    epsilon=self.config.epsilon_robust,
                    num_repeats=self.config.sre_num_repeats,
                )
        except Exception:
            self._record_sre_solve_time(time.perf_counter() - solve_start)
            return self._uniform_policies()
        self._record_sre_solve_time(time.perf_counter() - solve_start)

        if not result.success or not result.policies:
            return self._uniform_policies()

        policies = [self._normalize_policy(policy) for policy in result.policies]
        if (
            len(policies) != self.config.num_agents
            or any(policy.shape[0] != self.config.num_actions for policy in policies)
        ):
            policies = self._uniform_policies()
        return policies

    def _solve_sre_batch(self, q_tensors):
        q_tensors = np.asarray(q_tensors, dtype=np.float32)
        expected_ndim = self.config.num_agents + 2
        if q_tensors.ndim != expected_ndim:
            raise ValueError(
                "Expected q_tensors with shape [B, A1, ..., AN, N], "
                f"got {q_tensors.shape}."
            )
        if tuple(q_tensors.shape[1:]) != self.q_tensor_shape:
            raise ValueError(
                f"Expected per-sample Q tensor shape {self.q_tensor_shape}, "
                f"got {q_tensors.shape[1:]}."
            )

        policies_by_index = [None] * q_tensors.shape[0]
        unique_q_tensors = []
        unique_keys = []
        key_to_unique_index = {}

        for batch_index, q_tensor in enumerate(q_tensors):
            batch_key = self._sre_batch_key(q_tensor)
            unique_index = key_to_unique_index.get(batch_key)
            if unique_index is None:
                unique_index = len(unique_q_tensors)
                key_to_unique_index[batch_key] = unique_index
                unique_q_tensors.append(q_tensor)
                unique_keys.append(batch_key)
            policies_by_index[batch_index] = unique_index

        if unique_q_tensors:
            solve_start = time.perf_counter()
            try:
                if hasattr(self.sre_solver, "solve_batch"):
                    try:
                        results = self.sre_solver.solve_batch(
                            unique_q_tensors,
                            epsilon=self.config.epsilon_robust,
                            num_repeats=self.config.sre_num_repeats,
                            include_pure_starts=self.config.sre_include_pure_starts,
                        )
                    except TypeError as exc:
                        if "include_pure_starts" not in str(exc):
                            raise
                        results = self.sre_solver.solve_batch(
                            unique_q_tensors,
                            epsilon=self.config.epsilon_robust,
                            num_repeats=self.config.sre_num_repeats,
                        )
                else:
                    results = [
                        self._solve_sre_result(q_tensor)
                        for q_tensor in unique_q_tensors
                    ]
            except Exception:
                elapsed = time.perf_counter() - solve_start
                self._record_sre_solve_time(elapsed, count=len(unique_q_tensors))
                results = [None] * len(unique_q_tensors)
            else:
                elapsed = time.perf_counter() - solve_start
                self._record_sre_solve_time(elapsed, count=len(unique_q_tensors))

            unique_policies = []
            for q_tensor, batch_key, result in zip(unique_q_tensors, unique_keys, results):
                if result is None or not result.success or not result.policies:
                    policies = self._uniform_policies()
                else:
                    policies = [
                        self._normalize_policy(policy) for policy in result.policies
                    ]
                    if (
                        len(policies) != self.config.num_agents
                        or any(
                            policy.shape[0] != self.config.num_actions
                            for policy in policies
                        )
                    ):
                        policies = self._uniform_policies()
                unique_policies.append(policies)

            for batch_index, entry in enumerate(policies_by_index):
                if isinstance(entry, int):
                    policies_by_index[batch_index] = [
                        policy.copy() for policy in unique_policies[entry]
                    ]

        return policies_by_index

    def _solve_sre_result(self, q_tensor):
        try:
            return self.sre_solver.solve(
                q_tensor,
                epsilon=self.config.epsilon_robust,
                num_repeats=self.config.sre_num_repeats,
                include_pure_starts=self.config.sre_include_pure_starts,
            )
        except TypeError as exc:
            if "include_pure_starts" not in str(exc):
                raise
            return self.sre_solver.solve(
                q_tensor,
                epsilon=self.config.epsilon_robust,
                num_repeats=self.config.sre_num_repeats,
            )

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

        policies = self._solve_sre(q_tensor)
        my_policy = self._normalize_policy(policies[agent_id])
        return int(np.random.choice(self.config.num_actions, p=my_policy))

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
            next_online = self.q_net(next_states_t).detach().cpu().numpy()
            next_target = self.target_net(next_states_t).detach().cpu().numpy()

            policies_batch = self._solve_sre_batch(next_online)
            next_values = self._sre_expected_values_batch(next_target, policies_batch)

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
            "update_calls": self._update_calls,
            "buffer_size": self.replay_buffer.buffer.maxlen,
            "update_times": list(self.update_times),
            "sre_solve_time_count": self.sre_solve_time_count,
            "sre_solve_time_sum": self.sre_solve_time_sum,
            "sre_solve_time_sumsq": self.sre_solve_time_sumsq,
            "sre_solve_time_min": self.sre_solve_time_min,
            "sre_solve_time_max": self.sre_solve_time_max,
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
