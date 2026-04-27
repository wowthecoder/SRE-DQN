import random
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from path_solver import PathSolverWrapper, solve_strategically_robust_bimatrix_game_path_lcp


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


class DuelingJointQNetwork(nn.Module):
    """
    Dueling network over joint actions for 2-player games.
    Output shape: [batch, num_actions, num_actions, num_agents].
    """

    def __init__(self, obs_dim, num_actions, num_agents):
        super().__init__()
        if num_agents != 2:
            raise ValueError("This network currently supports only num_agents == 2.")

        self.num_actions = num_actions
        self.num_agents = num_agents
        self.joint_action_count = num_actions ** num_agents

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
        return q_joint.view(-1, self.num_actions, self.num_actions, self.num_agents)


class DuelingDoubleDqnSreAgent:
    """
    Dueling Double DQN agent that uses SRE policies for action selection and targets.
    """

    def __init__(
        self,
        agent_id,
        obs_dim,
        num_agents,
        num_actions,
        pathwrap_path="pathwrap.so",
        epsilon_robust=1.0,
        epsilon_explore=1.0,
        lr=3e-4,
        gamma=0.9,
        decay_rate=0.999,
        buffer_size=10000,
        learning_starts=1000,
        grad_clip_norm=10.0,
        use_gpu=True,
        sre_num_repeats=20,
        sre_cache_size=4096,
        train_every=1,
    ):
        if num_agents != 2:
            raise ValueError("This implementation currently supports only 2 agents.")

        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.epsilon_robust = epsilon_robust
        self.epsilon_explore = epsilon_explore
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.learning_starts = learning_starts
        self.grad_clip_norm = grad_clip_norm
        self.sre_num_repeats = sre_num_repeats
        self.sre_cache_size = sre_cache_size
        self._sre_cache = OrderedDict()
        self.train_every = max(1, int(train_every))
        self._update_calls = 0

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.q_net = DuelingJointQNetwork(obs_dim, num_actions, num_agents).to(self.device)
        self.target_net = DuelingJointQNetwork(obs_dim, num_actions, num_agents).to(
            self.device
        )
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.update_times = []
        self.sre_solve_time_count = 0
        self.sre_solve_time_sum = 0.0
        self.sre_solve_time_sumsq = 0.0
        self.sre_solve_time_min = None
        self.sre_solve_time_max = None

        self.path_solver = PathSolverWrapper(pathwrap_path)

    def _record_sre_solve_time(self, duration):
        self.sre_solve_time_count += 1
        self.sre_solve_time_sum += duration
        self.sre_solve_time_sumsq += duration * duration
        if self.sre_solve_time_min is None or duration < self.sre_solve_time_min:
            self.sre_solve_time_min = duration
        if self.sre_solve_time_max is None or duration > self.sre_solve_time_max:
            self.sre_solve_time_max = duration

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
        if vector.shape[0] != self.obs_dim:
            raise ValueError(
                f"Expected state vector length {self.obs_dim}, got {vector.shape[0]}."
            )
        return vector

    def _normalize_policy(self, policy):
        p = np.asarray(policy, dtype=np.float32)
        p = np.clip(p, 0.0, None)
        s = float(p.sum())
        if s <= 0.0:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        return p / s

    def _uniform_policies(self):
        u = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        return [u.copy(), u.copy()]

    def _sre_cache_key(self, q_tensor):
        q_key = np.ascontiguousarray(np.round(q_tensor, 6), dtype=np.float32)
        return (round(float(self.epsilon_robust), 6), int(self.sre_num_repeats), q_key.tobytes())

    def _cache_sre_policies(self, key, policies):
        if self.sre_cache_size <= 0:
            return
        self._sre_cache[key] = [policy.copy() for policy in policies]
        self._sre_cache.move_to_end(key)
        while len(self._sre_cache) > self.sre_cache_size:
            self._sre_cache.popitem(last=False)

    def _solve_sre(self, q_tensor):
        cache_key = self._sre_cache_key(q_tensor)
        cached = self._sre_cache.get(cache_key)
        if cached is not None:
            self._sre_cache.move_to_end(cache_key)
            return [policy.copy() for policy in cached]

        u1 = q_tensor[:, :, 0]
        u2 = q_tensor[:, :, 1]

        solve_start = time.perf_counter()
        try:
            results = solve_strategically_robust_bimatrix_game_path_lcp(
                u1,
                u2,
                [self.epsilon_robust, self.epsilon_robust],
                self.sre_num_repeats,
                self.path_solver,
            )
            solutions = results[0]
        except Exception:
            self._record_sre_solve_time(time.perf_counter() - solve_start)
            return self._uniform_policies()
        self._record_sre_solve_time(time.perf_counter() - solve_start)

        if not solutions:
            policies = self._uniform_policies()
            self._cache_sre_policies(cache_key, policies)
            return policies

        best_joint_reward = -float("inf")
        best = None
        for sol in solutions:
            p1 = self._normalize_policy(sol["p1"])
            p2 = self._normalize_policy(sol["p2"])
            r1 = float(p1 @ u1 @ p2)
            r2 = float(p1 @ u2 @ p2)
            if r1 + r2 > best_joint_reward:
                best_joint_reward = r1 + r2
                best = [p1, p2]

        policies = best if best is not None else self._uniform_policies()
        self._cache_sre_policies(cache_key, policies)
        return policies

    def _sre_expected_values(self, q_tensor, policies):
        expected = q_tensor
        for policy in policies:
            expected = np.tensordot(policy, expected, axes=([0], [0]))
        return np.asarray(expected, dtype=np.float32)

    def act(self, state, agent_id=None):
        if agent_id is None:
            agent_id = self.agent_id
        if not 0 <= agent_id < self.num_agents:
            raise ValueError(f"Expected agent_id in [0, {self.num_agents}), got {agent_id}.")

        if np.random.rand() < self.epsilon_explore:
            return int(np.random.choice(self.num_actions))

        state_vec = self._state_to_vector(state)
        state_t = torch.as_tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_tensor = self.q_net(state_t).squeeze(0).detach().cpu().numpy()

        policies = self._solve_sre(q_tensor)
        my_policy = self._normalize_policy(policies[agent_id])
        return int(np.random.choice(self.num_actions, p=my_policy))

    def update(self, state, joint_actions, joint_rewards, next_state, done=False, batch_size=64):
        state_vec = self._state_to_vector(state)
        next_state_vec = self._state_to_vector(next_state)
        actions_arr = np.asarray(joint_actions, dtype=np.int64).reshape(-1)
        rewards_arr = np.asarray(joint_rewards, dtype=np.float32).reshape(-1)

        if actions_arr.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected joint action length {self.num_agents}, got {actions_arr.shape[0]}."
            )
        if rewards_arr.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected joint reward length {self.num_agents}, got {rewards_arr.shape[0]}."
            )

        self._update_calls += 1
        self.replay_buffer.push(state_vec, actions_arr, rewards_arr, next_state_vec, done)
        if self._update_calls % self.train_every != 0:
            return None
        return self.train_step(batch_size=batch_size)

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < max(batch_size, self.learning_starts):
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
        q_tensor = self.q_net(states_t)  # [B, A1, A2, N]
        batch_idx = torch.arange(states_t.shape[0], device=self.device)
        current_q = q_tensor[batch_idx, actions_t[:, 0], actions_t[:, 1], :]  # [B, N]

        with torch.no_grad():
            # Double-DQN style: choose policy from online net, evaluate with target net.
            next_online = self.q_net(next_states_t).detach().cpu().numpy()
            next_target = self.target_net(next_states_t).detach().cpu().numpy()

            next_values = []
            for i in range(states_t.shape[0]):
                policies = self._solve_sre(next_online[i])
                expected_next = self._sre_expected_values(next_target[i], policies)
                next_values.append(expected_next)

            next_values_t = torch.as_tensor(
                np.asarray(next_values, dtype=np.float32), device=self.device
            )
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_values_t

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.update_times.append(time.perf_counter() - update_start)
        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def soft_update_target_network(self, tau=0.005):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def decay_parameters(self):
        self.epsilon_robust *= self.decay_rate
        self.epsilon_explore *= self.decay_rate

    def save_checkpoint(self, path, include_replay_buffer=False):
        payload = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon_robust": self.epsilon_robust,
            "epsilon_explore": self.epsilon_explore,
            "gamma": self.gamma,
            "decay_rate": self.decay_rate,
            "learning_starts": self.learning_starts,
            "grad_clip_norm": self.grad_clip_norm,
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "agent_id": self.agent_id,
            "sre_num_repeats": self.sre_num_repeats,
            "sre_cache_size": self.sre_cache_size,
            "train_every": self.train_every,
            "update_calls": self._update_calls,
            "buffer_size": self.replay_buffer.buffer.maxlen,
            "update_times": list(self.update_times),
            "sre_solve_time_count": self.sre_solve_time_count,
            "sre_solve_time_sum": self.sre_solve_time_sum,
            "sre_solve_time_sumsq": self.sre_solve_time_sumsq,
            "sre_solve_time_min": self.sre_solve_time_min,
            "sre_solve_time_max": self.sre_solve_time_max,
        }
        if include_replay_buffer:
            payload["replay_buffer"] = list(self.replay_buffer.buffer)
        torch.save(payload, path)

    def load_checkpoint(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        if "q_net" in checkpoint:
            self.q_net.load_state_dict(checkpoint["q_net"])
        if "target_net" in checkpoint:
            self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epsilon_robust" in checkpoint:
            self.epsilon_robust = checkpoint["epsilon_robust"]
        if "epsilon_explore" in checkpoint:
            self.epsilon_explore = checkpoint["epsilon_explore"]
        if "gamma" in checkpoint:
            self.gamma = checkpoint["gamma"]
        if "decay_rate" in checkpoint:
            self.decay_rate = checkpoint["decay_rate"]
        if "learning_starts" in checkpoint:
            self.learning_starts = checkpoint["learning_starts"]
        if "grad_clip_norm" in checkpoint:
            self.grad_clip_norm = checkpoint["grad_clip_norm"]
        if "sre_num_repeats" in checkpoint:
            self.sre_num_repeats = checkpoint["sre_num_repeats"]
        if "sre_cache_size" in checkpoint:
            self.sre_cache_size = checkpoint["sre_cache_size"]
        if "train_every" in checkpoint:
            self.train_every = max(1, int(checkpoint["train_every"]))
        if "update_calls" in checkpoint:
            self._update_calls = int(checkpoint["update_calls"])
        replay_buffer_items = checkpoint.get("replay_buffer")
        if replay_buffer_items is not None:
            self.replay_buffer = ReplayBuffer(checkpoint.get("buffer_size", len(replay_buffer_items)))
            self.replay_buffer.buffer.extend(replay_buffer_items)
        if "update_times" in checkpoint:
            self.update_times = list(checkpoint["update_times"])
        if "sre_solve_time_count" in checkpoint:
            self.sre_solve_time_count = checkpoint["sre_solve_time_count"]
        if "sre_solve_time_sum" in checkpoint:
            self.sre_solve_time_sum = checkpoint["sre_solve_time_sum"]
        if "sre_solve_time_sumsq" in checkpoint:
            self.sre_solve_time_sumsq = checkpoint["sre_solve_time_sumsq"]
        if "sre_solve_time_min" in checkpoint:
            self.sre_solve_time_min = checkpoint["sre_solve_time_min"]
        if "sre_solve_time_max" in checkpoint:
            self.sre_solve_time_max = checkpoint["sre_solve_time_max"]

    def close(self):
        if hasattr(self, "path_solver") and self.path_solver is not None:
            self.path_solver.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def flatten_gridworld_obs(obs):
    return np.asarray([coord for pos in obs for coord in pos], dtype=np.float32)
