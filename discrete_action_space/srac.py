import random
import sys
import time
from collections import deque
from dataclasses import dataclass, fields
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sre_solvers import make_sre_solver, robust_policy_values


def _linear_schedule(start, end, step, total_steps):
    if total_steps <= 1:
        return float(end)
    fraction = min(max(step, 0) / float(total_steps - 1), 1.0)
    return float(start + fraction * (end - start))


class SracReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def push(
        self,
        state,
        local_obs,
        joint_actions,
        joint_rewards,
        next_state,
        next_local_obs,
        done,
        action_masks=None,
        next_action_masks=None,
    ):
        self.buffer.append(
            (
                state,
                local_obs,
                joint_actions,
                joint_rewards,
                next_state,
                next_local_obs,
                done,
                action_masks,
                next_action_masks,
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, int(batch_size))
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.buffer)


@dataclass
class SracConfig:
    state_dim: int = 4
    actor_obs_dim: int = 4
    num_agents: int = 2
    num_actions: int = 4
    actor_hidden_dims: tuple = (128, 128)
    critic_hidden_dims: tuple = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 20000
    batch_size: int = 32
    learning_starts: int = 500
    train_every: int = 4
    actor_update_every: int = 1
    grad_clip_norm: float = 10.0
    use_gpu: bool = True
    epsilon_robust: float = 0.5
    epsilon_robust_initial: float = 0.5
    epsilon_schedule: str = "constant"
    decay_rate: float = 0.999
    epsilon_explore: float = 1.0
    action_epsilon_start: float = 1.0
    action_epsilon_end: float = 0.05
    action_epsilon_decay_fraction: float = 0.6
    target_tau: Optional[float] = None
    target_update_steps: int = 250
    pathwrap_path: str = "pathwrap.so"
    sre_solver_name: str = "path_mcp_nplayer"
    sre_solver_workers: int = 4
    sre_solver_start_method: Optional[str] = None
    sre_solver: Any = None
    sre_num_repeats: int = 5
    sre_include_pure_starts: bool = False
    sre_solver_exploitability_tol: float = 1e-4
    sre_approx_accept_tol: float = 1e-2
    sre_exploitability_filter_enabled: bool = True
    sre_solver_early_exit: bool = True
    sre_candidate_selection: str = "robust_exploitability"
    sre_target_value_mode: str = "nominal"
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


def _make_mlp(input_dim, hidden_dims, output_dim):
    dims = [int(input_dim)] + [int(dim) for dim in hidden_dims]
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"MLP dimensions must be positive, got {dims}.")
    layers = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
    layers.append(nn.Linear(dims[-1], int(output_dim)))
    return nn.Sequential(*layers)


class SracActor(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dims=(128, 128)):
        super().__init__()
        self.net = _make_mlp(obs_dim, hidden_dims, num_actions)

    def forward(self, obs):
        return self.net(obs)


class SracQueryCritic(nn.Module):
    def __init__(self, state_dim, num_agents, num_actions, hidden_dims=(256, 256)):
        super().__init__()
        self.num_agents = int(num_agents)
        self.num_actions = int(num_actions)
        input_dim = int(state_dim) + self.num_agents * self.num_actions
        self.net = _make_mlp(input_dim, hidden_dims, self.num_agents)

    def forward(self, states, joint_actions):
        if joint_actions.dtype != torch.long:
            joint_actions = joint_actions.long()
        one_hot = torch.nn.functional.one_hot(
            joint_actions,
            num_classes=self.num_actions,
        ).to(dtype=states.dtype)
        flat_actions = one_hot.reshape(joint_actions.shape[0], -1)
        return self.net(torch.cat([states, flat_actions], dim=-1))


class SracAgent:
    def __init__(self, config: SracConfig):
        self.config = config
        self.config.train_every = max(1, int(config.train_every))
        self.config.actor_update_every = max(1, int(config.actor_update_every))
        self._update_calls = 0
        self.gradient_steps = 0
        self.sre_solve_time_count = 0
        self.sre_solve_time_sum = 0.0
        self.sre_solve_time_sumsq = 0.0
        self.sre_solve_time_min = None
        self.sre_solve_time_max = None
        self.sre_solver_failures = 0
        self.sre_candidates_used = 0
        self.sre_candidates_skipped = 0

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.actors = nn.ModuleList(
            [
                SracActor(
                    config.actor_obs_dim,
                    config.num_actions,
                    config.actor_hidden_dims,
                )
                for _ in range(config.num_agents)
            ]
        ).to(self.device)
        self.critic = SracQueryCritic(
            config.state_dim,
            config.num_agents,
            config.num_actions,
            config.critic_hidden_dims,
        ).to(self.device)
        self.target_critic = SracQueryCritic(
            config.state_dim,
            config.num_agents,
            config.num_actions,
            config.critic_hidden_dims,
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay_buffer = SracReplayBuffer(config.buffer_size)

        if config.sre_solver is None:
            self.sre_solver = make_sre_solver(
                config.sre_solver_name,
                pathwrap_path=config.pathwrap_path,
                max_workers=config.sre_solver_workers,
                start_method=config.sre_solver_start_method,
            )
        else:
            self.sre_solver = config.sre_solver

    def _record_sre_solve_time(self, duration, count=1):
        count = max(1, int(count))
        per_solve = float(duration) / count
        self.sre_solve_time_count += count
        self.sre_solve_time_sum += float(duration)
        self.sre_solve_time_sumsq += count * per_solve * per_solve
        if self.sre_solve_time_min is None or per_solve < self.sre_solve_time_min:
            self.sre_solve_time_min = per_solve
        if self.sre_solve_time_max is None or per_solve > self.sre_solve_time_max:
            self.sre_solve_time_max = per_solve

    def get_sre_solve_time_summary(self):
        count = int(self.sre_solve_time_count)
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
            "count": count,
            "mean_seconds": float(mean),
            "min_seconds": float(self.sre_solve_time_min),
            "max_seconds": float(self.sre_solve_time_max),
            "std_seconds": std,
            "mean_microseconds": float(mean * 1_000_000.0),
            "min_microseconds": float(self.sre_solve_time_min * 1_000_000.0),
            "max_microseconds": float(self.sre_solve_time_max * 1_000_000.0),
            "std_microseconds": float(std * 1_000_000.0),
        }

    def get_usage_summary(self):
        return {
            "algorithm": "srac",
            "sre_candidates_used": int(self.sre_candidates_used),
            "sre_candidates_skipped": int(self.sre_candidates_skipped),
            "sre_solver_failures": int(self.sre_solver_failures),
            "solve_time": self.get_sre_solve_time_summary(),
        }

    def _state_to_vector(self, state):
        vector = np.asarray(state, dtype=np.float32).reshape(-1)
        if vector.shape[0] != int(self.config.state_dim):
            raise ValueError(
                f"Expected state vector length {self.config.state_dim}, got {vector.shape[0]}."
            )
        return vector

    def _local_obs_to_matrix(self, local_obs):
        matrix = np.asarray(local_obs, dtype=np.float32)
        matrix = matrix.reshape(int(self.config.num_agents), -1)
        if matrix.shape[1] != int(self.config.actor_obs_dim):
            raise ValueError(
                f"Expected local obs dim {self.config.actor_obs_dim}, got {matrix.shape[1]}."
            )
        return matrix

    def _normalize_action_masks(self, action_masks):
        if action_masks is None:
            return None
        masks = [np.asarray(mask, dtype=bool).reshape(-1) for mask in action_masks]
        if len(masks) != int(self.config.num_agents):
            raise ValueError(
                f"Expected {self.config.num_agents} action masks, got {len(masks)}."
            )
        normalized = []
        for agent_id, mask in enumerate(masks):
            if mask.shape[0] != int(self.config.num_actions):
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
        masks_list = list(action_masks_batch)
        if len(masks_list) != int(batch_size):
            raise ValueError(
                f"Expected {batch_size} action-mask entries, got {len(masks_list)}."
            )
        return [self._normalize_action_masks(masks) for masks in masks_list]

    def _sample_action_with_mask(self, probs, mask):
        if mask is None:
            return int(np.random.choice(len(probs), p=probs))
        valid = np.flatnonzero(mask)
        masked = np.zeros_like(probs, dtype=np.float64)
        masked[valid] = probs[valid]
        total = float(masked.sum())
        if total <= 0.0:
            return int(np.random.choice(valid))
        return int(np.random.choice(len(probs), p=masked / total))

    def _actor_policy(self, actor_id, obs, mask=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actors[int(actor_id)](obs_t).squeeze(0)
            if mask is not None:
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                logits = logits.masked_fill(~mask_t, -1e9)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs = np.clip(probs.astype(np.float64), 0.0, None)
        total = float(probs.sum())
        if total <= 0.0:
            probs = np.full(int(self.config.num_actions), 1.0 / self.config.num_actions)
        else:
            probs /= total
        return probs

    def act_joint(self, state, local_obs, action_masks=None):
        del state
        local_obs = self._local_obs_to_matrix(local_obs)
        masks = self._normalize_action_masks(action_masks)
        actions = []
        for agent_id in range(int(self.config.num_agents)):
            mask = None if masks is None else masks[agent_id]
            if np.random.rand() < float(self.config.epsilon_explore):
                valid = (
                    np.arange(int(self.config.num_actions))
                    if mask is None
                    else np.flatnonzero(mask)
                )
                actions.append(int(np.random.choice(valid)))
                continue
            policy = self._actor_policy(agent_id, local_obs[agent_id], mask=mask)
            actions.append(self._sample_action_with_mask(policy, mask))
        return actions

    def act_joint_batch(self, states, local_obs_batch, action_masks_batch=None):
        del states
        local_obs_batch = np.asarray(local_obs_batch, dtype=np.float32)
        batch_size = int(local_obs_batch.shape[0])
        masks_batch = self._normalize_action_masks_batch(action_masks_batch, batch_size)
        return [
            self.act_joint(
                None,
                local_obs_batch[batch_index],
                action_masks=masks_batch[batch_index],
            )
            for batch_index in range(batch_size)
        ]

    def _legal_action_indices(self, action_masks):
        if action_masks is None:
            return [
                np.arange(int(self.config.num_actions), dtype=np.int64)
                for _ in range(int(self.config.num_agents))
            ]
        return [np.flatnonzero(mask).astype(np.int64) for mask in action_masks]

    def _payoff_game_from_critic(self, critic, state, action_masks=None):
        state_vec = self._state_to_vector(state)
        action_indices = self._legal_action_indices(action_masks)
        joint_actions = np.asarray(list(product(*action_indices)), dtype=np.int64)
        if joint_actions.ndim != 2:
            joint_actions = joint_actions.reshape(-1, int(self.config.num_agents))
        states = np.repeat(state_vec[None, :], joint_actions.shape[0], axis=0)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(joint_actions, dtype=torch.long, device=self.device)
        with torch.no_grad():
            q_values = critic(states_t, actions_t).detach().cpu().numpy()
        q_tensor = q_values.reshape(
            *[len(indices) for indices in action_indices],
            int(self.config.num_agents),
        ).astype(np.float32)
        return q_tensor, action_indices

    def _call_sre_solver_batch(self, q_tensors):
        kwargs = {
            "epsilon": self.config.epsilon_robust,
            "num_repeats": self.config.sre_num_repeats,
            "include_pure_starts": self.config.sre_include_pure_starts,
            "exploitability_tol": self.config.sre_solver_exploitability_tol,
            "early_exit": self.config.sre_solver_early_exit,
            "candidate_selection": self.config.sre_candidate_selection,
        }
        while True:
            try:
                if hasattr(self.sre_solver, "solve_batch"):
                    return self.sre_solver.solve_batch(q_tensors, **kwargs)
                return [self.sre_solver.solve(q_tensor, **kwargs) for q_tensor in q_tensors]
            except TypeError as exc:
                message = str(exc)
                removed = False
                for key in (
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

    def _expand_legal_policies(self, legal_policies, action_indices):
        full = []
        if len(legal_policies) != int(self.config.num_agents):
            raise ValueError("Solver returned the wrong number of policies.")
        for policy, indices in zip(legal_policies, action_indices):
            policy = np.asarray(policy, dtype=np.float64).reshape(-1)
            if policy.shape[0] != len(indices):
                raise ValueError(
                    f"Expected policy length {len(indices)}, got {policy.shape[0]}."
                )
            target = np.zeros(int(self.config.num_actions), dtype=np.float32)
            clipped = np.clip(policy, 0.0, None)
            total = float(clipped.sum())
            if total <= 0.0:
                clipped = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
            else:
                clipped = clipped / total
            target[np.asarray(indices, dtype=np.int64)] = clipped.astype(np.float32)
            full.append(target)
        return full

    def _policies_from_result(self, result, action_indices):
        if result is None:
            self.sre_candidates_skipped += 1
            return None
        metadata = dict(getattr(result, "metadata", None) or {})
        if not getattr(result, "policies", None) or metadata.get("path_failed", False):
            self.sre_candidates_skipped += 1
            return None
        try:
            policies = self._expand_legal_policies(result.policies, action_indices)
        except ValueError:
            self.sre_candidates_skipped += 1
            return None
        if result.success or bool(getattr(self.sre_solver, "trust_approximate_policies", False)):
            self.sre_candidates_used += 1
            return policies
        if not bool(self.config.sre_exploitability_filter_enabled):
            self.sre_candidates_used += 1
            return policies
        gap = metadata.get("robust_exploitability")
        if gap is not None and float(gap) <= float(self.config.sre_approx_accept_tol):
            self.sre_candidates_used += 1
            return policies
        self.sre_candidates_skipped += 1
        return None

    def _solve_sre_games(self, q_tensors, action_indices_batch):
        if not q_tensors:
            return []
        solve_start = time.perf_counter()
        try:
            results = self._call_sre_solver_batch(q_tensors)
        except Exception:
            self.sre_solver_failures += len(q_tensors)
            self._record_sre_solve_time(time.perf_counter() - solve_start, len(q_tensors))
            return [None] * len(q_tensors)
        self._record_sre_solve_time(time.perf_counter() - solve_start, len(q_tensors))
        return [
            self._policies_from_result(result, action_indices)
            for result, action_indices in zip(results, action_indices_batch)
        ]

    def _expected_values(self, q_tensor, full_policies, action_indices):
        policies = [
            np.asarray(policy, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]
            for policy, indices in zip(full_policies, action_indices)
        ]
        expected = np.asarray(q_tensor, dtype=np.float32)
        for policy in policies:
            expected = np.tensordot(policy, expected, axes=([0], [0]))
        return np.asarray(expected, dtype=np.float32)

    def _target_values(self, q_tensor, full_policies, action_indices):
        mode = str(self.config.sre_target_value_mode).lower()
        if mode == "nominal":
            return self._expected_values(q_tensor, full_policies, action_indices)
        if mode == "robust":
            policies = [
                np.asarray(policy, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]
                for policy, indices in zip(full_policies, action_indices)
            ]
            return np.asarray(
                robust_policy_values(q_tensor, policies, self.config.epsilon_robust),
                dtype=np.float32,
            )
        raise ValueError(
            "sre_target_value_mode must be 'nominal' or 'robust', "
            f"got {self.config.sre_target_value_mode!r}."
        )

    def _critic_q_for_actions(self, states_t, actions_t):
        return self.critic(states_t, actions_t)

    def update(
        self,
        state,
        local_obs,
        joint_actions,
        joint_rewards,
        next_state,
        next_local_obs,
        done=False,
        batch_size=None,
        action_masks=None,
        next_action_masks=None,
    ):
        self._update_calls += 1
        self.replay_buffer.push(
            self._state_to_vector(state),
            self._local_obs_to_matrix(local_obs),
            np.asarray(joint_actions, dtype=np.int64).reshape(-1),
            np.asarray(joint_rewards, dtype=np.float32).reshape(-1),
            self._state_to_vector(next_state),
            self._local_obs_to_matrix(next_local_obs),
            np.asarray(done, dtype=np.float32),
            self._normalize_action_masks(action_masks),
            self._normalize_action_masks(next_action_masks),
        )
        if self._update_calls % int(self.config.train_every) != 0:
            return None
        return self.train_step(batch_size=batch_size or self.config.batch_size)

    def train_step(self, batch_size=None):
        batch_size = int(batch_size or self.config.batch_size)
        if len(self.replay_buffer) < max(batch_size, int(self.config.learning_starts)):
            return None

        (
            states,
            local_obs,
            actions,
            rewards,
            next_states,
            next_local_obs,
            dones,
            action_masks,
            next_action_masks,
        ) = self.replay_buffer.sample(batch_size)
        del next_local_obs
        states_arr = np.stack([self._state_to_vector(state) for state in states], axis=0)
        local_obs_arr = np.stack(
            [self._local_obs_to_matrix(obs) for obs in local_obs],
            axis=0,
        )
        actions_arr = np.stack(actions, axis=0).astype(np.int64)
        rewards_arr = np.stack(rewards, axis=0).astype(np.float32)
        next_states_arr = np.stack(
            [self._state_to_vector(state) for state in next_states],
            axis=0,
        )
        dones_arr = np.asarray(dones, dtype=np.float32)
        if dones_arr.ndim == 1:
            if dones_arr.shape[0] == batch_size:
                dones_arr = np.repeat(dones_arr[:, None], int(self.config.num_agents), axis=1)
            else:
                dones_arr = dones_arr.reshape(batch_size, int(self.config.num_agents))
        elif dones_arr.ndim == 0:
            dones_arr = np.full((batch_size, int(self.config.num_agents)), float(dones_arr))
        else:
            dones_arr = dones_arr.reshape(batch_size, int(self.config.num_agents))

        states_t = torch.as_tensor(states_arr, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions_arr, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards_arr, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones_arr, dtype=torch.float32, device=self.device)

        current_q = self._critic_q_for_actions(states_t, actions_t)
        target_values = np.zeros((batch_size, int(self.config.num_agents)), dtype=np.float32)
        valid_rows = np.all(dones_arr >= 1.0, axis=1)
        nonterminal_indices = np.flatnonzero(~valid_rows)

        next_masks_batch = self._normalize_action_masks_batch(next_action_masks, batch_size)
        q_tensors = []
        action_indices_batch = []
        pending_rows = []
        with torch.no_grad():
            for row in nonterminal_indices:
                q_tensor, action_indices = self._payoff_game_from_critic(
                    self.target_critic,
                    next_states_arr[int(row)],
                    next_masks_batch[int(row)],
                )
                q_tensors.append(q_tensor)
                action_indices_batch.append(action_indices)
                pending_rows.append(int(row))
        policies_batch = self._solve_sre_games(q_tensors, action_indices_batch)
        for row, q_tensor, action_indices, policies in zip(
            pending_rows,
            q_tensors,
            action_indices_batch,
            policies_batch,
        ):
            if policies is None:
                continue
            target_values[row] = self._target_values(q_tensor, policies, action_indices)
            valid_rows[row] = True

        if not np.any(valid_rows):
            return {
                "critic_loss": None,
                "actor_loss": None,
                "valid_critic_rows": 0,
                "valid_actor_rows": 0,
            }

        target_t = rewards_t + (1.0 - dones_t) * float(self.config.gamma) * torch.as_tensor(
            target_values,
            dtype=torch.float32,
            device=self.device,
        )
        valid_t = torch.as_tensor(valid_rows, dtype=torch.bool, device=self.device)
        critic_loss = torch.nn.functional.mse_loss(current_q[valid_t], target_t[valid_t])

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        self.critic_optimizer.step()

        self.gradient_steps += 1
        actor_loss_value = None
        actor_rows = 0
        if self.gradient_steps % int(self.config.actor_update_every) == 0:
            actor_loss, actor_rows = self._actor_imitation_loss(
                states_arr,
                local_obs_arr,
                action_masks,
            )
            if actor_loss is not None:
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.config.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.actors.parameters(),
                        self.config.grad_clip_norm,
                    )
                self.actor_optimizer.step()
                actor_loss_value = float(actor_loss.item())

        if self.config.target_tau is not None:
            self.soft_update_target_critic(float(self.config.target_tau))
        elif (
            self.config.target_update_steps
            and self.gradient_steps % int(self.config.target_update_steps) == 0
        ):
            self.update_target_critic()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "valid_critic_rows": int(np.sum(valid_rows)),
            "valid_actor_rows": int(actor_rows),
        }

    def _actor_imitation_loss(self, states_arr, local_obs_arr, action_masks_batch):
        masks_batch = self._normalize_action_masks_batch(
            action_masks_batch,
            int(states_arr.shape[0]),
        )
        q_tensors = []
        action_indices_batch = []
        for row in range(int(states_arr.shape[0])):
            q_tensor, action_indices = self._payoff_game_from_critic(
                self.critic,
                states_arr[row],
                masks_batch[row],
            )
            q_tensors.append(q_tensor)
            action_indices_batch.append(action_indices)
        policies_batch = self._solve_sre_games(q_tensors, action_indices_batch)

        losses = []
        valid_rows = 0
        for row, policies in enumerate(policies_batch):
            if policies is None:
                continue
            valid_rows += 1
            for agent_id, target_policy in enumerate(policies):
                obs_t = torch.as_tensor(
                    local_obs_arr[row, agent_id],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                logits = self.actors[agent_id](obs_t).squeeze(0)
                masks = masks_batch[row]
                if masks is not None:
                    mask_t = torch.as_tensor(
                        masks[agent_id],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    logits = logits.masked_fill(~mask_t, -1e9)
                log_probs = torch.log_softmax(logits, dim=-1)
                target_t = torch.as_tensor(
                    target_policy,
                    dtype=torch.float32,
                    device=self.device,
                )
                losses.append(-(target_t * log_probs).sum())
        if not losses:
            return None, 0
        return torch.stack(losses).mean(), valid_rows

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_target_critic(self, tau=0.005):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def decay_parameters(self, episode_idx, n_episodes):
        cfg = self.config
        if cfg.epsilon_schedule == "constant":
            cfg.epsilon_robust = float(cfg.epsilon_robust_initial)
        elif cfg.epsilon_schedule == "linear":
            cfg.epsilon_robust = _linear_schedule(
                cfg.epsilon_robust_initial,
                0.0,
                episode_idx,
                n_episodes,
            )
        elif cfg.epsilon_schedule == "exponential":
            cfg.epsilon_robust = float(cfg.epsilon_robust_initial) * (
                cfg.decay_rate ** int(episode_idx)
            )
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
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": _checkpoint_config_dict(self.config),
            "update_calls": self._update_calls,
            "gradient_steps": self.gradient_steps,
            "buffer_size": self.replay_buffer.buffer.maxlen,
            "sre_usage_summary": self.get_usage_summary(),
            "sre_solver_name": getattr(
                self.sre_solver,
                "name",
                type(self.sre_solver).__name__,
            ),
        }
        if include_replay_buffer:
            payload["replay_buffer"] = list(self.replay_buffer.buffer)
        torch.save(payload, path)

    def load_checkpoint(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)

        _restore_checkpoint_config(self.config, checkpoint.get("config", {}))
        if (state := checkpoint.get("actors")) is not None:
            self.actors.load_state_dict(state)
        if (state := checkpoint.get("critic")) is not None:
            self.critic.load_state_dict(state)
        if (state := checkpoint.get("target_critic")) is not None:
            self.target_critic.load_state_dict(state)
        if (state := checkpoint.get("actor_optimizer")) is not None:
            self.actor_optimizer.load_state_dict(state)
        if (state := checkpoint.get("critic_optimizer")) is not None:
            self.critic_optimizer.load_state_dict(state)

        self._update_calls = int(checkpoint.get("update_calls", self._update_calls))
        self.gradient_steps = int(checkpoint.get("gradient_steps", self.gradient_steps))
        replay_items = checkpoint.get("replay_buffer")
        if replay_items is not None:
            self.replay_buffer = SracReplayBuffer(
                checkpoint.get("buffer_size", len(replay_items))
            )
            self.replay_buffer.buffer.extend(replay_items)

    def close(self):
        if hasattr(self, "sre_solver") and self.sre_solver is not None:
            self.sre_solver.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
