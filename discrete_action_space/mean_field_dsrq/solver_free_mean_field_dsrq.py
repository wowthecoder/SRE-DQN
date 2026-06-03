"""Solver-free mean-field Deep SRQ for MAgent2-style discrete actions.

This module keeps the reference MF-Q shape: a critic conditioned on the local
observation and neighbour mean-action distribution.  The SRE part is local and
mean-field: action selection and Bellman targets use a distributionally robust
best response against a Wasserstein/TV ball around the observed mean action,
without constructing a representative bimatrix game or calling PATH.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linprog


class PairwiseMeanFieldQNetwork(nn.Module):
    """Reference-style MF-Q CNN producing pairwise mean-field payoffs."""

    def __init__(
        self,
        obs_channels: int,
        obs_height: int,
        obs_width: int,
        n_own_actions: int,
        n_mean_actions: int,
        feature_dim: int = 0,
    ):
        super().__init__()
        self.n_own_actions = int(n_own_actions)
        self.n_mean_actions = int(n_mean_actions)
        self.feature_dim = int(feature_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.obs_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * obs_height * obs_width, 256),
            nn.ReLU(),
        )
        if self.feature_dim > 0:
            self.feature_fc = nn.Sequential(
                nn.Linear(self.feature_dim, 32),
                nn.ReLU(),
            )
            head_input_dim = 256 + 32
        else:
            self.feature_fc = None
            head_input_dim = 256
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_own_actions * self.n_mean_actions),
        )

    def _feature_tensor(self, obs: torch.Tensor, feature: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.feature_dim <= 0:
            return None
        if feature is None:
            return torch.zeros(
                (obs.shape[0], self.feature_dim),
                dtype=obs.dtype,
                device=obs.device,
            )
        feature = feature.to(dtype=obs.dtype, device=obs.device)
        feature = feature.reshape(feature.shape[0], -1)
        if feature.shape[0] != obs.shape[0] or feature.shape[1] != self.feature_dim:
            raise ValueError(
                "Expected feature shape "
                f"[{obs.shape[0]}, {self.feature_dim}], got {tuple(feature.shape)}."
            )
        return feature

    def payoff_matrix(
        self,
        obs: torch.Tensor,
        feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obs_feat = self.obs_fc(self.conv(obs))
        feature_t = self._feature_tensor(obs, feature)
        if feature_t is not None:
            obs_feat = torch.cat([obs_feat, self.feature_fc(feature_t)], dim=-1)
        payoff = self.head(obs_feat)
        return payoff.reshape(-1, self.n_own_actions, self.n_mean_actions)

    def forward(
        self,
        obs: torch.Tensor,
        mean_action: torch.Tensor,
        feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        payoff = self.payoff_matrix(obs, feature)
        return torch.bmm(payoff, mean_action.unsqueeze(-1)).squeeze(-1)


class SolverFreeMFReplayBuffer:
    """Ring buffer of per-agent mean-field transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        mean_a: np.ndarray,
        next_mean_a: np.ndarray,
        done: bool,
        valid: bool = True,
        feature: Optional[np.ndarray] = None,
        next_feature: Optional[np.ndarray] = None,
    ) -> None:
        if feature is None:
            feature = np.zeros(0, dtype=np.float32)
        if next_feature is None:
            next_feature = np.zeros_like(feature, dtype=np.float32)
        self.buffer.append(
            (
                np.asarray(obs, dtype=np.float32),
                np.asarray(feature, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                np.asarray(next_feature, dtype=np.float32),
                _normalize_distribution(mean_a),
                _normalize_distribution(next_mean_a),
                float(done),
                float(valid),
            )
        )

    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
        idxs = np.random.randint(0, len(self.buffer), size=int(batch_size))
        batch = [self.buffer[int(i)] for i in idxs]
        obs, features, actions, rewards, next_obs, next_features, mean_a, next_mean_a, dones, valids = zip(*batch)

        def tensor(values, dtype):
            out = torch.as_tensor(np.stack(values), dtype=dtype)
            return out.to(device) if device is not None else out

        return {
            "obs": tensor(obs, torch.float32),
            "feature": tensor(features, torch.float32),
            "action": tensor(actions, torch.long),
            "reward": tensor(rewards, torch.float32),
            "next_obs": tensor(next_obs, torch.float32),
            "next_feature": tensor(next_features, torch.float32),
            "mean_a": tensor(mean_a, torch.float32),
            "next_mean_a": tensor(next_mean_a, torch.float32),
            "done": tensor(dones, torch.float32),
            "valid": tensor(valids, torch.float32),
        }

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass(frozen=True)
class RobustMeanFieldResult:
    policy: np.ndarray
    value: float
    worst_mean: np.ndarray
    lambda_value: float
    success: bool
    message: str = ""


def _normalize_distribution(values: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64).reshape(-1)
    if size is not None and p.size != int(size):
        raise ValueError(f"Expected distribution length {size}, got {p.size}.")
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        n = int(size) if size is not None else max(int(p.size), 1)
        return np.full(n, 1.0 / n, dtype=np.float32)
    return (p / total).astype(np.float32)


def _tv_worst_case_mean(mu: np.ndarray, values: np.ndarray, epsilon: float) -> np.ndarray:
    p = _normalize_distribution(mu).astype(np.float64)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if p.size != v.size:
        raise ValueError(f"Expected values length {p.size}, got {v.size}.")
    budget = float(np.clip(epsilon, 0.0, 1.0))
    if budget <= 0.0 or p.size <= 1:
        return p.astype(np.float32)

    q = p.copy()
    high_order = np.argsort(-v)
    low_order = np.argsort(v)
    high_pos = 0
    low_pos = 0
    while budget > 1e-12 and high_pos < p.size and low_pos < p.size:
        hi = int(high_order[high_pos])
        lo = int(low_order[low_pos])
        if v[hi] <= v[lo] + 1e-12:
            break
        movable = min(q[hi], 1.0 - q[lo], budget)
        if movable <= 1e-12:
            if q[hi] <= 1e-12:
                high_pos += 1
            if q[lo] >= 1.0 - 1e-12:
                low_pos += 1
            continue
        q[hi] -= movable
        q[lo] += movable
        budget -= movable
        if q[hi] <= 1e-12:
            high_pos += 1
        if q[lo] >= 1.0 - 1e-12:
            low_pos += 1
    return _normalize_distribution(q, p.size)


def _tv_worst_case_value(mu: np.ndarray, values: np.ndarray, epsilon: float) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    q = _tv_worst_case_mean(mu, v, epsilon).astype(np.float64)
    return float(q @ v)


class RobustMeanFieldSreOperator:
    """Local robust best-response oracle over a neighbour mean distribution."""

    def __init__(
        self,
        num_actions: int,
        mean_actions: Optional[int] = None,
        *,
        epsilon: float = 0.1,
        distance: str = "tv",
        fallback: str = "greedy_tv",
    ):
        if distance != "tv":
            raise ValueError("v1 solver-free MF-SRQ supports distance='tv' only.")
        if fallback != "greedy_tv":
            raise ValueError("v1 solver-free MF-SRQ supports fallback='greedy_tv' only.")
        self.num_actions = int(num_actions)
        self.mean_actions = int(mean_actions if mean_actions is not None else num_actions)
        self.epsilon = float(epsilon)
        self.distance = distance
        self.fallback = fallback
        self.cost = np.ones((self.mean_actions, self.mean_actions), dtype=np.float64)
        np.fill_diagonal(self.cost, 0.0)
        self._lambda_idx = self.num_actions
        self._beta_start = self.num_actions + 1
        self._n_vars = self.num_actions + 1 + self.mean_actions
        self._solve_c_template = np.zeros(self._n_vars, dtype=np.float64)
        self._solve_a_eq = np.zeros((1, self._n_vars), dtype=np.float64)
        self._solve_a_eq[0, : self.num_actions] = 1.0
        self._solve_b_eq = np.array([1.0], dtype=np.float64)
        self._solve_a_ub_template = np.zeros(
            (self.mean_actions * self.mean_actions, self._n_vars),
            dtype=np.float64,
        )
        row = 0
        for b in range(self.mean_actions):
            for c_idx in range(self.mean_actions):
                self._solve_a_ub_template[row, self._beta_start + b] = 1.0
                self._solve_a_ub_template[row, self._lambda_idx] = -self.cost[b, c_idx]
                row += 1
        self._solve_b_ub = np.zeros(self.mean_actions * self.mean_actions, dtype=np.float64)
        self._solve_bounds = (
            [(0.0, 1.0)] * self.num_actions
            + [(0.0, None)]
            + [(None, None)] * self.mean_actions
        )
        self.solve_calls = 0
        self.solve_failures = 0

    def solve(
        self,
        payoff_matrix: np.ndarray,
        mean_action: np.ndarray,
        epsilon: Optional[float] = None,
    ) -> RobustMeanFieldResult:
        self.solve_calls += 1
        matrix = np.asarray(payoff_matrix, dtype=np.float64)
        if matrix.shape != (self.num_actions, self.mean_actions):
            raise ValueError(
                "Expected payoff matrix shape "
                f"{(self.num_actions, self.mean_actions)}, got {matrix.shape}."
            )
        mu = _normalize_distribution(mean_action, self.mean_actions).astype(np.float64)
        eps = float(self.epsilon if epsilon is None else epsilon)
        eps = float(np.clip(eps, 0.0, 1.0))

        c = self._solve_c_template.copy()
        c[self._lambda_idx] = eps
        c[self._beta_start :] = -mu

        a_ub = self._solve_a_ub_template.copy()
        payoff_columns = -np.tile(matrix.T, (self.mean_actions, 1))
        a_ub[:, : self.num_actions] = payoff_columns

        result = linprog(
            c,
            A_ub=a_ub,
            b_ub=self._solve_b_ub,
            A_eq=self._solve_a_eq,
            b_eq=self._solve_b_eq,
            bounds=self._solve_bounds,
            method="highs",
        )
        if not result.success:
            self.solve_failures += 1
            return self._fallback_result(matrix, mu, eps, result.message)

        policy = _normalize_distribution(result.x[: self.num_actions], self.num_actions).astype(np.float64)
        value = float(-result.fun)
        worst_mean = self.worst_case_mean(policy @ matrix, mu, eps)
        return RobustMeanFieldResult(
            policy=policy.astype(np.float32),
            value=value,
            worst_mean=worst_mean.astype(np.float32),
            lambda_value=float(result.x[self._lambda_idx]),
            success=True,
        )

    def worst_case_mean(
        self,
        values: np.ndarray,
        mean_action: np.ndarray,
        epsilon: Optional[float] = None,
    ) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        mu = _normalize_distribution(mean_action, self.mean_actions).astype(np.float64)
        if v.size != self.mean_actions:
            raise ValueError(f"Expected values length {self.mean_actions}, got {v.size}.")
        eps = float(self.epsilon if epsilon is None else epsilon)
        eps = float(np.clip(eps, 0.0, 1.0))
        return _tv_worst_case_mean(mu, v, eps)

    def _fallback_result(
        self,
        matrix: np.ndarray,
        mu: np.ndarray,
        epsilon: float,
        message: str,
    ) -> RobustMeanFieldResult:
        robust_values = np.array(
            [_tv_worst_case_value(mu, matrix[a], epsilon) for a in range(self.num_actions)],
            dtype=np.float64,
        )
        best_action = int(np.argmax(robust_values))
        policy = np.zeros(self.num_actions, dtype=np.float32)
        policy[best_action] = 1.0
        worst_mean = self.worst_case_mean(matrix[best_action], mu, epsilon)
        return RobustMeanFieldResult(
            policy=policy,
            value=float(robust_values[best_action]),
            worst_mean=worst_mean,
            lambda_value=0.0,
            success=False,
            message=str(message),
        )

    def solve_batch(
        self,
        payoff_matrices: np.ndarray,
        mean_actions: np.ndarray,
        epsilon: Optional[float] = None,
    ) -> list[RobustMeanFieldResult]:
        return [
            self.solve(matrix, mean, epsilon)
            for matrix, mean in zip(payoff_matrices, mean_actions)
        ]


class SolverFreeMFDsrqAgent:
    """One shared solver-free mean-field DSRQ learner for one population."""

    algorithm_name = "mf_srq_lp"

    def __init__(
        self,
        type_id: int,
        obs_channels: int,
        obs_height: int,
        obs_width: int,
        n_own_actions: int,
        n_nbr_actions: int,
        *,
        feature_dim: int = 0,
        epsilon_robust: float = 0.1,
        gamma: float = 0.95,
        lr: float = 1e-4,
        batch_size: int = 64,
        buffer_capacity: int = 80_000,
        learning_starts: int = 5_000,
        train_every: int = 5,
        target_tau: float = 0.005,
        grad_clip: Optional[float] = 10.0,
        epsilon_explore: float = 1.0,
        robust_distance: str = "tv",
        robust_lp_fallback: str = "greedy_tv",
        robust_policy_cache_enabled: bool = True,
        robust_policy_cache_size: int = 4096,
        robust_policy_cache_round_digits: int = 6,
        device: Optional[torch.device] = None,
    ):
        self.type_id = int(type_id)
        self.n_own_actions = int(n_own_actions)
        self.n_nbr_actions = int(n_nbr_actions)
        self.feature_dim = int(feature_dim)
        self.num_actions = self.n_own_actions
        self.epsilon_robust = float(epsilon_robust)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_every = max(1, int(train_every))
        self.target_tau = float(target_tau)
        self.grad_clip = grad_clip
        self.epsilon_explore = float(epsilon_explore)
        self.robust_distance = str(robust_distance)
        self.robust_lp_fallback = str(robust_lp_fallback)
        self.robust_policy_cache_enabled = bool(robust_policy_cache_enabled)
        self.robust_policy_cache_size = int(robust_policy_cache_size)
        self.robust_policy_cache_round_digits = int(robust_policy_cache_round_digits)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.q_net = PairwiseMeanFieldQNetwork(
            obs_channels,
            obs_height,
            obs_width,
            self.n_own_actions,
            self.n_nbr_actions,
            self.feature_dim,
        ).to(device)
        self.target_net = PairwiseMeanFieldQNetwork(
            obs_channels,
            obs_height,
            obs_width,
            self.n_own_actions,
            self.n_nbr_actions,
            self.feature_dim,
        ).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.opt = optim.Adam(self.q_net.parameters(), lr=float(lr))
        self.buffer = SolverFreeMFReplayBuffer(buffer_capacity)
        self.robust_operator = RobustMeanFieldSreOperator(
            self.n_own_actions,
            self.n_nbr_actions,
            epsilon=self.epsilon_robust,
            distance=self.robust_distance,
            fallback=self.robust_lp_fallback,
        )

        self._update_calls = 0
        self._total_train_steps = 0
        self._last_loss: Optional[float] = None
        self._update_times: list[float] = []
        self._policy_cache: OrderedDict[tuple, RobustMeanFieldResult] = OrderedDict()
        self.robust_cache_exact_hits = 0
        self.robust_cache_misses = 0
        self.robust_lp_failures = 0

    def _policy_cache_key(self, payoff_matrix: np.ndarray, mean_action: np.ndarray) -> tuple:
        payoff = np.round(
            np.asarray(payoff_matrix, dtype=np.float32),
            self.robust_policy_cache_round_digits,
        )
        mean = np.round(
            _normalize_distribution(mean_action, self.n_nbr_actions),
            self.robust_policy_cache_round_digits,
        )
        return (
            float(self.epsilon_robust),
            payoff.shape,
            payoff.tobytes(),
            mean.tobytes(),
        )

    def _store_policy_cache(self, key: tuple, result: RobustMeanFieldResult) -> None:
        if not self.robust_policy_cache_enabled or self.robust_policy_cache_size <= 0:
            return
        self._policy_cache[key] = result
        self._policy_cache.move_to_end(key)
        while len(self._policy_cache) > self.robust_policy_cache_size:
            self._policy_cache.popitem(last=False)

    @torch.no_grad()
    def _payoff_matrices_from_net(
        self,
        net: nn.Module,
        obs: torch.Tensor,
        feature: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        return net.payoff_matrix(obs, feature).detach().cpu().numpy().astype(np.float32, copy=False)

    def _solve_policy_batch(
        self,
        payoff_matrices: np.ndarray,
        mean_actions: np.ndarray,
    ) -> list[RobustMeanFieldResult]:
        payoff_matrices = np.asarray(payoff_matrices, dtype=np.float32)
        mean_actions = np.asarray(mean_actions, dtype=np.float32)
        results: list[Optional[RobustMeanFieldResult]] = [None] * int(payoff_matrices.shape[0])
        for idx, (matrix, mean_action) in enumerate(zip(payoff_matrices, mean_actions)):
            key = self._policy_cache_key(matrix, mean_action)
            cached = self._policy_cache.get(key) if self.robust_policy_cache_enabled else None
            if cached is not None:
                self.robust_cache_exact_hits += 1
                self._policy_cache.move_to_end(key)
                results[idx] = cached
                continue
            self.robust_cache_misses += 1
            result = self.robust_operator.solve(
                matrix,
                mean_action,
                epsilon=self.epsilon_robust,
            )
            if not result.success:
                self.robust_lp_failures += 1
            self._store_policy_cache(key, result)
            results[idx] = result
        return [result for result in results if result is not None]

    @staticmethod
    def _robust_values(results: list[RobustMeanFieldResult]) -> np.ndarray:
        return np.asarray([result.value for result in results], dtype=np.float32)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        mean_a: Optional[np.ndarray] = None,
        feature: Optional[np.ndarray] = None,
    ) -> int:
        feature_batch = None if feature is None else np.expand_dims(feature, axis=0)
        return int(
            self.act_batch(
                np.expand_dims(obs, axis=0),
                None if mean_a is None else np.expand_dims(mean_a, axis=0),
                feature_batch,
            )[0]
        )

    @torch.no_grad()
    def act_batch(
        self,
        obs_batch: np.ndarray,
        mean_a_batch: Optional[np.ndarray] = None,
        feature_batch: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        batch_size = int(len(obs_batch))
        if mean_a_batch is None:
            mean_a_batch = np.full(
                (batch_size, self.n_nbr_actions),
                1.0 / self.n_nbr_actions,
                dtype=np.float32,
            )
        explore = np.random.rand(batch_size) < self.epsilon_explore
        actions = np.random.randint(0, self.n_own_actions, size=batch_size, dtype=np.int64)
        pending = np.flatnonzero(~explore)
        if pending.size:
            obs_t = torch.as_tensor(
                obs_batch[pending], dtype=torch.float32, device=self.device
            )
            feature_t = None
            if self.feature_dim > 0:
                if feature_batch is None:
                    feature_batch = np.zeros((batch_size, self.feature_dim), dtype=np.float32)
                feature_t = torch.as_tensor(
                    np.asarray(feature_batch)[pending],
                    dtype=torch.float32,
                    device=self.device,
                )
            matrices = self._payoff_matrices_from_net(self.q_net, obs_t, feature_t)
            results = self._solve_policy_batch(matrices, np.asarray(mean_a_batch)[pending])
            for local_idx, batch_idx in enumerate(pending):
                policy = results[local_idx].policy
                actions[int(batch_idx)] = np.random.choice(
                    self.n_own_actions,
                    p=_normalize_distribution(policy, self.n_own_actions),
                )
        return actions.astype(np.int64)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        mean_a: np.ndarray,
        next_mean_a: np.ndarray,
        done: bool,
        valid: bool = True,
        feature: Optional[np.ndarray] = None,
        next_feature: Optional[np.ndarray] = None,
    ) -> None:
        if self.feature_dim > 0:
            if feature is None:
                feature = np.zeros(self.feature_dim, dtype=np.float32)
            if next_feature is None:
                next_feature = np.zeros(self.feature_dim, dtype=np.float32)
        self.buffer.push(
            obs,
            action,
            reward,
            next_obs,
            mean_a,
            next_mean_a,
            done,
            valid,
            feature=feature,
            next_feature=next_feature,
        )
        self._update_calls += 1

    def maybe_train(self) -> Optional[float]:
        if len(self.buffer) < self.learning_starts:
            return None
        if self._update_calls % self.train_every != 0:
            return None
        return self.train_step()

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None
        t0 = time.perf_counter()
        batch = self.buffer.sample(self.batch_size, self.device)

        obs = batch["obs"]
        feature = batch["feature"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_obs = batch["next_obs"]
        next_feature = batch["next_feature"]
        mean_a = batch["mean_a"]
        next_mean_a = batch["next_mean_a"]
        dones = batch["done"]
        valid = batch["valid"]

        q_values = self.q_net(obs, mean_a, feature=feature)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_values = torch.zeros_like(rewards)
            nonterminal = torch.nonzero(dones < 1.0, as_tuple=False).flatten()
            if nonterminal.numel():
                next_policy_payoff = self._payoff_matrices_from_net(
                    self.q_net,
                    next_obs[nonterminal],
                    next_feature[nonterminal],
                )
                policies = self._solve_policy_batch(
                    next_policy_payoff,
                    next_mean_a[nonterminal].detach().cpu().numpy(),
                )
                next_target_payoff = self._payoff_matrices_from_net(
                    self.target_net,
                    next_obs[nonterminal],
                    next_feature[nonterminal],
                )
                next_mean_np = next_mean_a[nonterminal].detach().cpu().numpy()
                values = []
                for matrix, next_mean, policy in zip(next_target_payoff, next_mean_np, policies):
                    # Keep Double-DQN-style selection: online policy, target payoff.
                    selected_policy = policy.policy
                    target_values_by_mean = selected_policy @ matrix
                    worst_mean = self.robust_operator.worst_case_mean(
                        target_values_by_mean,
                        next_mean,
                        epsilon=self.epsilon_robust,
                    )
                    values.append(float(target_values_by_mean @ worst_mean))
                target_values[nonterminal] = torch.as_tensor(
                    values, dtype=torch.float32, device=self.device
                )
            y = rewards + (1.0 - dones) * self.gamma * target_values

        loss_per = nn.functional.mse_loss(q_taken, y, reduction="none")
        loss = (loss_per * valid).sum() / valid.sum().clamp(min=1.0)

        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), float(self.grad_clip))
        self.opt.step()
        self.soft_update_target_network(self.target_tau)

        self._total_train_steps += 1
        self._last_loss = float(loss.item())
        self._update_times.append(time.perf_counter() - t0)
        return self._last_loss

    def soft_update_target_network(self, tau: float) -> None:
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def save_checkpoint(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.algorithm_name,
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.opt.state_dict(),
                "epsilon_robust": self.epsilon_robust,
                "epsilon_explore": self.epsilon_explore,
                "total_train_steps": self._total_train_steps,
                "n_own_actions": self.n_own_actions,
                "n_nbr_actions": self.n_nbr_actions,
                "feature_dim": self.feature_dim,
                "robust_distance": self.robust_distance,
                "robust_lp_fallback": self.robust_lp_fallback,
            },
            path,
        )

    def load_checkpoint(self, path, map_location=None) -> None:
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.opt.load_state_dict(checkpoint["optimizer"])
        self.epsilon_robust = float(checkpoint.get("epsilon_robust", self.epsilon_robust))
        self.epsilon_explore = float(checkpoint.get("epsilon_explore", self.epsilon_explore))
        self._total_train_steps = int(checkpoint.get("total_train_steps", 0))

    def get_avg_update_time_ms(self) -> Optional[float]:
        if not self._update_times:
            return None
        return float(np.mean(self._update_times)) * 1000.0

    def close(self) -> None:
        return None
