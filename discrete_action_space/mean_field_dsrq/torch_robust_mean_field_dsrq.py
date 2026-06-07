"""Torch-native robust-action mean-field Deep SRQ."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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


def _normalize_torch_distribution(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = torch.clamp(values, min=0.0)
    total = values.sum(dim=-1, keepdim=True)
    size = max(int(values.shape[-1]), 1)
    uniform = torch.full_like(values, 1.0 / size)
    return torch.where(total > eps, values / total.clamp_min(eps), uniform)


def _finite_for_policy(values: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(values).all():
        return values
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return torch.zeros_like(values)
    high = finite.max().detach()
    low = finite.min().detach()
    return torch.nan_to_num(values, nan=0.0, posinf=float(high), neginf=float(low))


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

    def _feature_tensor(
        self,
        obs: torch.Tensor,
        feature: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
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


class MeanFieldReplayBuffer:
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
                np.array(obs, dtype=np.float32, copy=True),
                np.array(feature, dtype=np.float32, copy=True),
                int(action),
                float(reward),
                np.array(next_obs, dtype=np.float32, copy=True),
                np.array(next_feature, dtype=np.float32, copy=True),
                _normalize_distribution(mean_a),
                _normalize_distribution(next_mean_a),
                float(done),
                float(valid),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
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


def torch_tv_worst_case_values(
    mean_action: torch.Tensor,
    values: torch.Tensor,
    epsilon: float | torch.Tensor,
) -> torch.Tensor:
    """Return worst-case expectations under the 0/1-cost TV transport ball."""
    if values.ndim != 3:
        raise ValueError(f"Expected values shape [B,A,K], got {tuple(values.shape)}.")
    if mean_action.ndim != 2:
        raise ValueError(f"Expected mean_action shape [B,K], got {tuple(mean_action.shape)}.")
    if values.shape[0] != mean_action.shape[0] or values.shape[-1] != mean_action.shape[-1]:
        raise ValueError(
            "values and mean_action batch/action dimensions do not agree: "
            f"{tuple(values.shape)} vs {tuple(mean_action.shape)}."
        )

    batch, own_actions, mean_actions = values.shape
    mu = _normalize_torch_distribution(mean_action).to(dtype=values.dtype, device=values.device)
    if mean_actions <= 1:
        return (values * mu.unsqueeze(1)).sum(dim=-1)

    if not torch.is_tensor(epsilon):
        eps_t = torch.tensor(float(epsilon), dtype=values.dtype, device=values.device)
    else:
        eps_t = epsilon.to(dtype=values.dtype, device=values.device)
    eps_t = torch.clamp(eps_t, min=0.0, max=1.0)
    if bool(torch.all(eps_t <= 0.0)):
        return (values * mu.unsqueeze(1)).sum(dim=-1)

    flat_values = values.reshape(batch * own_actions, mean_actions)
    flat_q = mu.unsqueeze(1).expand(batch, own_actions, mean_actions).reshape(
        batch * own_actions, mean_actions
    ).clone()
    budget = eps_t.expand(batch * own_actions).clone() if eps_t.ndim == 0 else eps_t.reshape(-1).clone()
    if budget.numel() == batch:
        budget = budget.unsqueeze(1).expand(batch, own_actions).reshape(-1).clone()
    if budget.numel() != batch * own_actions:
        raise ValueError(
            "epsilon tensor must be scalar, [B], or [B,A], "
            f"got {tuple(eps_t.shape)} for values {tuple(values.shape)}."
        )

    high_order = torch.argsort(flat_values, dim=-1, descending=True)
    low_order = torch.argsort(flat_values, dim=-1, descending=False)
    rows = torch.arange(flat_values.shape[0], device=values.device)
    high_pos = torch.zeros_like(rows)
    low_pos = torch.zeros_like(rows)
    tol = torch.finfo(values.dtype).eps * 16.0

    for _ in range(2 * mean_actions + 1):
        active = (
            (budget > tol)
            & (high_pos < mean_actions)
            & (low_pos < mean_actions)
        )
        if not bool(active.any()):
            break

        hi = high_order[rows, high_pos.clamp(max=mean_actions - 1)]
        lo = low_order[rows, low_pos.clamp(max=mean_actions - 1)]
        hi_value = flat_values[rows, hi]
        lo_value = flat_values[rows, lo]
        active = active & (hi_value > lo_value + tol)
        if not bool(active.any()):
            break

        hi_mass = flat_q[rows, hi]
        lo_cap = 1.0 - flat_q[rows, lo]
        movable = torch.minimum(torch.minimum(hi_mass, lo_cap), budget)
        movable = torch.where(active, movable.clamp_min(0.0), torch.zeros_like(movable))

        flat_q[rows, hi] = flat_q[rows, hi] - movable
        flat_q[rows, lo] = flat_q[rows, lo] + movable
        budget = budget - movable

        high_done = active & (flat_q[rows, hi] <= tol)
        low_done = active & (flat_q[rows, lo] >= 1.0 - tol)
        stuck = active & (movable <= tol)
        high_pos = high_pos + (high_done | stuck).to(high_pos.dtype)
        low_pos = low_pos + (low_done | stuck).to(low_pos.dtype)

    worst_values = (flat_q * flat_values).sum(dim=-1)
    return worst_values.reshape(batch, own_actions)


class TorchRobustActionValueOperator:
    """Batched robust action-value operator used by MF-SRQ-Torch."""

    def __init__(
        self,
        num_actions: int,
        mean_actions: Optional[int] = None,
        *,
        epsilon: float = 0.1,
        temperature: float = 0.1,
    ):
        self.num_actions = int(num_actions)
        self.mean_actions = int(mean_actions if mean_actions is not None else num_actions)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)

    def robust_action_values(
        self,
        payoff_matrices: torch.Tensor,
        mean_actions: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        eps = self.epsilon if epsilon is None else float(epsilon)
        return torch_tv_worst_case_values(mean_actions, payoff_matrices, eps)

    def policy_from_values(self, robust_values: torch.Tensor) -> torch.Tensor:
        robust_values = _finite_for_policy(robust_values)
        if self.temperature <= 0.0:
            best = robust_values.argmax(dim=-1)
            return nn.functional.one_hot(best, num_classes=robust_values.shape[-1]).to(
                dtype=robust_values.dtype
            )
        logits = robust_values / max(self.temperature, 1e-8)
        logits = _finite_for_policy(logits)
        policy = torch.softmax(logits, dim=-1)
        return _normalize_torch_distribution(policy)


class TorchRobustMFDsrqAgent:
    """Mean-field DSRQ learner with a Torch batched robust value operator."""

    algorithm_name = "mf_srq_torch"

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
        robust_policy_temperature: float = 0.1,
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
        self.robust_policy_temperature = float(robust_policy_temperature)

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
        self.buffer = MeanFieldReplayBuffer(buffer_capacity)
        self.robust_operator = TorchRobustActionValueOperator(
            self.n_own_actions,
            self.n_nbr_actions,
            epsilon=self.epsilon_robust,
            temperature=self.robust_policy_temperature,
        )
        self.robust_torch_operator_calls = 0

        self._update_calls = 0
        self._total_train_steps = 0
        self._last_loss: Optional[float] = None
        self._update_times: list[float] = []

    def _robust_action_values(
        self,
        payoff_matrices: torch.Tensor,
        mean_actions: torch.Tensor,
    ) -> torch.Tensor:
        self.robust_operator.epsilon = float(self.epsilon_robust)
        self.robust_operator.temperature = float(self.robust_policy_temperature)
        self.robust_torch_operator_calls += int(payoff_matrices.shape[0])
        return self.robust_operator.robust_action_values(
            payoff_matrices,
            mean_actions,
            epsilon=self.epsilon_robust,
        )

    def _policy_from_payoffs(
        self,
        payoff_matrices: torch.Tensor,
        mean_actions: torch.Tensor,
    ) -> torch.Tensor:
        robust_q = self._robust_action_values(payoff_matrices, mean_actions)
        return self.robust_operator.policy_from_values(robust_q)

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
            mean_t = torch.as_tensor(
                np.asarray(mean_a_batch)[pending],
                dtype=torch.float32,
                device=self.device,
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
            payoff = self.q_net.payoff_matrix(obs_t, feature_t)
            policy = self._policy_from_payoffs(payoff, mean_t)
            policy = _normalize_torch_distribution(policy)
            sampled = torch.multinomial(policy, num_samples=1).squeeze(1)
            actions[pending] = sampled.detach().cpu().numpy().astype(np.int64)
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
                next_mean = next_mean_a[nonterminal]
                online_payoff = self.q_net.payoff_matrix(
                    next_obs[nonterminal],
                    next_feature[nonterminal],
                )
                online_policy = self._policy_from_payoffs(online_payoff, next_mean)

                target_payoff = self.target_net.payoff_matrix(
                    next_obs[nonterminal],
                    next_feature[nonterminal],
                )
                target_values_by_mean = (
                    online_policy.unsqueeze(-1) * target_payoff
                ).sum(dim=1)
                target_values[nonterminal] = torch_tv_worst_case_values(
                    next_mean,
                    target_values_by_mean.unsqueeze(1),
                    self.epsilon_robust,
                ).squeeze(1)
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
                "robust_policy_temperature": self.robust_policy_temperature,
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
        self.robust_policy_temperature = float(
            checkpoint.get("robust_policy_temperature", self.robust_policy_temperature)
        )
        self.robust_operator.temperature = self.robust_policy_temperature

    def get_avg_update_time_ms(self) -> Optional[float]:
        if not self._update_times:
            return None
        return float(np.mean(self._update_times)) * 1000.0

    def close(self) -> None:
        return None
