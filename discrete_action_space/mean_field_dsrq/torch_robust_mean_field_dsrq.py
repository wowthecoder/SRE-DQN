"""Torch-native robust-action mean-field Deep SRQ.

This variant keeps the solver-free MF-SRQ network, replay buffer, and training
surface, but replaces the SciPy LP robust best response with a batched Torch
operator.  The operator robustifies each own-action value against a TV ball
around the neighbour mean action, then uses an MF-Q-style Boltzmann policy.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .solver_free_mean_field_dsrq import SolverFreeMFDsrqAgent


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


def torch_tv_worst_case_values(
    mean_action: torch.Tensor,
    values: torch.Tensor,
    epsilon: float | torch.Tensor,
) -> torch.Tensor:
    """Return worst-case expectations under the 0/1-cost TV transport ball.

    Args:
        mean_action: Nominal distributions with shape ``[batch, mean_actions]``.
        values: Payoffs with shape ``[batch, own_actions, mean_actions]``.
        epsilon: Transport budget. With 0/1 ground cost this is the amount of
            probability mass that may be moved from high-value neighbour actions
            to low-value neighbour actions.

    Returns:
        Tensor of robust action values with shape ``[batch, own_actions]``.
    """
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


class TorchRobustMFDsrqAgent(SolverFreeMFDsrqAgent):
    """MF-SRQ using a Torch batched robust-action-value operator."""

    algorithm_name = "mf_srq_torch"

    def __init__(
        self,
        *args,
        robust_policy_temperature: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.robust_policy_temperature = float(robust_policy_temperature)
        self.robust_operator = TorchRobustActionValueOperator(
            self.n_own_actions,
            self.n_nbr_actions,
            epsilon=self.epsilon_robust,
            temperature=self.robust_policy_temperature,
        )
        self.robust_torch_operator_calls = 0

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
    def act_batch(
        self,
        obs_batch,
        mean_a_batch=None,
        feature_batch=None,
    ):
        import numpy as np

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

    def train_step(self) -> Optional[float]:
        import time

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

    def save_checkpoint(self, path) -> None:
        super().save_checkpoint(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["robust_policy_temperature"] = self.robust_policy_temperature
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, map_location=None) -> None:
        super().load_checkpoint(path, map_location=map_location)
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.robust_policy_temperature = float(
            checkpoint.get("robust_policy_temperature", self.robust_policy_temperature)
        )
        self.robust_operator.temperature = self.robust_policy_temperature
