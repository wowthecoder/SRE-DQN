"""Torch projected-gradient solver for continuous-action security strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


PayoffFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class SecuritySolution:
    actions: torch.Tensor
    adversarial_actions: torch.Tensor
    nominal_values: torch.Tensor
    security_values: torch.Tensor
    residual: torch.Tensor
    iterations: int


class SecurityStrategySolver:
    """Approximate maximin stage-game solver over bounded continuous actions."""

    def __init__(
        self,
        num_players: int,
        action_dim: int = 1,
        action_low: float | Iterable[float] = -1.0,
        action_high: float | Iterable[float] = 1.0,
        outer_iters: int = 8,
        adversary_iters: int = 4,
        action_lr: float = 0.05,
        adversary_lr: float = 0.05,
        prox_weight: float = 1e-2,
        tol: float = 1e-4,
    ) -> None:
        self.num_players = int(num_players)
        self.action_dim = int(action_dim)
        self.outer_iters = int(outer_iters)
        self.adversary_iters = int(adversary_iters)
        self.action_lr = float(action_lr)
        self.adversary_lr = float(adversary_lr)
        self.prox_weight = float(prox_weight)
        self.tol = float(tol)

        low = torch.as_tensor(action_low, dtype=torch.float32).flatten()
        high = torch.as_tensor(action_high, dtype=torch.float32).flatten()
        if low.numel() == 1:
            low = low.repeat(self.action_dim)
        if high.numel() == 1:
            high = high.repeat(self.action_dim)
        if low.numel() != self.action_dim or high.numel() != self.action_dim:
            raise ValueError("action bounds must be scalar or match action_dim")
        if torch.any(high <= low):
            raise ValueError("action_high must be greater than action_low")
        self.action_low = low
        self.action_high = high

    def _bounds(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.action_low.to(device=ref.device, dtype=ref.dtype).view(1, 1, self.action_dim)
        high = self.action_high.to(device=ref.device, dtype=ref.dtype).view(1, 1, self.action_dim)
        return low, high

    def project_actions(self, actions: torch.Tensor) -> torch.Tensor:
        low, high = self._bounds(actions)
        return torch.max(torch.min(actions, high), low)

    def _adversarial_profile(
        self,
        states: torch.Tensor,
        center_actions: torch.Tensor,
        player_idx: int,
        payoff_fn: PayoffFn,
    ) -> torch.Tensor:
        adv_actions = center_actions.detach().clone()
        low, high = self._bounds(center_actions)

        for _ in range(self.adversary_iters):
            adv_var = adv_actions.detach().clone().requires_grad_(True)
            joint = adv_var.clone()
            joint[:, player_idx, :] = center_actions[:, player_idx, :].detach()
            payoff = payoff_fn(states, joint)[:, player_idx]
            grad = torch.autograd.grad(payoff.sum(), adv_var, allow_unused=False)[0]
            next_adv = adv_var - self.adversary_lr * grad
            next_adv = torch.max(torch.min(next_adv, high), low)
            next_adv[:, player_idx, :] = center_actions[:, player_idx, :].detach()
            adv_actions = next_adv.detach()

        return adv_actions

    def _security_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        payoff_fn: PayoffFn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adversarial_profiles = []
        values = []
        for player_idx in range(self.num_players):
            adv = self._adversarial_profile(states, actions, player_idx, payoff_fn)
            adversarial_profiles.append(adv)
            values.append(payoff_fn(states, adv)[:, player_idx])
        return torch.stack(values, dim=1), torch.stack(adversarial_profiles, dim=1)

    @torch.enable_grad()
    def solve(
        self,
        states: torch.Tensor,
        initial_actions: torch.Tensor,
        payoff_fn: PayoffFn,
    ) -> SecuritySolution:
        if initial_actions.dim() != 3:
            raise ValueError("initial_actions must have shape [batch, num_players, action_dim]")
        if initial_actions.shape[1] != self.num_players:
            raise ValueError("initial_actions player dimension does not match solver")

        actions = self.project_actions(initial_actions.detach().clone())
        residual = torch.full((), float("inf"), device=actions.device, dtype=actions.dtype)
        iterations = 0

        for iterations in range(1, self.outer_iters + 1):
            old_actions = actions
            next_actions = actions.clone()
            adversarial_profiles = [
                self._adversarial_profile(states, actions, player_idx, payoff_fn)
                for player_idx in range(self.num_players)
            ]

            for player_idx in range(self.num_players):
                a_i = actions[:, player_idx, :].detach().clone().requires_grad_(True)
                joint = adversarial_profiles[player_idx].detach().clone()
                joint[:, player_idx, :] = a_i
                payoff = payoff_fn(states, joint)[:, player_idx]
                prox = self.prox_weight * ((a_i - actions[:, player_idx, :].detach()) ** 2).sum(dim=1)
                objective = payoff - prox
                grad_a = torch.autograd.grad(objective.sum(), a_i, allow_unused=False)[0]
                next_actions[:, player_idx, :] = (a_i + self.action_lr * grad_a).detach()

            actions = self.project_actions(next_actions)
            residual = (actions - old_actions).abs().amax()
            if torch.isfinite(residual) and residual.item() <= self.tol:
                break

        nominal_values = payoff_fn(states, actions).detach()
        security_values, adversarial_actions = self._security_values(states, actions, payoff_fn)
        return SecuritySolution(
            actions=actions.detach(),
            adversarial_actions=adversarial_actions.detach(),
            nominal_values=nominal_values,
            security_values=security_values.detach(),
            residual=residual.detach(),
            iterations=iterations,
        )

