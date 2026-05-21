"""Torch-based surrogate SRE stage-game solver.

This solver implements the continuous concave surrogate game from the SRE theory
paper as a nested projected-gradient routine. It is intended as a practical Bellman
target generator for learned neural critics, not as a replacement for exact CVXPY
solvers when the static game has a closed convex formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


PayoffFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class SRESolution:
    actions: torch.Tensor
    lambdas: torch.Tensor
    nominal_values: torch.Tensor
    robust_values: torch.Tensor
    residual: torch.Tensor
    iterations: int


class SurrogateSRESolver:
    """Projected-gradient proximal solver for the surrogate SRE game."""

    def __init__(
        self,
        num_players: int,
        action_dim: int = 1,
        action_low: float | Iterable[float] = -1.0,
        action_high: float | Iterable[float] = 1.0,
        lambda_min: float = 1e-4,
        lambda_max: float = 100.0,
        outer_iters: int = 8,
        adversary_iters: int = 4,
        action_lr: float = 0.05,
        lambda_lr: float = 0.05,
        adversary_lr: float = 0.05,
        prox_weight: float = 1e-2,
        tol: float = 1e-4,
        distance_power: int = 2,
        vectorized: bool = True,
    ) -> None:
        if distance_power != 2:
            raise ValueError("v1 solver supports squared Euclidean transport cost only")
        self.num_players = int(num_players)
        self.action_dim = int(action_dim)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.outer_iters = int(outer_iters)
        self.adversary_iters = int(adversary_iters)
        self.action_lr = float(action_lr)
        self.lambda_lr = float(lambda_lr)
        self.adversary_lr = float(adversary_lr)
        self.prox_weight = float(prox_weight)
        self.tol = float(tol)
        self.vectorized = bool(vectorized)

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
        shape = (1,) * (ref.dim() - 1) + (self.action_dim,)
        low = self.action_low.to(device=ref.device, dtype=ref.dtype).view(shape)
        high = self.action_high.to(device=ref.device, dtype=ref.dtype).view(shape)
        return low, high

    def _project_actions(self, actions: torch.Tensor) -> torch.Tensor:
        low, high = self._bounds(actions)
        return torch.max(torch.min(actions, high), low)

    def _project_lambdas(self, lambdas: torch.Tensor) -> torch.Tensor:
        return lambdas.clamp(self.lambda_min, self.lambda_max)

    def _opponent_mask(self, player_idx: int, ref: torch.Tensor) -> torch.Tensor:
        mask = torch.ones((1, self.num_players, 1), device=ref.device, dtype=ref.dtype)
        mask[:, player_idx, :] = 0.0
        return mask

    def _distance(self, center: torch.Tensor, perturbed: torch.Tensor, player_idx: int) -> torch.Tensor:
        mask = self._opponent_mask(player_idx, center)
        return (((center - perturbed) * mask) ** 2).sum(dim=(1, 2))

    def _player_masks(self, ref: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(self.num_players, device=ref.device, dtype=ref.dtype)
        return (1.0 - eye).view(self.num_players, 1, self.num_players, 1)

    def _states_for_all_players(self, states: torch.Tensor) -> torch.Tensor:
        return states.repeat(self.num_players, 1)

    def _payoffs_for_player_profiles(
        self,
        states: torch.Tensor,
        joint_profiles: torch.Tensor,
        payoff_fn: PayoffFn,
    ) -> torch.Tensor:
        batch_size = joint_profiles.shape[1]
        flat_profiles = joint_profiles.reshape(
            self.num_players * batch_size,
            self.num_players,
            self.action_dim,
        )
        flat_payoffs = payoff_fn(self._states_for_all_players(states), flat_profiles)
        payoffs = flat_payoffs.view(self.num_players, batch_size, self.num_players)
        player_ids = torch.arange(self.num_players, device=joint_profiles.device)
        return payoffs[player_ids, :, player_ids]

    def _distance_all_players(self, center: torch.Tensor, perturbed: torch.Tensor) -> torch.Tensor:
        masks = self._player_masks(center)
        center_all = center.unsqueeze(0)
        return (((center_all - perturbed) * masks) ** 2).sum(dim=(2, 3))

    def _adversarial_profile(
        self,
        states: torch.Tensor,
        center_actions: torch.Tensor,
        lambdas: torch.Tensor,
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
            transport_cost = self._distance(center_actions.detach(), joint, player_idx)
            objective = payoff + lambdas[:, player_idx].detach() * transport_cost
            grad = torch.autograd.grad(objective.sum(), adv_var, allow_unused=False)[0]
            next_adv = adv_var - self.adversary_lr * grad
            next_adv = torch.max(torch.min(next_adv, high), low)
            next_adv[:, player_idx, :] = center_actions[:, player_idx, :].detach()
            adv_actions = next_adv.detach()

        return adv_actions

    def _adversarial_profiles_all(
        self,
        states: torch.Tensor,
        center_actions: torch.Tensor,
        lambdas: torch.Tensor,
        payoff_fn: PayoffFn,
    ) -> torch.Tensor:
        masks = self._player_masks(center_actions)
        center_all = center_actions.unsqueeze(0)
        adv_actions = center_all.repeat(self.num_players, 1, 1, 1).detach().clone()
        lambda_by_player = lambdas.transpose(0, 1)
        low, high = self._bounds(adv_actions)

        for _ in range(self.adversary_iters):
            adv_var = adv_actions.detach().clone().requires_grad_(True)
            joint = adv_var * masks + center_all.detach() * (1.0 - masks)
            payoff = self._payoffs_for_player_profiles(states, joint, payoff_fn)
            transport_cost = self._distance_all_players(center_actions.detach(), joint)
            objective = payoff + lambda_by_player.detach() * transport_cost
            grad = torch.autograd.grad(objective.sum(), adv_var, allow_unused=False)[0]
            next_adv = adv_var - self.adversary_lr * grad
            next_adv = torch.max(torch.min(next_adv, high), low)
            adv_actions = (next_adv * masks + center_all.detach() * (1.0 - masks)).detach()

        return adv_actions

    def _robust_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        lambdas: torch.Tensor,
        eps: float,
        payoff_fn: PayoffFn,
    ) -> torch.Tensor:
        values = []
        eps_sq = float(eps) ** 2
        for player_idx in range(self.num_players):
            adv = self._adversarial_profile(states, actions, lambdas, player_idx, payoff_fn)
            payoff = payoff_fn(states, adv)[:, player_idx]
            transport_cost = self._distance(actions, adv, player_idx)
            value = payoff + lambdas[:, player_idx] * transport_cost - lambdas[:, player_idx] * eps_sq
            values.append(value)
        return torch.stack(values, dim=1)

    def _robust_values_vectorized(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        lambdas: torch.Tensor,
        eps: float,
        payoff_fn: PayoffFn,
    ) -> torch.Tensor:
        eps_sq = float(eps) ** 2
        adv = self._adversarial_profiles_all(states, actions, lambdas, payoff_fn)
        payoff = self._payoffs_for_player_profiles(states, adv, payoff_fn)
        transport_cost = self._distance_all_players(actions, adv)
        value = payoff + lambdas.transpose(0, 1) * transport_cost - lambdas.transpose(0, 1) * eps_sq
        return value.transpose(0, 1)

    def _solve_loop(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        lambdas: torch.Tensor,
        eps: float,
        payoff_fn: PayoffFn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        residual = torch.full((), float("inf"), device=actions.device, dtype=actions.dtype)
        iterations = 0
        eps_sq = float(eps) ** 2

        for iterations in range(1, self.outer_iters + 1):
            old_actions = actions
            old_lambdas = lambdas
            next_actions = actions.clone()
            next_lambdas = lambdas.clone()

            adversarial_profiles = [
                self._adversarial_profile(states, actions, lambdas, player_idx, payoff_fn)
                for player_idx in range(self.num_players)
            ]

            for player_idx in range(self.num_players):
                a_i = actions[:, player_idx, :].detach().clone().requires_grad_(True)
                lambda_i = lambdas[:, player_idx].detach().clone().requires_grad_(True)
                joint = adversarial_profiles[player_idx].detach().clone()
                joint[:, player_idx, :] = a_i
                payoff = payoff_fn(states, joint)[:, player_idx]
                transport_cost = self._distance(actions.detach(), joint, player_idx)
                prox = self.prox_weight * ((a_i - actions[:, player_idx, :].detach()) ** 2).sum(dim=1)
                prox = prox + self.prox_weight * (lambda_i - lambdas[:, player_idx].detach()) ** 2
                surrogate = payoff + lambda_i * transport_cost - lambda_i * eps_sq - prox
                grad_a, grad_lam = torch.autograd.grad(surrogate.sum(), (a_i, lambda_i), allow_unused=False)
                next_actions[:, player_idx, :] = (a_i + self.action_lr * grad_a).detach()
                next_lambdas[:, player_idx] = (lambda_i + self.lambda_lr * grad_lam).detach()

            actions = self._project_actions(next_actions)
            lambdas = self._project_lambdas(next_lambdas)
            action_delta = (actions - old_actions).abs().amax()
            lambda_delta = (lambdas - old_lambdas).abs().amax()
            residual = torch.maximum(action_delta, lambda_delta)
            if torch.isfinite(residual) and residual.item() <= self.tol:
                break

        return actions, lambdas, residual, iterations

    def _solve_vectorized(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        lambdas: torch.Tensor,
        eps: float,
        payoff_fn: PayoffFn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        residual = torch.full((), float("inf"), device=actions.device, dtype=actions.dtype)
        iterations = 0
        eps_sq = float(eps) ** 2

        for iterations in range(1, self.outer_iters + 1):
            old_actions = actions
            old_lambdas = lambdas
            masks = self._player_masks(actions)
            center_all = actions.unsqueeze(0)
            adversarial_profiles = self._adversarial_profiles_all(states, actions, lambdas, payoff_fn)

            a_i = actions.transpose(0, 1).detach().clone().requires_grad_(True)
            lambda_i = lambdas.transpose(0, 1).detach().clone().requires_grad_(True)
            joint = adversarial_profiles.detach().clone()
            joint = joint * masks + a_i.unsqueeze(2) * (1.0 - masks)
            payoff = self._payoffs_for_player_profiles(states, joint, payoff_fn)
            transport_cost = self._distance_all_players(actions.detach(), joint)
            prox = self.prox_weight * ((a_i - actions.transpose(0, 1).detach()) ** 2).sum(dim=2)
            prox = prox + self.prox_weight * (lambda_i - lambdas.transpose(0, 1).detach()) ** 2
            surrogate = payoff + lambda_i * transport_cost - lambda_i * eps_sq - prox
            grad_a, grad_lam = torch.autograd.grad(surrogate.sum(), (a_i, lambda_i), allow_unused=False)

            next_actions = actions.clone()
            next_lambdas = lambdas.clone()
            next_actions = (a_i + self.action_lr * grad_a).transpose(0, 1).detach()
            next_lambdas = (lambda_i + self.lambda_lr * grad_lam).transpose(0, 1).detach()

            actions = self._project_actions(next_actions)
            lambdas = self._project_lambdas(next_lambdas)
            action_delta = (actions - old_actions).abs().amax()
            lambda_delta = (lambdas - old_lambdas).abs().amax()
            residual = torch.maximum(action_delta, lambda_delta)
            if torch.isfinite(residual) and residual.item() <= self.tol:
                break

        return actions, lambdas, residual, iterations

    def _solve_nominal(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        lambdas: torch.Tensor,
        payoff_fn: PayoffFn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Projected simultaneous best-response ascent for the eps=0 nominal game."""
        residual = torch.full((), float("inf"), device=actions.device, dtype=actions.dtype)
        iterations = 0

        for iterations in range(1, self.outer_iters + 1):
            old_actions = actions
            masks = self._player_masks(actions)
            center_all = actions.unsqueeze(0)

            a_i = actions.transpose(0, 1).detach().clone().requires_grad_(True)
            joint = center_all.detach() * masks + a_i.unsqueeze(2) * (1.0 - masks)
            payoff = self._payoffs_for_player_profiles(states, joint, payoff_fn)
            prox = self.prox_weight * ((a_i - actions.transpose(0, 1).detach()) ** 2).sum(dim=2)
            surrogate = payoff - prox
            grad_a = torch.autograd.grad(surrogate.sum(), a_i, allow_unused=False)[0]

            actions = self._project_actions((a_i + self.action_lr * grad_a).transpose(0, 1).detach())
            residual = (actions - old_actions).abs().amax()
            if torch.isfinite(residual) and residual.item() <= self.tol:
                break

        lambdas = torch.full_like(lambdas, self.lambda_min)
        return actions, lambdas, residual, iterations

    @torch.enable_grad()
    def solve(
        self,
        states: torch.Tensor,
        initial_actions: torch.Tensor,
        initial_lambdas: torch.Tensor | None,
        eps: float,
        payoff_fn: PayoffFn,
    ) -> SRESolution:
        if initial_actions.dim() != 3:
            raise ValueError("initial_actions must have shape [batch, num_players, action_dim]")
        if initial_actions.shape[1] != self.num_players:
            raise ValueError("initial_actions player dimension does not match solver")
        actions = self._project_actions(initial_actions.detach().clone())
        batch_size = actions.shape[0]

        if initial_lambdas is None:
            lambdas = torch.ones(batch_size, self.num_players, device=actions.device, dtype=actions.dtype)
        else:
            lambdas = initial_lambdas.detach().clone().to(device=actions.device, dtype=actions.dtype)
        lambdas = self._project_lambdas(lambdas)

        if float(eps) <= 0.0:
            actions, lambdas, residual, iterations = self._solve_nominal(
                states, actions, lambdas, payoff_fn
            )
        elif self.vectorized:
            actions, lambdas, residual, iterations = self._solve_vectorized(
                states, actions, lambdas, eps, payoff_fn
            )
        else:
            actions, lambdas, residual, iterations = self._solve_loop(
                states, actions, lambdas, eps, payoff_fn
            )

        nominal_values = payoff_fn(states, actions).detach()
        if float(eps) <= 0.0:
            robust_values = nominal_values
        elif self.vectorized:
            robust_values = self._robust_values_vectorized(states, actions, lambdas, eps, payoff_fn).detach()
        else:
            robust_values = self._robust_values(states, actions, lambdas, eps, payoff_fn).detach()
        return SRESolution(
            actions=actions.detach(),
            lambdas=lambdas.detach(),
            nominal_values=nominal_values,
            robust_values=robust_values,
            residual=residual.detach(),
            iterations=iterations,
        )
