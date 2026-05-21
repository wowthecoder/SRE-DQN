"""Just-concave SRE-DQN agent.

The agent keeps the existing training-loop interface but replaces the local LQ
advantage representation with a learned joint-action critic and a surrogate-SRE
stage-game solver.
"""

from __future__ import annotations

from copy import deepcopy as dc
from typing import Iterable
import warnings

import torch
import torch.nn.functional as F

from .networks import ActorLambdaNet, JointQCritic
from .solver import SRESolution, SurrogateSRESolver


class JustConcaveSREAgent:
    """Continuous-action SRE-DQN agent using a learned critic stage game."""

    def __init__(
        self,
        state_dim: int,
        n_players: int,
        action_dim: int = 1,
        action_low: float | Iterable[float] = -50.0,
        action_high: float | Iterable[float] = 50.0,
        hidden_sizes: Iterable[int] | int = (128, 128, 128),
        lr: float = 1e-3,
        gamma: float = 1.0,
        tau: float = 0.01,
        lambda_min: float = 1e-4,
        lambda_max: float = 100.0,
        solver_iters: int = 8,
        adversary_iters: int = 4,
        solver_lr: float = 0.05,
        adversary_lr: float = 0.05,
        solver_tol: float = 1e-4,
        imitation_weight: float = 1.0,
        lambda_imitation_weight: float = 1e-3,
        policy_value_weight: float = 1e-2,
        compile_networks: bool = False,
        solver_vectorized: bool = True,
        use_cuda: bool | None = None,
    ) -> None:
        self.num_players = int(n_players)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.imitation_weight = float(imitation_weight)
        self.lambda_imitation_weight = float(lambda_imitation_weight)
        self.policy_value_weight = float(policy_value_weight)
        self.compile_networks = bool(compile_networks)
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else bool(use_cuda)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.action_net = ActorLambdaNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            action_low=action_low,
            action_high=action_high,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        ).to(self.device)
        self.value_net = JointQCritic(
            state_dim=state_dim,
            num_players=n_players,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        self.slow_val_net = JointQCritic(
            state_dim=state_dim,
            num_players=n_players,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        self.update_slow(hard=True)

        self.optimizer_DQN = torch.optim.Adam(self.action_net.parameters(), lr=lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=lr)

        self.solver = SurrogateSRESolver(
            num_players=n_players,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            outer_iters=solver_iters,
            adversary_iters=adversary_iters,
            action_lr=solver_lr,
            lambda_lr=solver_lr,
            adversary_lr=adversary_lr,
            tol=solver_tol,
            vectorized=solver_vectorized,
        )
        if self.compile_networks:
            self._compile_network_forwards()

    def __repr__(self) -> str:
        return (
            "JustConcaveSREAgent("
            f"players={self.num_players}, state_dim={self.state_dim}, action_dim={self.action_dim})"
        )

    def _to_device(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.to(self.device)

    def _compile_network_forwards(self) -> None:
        if not hasattr(torch, "compile"):
            warnings.warn("torch.compile is unavailable; using eager networks.", RuntimeWarning)
            return
        try:
            self.action_net.forward = torch.compile(self.action_net.forward)
            self.value_net.forward = torch.compile(self.value_net.forward)
            self.slow_val_net.forward = torch.compile(self.slow_val_net.forward)
        except Exception as exc:
            warnings.warn(
                "torch.compile failed for JustConcaveSREAgent; using eager networks. "
                f"Original error: {exc}",
                RuntimeWarning,
            )

    def _batch_size_from_states(self, states: torch.Tensor) -> int:
        if states.shape[0] % self.num_players != 0:
            raise ValueError("state rows must equal batch_size * num_players")
        return states.shape[0] // self.num_players

    def _reshape_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.to(self.device)
        if actions.dim() == 1:
            actions = actions.view(-1, self.num_players, self.action_dim)
        elif actions.dim() == 2:
            if self.action_dim == 1 and actions.shape[1] == self.num_players:
                actions = actions.unsqueeze(-1)
            elif actions.shape[1] == self.num_players * self.action_dim:
                actions = actions.view(-1, self.num_players, self.action_dim)
            else:
                raise ValueError("cannot infer joint action shape")
        elif actions.dim() != 3:
            raise ValueError("actions must be 1D, 2D, or 3D")
        return actions

    def _actor_warm_start(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions_flat, lambdas_flat = self.action_net(states)
        batch_size = self._batch_size_from_states(states)
        actions = actions_flat.view(batch_size, self.num_players, self.action_dim)
        lambdas = lambdas_flat.view(batch_size, self.num_players)
        return actions, lambdas

    def _payoffs(self, critic: JointQCritic, states: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        return critic(states, joint_actions)

    def _solve_sre(self, states: torch.Tensor, eps: float, critic: JointQCritic) -> SRESolution:
        with torch.no_grad():
            init_actions, init_lambdas = self._actor_warm_start(states)

        def payoff_fn(s: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
            return self._payoffs(critic, s, joint_actions)

        return self.solver.solve(
            states=states,
            initial_actions=init_actions,
            initial_lambdas=init_lambdas,
            eps=eps,
            payoff_fn=payoff_fn,
        )

    def predict_action(self, states: torch.Tensor, invt_states=None) -> torch.Tensor:
        states = self._to_device(states)
        actions, lambdas = self.action_net(states)
        return torch.cat([actions, lambdas.unsqueeze(1)], dim=1)

    def compute_sre_action(
        self,
        states: torch.Tensor,
        invt_states=None,
        eps: float = 0.0,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        states = self._to_device(states)
        solution = self._solve_sre(states, eps, self.value_net)
        actions = solution.actions.reshape(-1, self.action_dim)
        if noise_std > 0:
            actions = actions + torch.randn_like(actions) * float(noise_std)
            actions = self.solver._project_actions(actions.view(-1, self.num_players, self.action_dim))
            actions = actions.reshape(-1, self.action_dim)
        if self.action_dim == 1:
            return actions.view(-1)
        return actions

    def compute_actor_action(
        self,
        states: torch.Tensor,
        invt_states=None,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        del invt_states
        states = self._to_device(states)
        actions, _ = self._actor_warm_start(states)
        if noise_std > 0:
            actions = actions + torch.randn_like(actions) * float(noise_std)
            actions = self.solver._project_actions(actions)
        actions = actions.reshape(-1, self.action_dim)
        if self.action_dim == 1:
            return actions.view(-1)
        return actions

    def compute_training_diagnostics(
        self,
        state_tuples,
        eps: float = 0.0,
        max_transitions: int | None = 256,
    ) -> dict[str, float]:
        """Return scalar health metrics for the SRE target generator."""
        cur_s = self._to_device(state_tuples[0])
        next_s = self._to_device(state_tuples[2])
        rollout_actions = self._reshape_actions(state_tuples[6])

        batch_size = self._batch_size_from_states(next_s)
        if max_transitions is not None and batch_size > max_transitions:
            keep = int(max_transitions)
            rows = keep * self.num_players
            cur_s = cur_s[:rows]
            next_s = next_s[:rows]
            rollout_actions = rollout_actions[:keep]

        with torch.enable_grad():
            next_solution = self._solve_sre(next_s, eps, self.slow_val_net)

            def payoff_fn(s: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
                return self._payoffs(self.slow_val_net, s, joint_actions)

            if self.solver.vectorized:
                adv = self.solver._adversarial_profiles_all(
                    next_s,
                    next_solution.actions,
                    next_solution.lambdas,
                    payoff_fn,
                )
                transport_cost = self.solver._distance_all_players(next_solution.actions, adv)
            else:
                costs = []
                for player_idx in range(self.num_players):
                    adv = self.solver._adversarial_profile(
                        next_s,
                        next_solution.actions,
                        next_solution.lambdas,
                        player_idx,
                        payoff_fn,
                    )
                    costs.append(self.solver._distance(next_solution.actions, adv, player_idx))
                transport_cost = torch.stack(costs, dim=0)

        low, high = self.solver._bounds(next_solution.actions)
        action_tol = 1e-3
        sre_at_low = next_solution.actions <= (low + action_tol)
        sre_at_high = next_solution.actions >= (high - action_tol)
        rollout_at_low = rollout_actions <= (low + action_tol)
        rollout_at_high = rollout_actions >= (high - action_tol)

        actor_actions, actor_lambdas = self._actor_warm_start(cur_s)

        metrics = {
            "sre_robust_abs_max": next_solution.robust_values.abs().max(),
            "sre_robust_abs_mean": next_solution.robust_values.abs().mean(),
            "sre_nominal_abs_max": next_solution.nominal_values.abs().max(),
            "sre_nominal_abs_mean": next_solution.nominal_values.abs().mean(),
            "sre_lambda_max": next_solution.lambdas.max(),
            "sre_lambda_mean": next_solution.lambdas.mean(),
            "sre_transport_max": transport_cost.max(),
            "sre_transport_mean": transport_cost.mean(),
            "sre_residual": next_solution.residual,
            "sre_action_abs_max": next_solution.actions.abs().max(),
            "sre_action_saturation": (sre_at_low | sre_at_high).float().mean(),
            "rollout_action_abs_max": rollout_actions.abs().max(),
            "rollout_action_saturation": (rollout_at_low | rollout_at_high).float().mean(),
            "actor_action_abs_max": actor_actions.abs().max(),
            "actor_lambda_max": actor_lambdas.max(),
            "actor_lambda_mean": actor_lambdas.mean(),
        }
        return {name: float(value.detach().cpu()) for name, value in metrics.items()}

    def compute_value_Loss(self, state_tuples, eps: float = 0.0) -> torch.Tensor:
        cur_s = self._to_device(state_tuples[0])
        next_s = self._to_device(state_tuples[2])
        is_last = self._to_device(state_tuples[4])
        rewards = self._to_device(state_tuples[5])
        actions = self._reshape_actions(state_tuples[6])

        current_q = self.value_net(cur_s, actions)
        with torch.no_grad():
            next_solution = self._solve_sre(next_s, eps, self.slow_val_net)
            next_q = next_solution.robust_values
            not_last = 1.0 - is_last.view_as(rewards)
            target = rewards + self.gamma * not_last * next_q
        return F.mse_loss(current_q, target)

    def compute_action_Loss(self, state_tuples, eps: float = 0.0) -> torch.Tensor:
        cur_s = self._to_device(state_tuples[0])
        with torch.no_grad():
            solution = self._solve_sre(cur_s, eps, self.value_net)

        pred_actions, pred_lambdas = self._actor_warm_start(cur_s)
        imitation = F.mse_loss(pred_actions, solution.actions)
        lambda_imitation = F.mse_loss(pred_lambdas, solution.lambdas)
        if self.policy_value_weight:
            policy_value = self.value_net(cur_s, pred_actions).mean()
        else:
            policy_value = pred_actions.new_zeros(())
        return (
            self.imitation_weight * imitation
            + self.lambda_imitation_weight * lambda_imitation
            - self.policy_value_weight * policy_value
        )

    def update_slow(self, hard: bool = False) -> None:
        if hard:
            self.slow_val_net.load_state_dict(dc(self.value_net.state_dict()))
            return
        with torch.no_grad():
            for target_param, param in zip(self.slow_val_net.parameters(), self.value_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
