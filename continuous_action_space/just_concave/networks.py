"""Neural modules for the just-concave SRE-DQN prototype."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn


def _as_tuple(hidden_sizes: Iterable[int] | int) -> Tuple[int, ...]:
    if isinstance(hidden_sizes, int):
        return (hidden_sizes, hidden_sizes, hidden_sizes)
    return tuple(int(size) for size in hidden_sizes)


class MLP(nn.Module):
    """Small SiLU MLP used by the actor, lambda head, and critic."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Iterable[int] | int = (128, 128, 128),
    ) -> None:
        super().__init__()
        sizes = _as_tuple(hidden_sizes)
        layers = []
        last_dim = input_dim
        for width in sizes:
            layers.append(nn.Linear(last_dim, width))
            layers.append(nn.SiLU())
            last_dim = width
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorLambdaNet(nn.Module):
    """
    Per-agent amortized warm-start network.

    The input state tensor is flattened as [batch * num_players, state_dim], matching
    the locally-linear-quadratic training utilities. The action output has shape
    [batch * num_players, action_dim], and lambda has shape [batch * num_players].
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_sizes: Iterable[int] | int = (128, 128, 128),
        action_low: float | Iterable[float] = -1.0,
        action_high: float | Iterable[float] = 1.0,
        lambda_min: float = 1e-4,
        lambda_max: float = 100.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.body = MLP(state_dim, self.action_dim + 1, hidden_sizes)

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

        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.body(states)
        raw_action = raw[:, : self.action_dim]
        raw_lambda = raw[:, self.action_dim]
        center = 0.5 * (self.action_low + self.action_high)
        radius = 0.5 * (self.action_high - self.action_low)
        actions = center + radius * torch.tanh(raw_action)
        lambdas = self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(raw_lambda)
        return actions, lambdas


class JointQCritic(nn.Module):
    """
    Shared per-agent critic Q_i(s_i, a_joint).

    The state row is the per-agent perspective produced by existing state expansion.
    The joint action is repeated for each agent row before calling this module.
    """

    def __init__(
        self,
        state_dim: int,
        num_players: int,
        action_dim: int = 1,
        hidden_sizes: Iterable[int] | int = (128, 128, 128),
    ) -> None:
        super().__init__()
        self.num_players = int(num_players)
        self.action_dim = int(action_dim)
        input_dim = int(state_dim) + self.num_players * self.action_dim
        self.q_net = MLP(input_dim, 1, hidden_sizes)

    def forward(self, states: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        if joint_actions.dim() != 3:
            raise ValueError("joint_actions must have shape [batch, num_players, action_dim]")
        batch_size = joint_actions.shape[0]
        flat_actions = joint_actions.reshape(batch_size, self.num_players * self.action_dim)
        repeated_actions = flat_actions.repeat_interleave(self.num_players, dim=0)
        if states.shape[0] != repeated_actions.shape[0]:
            raise ValueError("states must have batch*num_players rows")
        return self.q_net(torch.cat([states, repeated_actions], dim=1)).view(batch_size, self.num_players)
