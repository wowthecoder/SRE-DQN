"""Mean-field Q-network for MF-DSRQ.

Architecture:
    ConvTrunk(obs) → feature [128]
    Linear head:   feature → [A_own × A_nbr]
    Reshape:       → [A_own, A_nbr]

Output q_grid[a_own, a_nbr] = Q(o, a_own, a_nbr).
Mean-field aggregation: Q(o, a_own ; ā) = Σ_k ā[k] · q_grid[a_own, k].

One network per agent-type; all agents of the same type share weights.
"""

import torch
import torch.nn as nn


class MeanFieldQNetwork(nn.Module):
    def __init__(
        self,
        obs_channels: int,
        obs_height: int,
        obs_width: int,
        n_own_actions: int,
        n_nbr_actions: int,
    ):
        super().__init__()
        self.n_own_actions = n_own_actions
        self.n_nbr_actions = n_nbr_actions

        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        conv_out = 32 * obs_height * obs_width
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, n_own_actions * n_nbr_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, C, H, W]

        Returns:
            q_grid: [B, A_own, A_nbr]
        """
        feat = self.trunk(self.conv(obs))
        B = obs.shape[0]
        return self.head(feat).reshape(B, self.n_own_actions, self.n_nbr_actions)
