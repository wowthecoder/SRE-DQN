"""Policy network for SR-ADIDAS.

SharedTrunkPolicyNet: shared state encoder with N per-agent softmax heads,
mirroring DuelingSharedTrunkPerAgentJointQNetwork from dueling_double_dqn_sre.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedTrunkPolicyNet(nn.Module):
    """Shared encoder + N per-agent softmax policy heads.

    Output is a list of N tensors of shape (B, num_actions), each a valid
    probability distribution over the discrete action space.
    """

    def __init__(self, obs_dim, num_actions, num_agents, hidden_dim=128):
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_actions) for _ in range(num_agents)]
        )

    def forward(self, state):
        """
        Args:
            state: (B, obs_dim) float tensor.
        Returns:
            List of N tensors, each (B, num_actions), on the probability simplex.
        """
        features = self.trunk(state)
        return [F.softmax(head(features), dim=-1) for head in self.heads]
