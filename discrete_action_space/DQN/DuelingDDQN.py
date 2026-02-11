from dqn_common import BaseDDqnAgent, DuelingQNetwork


class DuelingDDqnAgent(BaseDDqnAgent):
    """Dueling Double DQN."""

    def _build_network(self) -> DuelingQNetwork:
        return DuelingQNetwork(self.obs_dim, self.num_actions)
