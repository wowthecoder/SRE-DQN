from dqn_common import BaseDDqnSrqAgent, DuelingQNetwork


class DuelingDDqnSrqAgent(BaseDDqnSrqAgent):
    """Dueling Double DQN."""

    def _build_network(self) -> DuelingQNetwork:
        return DuelingQNetwork(self.obs_dim, self.num_actions)
