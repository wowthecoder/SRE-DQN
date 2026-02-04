from dqn_common import BaseDqnSrqAgent


class VanillaDqnSrqAgent(BaseDqnSrqAgent):
    """Vanilla DQN bootstrap: max over target network."""

    def _compute_next_q(self, next_states_t):
        return self.target_net(next_states_t).max(dim=1).values
