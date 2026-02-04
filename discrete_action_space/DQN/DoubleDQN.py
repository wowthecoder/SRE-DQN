from dqn_common import BaseDqnSrqAgent


class DDqnSrqAgent(BaseDqnSrqAgent):
    """Double DQN bootstrap: select via online net, evaluate via target net."""

    def _compute_next_q(self, next_states_t):
        next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
        return self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
