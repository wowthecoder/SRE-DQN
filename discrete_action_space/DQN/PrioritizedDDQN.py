from dqn_common import BasePrioritizedDqnSrqAgent


class PrioritizedDDqnSrqAgent(BasePrioritizedDqnSrqAgent):
    """Double DQN with prioritized experience replay."""

    def _compute_next_q(self, next_states_t):
        next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
        return self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
