import numpy as np

from SRQagent import SRQAgent

try:
    import pygambit as gambit
except ImportError:  # pragma: no cover - depends on local environment
    gambit = None


class NashQAgent(SRQAgent):
    """
    Tabular Nash-Q agent following the same joint-Q structure as SRQAgent.

    This mirrors Jack Brand's NashQ design choice of:
    - each agent modelling all agents' Q-values
    - computing a stage-game equilibrium from those Q-values
    - selecting the equilibrium with highest joint reward when multiple exist

    Nash equilibria are computed with pygambit, matching Jack's notebook.
    """

    def __init__(
        self,
        agent_id,
        num_agents,
        num_actions,
        epsilon_robust=1.0,
        epsilon_explore=1.0,
        alpha=0.1,
        gamma=0.9,
        decay_rate=0.998,
        epsilon_robust_min=0.01,
        epsilon_explore_min=0.01,
        alpha_min=1 / 3000,
        pathwrap_path="pathwrap.so",
    ):
        del pathwrap_path
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.epsilon_robust = epsilon_robust
        self.epsilon_explore = epsilon_explore
        self.alpha = alpha
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.epsilon_robust_min = epsilon_robust_min
        self.epsilon_explore_min = epsilon_explore_min
        self.alpha_min = alpha_min
        self.q_table = {}

        if gambit is None:
            raise ImportError(
                "pygambit is required for NashQAgent. Install it in the active environment."
            )

    def get_nash_equilibrium_probabilities(self, u1, u2):
        game = gambit.Game.from_arrays(u1, u2)
        result = gambit.nash.enummixed_solve(game)

        equilibria = list(result.equilibria)
        if not equilibria:
            return None

        num_players = 2
        num_strategies = self.num_actions
        num_equilibria = len(equilibria)

        float_equilibria = [
            [[0.0 for _ in range(num_equilibria)] for _ in range(num_strategies)]
            for _ in range(num_players)
        ]

        for eq_idx, eq in enumerate(equilibria):
            for player_idx, player in enumerate(game.players):
                for strat_idx, strategy in enumerate(player.strategies):
                    prob = float(eq[strategy])
                    float_equilibria[player_idx][strat_idx][eq_idx] = prob

        best_equilibrium = None
        highest_joint_payoff = -float("inf")

        for eq_idx in range(num_equilibria):
            probs_player_1 = np.asarray(
                [float_equilibria[0][i][eq_idx] for i in range(num_strategies)],
                dtype=np.float64,
            )
            probs_player_2 = np.asarray(
                [float_equilibria[1][i][eq_idx] for i in range(num_strategies)],
                dtype=np.float64,
            )

            player_1_payoff = float(probs_player_1 @ u1 @ probs_player_2)
            player_2_payoff = float(probs_player_1 @ u2 @ probs_player_2)
            joint_payoff = player_1_payoff + player_2_payoff

            if joint_payoff > highest_joint_payoff:
                highest_joint_payoff = joint_payoff
                best_equilibrium = eq_idx

        if best_equilibrium is None:
            return None

        best_equilibrium_probs = np.array(
            [
                [
                    float_equilibria[player][i][best_equilibrium]
                    for i in range(num_strategies)
                ]
                for player in range(num_players)
            ],
            dtype=np.float64,
        )
        return best_equilibrium_probs

    def solve_nash_from_q_values(self, q_tensor):
        u1 = q_tensor[:, :, 0]
        u2 = q_tensor[:, :, 1]
        equilibrium = self.get_nash_equilibrium_probabilities(u1, u2)
        if equilibrium is None:
            return self._uniform_policies()
        return [
            self._normalize_policy(equilibrium[0]),
            self._normalize_policy(equilibrium[1]),
        ]

    def solve_sre_from_q_values(self, q_tensor):
        return self.solve_nash_from_q_values(q_tensor)

    def solve_nash(self, state):
        return self.solve_nash_from_q_values(self.get_q_values(state))

    def solve_sre(self, state):
        return self.solve_nash(state)

    def calculate_nashq_value(self, state, policies=None):
        return self.calculate_srq_value(state, policies=policies)

    @staticmethod
    def solve_shared_nash(agents, state, solver_agent_idx=0):
        if not agents:
            raise ValueError("Expected at least one NashQAgent.")
        solver_agent = agents[solver_agent_idx]
        q_tensor = solver_agent.get_q_values(state)
        return solver_agent.solve_nash_from_q_values(q_tensor)

    @staticmethod
    def solve_shared_sre(agents, state, solver_agent_idx=0):
        return NashQAgent.solve_shared_nash(
            agents, state, solver_agent_idx=solver_agent_idx
        )

    def close(self):
        return None
