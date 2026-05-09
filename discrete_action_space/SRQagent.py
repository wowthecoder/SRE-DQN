import numpy as np
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from path_solver import PathSolverWrapper, solve_strategically_robust_bimatrix_game_path_lcp


@dataclass
class SRQAgentConfig:
    agent_id: int
    num_agents: int
    num_actions: int
    epsilon_robust: float = 1.0
    epsilon_explore: float = 1.0
    alpha: float = 0.1
    gamma: float = 0.9
    decay_rate: float = 0.998
    epsilon_robust_min: float = 0.01
    epsilon_explore_min: float = 0.01
    alpha_min: float = 1 / 3000
    pathwrap_path: str = "pathwrap.so"
    epsilon_schedule: str = "exponential"


class SRQAgent:
    """
    Implements Strategically Robust Q-Learning (SRQ).

    Each agent models the Q-functions of ALL agents to compute the
    Strategically Robust Equilibrium (SRE) at each step.
    """

    algorithm = "srq"

    def __init__(self, config: SRQAgentConfig):
        self.config = config
        self.initial_epsilon_robust = float(config.epsilon_robust)

        # Initialize Q-tables: Q_i^j(s, a1, ..., an) = 0
        # Structure: Dictionary mapping state -> Tensor of shape (num_actions_for agent 1, ..., num_actions for agent n, num_agents)
        # We store Q-values for ALL agents (j=1...n) to calculate equilibrium.
        self.q_table = {}
        self.update_times = []
        self.sre_solve_time_count = 0
        self.sre_solve_time_sum = 0.0
        self.sre_solve_time_sumsq = 0.0
        self.sre_solve_time_min = None
        self.sre_solve_time_max = None

        # --- PATH SETUP (ctypes wrapper) ---
        print(f"Loading PATH wrapper from {config.pathwrap_path}...")
        self.path_solver = PathSolverWrapper(config.pathwrap_path)
        print("PATH Solver Ready.")

    def _record_sre_solve_time(self, duration):
        self.sre_solve_time_count += 1
        self.sre_solve_time_sum += duration
        self.sre_solve_time_sumsq += duration * duration
        if self.sre_solve_time_min is None or duration < self.sre_solve_time_min:
            self.sre_solve_time_min = duration
        if self.sre_solve_time_max is None or duration > self.sre_solve_time_max:
            self.sre_solve_time_max = duration

    def get_sre_solve_time_summary(self):
        count = self.sre_solve_time_count
        if count == 0:
            return {
                "count": 0,
                "mean_seconds": None,
                "min_seconds": None,
                "max_seconds": None,
                "std_seconds": None,
                "mean_microseconds": None,
                "min_microseconds": None,
                "max_microseconds": None,
                "std_microseconds": None,
            }
        mean = self.sre_solve_time_sum / count
        variance = max(self.sre_solve_time_sumsq / count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        return {
            "count": int(count),
            "mean_seconds": float(mean),
            "min_seconds": float(self.sre_solve_time_min),
            "max_seconds": float(self.sre_solve_time_max),
            "std_seconds": std,
            "mean_microseconds": float(mean * 1_000_000.0),
            "min_microseconds": float(self.sre_solve_time_min * 1_000_000.0),
            "max_microseconds": float(self.sre_solve_time_max * 1_000_000.0),
            "std_microseconds": float(std * 1_000_000.0),
        }

    def _uniform_policies(self):
        uniform = np.ones(self.config.num_actions) / self.config.num_actions
        return [uniform.copy() for _ in range(self.config.num_agents)]

    def _normalize_policy(self, policy):
        p = np.asarray(policy, dtype=np.float64)
        p = np.clip(p, 0.0, None)
        total = float(np.sum(p))
        if total <= 0.0:
            return np.ones(self.config.num_actions, dtype=np.float64) / self.config.num_actions
        return p / total

    def solve_sre_from_q_values(self, q_tensor):
        """Solves the SRE for a provided joint Q-tensor."""
        # Extract Payoff Matrices
        U1 = q_tensor[:, :, 0]
        U2 = q_tensor[:, :, 1]

        solve_start = time.perf_counter()
        try:
            results = solve_strategically_robust_bimatrix_game_path_lcp(
                U1,
                U2,
                [self.config.epsilon_robust, self.config.epsilon_robust],
                20,
                self.path_solver,
            )
            solutions = results[0]
        except Exception as e:
            self._record_sre_solve_time(time.perf_counter() - solve_start)
            print(f"PATH Solver Error: {e}")
            return self._uniform_policies()
        self._record_sre_solve_time(time.perf_counter() - solve_start)

        if len(solutions) == 0:
            # Fallback if solver fails to converge
            return self._uniform_policies()

        # Equilibrium Selection: Highest Joint Reward
        best_sol = None
        best_joint_reward = -float("inf")

        for sol in solutions:
            p1 = self._normalize_policy(sol["p1"])
            p2 = self._normalize_policy(sol["p2"])

            # Calculate Expected Joint Reward
            r1 = p1 @ U1 @ p2
            r2 = p1 @ U2 @ p2

            current_reward = r1 + r2

            if current_reward > best_joint_reward:
                best_joint_reward = current_reward
                best_sol = [p1, p2]

        return best_sol if best_sol is not None else self._uniform_policies()

    def get_q_values(self, state):
        """Returns the Q-matrix for a given state, initializing if necessary."""
        state_key = str(state)
        if state_key not in self.q_table:
            # Shape: (A_1, A_2, ..., A_n, num_agents)
            shape = [self.config.num_actions] * self.config.num_agents + [self.config.num_agents]
            self.q_table[state_key] = np.zeros(shape)
        return self.q_table[state_key]

    def solve_sre(self, state):
        """
        Calculates the Strategically Robust Equilibrium using the PATH solver.
        """
        q_tensor = self.get_q_values(state)
        return self.solve_sre_from_q_values(q_tensor)

    @staticmethod
    def solve_shared_sre(agents, state, solver_agent_idx=0):
        """
        Solves one shared SRE policy pair for all agents using a designated solver agent.
        """
        if not agents:
            raise ValueError("Expected at least one SRQAgent.")
        solver_agent = agents[solver_agent_idx]
        q_tensor = solver_agent.get_q_values(state)
        return solver_agent.solve_sre_from_q_values(q_tensor)

    def calculate_srq_value(self, state, policies=None):
        """
        Calculates the SRQ value: The expected reward under SRE policies.
        SRQ_t^i(s', epsilon) = Product(pi_SR) * Q_t^i(s')
        """
        # 1. Get SRE policies for the next state [cite: 419]
        if policies is None:
            policies = self.solve_sre(state) # List of arrays [pi_1, pi_2, ...]

        # 2. Retrieve Q-values for next state
        q_values = self.get_q_values(state) # Shape: (A1, A2, ..., N_agents)

        # 3. Compute expected value: sum(p(a) * Q(s, a))
        # This computes the expectation over the joint action space
        expected_values = q_values
        for policy in policies:
            # Contract the tensor along the axis of the agent's actions
            expected_values = np.tensordot(policy, expected_values, axes=([0], [0]))

        # expected_values is now an array of shape (num_agents,) containing the
        # expected return for each agent j at the SRE of state s'.
        return expected_values

    def act(self, state, policies=None):
        """
        Choose action using e-greedy sampling from the SRE on Q_i(s).
        """
        # 1. Calculate SRE policies
        if policies is None:
            policies = self.solve_sre(state)
        my_policy = self._normalize_policy(policies[self.config.agent_id])

        # 2. Epsilon-greedy exploration
        if np.random.rand() < self.config.epsilon_explore:
            return int(np.random.choice(self.config.num_actions))
        else:
            # Sample from the SRE probability distribution
            return int(np.random.choice(self.config.num_actions, p=my_policy))

    def update(self, state, actions, rewards, next_state, done=False, next_policies=None):
        """
        Update Q-values for ALL agents j=1...n based on experience.

        Formula:
        Q_i^j(s, a) = (1 - alpha) * Q_i^j(s, a) + alpha * [r^j + gamma * SRQ_i^j(s', epsilon)]

        """
        # 1. Get current Q-values
        state_key = str(state)
        q_tensor = self.get_q_values(state)

        # 2. Calculate the bootstrapped next-state value unless the transition is terminal.
        if np.isscalar(done):
            if done:
                srq_values_next = np.zeros(self.config.num_agents, dtype=np.float64)
            else:
                srq_values_next = self.calculate_srq_value(
                    next_state, policies=next_policies
                )
        else:
            done_mask = np.asarray(done, dtype=np.float64).reshape(-1)
            if done_mask.shape[0] != self.config.num_agents:
                raise ValueError(
                    f"Expected done mask length {self.config.num_agents}, got {done_mask.shape[0]}."
                )
            srq_values_next = self.calculate_srq_value(
                next_state, policies=next_policies
            ) * (1.0 - done_mask)

        # 3. Create indices tuple to access the specific joint action cell in tensor
        # actions is list [a1, a2], we append slice(None) to update all agents' Q-values at once
        idx = tuple(actions) + (slice(None),)

        # 4. Perform the update
        current_vals = q_tensor[idx] # Array of shape (num_agents,)

        # Bellman update
        update_start = time.perf_counter()
        new_vals = (1 - self.config.alpha) * current_vals + \
                   self.config.alpha * (np.array(rewards) + self.config.gamma * srq_values_next)
        self.q_table[state_key][idx] = new_vals
        self.update_times.append(time.perf_counter() - update_start)

    def on_episode_end(self, *_):
        pass

    def save_checkpoint(self, path):
        self.save_q_table(path)

    def decay_parameters(
        self,
        *,
        epsilon_schedule=None,
        epsilon_robust_initial=None,
        episode_idx=None,
        n_episodes=None,
    ):
        if epsilon_schedule is None:
            epsilon_schedule = self.config.epsilon_schedule
        if epsilon_robust_initial is None:
            epsilon_robust_initial = self.initial_epsilon_robust

        if epsilon_schedule == "constant":
            self.config.epsilon_robust = float(epsilon_robust_initial)
        elif epsilon_schedule == "linear":
            if episode_idx is None or n_episodes is None:
                raise ValueError(
                    "episode_idx and n_episodes are required for linear epsilon decay."
                )
            if n_episodes <= 1:
                self.config.epsilon_robust = 0.0
            else:
                fraction = min(max(episode_idx, 0) / float(n_episodes - 1), 1.0)
                self.config.epsilon_robust = float(epsilon_robust_initial) * (1.0 - fraction)
        elif epsilon_schedule == "exponential":
            self.config.epsilon_robust = max(
                self.config.epsilon_robust_min,
                self.config.epsilon_robust * self.config.decay_rate,
            )
        else:
            raise ValueError(f"Unsupported epsilon schedule: {epsilon_schedule}")
        self.config.epsilon_explore = max(
            self.config.epsilon_explore_min,
            self.config.epsilon_explore * self.config.decay_rate,
        )
        self.config.alpha = max(
            self.config.alpha_min,
            self.config.alpha * self.config.decay_rate,
        )

    def save_q_table(self, path):
        payload = {
            "format_version": 2,
            "agent_class": type(self).__name__,
            "agent_id": self.config.agent_id,
            "num_agents": self.config.num_agents,
            "num_actions": self.config.num_actions,
            "epsilon_robust": self.config.epsilon_robust,
            "epsilon_explore": self.config.epsilon_explore,
            "alpha": self.config.alpha,
            "gamma": self.config.gamma,
            "decay_rate": self.config.decay_rate,
            "epsilon_robust_min": self.config.epsilon_robust_min,
            "epsilon_explore_min": self.config.epsilon_explore_min,
            "alpha_min": self.config.alpha_min,
            "q_table": self.q_table,
            "update_times": list(self.update_times),
            "sre_solve_time_count": self.sre_solve_time_count,
            "sre_solve_time_sum": self.sre_solve_time_sum,
            "sre_solve_time_sumsq": self.sre_solve_time_sumsq,
            "sre_solve_time_min": self.sre_solve_time_min,
            "sre_solve_time_max": self.sre_solve_time_max,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_q_table(self, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "q_table" in payload:
            self.config.agent_id = payload.get("agent_id", self.config.agent_id)
            self.config.num_agents = payload.get("num_agents", self.config.num_agents)
            self.config.num_actions = payload.get("num_actions", self.config.num_actions)
            self.config.epsilon_robust = payload.get(
                "epsilon_robust", self.config.epsilon_robust
            )
            self.config.epsilon_explore = payload.get(
                "epsilon_explore", self.config.epsilon_explore
            )
            self.config.alpha = payload.get("alpha", self.config.alpha)
            self.config.gamma = payload.get("gamma", self.config.gamma)
            self.config.decay_rate = payload.get("decay_rate", self.config.decay_rate)
            self.config.epsilon_robust_min = payload.get(
                "epsilon_robust_min", self.config.epsilon_robust_min
            )
            self.config.epsilon_explore_min = payload.get(
                "epsilon_explore_min", self.config.epsilon_explore_min
            )
            self.config.alpha_min = payload.get("alpha_min", self.config.alpha_min)
            self.q_table = payload["q_table"]
            self.update_times = list(payload.get("update_times", []))
            self.sre_solve_time_count = payload.get(
                "sre_solve_time_count", self.sre_solve_time_count
            )
            self.sre_solve_time_sum = payload.get(
                "sre_solve_time_sum", self.sre_solve_time_sum
            )
            self.sre_solve_time_sumsq = payload.get(
                "sre_solve_time_sumsq", self.sre_solve_time_sumsq
            )
            self.sre_solve_time_min = payload.get(
                "sre_solve_time_min", self.sre_solve_time_min
            )
            self.sre_solve_time_max = payload.get(
                "sre_solve_time_max", self.sre_solve_time_max
            )
        else:
            self.q_table = payload

    def close(self):
        self.path_solver.close()
