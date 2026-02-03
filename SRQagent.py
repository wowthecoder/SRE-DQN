import numpy as np
from path_solver import PathSolverWrapper, solve_strategically_robust_bimatrix_game_path

class SRQAgent:
    """
    Implements Strategically Robust Q-Learning (SRQ).
    
    Each agent models the Q-functions of ALL agents to compute the 
    Strategically Robust Equilibrium (SRE) at each step.
    """
    def __init__(self, agent_id, num_agents, num_actions,
                 epsilon_robust=1.0, epsilon_explore=1.0,
                 alpha=0.1, gamma=0.9, decay_rate=0.999,
                 pathwrap_path="pathwrap.so"):
        """
        Args:
            agent_id (int): Index of this agent (0 to n-1).
            num_agents (int): Total number of agents (n).
            num_actions (int): Number of actions available to each agent.
            epsilon_robust (float): Robustness parameter (epsilon in paper).
            epsilon_explore (float): Exploration parameter (e-greedy).
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            decay_rate (float): Decay rate for alpha, epsilon_robust, epsilon_explore.
            pathwrap_path (str): Path to the compiled pathwrap shared library.
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.num_actions = num_actions
        
        # Parameters
        self.epsilon_robust = epsilon_robust
        self.epsilon_explore = epsilon_explore
        self.alpha = alpha
        self.gamma = gamma
        self.decay_rate = decay_rate

        # Initialize Q-tables: Q_i^j(s, a1, ..., an) = 0
        # Structure: Dictionary mapping state -> Tensor of shape (num_actions_for agent 1, ..., num_actions for agent n, num_agents)
        # We store Q-values for ALL agents (j=1...n) to calculate equilibrium.
        self.q_table = {}

        # --- PATH SETUP (ctypes wrapper) ---
        print(f"Loading PATH wrapper from {pathwrap_path}...")
        self.path_solver = PathSolverWrapper(pathwrap_path)
        print("PATH Solver Ready.")

    def get_q_values(self, state):
        """Returns the Q-matrix for a given state, initializing if necessary."""
        state_key = str(state)
        if state_key not in self.q_table:
            # Shape: (A_1, A_2, ..., A_n, num_agents)
            shape = [self.num_actions] * self.num_agents + [self.num_agents]
            self.q_table[state_key] = np.zeros(shape)
        return self.q_table[state_key]

    def solve_sre(self, state):
        """
        Calculates the Strategically Robust Equilibrium using the PATH solver.
        """
        q_tensor = self.get_q_values(state)
        
        # Extract Payoff Matrices
        U1 = q_tensor[:, :, 0]
        U2 = q_tensor[:, :, 1]

        try:
            results = solve_strategically_robust_bimatrix_game_path(
                U1, U2,
                [self.epsilon_robust, self.epsilon_robust],
                3,
                self.path_solver
            )
            solutions = results[0]
        except Exception as e:
            print(f"PATH Solver Error: {e}")
            return [np.ones(self.num_actions) / self.num_actions] * 2

        if len(solutions) == 0:
            # Fallback if solver fails to converge
            uniform = np.ones(self.num_actions) / self.num_actions
            return [uniform, uniform]

        # Equilibrium Selection: Highest Joint Reward
        best_sol = None
        best_joint_reward = -float('inf')

        for sol in solutions:
            p1 = np.array(sol["p1"])
            p2 = np.array(sol["p2"])
            
            # Calculate Expected Joint Reward
            # R1 = p1 . U1 . p2
            r1 = p1 @ U1 @ p2
            # R2 = p1 . U2 . p2
            r2 = p1 @ U2 @ p2
            
            current_reward = r1 + r2
            
            if current_reward > best_joint_reward:
                best_joint_reward = current_reward
                best_sol = [p1, p2]

        return best_sol

    def calculate_srq_value(self, state):
        """
        Calculates the SRQ value: The expected reward under SRE policies.
        SRQ_t^i(s', epsilon) = Product(pi_SR) * Q_t^i(s') 
        """
        # 1. Get SRE policies for the next state [cite: 419]
        policies = self.solve_sre(state) # List of arrays [pi_1, pi_2, ...]
        
        # 2. Retrieve Q-values for next state
        q_values = self.get_q_values(state) # Shape: (A1, A2, ..., N_agents)
        
        # 3. Compute expected value: sum(p(a) * Q(s, a))
        # This computes the expectation over the joint action space
        expected_values = q_values
        for agent_idx, policy in enumerate(policies):
            # Contract the tensor along the axis of the agent's actions
            expected_values = np.tensordot(policy, expected_values, axes=([0], [0]))
            
        # expected_values is now an array of shape (num_agents,) containing the
        # expected return for each agent j at the SRE of state s'.
        return expected_values

    def act(self, state):
        """
        Choose action using e-greedy sampling from the SRE on Q_i(s).
        """
        # 1. Calculate SRE policies
        policies = self.solve_sre(state)
        my_policy = policies[self.agent_id]
        
        # 2. Normalize to ensure probabilities sum to 1 (numerical stability)
        my_policy = my_policy / np.sum(my_policy)
        
        # 3. Epsilon-greedy exploration
        if np.random.rand() < self.epsilon_explore:
            return np.random.choice(self.num_actions)
        else:
            # Sample from the SRE probability distribution
            return np.random.choice(self.num_actions, p=my_policy)

    def update(self, state, actions, rewards, next_state):
        """
        Update Q-values for ALL agents j=1...n based on experience.
        
        Formula:
        Q_i^j(s, a) = (1 - alpha) * Q_i^j(s, a) + alpha * [r^j + gamma * SRQ_i^j(s', epsilon)]
       
        """
        # 1. Get current Q-values
        state_key = str(state)
        q_tensor = self.get_q_values(state)
        
        # 2. Calculate the 'target' value (SRQ value of next state) 
        srq_values_next = self.calculate_srq_value(next_state) # Returns array of shape (num_agents,)
        
        # 3. Create indices tuple to access the specific joint action cell in tensor
        # actions is list [a1, a2], we append slice(None) to update all agents' Q-values at once
        idx = tuple(actions) + (slice(None),)
        
        # 4. Perform the update
        current_vals = q_tensor[idx] # Array of shape (num_agents,)
        
        # Bellman update
        new_vals = (1 - self.alpha) * current_vals + \
                   self.alpha * (np.array(rewards) + self.gamma * srq_values_next)
                   
        self.q_table[state_key][idx] = new_vals

    def decay_parameters(self):
        """
        Linearly decay epsilon, epsilon_explore, and alpha.
       
        """
        self.epsilon_robust *= self.decay_rate
        self.epsilon_explore *= self.decay_rate
        self.alpha *= self.decay_rate
