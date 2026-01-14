import numpy as np
from juliacall import Main as jl

class SRQAgent:
    """
    Implements Algorithm 3: Strategically Robust Q-Learning (SRQ).
    
    Each agent models the Q-functions of ALL agents to compute the 
    Strategically Robust Equilibrium (SRE) at each step.
    """
    def __init__(self, agent_id, num_agents, num_actions, 
                 epsilon_robust=1.0, epsilon_explore=1.0, 
                 alpha=0.1, gamma=0.9, decay_rate=0.999,
                 julia_source_path="bimatrix_asym_solver.jl"):
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
        # Structure: Dictionary mapping state -> Tensor of shape (num_actions, ..., num_actions, num_agents)
        # We store Q-values for ALL agents (j=1...n) to calculate equilibrium.
        self.q_table = {}

        # --- JULIA SETUP (juliacall) ---
        print(f"Loading Julia solver from {julia_source_path}...")
        
        # 1. Include the source file
        jl.include(julia_source_path)
        
        # 2. Bind the function. 
        # Note: juliacall wraps functions as Python objects automatically.
        self.julia_solve_func = jl.solve_strategically_robust_bimatrix_game
        
        # 3. Create a re-usable Symbol for dictionary access
        # In Julia, keys are :p1, :p2. In PythonCall, we need the Symbol object.
        self.sym_p1 = jl.Symbol("p1")
        self.sym_p2 = jl.Symbol("p2")

        # 4. JIT Warm-up
        print("Warming up JIT compilation (this may take a few seconds)...")
        dummy_u1 = np.random.rand(num_actions, num_actions)
        dummy_u2 = np.random.rand(num_actions, num_actions)
        # Call with dummy data
        self.julia_solve_func(dummy_u1, dummy_u2, [1.0, 1.0], 1)
        print("Julia Solver Ready.")

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
        Calculates the Strategically Robust Equilibrium using the Julia PATH solver.
        """
        q_tensor = self.get_q_values(state)
        
        # Extract Payoff Matrices
        U1 = q_tensor[:, :, 0]
        U2 = q_tensor[:, :, 1]

        # Call Julia Solver
        # juliacall handles NumPy -> Julia Array conversion automatically.
        try:
            # Returns tuple: (solutions, utilities_sr, utilities_nominal)
            results = self.julia_solve_func(
                U1, U2, 
                [self.epsilon_robust, self.epsilon_robust], 
                3  # num_repeats
            )
            # Access the first element of the tuple (the solutions vector)
            solutions_jl = results[0]
            
        except Exception as e:
            print(f"Julia Solver Error: {e}")
            return [np.ones(self.num_actions)/self.num_actions] * 2

        if len(solutions_jl) == 0:
            # Fallback if solver fails to converge
            uniform = np.ones(self.num_actions) / self.num_actions
            return [uniform, uniform]

        # Equilibrium Selection: Highest Joint Reward
        best_sol = None
        best_joint_reward = -float('inf')

        # Iterate over the Julia Vector of Dictionaries
        for sol_jl in solutions_jl:
            # sol_jl is a Julia Dict. We access keys using the symbols we created.
            # Then we convert the Julia Array to a NumPy array.
            p1 = np.array(sol_jl[self.sym_p1])
            p2 = np.array(sol_jl[self.sym_p2])
            
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

# --- Example Training Loop (Pseudocode for Context) ---
# See
def train_srq(env, num_episodes=3000):
    # Initialize agents (assuming 2 agents, 4 actions each)
    agents = [SRQAgent(i, 2, 4) for i in range(2)]
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 1. Choose actions
            actions = [agent.act(state) for agent in agents]
            
            # 2. Step environment
            next_state, rewards, done, _ = env.step(actions)
            
            # 3. Update Q-tables
            # Note: Each agent acts independently but models the other.
            # In a centralized training simulation, we can loop:
            for agent in agents:
                agent.update(state, actions, rewards, next_state)
            
            state = next_state
        
        # 4. Decay parameters
        for agent in agents:
            agent.decay_parameters()