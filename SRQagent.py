import ctypes as ct
import os
import sys
import numpy as np


class PathSolverWrapper:
    def __init__(self, lib_path="pathwrap.so", lib_dir=None):
        lib_path = os.fspath(lib_path)
        if not os.path.isabs(lib_path):
            candidate = os.path.join(os.path.dirname(__file__), lib_path)
            if os.path.exists(candidate):
                lib_path = candidate

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"PATH wrapper library not found at: {lib_path}. "
                "Build pathwrap.so and/or pass an absolute path."
            )

        self._preload_path_libs(lib_dir, lib_path)
        mode = getattr(ct, "RTLD_GLOBAL", 0) | getattr(ct, "RTLD_NOW", 0)
        try:
            self.lib = ct.CDLL(lib_path, mode=mode)
        except OSError as e:
            raise OSError(
                f"Failed to load {lib_path}: {e}. "
                "Check that libpath50 is accessible (pathlib/lib_lnx) "
                "or preload it with ctypes.CDLL."
            ) from e

        self._func_type = ct.CFUNCTYPE(
            ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)
        )
        self._jac_type = ct.CFUNCTYPE(
            ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_int), ct.POINTER(ct.c_double)
        )

        self.lib.path_solve.argtypes = [
            ct.c_int, ct.c_int,
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
            self._func_type, self._jac_type,
            ct.POINTER(ct.c_int),
        ]

    def solve(self, n, nnz, z, f, lb, ub, func_eval, jac_eval):
        fe = self._func_type(func_eval)
        je = self._jac_type(jac_eval)
        self._last_fe = fe
        self._last_je = je
        status = ct.c_int()
        self.lib.path_solve(
            n, nnz,
            z.ctypes.data_as(ct.POINTER(ct.c_double)),
            f.ctypes.data_as(ct.POINTER(ct.c_double)),
            lb.ctypes.data_as(ct.POINTER(ct.c_double)),
            ub.ctypes.data_as(ct.POINTER(ct.c_double)),
            fe, je,
            ct.byref(status),
        )
        return status.value

    def _preload_path_libs(self, lib_dir, lib_path):
        base_dir = os.path.dirname(__file__)
        wrap_dir = os.path.dirname(os.path.abspath(lib_path))
        candidates = []
        if lib_dir:
            candidates.append(os.fspath(lib_dir))

        if sys.platform.startswith("linux"):
            lib_name = "libpath50.so"
            candidates.extend([
                os.path.join(base_dir, "pathlib", "lib_lnx"),
                os.path.join(wrap_dir, "pathlib", "lib_lnx"),
                os.path.join(os.getcwd(), "pathlib", "lib_lnx"),
            ])
        elif sys.platform == "darwin":
            lib_name = "libpath50.dylib"
            candidates.extend([
                os.path.join(base_dir, "pathlib", "lib_osx"),
                os.path.join(wrap_dir, "pathlib", "lib_osx"),
                os.path.join(os.getcwd(), "pathlib", "lib_osx"),
            ])
        elif sys.platform.startswith("win"):
            lib_name = "libpath50.dll"
            candidates.extend([
                os.path.join(base_dir, "pathlib", "lib_win"),
                os.path.join(wrap_dir, "pathlib", "lib_win"),
                os.path.join(os.getcwd(), "pathlib", "lib_win"),
            ])
        else:
            lib_name = "libpath50.so"
            candidates.extend([
                os.path.join(base_dir, "pathlib", "lib_lnx"),
                os.path.join(wrap_dir, "pathlib", "lib_lnx"),
                os.path.join(os.getcwd(), "pathlib", "lib_lnx"),
            ])

        mode = getattr(ct, "RTLD_GLOBAL", 0) | getattr(ct, "RTLD_NOW", 0)
        for d in candidates:
            dep = os.path.join(d, lib_name)
            if os.path.exists(dep):
                ct.CDLL(dep, mode=mode)
                return

        # If we got here, preload failed; fall through and let dlopen error out.


def _build_index_map(nA, nB):
    idx = 0
    s = {}
    s["p1"] = slice(idx, idx + nA); idx += nA
    s["p2"] = slice(idx, idx + nB); idx += nB
    s["lambda1"] = idx; idx += 1
    s["lambda2"] = idx; idx += 1
    s["xi1"] = slice(idx, idx + nB); idx += nB
    s["xi2"] = slice(idx, idx + nA); idx += nA
    s["eta1"] = slice(idx, idx + nB * nB); idx += nB * nB
    s["eta2"] = slice(idx, idx + nA * nA); idx += nA * nA
    s["kappa1"] = idx; idx += 1
    s["kappa2"] = idx; idx += 1
    return s, idx


def solve_strategically_robust_bimatrix_game_path(
    U1, U2, epsilon_values, num_repeats, path_solver, verbose=False
):
    if U2.shape == U1.shape:
        U2 = U2.T

    nA = U1.shape[0]
    nB = U2.shape[0]

    dist_A = np.ones((nB, nB)) - np.eye(nB)
    dist_B = np.ones((nA, nA)) - np.eye(nA)

    s, n_vars = _build_index_map(nA, nB)
    nnz = n_vars * n_vars

    INF = 1e20
    lb = np.full(n_vars, -INF, dtype=np.float64)
    ub = np.full(n_vars, INF, dtype=np.float64)

    lb[s["p1"]] = 0.0
    lb[s["p2"]] = 0.0
    lb[s["lambda1"]] = 0.0
    lb[s["lambda2"]] = 0.0
    lb[s["eta1"]] = 0.0
    lb[s["eta2"]] = 0.0

    solutions_p = []
    utilities_sr = []
    utilities_nominal = []

    total_other = 2 + nB + nA + 2 + nB * nB + nA * nA

    def compute_f(z):
        p1 = z[s["p1"]]
        p2 = z[s["p2"]]
        lambda1 = z[s["lambda1"]]
        lambda2 = z[s["lambda2"]]
        xi1 = z[s["xi1"]]
        xi2 = z[s["xi2"]]
        eta1 = z[s["eta1"]].reshape(nB, nB)
        eta2 = z[s["eta2"]].reshape(nA, nA)
        kappa1 = z[s["kappa1"]]
        kappa2 = z[s["kappa2"]]

        f = np.zeros_like(z)

        s1 = eta1.sum(axis=0)
        s2 = eta2.sum(axis=0)

        f[s["p1"]] = -kappa1 - U1 @ s1
        f[s["p2"]] = -kappa2 - U2 @ s2
        f[s["lambda1"]] = epsilon_values[0] - np.sum(eta1 * dist_A)
        f[s["lambda2"]] = epsilon_values[1] - np.sum(eta2 * dist_B)
        f[s["xi1"]] = -p2 + eta1.sum(axis=1)
        f[s["xi2"]] = -p1 + eta2.sum(axis=1)

        p1U1 = p1 @ U1
        p2U2 = p2 @ U2

        f_eta1 = np.empty((nB, nB))
        for i in range(nB):
            f_eta1[i, :] = -xi1[i] + p1U1 + lambda1 * dist_A[i, :]
        f[s["eta1"]] = f_eta1.ravel()

        f_eta2 = np.empty((nA, nA))
        for i in range(nA):
            f_eta2[i, :] = -xi2[i] + p2U2 + lambda2 * dist_B[i, :]
        f[s["eta2"]] = f_eta2.ravel()

        f[s["kappa1"]] = 1.0 - np.sum(p1)
        f[s["kappa2"]] = 1.0 - np.sum(p2)

        return f

    def compute_jacobian(z):
        J = np.zeros((n_vars, n_vars), dtype=np.float64)

        p1_start = s["p1"].start
        p2_start = s["p2"].start
        xi1_start = s["xi1"].start
        xi2_start = s["xi2"].start
        eta1_start = s["eta1"].start
        eta2_start = s["eta2"].start

        for i in range(nA):
            row = p1_start + i
            J[row, s["kappa1"]] = -1.0
            for k in range(nB):
                for l in range(nB):
                    J[row, eta1_start + k * nB + l] = -U1[i, l]

        for j in range(nB):
            row = p2_start + j
            J[row, s["kappa2"]] = -1.0
            for k in range(nA):
                for l in range(nA):
                    J[row, eta2_start + k * nA + l] = -U2[j, l]

        row = s["lambda1"]
        for i in range(nB):
            for j in range(nB):
                J[row, eta1_start + i * nB + j] = -dist_A[i, j]

        row = s["lambda2"]
        for i in range(nA):
            for j in range(nA):
                J[row, eta2_start + i * nA + j] = -dist_B[i, j]

        for j in range(nB):
            row = xi1_start + j
            J[row, p2_start + j] = -1.0
            for k in range(nB):
                J[row, eta1_start + j * nB + k] = 1.0

        for i in range(nA):
            row = xi2_start + i
            J[row, p1_start + i] = -1.0
            for k in range(nA):
                J[row, eta2_start + i * nA + k] = 1.0

        for i in range(nB):
            for j in range(nB):
                row = eta1_start + i * nB + j
                J[row, xi1_start + i] = -1.0
                J[row, s["lambda1"]] = dist_A[i, j]
                for k in range(nA):
                    J[row, p1_start + k] = U1[k, j]

        for i in range(nA):
            for j in range(nA):
                row = eta2_start + i * nA + j
                J[row, xi2_start + i] = -1.0
                J[row, s["lambda2"]] = dist_B[i, j]
                for k in range(nB):
                    J[row, p2_start + k] = U2[k, j]

        row = s["kappa1"]
        for i in range(nA):
            J[row, p1_start + i] = -1.0

        row = s["kappa2"]
        for j in range(nB):
            J[row, p2_start + j] = -1.0

        return J

    def func_eval(n, z_ptr, f_ptr):
        z = np.ctypeslib.as_array(z_ptr, shape=(n,))
        f = np.ctypeslib.as_array(f_ptr, shape=(n,))
        f[:] = compute_f(z)
        return 0

    def jac_eval(n, nnz_in, z_ptr, col_start_ptr, col_len_ptr, row_ptr, data_ptr):
        z = np.ctypeslib.as_array(z_ptr, shape=(n,))
        col_start = np.ctypeslib.as_array(col_start_ptr, shape=(n,))
        col_len = np.ctypeslib.as_array(col_len_ptr, shape=(n,))
        row = np.ctypeslib.as_array(row_ptr, shape=(nnz_in,))
        data = np.ctypeslib.as_array(data_ptr, shape=(nnz_in,))

        J = compute_jacobian(z)
        idx = 0
        for col in range(n):
            col_start[col] = idx + 1
            col_len[col] = n
            for r in range(n):
                row[idx] = r + 1
                data[idx] = J[r, col]
                idx += 1
        return 0

    for _ in range(num_repeats):
        z = np.zeros(n_vars, dtype=np.float64)
        f = np.zeros(n_vars, dtype=np.float64)

        starting_points_p = np.random.rand(nA + nB)
        starting_points_other = 100 * np.random.rand(total_other) - 50

        z[s["p1"]] = starting_points_p[:nA]
        z[s["p2"]] = starting_points_p[nA:]

        idx = 0
        z[s["lambda1"]] = starting_points_other[idx]; idx += 1
        z[s["lambda2"]] = starting_points_other[idx]; idx += 1
        z[s["xi1"]] = starting_points_other[idx:idx + nB]; idx += nB
        z[s["xi2"]] = starting_points_other[idx:idx + nA]; idx += nA
        z[s["kappa1"]] = starting_points_other[idx]; idx += 1
        z[s["kappa2"]] = starting_points_other[idx]; idx += 1
        z[s["eta1"]] = starting_points_other[idx:idx + nB * nB]; idx += nB * nB
        z[s["eta2"]] = starting_points_other[idx:idx + nA * nA]

        status = path_solver.solve(n_vars, nnz, z, f, lb, ub, func_eval, jac_eval)

        if verbose:
            print(f"PATH status: {status}")

        if status in (1, 2):
            p1 = z[s["p1"]].copy()
            p2 = z[s["p2"]].copy()
            xi1 = z[s["xi1"]].copy()
            xi2 = z[s["xi2"]].copy()
            lambda1 = float(z[s["lambda1"]])
            lambda2 = float(z[s["lambda2"]])

            sol = {"p1": np.round(p1, 4).tolist(), "p2": np.round(p2, 4).tolist()}
            if sol not in solutions_p:
                solutions_p.append(sol)
                utility1_sr = np.sum(p2 * xi1) - lambda1 * epsilon_values[0]
                utility2_sr = np.sum(p1 * xi2) - lambda2 * epsilon_values[1]
                utilities_sr.append([utility1_sr, utility2_sr])

                utility1_nominal = float(p1 @ U1 @ p2)
                utility2_nominal = float(p2 @ U2 @ p1)
                utilities_nominal.append([utility1_nominal, utility2_nominal])

    return solutions_p, utilities_sr, utilities_nominal

class SRQAgent:
    """
    Implements Algorithm 3: Strategically Robust Q-Learning (SRQ).
    
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
        # Structure: Dictionary mapping state -> Tensor of shape (num_actions, ..., num_actions, num_agents)
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
