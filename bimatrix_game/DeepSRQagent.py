import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from path_solver import PathSolverWrapper, solve_strategically_robust_bimatrix_game_path

# --- 1. Q-Network Architecture ---
class SRQNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, num_agents):
        """
        Approximates the Joint Q-function Q(s, a1, a2, ...) for ALL agents.
        Output is flattened, then reshaped during usage.
        """
        super(SRQNetwork, self).__init__()
        self.num_actions = num_actions
        self.num_agents = num_agents
        
        # Calculate output size: (Num Actions ^ Num Agents) * Num Agents
        # e.g., 2 Agents, 4 Actions -> 4*4 * 2 = 32 outputs
        self.output_dim = (num_actions ** num_agents) * num_agents
        
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        x = self.fc(state)
        # Reshape to [Batch, A1, A2, ..., An, Agent_Index]
        # Assuming 2 agents for the bimatrix solver
        shape = [-1, self.num_actions, self.num_actions, self.num_agents]
        return x.view(*shape)

# --- 2. Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, actions, rewards, next_state, done):
        self.buffer.append((state, actions, rewards, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

# --- 3. Deep SRQ Agent ---
class DeepSRQAgent:
    def __init__(self, agent_id, obs_dim, num_agents, num_actions, 
                 pathwrap_path="pathwrap.so",
                 epsilon_robust=1.0, epsilon_explore=1.0, 
                 lr=1e-3, gamma=0.9, decay_rate=0.999, buffer_size=10000,
                 use_gpu=True):
        
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.epsilon_robust = epsilon_robust
        self.epsilon_explore = epsilon_explore
        self.gamma = gamma
        self.decay_rate = decay_rate
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Neural Networks
        self.q_net = SRQNetwork(obs_dim, num_actions, num_agents).to(self.device)
        self.target_net = SRQNetwork(obs_dim, num_actions, num_agents).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size)

        # --- PATH Solver Setup (Same as tabular) ---
        print(f"Loading PATH wrapper from {pathwrap_path}...")
        self.path_solver = PathSolverWrapper(pathwrap_path)
        print("PATH Solver Ready.")

    def solve_sre_from_tensor(self, q_tensor_np):
        """
        Extracts matrices from a numpy Q-tensor and solves for SRE using PATH.
        Expects q_tensor_np shape: (A1, A2, Num_Agents)
        """
        # Extract Payoff Matrices for Agent 0 and Agent 1
        U1 = q_tensor_np[:, :, 0]
        U2 = q_tensor_np[:, :, 1]

        try:
            results = solve_strategically_robust_bimatrix_game_path(
                U1, U2,
                [self.epsilon_robust, self.epsilon_robust],
                3,
                self.path_solver
            )
            solutions = results[0]
        except Exception:
            # Fallback to uniform if solver fails
            uniform = np.ones(self.num_actions) / self.num_actions
            return [uniform, uniform]

        if len(solutions) == 0:
            uniform = np.ones(self.num_actions) / self.num_actions
            return [uniform, uniform]

        # Equilibrium Selection: Highest Joint Reward
        best_sol = None
        best_joint_reward = -float('inf')

        for sol in solutions:
            p1 = np.array(sol["p1"])
            p2 = np.array(sol["p2"])
            
            r1 = p1 @ U1 @ p2
            r2 = p1 @ U2 @ p2
            
            if (r1 + r2) > best_joint_reward:
                best_joint_reward = (r1 + r2)
                best_sol = [p1, p2]
        
        return best_sol

    def act(self, state):
        """
        Epsilon-greedy action selection based on SRE of Q-network predictions.
        """
        if np.random.rand() < self.epsilon_explore:
            return np.random.choice(self.num_actions)

        # 1. Forward pass to get Q-values
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_tensor = self.q_net(state_t) # Shape: (1, A, A, 2)
        
        # 2. Convert to numpy for Julia
        q_np = q_tensor.cpu().numpy()[0] 
        
        # 3. Solve SRE
        policies = self.solve_sre_from_tensor(q_np)
        my_policy = policies[self.agent_id]
        
        # 4. Sample action
        return np.random.choice(self.num_actions, p=my_policy)

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        # 1. Sample Batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device) # Shape: (Batch, 2) -> [a1, a2]
        rewards_t = torch.FloatTensor(rewards).to(self.device) # Shape: (Batch, 2)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 2. Calculate Current Q(s, a1, a2)
        # q_preds shape: (Batch, A, A, 2)
        q_preds = self.q_net(states_t)
        
        # We need to gather the specific Q-values for the actions taken.
        # This indexing is tricky. We want q_preds[b, a1[b], a2[b], :]
        # A simple way for 2 agents:
        batch_indices = torch.arange(batch_size).long().to(self.device)
        a1 = actions_t[:, 0]
        a2 = actions_t[:, 1]
        
        # Shape: (Batch, 2) -> The Q-values for both agents at the chosen joint action
        current_q = q_preds[batch_indices, a1, a2, :] 

        # 3. Calculate Target Values (The Computationally Expensive Part)
        with torch.no_grad():
            # Get full Q-matrices for next states
            next_q_preds = self.target_net(next_states_t) # Shape: (Batch, A, A, 2)
            next_q_np = next_q_preds.cpu().numpy()
            
            srq_values = []
            
            # Loop through batch to solve SRE for each transition
            # Note: This loop is slow because we call Julia inside. 
            # In production, you would want to vectorize the Julia call or use multiprocess.
            for i in range(batch_size):
                if dones[i]:
                    srq_values.append(np.zeros(self.num_agents))
                    continue
                
                # Solve SRE for next state
                q_matrix = next_q_np[i] # (A, A, 2)
                policies = self.solve_sre_from_tensor(q_matrix) # [pi_1, pi_2]
                pi1, pi2 = policies
                
                # Calculate Expected Value under Equilibrium: V = pi1 * U * pi2
                # U1 = q_matrix[:,:,0], U2 = q_matrix[:,:,1]
                val_1 = pi1 @ q_matrix[:,:,0] @ pi2
                val_2 = pi1 @ q_matrix[:,:,1] @ pi2
                
                srq_values.append(np.array([val_1, val_2]))
            
            srq_values_t = torch.FloatTensor(np.array(srq_values)).to(self.device)
            
            # Bellman Target
            target_q = rewards_t + (1 - dones_t.unsqueeze(1)) * self.gamma * srq_values_t

        # 4. Update Network
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_parameters(self):
        self.epsilon_robust *= self.decay_rate
        self.epsilon_explore *= self.decay_rate

    def save_checkpoint(self, path):
        checkpoint = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon_robust": self.epsilon_robust,
            "epsilon_explore": self.epsilon_explore,
            "gamma": self.gamma,
            "decay_rate": self.decay_rate,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        if "q_net" in checkpoint:
            self.q_net.load_state_dict(checkpoint["q_net"])
        if "target_net" in checkpoint:
            self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epsilon_robust" in checkpoint:
            self.epsilon_robust = checkpoint["epsilon_robust"]
        if "epsilon_explore" in checkpoint:
            self.epsilon_explore = checkpoint["epsilon_explore"]
        if "gamma" in checkpoint:
            self.gamma = checkpoint["gamma"]
        if "decay_rate" in checkpoint:
            self.decay_rate = checkpoint["decay_rate"]
