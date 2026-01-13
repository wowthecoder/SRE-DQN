import numpy as np
import random

class GridWorldEnv:
    """
    A Multi-Agent Grid-World Environment based on Section 3.2 of 
    'Strategically Robust Q-Learning'[cite: 448].
    
    Attributes:
        grid_size (int): Dimensions of the grid (default 3x3)[cite: 449].
        p (float): Probability of successful action transition. 
                   (1-p)/3 probability of slipping to other directions.
    """
    def __init__(self, grid_size=3, p=1.0, max_steps=500):
        self.grid_size = grid_size
        self.p = p
        self.max_steps = max_steps
        self.current_step = 0
        
        # Actions: 0: Up, 1: Right, 2: Down, 3: Left [cite: 450]
        self.action_space = [0, 1, 2, 3]
        self.n_agents = 2
        
        # Define Start and Goal States (coordinates are (row, col))
        # Agent 1: Start Bottom Left (2, 0) -> Goal Top Right (0, 2) [cite: 454]
        # Agent 2: Start Bottom Right (2, 2) -> Goal Top Left (0, 0) [cite: 456]
        # Note: Adjusting for dynamic grid size, assuming corners.
        self.start_positions = [
            (self.grid_size - 1, 0),                  # Agent 1 Start
            (self.grid_size - 1, self.grid_size - 1)  # Agent 2 Start
        ]
        self.goal_positions = [
            (0, self.grid_size - 1),                  # Agent 1 Goal
            (0, 0)                                    # Agent 2 Goal
        ]
        
        self.agent_positions = list(self.start_positions)
        self.agents_finished = [False, False]

    def reset(self):
        """Resets the environment to start conditions."""
        self.agent_positions = list(self.start_positions)
        self.agents_finished = [False, False]
        self.current_step = 0
        return self._get_obs()

    def step(self, actions):
        """
        Executes a step in the environment.
        
        Args:
            actions (list): A list of two integers [action_agent_1, action_agent_2].
            
        Returns:
            obs, rewards, done, info
        """
        if self.current_step >= self.max_steps:
            return self._get_obs(), [0, 0], True, {"reason": "max_steps"}
        
        self.current_step += 1
        proposed_positions = [None, None]
        rewards = [0, 0]
        
        # 1. Calculate Proposed Movements (Stochasticity Handling)
        for i in range(self.n_agents):
            if self.agents_finished[i]:
                # If agent is at goal, it stays there and takes no actions [cite: 473]
                proposed_positions[i] = self.agent_positions[i]
                continue

            # Apply transition probability p [cite: 492]
            chosen_action = actions[i]
            if np.random.rand() > self.p:
                # Select one of the other 3 actions uniformly
                other_actions = [a for a in self.action_space if a != chosen_action]
                actual_action = np.random.choice(other_actions)
            else:
                actual_action = chosen_action
                
            # Determine target coordinate
            current_r, current_c = self.agent_positions[i]
            delta_r, delta_c = self._action_to_delta(actual_action)
            new_r, new_c = current_r + delta_r, current_c + delta_c
            
            # Boundary Check: If out of bounds, stay in current state [cite: 451]
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                proposed_positions[i] = (new_r, new_c)
            else:
                proposed_positions[i] = (current_r, current_c)

        # 2. Check for Collisions
        # "If both agents attempt to move to the same state (collision)... 
        # each agent is moved back to its previous position." 
        if proposed_positions[0] == proposed_positions[1]:
            collision = True
            # Both bounce back
            final_positions = list(self.agent_positions) 
            rewards = [-100, -100] # Penalty for collision 
        else:
            collision = False
            final_positions = proposed_positions
        
        # 3. Assign Standard Rewards and Update State
        for i in range(self.n_agents):
            if self.agents_finished[i]:
                # Already finished agents get 0 reward this step (assumed)
                continue
                
            if collision:
                # Reward already handled (-100)
                pass 
            else:
                # Check if reached goal
                if final_positions[i] == self.goal_positions[i]:
                    rewards[i] = 100 # Transition to goal 
                    self.agents_finished[i] = True
                else:
                    rewards[i] = -1  # Transition to non-goal state 
            
            # Update position
            self.agent_positions[i] = final_positions[i]

        # 4. Check Termination
        # "An episode terminates when both agents reach their goal state" 
        done = all(self.agents_finished)
        
        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        """
        Returns the current state.
        Text implies agents observe state of themselves and opponent[cite: 453].
        """
        return list(self.agent_positions)

    def _action_to_delta(self, action):
        """Maps action index to (row, col) delta."""
        # 0: Up, 1: Right, 2: Down, 3: Left
        if action == 0: return (-1, 0)
        if action == 1: return (0, 1)
        if action == 2: return (1, 0)
        if action == 3: return (0, -1)
        return (0, 0)

    def render(self):
        """Visualizes the grid for debugging."""
        grid = np.full((self.grid_size, self.grid_size), '_')
        
        # Mark goals
        g1 = self.goal_positions[0]
        g2 = self.goal_positions[1]
        grid[g1[0], g1[1]] = 'G' # Goal 1 (shared visual for simplicity or use G1)
        grid[g2[0], g2[1]] = 'g' # Goal 2
        
        # Mark agents
        a1 = self.agent_positions[0]
        a2 = self.agent_positions[1]
        
        if a1 == a2:
            grid[a1[0], a1[1]] = 'X' # Collision/Overlay
        else:
            grid[a1[0], a1[1]] = '1'
            grid[a2[0], a2[1]] = '2'
            
        print("\n".join([" ".join(row) for row in grid]))
        print(f"P1: {a1}, P2: {a2}")
        print("-" * 10)

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize environment with p=0.8 (uncertainty) as described in Results 3.3
    env = RobustGridWorldEnv(grid_size=3, p=0.8)
    obs = env.reset()
    env.render()
    
    # Example Step: Agent 1 moves Up (0), Agent 2 moves Left (3)
    # Note: A2 starts at (2,2), Left is (2,1). A1 starts (2,0), Up is (1,0).
    obs, rewards, done, info = env.step([0, 3]) 
    
    print(f"Rewards: {rewards}")
    env.render()