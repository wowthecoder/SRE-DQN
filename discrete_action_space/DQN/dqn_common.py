import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, state):
        return self.net(state)


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, num_actions)

    def forward(self, state):
        features = self.feature(state)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, rewards, next_state, done):
        self.buffer.append((state, actions, rewards, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, actions, rewards, next_state, done = zip(*batch)
        return state, actions, rewards, next_state, done

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Proportional prioritized replay buffer (Schaul et al. 2015).
    """

    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, state, actions, rewards, next_state, done):
        transition = (state, actions, rewards, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        size = len(self.buffer)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        priorities = self.priorities[:size]
        scaled = priorities ** self.alpha
        scaled_sum = scaled.sum()
        if scaled_sum <= 0:
            probs = np.full(size, 1.0 / size, dtype=np.float32)
        else:
            probs = scaled / scaled_sum

        indices = np.random.choice(size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        weights = (size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        new_priorities = np.abs(td_errors) + self.eps
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, float(new_priorities.max()))

    def __len__(self):
        return len(self.buffer)


class BaseDDqnSrqAgent:
    """
    Common DQN agent logic with SRQAgent-style API compatibility.
    Uses Double DQN bootstrap by default. Subclasses can override network type
    and/or next-q computation.
    """

    def __init__(
        self,
        agent_id,
        obs_dim,
        num_agents,
        num_actions,
        pathwrap_path="pathwrap.so",
        epsilon_robust=1.0,
        epsilon_explore=1.0,
        lr=1e-3,
        gamma=0.9,
        decay_rate=0.999,
        buffer_size=10000,
        use_gpu=True,
    ):
        # pathwrap_path is kept for constructor compatibility.
        _ = pathwrap_path

        self.agent_id = agent_id
        self.obs_dim = obs_dim
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

        self.q_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)

    def _build_network(self) -> nn.Module:
        return QNetwork(self.obs_dim, self.num_actions)

    def _state_to_vector(self, state):
        vector = np.asarray(state, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.obs_dim:
            raise ValueError(
                f"Expected state vector length {self.obs_dim}, got {vector.shape[0]}."
            )
        return vector

    def _extract_agent_action_reward(self, actions, rewards):
        if isinstance(actions, (list, tuple, np.ndarray)):
            action = int(actions[self.agent_id])
        else:
            action = int(actions)

        if isinstance(rewards, (list, tuple, np.ndarray)):
            reward = float(rewards[self.agent_id])
        else:
            reward = float(rewards)

        return action, reward

    def _compute_next_q(self, next_states_t):
        next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
        return self.target_net(next_states_t).gather(1, next_actions).squeeze(1)

    def _prepare_batch_tensors(self, states, actions, rewards, next_states, dones):
        states_arr = np.stack([self._state_to_vector(s) for s in states])
        next_states_arr = np.stack([self._state_to_vector(s) for s in next_states])

        actions_arr = np.asarray(actions)
        rewards_arr = np.asarray(rewards)
        if actions_arr.ndim > 1:
            actions_arr = actions_arr[:, self.agent_id]
        if rewards_arr.ndim > 1:
            rewards_arr = rewards_arr[:, self.agent_id]

        states_t = torch.as_tensor(states_arr, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions_arr, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards_arr, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            next_states_arr, dtype=torch.float32, device=self.device
        )
        dones_t = torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=self.device)
        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def act(self, state):
        if np.random.rand() < self.epsilon_explore:
            return np.random.choice(self.num_actions)

        state_vec = self._state_to_vector(state)
        state_t = torch.as_tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())

        return action

    def update(self, state, actions, rewards, next_state, done=False, batch_size=32):
        state_vec = self._state_to_vector(state)
        next_state_vec = self._state_to_vector(next_state)
        action, reward = self._extract_agent_action_reward(actions, rewards)

        self.replay_buffer.push(state_vec, action, reward, next_state_vec, done)
        self.train_step(batch_size=batch_size)

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        states_t, actions_t, rewards_t, next_states_t, dones_t = self._prepare_batch_tensors(
            states, actions, rewards, next_states, dones
        )

        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self._compute_next_q(next_states_t)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_parameters(self):
        self.epsilon_robust *= self.decay_rate
        self.epsilon_explore *= self.decay_rate

    def _checkpoint_payload(self):
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon_robust": self.epsilon_robust,
            "epsilon_explore": self.epsilon_explore,
            "gamma": self.gamma,
            "decay_rate": self.decay_rate,
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "agent_id": self.agent_id,
        }

    def _restore_common_checkpoint(self, checkpoint):
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

    def save_checkpoint(self, path):
        torch.save(self._checkpoint_payload(), path)

    def load_checkpoint(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        self._restore_common_checkpoint(checkpoint)

    # SRQAgent naming compatibility
    def save_q_table(self, path):
        self.save_checkpoint(path)

    def load_q_table(self, path, map_location=None):
        self.load_checkpoint(path, map_location=map_location)
