import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow loading path_solver.py from discrete_action_space/.
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from path_solver import PathSolverWrapper, solve_strategically_robust_bimatrix_game_path


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, joint_actions, joint_rewards, next_state, done):
        self.buffer.append((state, joint_actions, joint_rewards, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DuelingJointQNetwork(nn.Module):
    """
    Dueling network over joint actions for 2-player games.
    Output shape: [batch, num_actions, num_actions, num_agents].
    """

    def __init__(self, obs_dim, num_actions, num_agents):
        super().__init__()
        if num_agents != 2:
            raise ValueError("This network currently supports only num_agents == 2.")

        self.num_actions = num_actions
        self.num_agents = num_agents
        self.joint_action_count = num_actions ** num_agents

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(128, num_agents)
        self.adv_head = nn.Linear(128, self.joint_action_count * num_agents)

    def forward(self, state):
        features = self.feature(state)
        value = self.value_head(features)  # [B, N]
        advantage = self.adv_head(features).view(
            -1, self.joint_action_count, self.num_agents
        )  # [B, |A_joint|, N]

        q_joint = value.unsqueeze(1) + (
            advantage - advantage.mean(dim=1, keepdim=True)
        )
        return q_joint.view(-1, self.num_actions, self.num_actions, self.num_agents)


class DuelingDoubleDqnSreAgent:
    """
    Dueling Double DQN agent that uses SRE policies for action selection and targets.
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
        sre_num_repeats=20,
    ):
        if num_agents != 2:
            raise ValueError("This implementation currently supports only 2 agents.")

        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.epsilon_robust = epsilon_robust
        self.epsilon_explore = epsilon_explore
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.sre_num_repeats = sre_num_repeats

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.q_net = DuelingJointQNetwork(obs_dim, num_actions, num_agents).to(self.device)
        self.target_net = DuelingJointQNetwork(obs_dim, num_actions, num_agents).to(
            self.device
        )
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.path_solver = PathSolverWrapper(pathwrap_path)

    def _state_to_vector(self, state):
        vector = np.asarray(state, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.obs_dim:
            raise ValueError(
                f"Expected state vector length {self.obs_dim}, got {vector.shape[0]}."
            )
        return vector

    def _normalize_policy(self, policy):
        p = np.asarray(policy, dtype=np.float32)
        p = np.clip(p, 0.0, None)
        s = float(p.sum())
        if s <= 0.0:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        return p / s

    def _uniform_policies(self):
        u = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        return [u.copy(), u.copy()]

    def _solve_sre(self, q_tensor):
        u1 = q_tensor[:, :, 0]
        u2 = q_tensor[:, :, 1]

        try:
            results = solve_strategically_robust_bimatrix_game_path(
                u1,
                u2,
                [self.epsilon_robust, self.epsilon_robust],
                self.sre_num_repeats,
                self.path_solver,
            )
            solutions = results[0]
        except Exception:
            return self._uniform_policies()

        if not solutions:
            return self._uniform_policies()

        best_joint_reward = -float("inf")
        best = None
        for sol in solutions:
            p1 = self._normalize_policy(sol["p1"])
            p2 = self._normalize_policy(sol["p2"])
            r1 = float(p1 @ u1 @ p2)
            r2 = float(p1 @ u2 @ p2)
            if r1 + r2 > best_joint_reward:
                best_joint_reward = r1 + r2
                best = [p1, p2]

        return best if best is not None else self._uniform_policies()

    def _sre_expected_values(self, q_tensor, policies):
        expected = q_tensor
        for policy in policies:
            expected = np.tensordot(policy, expected, axes=([0], [0]))
        return np.asarray(expected, dtype=np.float32)

    def act(self, state):
        if np.random.rand() < self.epsilon_explore:
            return int(np.random.choice(self.num_actions))

        state_vec = self._state_to_vector(state)
        state_t = torch.as_tensor(
            state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_tensor = self.q_net(state_t).squeeze(0).detach().cpu().numpy()

        policies = self._solve_sre(q_tensor)
        my_policy = self._normalize_policy(policies[self.agent_id])
        return int(np.random.choice(self.num_actions, p=my_policy))

    def update(self, state, joint_actions, joint_rewards, next_state, done=False, batch_size=32):
        state_vec = self._state_to_vector(state)
        next_state_vec = self._state_to_vector(next_state)
        actions_arr = np.asarray(joint_actions, dtype=np.int64).reshape(-1)
        rewards_arr = np.asarray(joint_rewards, dtype=np.float32).reshape(-1)

        if actions_arr.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected joint action length {self.num_agents}, got {actions_arr.shape[0]}."
            )
        if rewards_arr.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected joint reward length {self.num_agents}, got {rewards_arr.shape[0]}."
            )

        self.replay_buffer.push(state_vec, actions_arr, rewards_arr, next_state_vec, done)
        self.train_step(batch_size=batch_size)

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states_arr = np.stack([self._state_to_vector(s) for s in states], axis=0)
        next_states_arr = np.stack([self._state_to_vector(s) for s in next_states], axis=0)
        actions_arr = np.stack(actions, axis=0)
        rewards_arr = np.stack(rewards, axis=0)

        states_t = torch.as_tensor(states_arr, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            next_states_arr, dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(actions_arr, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards_arr, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=self.device)

        q_tensor = self.q_net(states_t)  # [B, A1, A2, N]
        batch_idx = torch.arange(states_t.shape[0], device=self.device)
        current_q = q_tensor[batch_idx, actions_t[:, 0], actions_t[:, 1], :]  # [B, N]

        with torch.no_grad():
            # Double-DQN style: choose policy from online net, evaluate with target net.
            next_online = self.q_net(next_states_t).detach().cpu().numpy()
            next_target = self.target_net(next_states_t).detach().cpu().numpy()

            next_values = []
            for i in range(states_t.shape[0]):
                policies = self._solve_sre(next_online[i])
                expected_next = self._sre_expected_values(next_target[i], policies)
                next_values.append(expected_next)

            next_values_t = torch.as_tensor(
                np.asarray(next_values, dtype=np.float32), device=self.device
            )
            target_q = rewards_t + (1.0 - dones_t.unsqueeze(1)) * self.gamma * next_values_t

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

    def save_checkpoint(self, path):
        payload = {
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
            "sre_num_repeats": self.sre_num_repeats,
        }
        torch.save(payload, path)

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
        if "sre_num_repeats" in checkpoint:
            self.sre_num_repeats = checkpoint["sre_num_repeats"]

    def close(self):
        if hasattr(self, "path_solver") and self.path_solver is not None:
            self.path_solver.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def flatten_gridworld_obs(obs):
    return np.asarray([coord for pos in obs for coord in pos], dtype=np.float32)


def train_gridworld_dueling_double_dqn_sre(
    n_episodes=1000,
    p_env=0.8,
    pathwrap_path=None,
    batch_size=32,
    target_update=10,
    checkpoint_interval=100,
    checkpoint_dir="checkpoints_dueling_sre_ddqn",
    use_gpu=True,
):
    from GridWorld import GridWorldEnv

    env = GridWorldEnv(p=p_env)
    if pathwrap_path is None:
        pathwrap_path = str((_THIS_DIR / "pathwrap.so").resolve())

    num_agents = 2
    num_actions = len(env.action_space)
    obs_dim = len(flatten_gridworld_obs(env.reset()))

    agents = [
        DuelingDoubleDqnSreAgent(
            agent_id=0,
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_actions=num_actions,
            pathwrap_path=pathwrap_path,
            use_gpu=use_gpu,
        ),
        DuelingDoubleDqnSreAgent(
            agent_id=1,
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_actions=num_actions,
            pathwrap_path=pathwrap_path,
            use_gpu=use_gpu,
        ),
    ]

    rewards_history = [[], []]
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = [0.0, 0.0]

        while not done:
            actions = [agents[0].act(state), agents[1].act(state)]
            next_state, rewards, done, _ = env.step(actions)

            for i in range(num_agents):
                agents[i].update(
                    state=flatten_gridworld_obs(state),
                    joint_actions=actions,
                    joint_rewards=rewards,
                    next_state=flatten_gridworld_obs(next_state),
                    done=done,
                    batch_size=batch_size,
                )
                ep_reward[i] += float(rewards[i])

            state = next_state

        for i in range(num_agents):
            rewards_history[i].append(ep_reward[i])
            agents[i].decay_parameters()

        if episode % target_update == 0:
            for agent in agents:
                agent.update_target_network()

        if episode % checkpoint_interval == 0:
            for i, agent in enumerate(agents):
                agent.save_checkpoint(checkpoint_path / f"dueling_sre_agent{i}_ep{episode}.pt")

    for i, agent in enumerate(agents):
        agent.save_checkpoint(checkpoint_path / f"dueling_sre_agent{i}_final.pt")

    return rewards_history, agents
