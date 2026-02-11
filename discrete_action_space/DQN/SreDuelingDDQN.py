import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dqn_common import BaseDDqnAgent

# Support importing path_solver.py when this module is imported from DQN folder.
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from path_solver import PathSolverWrapper, solve_strategically_robust_bimatrix_game_path


class DuelingJointQNetwork(nn.Module):
    """
    Dueling network over joint actions.
    Output shape for 2-agent games: [batch, A1, A2, num_agents].
    """

    def __init__(self, obs_dim, num_actions, num_agents):
        super().__init__()
        if num_agents != 2:
            raise ValueError(
                "DuelingJointQNetwork currently supports only 2-agent games."
            )

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
        self.advantage_head = nn.Linear(
            128, self.joint_action_count * num_agents
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value_head(features)  # [B, N]
        advantage = self.advantage_head(features).view(
            -1, self.joint_action_count, self.num_agents
        )  # [B, |A_joint|, N]

        q_joint = value.unsqueeze(1) + (
            advantage - advantage.mean(dim=1, keepdim=True)
        )
        return q_joint.view(-1, self.num_actions, self.num_actions, self.num_agents)


class SreDuelingDDqnAgent(BaseDDqnAgent):
    """
    Dueling DDQN variant that computes SRE at every step for Bellman targets.

    This mirrors SRQAgent's target style for 2-player bimatrix games:
    expected next value is computed under SRE policy at s'.
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
        sre_num_repeats=3,
    ):
        if num_agents != 2:
            raise ValueError(
                "SreDuelingDDqnAgent currently supports only num_agents == 2."
            )

        self.sre_num_repeats = sre_num_repeats
        self.path_solver = PathSolverWrapper(pathwrap_path)

        super().__init__(
            agent_id=agent_id,
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_actions=num_actions,
            epsilon_robust=epsilon_robust,
            epsilon_explore=epsilon_explore,
            lr=lr,
            gamma=gamma,
            decay_rate=decay_rate,
            buffer_size=buffer_size,
            use_gpu=use_gpu,
        )

    def _build_network(self):
        return DuelingJointQNetwork(self.obs_dim, self.num_actions, self.num_agents)

    def _uniform_policies(self):
        uniform = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        return [uniform.copy(), uniform.copy()]

    def _normalize_policy(self, p):
        p = np.asarray(p, dtype=np.float32)
        p = np.clip(p, 0.0, None)
        s = float(p.sum())
        if s <= 0.0:
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        return p / s

    def _solve_sre(self, q_tensor):
        # q_tensor shape: [A1, A2, N]
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
            joint_reward = r1 + r2
            if joint_reward > best_joint_reward:
                best_joint_reward = joint_reward
                best = [p1, p2]

        return best if best is not None else self._uniform_policies()

    def _sre_expected_values(self, q_tensor, policies):
        # q_tensor shape: [A1, A2, N]
        expected = q_tensor
        for policy in policies:
            expected = np.tensordot(policy, expected, axes=([0], [0]))
        return np.asarray(expected, dtype=np.float32)

    def _extract_joint_actions_rewards(self, actions, rewards):
        actions_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
        rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(-1)

        if actions_arr.shape[0] != self.num_agents:
            raise ValueError(
                "SreDuelingDDqnAgent expects full joint actions with length "
                f"{self.num_agents}, got shape {actions_arr.shape}."
            )
        if rewards_arr.shape[0] != self.num_agents:
            raise ValueError(
                "SreDuelingDDqnAgent expects full reward vector with length "
                f"{self.num_agents}, got shape {rewards_arr.shape}."
            )
        return actions_arr, rewards_arr

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

    def update(self, state, actions, rewards, next_state, done=False, batch_size=32):
        state_vec = self._state_to_vector(state)
        next_state_vec = self._state_to_vector(next_state)
        joint_actions, reward_vec = self._extract_joint_actions_rewards(actions, rewards)

        self.replay_buffer.push(state_vec, joint_actions, reward_vec, next_state_vec, done)
        self.train_step(batch_size=batch_size)

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        states_arr = np.stack([self._state_to_vector(s) for s in states])
        next_states_arr = np.stack([self._state_to_vector(s) for s in next_states])
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
        batch_idx = torch.arange(batch_size, device=self.device)
        current_q = q_tensor[
            batch_idx, actions_t[:, 0], actions_t[:, 1], :
        ]  # [B, N]

        with torch.no_grad():
            next_online = self.q_net(next_states_t).detach().cpu().numpy()
            next_target = self.target_net(next_states_t).detach().cpu().numpy()

            next_values = []
            for i in range(batch_size):
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

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload.update({"sre_num_repeats": self.sre_num_repeats})
        return payload

    def load_checkpoint(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        self._restore_common_checkpoint(checkpoint)
        if "sre_num_repeats" in checkpoint:
            self.sre_num_repeats = checkpoint["sre_num_repeats"]

    def close(self):
        if getattr(self, "path_solver", None) is not None:
            self.path_solver.close()


# Backward compatibility
DuelingSreDDqnSrqAgent = SreDuelingDDqnAgent
