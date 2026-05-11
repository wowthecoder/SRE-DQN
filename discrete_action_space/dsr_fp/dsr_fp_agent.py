"""Deep Strategically Robust Fictitious Play (DSR-FP) agent.

Two networks per agent (parameter-shared in self-play, like Deep SRQ):
  - Q-network  Q^θ(s) → [A_1, ..., A_N, N]  (joint payoff tensor)
  - Policy net π̄^φ(s) → list of N softmax distributions over A_i

The Q-network uses the same architecture as Deep SRQ (make_q_network from
dueling_double_dqn_sre.py).  The policy net is SharedTrunkPolicyNet from
sr_adidas/networks.py.

NFSP averaging: BR actions sampled at rollout time are pushed to a reservoir
buffer.  A supervised cross-entropy loss trains π̄^φ to imitate the time-average
of past SR best-responses.  This is the fictitious-play averaging step.

No stage-game solver is called at training time.  The SR best-response is
computed by Frank-Wolfe on the TV-DRO objective (see robust_br.py).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim  # noqa: F401 used via optim.Adam

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
_SRADS_DIR = _DISCRETE_DIR / "sr_adidas"
for _path in (str(_THIS_DIR), str(_DISCRETE_DIR), str(_SRADS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dueling_double_dqn_sre import ReplayBuffer, make_q_network
from networks import SharedTrunkPolicyNet  # sr_adidas/networks.py

try:
    from .reservoir_buffer import ReservoirBuffer
    from .robust_br import sr_br_policies_batch, sr_value_batch
except ImportError:
    from reservoir_buffer import ReservoirBuffer  # type: ignore[no-redef]
    from robust_br import sr_br_policies_batch, sr_value_batch  # type: ignore[no-redef]


class DsrFpAgent:
    """DSR-FP agent: solver-free SRE approximation via fictitious play.

    Public API is compatible with DuelingDoubleDqnSreAgent so the same
    training harnesses can drive both agents.
    """

    def __init__(
        self,
        agent_id,
        obs_dim,
        num_agents,
        num_actions,
        *,
        epsilon_robust=0.5,
        epsilon_explore=1.0,
        epsilon_robust_initial=None,
        epsilon_schedule="linear",
        decay_rate=0.999,
        action_epsilon_start=1.0,
        action_epsilon_end=0.05,
        action_epsilon_decay_fraction=0.5,
        eta=0.1,
        lr_q=3e-4,
        lr_pi=3e-4,
        gamma=0.9,
        buffer_size=10_000,
        reservoir_size=100_000,
        learning_starts=1_000,
        grad_clip_norm=10.0,
        target_update_steps=250,
        br_n_iter=40,
        network_type="shared_trunk_separate_heads",
        use_gpu=True,
        train_every=1,
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.q_tensor_shape = tuple([num_actions] * num_agents + [num_agents])

        self.epsilon_robust = float(epsilon_robust)
        self.epsilon_explore = float(epsilon_explore)
        self.epsilon_robust_initial = float(
            epsilon_robust if epsilon_robust_initial is None else epsilon_robust_initial
        )
        self.epsilon_schedule = str(epsilon_schedule)
        self.decay_rate = float(decay_rate)
        self.action_epsilon_start = float(action_epsilon_start)
        self.action_epsilon_end = float(action_epsilon_end)
        self.action_epsilon_decay_fraction = float(action_epsilon_decay_fraction)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.learning_starts = int(learning_starts)
        self.grad_clip_norm = grad_clip_norm
        self.br_n_iter = int(br_n_iter)
        self.network_type = network_type
        self.train_every = max(1, int(train_every))
        self._update_calls = 0
        self.gradient_steps = 0
        self._target_update_steps = int(target_update_steps)  # stored for checkpoint only

        self.device = (
            torch.device("cuda")
            if use_gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.q_net = make_q_network(obs_dim, num_actions, num_agents, network_type).to(self.device)
        self.q_target = make_q_network(obs_dim, num_actions, num_agents, network_type).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.pi_net = SharedTrunkPolicyNet(obs_dim, num_actions, num_agents).to(self.device)
        self.pi_target = SharedTrunkPolicyNet(obs_dim, num_actions, num_agents).to(self.device)
        self.pi_target.load_state_dict(self.pi_net.state_dict())

        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q)
        self.pi_optimizer = optim.Adam(self.pi_net.parameters(), lr=lr_pi)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.reservoir = ReservoirBuffer(reservoir_size)

        self.update_times: list[float] = []

    # ------------------------------------------------------------------
    # Epsilon scheduling
    # ------------------------------------------------------------------

    def decay_parameters(self, episode_idx, n_episodes):
        if self.epsilon_schedule == "constant":
            self.epsilon_robust = self.epsilon_robust_initial
        elif self.epsilon_schedule == "linear":
            if n_episodes <= 1:
                self.epsilon_robust = 0.0
            else:
                fraction = min(max(int(episode_idx), 0) / float(n_episodes - 1), 1.0)
                self.epsilon_robust = self.epsilon_robust_initial * (1.0 - fraction)
        elif self.epsilon_schedule == "exponential":
            self.epsilon_robust = self.epsilon_robust_initial * (
                self.decay_rate ** int(episode_idx)
            )
        else:
            raise ValueError(f"Unsupported epsilon schedule: {self.epsilon_schedule}")

        decay_episodes = max(1, int(n_episodes * self.action_epsilon_decay_fraction))
        step = min(int(episode_idx), decay_episodes - 1)
        fraction = min(max(step, 0) / float(max(decay_episodes - 1, 1)), 1.0)
        self.epsilon_explore = self.action_epsilon_start + fraction * (
            self.action_epsilon_end - self.action_epsilon_start
        )

    # ------------------------------------------------------------------
    # Action selection (NFSP: explore / anticipatory-BR / avg-policy)
    # ------------------------------------------------------------------

    def act(self, state, agent_id=None):
        """Single-env action selection (used in the LBF serial loop)."""
        if agent_id is None:
            agent_id = self.agent_id

        if np.random.rand() < self.epsilon_explore:
            return int(np.random.randint(self.num_actions))

        state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
        state_t = torch.as_tensor(state_vec, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_batch = self.q_net(state_t).cpu().numpy()   # [1, A1, ..., AN, N]
            pi_lists = [p.cpu().numpy() for p in self.pi_target(state_t)]  # N x [1, Ai]
            pi_avg_batch = [p for p in pi_lists]          # N x [1, Ai]

        use_br = np.random.rand() < self.eta and len(self.replay_buffer) >= self.learning_starts
        if use_br:
            br_policies, _ = sr_br_policies_batch(
                q_batch, pi_avg_batch, self.epsilon_robust,
                self.num_agents, self.num_actions, self.br_n_iter,
            )
            p_i = br_policies[0][agent_id]
            action = int(np.random.choice(self.num_actions, p=p_i))
            self.reservoir.push(state_vec, agent_id, action)
        else:
            p_i = np.asarray(pi_avg_batch[agent_id][0], dtype=np.float64)
            p_i = np.clip(p_i, 0.0, None)
            p_i /= p_i.sum() if p_i.sum() > 0 else 1.0
            action = int(np.random.choice(self.num_actions, p=p_i))

        return action

    def act_batch(self, obs_batch):
        """Batched action selection for vectorized environments.

        Args:
            obs_batch: float array [B, obs_dim].

        Returns:
            actions: int array [B, num_agents].
            br_obs_list: list of (obs, agent_id, action) for reservoir pushes
                (only from η-branch steps).
        """
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        B = obs_batch.shape[0]

        with torch.no_grad():
            q_batch = self.q_net(obs_t).cpu().numpy()               # [B, A1,...,AN, N]
            pi_lists = [p.cpu().numpy() for p in self.pi_target(obs_t)]  # N x [B, Ai]

        explore_mask = np.random.rand(B, self.num_agents) < self.epsilon_explore
        eta_mask = (
            (np.random.rand(B, self.num_agents) < self.eta)
            & ~explore_mask
            & (len(self.replay_buffer) >= self.learning_starts)
        )

        # Compute SR-BR only for envs where at least one agent uses it
        needs_br = np.any(eta_mask, axis=1)  # [B]
        br_policies_all = [None] * B
        if np.any(needs_br):
            br_q = q_batch[needs_br]
            br_pi = [pi_lists[i][needs_br] for i in range(self.num_agents)]
            br_pols, _ = sr_br_policies_batch(
                br_q, br_pi, self.epsilon_robust,
                self.num_agents, self.num_actions, self.br_n_iter,
            )
            br_idx = 0
            for b in range(B):
                if needs_br[b]:
                    br_policies_all[b] = br_pols[br_idx]
                    br_idx += 1

        actions = np.zeros((B, self.num_agents), dtype=np.int64)
        reservoir_pushes = []

        for b in range(B):
            for i in range(self.num_agents):
                if explore_mask[b, i]:
                    actions[b, i] = np.random.randint(self.num_actions)
                elif eta_mask[b, i] and br_policies_all[b] is not None:
                    p_i = br_policies_all[b][i]
                    a = int(np.random.choice(self.num_actions, p=p_i))
                    actions[b, i] = a
                    reservoir_pushes.append((obs_batch[b], i, a))
                else:
                    p_i = np.asarray(pi_lists[i][b], dtype=np.float64)
                    p_i = np.clip(p_i, 0.0, None)
                    s = p_i.sum()
                    p_i = p_i / s if s > 0 else np.full(self.num_actions, 1.0 / self.num_actions)
                    actions[b, i] = int(np.random.choice(self.num_actions, p=p_i))

        return actions, reservoir_pushes

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def push_transition(self, state, joint_actions, joint_rewards, next_state, done):
        self._update_calls += 1
        self.replay_buffer.push(
            np.asarray(state, dtype=np.float32),
            np.asarray(joint_actions, dtype=np.int64),
            np.asarray(joint_rewards, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            done,
        )

    def push_reservoir(self, obs, agent_id, action):
        self.reservoir.push(obs, agent_id, action)

    # ------------------------------------------------------------------
    # Q-network training (Bellman with SR value target)
    # ------------------------------------------------------------------

    def train_q_step(self, batch_size=32):
        if len(self.replay_buffer) < max(batch_size, self.learning_starts):
            return None

        t0 = time.perf_counter()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states_arr = np.stack(states, axis=0).astype(np.float32)
        next_states_arr = np.stack(next_states, axis=0).astype(np.float32)
        actions_arr = np.stack(actions, axis=0)    # [B, N]
        rewards_arr = np.stack(rewards, axis=0)    # [B, N]
        dones_arr = np.asarray(dones, dtype=np.float32)

        states_t = torch.as_tensor(states_arr, device=self.device)
        next_t = torch.as_tensor(next_states_arr, device=self.device)
        actions_t = torch.as_tensor(actions_arr, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards_arr, device=self.device)
        dones_t = torch.as_tensor(dones_arr, device=self.device)
        if dones_t.ndim == 1:
            dones_t = dones_t.unsqueeze(1)

        # Current Q values at the taken joint actions
        q_vals = self.q_net(states_t)  # [B, A1,...,AN, N]
        batch_idx = torch.arange(batch_size, device=self.device)
        action_indices = [actions_t[:, i] for i in range(self.num_agents)]
        current_q = q_vals[(batch_idx, *action_indices, slice(None))]  # [B, N]

        # SR value targets using target Q-net and target policy net
        with torch.no_grad():
            q_next = self.q_target(next_t).cpu().numpy()                   # [B, A1,...,AN, N]
            pi_next = [p.cpu().numpy() for p in self.pi_target(next_t)]   # N x [B, Ai]

        next_values = sr_value_batch(
            q_next, pi_next, self.epsilon_robust,
            self.num_agents, self.num_actions, self.br_n_iter,
        )  # [B, N]

        next_values_t = torch.as_tensor(next_values, dtype=torch.float32, device=self.device)
        target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_values_t

        loss = self.loss_fn(current_q, target_q.detach())
        self.q_optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.q_optimizer.step()
        self.gradient_steps += 1
        self.update_times.append(time.perf_counter() - t0)
        return float(loss.item())

    # ------------------------------------------------------------------
    # Policy network training (supervised FP averaging on reservoir)
    # ------------------------------------------------------------------

    def train_pi_step(self, batch_size=32):
        if len(self.reservoir) < max(batch_size, self.learning_starts):
            return None

        obs_batch, agent_ids, actions = self.reservoir.sample(batch_size)
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)

        pi_lists = self.pi_net(obs_t)  # N x [B, num_actions]
        pi_stack = torch.stack(pi_lists, dim=1)  # [B, N, num_actions]

        # For each sample, pick the policy head of the stored agent_id
        agent_ids_t = torch.as_tensor(agent_ids, dtype=torch.long, device=self.device)
        # [B, num_actions]: select the row for the relevant agent
        selected_pi = pi_stack[
            torch.arange(batch_size, device=self.device),
            agent_ids_t,
        ]  # [B, num_actions]

        log_probs = torch.log(selected_pi.clamp(min=1e-8))
        loss = -log_probs[torch.arange(batch_size, device=self.device), actions_t].mean()

        self.pi_optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.pi_net.parameters(), self.grad_clip_norm)
        self.pi_optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    # Target network updates
    # ------------------------------------------------------------------

    def update_target_networks(self):
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.pi_target.load_state_dict(self.pi_net.state_dict())

    def soft_update_target_networks(self, tau=0.005):
        for target_p, p in zip(self.q_target.parameters(), self.q_net.parameters()):
            target_p.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        for target_p, p in zip(self.pi_target.parameters(), self.pi_net.parameters()):
            target_p.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    # ------------------------------------------------------------------
    # Compatibility shim: update() mirrors DuelingDoubleDqnSreAgent.update()
    # ------------------------------------------------------------------

    def update(self, state, joint_actions, joint_rewards, next_state, done=False, batch_size=32):
        """Serial-loop convenience method (used in LBF runner)."""
        state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
        next_vec = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.push_transition(state_vec, joint_actions, joint_rewards, next_vec, done)
        if self._update_calls % self.train_every != 0:
            return None
        q_loss = self.train_q_step(batch_size=batch_size)
        self.train_pi_step(batch_size=batch_size)
        return q_loss

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, include_replay_buffer=False):
        payload = {
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "pi_net": self.pi_net.state_dict(),
            "pi_target": self.pi_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "epsilon_robust": self.epsilon_robust,
            "epsilon_explore": self.epsilon_explore,
            "eta": self.eta,
            "gamma": self.gamma,
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "num_agents": self.num_agents,
            "agent_id": self.agent_id,
            "network_type": self.network_type,
            "learning_starts": self.learning_starts,
            "grad_clip_norm": self.grad_clip_norm,
            "br_n_iter": self.br_n_iter,
            "train_every": self.train_every,
            "gradient_steps": self.gradient_steps,
            "_update_calls": self._update_calls,
            "buffer_size": self.replay_buffer.buffer.maxlen,
            "reservoir_size": self.reservoir.capacity,
            "update_times": list(self.update_times),
        }
        if include_replay_buffer:
            payload["replay_buffer"] = list(self.replay_buffer.buffer)
        torch.save(payload, path)

    def load_checkpoint(self, path, map_location=None):
        cp = torch.load(path, map_location=map_location or self.device)
        for key, net in [
            ("q_net", self.q_net),
            ("q_target", self.q_target),
            ("pi_net", self.pi_net),
            ("pi_target", self.pi_target),
        ]:
            if key in cp:
                net.load_state_dict(cp[key])
        for key, opt in [("q_optimizer", self.q_optimizer), ("pi_optimizer", self.pi_optimizer)]:
            if key in cp:
                opt.load_state_dict(cp[key])
        for attr in [
            "epsilon_robust", "epsilon_explore", "eta", "gamma",
            "learning_starts", "grad_clip_norm", "br_n_iter",
            "train_every", "gradient_steps", "_update_calls",
        ]:
            if attr in cp:
                setattr(self, attr, cp[attr])
        if "update_times" in cp:
            self.update_times = list(cp["update_times"])
        if "replay_buffer" in cp:
            buf_size = cp.get("buffer_size", len(cp["replay_buffer"]))
            self.replay_buffer = ReplayBuffer(buf_size)
            self.replay_buffer.buffer.extend(cp["replay_buffer"])

    def get_avg_update_time(self):
        if not self.update_times:
            return None
        return float(np.mean(self.update_times))

    def close(self):
        pass
