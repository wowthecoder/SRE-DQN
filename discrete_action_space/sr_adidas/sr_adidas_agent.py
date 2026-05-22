"""SR-ADIDAS agent: policy network + Q-network trained jointly via robust ADI loss.

Differences from Deep SRQ (dueling_double_dqn_sre.py):
- Maintains an explicit policy network pi_theta(s) per agent (SharedTrunkPolicyNet).
- No per-step stage-game solver; equilibrium emerges from ADI gradient descent on pi.
- Q-network is bootstrapped against pi_theta, not the SRE oracle.
- Temperature tau is annealed globally (homotopy) rather than per state.
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover
    torch = None

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_DISCRETE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dueling_double_dqn_sre import (
    DuelingSharedTrunkPerAgentJointQNetwork,
    make_q_network,
    ReplayBuffer,
)
from sre_solvers.nplayer_common import robust_action_values, robust_exploitability

from .networks import SharedTrunkPolicyNet
from .robust_polymatrix import build_nominal_polymatrix
from .schedules import TauSchedule, EpsilonSchedule


def _expected_q_under_policy(q_tensor_t, policies_t):
    """Contract Q tensor with joint policy, returning per-agent expected value.

    Args:
        q_tensor_t: (B, A_1, ..., A_N, N) torch tensor.
        policies_t: list of N (B, A_k) torch tensors.
    Returns:
        (B, N) tensor of expected Q values.
    """
    val = q_tensor_t
    for pi_k in policies_t:
        val = torch.einsum("ba,ba...->b...", pi_k, val)
    return val  # (B, N)


class SrAdidasAgent:
    """Joint policy + Q-learning agent using the SR-ADIDAS algorithm.

    Parameters
    ----------
    obs_dim:
        Flattened observation dimension.
    num_agents:
        Number of agents N.
    num_actions:
        Shared action count A (assumed uniform across agents).
    epsilon_robust:
        Initial TV-Wasserstein robustness radius.
    tau_init:
        Initial entropy temperature for the ADI homotopy.
    tau_min:
        Minimum temperature; annealing stops here.
    tau_threshold:
        ADI convergence threshold that triggers tau halving.
    eta_y:
        EMA learning rate for y_k estimates (0 = no update, 1 = full refresh each step).
    lr_q, lr_pi:
        Learning rates for Q and policy networks.
    gamma:
        Discount factor.
    buffer_size, batch_size, learning_starts:
        Replay buffer settings.
    grad_clip:
        Gradient clipping max-norm.
    target_update_steps:
        Hard target-network update interval (in train_step calls).
    train_every:
        Number of update() calls between train_step() invocations.
    network_type:
        Q-network architecture; passed to make_q_network().
    action_epsilon_start/end/decay_fraction/total_steps:
        Exploration epsilon (greedy) schedule.
    use_gpu:
        Whether to use CUDA if available.
    """

    def __init__(
        self,
        obs_dim,
        num_agents,
        num_actions,
        *,
        epsilon_robust=1.0,
        tau_init=100.0,
        tau_min=1e-3,
        tau_decay=0.5,
        tau_threshold=1e-3,
        eta_y=1.0,
        lr_q=3e-4,
        lr_pi=1e-3,
        gamma=0.99,
        buffer_size=20_000,
        batch_size=32,
        learning_starts=500,
        grad_clip=10.0,
        target_update_steps=250,
        train_every=4,
        network_type="shared_trunk_separate_heads",
        action_epsilon_start=1.0,
        action_epsilon_end=0.05,
        action_epsilon_decay_fraction=0.6,
        total_steps=50_000,
        epsilon_robust_end=0.01,
        epsilon_robust_decay_fraction=0.8,
        use_gpu=True,
    ):
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.grad_clip = float(grad_clip)
        self.target_update_steps = int(target_update_steps)
        self.train_every = max(1, int(train_every))
        self.eta_y = float(eta_y)
        self._update_calls = 0
        self._train_step_calls = 0

        if use_gpu and torch is not None and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Q-networks
        self.q_net = make_q_network(obs_dim, num_actions, num_agents, network_type).to(self.device)
        self.q_target = make_q_network(obs_dim, num_actions, num_agents, network_type).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        # Policy network
        self.pi_net = SharedTrunkPolicyNet(obs_dim, num_actions, num_agents).to(self.device)

        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q)
        self.pi_optimizer = optim.Adam(self.pi_net.parameters(), lr=lr_pi)

        self.replay_buffer = ReplayBuffer(buffer_size)

        # Schedules
        self.tau_schedule = TauSchedule(tau_init, tau_min, tau_decay, tau_threshold)
        self.action_eps_schedule = EpsilonSchedule(
            action_epsilon_start, action_epsilon_end,
            action_epsilon_decay_fraction, total_steps, mode="linear",
        )
        self.robust_eps_schedule = EpsilonSchedule(
            epsilon_robust, epsilon_robust_end,
            epsilon_robust_decay_fraction, total_steps, mode="linear",
        )

        # EMA payoff-gradient cache: maps replay-buffer sample index → y_k vectors
        # We use a lightweight dict; evicted lazily when buffer wraps.
        self._y_cache: dict[int, list[np.ndarray]] = {}
        self._buf_counter = 0  # monotonic index for replay buffer entries

        # Diagnostics
        self.train_losses_q: list[float] = []
        self.train_losses_pi: list[float] = []
        self.adi_estimates: list[float] = []
        self.train_times: list[float] = []

    # ------------------------------------------------------------------ helpers

    def _state_to_vec(self, state):
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def _uniform_policies(self):
        u = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        return [u.copy() for _ in range(self.num_agents)]

    @property
    def epsilon_robust(self):
        return self.robust_eps_schedule.value()

    @property
    def epsilon_explore(self):
        return self.action_eps_schedule.value()

    @property
    def tau(self):
        return self.tau_schedule.value()

    # -------------------------------------------------------------- acting

    def act(self, state, agent_id=0):
        """Epsilon-greedy action from the current policy network."""
        if np.random.rand() < self.epsilon_explore:
            return int(np.random.randint(self.num_actions))

        s = torch.as_tensor(
            self._state_to_vec(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            policies = self.pi_net(s)
        pi_k = policies[agent_id].squeeze(0).cpu().numpy()
        pi_k = np.clip(pi_k, 0.0, None)
        pi_k /= pi_k.sum() + 1e-12
        return int(np.random.choice(self.num_actions, p=pi_k))

    def act_all(self, state):
        """Return joint actions for all agents (list of ints)."""
        if np.random.rand() < self.epsilon_explore:
            return [int(np.random.randint(self.num_actions)) for _ in range(self.num_agents)]

        s = torch.as_tensor(
            self._state_to_vec(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            policies = self.pi_net(s)
        actions = []
        for k in range(self.num_agents):
            pi_k = policies[k].squeeze(0).cpu().numpy()
            pi_k = np.clip(pi_k, 0.0, None)
            pi_k /= pi_k.sum() + 1e-12
            actions.append(int(np.random.choice(self.num_actions, p=pi_k)))
        return actions

    # -------------------------------------------------------------- update interface

    def push(self, state, joint_actions, joint_rewards, next_state, done):
        sv = self._state_to_vec(state)
        nsv = self._state_to_vec(next_state)
        a = np.asarray(joint_actions, dtype=np.int64)
        r = np.asarray(joint_rewards, dtype=np.float32)
        self.replay_buffer.push(sv, a, r, nsv, done)
        self._buf_counter += 1
        self.action_eps_schedule.step()
        self.robust_eps_schedule.step()

    def maybe_train(self):
        """Call after each push; runs train_step every `train_every` calls."""
        self._update_calls += 1
        if self._update_calls % self.train_every != 0:
            return None
        return self.train_step()

    # -------------------------------------------------------------- training

    def train_step(self):
        if len(self.replay_buffer) < max(self.batch_size, self.learning_starts):
            return None

        t0 = time.perf_counter()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # -- numpy arrays
        states_np = np.stack([self._state_to_vec(s) for s in states])        # (B, obs)
        next_states_np = np.stack([self._state_to_vec(s) for s in next_states])
        actions_np = np.stack(actions)   # (B, N) int
        rewards_np = np.stack(rewards)   # (B, N) float
        dones_np = np.asarray(dones, dtype=np.float32).reshape(-1)            # (B,)

        # -- tensors
        s_t = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        ns_t = torch.as_tensor(next_states_np, dtype=torch.float32, device=self.device)
        r_t = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)  # (B,N)
        d_t = torch.as_tensor(dones_np, dtype=torch.float32, device=self.device)    # (B,)

        # ---- Q-network update ----------------------------------------
        # Current Q values at observed joint actions
        q_all = self.q_net(s_t)  # (B, A,...,A, N)
        # Index: q_all[b, a_0, ..., a_{N-1}, k]
        idx = tuple(actions_np[:, k] for k in range(self.num_agents))
        q_sa = q_all[torch.arange(self.batch_size, device=self.device), *idx, :]
        # q_sa shape: (B, N)

        with torch.no_grad():
            # Policy at next state (no grad, target for Q bootstrap)
            pi_next = self.pi_net(ns_t)  # list of N (B,A) tensors
            q_next = self.q_target(ns_t)  # (B, A,...,A, N)
            v_next = _expected_q_under_policy(q_next, pi_next)  # (B, N)
            targets = r_t + self.gamma * v_next * (1.0 - d_t.unsqueeze(-1))  # (B,N)

        q_loss = nn.functional.mse_loss(q_sa, targets)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.q_optimizer.step()

        # ---- Policy (ADI) update ------------------------------------
        eps = self.epsilon_robust
        tau = self.tau

        # Compute robust action values and EMA estimates in numpy (per-state)
        with torch.no_grad():
            q_np = self.q_net(s_t).detach().cpu().numpy()  # (B, A,...,A, N)
            pi_np_batch = [
                self.pi_net(s_t)[k].detach().cpu().numpy()
                for k in range(self.num_agents)
            ]  # each (B, A)

        y_batch = []       # list of N (B,A) arrays
        adi_sum = 0.0

        for k in range(self.num_agents):
            y_k_batch = np.zeros((self.batch_size, self.num_actions), dtype=np.float64)
            for b in range(self.batch_size):
                q_b = q_np[b]
                policies_b = [pi_np_batch[j][b].astype(np.float64) for j in range(self.num_agents)]
                v_k_rob = robust_action_values(q_b, policies_b, eps, k)  # (A,)
                y_k_prev = self._get_y(b, k)
                y_k_new = y_k_prev + self.eta_y * (v_k_rob - y_k_prev)
                self._set_y(b, k, y_k_new)
                y_k_batch[b] = y_k_new

                # accumulate ADI estimate contribution (Eq. 8 simplified)
                br_k = self._softmax(y_k_new, tau)
                adi_sum += float(y_k_new @ (br_k - policies_b[k]))

            y_batch.append(y_k_batch)

        adi_est = adi_sum / (self.batch_size * self.num_agents)
        self.tau_schedule.step(adi_est)
        self.adi_estimates.append(adi_est)

        # Policy gradient: L_pi = sum_k [-y_k @ pi_k - tau * H(pi_k)]
        pi_policies = self.pi_net(s_t)  # list of N (B, A) tensors, differentiable

        pi_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        pi_loss_list = []
        for k in range(self.num_agents):
            y_k_t = torch.as_tensor(y_batch[k], dtype=torch.float32, device=self.device)
            pi_k = pi_policies[k]  # (B, A) differentiable
            # -y_k @ pi_k: per-sample dot product
            neg_utility = -(y_k_t * pi_k).sum(dim=-1).mean()
            # +tau * sum_a pi_k[a] * log(pi_k[a])  (negative entropy, we minimise)
            entropy_penalty = tau * (pi_k * torch.log(pi_k + 1e-12)).sum(dim=-1).mean()
            pi_loss_list.append(neg_utility + entropy_penalty)

        pi_loss = sum(pi_loss_list)
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.pi_net.parameters(), self.grad_clip)
        self.pi_optimizer.step()

        # Target network update
        self._train_step_calls += 1
        if self._train_step_calls % self.target_update_steps == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        elapsed = time.perf_counter() - t0
        self.train_times.append(elapsed)
        self.train_losses_q.append(float(q_loss.detach()))
        self.train_losses_pi.append(float(pi_loss.detach()))

        return {
            "q_loss": float(q_loss.detach()),
            "pi_loss": float(pi_loss.detach()),
            "adi": float(adi_est),
            "tau": float(tau),
            "epsilon_robust": float(eps),
        }

    # -------------------------------------------------------------- EMA cache

    def _get_y(self, batch_idx, player_k):
        key = (batch_idx, player_k)
        return self._y_cache.get(key, np.zeros(self.num_actions, dtype=np.float64))

    def _set_y(self, batch_idx, player_k, value):
        self._y_cache[(batch_idx, player_k)] = np.asarray(value, dtype=np.float64)

    @staticmethod
    def _softmax(values, tau):
        v = np.asarray(values, dtype=np.float64)
        if tau <= 1e-12:
            best = v >= np.max(v) - 1e-10
            return best.astype(np.float64) / float(best.sum())
        centered = v - float(np.max(v))
        w = np.exp(centered / float(tau))
        return w / float(w.sum())

    # -------------------------------------------------------------- evaluation

    def eval_exploitability(self, states):
        """Compute mean robust exploitability over a set of states."""
        eps = self.epsilon_robust
        gaps = []
        s_t = torch.as_tensor(
            np.stack([self._state_to_vec(s) for s in states]),
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            q_batch = self.q_net(s_t).detach().cpu().numpy()
            pi_batch = [self.pi_net(s_t)[k].detach().cpu().numpy() for k in range(self.num_agents)]

        for b in range(len(states)):
            q_b = q_batch[b]
            pi_b = [pi_batch[k][b].astype(np.float64) for k in range(self.num_agents)]
            gap, _, _ = robust_exploitability(q_b, pi_b, eps)
            gaps.append(gap)
        return float(np.mean(gaps))

    def get_policies(self, state):
        """Return current mixed strategies for all agents at a single state."""
        s = torch.as_tensor(
            self._state_to_vec(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            policies = self.pi_net(s)
        return [p.squeeze(0).cpu().numpy() for p in policies]

    def save_checkpoint(self, path):
        """Save enough state for notebook evaluation and training inspection."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "num_agents": self.num_agents,
                "num_actions": self.num_actions,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "learning_starts": self.learning_starts,
                "grad_clip": self.grad_clip,
                "target_update_steps": self.target_update_steps,
                "train_every": self.train_every,
                "eta_y": self.eta_y,
                "q_net": self.q_net.state_dict(),
                "q_target": self.q_target.state_dict(),
                "pi_net": self.pi_net.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "pi_optimizer": self.pi_optimizer.state_dict(),
                "tau": self.tau_schedule.tau,
                "action_epsilon_step": self.action_eps_schedule._step,
                "robust_epsilon_step": self.robust_eps_schedule._step,
                "train_losses_q": self.train_losses_q,
                "train_losses_pi": self.train_losses_pi,
                "adi_estimates": self.adi_estimates,
                "train_times": self.train_times,
            },
            path,
        )

    def load_checkpoint(self, path, map_location=None):
        """Load a checkpoint saved by :meth:`save_checkpoint`."""
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        self.pi_net.load_state_dict(checkpoint["pi_net"])
        if "q_optimizer" in checkpoint:
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        if "pi_optimizer" in checkpoint:
            self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer"])
        self.tau_schedule.tau = float(checkpoint.get("tau", self.tau_schedule.tau))
        self.action_eps_schedule._step = int(
            checkpoint.get("action_epsilon_step", self.action_eps_schedule._step)
        )
        self.robust_eps_schedule._step = int(
            checkpoint.get("robust_epsilon_step", self.robust_eps_schedule._step)
        )
        self.train_losses_q = list(checkpoint.get("train_losses_q", self.train_losses_q))
        self.train_losses_pi = list(checkpoint.get("train_losses_pi", self.train_losses_pi))
        self.adi_estimates = list(checkpoint.get("adi_estimates", self.adi_estimates))
        self.train_times = list(checkpoint.get("train_times", self.train_times))
        return checkpoint
