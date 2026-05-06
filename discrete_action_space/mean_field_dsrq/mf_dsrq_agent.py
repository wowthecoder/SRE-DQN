"""MF-DSRQ agent (per agent-type).

One MFDsrqAgent instance manages training for all agents of the same type.
Weight sharing is handled by the single Q-network.

Robust Bellman target (from observed ā, no fixed-point iteration):
    z_target(a)  = TV_worst(ā'_observed, Q_φ̄(o', a, ·), ε)   for each own-action a
    π_target(a)  = softmax(β · z_target(a))
    y            = r + γ · (1 - done) · Σ_a π_target(a) · z_target(a)
"""

import copy
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .mf_q_network import MeanFieldQNetwork
from .mf_replay_buffer import MFReplayBuffer
from .mf_robust_value import boltzmann_policy, robust_q_grid


class MFDsrqAgent:
    def __init__(
        self,
        type_id: int,
        obs_channels: int,
        obs_height: int,
        obs_width: int,
        n_own_actions: int,
        n_nbr_actions: int,
        *,
        epsilon_tv: float = 0.10,
        beta: float = 1.0,
        gamma: float = 0.95,
        lr: float = 1e-4,
        batch_size: int = 256,
        buffer_capacity: int = 1_000_000,
        learning_starts: int = 5_000,
        train_every: int = 4,
        target_tau: float = 0.005,
        grad_clip: float = 10.0,
        epsilon_explore: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.type_id = type_id
        self.n_own_actions = n_own_actions
        self.n_nbr_actions = n_nbr_actions
        self.epsilon_tv = float(epsilon_tv)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_every = int(train_every)
        self.target_tau = float(target_tau)
        self.grad_clip = float(grad_clip)
        self.epsilon_explore = float(epsilon_explore)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.q_net = MeanFieldQNetwork(
            obs_channels, obs_height, obs_width, n_own_actions, n_nbr_actions
        ).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = MFReplayBuffer(
            buffer_capacity,
            obs_shape=(obs_channels, obs_height, obs_width),
            n_actions=n_nbr_actions,
        )

        self._update_calls = 0
        self._total_train_steps = 0
        self._last_loss: Optional[float] = None
        self._update_times: list[float] = []

    # ------------------------------------------------------------------ #
    # Action selection                                                     #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        mean_a: np.ndarray,
    ) -> int:
        """Select action for a single agent given obs and current mean field.

        Args:
            obs: [C, H, W] float32 observation.
            mean_a: [A_nbr] float32 neighborhood mean-action distribution.

        Returns:
            Chosen action index.
        """
        if np.random.rand() < self.epsilon_explore:
            return int(np.random.randint(self.n_own_actions))

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        ma_t = torch.as_tensor(mean_a, dtype=torch.float32, device=self.device).unsqueeze(0)

        q_grid = self.q_net(obs_t)                          # [1, A_own, A_nbr]
        q_rob = robust_q_grid(q_grid, ma_t, self.epsilon_tv)  # [1, A_own]
        pi = boltzmann_policy(q_rob, self.beta)               # [1, A_own]
        return int(torch.multinomial(pi[0], 1).item())

    @torch.no_grad()
    def act_batch(
        self,
        obs_batch: np.ndarray,
        mean_a_batch: np.ndarray,
    ) -> np.ndarray:
        """Vectorised action selection for a batch of agents of this type.

        Args:
            obs_batch:   [N, C, H, W]
            mean_a_batch: [N, A_nbr]

        Returns:
            actions: [N] int64
        """
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        ma_t = torch.as_tensor(mean_a_batch, dtype=torch.float32, device=self.device)

        q_grid = self.q_net(obs_t)                             # [N, A_own, A_nbr]
        q_rob = robust_q_grid(q_grid, ma_t, self.epsilon_tv)   # [N, A_own]

        explore_mask = np.random.rand(obs_t.shape[0]) < self.epsilon_explore
        pi = boltzmann_policy(q_rob, self.beta)                 # [N, A_own]
        greedy = torch.multinomial(pi, 1).squeeze(1).cpu().numpy()
        random_acts = np.random.randint(0, self.n_own_actions, size=obs_t.shape[0])

        actions = np.where(explore_mask, random_acts, greedy)
        return actions.astype(np.int64)

    # ------------------------------------------------------------------ #
    # Buffer                                                               #
    # ------------------------------------------------------------------ #

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        mean_a: np.ndarray,
        next_mean_a: np.ndarray,
        done: bool,
        valid: bool = True,
    ) -> None:
        self.buffer.push(obs, action, reward, next_obs, mean_a, next_mean_a, done, valid)
        self._update_calls += 1

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def maybe_train(self) -> Optional[float]:
        """Train if enough steps have accumulated. Returns loss or None."""
        if len(self.buffer) < self.learning_starts:
            return None
        if self._update_calls % self.train_every != 0:
            return None
        return self.train_step()

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None

        t0 = time.perf_counter()
        batch = self.buffer.sample(self.batch_size, device=self.device)

        obs = batch["obs"]               # [B, C, H, W]
        actions = batch["action"]        # [B]
        rewards = batch["reward"]        # [B]
        next_obs = batch["next_obs"]     # [B, C, H, W]
        mean_a = batch["mean_a"]         # [B, A_nbr]
        next_mean_a = batch["next_mean_a"]  # [B, A_nbr]
        dones = batch["done"]            # [B]
        valid = batch["valid"]           # [B]

        # Online Q for own action a taken.
        q_grid_online = self.q_net(obs)                         # [B, A_own, A_nbr]
        # Gather the neighbor-payoff row for the action that was taken.
        a_idx = actions.view(-1, 1, 1).expand(-1, 1, self.n_nbr_actions)
        q_taken_row = q_grid_online.gather(1, a_idx).squeeze(1) # [B, A_nbr]
        q_taken = tv_worst_case_batch_own(q_taken_row, mean_a, self.epsilon_tv)  # [B]

        with torch.no_grad():
            # Target network Q-grid.
            q_grid_target = self.target_net(next_obs)           # [B, A_own, A_nbr]
            # Robust Q for all own-actions using observed next mean field.
            z_target = robust_q_grid(q_grid_target, next_mean_a, self.epsilon_tv)  # [B, A_own]
            pi_target = boltzmann_policy(z_target, self.beta)   # [B, A_own]
            v_target = (pi_target * z_target).sum(dim=-1)       # [B]
            y = rewards + (1.0 - dones) * self.gamma * v_target

        loss_per = nn.functional.smooth_l1_loss(q_taken, y, reduction="none")
        loss = (loss_per * valid).sum() / valid.sum().clamp(min=1.0)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.opt.step()

        self._soft_update_target()
        self._total_train_steps += 1
        self._last_loss = float(loss.item())
        self._update_times.append(time.perf_counter() - t0)
        return self._last_loss

    def _soft_update_target(self) -> None:
        tau = self.target_tau
        for p_online, p_target in zip(self.q_net.parameters(), self.target_net.parameters()):
            p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "opt": self.opt.state_dict(),
            "epsilon_tv": self.epsilon_tv,
            "beta": self.beta,
            "epsilon_explore": self.epsilon_explore,
            "total_train_steps": self._total_train_steps,
        }, path)

    def load_checkpoint(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.epsilon_tv = ckpt.get("epsilon_tv", self.epsilon_tv)
        self.beta = ckpt.get("beta", self.beta)
        self.epsilon_explore = ckpt.get("epsilon_explore", self.epsilon_explore)
        self._total_train_steps = ckpt.get("total_train_steps", 0)

    def get_avg_update_time_ms(self) -> Optional[float]:
        if not self._update_times:
            return None
        return float(np.mean(self._update_times)) * 1000.0


# Module-level helper so train_step can import without circular dep.
def tv_worst_case_batch_own(v_row: torch.Tensor, mean_a: torch.Tensor, epsilon: float) -> torch.Tensor:
    """TV worst-case for a single (batch of) own-action rows."""
    from .mf_robust_value import tv_worst_case_batch
    return tv_worst_case_batch(mean_a, v_row, epsilon)
