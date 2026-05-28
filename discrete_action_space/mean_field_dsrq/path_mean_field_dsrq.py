"""PATH-backed mean-field Deep SRQ for MAgent2 Battle.

This module mirrors the MF-Q network used in the reference MFRL code: a
convolutional observation branch plus a mean-action branch. The Bellman target
replaces MF-Q's greedy value with a two-player representative SRE stage game
solved by the PATH bimatrix LCP solver.
"""

from __future__ import annotations

import time
import sys
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
if str(_DISCRETE_DIR) not in sys.path:
    sys.path.insert(0, str(_DISCRETE_DIR))

try:
    from discrete_action_space.sre_solvers import make_sre_solver
except ImportError:  # pragma: no cover - supports notebook-local sys.path setup.
    from sre_solvers import make_sre_solver


_DEFAULT_PATHWRAP = _THIS_DIR.parent / "sre_solvers" / "pathwrap.so"


class PathMeanFieldQNetwork(nn.Module):
    """Reference-style MFQ network: Conv(obs) + MLP(mean action) -> Q(a)."""

    def __init__(
        self,
        obs_channels: int,
        obs_height: int,
        obs_width: int,
        num_actions: int,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.obs_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * obs_height * obs_width, 256),
            nn.ReLU(),
        )
        self.mean_fc = nn.Sequential(
            nn.Linear(num_actions, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, obs: torch.Tensor, mean_action: torch.Tensor) -> torch.Tensor:
        obs_feat = self.obs_fc(self.conv(obs))
        mean_feat = self.mean_fc(mean_action)
        return self.head(torch.cat([obs_feat, mean_feat], dim=-1))


class PathMFReplayBuffer:
    """Ring buffer of per-agent mean-field transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))

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
        self.buffer.append(
            (
                np.asarray(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                np.asarray(mean_a, dtype=np.float32),
                np.asarray(next_mean_a, dtype=np.float32),
                float(done),
                float(valid),
            )
        )

    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
        idxs = np.random.randint(0, len(self.buffer), size=int(batch_size))
        batch = [self.buffer[int(i)] for i in idxs]
        obs, actions, rewards, next_obs, mean_a, next_mean_a, dones, valids = zip(*batch)

        def tensor(values, dtype):
            out = torch.as_tensor(np.stack(values), dtype=dtype)
            return out.to(device) if device is not None else out

        return {
            "obs": tensor(obs, torch.float32),
            "action": tensor(actions, torch.long),
            "reward": tensor(rewards, torch.float32),
            "next_obs": tensor(next_obs, torch.float32),
            "mean_a": tensor(mean_a, torch.float32),
            "next_mean_a": tensor(next_mean_a, torch.float32),
            "done": tensor(dones, torch.float32),
            "valid": tensor(valids, torch.float32),
        }

    def __len__(self) -> int:
        return len(self.buffer)


def _normalize_policy(policy: np.ndarray, size: int) -> np.ndarray:
    p = np.asarray(policy, dtype=np.float64).reshape(-1)
    if p.shape[0] != int(size):
        raise RuntimeError(f"Expected policy length {size}, got {p.shape[0]}.")
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        return np.full(size, 1.0 / size, dtype=np.float64)
    return p / total


class MFDsrqAgent:
    """One shared PATH mean-field DSRQ learner for one Battle population."""

    def __init__(
        self,
        type_id: int,
        obs_channels: int,
        obs_height: int,
        obs_width: int,
        n_own_actions: int,
        n_nbr_actions: int,
        *,
        epsilon_robust: float = 0.1,
        gamma: float = 0.95,
        lr: float = 1e-4,
        batch_size: int = 64,
        buffer_capacity: int = 80_000,
        learning_starts: int = 5_000,
        train_every: int = 5,
        target_tau: float = 0.005,
        grad_clip: Optional[float] = 10.0,
        epsilon_explore: float = 1.0,
        pathwrap_path: object = _DEFAULT_PATHWRAP,
        sre_solver_name: str = "path_c_pool",
        sre_solver_workers: int = 8,
        sre_solver_start_method: Optional[str] = None,
        sre_num_random_starts: int = 5,
        sre_num_pure_starts: int = 5,
        sre_policy_cache_enabled: bool = True,
        sre_policy_cache_size: int = 4096,
        sre_policy_cache_round_digits: int = 6,
        sre_uniform_fallback_on_failure: bool = True,
        sre_solver: Any = None,
        device: Optional[torch.device] = None,
    ):
        if int(n_own_actions) != int(n_nbr_actions):
            raise ValueError(
                "PATH mean-field DSRQ currently supports Battle-style symmetric "
                f"action spaces only; got own={n_own_actions}, neighbour={n_nbr_actions}."
            )
        self.type_id = int(type_id)
        self.n_own_actions = int(n_own_actions)
        self.n_nbr_actions = int(n_nbr_actions)
        self.num_actions = int(n_own_actions)
        self.epsilon_robust = float(epsilon_robust)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_every = max(1, int(train_every))
        self.target_tau = float(target_tau)
        self.grad_clip = grad_clip
        self.epsilon_explore = float(epsilon_explore)
        self.pathwrap_path = str(pathwrap_path)
        self.sre_solver_name = str(sre_solver_name)
        self.sre_solver_workers = int(sre_solver_workers)
        self.sre_solver_start_method = sre_solver_start_method
        self.sre_num_random_starts = max(0, int(sre_num_random_starts))
        self.sre_num_pure_starts = max(0, int(sre_num_pure_starts))
        self.sre_policy_cache_enabled = bool(sre_policy_cache_enabled)
        self.sre_policy_cache_size = int(sre_policy_cache_size)
        self.sre_policy_cache_round_digits = int(sre_policy_cache_round_digits)
        self.sre_uniform_fallback_on_failure = bool(sre_uniform_fallback_on_failure)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.q_net = PathMeanFieldQNetwork(
            obs_channels, obs_height, obs_width, self.num_actions
        ).to(device)
        self.target_net = PathMeanFieldQNetwork(
            obs_channels, obs_height, obs_width, self.num_actions
        ).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.opt = optim.Adam(self.q_net.parameters(), lr=float(lr))
        self.buffer = PathMFReplayBuffer(buffer_capacity)
        self.sre_solver = sre_solver or make_sre_solver(
            solver_name=self.sre_solver_name,
            pathwrap_path=self.pathwrap_path,
            max_workers=self.sre_solver_workers,
            start_method=self.sre_solver_start_method,
        )

        self._update_calls = 0
        self._total_train_steps = 0
        self._last_loss: Optional[float] = None
        self._update_times: list[float] = []
        self._sre_policy_cache: OrderedDict[tuple, list[np.ndarray]] = OrderedDict()
        self.sre_cache_exact_hits = 0
        self.sre_cache_misses = 0
        self.sre_failure_fallbacks = 0

    @property
    def _one_hot_mean_actions(self) -> torch.Tensor:
        return torch.eye(self.num_actions, dtype=torch.float32, device=self.device)

    def _policy_cache_key(self, q_tensor: np.ndarray) -> tuple:
        rounded = np.round(
            np.asarray(q_tensor, dtype=np.float32),
            self.sre_policy_cache_round_digits,
        )
        return (
            float(self.epsilon_robust),
            rounded.shape,
            rounded.tobytes(),
        )

    def _store_policy_cache(self, key: tuple, policies: list[np.ndarray]) -> None:
        if not self.sre_policy_cache_enabled or self.sre_policy_cache_size <= 0:
            return
        self._sre_policy_cache[key] = [p.copy() for p in policies]
        self._sre_policy_cache.move_to_end(key)
        while len(self._sre_policy_cache) > self.sre_policy_cache_size:
            self._sre_policy_cache.popitem(last=False)

    def _policies_from_result(self, result) -> list[np.ndarray]:
        metadata = dict(getattr(result, "metadata", None) or {})
        if not getattr(result, "success", False) or not getattr(result, "policies", None):
            if self.sre_uniform_fallback_on_failure:
                self.sre_failure_fallbacks += 1
                uniform = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float64)
                return [uniform, uniform.copy()]
            message = getattr(result, "message", "") or "PATH returned no SRE policy."
            raise RuntimeError(f"SRE solve failed. {message} Metadata: {metadata}")
        policies = [
            _normalize_policy(policy, self.num_actions)
            for policy in result.policies[:2]
        ]
        if len(policies) != 2:
            raise RuntimeError(f"Expected two SRE policies, got {len(policies)}.")
        return policies

    def _q_matrices_from_net(self, net: nn.Module, obs: torch.Tensor) -> torch.Tensor:
        """Return U1 matrices with shape [B, A_own, A_neighbor]."""
        batch_size = int(obs.shape[0])
        obs_rep = obs.repeat_interleave(self.num_actions, dim=0)
        mean_rep = self._one_hot_mean_actions.repeat(batch_size, 1)
        q_neighbor_first = net(obs_rep, mean_rep).reshape(
            batch_size, self.num_actions, self.num_actions
        )
        return q_neighbor_first.permute(0, 2, 1).contiguous()

    def _q_tensors_from_matrices(self, u1_batch: torch.Tensor) -> np.ndarray:
        u1_np = u1_batch.detach().cpu().numpy().astype(np.float32, copy=False)
        u2_np = np.swapaxes(u1_np, 1, 2)
        return np.stack([u1_np, u2_np], axis=-1)

    def _q_tensors_from_net(self, net: nn.Module, obs: torch.Tensor) -> np.ndarray:
        return self._q_tensors_from_matrices(self._q_matrices_from_net(net, obs))

    def _solve_sre_batch(self, q_tensors: np.ndarray) -> list[list[np.ndarray]]:
        q_tensors = np.asarray(q_tensors, dtype=np.float32)
        policies_by_index: list[Optional[list[np.ndarray]]] = [None] * int(q_tensors.shape[0])
        pending = []
        pending_indices = []
        pending_keys = []

        for idx, q_tensor in enumerate(q_tensors):
            key = self._policy_cache_key(q_tensor)
            cached = self._sre_policy_cache.get(key) if self.sre_policy_cache_enabled else None
            if cached is not None:
                self.sre_cache_exact_hits += 1
                self._sre_policy_cache.move_to_end(key)
                policies_by_index[idx] = [p.copy() for p in cached]
                continue
            self.sre_cache_misses += 1
            pending.append(q_tensor)
            pending_indices.append(idx)
            pending_keys.append(key)

        if pending:
            kwargs = {
                "epsilon": self.epsilon_robust,
                "num_random_starts": self.sre_num_random_starts,
                "num_pure_starts": self.sre_num_pure_starts,
            }
            if hasattr(self.sre_solver, "solve_batch"):
                results = self.sre_solver.solve_batch(pending, **kwargs)
            else:
                results = [self.sre_solver.solve(q_tensor, **kwargs) for q_tensor in pending]
            for idx, key, result in zip(pending_indices, pending_keys, results):
                policies = self._policies_from_result(result)
                policies_by_index[idx] = policies
                self._store_policy_cache(key, policies)

        return [[p.copy() for p in policies] for policies in policies_by_index]  # type: ignore[arg-type]

    def _sre_values(self, q_tensors: np.ndarray, policies_batch: list[list[np.ndarray]]) -> np.ndarray:
        values = []
        for q_tensor, policies in zip(q_tensors, policies_batch):
            p1, p2 = policies
            values.append(float(p1 @ q_tensor[:, :, 0] @ p2))
        return np.asarray(values, dtype=np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mean_a: Optional[np.ndarray] = None) -> int:
        del mean_a
        return int(self.act_batch(np.expand_dims(obs, axis=0), None)[0])

    @torch.no_grad()
    def act_batch(
        self,
        obs_batch: np.ndarray,
        mean_a_batch: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        del mean_a_batch
        batch_size = int(len(obs_batch))
        explore = np.random.rand(batch_size) < self.epsilon_explore
        actions = np.random.randint(0, self.num_actions, size=batch_size, dtype=np.int64)
        pending = np.flatnonzero(~explore)
        if pending.size:
            obs_t = torch.as_tensor(
                obs_batch[pending], dtype=torch.float32, device=self.device
            )
            q_tensors = self._q_tensors_from_net(self.q_net, obs_t)
            policies_batch = self._solve_sre_batch(q_tensors)
            for local_idx, batch_idx in enumerate(pending):
                policy = policies_batch[local_idx][0]
                actions[int(batch_idx)] = np.random.choice(self.num_actions, p=policy)
        return actions.astype(np.int64)

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

    def maybe_train(self) -> Optional[float]:
        if len(self.buffer) < self.learning_starts:
            return None
        if self._update_calls % self.train_every != 0:
            return None
        return self.train_step()

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None
        t0 = time.perf_counter()
        batch = self.buffer.sample(self.batch_size, self.device)

        obs = batch["obs"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_obs = batch["next_obs"]
        mean_a = batch["mean_a"]
        dones = batch["done"]
        valid = batch["valid"]

        q_values = self.q_net(obs, mean_a)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_values = torch.zeros_like(rewards)
            nonterminal = torch.nonzero(dones < 1.0, as_tuple=False).flatten()
            if nonterminal.numel():
                next_policy_q = self._q_tensors_from_net(self.q_net, next_obs[nonterminal])
                policies_batch = self._solve_sre_batch(next_policy_q)
                next_target_q = self._q_tensors_from_net(
                    self.target_net, next_obs[nonterminal]
                )
                values_np = self._sre_values(next_target_q, policies_batch)
                target_values[nonterminal] = torch.as_tensor(
                    values_np, dtype=torch.float32, device=self.device
                )
            y = rewards + (1.0 - dones) * self.gamma * target_values

        loss_per = nn.functional.mse_loss(q_taken, y, reduction="none")
        loss = (loss_per * valid).sum() / valid.sum().clamp(min=1.0)

        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), float(self.grad_clip))
        self.opt.step()
        self.soft_update_target_network(self.target_tau)

        self._total_train_steps += 1
        self._last_loss = float(loss.item())
        self._update_times.append(time.perf_counter() - t0)
        return self._last_loss

    def soft_update_target_network(self, tau: float) -> None:
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def save_checkpoint(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.opt.state_dict(),
                "epsilon_robust": self.epsilon_robust,
                "epsilon_explore": self.epsilon_explore,
                "total_train_steps": self._total_train_steps,
                "num_actions": self.num_actions,
                "sre_solver_name": self.sre_solver_name,
            },
            path,
        )

    def load_checkpoint(self, path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        if "optimizer" in checkpoint:
            self.opt.load_state_dict(checkpoint["optimizer"])
        self.epsilon_robust = float(checkpoint.get("epsilon_robust", self.epsilon_robust))
        self.epsilon_explore = float(checkpoint.get("epsilon_explore", self.epsilon_explore))
        self._total_train_steps = int(checkpoint.get("total_train_steps", 0))

    def get_avg_update_time_ms(self) -> Optional[float]:
        if not self._update_times:
            return None
        return float(np.mean(self._update_times)) * 1000.0

    def close(self) -> None:
        if hasattr(self.sre_solver, "close"):
            self.sre_solver.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
