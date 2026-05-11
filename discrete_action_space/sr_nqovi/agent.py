"""
SR-NQOVI: Strategically Robust Nash Q-Learning with Optimistic Value Iteration.

Replaces the Nash equilibrium operator in NQOVI (Cisneros-Velarde & Koyejo, 2023)
with the SRE operator, giving a sample-efficient, theory-aligned MARL algorithm
with tunable robustness via epsilon.

Key change vs NQOVI: lines 7 and 14 of Algorithm 1 call SRE instead of NE.

Performance note: with tabular one-hot features, Lambda_h is always diagonal
(rank-1 updates from one-hot outer products).  Use `diagonal_lambda=True` (default)
to exploit this: all Q lookups become O(1) instead of O(d²).
"""
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_DISCRETE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sre_solvers import make_sre_solver


def _linear_schedule(start, end, step, total_steps):
    if total_steps <= 1:
        return float(end)
    frac = min(max(step, 0) / float(total_steps - 1), 1.0)
    return float(start + frac * (end - start))


@dataclass
class SrNqoviConfig:
    num_agents: int = 2
    num_actions: int = 4
    feature_dim: int = 144           # d (num_states * num_joint_actions for tabular)
    horizon: int = 100               # H in the paper (max episode length)
    ridge: float = 1.0               # lambda in the paper
    ucb_constant: float = 1.0        # c_beta multiplier; beta = c_beta * sqrt(d) * H * iota
    total_episodes: int = 3000       # K (used to set iota at init time)
    delta: float = 0.05              # failure probability for iota
    epsilon_robust: float = 1.0      # ε_k (robustness parameter; can be scheduled)
    epsilon_robust_initial: float = 1.0
    epsilon_schedule: str = "linear" # "constant" | "linear" | "exponential"
    epsilon_explore: float = 1.0     # action-selection exploration
    action_epsilon_start: float = 1.0
    action_epsilon_end: float = 0.05
    action_epsilon_decay_fraction: float = 0.5
    decay_rate: float = 0.999        # used only for exponential schedule
    sre_solver_name: str = "path_c"
    sre_num_repeats: int = 4
    sre_include_pure_starts: bool = False
    pathwrap_path: str = "pathwrap.so"
    diagonal_lambda: bool = True     # exploit diagonal structure for one-hot features
    sre_solver: Any = field(default=None, repr=False)


class SrNqoviAgent:
    """
    SR-NQOVI agent: linear Q-functions with per-step UCB optimism and SRE policies.

    Stores per-step:
      W[h, i, j]          weight for agent i, step h, feature j
      lambda_diag[h, j]   diagonal of Lambda_h (diagonal_lambda=True, exact for one-hot)
      Lambda[h]           full Lambda_h (diagonal_lambda=False, needed for RBF/Fourier)
    """

    def __init__(
        self,
        config: SrNqoviConfig,
        feature_fn,
        state_to_index,
        joint_action_to_index,
    ):
        self.config = config
        self.feature_fn = feature_fn
        self.state_to_index = state_to_index
        self.joint_action_to_index = joint_action_to_index

        H = config.horizon
        n = config.num_agents
        d = config.feature_dim

        self.W = np.zeros((H, n, d), dtype=np.float64)

        if config.diagonal_lambda:
            self.lambda_diag = np.full((H, d), config.ridge, dtype=np.float64)
        else:
            self.Lambda = np.array([config.ridge * np.eye(d) for _ in range(H)])
            self.Lchol = np.array([np.sqrt(config.ridge) * np.eye(d) for _ in range(H)])

        # trajectory[h] = [(s_idx, joint_a_idx, r_vec, s_next_idx), ...]
        self.trajectory: list[list] = [[] for _ in range(H)]
        # For diagonal_lambda, track which feature index each transition hit
        self.traj_feat_idx: list[list] = [[] for _ in range(H)]
        # For full-matrix lambda, cache phi vectors
        self.trajectory_phi: list[list] = [[] for _ in range(H)]

        # UCB constant: beta = c_beta * sqrt(d) * H * iota
        iota = math.log(max(d * config.total_episodes * H / max(config.delta, 1e-10), 2.0))
        self.beta = config.ucb_constant * math.sqrt(d) * H * iota

        if config.sre_solver is not None:
            self.sre_solver = config.sre_solver
        else:
            self.sre_solver = make_sre_solver(
                solver_name=config.sre_solver_name,
                pathwrap_path=config.pathwrap_path,
            )

        self.sre_solve_times: list[float] = []

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _flat_idx(self, state_index, joint_action_index) -> int:
        """Feature index for tabular one-hot encoding."""
        return int(state_index) * (self.config.num_actions ** self.config.num_agents) + int(joint_action_index)

    def _joint_index_to_actions(self, joint_idx):
        n_a = self.config.num_actions
        n = self.config.num_agents
        actions = []
        for _ in range(n):
            actions.append(joint_idx % n_a)
            joint_idx //= n_a
        return list(reversed(actions))

    # ------------------------------------------------------------------
    # Q-value computation — two paths: diagonal and full-matrix
    # ------------------------------------------------------------------

    def _bonus(self, flat_idx_or_phi, h):
        """UCB bonus for a feature vector (or flat index when diagonal_lambda=True)."""
        if self.config.diagonal_lambda:
            idx = int(flat_idx_or_phi)
            lam_val = float(self.lambda_diag[h, idx])
            return self.beta * math.sqrt(max(1.0 / lam_val, 0.0))
        else:
            phi = flat_idx_or_phi
            Lh = self.Lchol[h]
            try:
                v = np.linalg.solve(Lh, phi)
                return self.beta * math.sqrt(max(float(v @ v), 0.0))
            except np.linalg.LinAlgError:
                return float(self.beta)

    def q_value_flat(self, flat_idx, h) -> np.ndarray:
        """Q-vector (shape [n]) at step h via flat feature index (diagonal_lambda path)."""
        bonus = self._bonus(flat_idx, h)
        q = self.W[h, :, flat_idx] + bonus
        return np.minimum(q, float(self.config.horizon))

    def q_value(self, state_index, joint_action_index, h) -> np.ndarray:
        """Q-vector (shape [n]) at step h for a given (state, joint_action)."""
        if self.config.diagonal_lambda:
            return self.q_value_flat(self._flat_idx(state_index, joint_action_index), h)
        phi = self.feature_fn(state_index, joint_action_index)
        bonus = self._bonus(phi, h)
        q = self.W[h] @ phi + bonus
        return np.minimum(q, float(self.config.horizon))

    def q_tensor(self, state_index, h) -> np.ndarray:
        """
        Full Q-tensor at (state, step h), shape (A, ..., A, n_agents).
        Fast path for diagonal_lambda: all 16 entries are simple array slices.
        """
        n_a = self.config.num_actions
        n = self.config.num_agents
        num_joint = n_a ** n
        shape = tuple([n_a] * n + [n])
        qt = np.zeros(shape, dtype=np.float64)

        if self.config.diagonal_lambda:
            base = int(state_index) * num_joint
            for j in range(num_joint):
                flat = base + j
                bonus = self._bonus(flat, h)
                q = np.minimum(self.W[h, :, flat] + bonus, float(self.config.horizon))
                actions = self._joint_index_to_actions(j)
                qt[tuple(actions)] = q
        else:
            for j in range(num_joint):
                actions = self._joint_index_to_actions(j)
                phi = self.feature_fn(state_index, j)
                bonus = self._bonus(phi, h)
                q = np.minimum(self.W[h] @ phi + bonus, float(self.config.horizon))
                qt[tuple(actions)] = q

        return qt

    # ------------------------------------------------------------------
    # SRE solving
    # ------------------------------------------------------------------

    def _solve_sre(self, q_t) -> list:
        n_a = self.config.num_actions
        n = self.config.num_agents
        uniform = [np.full(n_a, 1.0 / n_a) for _ in range(n)]

        t0 = time.perf_counter()
        try:
            result = self.sre_solver.solve(
                q_t,
                epsilon=self.config.epsilon_robust,
                num_repeats=self.config.sre_num_repeats,
                include_pure_starts=self.config.sre_include_pure_starts,
            )
        except TypeError:
            try:
                result = self.sre_solver.solve(
                    q_t,
                    epsilon=self.config.epsilon_robust,
                    num_repeats=self.config.sre_num_repeats,
                )
            except Exception:
                result = None
        except Exception:
            result = None
        self.sre_solve_times.append(time.perf_counter() - t0)

        if result is None or not result.success or not result.policies:
            return uniform
        policies = []
        for p in result.policies:
            p = np.asarray(p, dtype=np.float64)
            p = np.clip(p, 0.0, None)
            s = p.sum()
            policies.append(p / s if s > 1e-12 else np.full(n_a, 1.0 / n_a))
        if len(policies) != n or any(p.shape[0] != n_a for p in policies):
            return uniform
        return policies

    def sre_policy(self, state_index, h) -> list:
        qt = self.q_tensor(state_index, h)
        return self._solve_sre(qt)

    def act(self, state_index, h, agent_id=0) -> int:
        if np.random.rand() < self.config.epsilon_explore:
            return int(np.random.randint(self.config.num_actions))
        policies = self.sre_policy(state_index, h)
        p = policies[agent_id]
        return int(np.random.choice(self.config.num_actions, p=p))

    # ------------------------------------------------------------------
    # Backward optimistic value iteration (core of SR-NQOVI)
    # ------------------------------------------------------------------

    def backward_pass(self, current_episode_trajectory):
        """
        Run backward OVI over all H steps for the current episode k.

        current_episode_trajectory: list of length <= H, each element:
            (state_index, joint_action_index, reward_vec [n], next_state_index)
        """
        H = self.config.horizon
        n = self.config.num_agents
        d = self.config.feature_dim
        n_a = self.config.num_actions
        num_joint = n_a ** n
        ep_len = len(current_episode_trajectory)

        if ep_len == 0:
            return

        # --- Append current episode to trajectory store and update Lambda ---
        for h, (s_idx, a_idx, r_vec, sn_idx) in enumerate(current_episode_trajectory):
            self.trajectory[h].append((s_idx, a_idx, r_vec, sn_idx))

            if self.config.diagonal_lambda:
                flat = self._flat_idx(s_idx, a_idx)
                self.traj_feat_idx[h].append(flat)
                self.lambda_diag[h, flat] += 1.0
            else:
                phi_h = self.feature_fn(s_idx, a_idx)
                self.trajectory_phi[h].append(phi_h)
                self.Lambda[h] += np.outer(phi_h, phi_h)
                try:
                    self.Lchol[h] = np.linalg.cholesky(self.Lambda[h])
                except np.linalg.LinAlgError:
                    self.Lchol[h] = np.diag(np.sqrt(np.abs(np.diag(self.Lambda[h]))))

        # --- Build joint policy distribution from a list of per-agent policies ---
        def _joint_prob(policies):
            prob = np.ones(num_joint, dtype=np.float64)
            for j in range(num_joint):
                actions = self._joint_index_to_actions(j)
                for agent_id, a in enumerate(actions):
                    prob[j] *= float(policies[agent_id][a])
            return prob

        # --- Backward pass: h = H-1, ..., 0 ---
        for h in range(H - 1, -1, -1):
            if not self.trajectory[h]:
                continue

            num_past = len(self.trajectory[h])

            # ---- Line 7: π* = SRE at x^k_{h+1} (leaf of current episode) ----
            # Use Q^k_{h+1} which has just been updated above.
            if h + 1 < H:
                if h < ep_len:
                    s_next_k = current_episode_trajectory[h][3]
                else:
                    s_next_k = current_episode_trajectory[-1][3]
                qt_leaf = self.q_tensor(s_next_k, h + 1)
                pi_star = self._solve_sre(qt_leaf)
                joint_prob = _joint_prob(pi_star)  # shape (num_joint,)
            else:
                pi_star = None
                joint_prob = None

            # ---- Line 9: compute bootstrap targets and rhs ----
            # For diagonal case: Q(sn, j) = W[h+1, :, flat(sn,j)] + bonus[flat(sn,j)]
            # For each tau: bootstrap[tau, i] = sum_j joint_prob[j] * Q_i(sn_tau, j)

            rhs = np.zeros((n, d), dtype=np.float64)  # will accumulate Σ_τ phi_τ * target_τ

            for tau in range(num_past):
                s_idx, a_idx, r_vec, sn_idx = self.trajectory[h][tau]
                r_vec = np.asarray(r_vec, dtype=np.float64)

                if h + 1 < H and joint_prob is not None:
                    # E_{a~π*}[Q^{i,k}_{h+1}(x^τ_{h+1}, a)] (same π* for all τ, per paper)
                    if self.config.diagonal_lambda:
                        base_next = int(sn_idx) * num_joint
                        flat_indices = np.arange(base_next, base_next + num_joint)
                        bonus_vec = self.beta * np.sqrt(
                            np.maximum(1.0 / self.lambda_diag[h + 1, flat_indices], 0.0)
                        )
                        # np.take avoids advanced-indexing axis promotion
                        W_slice = np.take(self.W[h + 1], flat_indices, axis=1)  # (n, num_joint)
                        q_next_by_agent = np.minimum(
                            W_slice + bonus_vec[np.newaxis, :],
                            float(self.config.horizon),
                        )  # (n, num_joint)
                        bootstrap = q_next_by_agent @ joint_prob  # (n,)
                    else:
                        bootstrap = np.zeros(n, dtype=np.float64)
                        for j in range(num_joint):
                            phi_j = self.feature_fn(sn_idx, j)
                            bonus_j = self._bonus(phi_j, h + 1)
                            q_j = np.minimum(self.W[h + 1] @ phi_j + bonus_j, float(self.config.horizon))
                            bootstrap += joint_prob[j] * q_j
                else:
                    bootstrap = np.zeros(n, dtype=np.float64)

                target = r_vec + bootstrap  # shape (n,)

                if self.config.diagonal_lambda:
                    flat = self.traj_feat_idx[h][tau]
                    for i in range(n):
                        rhs[i, flat] += target[i]
                else:
                    phi_tau = self.trajectory_phi[h][tau]
                    for i in range(n):
                        rhs[i] += phi_tau * target[i]

            # ---- Solve (Λ_h)^{-1} rhs[i] -> w^{i,k}_h ----
            if self.config.diagonal_lambda:
                # Diagonal solve: w = rhs / lambda_diag
                for i in range(n):
                    self.W[h, i] = rhs[i] / self.lambda_diag[h]
            else:
                Lh = self.Lchol[h]
                for i in range(n):
                    try:
                        y = np.linalg.solve(Lh, rhs[i])
                        self.W[h, i] = np.linalg.solve(Lh.T, y)
                    except np.linalg.LinAlgError:
                        self.W[h, i] = np.linalg.lstsq(self.Lambda[h], rhs[i], rcond=None)[0]

    # ------------------------------------------------------------------
    # Parameter scheduling
    # ------------------------------------------------------------------

    def decay_parameters(self, episode_idx, n_episodes):
        cfg = self.config
        if cfg.epsilon_schedule == "constant":
            cfg.epsilon_robust = float(cfg.epsilon_robust_initial)
        elif cfg.epsilon_schedule == "linear":
            cfg.epsilon_robust = _linear_schedule(
                cfg.epsilon_robust_initial, 0.0, episode_idx, n_episodes
            )
        elif cfg.epsilon_schedule == "exponential":
            cfg.epsilon_robust = float(cfg.epsilon_robust_initial) * (cfg.decay_rate ** episode_idx)
        else:
            raise ValueError(f"Unknown epsilon_schedule: {cfg.epsilon_schedule}")

        decay_eps = max(1, int(n_episodes * cfg.action_epsilon_decay_fraction))
        cfg.epsilon_explore = _linear_schedule(
            cfg.action_epsilon_start,
            cfg.action_epsilon_end,
            min(int(episode_idx), decay_eps - 1),
            decay_eps,
        )

    def get_timing_summary(self):
        times = np.asarray(self.sre_solve_times)
        if times.size == 0:
            return {"count": 0}
        return {
            "count": int(times.size),
            "mean_seconds": float(times.mean()),
            "min_seconds": float(times.min()),
            "max_seconds": float(times.max()),
            "std_seconds": float(times.std()),
        }

    def close(self):
        if hasattr(self.sre_solver, "close"):
            self.sre_solver.close()
