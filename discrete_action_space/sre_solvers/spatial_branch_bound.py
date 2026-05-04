"""Spatial Branch-and-Bound (SBB) solver for finite-action N-player SRE stage games.

Two-stage algorithm following the SBB paper:
  Stage 1: SLSQP NLP local warm-start, minimising the robust exploitability directly.
  Stage 2: Best-first branch-and-bound tree with McCormick LP relaxations (scipy HiGHS).

The McCormick LP relaxes the bilinear products w_ia ≈ δ_ia * v_i in formulation (R):
    min ϖ  s.t.  δ_ia*(v_i - u_rob_ia) ≤ ϖ,  v_i ≥ u_rob_ia,  Σ_a δ_ia = 1.

Reference: Spatial Branch-and-Bound for Computing Multiplayer Nash Equilibrium.
"""

import heapq
import time

import numpy as np
from scipy.optimize import linprog, minimize

from .base import SreSolveResult, SreStageGameSolver, _empty_duration_summary
from .baseline_nplayer import IterativeNPlayerSreSolver
from .nplayer_common import (
    _expected_nominal_values,
    _solution_dict_from_policies,
    _uniform_nplayer_policies,
    robust_action_values,
    robust_exploitability,
    validate_nplayer_q_tensor,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_offsets(action_sizes):
    offsets = [0] * len(action_sizes)
    for i in range(1, len(action_sizes)):
        offsets[i] = offsets[i - 1] + action_sizes[i - 1]
    return offsets


def _policies_from_flat(x_flat, action_sizes, offsets):
    pols = []
    for i, size in enumerate(action_sizes):
        d = np.clip(x_flat[offsets[i]:offsets[i] + size], 0.0, 1.0)
        s = float(d.sum())
        pols.append(d / s if s > 1e-12 else np.full(size, 1.0 / size))
    return pols


def _stage1_nlp(q_tensor, epsilon, start_policies, action_sizes, offsets, max_iter, tol):
    """SLSQP minimising robust exploitability on the product of simplices."""
    def objective(x):
        pols = _policies_from_flat(x, action_sizes, offsets)
        gap, _, _ = robust_exploitability(q_tensor, pols, epsilon)
        return gap

    constraints = []
    for off, sz in zip(offsets, action_sizes):
        constraints.append({
            'type': 'eq',
            'fun': lambda x, o=off, s=sz: float(x[o:o + s].sum()) - 1.0,
        })

    bounds = [(0.0, 1.0)] * sum(action_sizes)
    x0 = np.clip(
        np.concatenate([np.asarray(p, dtype=np.float64) for p in start_policies]),
        0.0, 1.0,
    )

    res = minimize(
        objective, x0,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol * 0.1, 'disp': False},
    )
    pols = _policies_from_flat(res.x, action_sizes, offsets)
    obj = float(res.fun) if np.isfinite(res.fun) else float('inf')
    return pols, obj


def _u_rob_midpoint(q_tensor, epsilon, action_sizes, delta_lo, delta_hi):
    """Evaluate robust action values at the midpoint of the current box."""
    N = len(action_sizes)
    mid_pols = []
    for i in range(N):
        mid = np.clip((delta_lo[i] + delta_hi[i]) * 0.5, 0.0, 1.0)
        s = float(mid.sum())
        mid_pols.append(mid / s if s > 1e-12 else np.full(action_sizes[i], 1.0 / action_sizes[i]))
    return [robust_action_values(q_tensor, mid_pols, epsilon, i) for i in range(N)]


def _solve_mccormick_lp(action_sizes, N, n_delta, offsets,
                        delta_lo, delta_hi, v_lo, v_hi, u_rob):
    """McCormick LP relaxation at a B&B node.

    Variables: [δ (n_delta), v (N), w (n_delta), ϖ (1)]
    w_ia is the McCormick variable approximating δ_ia * v_i.
    Objective: minimise ϖ.
    """
    idx_v = n_delta
    idx_w = n_delta + N
    idx_vp = 2 * n_delta + N
    n_vars = 2 * n_delta + N + 1

    c = np.zeros(n_vars)
    c[idx_vp] = 1.0

    # Equality: simplex per player
    A_eq_rows, b_eq = [], []
    for i in range(N):
        row = np.zeros(n_vars)
        row[offsets[i]:offsets[i] + action_sizes[i]] = 1.0
        A_eq_rows.append(row)
        b_eq.append(1.0)

    # Inequalities
    A_ub_rows, b_ub = [], []

    def add(row, rhs):
        A_ub_rows.append(row)
        b_ub.append(rhs)

    for i in range(N):
        for a in range(action_sizes[i]):
            flat = offsets[i] + a
            w_idx = idx_w + flat
            u = float(u_rob[i][a])
            dlo = float(delta_lo[i][a])
            dhi = float(delta_hi[i][a])
            vlo = float(v_lo[i])
            vhi = float(v_hi[i])

            # v_i >= u_rob_ia
            r = np.zeros(n_vars)
            r[idx_v + i] = -1.0
            add(r, -u)

            # complementarity residual: w_ia - u*δ_ia <= ϖ
            r = np.zeros(n_vars)
            r[w_idx] = 1.0
            r[flat] = -u
            r[idx_vp] = -1.0
            add(r, 0.0)

            # McCormick lower 1: w >= dlo*v + vlo*δ - dlo*vlo
            r = np.zeros(n_vars)
            r[w_idx] = -1.0
            r[idx_v + i] = dlo
            r[flat] = vlo
            add(r, dlo * vlo)

            # McCormick lower 2: w >= dhi*v + vhi*δ - dhi*vhi
            r = np.zeros(n_vars)
            r[w_idx] = -1.0
            r[idx_v + i] = dhi
            r[flat] = vhi
            add(r, dhi * vhi)

            # McCormick upper 1: w <= dhi*v + vlo*δ - dhi*vlo
            r = np.zeros(n_vars)
            r[w_idx] = 1.0
            r[idx_v + i] = -dhi
            r[flat] = -vlo
            add(r, -dhi * vlo)

            # McCormick upper 2: w <= dlo*v + vhi*δ - dlo*vhi
            r = np.zeros(n_vars)
            r[w_idx] = 1.0
            r[idx_v + i] = -dlo
            r[flat] = -vhi
            add(r, -dlo * vhi)

    A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
    b_ub_arr = np.array(b_ub, dtype=np.float64)
    A_eq_arr = np.array(A_eq_rows, dtype=np.float64)
    b_eq_arr = np.array(b_eq, dtype=np.float64)

    bounds = []
    for i in range(N):
        for a in range(action_sizes[i]):
            bounds.append((float(delta_lo[i][a]), float(delta_hi[i][a])))
    for i in range(N):
        bounds.append((float(v_lo[i]), float(v_hi[i])))
    for i in range(N):
        for a in range(action_sizes[i]):
            dlo_, dhi_ = float(delta_lo[i][a]), float(delta_hi[i][a])
            vlo_, vhi_ = float(v_lo[i]), float(v_hi[i])
            prods = [dlo_ * vlo_, dlo_ * vhi_, dhi_ * vlo_, dhi_ * vhi_]
            bounds.append((min(prods), max(prods)))
    bounds.append((0.0, None))

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub_arr, A_eq=A_eq_arr, b_eq=b_eq_arr,
                      bounds=bounds, method='highs')
    except Exception:
        return None, float('inf')

    if res.status != 0:
        return None, float('inf')
    return res.x, float(res.fun)


def _mccormick_gap(lp_x, action_sizes, N, offsets, n_delta):
    """Return (max McCormick gap, branch player, branch action)."""
    idx_v = n_delta
    idx_w = n_delta + N
    max_gap = -1.0
    bi, ba = 0, 0
    for i in range(N):
        v_i = float(lp_x[idx_v + i])
        for a in range(action_sizes[i]):
            flat = offsets[i] + a
            gap = abs(float(lp_x[idx_w + flat]) - float(lp_x[flat]) * v_i)
            if gap > max_gap:
                max_gap = gap
                bi, ba = i, a
    return max_gap, bi, ba


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class SpatialBranchBoundNPlayerSreSolver(SreStageGameSolver):
    """Spatial Branch-and-Bound solver for N-player finite-action SRE.

    Stage 1: SLSQP NLP warm-start (minimises robust exploitability).
    Stage 2: Best-first B&B with McCormick LP relaxations (scipy HiGHS).
    """

    name = "sbb_nplayer"

    def __init__(
        self,
        *,
        max_iter: int = 250,
        tol: float = 1e-5,
        max_nodes: int = 256,
        damping: float = 0.35,
        temperature: float = 0.02,
        random_seed=None,
    ):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.max_nodes = int(max_nodes)
        self.rng = np.random.default_rng(random_seed)
        self._fallback = IterativeNPlayerSreSolver(
            max_iter=max_iter,
            tol=tol,
            damping=damping,
            temperature=temperature,
            random_seed=random_seed,
        )
        self.solve_durations: list[float] = []

    def _record_duration(self, d: float) -> None:
        self.solve_durations.append(float(d))

    def get_solve_time_summary(self) -> dict:
        if not self.solve_durations:
            return _empty_duration_summary()
        arr = np.asarray(self.solve_durations, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean_seconds": float(np.mean(arr)),
            "min_seconds": float(np.min(arr)),
            "max_seconds": float(np.max(arr)),
            "std_seconds": float(np.std(arr)),
            "mean_microseconds": float(np.mean(arr) * 1e6),
            "min_microseconds": float(np.min(arr) * 1e6),
            "max_microseconds": float(np.max(arr) * 1e6),
            "std_microseconds": float(np.std(arr) * 1e6),
        }

    def _initial_policies(self, q_tensor, num_repeats, include_pure_starts):
        return self._fallback._initial_policies(q_tensor, num_repeats, include_pure_starts)

    def _run_sbb(self, q_tensor, epsilon, start_policies,
                 action_sizes, N, n_delta, offsets, q_lo, q_hi):
        """Stage 2 best-first B&B returning (best_policies, nodes_explored, gap)."""
        delta_lo = [np.zeros(action_sizes[i]) for i in range(N)]
        delta_hi = [np.ones(action_sizes[i]) for i in range(N)]
        v_lo_root = np.full(N, q_lo)
        v_hi_root = np.full(N, q_hi)

        inc_obj, _, _ = robust_exploitability(q_tensor, start_policies, epsilon)
        inc_obj = max(0.0, float(inc_obj))
        inc_pols = [p.copy() for p in start_policies]

        node_id = [0]

        def nid():
            node_id[0] += 1
            return node_id[0]

        # heap entries: (lb, id, delta_lo, delta_hi, v_lo, v_hi)
        heap: list = [(0.0, nid(),
                       [d.copy() for d in delta_lo],
                       [d.copy() for d in delta_hi],
                       v_lo_root.copy(), v_hi_root.copy())]
        heapq.heapify(heap)

        nodes_explored = 0

        while heap and nodes_explored < self.max_nodes:
            _, _, d_lo, d_hi, v_lo, v_hi = heapq.heappop(heap)
            nodes_explored += 1

            u_rob = _u_rob_midpoint(q_tensor, epsilon, action_sizes, d_lo, d_hi)

            # Tighten v lower bounds: v_i >= max_a u_rob_ia
            v_lo_eff = v_lo.copy()
            v_hi_eff = v_hi.copy()
            for i in range(N):
                v_lo_eff[i] = max(float(v_lo[i]), float(np.max(u_rob[i])))
            if np.any(v_lo_eff > v_hi_eff + self.tol):
                continue  # infeasible box

            lp_x, lp_obj = _solve_mccormick_lp(
                action_sizes, N, n_delta, offsets,
                d_lo, d_hi, v_lo_eff, v_hi_eff, u_rob,
            )

            if lp_obj >= inc_obj - self.tol:
                # Best-first: all remaining nodes also have lb >= lp_obj >= inc_obj
                if not heap or heap[0][0] >= inc_obj - self.tol:
                    break
                continue

            if lp_x is None:
                continue

            mc_gap, bi, ba = _mccormick_gap(lp_x, action_sizes, N, offsets, n_delta)

            if mc_gap <= self.tol * 10:
                # LP solution is bilinear-tight: evaluate true objective
                lp_pols = _policies_from_flat(lp_x, action_sizes, offsets)
                true_obj, _, _ = robust_exploitability(q_tensor, lp_pols, epsilon)
                true_obj = float(true_obj)
                if true_obj < inc_obj:
                    inc_obj = true_obj
                    inc_pols = lp_pols
                if inc_obj <= self.tol:
                    break
                continue

            # Branch on delta[bi][ba] at midpoint
            dlo_ba = float(d_lo[bi][ba])
            dhi_ba = float(d_hi[bi][ba])
            if dhi_ba - dlo_ba < 1e-10:
                # Cannot branch further — treat as leaf
                lp_pols = _policies_from_flat(lp_x, action_sizes, offsets)
                true_obj, _, _ = robust_exploitability(q_tensor, lp_pols, epsilon)
                if float(true_obj) < inc_obj:
                    inc_obj = float(true_obj)
                    inc_pols = lp_pols
                continue

            d_mid = (dlo_ba + dhi_ba) * 0.5
            for lo_new, hi_new in [(dlo_ba, d_mid), (d_mid, dhi_ba)]:
                nd_lo = [d.copy() for d in d_lo]
                nd_hi = [d.copy() for d in d_hi]
                nd_lo[bi][ba] = lo_new
                nd_hi[bi][ba] = hi_new
                heapq.heappush(heap, (lp_obj, nid(), nd_lo, nd_hi,
                                      v_lo_eff.copy(), v_hi_eff.copy()))

        # Compute final optimality gap
        final_lb = heap[0][0] if heap else inc_obj
        gap = max(0.0, inc_obj - final_lb)
        return inc_pols, nodes_explored, gap

    def _solve_impl(
        self, q_tensor, epsilon,
        *, num_repeats=20, round_digits=4, include_pure_starts=True,
    ) -> SreSolveResult:
        q_tensor = validate_nplayer_q_tensor(q_tensor)
        action_sizes = list(q_tensor.shape[:-1])
        N = len(action_sizes)
        n_delta = sum(action_sizes)
        offsets = _make_offsets(action_sizes)
        q_lo = float(q_tensor.min())
        q_hi = float(q_tensor.max())

        starts = self._initial_policies(q_tensor, num_repeats, include_pure_starts)

        best_policies = None
        best_exp = float('inf')
        total_nodes = 0
        final_gap = float('inf')

        for start_pols in starts:
            # Stage 1: NLP warm-start
            try:
                s1_pols, _ = _stage1_nlp(
                    q_tensor, epsilon, start_pols, action_sizes, offsets,
                    self.max_iter, self.tol,
                )
            except Exception:
                s1_pols = start_pols

            # Stage 2: B&B
            try:
                sbb_pols, nodes, gap = self._run_sbb(
                    q_tensor, epsilon, s1_pols,
                    action_sizes, N, n_delta, offsets, q_lo, q_hi,
                )
            except Exception:
                sbb_pols = s1_pols
                nodes = 0
                gap = float('inf')

            total_nodes += nodes
            exp, _, _ = robust_exploitability(q_tensor, sbb_pols, epsilon)
            if float(exp) < best_exp:
                best_exp = float(exp)
                best_policies = [p.copy() for p in sbb_pols]
                final_gap = gap

        if best_policies is None:
            best_policies = _uniform_nplayer_policies(q_tensor)
            best_exp, _, _ = robust_exploitability(q_tensor, best_policies, epsilon)
            best_exp = float(best_exp)
            final_gap = float('inf')

        exp, player_gaps, robust_vals = robust_exploitability(q_tensor, best_policies, epsilon)
        nominal = _expected_nominal_values(q_tensor, best_policies)
        robust_policy_vals = [
            float(np.asarray(best_policies[i]) @ robust_vals[i]) for i in range(N)
        ]

        success = bool(float(exp) <= max(10.0 * self.tol, 1e-4))
        solution = _solution_dict_from_policies(best_policies, round_digits=round_digits)

        return SreSolveResult(
            policies=best_policies,
            solutions=[solution],
            utilities_sr=[robust_policy_vals],
            utilities_nominal=[[float(v) for v in nominal]],
            success=success,
            message="" if success else "SBB returned best approximate SRE candidate.",
            metadata={
                "solver": self.name,
                "algorithm_family": "spatial_branch_and_bound",
                "sbb_backend": "scipy_highs",
                "sbb_nodes": total_nodes,
                "sbb_gap": float(final_gap) if np.isfinite(final_gap) else -1.0,
                "epsilon": float(epsilon),
                "num_agents": N,
                "action_sizes": [int(s) for s in action_sizes],
                "num_starts": len(starts),
                "robust_exploitability": float(exp),
                "player_robust_gaps": [float(g) for g in player_gaps],
                "robust_policy_values": robust_policy_vals,
                "nominal_values": [float(v) for v in nominal],
                "joint_nominal_welfare": float(np.sum(nominal)),
                "converged_to_tolerance": success,
            },
        )

    def solve(
        self, q_tensor, epsilon,
        *, num_repeats=20, round_digits=4, include_pure_starts=True,
    ) -> SreSolveResult:
        t0 = time.perf_counter()
        try:
            return self._solve_impl(
                q_tensor, epsilon,
                num_repeats=num_repeats,
                round_digits=round_digits,
                include_pure_starts=include_pure_starts,
            )
        finally:
            self._record_duration(time.perf_counter() - t0)
