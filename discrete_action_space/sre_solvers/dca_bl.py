"""DCA-BL solver for finite-action N-player SRE stage games.

Implements the Difference-of-Convex Algorithm for Bilinear terms (DCA-BL)
from Gabriel et al. (2024) "A Heuristic for Complementarity Problems Using
Difference of Convex Functions" and Flocco & Gabriel (2026)
"Efficient Warm-Start Strategies for Nash-based LCPs".

The SRE complementarity conditions
    0 ≤ s_ia := (v_i - u_rob_ia) ⊥ delta_ia ≥ 0
are DC-decomposed as s_ia * delta_ia = U_ia² - W_ia² using the
transformation N = [[1, 1], [1, -1]]:
    s_ia = U_ia + W_ia,   delta_ia = U_ia - W_ia   (alpha = beta = 1).

At each iteration a convex QCQP is solved by Gurobi. The robust action value
u_rob_ia is linearised around the current iterate (zero-order), matching the
GDCA2 pattern from the paper.

References
----------
Algorithm 1 (GDCA2) in Gabriel et al. (2024), equations (11a-11j).
Appendix A in Flocco & Gabriel (2026).
"""

import time

import numpy as np

try:
    import gurobipy  # noqa: F401 — availability check; functions import locally
    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False

from .base import SreSolveResult, SreStageGameSolver, _empty_duration_summary, _normalize_policy
from .baseline_nplayer import IterativeNPlayerSreSolver
from .nplayer_common import (
    _expected_nominal_values,
    _normalize_nplayer_policies,
    _solution_dict_from_policies,
    _uniform_nplayer_policies,
    robust_action_values,
    robust_exploitability,
    validate_nplayer_q_tensor,
)


class DcaBlNPlayerSreSolver(SreStageGameSolver):
    """DCA-BL solver for N-player finite-action SRE stage games.

    Falls back to IterativeNPlayerSreSolver when gurobipy is not installed
    or no licence is available. The metadata field ``subproblem_backend``
    distinguishes the two paths ("gurobi" vs "python_fallback").
    """

    name = "dca_bl_nplayer"

    def __init__(
        self,
        *,
        max_iter: int = 100,
        tol: float = 1e-5,
        rho_init: float = 1.0,
        delta1: float = 1.0,
        delta2: float = 1.0,
        damping: float = 0.35,
        temperature: float = 0.02,
        random_seed=None,
    ):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rho_init = float(rho_init)
        self.delta1 = float(delta1)
        self.delta2 = float(delta2)
        self.rng = np.random.default_rng(random_seed)
        self._fallback = IterativeNPlayerSreSolver(
            max_iter=max_iter,
            tol=tol,
            damping=damping,
            temperature=temperature,
            random_seed=random_seed,
        )
        self.solve_durations: list[float] = []

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Multi-start generation (delegates to baseline)
    # ------------------------------------------------------------------

    def _initial_policies(self, q_tensor, num_repeats: int, include_pure_starts: bool):
        return self._fallback._initial_policies(q_tensor, num_repeats, include_pure_starts)

    # ------------------------------------------------------------------
    # DCA-BL inner iteration
    # ------------------------------------------------------------------

    def _init_dc_vars(self, policies, u_rob: list[np.ndarray]) -> tuple:
        """Initialise (U, W, v) from current policies and robust values.

        With N = [[1,1],[1,-1]]:  s = U+W,  delta = U-W
        → U = (s + delta)/2,  W = (s - delta)/2.
        """
        num_agents = len(policies)
        v = np.array([float(np.max(u_rob[i])) for i in range(num_agents)])
        U, W = [], []
        for i in range(num_agents):
            delta_i = np.asarray(policies[i], dtype=np.float64)
            s_i = np.maximum(v[i] - u_rob[i], 0.0)
            U.append((s_i + delta_i) / 2.0)
            W.append((s_i - delta_i) / 2.0)
        return U, W, v

    def _run_dca_bl(
        self,
        q_tensor: np.ndarray,
        epsilon: float,
        start_policies: list,
    ) -> tuple:
        """Run DCA-BL from one starting policy profile.

        Returns (policies, total_iterations, t_final, lambda_1_norm_last).
        """
        action_sizes = list(q_tensor.shape[:-1])
        num_agents = len(action_sizes)
        q_lo = float(q_tensor.min())
        q_hi = float(q_tensor.max())

        policies = _normalize_nplayer_policies(start_policies, action_sizes)
        u_rob = [robust_action_values(q_tensor, policies, epsilon, i) for i in range(num_agents)]
        U, W, v = self._init_dc_vars(policies, u_rob)

        rho_k = self.rho_init
        t_final = float("inf")
        lambda_norm = self.delta1

        for iteration in range(self.max_iter):
            # Linearisation point: robust values at current iterate (zero-order)
            u_rob_k = [robust_action_values(q_tensor, policies, epsilon, i) for i in range(num_agents)]

            try:
                policies_new, v_new, U_new, W_new, t_k, lambda_norm = _solve_qcqp(
                    action_sizes=action_sizes,
                    num_agents=num_agents,
                    q_lo=q_lo,
                    q_hi=q_hi,
                    u_rob_k=u_rob_k,
                    U_k=U,
                    W_k=W,
                    rho_k=rho_k,
                )
            except Exception:
                break

            # Penalty parameter update (Algorithm 1, step 4)
            z_old = np.concatenate([np.asarray(p) for p in policies] + [np.atleast_1d(vi) for vi in v])
            z_new = np.concatenate([np.asarray(p) for p in policies_new] + [np.atleast_1d(vi) for vi in v_new])
            step = float(np.linalg.norm(z_new - z_old))
            r_k = min(1.0 / step if step > 1e-12 else float("inf"), lambda_norm + self.delta1)
            if rho_k < r_k:
                rho_k += self.delta2

            policies, v, U, W = policies_new, v_new, U_new, W_new
            t_final = t_k

            if t_k <= self.tol and step <= self.tol:
                return policies, iteration + 1, t_final, lambda_norm

        return policies, self.max_iter, t_final, lambda_norm

    # ------------------------------------------------------------------
    # Core solve
    # ------------------------------------------------------------------

    def _solve_impl(
        self,
        q_tensor: np.ndarray,
        epsilon: float,
        *,
        num_repeats: int = 20,
        round_digits: int = 4,
        include_pure_starts: bool = True,
    ) -> SreSolveResult:
        if not _GUROBI_AVAILABLE or not _gurobi_licensed():
            result = self._fallback._solve_impl(
                q_tensor, epsilon,
                num_repeats=num_repeats,
                round_digits=round_digits,
                include_pure_starts=include_pure_starts,
            )
            result.metadata.update({
                "solver": self.name,
                "algorithm_family": "dca_bl",
                "dc_decomposition": "bilinear_complementarity_quadratic_difference",
                "subproblem_backend": "python_fallback",
                "dca_iterations": result.metadata.get("iterations", 0),
            })
            return result

        q_tensor = validate_nplayer_q_tensor(q_tensor)
        starts = self._initial_policies(q_tensor, num_repeats, include_pure_starts)

        best_policies = None
        best_meta: dict = {}
        best_score = float("inf")
        total_dca_iters = 0

        for start_policies in starts:
            policies, iters, t_final, _ = self._run_dca_bl(q_tensor, epsilon, start_policies)
            total_dca_iters += iters

            exp, player_gaps, robust_vals = robust_exploitability(q_tensor, policies, epsilon)
            nominal = _expected_nominal_values(q_tensor, policies)
            robust_policy_vals = [
                float(np.asarray(policies[i]) @ robust_vals[i])
                for i in range(len(policies))
            ]
            score = (float(exp), -float(np.sum(nominal)))
            if score[0] < best_score:
                best_score = score[0]
                best_policies = policies
                best_meta = {
                    "robust_exploitability": float(exp),
                    "player_robust_gaps": [float(g) for g in player_gaps],
                    "robust_policy_values": robust_policy_vals,
                    "nominal_values": [float(vi) for vi in nominal],
                    "joint_nominal_welfare": float(np.sum(nominal)),
                    "t_final": float(t_final),
                }

        if best_policies is None:
            best_policies = _uniform_nplayer_policies(q_tensor)
            exp, player_gaps, robust_vals = robust_exploitability(q_tensor, best_policies, epsilon)
            nominal = _expected_nominal_values(q_tensor, best_policies)
            best_meta = {
                "robust_exploitability": float(exp),
                "player_robust_gaps": [float(g) for g in player_gaps],
                "robust_policy_values": [float(np.asarray(best_policies[i]) @ robust_vals[i]) for i in range(len(best_policies))],
                "nominal_values": [float(vi) for vi in nominal],
                "joint_nominal_welfare": float(np.sum(nominal)),
                "t_final": float("inf"),
            }

        success = bool(best_meta["robust_exploitability"] <= max(10.0 * self.tol, 1e-4))
        solution = _solution_dict_from_policies(best_policies, round_digits=round_digits)

        return SreSolveResult(
            policies=best_policies,
            solutions=[solution],
            utilities_sr=[best_meta["robust_policy_values"]],
            utilities_nominal=[best_meta["nominal_values"]],
            success=success,
            message="" if success else "DCA-BL returned best approximate SRE candidate.",
            metadata={
                "solver": self.name,
                "algorithm_family": "dca_bl",
                "dc_decomposition": "bilinear_complementarity_quadratic_difference",
                "subproblem_backend": "gurobi",
                "dca_iterations": total_dca_iters,
                "epsilon": float(epsilon),
                "num_agents": int(q_tensor.shape[-1]),
                "action_sizes": [int(s) for s in q_tensor.shape[:-1]],
                "num_starts": len(starts),
                "converged_to_tolerance": success,
                **best_meta,
            },
        )

    def solve(
        self,
        q_tensor,
        epsilon: float,
        *,
        num_repeats: int = 20,
        round_digits: int = 4,
        include_pure_starts: bool = True,
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


# ---------------------------------------------------------------------------
# Gurobi QCQP subproblem (module-level for clarity)
# ---------------------------------------------------------------------------

def _solve_qcqp(
    *,
    action_sizes: list,
    num_agents: int,
    q_lo: float,
    q_hi: float,
    u_rob_k: list,
    U_k: list,
    W_k: list,
    rho_k: float,
) -> tuple:
    """Solve one DCA-BL convex QCQP subproblem (eq. 11, Gabriel et al.).

    Variables: delta_ia ∈ [0,1], v_i ∈ [q_lo, q_hi], U_ia ∈ ℝ, W_ia ∈ ℝ, t ≥ 0.

    Linking equalities (N = [[1,1],[1,-1]]):
        delta_ia = U_ia - W_ia
        s_ia     = U_ia + W_ia    (s_ia ≡ v_i - u_rob_ia, the slack)

    Linearised best-response constraints (eqs 11b/11c, with H^1 = u_rob^k const):
        v_i - u_rob_k_ia - (U_ia + W_ia) ≤ t     [11b]
        (U_ia + W_ia) + u_rob_k_ia - v_i  ≤ t     [11c]

    Bilinear DC constraints (eqs 11e/11f, alpha = beta = 1):
        U_ia² - 2*W^k_ia * W_ia + (W^k_ia)² ≤ t  [11e]
        W_ia² - 2*U^k_ia * U_ia + (U^k_ia)² ≤ t  [11f]

    Simplex: sum_a delta_ia = 1 for each i.

    Returns (policies, v, U_new, W_new, t_val, lambda_1norm).
    """
    import gurobipy as gp
    from gurobipy import GRB
    from ._gurobi_backend import _get_env

    env = _get_env()
    with gp.Model(env=env) as m:
        m.Params.OutputFlag = 0
        m.Params.NonConvex = 0   # convex QCQP — Gurobi must recognise it

        t = m.addVar(lb=0.0, name="t")

        delta = [m.addVars(action_sizes[i], lb=0.0, ub=1.0, name=f"d{i}") for i in range(num_agents)]
        v_var = m.addVars(num_agents, lb=q_lo, ub=q_hi, name="v")
        U_var = [m.addVars(action_sizes[i], lb=-GRB.INFINITY, name=f"U{i}") for i in range(num_agents)]
        W_var = [m.addVars(action_sizes[i], lb=-GRB.INFINITY, name=f"W{i}") for i in range(num_agents)]

        m.setObjective(rho_k * t, GRB.MINIMIZE)

        lin_constrs = []   # collect linear constraints to read Pi later

        for i in range(num_agents):
            A = action_sizes[i]

            # Simplex
            c = m.addConstr(gp.quicksum(delta[i][a] for a in range(A)) == 1.0)
            lin_constrs.append(c)

            for a in range(A):
                u_k = float(u_rob_k[i][a])
                Uk = float(U_k[i][a])
                Wk = float(W_k[i][a])

                # Linking: delta = U - W
                c = m.addConstr(delta[i][a] - U_var[i][a] + W_var[i][a] == 0.0)
                lin_constrs.append(c)

                # 11b: v_i - u_k - (U + W) ≤ t
                c = m.addConstr(v_var[i] - u_k - U_var[i][a] - W_var[i][a] <= t)
                lin_constrs.append(c)

                # 11c: (U + W) + u_k - v_i ≤ t
                c = m.addConstr(U_var[i][a] + W_var[i][a] + u_k - v_var[i] <= t)
                lin_constrs.append(c)

                # 11e: U² - 2*Wk*W + Wk² ≤ t
                m.addQConstr(
                    U_var[i][a] * U_var[i][a]
                    - 2.0 * Wk * W_var[i][a]
                    + Wk * Wk - t <= 0.0
                )

                # 11f: W² - 2*Uk*U + Uk² ≤ t
                m.addQConstr(
                    W_var[i][a] * W_var[i][a]
                    - 2.0 * Uk * U_var[i][a]
                    + Uk * Uk - t <= 0.0
                )

        m.optimize()

        if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Gurobi QCQP status {m.Status}")

        # Extract iterate
        policies_out, U_out, W_out = [], [], []
        for i in range(num_agents):
            A = action_sizes[i]
            d_arr = np.array([delta[i][a].X for a in range(A)], dtype=np.float64)
            d_arr = np.clip(d_arr, 0.0, 1.0)
            p = _normalize_policy(d_arr)
            if p is None:
                p = np.full(A, 1.0 / A, dtype=np.float64)
            policies_out.append(p)

            U_arr = np.array([U_var[i][a].X for a in range(A)], dtype=np.float64)
            W_arr = np.array([W_var[i][a].X for a in range(A)], dtype=np.float64)
            U_out.append(U_arr)
            W_out.append(W_arr)

        v_out = np.array([v_var[k].X for k in range(num_agents)], dtype=np.float64)
        t_val = float(t.X)

        # L1 norm of linear constraint dual variables (approximates lambda_norm)
        lambda_1norm = sum(abs(c.Pi) for c in lin_constrs if abs(c.Pi) < 1e20)

        return policies_out, v_out, U_out, W_out, t_val, lambda_1norm


# ---------------------------------------------------------------------------
# Licence check (run once, cache result)
# ---------------------------------------------------------------------------

_GUROBI_LICENSED: bool | None = None


def _gurobi_licensed() -> bool:
    global _GUROBI_LICENSED
    if _GUROBI_LICENSED is not None:
        return _GUROBI_LICENSED
    try:
        import gurobipy as _gp
        from ._gurobi_backend import _get_env
        _get_env()
        with _gp.Model(env=_get_env()):
            pass
        _GUROBI_LICENSED = True
    except Exception:
        _GUROBI_LICENSED = False
    return _GUROBI_LICENSED
