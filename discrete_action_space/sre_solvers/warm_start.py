"""Efficient Warm-Start solver for N-player SRE stage games.

Implements the warm-start strategy from Flocco & Gabriel (2026)
"Efficient Warm-Start Strategies for Nash-based LCPs":
  1. Run DCA-BL and SBB with halved iteration/node budgets to produce
     candidate policies.
  2. Feed each candidate as a warm start to the PATH MCP solver.
  3. Also run PATH with a cold start.
  4. Select the best result among all variants by robust exploitability.
  5. Fall back to the best of DCA-BL / SBB when PATH is unavailable.

Metadata key: ``selected_warm_start_source ∈ {"dca_bl", "sbb", "cold"}``.
"""

import time

import numpy as np

from .base import SreSolveResult, SreStageGameSolver, _empty_duration_summary
from .dca_bl import DcaBlNPlayerSreSolver
from .spatial_branch_bound import SpatialBranchBoundNPlayerSreSolver
from .nplayer_common import (
    _expected_nominal_values,
    _solution_dict_from_policies,
    robust_exploitability,
    validate_nplayer_q_tensor,
)


def _expl(result: SreSolveResult) -> float:
    return float(result.metadata.get("robust_exploitability", float("inf")))


class WarmStartNPlayerSreSolver(SreStageGameSolver):
    """Efficient warm-start solver: DCA-BL / SBB → PATH.

    DCA-BL and SBB are run with halved iteration/node budgets to produce
    warm-start policies that are fed to PATH for polishing. When PATH is
    unavailable, the best of DCA-BL and SBB is returned directly.
    """

    name = "warm_start_nplayer"

    def __init__(
        self,
        *,
        max_iter: int = 250,
        tol: float = 1e-5,
        damping: float = 0.35,
        temperature: float = 0.02,
        random_seed=None,
    ):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        half_iter = max(1, self.max_iter // 2)

        self._dca = DcaBlNPlayerSreSolver(
            max_iter=half_iter, tol=tol,
            damping=damping, temperature=temperature, random_seed=random_seed,
        )
        self._sbb = SpatialBranchBoundNPlayerSreSolver(
            max_iter=half_iter, tol=tol, max_nodes=128,
            damping=damping, temperature=temperature, random_seed=random_seed,
        )
        self._path = None  # PathMcpNPlayerSreSolver | None
        try:
            from .path_mcp_nplayer import PathMcpNPlayerSreSolver as _PMCP_init
            self._path = _PMCP_init()
        except Exception:
            pass

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

    def close(self) -> None:
        if self._path is not None:
            self._path.close()

    # ------------------------------------------------------------------
    # PATH warm-start helper
    # ------------------------------------------------------------------

    def _run_path_warmstart(
        self,
        q_tensor: np.ndarray,
        epsilon: float,
        warm_policies: list,
        round_digits: int,
    ) -> "SreSolveResult | None":
        """Run one PATH solve seeded with warm_policies. Returns None on failure."""
        if self._path is None:
            return None
        try:
            from .path_mcp_nplayer import PathMcpNPlayerSreSolver as _PMCP
            action_sizes = q_tensor.shape[:-1]
            index, opponent_data, _ = _PMCP._build_index(action_sizes)
            payoffs = [
                _PMCP._payoff_by_own_and_opponent_profile(
                    q_tensor, pid, opponent_data[pid][1]
                )
                for pid in range(len(action_sizes))
            ]
            z = self._path._make_start(
                index, action_sizes, opponent_data, policies=warm_policies
            )
            status, z_sol = self._path._solve_from_start(
                z, q_tensor, epsilon, index, opponent_data, payoffs
            )
        except Exception:
            return None

        if status not in (1, 2):
            return None

        policies = self._path._policies_from_z(z_sol, index, action_sizes)
        if policies is None:
            return None

        exp, player_gaps, robust_vals = robust_exploitability(q_tensor, policies, epsilon)
        nominal = _expected_nominal_values(q_tensor, policies)
        robust_policy_vals = [
            float(np.asarray(policies[i]) @ robust_vals[i])
            for i in range(len(policies))
        ]
        solution = _solution_dict_from_policies(policies, round_digits=round_digits)
        return SreSolveResult(
            policies=policies,
            solutions=[solution],
            utilities_sr=[robust_policy_vals],
            utilities_nominal=[[float(v) for v in nominal]],
            success=bool(float(exp) <= 1e-4),
            message="" if float(exp) <= 1e-4 else "PATH warm-start returned best candidate.",
            metadata={
                "solver": self.name,
                "algorithm_family": "efficient_warm_start",
                "robust_exploitability": float(exp),
                "player_robust_gaps": [float(g) for g in player_gaps],
                "robust_policy_values": robust_policy_vals,
                "nominal_values": [float(v) for v in nominal],
                "joint_nominal_welfare": float(np.sum(nominal)),
                "path_status": int(status),
            },
        )

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
        q_tensor = validate_nplayer_q_tensor(q_tensor)
        half_repeats = max(1, num_repeats // 2)

        # --- Stage A: DCA-BL warm-start ---
        t0 = time.perf_counter()
        dca_result = self._dca._solve_impl(
            q_tensor, epsilon,
            num_repeats=half_repeats,
            round_digits=round_digits,
            include_pure_starts=include_pure_starts,
        )
        t_dca = time.perf_counter() - t0

        # --- Stage B: SBB warm-start ---
        t0 = time.perf_counter()
        sbb_result = self._sbb._solve_impl(
            q_tensor, epsilon,
            num_repeats=half_repeats,
            round_digits=round_digits,
            include_pure_starts=include_pure_starts,
        )
        t_sbb = time.perf_counter() - t0

        # --- Stage C: PATH polishing (optional) ---
        t_path_dca = t_path_sbb = t_path_cold = 0.0
        path_dca_result: "SreSolveResult | None" = None
        path_sbb_result: "SreSolveResult | None" = None
        path_cold_result: "SreSolveResult | None" = None

        if self._path is not None:
            t0 = time.perf_counter()
            path_dca_result = self._run_path_warmstart(
                q_tensor, epsilon, dca_result.policies, round_digits
            )
            t_path_dca = time.perf_counter() - t0

            t0 = time.perf_counter()
            path_sbb_result = self._run_path_warmstart(
                q_tensor, epsilon, sbb_result.policies, round_digits
            )
            t_path_sbb = time.perf_counter() - t0

            t0 = time.perf_counter()
            try:
                cold = self._path.solve(
                    q_tensor, epsilon,
                    num_repeats=num_repeats,
                    round_digits=round_digits,
                    include_pure_starts=include_pure_starts,
                )
                path_cold_result = cold if cold.policies else None
            except Exception:
                path_cold_result = None
            t_path_cold = time.perf_counter() - t0

        # --- Stage D: Select best ---
        # Each entry: (source_label, result, time_seconds)
        candidates: list[tuple[str, SreSolveResult, float]] = [
            ("dca_bl", dca_result, t_dca),
            ("sbb",    sbb_result, t_sbb),
        ]
        if path_dca_result is not None:
            candidates.append(("dca_bl", path_dca_result, t_path_dca))
        if path_sbb_result is not None:
            candidates.append(("sbb",    path_sbb_result, t_path_sbb))
        if path_cold_result is not None:
            candidates.append(("cold",   path_cold_result, t_path_cold))

        best_source, best_result, _ = min(candidates, key=lambda x: _expl(x[1]))

        # --- Stage E: Assemble metadata ---
        variant_meta = [
            {
                "source": src,
                "robust_exploitability": _expl(r),
                "joint_nominal_welfare": float(r.metadata.get("joint_nominal_welfare", 0.0)),
                "time_seconds": float(t),
            }
            for src, r, t in candidates
        ]

        best_result.metadata.update({
            "solver": self.name,
            "algorithm_family": "efficient_warm_start",
            "selected_warm_start_source": best_source,
            "path_available": self._path is not None,
            "variants": variant_meta,
            "dca_exploitability": _expl(dca_result),
            "sbb_exploitability": _expl(sbb_result),
        })
        return best_result

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
