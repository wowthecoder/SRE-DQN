import importlib
import time

import numpy as np

try:
    from path_solver import build_robust_bimatrix_lcp
except ImportError:  # Package import from repository root.
    from ..path_solver import build_robust_bimatrix_lcp

from .base import (
    SreSolveResult,
    SreStageGameSolver,
    _empty_duration_summary,
    _uniform_policies,
    validate_bimatrix_q_tensor,
)
from .path_c import _select_best_solution, _solution_from_lcp_vector


class LemkeLcpBimatrixSreSolver(SreStageGameSolver):
    name = "lemkelcp"

    def __init__(self):
        self._lemke = self._load_solver()
        self.solve_time_count = 0
        self.solve_time_sum = 0.0
        self.solve_time_sumsq = 0.0
        self.solve_time_min = None
        self.solve_time_max = None

    @staticmethod
    def _load_solver():
        try:
            module = importlib.import_module("lemkelcp")
        except ImportError as exc:
            raise ImportError(
                "lemkelcp is required for LemkeLcpBimatrixSreSolver. "
                "Install it with `pip install lemkelcp==0.1`."
            ) from exc

        candidate = getattr(module, "lemkelcp", None)
        if callable(candidate):
            return candidate
        if candidate is not None and callable(getattr(candidate, "lemkelcp", None)):
            return candidate.lemkelcp

        try:
            submodule = importlib.import_module("lemkelcp.lemkelcp")
            candidate = getattr(submodule, "lemkelcp", None)
        except ImportError:
            candidate = None
        if callable(candidate):
            return candidate

        raise ImportError(
            "Installed lemkelcp package does not expose a callable lemkelcp solver."
        )

    def _record_solve_time(self, duration):
        self.solve_time_count += 1
        self.solve_time_sum += duration
        self.solve_time_sumsq += duration * duration
        if self.solve_time_min is None or duration < self.solve_time_min:
            self.solve_time_min = duration
        if self.solve_time_max is None or duration > self.solve_time_max:
            self.solve_time_max = duration

    def get_solve_time_summary(self):
        count = self.solve_time_count
        if count == 0:
            return _empty_duration_summary()

        mean = self.solve_time_sum / count
        variance = max(self.solve_time_sumsq / count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        return {
            "count": int(count),
            "mean_seconds": float(mean),
            "min_seconds": float(self.solve_time_min),
            "max_seconds": float(self.solve_time_max),
            "std_seconds": std,
            "mean_microseconds": float(mean * 1_000_000.0),
            "min_microseconds": float(self.solve_time_min * 1_000_000.0),
            "max_microseconds": float(self.solve_time_max * 1_000_000.0),
            "std_microseconds": float(std * 1_000_000.0),
        }

    @staticmethod
    def _parse_result(result):
        if isinstance(result, dict):
            if "z" in result:
                z = result["z"]
            else:
                z = result.get("solution")
            status = result.get("status") or result.get("exit_code")
            message = result.get("message") or result.get("exit_string") or ""
            return z, status, message

        if isinstance(result, tuple):
            z = result[0] if len(result) > 0 else None
            status = result[1] if len(result) > 1 else None
            message = result[2] if len(result) > 2 else ""
            return z, status, message

        return result, None, ""

    @staticmethod
    def _lcp_conditions_hold(M, q, z, tol=1e-5):
        w = M @ z + q
        return (
            np.all(z >= -tol)
            and np.all(w >= -tol)
            and float(np.max(np.abs(z * w))) <= tol
        )

    def solve(
        self,
        q_tensor,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
    ):
        q_tensor = validate_bimatrix_q_tensor(q_tensor)
        lcp = build_robust_bimatrix_lcp(q_tensor[:, :, 0], q_tensor[:, :, 1], epsilon)
        M = np.asarray(lcp["M"], dtype=np.float64)
        q = np.asarray(lcp["q"], dtype=np.float64)

        start = time.perf_counter()
        try:
            result = self._lemke(M, q)
        except Exception as exc:
            self._record_solve_time(time.perf_counter() - start)
            return SreSolveResult(
                policies=[],
                solutions=[],
                utilities_sr=[],
                utilities_nominal=[],
                success=False,
                message=str(exc),
                metadata={
                    "solver": self.name,
                    "num_repeats_ignored": int(num_repeats),
                    "include_pure_starts_ignored": bool(include_pure_starts),
                },
            )
        finally:
            if "result" in locals():
                self._record_solve_time(time.perf_counter() - start)

        z, status, message = self._parse_result(result)
        if z is None:
            return SreSolveResult(
                policies=[],
                solutions=[],
                utilities_sr=[],
                utilities_nominal=[],
                success=False,
                message="lemkelcp did not return a solution vector.",
                metadata={"solver": self.name, "raw_status": status},
            )

        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.shape[0] != q.shape[0] or not self._lcp_conditions_hold(M, q, z):
            return SreSolveResult(
                policies=[],
                solutions=[],
                utilities_sr=[],
                utilities_nominal=[],
                success=False,
                message=str(message or "lemkelcp solution failed LCP validation."),
                metadata={
                    "solver": self.name,
                    "raw_status": status,
                    "num_repeats_ignored": int(num_repeats),
                    "include_pure_starts_ignored": bool(include_pure_starts),
                },
            )

        sol, nominal = _solution_from_lcp_vector(z, lcp, round_digits=round_digits)
        if sol is None:
            return SreSolveResult(
                policies=[],
                solutions=[],
                utilities_sr=[],
                utilities_nominal=[],
                success=False,
                message="lemkelcp solution did not contain valid mixed strategies.",
                metadata={
                    "solver": self.name,
                    "raw_status": status,
                    "num_repeats_ignored": int(num_repeats),
                    "include_pure_starts_ignored": bool(include_pure_starts),
                },
            )

        policies = _select_best_solution(q_tensor, [sol])
        return SreSolveResult(
            policies=policies or _uniform_policies(q_tensor),
            solutions=[sol],
            utilities_sr=[nominal],
            utilities_nominal=[nominal],
            success=policies is not None,
            message="" if policies is not None else "No valid policy extracted.",
            metadata={
                "solver": self.name,
                "raw_status": status,
                "raw_message": str(message),
                "num_repeats_ignored": int(num_repeats),
                "include_pure_starts_ignored": bool(include_pure_starts),
            },
        )
