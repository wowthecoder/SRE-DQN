import importlib
from dataclasses import dataclass, replace
import multiprocessing as mp
import time

import numpy as np
from path_solver import build_robust_bimatrix_lcp

from .base import (
    SreSolveResult,
    SreStageGameSolver,
    _empty_duration_summary,
    _uniform_policies,
    validate_bimatrix_q_tensor,
)
from .path_c import _select_best_solution, _solution_from_lcp_vector


@dataclass(frozen=True)
class LemkeLcpBimatrixSreSolverConfig:
    pass


@dataclass(frozen=True)
class ProcessPoolLemkeLcpBimatrixSreSolverConfig:
    max_workers: int = 4
    start_method: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "max_workers", int(self.max_workers))
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive for process-pool SRE.")


class LemkeLcpBimatrixSreSolver(SreStageGameSolver):
    name = "lemkelcp"

    def __init__(self, config: LemkeLcpBimatrixSreSolverConfig | None = None, **overrides):
        if config is None:
            config = LemkeLcpBimatrixSreSolverConfig(**overrides)
        elif overrides:
            config = replace(config, **overrides)
        self.config = config
        self._lemke = self._load_solver()
        self.solve_time_count = 0
        self.solve_time_sum = 0.0
        self.solve_time_sumsq = 0.0
        self.solve_time_min = None
        self.solve_time_max = None

    @staticmethod
    def _patch_python3_tableau(submodule):
        tableau_cls = getattr(submodule, "lemketableau", None)
        if tableau_cls is None or getattr(tableau_cls, "_sre_dqn_py3_patch", False):
            return

        original_init = tableau_cls.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if isinstance(self.wPos, range):
                self.wPos = list(self.wPos)
            if isinstance(self.zPos, range):
                self.zPos = list(self.zPos)

        tableau_cls.__init__ = patched_init
        tableau_cls._sre_dqn_py3_patch = True

    @staticmethod
    def _load_solver():
        module = importlib.import_module("lemkelcp")
        submodule = importlib.import_module("lemkelcp.lemkelcp")
        LemkeLcpBimatrixSreSolver._patch_python3_tableau(submodule)

        candidate = getattr(module, "lemkelcp", None)
        if callable(candidate):
            return candidate
        if candidate is not None and callable(getattr(candidate, "lemkelcp", None)):
            return candidate.lemkelcp

        candidate = getattr(submodule, "lemkelcp", None)
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
            detail = str(message or status or "unknown status")
            return SreSolveResult(
                policies=[],
                solutions=[],
                utilities_sr=[],
                utilities_nominal=[],
                success=False,
                message=f"lemkelcp did not return a solution vector: {detail}",
                metadata={
                    "solver": self.name,
                    "raw_status": status,
                    "raw_message": str(message),
                    "num_repeats_ignored": int(num_repeats),
                    "include_pure_starts_ignored": bool(include_pure_starts),
                },
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


_LEMKE_POOL_SOLVER = None


def _lemke_pool_initializer():
    global _LEMKE_POOL_SOLVER
    _LEMKE_POOL_SOLVER = LemkeLcpBimatrixSreSolver()


def _lemke_pool_solve_task(payload):
    q_tensor, epsilon, num_repeats, round_digits, include_pure_starts = payload
    start = time.perf_counter()
    result = _LEMKE_POOL_SOLVER.solve(
        q_tensor,
        epsilon,
        num_repeats=num_repeats,
        round_digits=round_digits,
        include_pure_starts=include_pure_starts,
    )
    elapsed = time.perf_counter() - start
    result.metadata = dict(result.metadata)
    result.metadata["worker_sre_wall_seconds"] = float(elapsed)
    return result


class ProcessPoolLemkeLcpBimatrixSreSolver(SreStageGameSolver):
    name = "lemkelcp_pool"

    def __init__(
        self,
        config: ProcessPoolLemkeLcpBimatrixSreSolverConfig | None = None,
        **overrides,
    ):
        if config is None:
            config = ProcessPoolLemkeLcpBimatrixSreSolverConfig(**overrides)
        elif overrides:
            config = replace(config, **overrides)
        self.config = config
        self.solve_time_count = 0
        self.solve_time_sum = 0.0
        self.solve_time_sumsq = 0.0
        self.solve_time_min = None
        self.solve_time_max = None
        ctx = mp.get_context(self.config.start_method) if self.config.start_method else mp.get_context()
        self._pool = ctx.Pool(
            processes=self.config.max_workers,
            initializer=_lemke_pool_initializer,
        )

    def _record_solve_time(self, duration):
        self.solve_time_count += 1
        self.solve_time_sum += duration
        self.solve_time_sumsq += duration * duration
        if self.solve_time_min is None or duration < self.solve_time_min:
            self.solve_time_min = duration
        if self.solve_time_max is None or duration > self.solve_time_max:
            self.solve_time_max = duration

    def solve(
        self,
        q_tensor,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
    ):
        return self.solve_batch(
            [q_tensor],
            epsilon,
            num_repeats=num_repeats,
            round_digits=round_digits,
            include_pure_starts=include_pure_starts,
        )[0]

    def solve_batch(
        self,
        q_tensors,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
    ):
        q_tensors = [validate_bimatrix_q_tensor(q_tensor) for q_tensor in q_tensors]
        if not q_tensors:
            return []

        payloads = [
            (
                q_tensor,
                float(epsilon),
                int(num_repeats),
                round_digits,
                bool(include_pure_starts),
            )
            for q_tensor in q_tensors
        ]
        results = self._pool.map(_lemke_pool_solve_task, payloads)
        for result in results:
            duration = float(result.metadata.get("worker_sre_wall_seconds", 0.0))
            self._record_solve_time(duration)
            result.metadata["solver"] = self.name
            result.metadata["max_workers"] = self.config.max_workers
        return results

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

    def close(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.close()
            pool.join()
            self._pool = None
