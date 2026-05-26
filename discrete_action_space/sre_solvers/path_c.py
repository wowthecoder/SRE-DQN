from pathlib import Path
from dataclasses import dataclass, replace
import multiprocessing as mp
import time

import numpy as np
from path_solver import (
    PathSolverWrapper,
    solve_strategically_robust_bimatrix_game_path_lcp,
)

from .base import (
    SreSolveResult,
    SreStageGameSolver,
    _empty_duration_summary,
    _normalize_policy,
    validate_bimatrix_q_tensor,
)


DEFAULT_PATHWRAP_PATH = Path(__file__).resolve().with_name("pathwrap.so")


@dataclass(frozen=True)
class PathCBimatrixSreSolverConfig:
    pathwrap_path: object = DEFAULT_PATHWRAP_PATH


@dataclass(frozen=True)
class ProcessPoolPathCBimatrixSreSolverConfig:
    pathwrap_path: object = DEFAULT_PATHWRAP_PATH
    max_workers: int = 4
    start_method: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "max_workers", int(self.max_workers))
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive for process-pool SRE.")


def _select_best_solution(q_tensor, solutions):
    u1 = q_tensor[:, :, 0]
    u2 = q_tensor[:, :, 1]
    best_joint_reward = -float("inf")
    best = None

    for sol in solutions:
        p1 = _normalize_policy(sol.get("p1"))
        p2 = _normalize_policy(sol.get("p2"))
        if p1 is None or p2 is None:
            continue
        if p1.shape[0] != u1.shape[0] or p2.shape[0] != u1.shape[1]:
            continue

        r1 = float(p1 @ u1 @ p2)
        r2 = float(p1 @ u2 @ p2)
        if r1 + r2 > best_joint_reward:
            best_joint_reward = r1 + r2
            best = [p1, p2]

    return best


def _solution_from_lcp_vector(z, lcp, *, round_digits=4):
    k1 = lcp["K1"]
    k2 = lcp["K2"]
    n1 = lcp["n1"]
    p1 = np.asarray(z[:k1], dtype=np.float64)
    p2 = np.asarray(z[n1 : n1 + k2], dtype=np.float64)

    if (
        np.any(p1 < -1e-6)
        or np.any(p2 < -1e-6)
        or abs(float(np.sum(p1)) - 1.0) > 1e-5
        or abs(float(np.sum(p2)) - 1.0) > 1e-5
    ):
        return None, None

    p1 = _normalize_policy(p1)
    p2 = _normalize_policy(p2)
    if p1 is None or p2 is None:
        return None, None

    if round_digits is None:
        sol = {"p1": p1.tolist(), "p2": p2.tolist()}
    else:
        sol = {
            "p1": np.round(p1, round_digits).tolist(),
            "p2": np.round(p2, round_digits).tolist(),
        }

    u1 = lcp["U1_original"]
    u2 = lcp["U2_original"]
    nominal = [float(p1 @ u1 @ p2), float(p1 @ u2 @ p2)]
    return sol, nominal


class PathCBimatrixSreSolver(SreStageGameSolver):
    name = "path_c"

    def __init__(self, config: PathCBimatrixSreSolverConfig | None = None, **overrides):
        if config is None:
            config = PathCBimatrixSreSolverConfig(**overrides)
        elif overrides:
            config = replace(config, **overrides)
        self.config = config
        self.path_solver = PathSolverWrapper(self.config.pathwrap_path)

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
        u1 = q_tensor[:, :, 0]
        u2 = q_tensor[:, :, 1]
        try:
            solutions, utilities_sr, utilities_nominal = (
                solve_strategically_robust_bimatrix_game_path_lcp(
                    u1,
                    u2,
                    [epsilon, epsilon],
                    num_repeats,
                    self.path_solver,
                    round_digits=round_digits,
                    include_pure_starts=include_pure_starts,
                )
            )
        except Exception as exc:
            return SreSolveResult(
                policies=[],
                solutions=[],
                utilities_sr=[],
                utilities_nominal=[],
                success=False,
                message=str(exc),
                metadata={"solver": self.name},
            )

        policies = _select_best_solution(q_tensor, solutions)
        if policies is None:
            return SreSolveResult(
                policies=[],
                solutions=solutions,
                utilities_sr=utilities_sr,
                utilities_nominal=utilities_nominal,
                success=False,
                message="No valid SRE solution was returned by PATH.",
                metadata={"solver": self.name},
            )

        return SreSolveResult(
            policies=policies,
            solutions=solutions,
            utilities_sr=utilities_sr,
            utilities_nominal=utilities_nominal,
            success=True,
            metadata={
                "solver": self.name,
                "num_repeats": int(num_repeats),
                "include_pure_starts": bool(include_pure_starts),
            },
        )

    def get_solve_time_summary(self):
        return self.path_solver.get_solve_time_summary()

    def close(self):
        self.path_solver.close()


_POOL_SOLVER = None


def _path_pool_initializer(pathwrap_path):
    global _POOL_SOLVER
    _POOL_SOLVER = PathCBimatrixSreSolver(pathwrap_path=pathwrap_path)


def _path_pool_solve_task(payload):
    q_tensor, epsilon, num_repeats, round_digits, include_pure_starts = payload
    start = time.perf_counter()
    result = _POOL_SOLVER.solve(
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


class ProcessPoolPathCBimatrixSreSolver(SreStageGameSolver):
    name = "path_c_pool"

    def __init__(self, config: ProcessPoolPathCBimatrixSreSolverConfig | None = None, **overrides):
        if config is None:
            config = ProcessPoolPathCBimatrixSreSolverConfig(**overrides)
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
            initializer=_path_pool_initializer,
            initargs=(self.config.pathwrap_path,),
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
        results = self._pool.map(_path_pool_solve_task, payloads)
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
