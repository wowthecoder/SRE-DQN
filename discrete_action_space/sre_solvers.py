from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib
import time

import numpy as np

from path_solver import (
    PathSolverWrapper,
    build_robust_bimatrix_lcp,
    solve_strategically_robust_bimatrix_game_path_lcp,
)


def _empty_duration_summary():
    return {
        "count": 0,
        "mean_seconds": None,
        "min_seconds": None,
        "max_seconds": None,
        "std_seconds": None,
        "mean_microseconds": None,
        "min_microseconds": None,
        "max_microseconds": None,
        "std_microseconds": None,
    }


@dataclass
class SreSolveResult:
    policies: list[np.ndarray]
    solutions: list[dict]
    utilities_sr: list[list[float]]
    utilities_nominal: list[list[float]]
    success: bool
    message: str = ""
    metadata: dict = field(default_factory=dict)


class SreStageGameSolver(ABC):
    name: str = "base"

    @abstractmethod
    def solve(self, q_tensor, epsilon, *, num_repeats=20, round_digits=4):
        raise NotImplementedError

    def get_solve_time_summary(self):
        return _empty_duration_summary()

    def close(self):
        return None


def validate_bimatrix_q_tensor(q_tensor):
    q_tensor = np.asarray(q_tensor, dtype=np.float64)
    if q_tensor.ndim != 3 or q_tensor.shape[-1] != 2:
        raise ValueError(
            "Expected a bimatrix Q tensor with shape (A1, A2, 2), "
            f"got {q_tensor.shape}."
        )
    return q_tensor


def _normalize_policy(policy):
    p = np.asarray(policy, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = float(np.sum(p))
    if total <= 0.0:
        return None
    return p / total


def _uniform_policies(q_tensor):
    return [
        np.full(q_tensor.shape[0], 1.0 / q_tensor.shape[0], dtype=np.float64),
        np.full(q_tensor.shape[1], 1.0 / q_tensor.shape[1], dtype=np.float64),
    ]


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

    def __init__(self, pathwrap_path="pathwrap.so"):
        self.path_solver = PathSolverWrapper(pathwrap_path)

    def solve(self, q_tensor, epsilon, *, num_repeats=20, round_digits=4):
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
            metadata={"solver": self.name, "num_repeats": int(num_repeats)},
        )

    def get_solve_time_summary(self):
        return self.path_solver.get_solve_time_summary()

    def close(self):
        self.path_solver.close()


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

    def solve(self, q_tensor, epsilon, *, num_repeats=20, round_digits=4):
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
            },
        )


def make_sre_solver(solver_name="path_c", *, pathwrap_path=None):
    if solver_name == "path_c":
        kwargs = {}
        if pathwrap_path is not None:
            kwargs["pathwrap_path"] = pathwrap_path
        return PathCBimatrixSreSolver(**kwargs)
    if solver_name == "lemkelcp":
        return LemkeLcpBimatrixSreSolver()
    raise ValueError(f"Unknown SRE solver: {solver_name}")
