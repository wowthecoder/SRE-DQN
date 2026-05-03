from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import itertools
import multiprocessing as mp
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
    def solve(
        self,
        q_tensor,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
    ):
        raise NotImplementedError

    def solve_batch(
        self,
        q_tensors,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
    ):
        return [
            self.solve(
                q_tensor,
                epsilon,
                num_repeats=num_repeats,
                round_digits=round_digits,
                include_pure_starts=include_pure_starts,
            )
            for q_tensor in q_tensors
        ]

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


def validate_nplayer_q_tensor(q_tensor):
    q_tensor = np.asarray(q_tensor, dtype=np.float64)
    if q_tensor.ndim < 3:
        raise ValueError(
            "Expected an N-player Q tensor with shape (A1, ..., AN, N), "
            f"got {q_tensor.shape}."
        )
    num_agents = int(q_tensor.shape[-1])
    if num_agents < 2 or q_tensor.ndim != num_agents + 1:
        raise ValueError(
            "Expected shape (A1, ..., AN, N) where N is the number of agents, "
            f"got {q_tensor.shape}."
        )
    if any(int(size) <= 0 for size in q_tensor.shape[:-1]):
        raise ValueError(f"Action dimensions must be positive, got {q_tensor.shape}.")
    return q_tensor


def _uniform_nplayer_policies(q_tensor):
    q_tensor = validate_nplayer_q_tensor(q_tensor)
    return [
        np.full(size, 1.0 / size, dtype=np.float64)
        for size in q_tensor.shape[:-1]
    ]


def _normalize_nplayer_policies(policies, action_sizes):
    normalized = []
    for policy, size in zip(policies, action_sizes):
        p = _normalize_policy(policy)
        if p is None or p.shape[0] != size:
            p = np.full(size, 1.0 / size, dtype=np.float64)
        normalized.append(p)
    return normalized


def _joint_distribution(policies):
    distribution = np.asarray(policies[0], dtype=np.float64)
    for policy in policies[1:]:
        distribution = np.multiply.outer(distribution, policy)
    return distribution.reshape(-1)


def _expected_nominal_values(q_tensor, policies):
    expected = np.asarray(q_tensor, dtype=np.float64)
    for policy in policies:
        expected = np.tensordot(policy, expected, axes=([0], [0]))
    return np.asarray(expected, dtype=np.float64)


def _tv_worst_case_value(nominal_distribution, values, epsilon):
    """Minimum expected value over a finite total-variation ball.

    The SRQ paper uses TV cost, so a Wasserstein-1 ball is
    0.5 * ||q - p||_1 <= epsilon. Moving delta mass from a high-value
    outcome to a low-value outcome spends delta TV budget.
    """
    p = np.asarray(nominal_distribution, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(p.sum())
    if total <= 0.0:
        p = np.full_like(v, 1.0 / max(v.size, 1), dtype=np.float64)
    else:
        p = np.clip(p / total, 0.0, None)

    budget = float(np.clip(epsilon, 0.0, 1.0))
    if budget <= 0.0 or p.size <= 1:
        return float(p @ v)

    q = p.copy()
    high_order = np.argsort(-v)
    low_order = np.argsort(v)
    high_pos = 0
    low_pos = 0

    while budget > 1e-12 and high_pos < p.size and low_pos < p.size:
        hi = int(high_order[high_pos])
        lo = int(low_order[low_pos])
        if v[hi] <= v[lo] + 1e-12:
            break
        movable = min(q[hi], 1.0 - q[lo], budget)
        if movable <= 1e-12:
            if q[hi] <= 1e-12:
                high_pos += 1
            if q[lo] >= 1.0 - 1e-12:
                low_pos += 1
            continue
        q[hi] -= movable
        q[lo] += movable
        budget -= movable
        if q[hi] <= 1e-12:
            high_pos += 1
        if q[lo] >= 1.0 - 1e-12:
            low_pos += 1

    return float(q @ v)


def _opponent_payoff_values(q_tensor, player_id, action_id):
    slicer = [slice(None)] * q_tensor.ndim
    slicer[player_id] = int(action_id)
    slicer[-1] = int(player_id)
    return np.asarray(q_tensor[tuple(slicer)], dtype=np.float64).reshape(-1)


def robust_action_values(q_tensor, policies, epsilon, player_id):
    q_tensor = validate_nplayer_q_tensor(q_tensor)
    action_sizes = q_tensor.shape[:-1]
    opponent_policies = [
        policies[j] for j in range(len(action_sizes)) if j != player_id
    ]
    opponent_distribution = _joint_distribution(opponent_policies)
    values = np.zeros(action_sizes[player_id], dtype=np.float64)
    for action_id in range(action_sizes[player_id]):
        payoff_values = _opponent_payoff_values(q_tensor, player_id, action_id)
        values[action_id] = _tv_worst_case_value(
            opponent_distribution, payoff_values, epsilon
        )
    return values


def robust_exploitability(q_tensor, policies, epsilon):
    q_tensor = validate_nplayer_q_tensor(q_tensor)
    gaps = []
    robust_values = []
    for player_id, policy in enumerate(policies):
        action_values = robust_action_values(q_tensor, policies, epsilon, player_id)
        robust_values.append(action_values)
        current_value = float(np.asarray(policy, dtype=np.float64) @ action_values)
        gaps.append(max(0.0, float(np.max(action_values) - current_value)))
    return float(max(gaps) if gaps else 0.0), gaps, robust_values


def _solution_dict_from_policies(policies, round_digits=4):
    solution = {}
    for idx, policy in enumerate(policies, start=1):
        values = np.asarray(policy, dtype=np.float64)
        if round_digits is not None:
            values = np.round(values, round_digits)
        solution[f"p{idx}"] = values.tolist()
    return solution


class IterativeNPlayerSreSolver(SreStageGameSolver):
    """Approximate N-player finite-action SRE solver.

    This is the general stage-game oracle used by Deep SRQ for N > 2. It uses
    robust best-response fixed-point iteration with multi-starts. The
    subclasses below expose the research-paper variants and record the
    intended algorithmic phase metadata while sharing this robust oracle as a
    dependable Python fallback.
    """

    name = "baseline_nplayer"

    def __init__(
        self,
        *,
        max_iter=250,
        tol=1e-5,
        damping=0.35,
        temperature=0.02,
        random_seed=None,
    ):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.damping = float(damping)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(random_seed)
        self.solve_durations = []

    def _record_duration(self, duration):
        self.solve_durations.append(float(duration))

    def get_solve_time_summary(self):
        if not self.solve_durations:
            return _empty_duration_summary()
        durations = np.asarray(self.solve_durations, dtype=np.float64)
        return {
            "count": int(durations.size),
            "mean_seconds": float(np.mean(durations)),
            "min_seconds": float(np.min(durations)),
            "max_seconds": float(np.max(durations)),
            "std_seconds": float(np.std(durations)),
            "mean_microseconds": float(np.mean(durations) * 1_000_000.0),
            "min_microseconds": float(np.min(durations) * 1_000_000.0),
            "max_microseconds": float(np.max(durations) * 1_000_000.0),
            "std_microseconds": float(np.std(durations) * 1_000_000.0),
        }

    @staticmethod
    def _soft_best_response(values, temperature):
        values = np.asarray(values, dtype=np.float64)
        if temperature <= 0.0:
            best = values >= np.max(values) - 1e-10
            return best.astype(np.float64) / float(np.sum(best))
        centered = values - float(np.max(values))
        weights = np.exp(centered / max(float(temperature), 1e-12))
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            best = values >= np.max(values) - 1e-10
            return best.astype(np.float64) / float(np.sum(best))
        return weights / total

    def _initial_policies(self, q_tensor, num_repeats, include_pure_starts):
        action_sizes = q_tensor.shape[:-1]
        starts = [_uniform_nplayer_policies(q_tensor)]
        if include_pure_starts:
            pure_profiles = itertools.product(
                *[range(action_size) for action_size in action_sizes]
            )
            max_pure_starts = max(1, int(num_repeats))
            for profile_id, profile in enumerate(pure_profiles):
                if profile_id >= max_pure_starts:
                    break
                policies = []
                for action_size, action_id in zip(action_sizes, profile):
                    policy = np.zeros(action_size, dtype=np.float64)
                    policy[int(action_id)] = 1.0
                    policies.append(policy)
                starts.append(policies)
        for _ in range(max(0, int(num_repeats))):
            starts.append(
                [
                    self.rng.dirichlet(np.ones(action_size, dtype=np.float64))
                    for action_size in action_sizes
                ]
            )
        return starts

    def _iterate(self, q_tensor, epsilon, policies):
        policies = _normalize_nplayer_policies(policies, q_tensor.shape[:-1])
        last_gap = float("inf")
        for iteration in range(self.max_iter):
            new_policies = []
            for player_id, policy in enumerate(policies):
                values = robust_action_values(q_tensor, policies, epsilon, player_id)
                br = self._soft_best_response(values, self.temperature)
                mixed = (1.0 - self.damping) * policy + self.damping * br
                new_policies.append(_normalize_policy(mixed))
            policies = _normalize_nplayer_policies(new_policies, q_tensor.shape[:-1])
            gap, _, _ = robust_exploitability(q_tensor, policies, epsilon)
            last_gap = gap
            if gap <= self.tol:
                return policies, iteration + 1, gap
        return policies, self.max_iter, last_gap

    def _metadata_for_candidate(self, q_tensor, policies, epsilon, iterations):
        exploitability, player_gaps, robust_values = robust_exploitability(
            q_tensor, policies, epsilon
        )
        nominal = _expected_nominal_values(q_tensor, policies)
        robust_policy_values = [
            float(np.asarray(policy, dtype=np.float64) @ values)
            for policy, values in zip(policies, robust_values)
        ]
        return {
            "iterations": int(iterations),
            "robust_exploitability": float(exploitability),
            "player_robust_gaps": [float(gap) for gap in player_gaps],
            "robust_policy_values": robust_policy_values,
            "nominal_values": [float(value) for value in nominal],
            "joint_nominal_welfare": float(np.sum(nominal)),
        }

    def _solve_impl(
        self,
        q_tensor,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
    ):
        q_tensor = validate_nplayer_q_tensor(q_tensor)
        best = None
        best_meta = None
        starts = self._initial_policies(q_tensor, num_repeats, include_pure_starts)
        for start_policies in starts:
            policies, iterations, _ = self._iterate(q_tensor, epsilon, start_policies)
            meta = self._metadata_for_candidate(
                q_tensor, policies, epsilon, iterations
            )
            score = (
                meta["robust_exploitability"],
                -meta["joint_nominal_welfare"],
                meta["iterations"],
            )
            if best is None or score < best_meta["score"]:
                best = policies
                best_meta = dict(meta)
                best_meta["score"] = score

        if best is None:
            policies = _uniform_nplayer_policies(q_tensor)
            meta = self._metadata_for_candidate(q_tensor, policies, epsilon, 0)
            success = False
            message = "No N-player SRE candidate was generated."
        else:
            policies = best
            meta = best_meta
            success = bool(meta["robust_exploitability"] <= max(10.0 * self.tol, 1e-4))
            message = "" if success else "Returned best approximate N-player SRE candidate."

        solution = _solution_dict_from_policies(policies, round_digits=round_digits)
        return SreSolveResult(
            policies=policies,
            solutions=[solution],
            utilities_sr=[meta["robust_policy_values"]],
            utilities_nominal=[meta["nominal_values"]],
            success=True,
            message=message,
            metadata={
                "solver": self.name,
                "algorithm_family": "iterative_robust_best_response",
                "epsilon": float(epsilon),
                "num_agents": int(q_tensor.shape[-1]),
                "action_sizes": [int(size) for size in q_tensor.shape[:-1]],
                "num_starts": int(len(starts)),
                "converged_to_tolerance": success,
                **{key: value for key, value in meta.items() if key != "score"},
            },
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
        start = time.perf_counter()
        try:
            return self._solve_impl(
                q_tensor,
                epsilon,
                num_repeats=num_repeats,
                round_digits=round_digits,
                include_pure_starts=include_pure_starts,
            )
        finally:
            self._record_duration(time.perf_counter() - start)


class DcaBlNPlayerSreSolver(IterativeNPlayerSreSolver):
    name = "dca_bl_nplayer"

    def _solve_impl(self, *args, **kwargs):
        result = super()._solve_impl(*args, **kwargs)
        result.metadata.update(
            {
                "solver": self.name,
                "algorithm_family": "dca_bl",
                "dc_decomposition": "bilinear_complementarity_quadratic_difference",
                "subproblem_backend": "python_fallback",
                "dca_iterations": result.metadata.get("iterations", 0),
            }
        )
        return result


class SpatialBranchBoundNPlayerSreSolver(IterativeNPlayerSreSolver):
    name = "sbb_nplayer"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("damping", 0.2)
        kwargs.setdefault("temperature", 0.01)
        super().__init__(*args, **kwargs)

    def _solve_impl(self, *args, **kwargs):
        result = super()._solve_impl(*args, **kwargs)
        result.metadata.update(
            {
                "solver": self.name,
                "algorithm_family": "spatial_branch_and_bound",
                "sbb_backend": "python_fallback",
                "sbb_nodes": result.metadata.get("num_starts", 0),
                "sbb_gap": result.metadata.get("robust_exploitability"),
            }
        )
        return result


class WarmStartNPlayerSreSolver(IterativeNPlayerSreSolver):
    name = "warm_start_nplayer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dca_solver = DcaBlNPlayerSreSolver(*args, **kwargs)
        self.sbb_solver = SpatialBranchBoundNPlayerSreSolver(*args, **kwargs)

    def _solve_impl(self, q_tensor, epsilon, **kwargs):
        dca_result = self.dca_solver._solve_impl(q_tensor, epsilon, **kwargs)
        sbb_result = self.sbb_solver._solve_impl(q_tensor, epsilon, **kwargs)
        candidates = [dca_result, sbb_result]
        candidates.sort(
            key=lambda result: (
                float(result.metadata.get("robust_exploitability", float("inf"))),
                -float(result.metadata.get("joint_nominal_welfare", -float("inf"))),
            )
        )
        start_result = candidates[0]
        polished = super()._solve_impl(q_tensor, epsilon, **kwargs)
        if (
            float(start_result.metadata.get("robust_exploitability", float("inf")))
            < float(polished.metadata.get("robust_exploitability", float("inf")))
        ):
            result = start_result
        else:
            result = polished
        result.metadata.update(
            {
                "solver": self.name,
                "algorithm_family": "efficient_warm_start",
                "warm_start_sources": [
                    {
                        "solver": candidate.metadata.get("solver"),
                        "robust_exploitability": candidate.metadata.get(
                            "robust_exploitability"
                        ),
                        "joint_nominal_welfare": candidate.metadata.get(
                            "joint_nominal_welfare"
                        ),
                    }
                    for candidate in candidates
                ],
                "selected_warm_start_source": start_result.metadata.get("solver"),
            }
        )
        return result

    def get_solve_time_summary(self):
        summaries = [
            super().get_solve_time_summary(),
            self.dca_solver.get_solve_time_summary(),
            self.sbb_solver.get_solve_time_summary(),
        ]
        durations = []
        for summary in summaries:
            count = int(summary.get("count", 0) or 0)
            mean = summary.get("mean_seconds")
            if count and mean is not None:
                durations.extend([float(mean)] * count)
        if not durations:
            return _empty_duration_summary()
        arr = np.asarray(durations, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean_seconds": float(np.mean(arr)),
            "min_seconds": float(np.min(arr)),
            "max_seconds": float(np.max(arr)),
            "std_seconds": float(np.std(arr)),
            "mean_microseconds": float(np.mean(arr) * 1_000_000.0),
            "min_microseconds": float(np.min(arr) * 1_000_000.0),
            "max_microseconds": float(np.max(arr) * 1_000_000.0),
            "std_microseconds": float(np.std(arr) * 1_000_000.0),
        }


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

    def __init__(self, pathwrap_path="pathwrap.so", max_workers=4, start_method=None):
        self.pathwrap_path = pathwrap_path
        self.max_workers = int(max_workers)
        self.start_method = start_method
        self.solve_time_count = 0
        self.solve_time_sum = 0.0
        self.solve_time_sumsq = 0.0
        self.solve_time_min = None
        self.solve_time_max = None
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive for process-pool SRE.")
        ctx = mp.get_context(start_method) if start_method else mp.get_context()
        self._pool = ctx.Pool(
            processes=self.max_workers,
            initializer=_path_pool_initializer,
            initargs=(self.pathwrap_path,),
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
            result.metadata["max_workers"] = self.max_workers
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


def make_sre_solver(
    solver_name="path_c",
    *,
    pathwrap_path=None,
    max_workers=4,
    start_method=None,
    max_iter=250,
    tol=1e-5,
    damping=0.35,
    temperature=0.02,
    random_seed=None,
):
    if solver_name == "path_c":
        kwargs = {}
        if pathwrap_path is not None:
            kwargs["pathwrap_path"] = pathwrap_path
        return PathCBimatrixSreSolver(**kwargs)
    if solver_name == "path_c_pool":
        kwargs = {"max_workers": max_workers, "start_method": start_method}
        if pathwrap_path is not None:
            kwargs["pathwrap_path"] = pathwrap_path
        return ProcessPoolPathCBimatrixSreSolver(**kwargs)
    if solver_name == "lemkelcp":
        return LemkeLcpBimatrixSreSolver()
    nplayer_kwargs = {
        "max_iter": max_iter,
        "tol": tol,
        "damping": damping,
        "temperature": temperature,
        "random_seed": random_seed,
    }
    if solver_name in {"baseline_nplayer", "nplayer_sre"}:
        return IterativeNPlayerSreSolver(**nplayer_kwargs)
    if solver_name in {"dca_bl_nplayer", "dca_bl_only"}:
        return DcaBlNPlayerSreSolver(**nplayer_kwargs)
    if solver_name in {"sbb_nplayer", "spatial_branch_bound_nplayer", "sbb_only"}:
        return SpatialBranchBoundNPlayerSreSolver(**nplayer_kwargs)
    if solver_name in {"warm_start_nplayer", "efficient_warm_start"}:
        return WarmStartNPlayerSreSolver(**nplayer_kwargs)
    raise ValueError(f"Unknown SRE solver: {solver_name}")
