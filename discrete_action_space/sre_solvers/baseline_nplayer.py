import itertools
import time

import numpy as np

from .base import SreSolveResult, SreStageGameSolver, _empty_duration_summary, _normalize_policy
from .nplayer_common import (
    _expected_nominal_values,
    _normalize_nplayer_policies,
    _solution_dict_from_policies,
    _uniform_nplayer_policies,
    robust_action_values,
    robust_exploitability,
    validate_nplayer_q_tensor,
)


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

