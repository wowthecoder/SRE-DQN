import itertools
import time

import numpy as np

from .base import SreSolveResult, SreStageGameSolver, _empty_duration_summary, _normalize_policy
from .nplayer_common import (
    _expected_nominal_values,
    _solution_dict_from_policies,
    _uniform_nplayer_policies,
    robust_exploitability,
    validate_nplayer_q_tensor,
)
from .path_mcp_nplayer import PathMcpNPlayerSreSolver


def smooth_min(t, a, b):
    """Stable log-sum-exp smoothing of min(a, b)."""
    t = max(float(t), 1e-300)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.minimum(a, b)
    exp_a = np.exp(-(a - c) / t)
    exp_b = np.exp(-(b - c) / t)
    return c - t * np.log(exp_a + exp_b)


def smooth_min_partials(t, a, b):
    """Return smooth_min and its partials with respect to t, a, and b."""
    t = max(float(t), 1e-300)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.minimum(a, b)
    exp_a = np.exp(-(a - c) / t)
    exp_b = np.exp(-(b - c) / t)
    total = exp_a + exp_b
    weight_a = exp_a / total
    weight_b = exp_b / total
    log_sum = -c / t + np.log(total)
    value = c - t * np.log(total)
    partial_t = -log_sum - (weight_a * a + weight_b * b) / t
    return value, partial_t, weight_a, weight_b


class SmoothingNewtonNPlayerSreSolver(SreStageGameSolver):
    """EVLCP-inspired smoothing Newton solver for the N-player SRE MCP.

    The finite-action N-player SRE conditions form a multilinear MCP. This
    solver applies the EVLCP paper's smooth-min idea to each lower-bound
    complementarity pair and runs a damped Newton method on the resulting
    smooth system. It is experimental and intentionally keeps the baseline
    iterative solver as the default backend elsewhere in the project.
    """

    name = "smoothing_newton_nplayer"

    def __init__(
        self,
        *,
        max_iter=100,
        tol=1e-6,
        t0=1e-2,
        t_min=1e-8,
        max_step_fraction=0.9,
        line_search_decay=0.5,
        random_seed=None,
    ):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.t0 = float(t0)
        self.t_min = float(t_min)
        self.max_step_fraction = float(max_step_fraction)
        self.line_search_decay = float(line_search_decay)
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
    def _build_index(action_sizes):
        return PathMcpNPlayerSreSolver._build_index(action_sizes)

    @staticmethod
    def _payoff_by_own_and_opponent_profile(q_tensor, player_id, profiles):
        return PathMcpNPlayerSreSolver._payoff_by_own_and_opponent_profile(
            q_tensor, player_id, profiles
        )

    @staticmethod
    def _policies_from_z(z, index, action_sizes):
        policies = []
        for player_id, action_size in enumerate(action_sizes):
            policy = _normalize_policy(z[index["prob"][player_id]])
            if policy is None or policy.shape[0] != action_size:
                return None
            policies.append(policy)
        return policies

    @staticmethod
    def _opponent_distribution_and_gradients(z, index, player_id, opponent_ids, profiles):
        return PathMcpNPlayerSreSolver._opponent_distribution_and_gradients(
            z, index, player_id, opponent_ids, profiles
        )

    def _make_start(self, index, action_sizes, policies=None):
        n_vars = index["kappa"][-1] + 1
        z = np.zeros(n_vars, dtype=np.float64)
        for player_id, action_size in enumerate(action_sizes):
            if policies is None:
                policy = self.rng.dirichlet(np.ones(action_size, dtype=np.float64))
            else:
                policy = np.asarray(policies[player_id], dtype=np.float64)
            z[index["prob"][player_id]] = policy
            z[index["lambda"][player_id]] = self.rng.random() + 1e-2
            xi_slice = index["xi"][player_id]
            eta_slice = index["eta"][player_id]
            z[xi_slice] = 2.0 * self.rng.random(xi_slice.stop - xi_slice.start) - 1.0
            z[eta_slice] = self.rng.random(eta_slice.stop - eta_slice.start)
            z[index["kappa"][player_id]] = 2.0 * self.rng.random() - 1.0
        return z

    def _initial_starts(self, index, action_sizes, num_repeats, include_pure_starts):
        starts = []
        starts.append(self._make_start(
            index,
            action_sizes,
            [np.full(size, 1.0 / size, dtype=np.float64) for size in action_sizes],
        ))
        if include_pure_starts:
            max_pure_starts = max(1, int(num_repeats))
            for profile_idx, pure_profile in enumerate(
                itertools.product(*[range(size) for size in action_sizes])
            ):
                if profile_idx >= max_pure_starts:
                    break
                policies = []
                for action_size, action_id in zip(action_sizes, pure_profile):
                    policy = np.zeros(action_size, dtype=np.float64)
                    policy[int(action_id)] = 1.0
                    policies.append(policy)
                starts.append(self._make_start(index, action_sizes, policies))
        for _ in range(max(0, int(num_repeats))):
            starts.append(self._make_start(index, action_sizes))
        return starts

    @staticmethod
    def _nonnegative_indices(index, num_players):
        indices = []
        for player_id in range(num_players):
            indices.extend(range(index["prob"][player_id].start, index["prob"][player_id].stop))
            indices.append(index["lambda"][player_id])
            indices.extend(range(index["eta"][player_id].start, index["eta"][player_id].stop))
        return np.asarray(indices, dtype=np.int64)

    def _compute_mcp_f_and_jacobian(self, z, q_tensor, epsilon, index, opponent_data, payoffs_by_player):
        action_sizes = q_tensor.shape[:-1]
        n_vars = z.shape[0]
        f = np.zeros(n_vars, dtype=np.float64)
        jac = np.zeros((n_vars, n_vars), dtype=np.float64)

        for player_id, action_size in enumerate(action_sizes):
            prob_slice = index["prob"][player_id]
            lambda_idx = index["lambda"][player_id]
            xi_slice = index["xi"][player_id]
            eta_slice = index["eta"][player_id]
            kappa_idx = index["kappa"][player_id]

            opponent_ids, profiles = opponent_data[player_id]
            num_opp_profiles = len(profiles)
            payoffs = payoffs_by_player[player_id]
            eta = z[eta_slice].reshape(num_opp_profiles, num_opp_profiles)
            xi = z[xi_slice]
            lambda_value = z[lambda_idx]
            distance = np.ones((num_opp_profiles, num_opp_profiles), dtype=np.float64)
            np.fill_diagonal(distance, 0.0)

            eta_col_sum = eta.sum(axis=0)
            f[prob_slice] = -z[kappa_idx] - payoffs @ eta_col_sum
            jac[prob_slice, kappa_idx] = -1.0
            eta_start = eta_slice.start
            for action_id in range(action_size):
                row = prob_slice.start + action_id
                for k in range(num_opp_profiles):
                    for l in range(num_opp_profiles):
                        jac[row, eta_start + k * num_opp_profiles + l] = -payoffs[action_id, l]

            f[lambda_idx] = float(epsilon) - float(np.sum(eta * distance))
            for k in range(num_opp_profiles):
                for l in range(num_opp_profiles):
                    jac[lambda_idx, eta_start + k * num_opp_profiles + l] = -distance[k, l]

            opponent_distribution, distribution_grads = self._opponent_distribution_and_gradients(
                z, index, player_id, opponent_ids, profiles
            )
            f[xi_slice] = -opponent_distribution + eta.sum(axis=1)
            for j, profile_grads in enumerate(distribution_grads):
                row = xi_slice.start + j
                for opponent_id, grad in profile_grads:
                    jac[row, index["prob"][opponent_id]] -= grad
                for l in range(num_opp_profiles):
                    jac[row, eta_start + j * num_opp_profiles + l] = 1.0

            own_expected_payoff = z[prob_slice] @ payoffs
            for k in range(num_opp_profiles):
                for l in range(num_opp_profiles):
                    row = eta_start + k * num_opp_profiles + l
                    f[row] = -xi[k] + own_expected_payoff[l] + lambda_value * distance[k, l]
                    jac[row, xi_slice.start + k] = -1.0
                    jac[row, prob_slice] = payoffs[:, l]
                    jac[row, lambda_idx] = distance[k, l]

            f[kappa_idx] = 1.0 - float(np.sum(z[prob_slice]))
            jac[kappa_idx, prob_slice] = -1.0

        return f, jac

    def _smoothed_system(self, t, z, q_tensor, epsilon, index, opponent_data, payoffs_by_player, nonnegative_idx):
        f, jac_f = self._compute_mcp_f_and_jacobian(
            z, q_tensor, epsilon, index, opponent_data, payoffs_by_player
        )
        n_vars = z.shape[0]
        residual = f.copy()
        jac_z = jac_f.copy()
        jac_t = np.zeros(n_vars, dtype=np.float64)

        values, partial_t, weight_y, weight_f = smooth_min_partials(
            t, z[nonnegative_idx], f[nonnegative_idx]
        )
        residual[nonnegative_idx] = values
        jac_t[nonnegative_idx] = partial_t
        jac_z[nonnegative_idx, :] = weight_f[:, None] * jac_f[nonnegative_idx, :]
        jac_z[nonnegative_idx, nonnegative_idx] += weight_y

        h = np.empty(n_vars + 1, dtype=np.float64)
        h[0] = t
        h[1:] = residual

        jac_h = np.zeros((n_vars + 1, n_vars + 1), dtype=np.float64)
        jac_h[0, 0] = 1.0
        jac_h[1:, 0] = jac_t
        jac_h[1:, 1:] = jac_z
        return h, jac_h

    @staticmethod
    def _newton_direction(jac_h, h):
        try:
            cond = np.linalg.cond(jac_h)
            if not np.isfinite(cond) or cond > 1e12:
                raise np.linalg.LinAlgError("ill-conditioned smoothing Newton system")
            direction, *_ = np.linalg.lstsq(jac_h, -h, rcond=None)
        except np.linalg.LinAlgError:
            damping = 1e-6
            lhs = jac_h.T @ jac_h + damping * np.eye(jac_h.shape[1], dtype=np.float64)
            rhs = -jac_h.T @ h
            direction = np.linalg.solve(lhs, rhs)
        return direction

    def _cap_step(self, t, z, dt, dz, nonnegative_idx):
        alpha = 1.0
        t_floor = max(self.t_min, (1.0 - self.max_step_fraction) * t)
        if dt < 0.0:
            alpha = min(alpha, max(0.0, (t - t_floor) / (-dt)))
        for idx in nonnegative_idx:
            if dz[idx] < -1e-12:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    ratio = self.max_step_fraction * z[idx] / (-dz[idx])
                if np.isfinite(ratio):
                    alpha = min(alpha, ratio)
        return float(np.clip(alpha, 0.0, 1.0))

    def _run_newton(self, z0, q_tensor, epsilon, index, opponent_data, payoffs_by_player, nonnegative_idx):
        t = max(self.t0, self.t_min)
        z = np.asarray(z0, dtype=np.float64).copy()
        line_search_failures = 0
        residual_norm = float("inf")
        converged = False

        for iteration in range(self.max_iter):
            h, jac_h = self._smoothed_system(
                t, z, q_tensor, epsilon, index, opponent_data, payoffs_by_player, nonnegative_idx
            )
            residual_norm = float(np.linalg.norm(h))
            if residual_norm <= self.tol:
                converged = True
                return z, {
                    "iterations": iteration,
                    "final_t": float(t),
                    "residual_norm": residual_norm,
                    "line_search_failures": line_search_failures,
                    "newton_converged": converged,
                }

            direction = self._newton_direction(jac_h, h)
            dt = float(direction[0])
            dz = direction[1:]
            alpha = self._cap_step(t, z, dt, dz, nonnegative_idx)
            current_merit = 0.5 * residual_norm * residual_norm
            accepted = False

            while alpha >= 1e-8:
                t_candidate = max(self.t_min, t + alpha * dt)
                z_candidate = z + alpha * dz
                h_candidate, _ = self._smoothed_system(
                    t_candidate,
                    z_candidate,
                    q_tensor,
                    epsilon,
                    index,
                    opponent_data,
                    payoffs_by_player,
                    nonnegative_idx,
                )
                candidate_merit = 0.5 * float(h_candidate @ h_candidate)
                if np.isfinite(candidate_merit) and candidate_merit < current_merit:
                    t = t_candidate
                    z = z_candidate
                    accepted = True
                    break
                alpha *= self.line_search_decay

            if not accepted:
                line_search_failures += 1
                t = max(self.t_min, min(t * 0.5, t + max(dt, -0.5 * t)))

        h, _ = self._smoothed_system(
            t, z, q_tensor, epsilon, index, opponent_data, payoffs_by_player, nonnegative_idx
        )
        residual_norm = float(np.linalg.norm(h))
        return z, {
            "iterations": self.max_iter,
            "final_t": float(t),
            "residual_norm": residual_norm,
            "line_search_failures": line_search_failures,
            "newton_converged": False,
        }

    def _candidate_from_policies(self, q_tensor, policies, epsilon, metadata):
        if policies is None:
            return None
        exploitability, player_gaps, robust_values = robust_exploitability(
            q_tensor, policies, epsilon
        )
        nominal = _expected_nominal_values(q_tensor, policies)
        robust_policy_values = [
            float(policy @ values) for policy, values in zip(policies, robust_values)
        ]
        return {
            "policies": policies,
            "robust_exploitability": float(exploitability),
            "player_robust_gaps": [float(gap) for gap in player_gaps],
            "robust_policy_values": robust_policy_values,
            "nominal_values": [float(value) for value in nominal],
            "joint_nominal_welfare": float(np.sum(nominal)),
            **metadata,
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
        action_sizes = q_tensor.shape[:-1]
        index, opponent_data, _ = self._build_index(action_sizes)
        payoffs_by_player = [
            self._payoff_by_own_and_opponent_profile(
                q_tensor, player_id, opponent_data[player_id][1]
            )
            for player_id in range(q_tensor.shape[-1])
        ]
        nonnegative_idx = self._nonnegative_indices(index, q_tensor.shape[-1])
        starts = self._initial_starts(index, action_sizes, num_repeats, include_pure_starts)
        candidates = []

        for start_index, z0 in enumerate(starts):
            start_policies = self._policies_from_z(z0, index, action_sizes)
            start_candidate = self._candidate_from_policies(
                q_tensor,
                start_policies,
                epsilon,
                {"source": "initial_start", "start_index": int(start_index)},
            )
            if start_candidate is not None:
                candidates.append(start_candidate)

            z_sol, newton_meta = self._run_newton(
                z0, q_tensor, epsilon, index, opponent_data, payoffs_by_player, nonnegative_idx
            )
            policies = self._policies_from_z(z_sol, index, action_sizes)
            candidate = self._candidate_from_policies(
                q_tensor,
                policies,
                epsilon,
                {"source": "smoothing_newton", "start_index": int(start_index), **newton_meta},
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            policies = _uniform_nplayer_policies(q_tensor)
            fallback = self._candidate_from_policies(
                q_tensor, policies, epsilon, {"source": "uniform_fallback"}
            )
            candidates = [fallback]

        candidates.sort(
            key=lambda candidate: (
                candidate["robust_exploitability"],
                -candidate["joint_nominal_welfare"],
            )
        )
        best = candidates[0]
        success = bool(best["robust_exploitability"] <= max(10.0 * self.tol, 1e-4))
        solution = _solution_dict_from_policies(best["policies"], round_digits=round_digits)
        return SreSolveResult(
            policies=best["policies"],
            solutions=[solution],
            utilities_sr=[best["robust_policy_values"]],
            utilities_nominal=[best["nominal_values"]],
            success=success,
            message="" if success else "Returned best smoothing Newton N-player SRE candidate.",
            metadata={
                "solver": self.name,
                "algorithm_family": "evlcp_inspired_smoothing_newton_nplayer_mcp",
                "epsilon": float(epsilon),
                "num_agents": int(q_tensor.shape[-1]),
                "action_sizes": [int(size) for size in action_sizes],
                "num_starts": int(len(starts)),
                "num_candidates": int(len(candidates)),
                "converged_to_tolerance": success,
                **{
                    key: value
                    for key, value in best.items()
                    if key not in {"policies", "robust_policy_values", "nominal_values"}
                },
                "robust_policy_values": best["robust_policy_values"],
                "nominal_values": best["nominal_values"],
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
