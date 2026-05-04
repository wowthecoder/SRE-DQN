import itertools
from pathlib import Path
import time

import numpy as np

try:
    from path_solver import PathSolverWrapper
except ImportError:  # Package import from repository root.
    from ..path_solver import PathSolverWrapper

from .base import SreSolveResult, SreStageGameSolver, _empty_duration_summary, _normalize_policy
from .nplayer_common import (
    _expected_nominal_values,
    _solution_dict_from_policies,
    _uniform_nplayer_policies,
    robust_exploitability,
    validate_nplayer_q_tensor,
)


DEFAULT_PATHWRAP_PATH = Path(__file__).resolve().with_name("pathwrap.so")
INF = 1e20


class PathMcpNPlayerSreSolver(SreStageGameSolver):
    """PATH-backed MCP solver for finite-action N-player SRE stage games.

    This ports the JuMP/PATH formulation in
    `strategically-robust-game-theory/sr_games_julia/solve_sr_N_player_game.jl`.
    Unlike the two-player PATH backend, this is not an LCP: for N > 2 the
    product distribution over opponents makes the MCP multilinear.
    """

    name = "path_mcp_nplayer"

    def __init__(self, pathwrap_path=DEFAULT_PATHWRAP_PATH):
        self.path_solver = PathSolverWrapper(pathwrap_path)

    @staticmethod
    def _opponent_profiles(action_sizes, player_id):
        opponent_ids = [idx for idx in range(len(action_sizes)) if idx != player_id]
        profiles = list(itertools.product(*[range(action_sizes[idx]) for idx in opponent_ids]))
        return opponent_ids, profiles

    @staticmethod
    def _payoff_by_own_and_opponent_profile(q_tensor, player_id, profiles):
        action_size = q_tensor.shape[player_id]
        payoffs = np.zeros((action_size, len(profiles)), dtype=np.float64)
        for action_id in range(action_size):
            for profile_idx, profile in enumerate(profiles):
                full_profile = []
                opponent_pos = 0
                for idx in range(q_tensor.shape[-1]):
                    if idx == player_id:
                        full_profile.append(action_id)
                    else:
                        full_profile.append(profile[opponent_pos])
                        opponent_pos += 1
                payoffs[action_id, profile_idx] = q_tensor[tuple(full_profile) + (player_id,)]
        return payoffs

    @staticmethod
    def _build_index(action_sizes):
        index = {"prob": [], "lambda": [], "xi": [], "eta": [], "kappa": []}
        pos = 0
        opponent_data = []
        for player_id, action_size in enumerate(action_sizes):
            opponent_ids, profiles = PathMcpNPlayerSreSolver._opponent_profiles(
                action_sizes, player_id
            )
            opponent_data.append((opponent_ids, profiles))
            num_opp_profiles = len(profiles)

            index["prob"].append(slice(pos, pos + action_size))
            pos += action_size
            index["lambda"].append(pos)
            pos += 1
            index["xi"].append(slice(pos, pos + num_opp_profiles))
            pos += num_opp_profiles
            index["eta"].append(slice(pos, pos + num_opp_profiles * num_opp_profiles))
            pos += num_opp_profiles * num_opp_profiles
            index["kappa"].append(pos)
            pos += 1
        return index, opponent_data, pos

    @staticmethod
    def _policies_from_z(z, index, action_sizes):
        policies = []
        for player_id, action_size in enumerate(action_sizes):
            policy = _normalize_policy(z[index["prob"][player_id]])
            if policy is None or policy.shape[0] != action_size:
                return None
            policies.append(policy)
        return policies

    def _make_start(self, index, action_sizes, opponent_data, policies=None):
        n_vars = index["kappa"][-1] + 1
        z = np.zeros(n_vars, dtype=np.float64)
        rng = np.random.default_rng()
        for player_id, action_size in enumerate(action_sizes):
            if policies is None:
                policy = rng.dirichlet(np.ones(action_size, dtype=np.float64))
            else:
                policy = np.asarray(policies[player_id], dtype=np.float64)
            z[index["prob"][player_id]] = policy
            z[index["lambda"][player_id]] = rng.random() + 1e-2
            xi_slice = index["xi"][player_id]
            eta_slice = index["eta"][player_id]
            z[xi_slice] = 100.0 * rng.random(xi_slice.stop - xi_slice.start) - 50.0
            z[eta_slice] = rng.random(eta_slice.stop - eta_slice.start)
            z[index["kappa"][player_id]] = 100.0 * rng.random() - 50.0
        return z

    def _initial_starts(self, index, action_sizes, opponent_data, num_repeats, include_pure_starts):
        starts = []
        if include_pure_starts:
            max_pure_starts = max(1, int(num_repeats))
            for profile_idx, pure_profile in enumerate(itertools.product(*[range(size) for size in action_sizes])):
                if profile_idx >= max_pure_starts:
                    break
                policies = []
                for action_size, action_id in zip(action_sizes, pure_profile):
                    policy = np.zeros(action_size, dtype=np.float64)
                    policy[action_id] = 1.0
                    policies.append(policy)
                starts.append(self._make_start(index, action_sizes, opponent_data, policies))

        for _ in range(max(0, int(num_repeats))):
            starts.append(self._make_start(index, action_sizes, opponent_data))
        return starts

    @staticmethod
    def _opponent_distribution_and_gradients(z, index, player_id, opponent_ids, profiles):
        probs = [z[index["prob"][opponent_id]] for opponent_id in opponent_ids]
        distribution = np.zeros(len(profiles), dtype=np.float64)
        gradients = []
        for profile_idx, profile in enumerate(profiles):
            product = 1.0
            for local_idx, action_id in enumerate(profile):
                product *= probs[local_idx][action_id]
            distribution[profile_idx] = product

            profile_grads = []
            for local_idx, opponent_id in enumerate(opponent_ids):
                grad = np.zeros_like(probs[local_idx])
                action_id = profile[local_idx]
                partial = 1.0
                for other_local_idx, other_action_id in enumerate(profile):
                    if other_local_idx == local_idx:
                        continue
                    partial *= probs[other_local_idx][other_action_id]
                grad[action_id] = partial
                profile_grads.append((opponent_id, grad))
            gradients.append(profile_grads)
        return distribution, gradients

    def _compute_f_and_jacobian(self, z, q_tensor, epsilon, index, opponent_data, payoffs_by_player):
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

            # d / d(prob_i): -kappa_i - sum_{k,l} eta_i[k,l] payoff_i(a_i,l)
            eta_col_sum = eta.sum(axis=0)
            f[prob_slice] = -z[kappa_idx] - payoffs @ eta_col_sum
            jac[prob_slice, kappa_idx] = -1.0
            eta_start = eta_slice.start
            for action_id in range(action_size):
                row = prob_slice.start + action_id
                for k in range(num_opp_profiles):
                    for l in range(num_opp_profiles):
                        jac[row, eta_start + k * num_opp_profiles + l] = -payoffs[action_id, l]

            # d / d(lambda_i): epsilon_i - transport cost.
            f[lambda_idx] = float(epsilon) - float(np.sum(eta * distance))
            for k in range(num_opp_profiles):
                for l in range(num_opp_profiles):
                    jac[lambda_idx, eta_start + k * num_opp_profiles + l] = -distance[k, l]

            opponent_distribution, distribution_grads = self._opponent_distribution_and_gradients(
                z, index, player_id, opponent_ids, profiles
            )

            # d / d(xi_i[j]): -prod opponents prob(profile_j) + sum_l eta_i[j,l]
            f[xi_slice] = -opponent_distribution + eta.sum(axis=1)
            for j, profile_grads in enumerate(distribution_grads):
                row = xi_slice.start + j
                for opponent_id, grad in profile_grads:
                    opponent_prob_slice = index["prob"][opponent_id]
                    jac[row, opponent_prob_slice] -= grad
                for l in range(num_opp_profiles):
                    jac[row, eta_start + j * num_opp_profiles + l] = 1.0

            # Primal/dual transport complementarity against eta_i[k,l].
            own_expected_payoff = z[prob_slice] @ payoffs
            for k in range(num_opp_profiles):
                for l in range(num_opp_profiles):
                    row = eta_start + k * num_opp_profiles + l
                    f[row] = -xi[k] + own_expected_payoff[l] + lambda_value * distance[k, l]
                    jac[row, xi_slice.start + k] = -1.0
                    jac[row, prob_slice] = payoffs[:, l]
                    jac[row, lambda_idx] = distance[k, l]

            # Simplex equation.
            f[kappa_idx] = 1.0 - float(np.sum(z[prob_slice]))
            jac[kappa_idx, prob_slice] = -1.0

        return f, jac

    @staticmethod
    def _fill_dense_csc(jac, col_start, col_len, row, data):
        n = jac.shape[0]
        idx = 0
        for col in range(n):
            col_start[col] = idx + 1
            col_len[col] = n
            for r in range(n):
                row[idx] = r + 1
                data[idx] = jac[r, col]
                idx += 1

    def _solve_from_start(self, z, q_tensor, epsilon, index, opponent_data, payoffs_by_player):
        n_vars = z.shape[0]
        nnz = n_vars * n_vars
        f = np.zeros(n_vars, dtype=np.float64)
        lb = np.full(n_vars, -INF, dtype=np.float64)
        ub = np.full(n_vars, INF, dtype=np.float64)
        for player_id in range(q_tensor.shape[-1]):
            lb[index["prob"][player_id]] = 0.0
            lb[index["lambda"][player_id]] = 0.0
            lb[index["eta"][player_id]] = 0.0

        def func_eval(n, z_ptr, f_ptr):
            z_view = np.ctypeslib.as_array(z_ptr, shape=(n,))
            f_view = np.ctypeslib.as_array(f_ptr, shape=(n,))
            f_value, _ = self._compute_f_and_jacobian(
                z_view, q_tensor, epsilon, index, opponent_data, payoffs_by_player
            )
            f_view[:] = f_value
            return 0

        def jac_eval(n, nnz_in, z_ptr, col_start_ptr, col_len_ptr, row_ptr, data_ptr):
            z_view = np.ctypeslib.as_array(z_ptr, shape=(n,))
            col_start = np.ctypeslib.as_array(col_start_ptr, shape=(n,))
            col_len = np.ctypeslib.as_array(col_len_ptr, shape=(n,))
            row = np.ctypeslib.as_array(row_ptr, shape=(nnz_in,))
            data = np.ctypeslib.as_array(data_ptr, shape=(nnz_in,))
            _, jac = self._compute_f_and_jacobian(
                z_view, q_tensor, epsilon, index, opponent_data, payoffs_by_player
            )
            self._fill_dense_csc(jac, col_start, col_len, row, data)
            return 0

        status = self.path_solver.solve(n_vars, nnz, z, f, lb, ub, func_eval, jac_eval)
        return status, z

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
        q_tensor = validate_nplayer_q_tensor(q_tensor)
        action_sizes = q_tensor.shape[:-1]
        index, opponent_data, _ = self._build_index(action_sizes)
        payoffs_by_player = [
            self._payoff_by_own_and_opponent_profile(q_tensor, player_id, opponent_data[player_id][1])
            for player_id in range(q_tensor.shape[-1])
        ]

        starts = self._initial_starts(
            index, action_sizes, opponent_data, num_repeats, include_pure_starts
        )
        candidates = []
        messages = []

        for z0 in starts:
            try:
                status, z_sol = self._solve_from_start(
                    z0, q_tensor, epsilon, index, opponent_data, payoffs_by_player
                )
            except Exception as exc:
                messages.append(str(exc))
                continue
            if status not in (1, 2):
                messages.append(f"PATH status {status}")
                continue
            policies = self._policies_from_z(z_sol, index, action_sizes)
            if policies is None:
                continue
            exploitability, player_gaps, robust_values = robust_exploitability(
                q_tensor, policies, epsilon
            )
            nominal = _expected_nominal_values(q_tensor, policies)
            robust_policy_values = [
                float(policy @ values) for policy, values in zip(policies, robust_values)
            ]
            candidates.append(
                {
                    "policies": policies,
                    "robust_exploitability": float(exploitability),
                    "player_robust_gaps": [float(gap) for gap in player_gaps],
                    "robust_policy_values": robust_policy_values,
                    "nominal_values": [float(value) for value in nominal],
                    "joint_nominal_welfare": float(np.sum(nominal)),
                    "status": int(status),
                }
            )

        elapsed = time.perf_counter() - start
        if not candidates:
            policies = _uniform_nplayer_policies(q_tensor)
            exploitability, player_gaps, robust_values = robust_exploitability(
                q_tensor, policies, epsilon
            )
            nominal = _expected_nominal_values(q_tensor, policies)
            return SreSolveResult(
                policies=[],
                solutions=[],
                utilities_sr=[],
                utilities_nominal=[],
                success=False,
                message="No N-player PATH MCP solution was returned. " + "; ".join(messages[:3]),
                metadata={
                    "solver": self.name,
                    "algorithm_family": "path_mcp_multilinear_complementarity",
                    "epsilon": float(epsilon),
                    "num_agents": int(q_tensor.shape[-1]),
                    "action_sizes": [int(size) for size in action_sizes],
                    "num_starts": int(len(starts)),
                    "wall_seconds": float(elapsed),
                    "fallback_uniform_robust_exploitability": float(exploitability),
                    "fallback_uniform_player_robust_gaps": [float(gap) for gap in player_gaps],
                    "fallback_uniform_nominal_values": [float(value) for value in nominal],
                },
            )

        candidates.sort(
            key=lambda candidate: (
                candidate["robust_exploitability"],
                -candidate["joint_nominal_welfare"],
            )
        )
        best = candidates[0]
        solution = _solution_dict_from_policies(best["policies"], round_digits=round_digits)
        return SreSolveResult(
            policies=best["policies"],
            solutions=[solution],
            utilities_sr=[best["robust_policy_values"]],
            utilities_nominal=[best["nominal_values"]],
            success=bool(best["robust_exploitability"] <= 1e-4),
            message="" if best["robust_exploitability"] <= 1e-4 else "Returned best PATH MCP candidate.",
            metadata={
                "solver": self.name,
                "algorithm_family": "path_mcp_multilinear_complementarity",
                "epsilon": float(epsilon),
                "num_agents": int(q_tensor.shape[-1]),
                "action_sizes": [int(size) for size in action_sizes],
                "num_starts": int(len(starts)),
                "num_candidates": int(len(candidates)),
                "path_status": best["status"],
                "wall_seconds": float(elapsed),
                "robust_exploitability": best["robust_exploitability"],
                "player_robust_gaps": best["player_robust_gaps"],
                "robust_policy_values": best["robust_policy_values"],
                "nominal_values": best["nominal_values"],
                "joint_nominal_welfare": best["joint_nominal_welfare"],
            },
        )

    def get_solve_time_summary(self):
        return self.path_solver.get_solve_time_summary()

    def close(self):
        self.path_solver.close()
