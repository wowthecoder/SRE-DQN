import itertools
from functools import lru_cache
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

try:
    from path_solver import PathSolverWrapper
except ImportError:  # Package import from repository root.
    from ...path_solver import PathSolverWrapper

from ..base import (
    SreSolveResult,
    SreStageGameSolver,
    _empty_duration_summary,
    _normalize_policy,
)
from ..nplayer_common import (
    _expected_nominal_values,
    _solution_dict_from_policies,
    _uniform_nplayer_policies,
    robust_exploitability,
    validate_nplayer_q_tensor,
)


DEFAULT_PATHWRAP_PATH = Path(__file__).resolve().parent.parent / "pathwrap.so"
INF = 1e20


class PathMcpNPlayerSreSolver(SreStageGameSolver):
    """PATH-backed MCP solver for finite-action N-player SRE stage games.

    This ports the JuMP/PATH formulation in
    `strategically-robust-game-theory/sr_games_julia/solve_sr_N_player_game.jl`.
    Unlike the two-player PATH backend, this is not an LCP: for N > 2 the
    product distribution over opponents makes the MCP multilinear.
    """

    name = "path_mcp_nplayer"

    def __init__(self, pathwrap_path=DEFAULT_PATHWRAP_PATH, random_seed=None):
        self.path_solver = PathSolverWrapper(pathwrap_path)
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def _opponent_profiles(action_sizes, player_id):
        opponent_ids = [idx for idx in range(len(action_sizes)) if idx != player_id]
        profiles = list(itertools.product(*[range(action_sizes[idx]) for idx in opponent_ids]))
        return opponent_ids, profiles

    @staticmethod
    def _payoff_by_own_and_opponent_profile(q_tensor, player_id, profiles):
        del profiles
        payoff_tensor = np.asarray(q_tensor[..., player_id], dtype=np.float64)
        return np.moveaxis(payoff_tensor, player_id, 0).reshape(q_tensor.shape[player_id], -1)

    @staticmethod
    @lru_cache(maxsize=64)
    def _distance_matrix(num_opp_profiles):
        distance = np.ones((int(num_opp_profiles), int(num_opp_profiles)), dtype=np.float64)
        np.fill_diagonal(distance, 0.0)
        return distance

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
        for player_id, action_size in enumerate(action_sizes):
            if policies is None:
                policy = self.rng.dirichlet(np.ones(action_size, dtype=np.float64))
            else:
                policy = np.asarray(policies[player_id], dtype=np.float64)
            z[index["prob"][player_id]] = policy
            z[index["lambda"][player_id]] = self.rng.random() + 1e-2
            xi_slice = index["xi"][player_id]
            eta_slice = index["eta"][player_id]
            z[xi_slice] = 100.0 * self.rng.random(xi_slice.stop - xi_slice.start) - 50.0
            z[eta_slice] = self.rng.random(eta_slice.stop - eta_slice.start)
            z[index["kappa"][player_id]] = 100.0 * self.rng.random() - 50.0
        return z

    @staticmethod
    def _normalize_initial_policies(action_sizes, policies):
        if policies is None or len(policies) != len(action_sizes):
            return None
        normalized = []
        for action_size, policy in zip(action_sizes, policies):
            p = _normalize_policy(policy)
            if p is None or p.shape[0] != action_size:
                return None
            normalized.append(p)
        return normalized

    def _initial_starts(
        self,
        index,
        action_sizes,
        opponent_data,
        num_repeats,
        include_pure_starts,
        initial_policies=None,
    ):
        starts = []
        initial_policies = self._normalize_initial_policies(
            action_sizes, initial_policies
        )
        if initial_policies is not None:
            starts.append(
                self._make_start(index, action_sizes, opponent_data, initial_policies)
            )
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

    def _compute_f_and_jacobian(
        self,
        z,
        q_tensor,
        epsilon,
        index,
        opponent_data,
        payoffs_by_player,
        distances_by_player=None,
        compute_jac=True,
    ):
        action_sizes = q_tensor.shape[:-1]
        n_vars = z.shape[0]
        f = np.zeros(n_vars, dtype=np.float64)
        jac = np.zeros((n_vars, n_vars), dtype=np.float64) if compute_jac else None

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
            if distances_by_player is None:
                distance = self._distance_matrix(num_opp_profiles)
            else:
                distance = distances_by_player[player_id]

            # d / d(prob_i): -kappa_i - sum_{k,l} eta_i[k,l] payoff_i(a_i,l)
            eta_col_sum = eta.sum(axis=0)
            f[prob_slice] = -z[kappa_idx] - payoffs @ eta_col_sum
            if compute_jac:
                jac[prob_slice, kappa_idx] = -1.0
                jac[prob_slice, eta_slice] = -np.broadcast_to(
                    payoffs[:, None, :],
                    (action_size, num_opp_profiles, num_opp_profiles),
                ).reshape(action_size, num_opp_profiles * num_opp_profiles)

            # d / d(lambda_i): epsilon_i - transport cost.
            f[lambda_idx] = float(epsilon) - float(np.sum(eta * distance))
            if compute_jac:
                jac[lambda_idx, eta_slice] = -distance.reshape(-1)

            opponent_distribution, distribution_grads = self._opponent_distribution_and_gradients(
                z, index, player_id, opponent_ids, profiles
            )

            # d / d(xi_i[j]): -prod opponents prob(profile_j) + sum_l eta_i[j,l]
            f[xi_slice] = -opponent_distribution + eta.sum(axis=1)
            if compute_jac:
                eta_start = eta_slice.start
                for j, profile_grads in enumerate(distribution_grads):
                    row = xi_slice.start + j
                    for opponent_id, grad in profile_grads:
                        opponent_prob_slice = index["prob"][opponent_id]
                        jac[row, opponent_prob_slice] -= grad
                    row_slice = slice(
                        eta_start + j * num_opp_profiles,
                        eta_start + (j + 1) * num_opp_profiles,
                    )
                    jac[row, row_slice] = 1.0

            # Primal/dual transport complementarity against eta_i[k,l].
            own_expected_payoff = z[prob_slice] @ payoffs
            f[eta_slice] = (
                -xi[:, None] + own_expected_payoff[None, :] + lambda_value * distance
            ).reshape(-1)
            if compute_jac:
                jac[eta_slice, prob_slice] = np.broadcast_to(
                    payoffs.T[None, :, :],
                    (num_opp_profiles, num_opp_profiles, action_size),
                ).reshape(num_opp_profiles * num_opp_profiles, action_size)
                jac[eta_slice, lambda_idx] = distance.reshape(-1)
                eta_start = eta_slice.start
                for k in range(num_opp_profiles):
                    row_slice = slice(
                        eta_start + k * num_opp_profiles,
                        eta_start + (k + 1) * num_opp_profiles,
                    )
                    jac[row_slice, xi_slice.start + k] = -1.0

            # Simplex equation.
            f[kappa_idx] = 1.0 - float(np.sum(z[prob_slice]))
            if compute_jac:
                jac[kappa_idx, prob_slice] = -1.0

        return f, jac

    @staticmethod
    def _build_sparse_jacobian_pattern(index, action_sizes, opponent_data, distances_by_player):
        n_vars = index["kappa"][-1] + 1
        entries_by_col = [[] for _ in range(n_vars)]

        def add(row, col, kind, payload):
            entries_by_col[int(col)].append((int(row), kind, payload))

        for player_id, action_size in enumerate(action_sizes):
            prob_slice = index["prob"][player_id]
            lambda_idx = index["lambda"][player_id]
            xi_slice = index["xi"][player_id]
            eta_slice = index["eta"][player_id]
            kappa_idx = index["kappa"][player_id]
            opponent_ids, profiles = opponent_data[player_id]
            num_opp_profiles = len(profiles)
            distance = distances_by_player[player_id]

            for action_id in range(action_size):
                row = prob_slice.start + action_id
                add(row, kappa_idx, "const", -1.0)
                for k in range(num_opp_profiles):
                    for l in range(num_opp_profiles):
                        col = eta_slice.start + k * num_opp_profiles + l
                        add(row, col, "neg_payoff", (player_id, action_id, l))

            for k in range(num_opp_profiles):
                for l in range(num_opp_profiles):
                    col = eta_slice.start + k * num_opp_profiles + l
                    add(lambda_idx, col, "const", -float(distance[k, l]))

            for profile_idx, profile in enumerate(profiles):
                row = xi_slice.start + profile_idx
                for local_idx, opponent_id in enumerate(opponent_ids):
                    action_id = profile[local_idx]
                    col = index["prob"][opponent_id].start + action_id
                    add(
                        row,
                        col,
                        "neg_profile_grad",
                        (tuple(opponent_ids), tuple(profile), local_idx),
                    )
                for l in range(num_opp_profiles):
                    col = eta_slice.start + profile_idx * num_opp_profiles + l
                    add(row, col, "const", 1.0)

            for k in range(num_opp_profiles):
                for l in range(num_opp_profiles):
                    row = eta_slice.start + k * num_opp_profiles + l
                    for action_id in range(action_size):
                        col = prob_slice.start + action_id
                        add(row, col, "payoff", (player_id, action_id, l))
                    add(row, lambda_idx, "const", float(distance[k, l]))
                    add(row, xi_slice.start + k, "const", -1.0)

            for action_id in range(action_size):
                add(kappa_idx, prob_slice.start + action_id, "const", -1.0)

        for col_entries in entries_by_col:
            col_entries.sort(key=lambda entry: entry[0])
        return entries_by_col

    @staticmethod
    def _fill_sparse_csc(z, payoffs_by_player, index, entries_by_col, col_start, col_len, row, data):
        idx = 0
        for col, entries in enumerate(entries_by_col):
            col_start[col] = idx + 1
            col_len[col] = len(entries)
            for row_id, kind, payload in entries:
                row[idx] = row_id + 1
                if kind == "const":
                    value = payload
                elif kind == "payoff":
                    player_id, action_id, opp_profile_id = payload
                    value = payoffs_by_player[player_id][action_id, opp_profile_id]
                elif kind == "neg_payoff":
                    player_id, action_id, opp_profile_id = payload
                    value = -payoffs_by_player[player_id][action_id, opp_profile_id]
                elif kind == "neg_profile_grad":
                    opponent_ids, profile, local_idx = payload
                    partial = 1.0
                    for other_idx, opponent_id in enumerate(opponent_ids):
                        if other_idx == local_idx:
                            continue
                        action_id = profile[other_idx]
                        partial *= z[index["prob"][opponent_id].start + action_id]
                    value = -partial
                else:
                    raise ValueError(f"Unknown sparse Jacobian entry kind: {kind}.")
                data[idx] = value
                idx += 1
        if idx != data.shape[0]:
            raise ValueError(
                f"Sparse Jacobian filled {idx} entries, expected {data.shape[0]}."
            )

    @staticmethod
    def _result_from_candidate(
        *,
        q_tensor,
        policies,
        epsilon,
        round_digits,
        start,
        metadata,
        success_tol,
    ):
        exploitability, player_gaps, robust_values = robust_exploitability(
            q_tensor, policies, epsilon
        )
        nominal = _expected_nominal_values(q_tensor, policies)
        robust_policy_values = [
            float(policy @ values) for policy, values in zip(policies, robust_values)
        ]
        success = bool(exploitability <= success_tol)
        result_metadata = {
            "solver": PathMcpNPlayerSreSolver.name,
            "algorithm_family": "path_mcp_multilinear_complementarity",
            "epsilon": float(epsilon),
            "exploitability_tol": float(success_tol),
            "num_agents": int(q_tensor.shape[-1]),
            "action_sizes": [int(size) for size in q_tensor.shape[:-1]],
            "wall_seconds": float(time.perf_counter() - start),
            "robust_exploitability": float(exploitability),
            "player_robust_gaps": [float(gap) for gap in player_gaps],
            "robust_policy_values": robust_policy_values,
            "nominal_values": [float(value) for value in nominal],
            "joint_nominal_welfare": float(np.sum(nominal)),
        }
        result_metadata.update(metadata)
        return SreSolveResult(
            policies=policies,
            solutions=[_solution_dict_from_policies(policies, round_digits=round_digits)],
            utilities_sr=[robust_policy_values],
            utilities_nominal=[[float(value) for value in nominal]],
            success=success,
            message="" if success else "Returned best PATH MCP candidate.",
            metadata=result_metadata,
        )

    @staticmethod
    def _dominant_action_policies(payoffs_by_player, tol=1e-12):
        policies = []
        for payoffs in payoffs_by_player:
            action_size = payoffs.shape[0]
            dominant_action = None
            for action_id in range(action_size):
                if np.all(payoffs[action_id][None, :] >= payoffs - tol):
                    dominant_action = action_id
                    break
            if dominant_action is None:
                return None
            policy = np.zeros(action_size, dtype=np.float64)
            policy[dominant_action] = 1.0
            policies.append(policy)
        return policies

    def _solve_from_start(
        self,
        z,
        q_tensor,
        epsilon,
        index,
        opponent_data,
        payoffs_by_player,
        distances_by_player=None,
        sparse_pattern=None,
    ):
        n_vars = z.shape[0]
        if sparse_pattern is None:
            sparse_pattern = self._build_sparse_jacobian_pattern(
                index, q_tensor.shape[:-1], opponent_data, distances_by_player
            )
        nnz = sum(len(entries) for entries in sparse_pattern)
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
                z_view,
                q_tensor,
                epsilon,
                index,
                opponent_data,
                payoffs_by_player,
                distances_by_player,
                compute_jac=False,
            )
            f_view[:] = f_value
            return 0

        def jac_eval(n, nnz_in, z_ptr, col_start_ptr, col_len_ptr, row_ptr, data_ptr):
            z_view = np.ctypeslib.as_array(z_ptr, shape=(n,))
            col_start = np.ctypeslib.as_array(col_start_ptr, shape=(n,))
            col_len = np.ctypeslib.as_array(col_len_ptr, shape=(n,))
            row = np.ctypeslib.as_array(row_ptr, shape=(nnz_in,))
            data = np.ctypeslib.as_array(data_ptr, shape=(nnz_in,))
            self._fill_sparse_csc(
                z_view,
                payoffs_by_player,
                index,
                sparse_pattern,
                col_start,
                col_len,
                row,
                data,
            )
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
        initial_policies=None,
        exploitability_tol=1e-4,
        early_exit=True,
    ):
        start = time.perf_counter()
        q_tensor = validate_nplayer_q_tensor(q_tensor)
        action_sizes = q_tensor.shape[:-1]
        index, opponent_data, _ = self._build_index(action_sizes)
        payoffs_by_player = [
            self._payoff_by_own_and_opponent_profile(q_tensor, player_id, opponent_data[player_id][1])
            for player_id in range(q_tensor.shape[-1])
        ]
        distances_by_player = [
            self._distance_matrix(len(opponent_data[player_id][1]))
            for player_id in range(q_tensor.shape[-1])
        ]
        sparse_pattern = self._build_sparse_jacobian_pattern(
            index, action_sizes, opponent_data, distances_by_player
        )

        initial_policies = self._normalize_initial_policies(
            action_sizes, initial_policies
        )
        if initial_policies is not None and early_exit:
            initial_gap, _, _ = robust_exploitability(
                q_tensor, initial_policies, epsilon
            )
            if initial_gap <= exploitability_tol:
                return self._result_from_candidate(
                    q_tensor=q_tensor,
                    policies=initial_policies,
                    epsilon=epsilon,
                    round_digits=round_digits,
                    start=start,
                    success_tol=exploitability_tol,
                    metadata={
                        "path_shortcut": "initial_policies_verified",
                        "num_starts": 0,
                        "num_candidates": 1,
                        "early_exit": True,
                    },
                )

        dominant_policies = self._dominant_action_policies(payoffs_by_player)
        if dominant_policies is not None:
            return self._result_from_candidate(
                q_tensor=q_tensor,
                policies=dominant_policies,
                epsilon=epsilon,
                round_digits=round_digits,
                start=start,
                success_tol=exploitability_tol,
                metadata={
                    "path_shortcut": "dominant_actions",
                    "num_starts": 0,
                    "num_candidates": 1,
                    "early_exit": True,
                },
            )

        starts = self._initial_starts(
            index,
            action_sizes,
            opponent_data,
            num_repeats,
            include_pure_starts,
            initial_policies=initial_policies,
        )
        candidates = []
        messages = []
        attempted_starts = 0

        for z0 in starts:
            attempted_starts += 1
            try:
                status, z_sol = self._solve_from_start(
                    z0,
                    q_tensor,
                    epsilon,
                    index,
                    opponent_data,
                    payoffs_by_player,
                    distances_by_player,
                    sparse_pattern,
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
            if early_exit and exploitability <= exploitability_tol:
                return self._result_from_candidate(
                    q_tensor=q_tensor,
                    policies=policies,
                    epsilon=epsilon,
                    round_digits=round_digits,
                    start=start,
                    success_tol=exploitability_tol,
                    metadata={
                        "num_starts": int(len(starts)),
                        "num_starts_attempted": int(attempted_starts),
                        "num_candidates": int(len(candidates)),
                        "path_status": int(status),
                        "early_exit": True,
                    },
                )

        elapsed = time.perf_counter() - start
        if not candidates:
            policies = _uniform_nplayer_policies(q_tensor)
            exploitability, player_gaps, robust_values = robust_exploitability(
                q_tensor, policies, epsilon
            )
            nominal = _expected_nominal_values(q_tensor, policies)
            robust_policy_values = [
                float(policy @ values) for policy, values in zip(policies, robust_values)
            ]
            return SreSolveResult(
                policies=policies,
                solutions=[_solution_dict_from_policies(policies, round_digits=round_digits)],
                utilities_sr=[robust_policy_values],
                utilities_nominal=[[float(value) for value in nominal]],
                success=False,
                message="PATH MCP failed to return a candidate. " + "; ".join(messages[:3]),
                metadata={
                    "solver": self.name,
                    "algorithm_family": "path_mcp_multilinear_complementarity",
                    "path_failed": True,
                    "path_failure_messages": messages[:3],
                    "epsilon": float(epsilon),
                    "num_agents": int(q_tensor.shape[-1]),
                    "action_sizes": [int(size) for size in action_sizes],
                    "num_starts": int(len(starts)),
                    "num_starts_attempted": int(attempted_starts),
                    "num_candidates": 0,
                    "wall_seconds": float(elapsed),
                    "robust_exploitability": float(exploitability),
                    "player_robust_gaps": [float(gap) for gap in player_gaps],
                    "robust_policy_values": robust_policy_values,
                    "nominal_values": [float(value) for value in nominal],
                    "joint_nominal_welfare": float(np.sum(nominal)),
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
            success=bool(best["robust_exploitability"] <= exploitability_tol),
            message="" if best["robust_exploitability"] <= exploitability_tol else "Returned best PATH MCP candidate.",
            metadata={
                "solver": self.name,
                "algorithm_family": "path_mcp_multilinear_complementarity",
                "epsilon": float(epsilon),
                "exploitability_tol": float(exploitability_tol),
                "num_agents": int(q_tensor.shape[-1]),
                "action_sizes": [int(size) for size in action_sizes],
                "num_starts": int(len(starts)),
                "num_starts_attempted": int(attempted_starts),
                "num_candidates": int(len(candidates)),
                "path_status": best["status"],
                "wall_seconds": float(elapsed),
                "robust_exploitability": best["robust_exploitability"],
                "player_robust_gaps": best["player_robust_gaps"],
                "robust_policy_values": best["robust_policy_values"],
                "nominal_values": best["nominal_values"],
                "joint_nominal_welfare": best["joint_nominal_welfare"],
                "early_exit": False,
            },
        )

    def get_solve_time_summary(self):
        return self.path_solver.get_solve_time_summary()

    def close(self):
        self.path_solver.close()


_NPLAYER_POOL_SOLVER = None


def _path_mcp_nplayer_pool_initializer(pathwrap_path, random_seed):
    global _NPLAYER_POOL_SOLVER
    _NPLAYER_POOL_SOLVER = PathMcpNPlayerSreSolver(
        pathwrap_path=pathwrap_path,
        random_seed=random_seed,
    )


def _path_mcp_nplayer_pool_solve_task(payload):
    (
        q_tensor,
        epsilon,
        num_repeats,
        round_digits,
        include_pure_starts,
        initial_policies,
        exploitability_tol,
        early_exit,
        task_seed,
    ) = payload
    if task_seed is not None:
        _NPLAYER_POOL_SOLVER.rng = np.random.default_rng(task_seed)

    start = time.perf_counter()
    result = _NPLAYER_POOL_SOLVER.solve(
        q_tensor,
        epsilon,
        num_repeats=num_repeats,
        round_digits=round_digits,
        include_pure_starts=include_pure_starts,
        initial_policies=initial_policies,
        exploitability_tol=exploitability_tol,
        early_exit=early_exit,
    )
    elapsed = time.perf_counter() - start
    result.metadata = dict(result.metadata)
    result.metadata["worker_sre_wall_seconds"] = float(elapsed)
    return result


class ProcessPoolPathMcpNPlayerSreSolver(SreStageGameSolver):
    """Multiprocessing batch wrapper for PATH-backed N-player MCP solves."""

    name = "path_mcp_nplayer_pool"

    def __init__(
        self,
        pathwrap_path=DEFAULT_PATHWRAP_PATH,
        max_workers=4,
        start_method=None,
        random_seed=None,
    ):
        self.pathwrap_path = pathwrap_path
        self.max_workers = int(max_workers)
        self.start_method = start_method
        self.random_seed = random_seed
        self._task_counter = 0
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
            initializer=_path_mcp_nplayer_pool_initializer,
            initargs=(self.pathwrap_path, self.random_seed),
        )

    def _record_solve_time(self, duration):
        self.solve_time_count += 1
        self.solve_time_sum += duration
        self.solve_time_sumsq += duration * duration
        if self.solve_time_min is None or duration < self.solve_time_min:
            self.solve_time_min = duration
        if self.solve_time_max is None or duration > self.solve_time_max:
            self.solve_time_max = duration

    def _task_seed(self, batch_offset):
        if self.random_seed is None:
            return None
        return int(self.random_seed) + int(self._task_counter) + int(batch_offset)

    def solve(
        self,
        q_tensor,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
        initial_policies=None,
        exploitability_tol=1e-4,
        early_exit=True,
    ):
        return self.solve_batch(
            [q_tensor],
            epsilon,
            num_repeats=num_repeats,
            round_digits=round_digits,
            include_pure_starts=include_pure_starts,
            initial_policies_batch=[initial_policies],
            exploitability_tol=exploitability_tol,
            early_exit=early_exit,
        )[0]

    def solve_batch(
        self,
        q_tensors,
        epsilon,
        *,
        num_repeats=20,
        round_digits=4,
        include_pure_starts=True,
        initial_policies_batch=None,
        exploitability_tol=1e-4,
        early_exit=True,
    ):
        q_tensors = [validate_nplayer_q_tensor(q_tensor) for q_tensor in q_tensors]
        if not q_tensors:
            return []
        if initial_policies_batch is None:
            initial_policies_batch = [None] * len(q_tensors)
        if len(initial_policies_batch) != len(q_tensors):
            raise ValueError(
                "initial_policies_batch must be None or match q_tensors length."
            )

        payloads = [
            (
                q_tensor,
                float(epsilon),
                int(num_repeats),
                round_digits,
                bool(include_pure_starts),
                initial_policies,
                float(exploitability_tol),
                bool(early_exit),
                self._task_seed(batch_offset),
            )
            for batch_offset, (q_tensor, initial_policies) in enumerate(
                zip(q_tensors, initial_policies_batch)
            )
        ]
        self._task_counter += len(payloads)

        results = self._pool.map(_path_mcp_nplayer_pool_solve_task, payloads)
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
