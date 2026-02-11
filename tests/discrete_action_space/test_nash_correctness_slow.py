import numpy as np
import pytest

from tests.discrete_action_space.math_assertions import (
    assert_is_nash,
    run_nash_solver,
    select_lowest_gap_solution,
)


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _to_arrays(sol):
    return np.asarray(sol["p1"], dtype=np.float64), np.asarray(sol["p2"], dtype=np.float64)


def _strategy_pair_close(sol_pair, candidate_pair, atol):
    p1_sol, p2_sol = sol_pair
    p1_can, p2_can = candidate_pair
    return np.allclose(p1_sol, p1_can, atol=atol) and np.allclose(p2_sol, p2_can, atol=atol)


def _oracle_equilibria_2x2(U1, U2, tol=1e-10):
    U1 = np.asarray(U1, dtype=np.float64)
    U2 = np.asarray(U2, dtype=np.float64)
    equilibria = []

    for i in range(2):
        for j in range(2):
            p1_best = U1[i, j] >= np.max(U1[:, j]) - tol
            p2_best = U2[i, j] >= np.max(U2[i, :]) - tol
            if p1_best and p2_best:
                p1 = np.zeros(2, dtype=np.float64)
                p2 = np.zeros(2, dtype=np.float64)
                p1[i] = 1.0
                p2[j] = 1.0
                equilibria.append((p1, p2))

    den_a = U1[0, 0] - U1[0, 1] - U1[1, 0] + U1[1, 1]
    den_b = U2[0, 0] - U2[1, 0] - U2[0, 1] + U2[1, 1]
    if abs(den_a) > tol and abs(den_b) > tol:
        q = (U1[1, 1] - U1[0, 1]) / den_a
        p = (U2[1, 1] - U2[1, 0]) / den_b
        if -tol <= p <= 1.0 + tol and -tol <= q <= 1.0 + tol:
            p = float(np.clip(p, 0.0, 1.0))
            q = float(np.clip(q, 0.0, 1.0))
            equilibria.append(
                (
                    np.array([p, 1.0 - p], dtype=np.float64),
                    np.array([q, 1.0 - q], dtype=np.float64),
                )
            )

    dedup = []
    for cand in equilibria:
        if not any(_strategy_pair_close(cand, existing, atol=1e-7) for existing in dedup):
            dedup.append(cand)
    return dedup


def test_multi_equilibrium_coordination_game_subset_known_equilibria(path_solver):
    U1 = np.array([[4.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    U2 = U1.copy()
    known_equilibria = [
        (np.array([1.0, 0.0]), np.array([1.0, 0.0])),
        (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        (np.array([1.0 / 3.0, 2.0 / 3.0]), np.array([1.0 / 3.0, 2.0 / 3.0])),
    ]

    solutions = run_nash_solver(path_solver, U1, U2, num_repeats=30, seed=3, round_digits=None)
    assert solutions, "Expected at least one PATH solution for coordination game."

    for sol in solutions:
        p1, p2 = _to_arrays(sol)
        assert_is_nash(U1, U2, p1, p2)
        assert any(
            _strategy_pair_close((p1, p2), candidate, atol=2e-2)
            for candidate in known_equilibria
        ), f"Solution is not close to any known equilibrium: p1={p1}, p2={p2}"


def test_action_permutation_invariance_unique_nash(path_solver):
    U1 = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    U2 = -U1

    baseline = run_nash_solver(path_solver, U1, U2, num_repeats=18, seed=5, round_digits=None)
    p1_base, p2_base, base_gap = select_lowest_gap_solution(U1, U2, baseline)
    assert base_gap <= 1e-6

    row_perm = np.array([1, 0])
    col_perm = np.array([1, 0])
    U1_perm = U1[row_perm][:, col_perm]
    U2_perm = U2[row_perm][:, col_perm]

    permuted = run_nash_solver(
        path_solver, U1_perm, U2_perm, num_repeats=18, seed=6, round_digits=None
    )
    p1_perm, p2_perm, perm_gap = select_lowest_gap_solution(U1_perm, U2_perm, permuted)
    assert perm_gap <= 1e-6

    p1_unpermuted = p1_perm[np.argsort(row_perm)]
    p2_unpermuted = p2_perm[np.argsort(col_perm)]

    assert np.allclose(p1_unpermuted, p1_base, atol=1e-4)
    assert np.allclose(p2_unpermuted, p2_base, atol=1e-4)


def test_positive_affine_payoff_invariance(path_solver):
    U1 = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    U2 = -U1

    transformed_U1 = 2.5 * U1 + 3.0
    transformed_U2 = 1.7 * U2 - 2.0

    base_solutions = run_nash_solver(path_solver, U1, U2, num_repeats=18, seed=9, round_digits=None)
    tr_solutions = run_nash_solver(
        path_solver, transformed_U1, transformed_U2, num_repeats=18, seed=10, round_digits=None
    )

    p1_base, p2_base, base_gap = select_lowest_gap_solution(U1, U2, base_solutions)
    p1_tr, p2_tr, tr_gap = select_lowest_gap_solution(transformed_U1, transformed_U2, tr_solutions)
    assert base_gap <= 1e-6
    assert tr_gap <= 1e-6

    assert np.allclose(p1_base, p1_tr, atol=1e-4)
    assert np.allclose(p2_base, p2_tr, atol=1e-4)


def test_random_2x2_games_match_analytic_oracle(path_solver):
    rng = np.random.default_rng(2026)
    checked = 0

    for idx in range(10):
        U1 = rng.uniform(-2.0, 2.0, size=(2, 2))
        U2 = rng.uniform(-2.0, 2.0, size=(2, 2))

        oracle = _oracle_equilibria_2x2(U1, U2)
        if not oracle:
            continue

        solutions = run_nash_solver(
            path_solver, U1, U2, num_repeats=25, seed=200 + idx, round_digits=None
        )
        assert solutions, f"Expected PATH solutions for random game {idx}."

        for sol in solutions:
            p1, p2 = _to_arrays(sol)
            assert_is_nash(U1, U2, p1, p2)

        matched = False
        for sol in solutions:
            p1, p2 = _to_arrays(sol)
            if any(_strategy_pair_close((p1, p2), candidate, atol=2e-2) for candidate in oracle):
                matched = True
                break

        assert matched, f"PATH solution did not match any oracle equilibrium for game {idx}."
        checked += 1

    assert checked >= 8, "Too many random games were skipped due oracle degeneracy."
