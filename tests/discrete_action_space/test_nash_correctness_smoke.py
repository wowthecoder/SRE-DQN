import numpy as np
import pytest

from tests.discrete_action_space.math_assertions import assert_is_nash, run_nash_solver


pytestmark = pytest.mark.integration


def test_prisoners_dilemma_unique_pure_nash(path_solver):
    # Action order: [Cooperate, Defect]
    U1 = np.array([[3.0, 0.0], [5.0, 1.0]], dtype=np.float64)
    U2 = np.array([[3.0, 5.0], [0.0, 1.0]], dtype=np.float64)

    solutions = run_nash_solver(path_solver, U1, U2, num_repeats=12, seed=11, round_digits=None)
    assert solutions, "Expected at least one PATH solution for Prisoner's Dilemma."

    for sol in solutions:
        p1 = np.asarray(sol["p1"], dtype=np.float64)
        p2 = np.asarray(sol["p2"], dtype=np.float64)
        assert_is_nash(U1, U2, p1, p2)
        assert np.allclose(p1, np.array([0.0, 1.0]), atol=1e-4)
        assert np.allclose(p2, np.array([0.0, 1.0]), atol=1e-4)


def test_matching_pennies_unique_mixed_nash(path_solver):
    U1 = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    U2 = -U1

    solutions = run_nash_solver(path_solver, U1, U2, num_repeats=16, seed=7, round_digits=None)
    assert solutions, "Expected at least one PATH solution for Matching Pennies."

    target = np.array([0.5, 0.5], dtype=np.float64)
    for sol in solutions:
        p1 = np.asarray(sol["p1"], dtype=np.float64)
        p2 = np.asarray(sol["p2"], dtype=np.float64)
        assert_is_nash(U1, U2, p1, p2)
        assert np.allclose(p1, target, atol=1e-4)
        assert np.allclose(p2, target, atol=1e-4)


def test_rock_paper_scissors_uniform_nash(path_solver):
    U1 = np.array(
        [
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    U2 = -U1

    solutions = run_nash_solver(path_solver, U1, U2, num_repeats=20, seed=21, round_digits=None)
    assert solutions, "Expected at least one PATH solution for Rock-Paper-Scissors."

    target = np.full(3, 1.0 / 3.0, dtype=np.float64)
    for sol in solutions:
        p1 = np.asarray(sol["p1"], dtype=np.float64)
        p2 = np.asarray(sol["p2"], dtype=np.float64)
        assert_is_nash(U1, U2, p1, p2)
        assert np.allclose(p1, target, atol=1e-4)
        assert np.allclose(p2, target, atol=1e-4)
