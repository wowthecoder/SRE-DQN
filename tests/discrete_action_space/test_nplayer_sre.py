import numpy as np
import pytest

from sre_solvers import (
    IterativeNPlayerSreSolver,
    PathCBimatrixSreSolver,
    SmoothingNewtonNPlayerSreSolver,
    make_sre_solver,
    robust_action_values,
    robust_exploitability,
)
from sre_solvers.smoothing_newton_nplayer import smooth_min, smooth_min_partials


def test_tv_robust_values_reduce_to_nominal_at_zero_epsilon():
    q_tensor = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q_tensor[:, :, :, 0] = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    policies = [
        np.array([0.25, 0.75]),
        np.array([0.4, 0.6]),
        np.array([0.7, 0.3]),
    ]

    values = robust_action_values(q_tensor, policies, epsilon=0.0, player_id=0)
    expected = []
    for action in range(2):
        payoff = q_tensor[action, :, :, 0]
        expected.append(float(np.einsum("j,k,jk->", policies[1], policies[2], payoff)))

    assert np.allclose(values, expected)


def test_nplayer_solver_finds_pure_dominant_profile():
    q_tensor = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q_tensor[1, :, :, 0] = 2.0
    q_tensor[:, 1, :, 1] = 3.0
    q_tensor[:, :, 1, 2] = 4.0

    solver = IterativeNPlayerSreSolver(max_iter=100, tol=1e-6, temperature=0.0)
    result = solver.solve(q_tensor, epsilon=0.0, num_repeats=4, round_digits=None)

    assert result.success
    assert len(result.policies) == 3
    assert np.allclose(result.policies[0], [0.0, 1.0], atol=1e-3)
    assert np.allclose(result.policies[1], [0.0, 1.0], atol=1e-3)
    assert np.allclose(result.policies[2], [0.0, 1.0], atol=1e-3)
    gap, _, _ = robust_exploitability(q_tensor, result.policies, epsilon=0.0)
    assert gap <= 1e-5


def test_smooth_min_approaches_exact_min_and_stays_finite():
    a = np.array([-1000.0, -0.2, 1.0, 1000.0], dtype=np.float64)
    b = np.array([999.0, 0.4, -3.0, 1001.0], dtype=np.float64)

    values = smooth_min(1e-8, a, b)

    assert np.all(np.isfinite(values))
    assert np.allclose(values, np.minimum(a, b), atol=1e-6)


def test_smooth_min_partials_match_finite_differences():
    t = 0.3
    a = np.array([0.7], dtype=np.float64)
    b = np.array([-0.1], dtype=np.float64)
    eps = 1e-6

    _, partial_t, partial_a, partial_b = smooth_min_partials(t, a, b)
    numeric_t = (smooth_min(t + eps, a, b) - smooth_min(t - eps, a, b)) / (2 * eps)
    numeric_a = (smooth_min(t, a + eps, b) - smooth_min(t, a - eps, b)) / (2 * eps)
    numeric_b = (smooth_min(t, a, b + eps) - smooth_min(t, a, b - eps)) / (2 * eps)

    assert np.allclose(partial_t, numeric_t, atol=1e-6)
    assert np.allclose(partial_a, numeric_a, atol=1e-6)
    assert np.allclose(partial_b, numeric_b, atol=1e-6)


def test_smoothing_newton_nplayer_solver_handles_single_action_game():
    q_tensor = np.ones((1, 1, 1, 3), dtype=np.float64)

    solver = SmoothingNewtonNPlayerSreSolver(max_iter=5, tol=1e-6, random_seed=1)
    result = solver.solve(q_tensor, epsilon=0.0, num_repeats=1, round_digits=None)

    assert result.success, result.message
    assert len(result.policies) == 3
    assert all(np.allclose(policy, [1.0]) for policy in result.policies)
    gap, _, _ = robust_exploitability(q_tensor, result.policies, epsilon=0.0)
    assert gap <= 1e-8


def test_smoothing_newton_nplayer_solver_finds_pure_dominant_profile():
    q_tensor = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q_tensor[1, :, :, 0] = 2.0
    q_tensor[:, 1, :, 1] = 3.0
    q_tensor[:, :, 1, 2] = 4.0

    solver = SmoothingNewtonNPlayerSreSolver(max_iter=20, tol=1e-6, random_seed=2)
    result = solver.solve(q_tensor, epsilon=0.0, num_repeats=8, round_digits=None)

    assert result.success, result.message
    assert np.allclose(result.policies[0], [0.0, 1.0], atol=1e-3)
    assert np.allclose(result.policies[1], [0.0, 1.0], atol=1e-3)
    assert np.allclose(result.policies[2], [0.0, 1.0], atol=1e-3)
    assert result.metadata["algorithm_family"] == "evlcp_inspired_smoothing_newton_nplayer_mcp"


def test_make_sre_solver_exposes_nplayer_variants():
    for name in [
        "baseline_nplayer",
        "dca_bl_nplayer",
        "sbb_nplayer",
        "warm_start_nplayer",
        "smoothing_newton_nplayer",
        "evlcp_smoothing_nplayer",
        "smoothing_newton",
    ]:
        solver = make_sre_solver(name, max_iter=2)
        assert solver.name in {
            name,
            "warm_start_nplayer",
            "smoothing_newton_nplayer",
        }


def test_smoothing_newton_no_worse_than_baseline_on_simple_nplayer_game():
    q_tensor = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q_tensor[1, :, :, 0] = 2.0
    q_tensor[:, 1, :, 1] = 3.0
    q_tensor[:, :, 1, 2] = 4.0

    baseline = IterativeNPlayerSreSolver(max_iter=100, tol=1e-6, temperature=0.0, random_seed=3)
    smoothing = SmoothingNewtonNPlayerSreSolver(max_iter=20, tol=1e-6, random_seed=3)

    baseline_result = baseline.solve(q_tensor, epsilon=0.0, num_repeats=8)
    smoothing_result = smoothing.solve(q_tensor, epsilon=0.0, num_repeats=8)

    baseline_gap, _, _ = robust_exploitability(q_tensor, baseline_result.policies, epsilon=0.0)
    smoothing_gap, _, _ = robust_exploitability(q_tensor, smoothing_result.policies, epsilon=0.0)
    assert smoothing_gap <= baseline_gap + 1e-6


@pytest.mark.integration
def test_smoothing_newton_matches_path_c_on_unique_mixed_bimatrix(path_runtime_available):
    q_tensor = np.array(
        [
            [[1.0, -1.0], [-1.0, 1.0]],
            [[-1.0, 1.0], [1.0, -1.0]],
        ],
        dtype=np.float64,
    )

    path_solver = PathCBimatrixSreSolver(pathwrap_path=path_runtime_available)
    smoothing_solver = SmoothingNewtonNPlayerSreSolver(max_iter=100, tol=1e-6, random_seed=4)
    try:
        path_result = path_solver.solve(q_tensor, epsilon=0.0, num_repeats=3, round_digits=None)
        smoothing_result = smoothing_solver.solve(
            q_tensor, epsilon=0.0, num_repeats=10, round_digits=None
        )
    finally:
        path_solver.close()

    assert path_result.success, path_result.message
    assert smoothing_result.success, smoothing_result.message
    assert np.allclose(smoothing_result.policies[0], path_result.policies[0], atol=1e-3)
    assert np.allclose(smoothing_result.policies[1], path_result.policies[1], atol=1e-3)


def test_dueling_double_dqn_supports_three_agents_with_fake_solver():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent
    from sre_solvers import SreSolveResult

    class FakeSolver:
        name = "fake_nplayer"

        def solve(self, q_tensor, epsilon, *, num_repeats=20, round_digits=4, include_pure_starts=True):
            del epsilon, num_repeats, round_digits, include_pure_starts
            assert q_tensor.shape == (2, 2, 2, 3)
            return SreSolveResult(
                policies=[
                    np.array([1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([0.5, 0.5]),
                ],
                solutions=[],
                utilities_sr=[],
                utilities_nominal=[],
                success=True,
            )

        def close(self):
            pass

    del torch
    agent = DuelingDoubleDqnSreAgent(
        agent_id=0,
        obs_dim=5,
        num_agents=3,
        num_actions=2,
        epsilon_explore=0.0,
        learning_starts=99,
        use_gpu=False,
        sre_solver=FakeSolver(),
    )
    try:
        q = agent.q_net(
            __import__("torch").zeros((4, 5), dtype=__import__("torch").float32)
        )
        assert tuple(q.shape) == (4, 2, 2, 2, 3)
        assert agent.act(np.zeros(5, dtype=np.float32), agent_id=0) == 0
    finally:
        agent.close()
