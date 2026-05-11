import numpy as np
import pytest

from sre_solvers import (
    PathCBimatrixSreSolver,
    PathMcpNPlayerSreSolver,
    make_sre_solver,
    robust_action_values,
    robust_exploitability,
)


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


def test_factory_only_exposes_path_mcp_for_nplayer():
    solver = make_sre_solver("path_mcp_nplayer", random_seed=5)
    try:
        assert isinstance(solver, PathMcpNPlayerSreSolver)
    finally:
        solver.close()

    for removed_name in (
        "baseline_nplayer",
        "dca_bl_nplayer",
        "sbb_nplayer",
        "warm_start_nplayer",
        "smoothing_newton_nplayer",
    ):
        with pytest.raises(ValueError):
            make_sre_solver(removed_name)


@pytest.mark.integration
def test_path_mcp_nplayer_returns_valid_candidate_on_pure_dominant_game(path_runtime_available):
    q_tensor = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q_tensor[1, :, :, 0] = 2.0
    q_tensor[:, 1, :, 1] = 3.0
    q_tensor[:, :, 1, 2] = 4.0

    solver = PathMcpNPlayerSreSolver(
        pathwrap_path=path_runtime_available,
        random_seed=5,
    )
    try:
        result = solver.solve(q_tensor, epsilon=0.0, num_repeats=8, round_digits=None)
    finally:
        solver.close()

    assert result.success, result.message
    gap, _, _ = robust_exploitability(q_tensor, result.policies, epsilon=0.0)
    assert gap <= 1e-4
    assert result.metadata["algorithm_family"] == "path_mcp_multilinear_complementarity"


@pytest.mark.integration
def test_path_mcp_nplayer_matches_path_c_on_unique_mixed_bimatrix(path_runtime_available):
    q_tensor = np.array(
        [
            [[1.0, -1.0], [-1.0, 1.0]],
            [[-1.0, 1.0], [1.0, -1.0]],
        ],
        dtype=np.float64,
    )

    path_solver = PathCBimatrixSreSolver(pathwrap_path=path_runtime_available)
    mcp_solver = PathMcpNPlayerSreSolver(
        pathwrap_path=path_runtime_available,
        random_seed=4,
    )
    try:
        path_result = path_solver.solve(q_tensor, epsilon=0.0, num_repeats=3, round_digits=None)
        mcp_result = mcp_solver.solve(q_tensor, epsilon=0.0, num_repeats=10, round_digits=None)
    finally:
        path_solver.close()
        mcp_solver.close()

    assert path_result.success, path_result.message
    assert mcp_result.success, mcp_result.message
    assert np.allclose(mcp_result.policies[0], path_result.policies[0], atol=1e-3)
    assert np.allclose(mcp_result.policies[1], path_result.policies[1], atol=1e-3)


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
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgentConfig
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            agent_id=0,
            obs_dim=5,
            num_agents=3,
            num_actions=2,
            epsilon_explore=0.0,
            learning_starts=99,
            use_gpu=False,
            sre_solver=FakeSolver(),
        )
    )
    try:
        q = agent.q_net(
            __import__("torch").zeros((4, 5), dtype=__import__("torch").float32)
        )
        assert tuple(q.shape) == (4, 2, 2, 2, 3)
        assert agent.act(np.zeros(5, dtype=np.float32), agent_id=0) == 0
    finally:
        agent.close()
