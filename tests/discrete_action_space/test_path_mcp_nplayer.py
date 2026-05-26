import numpy as np
import pytest

from sre_solvers import (
    PathCBimatrixSreSolver,
    PathMcpNPlayerSreSolver,
    PathTvcMcpNPlayerSreSolver,
    ProcessPoolPathMcpNPlayerSreSolver,
    ProcessPoolPathTvcMcpNPlayerSreSolver,
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

    solver = make_sre_solver("path_tvc_mcp_nplayer", random_seed=5)
    try:
        assert isinstance(solver, PathTvcMcpNPlayerSreSolver)
    finally:
        solver.close()

    solver = make_sre_solver("path_mcp_nplayer_pool", max_workers=1, random_seed=5)
    try:
        assert isinstance(solver, ProcessPoolPathMcpNPlayerSreSolver)
    finally:
        solver.close()

    solver = make_sre_solver("path_tvc_mcp_nplayer_pool", max_workers=1, random_seed=5)
    try:
        assert isinstance(solver, ProcessPoolPathTvcMcpNPlayerSreSolver)
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


def test_path_tvc_mcp_nplayer_structure_is_linear_in_opponent_profiles():
    _, _, generic_vars = PathMcpNPlayerSreSolver._build_index((6, 6, 6, 6))
    _, _, tvc_vars = PathTvcMcpNPlayerSreSolver._build_index((6, 6, 6, 6))

    assert generic_vars == 187520
    assert tvc_vars == 3492


def test_path_tvc_mcp_nplayer_dense_jacobian_matches_finite_difference():
    rng = np.random.default_rng(7)
    q_tensor = rng.normal(size=(2, 3, 2))
    solver = object.__new__(PathTvcMcpNPlayerSreSolver)
    solver._structure_cache = {}
    structure = solver._structure_for_action_sizes(q_tensor.shape[:-1])
    payoffs_by_player = [
        solver._payoff_by_own_and_opponent_profile(
            q_tensor,
            player_id,
            structure["opponent_data"][player_id][1],
        )
        for player_id in range(q_tensor.shape[-1])
    ]
    z = rng.normal(size=structure["n_vars"])
    for player_id, action_size in enumerate(q_tensor.shape[:-1]):
        z[structure["index"]["prob"][player_id]] = rng.dirichlet(np.ones(action_size))
        z[structure["index"]["lambda"][player_id]] = abs(z[structure["index"]["lambda"][player_id]]) + 0.1
        for key in ("alpha", "beta", "gamma"):
            z[structure["index"][key][player_id]] = np.abs(z[structure["index"][key][player_id]]) + 0.1

    f, jac = solver._compute_f_and_jacobian(
        z,
        q_tensor,
        0.25,
        structure["index"],
        structure["opponent_data"],
        payoffs_by_player,
        compute_jac=True,
    )
    eps = 1e-6
    for col in range(jac.shape[1]):
        z_hi = z.copy()
        z_lo = z.copy()
        z_hi[col] += eps
        z_lo[col] -= eps
        f_hi, _ = solver._compute_f_and_jacobian(
            z_hi,
            q_tensor,
            0.25,
            structure["index"],
            structure["opponent_data"],
            payoffs_by_player,
            compute_jac=False,
        )
        f_lo, _ = solver._compute_f_and_jacobian(
            z_lo,
            q_tensor,
            0.25,
            structure["index"],
            structure["opponent_data"],
            payoffs_by_player,
            compute_jac=False,
        )
        numeric = (f_hi - f_lo) / (2 * eps)
        assert np.allclose(jac[:, col], numeric, atol=1e-5)


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
def test_path_tvc_mcp_nplayer_returns_valid_candidate_on_pure_dominant_game(path_runtime_available):
    q_tensor = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q_tensor[1, :, :, 0] = 2.0
    q_tensor[:, 1, :, 1] = 3.0
    q_tensor[:, :, 1, 2] = 4.0

    solver = PathTvcMcpNPlayerSreSolver(
        pathwrap_path=path_runtime_available,
        random_seed=5,
    )
    try:
        result = solver.solve(q_tensor, epsilon=0.0, num_repeats=2, round_digits=None)
    finally:
        solver.close()

    assert result.success, result.message
    gap, _, _ = robust_exploitability(q_tensor, result.policies, epsilon=0.0)
    assert gap <= 1e-4
    assert result.metadata["algorithm_family"] == "path_tvc_mcp_multilinear_complementarity"


def test_path_mcp_nplayer_pool_batches_pure_dominant_game():
    q_tensor = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q_tensor[1, :, :, 0] = 2.0
    q_tensor[:, 1, :, 1] = 3.0
    q_tensor[:, :, 1, 2] = 4.0

    solver = ProcessPoolPathMcpNPlayerSreSolver(max_workers=1, random_seed=5)
    try:
        results = solver.solve_batch(
            [q_tensor, q_tensor],
            epsilon=0.0,
            num_repeats=2,
            round_digits=None,
        )
        summary = solver.get_solve_time_summary()
    finally:
        solver.close()

    assert len(results) == 2
    assert all(result.success for result in results)
    assert all(
        result.metadata["solver"] == "path_mcp_nplayer_pool"
        for result in results
    )
    assert summary["count"] == 2
    assert np.allclose(results[0].policies[0], [0.0, 1.0])
    assert np.allclose(results[0].policies[1], [0.0, 1.0])
    assert np.allclose(results[0].policies[2], [0.0, 1.0])


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


@pytest.mark.integration
def test_path_tvc_mcp_nplayer_matches_path_c_on_unique_mixed_bimatrix(path_runtime_available):
    q_tensor = np.array(
        [
            [[1.0, -1.0], [-1.0, 1.0]],
            [[-1.0, 1.0], [1.0, -1.0]],
        ],
        dtype=np.float64,
    )

    path_solver = PathCBimatrixSreSolver(pathwrap_path=path_runtime_available)
    tvc_solver = PathTvcMcpNPlayerSreSolver(
        pathwrap_path=path_runtime_available,
        random_seed=4,
    )
    try:
        path_result = path_solver.solve(q_tensor, epsilon=0.0, num_repeats=3, round_digits=None)
        tvc_result = tvc_solver.solve(q_tensor, epsilon=0.0, num_repeats=10, round_digits=None)
    finally:
        path_solver.close()
        tvc_solver.close()

    assert path_result.success, path_result.message
    assert tvc_result.success, tvc_result.message
    assert np.allclose(tvc_result.policies[0], path_result.policies[0], atol=1e-3)
    assert np.allclose(tvc_result.policies[1], path_result.policies[1], atol=1e-3)


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
