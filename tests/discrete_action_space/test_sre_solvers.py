import numpy as np
import pytest

from sre_solvers import (
    LemkeLcpBimatrixSreSolver,
    PathCBimatrixSreSolver,
    SreSolveResult,
)
from sre_solvers.n_player.path_mcp_nplayer import _sort_sre_candidates
from bimatrix_game.stats_utils import collect_timing_stats


def _assert_valid_policy_pair(policies):
    assert len(policies) == 2
    for policy in policies:
        policy = np.asarray(policy, dtype=np.float64)
        assert np.all(policy >= -1e-8)
        assert np.isclose(np.sum(policy), 1.0)


class _FakeSolver:
    def __init__(self, name="fake_solver"):
        self.name = name
        self.calls = 0
        self.closed = False

    def solve(self, q_tensor, epsilon, *, num_repeats=20, round_digits=4):
        del epsilon, num_repeats, round_digits
        self.calls += 1
        q_tensor = np.asarray(q_tensor)
        return SreSolveResult(
            policies=[
                np.array([1.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0], dtype=np.float64),
            ],
            solutions=[],
            utilities_sr=[],
            utilities_nominal=[],
            success=q_tensor.shape == (2, 2, 2),
        )

    def get_solve_time_summary(self):
        return {"count": self.calls}

    def close(self):
        self.closed = True


class _RecordingVariableShapeSolver:
    name = "recording_variable_shape_solver"

    def __init__(self):
        self.shapes = []
        self.closed = False

    def solve_batch(self, q_tensors, epsilon, **_kwargs):
        del epsilon
        results = []
        for q_tensor in q_tensors:
            q_tensor = np.asarray(q_tensor)
            self.shapes.append(q_tensor.shape)
            action_sizes = q_tensor.shape[:-1]
            results.append(
                SreSolveResult(
                    policies=[
                        np.full(size, 1.0 / size, dtype=np.float64)
                        for size in action_sizes
                    ],
                    solutions=[],
                    utilities_sr=[],
                    utilities_nominal=[],
                    success=True,
                )
            )
        return results

    def close(self):
        self.closed = True


@pytest.mark.integration
def test_path_c_solver_returns_valid_policies(path_runtime_available):
    q_tensor = np.array(
        [
            [[3.0, 3.0], [0.0, 5.0]],
            [[5.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    solver = PathCBimatrixSreSolver(pathwrap_path=path_runtime_available)
    try:
        result = solver.solve(q_tensor, epsilon=0.0, num_repeats=2, round_digits=None)
        assert result.success, result.message
        _assert_valid_policy_pair(result.policies)
    finally:
        solver.close()


@pytest.mark.integration
def test_path_c_solver_rejects_non_bimatrix_tensor(path_runtime_available):
    solver = PathCBimatrixSreSolver(pathwrap_path=path_runtime_available)
    try:
        with pytest.raises(ValueError):
            solver.solve(np.zeros((2, 2), dtype=np.float64), epsilon=0.0)
    finally:
        solver.close()


def test_lemke_solver_returns_valid_policies_when_installed():
    pytest.importorskip("lemkelcp")
    q_tensor = np.array(
        [
            [[1.0, -1.0], [-1.0, 1.0]],
            [[-1.0, 1.0], [1.0, -1.0]],
        ],
        dtype=np.float64,
    )
    solver = LemkeLcpBimatrixSreSolver()
    result = solver.solve(q_tensor, epsilon=0.0, num_repeats=5, round_digits=None)
    assert result.success, result.message
    _assert_valid_policy_pair(result.policies)
    assert result.metadata["num_repeats_ignored"] == 5


def test_dueling_double_dqn_uses_injected_solver_policy():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    solver = _FakeSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            agent_id=0,
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_explore=0.0,
            learning_starts=999,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
        )
    )
    try:
        action = agent.act(np.zeros(4, dtype=np.float32), agent_id=0)
        assert action == 0
        assert solver.calls == 1
    finally:
        agent.close()
    assert solver.closed


def test_dueling_double_dqn_slices_masked_stage_game_actions():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    solver = _RecordingVariableShapeSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=3,
            epsilon_explore=0.0,
            learning_starts=999,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
            sre_remove_fixed_players=False,
        )
    )
    try:
        action = agent.act_joint(
            np.zeros(4, dtype=np.float32),
            action_masks=[
                np.array([True, False, True]),
                np.array([False, True, False]),
            ],
        )
        assert action[0] in {0, 2}
        assert action[1] == 1
        assert solver.shapes == [(2, 1, 2)]
    finally:
        agent.close()
    assert solver.closed


def test_dueling_double_dqn_removes_fixed_masked_players_from_solver_game():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    solver = _RecordingVariableShapeSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=3,
            num_actions=3,
            epsilon_explore=0.0,
            learning_starts=999,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=False,
            sre_remove_fixed_players=True,
        )
    )
    try:
        q_batch = torch.zeros((1, 3, 3, 3, 3), dtype=torch.float32)
        policies = agent._solve_sre_batch_masked(
            q_batch,
            [
                [
                    np.array([True, False, True]),
                    np.array([False, True, False]),
                    np.array([True, True, False]),
                ]
            ],
        )
        assert solver.shapes == [(2, 2, 2)]
        assert len(policies[0]) == 3
        assert np.allclose(policies[0][1], [0.0, 1.0, 0.0])
    finally:
        agent.close()


def test_dueling_double_dqn_reuses_greedy_masked_warm_policy_on_path_failure():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FailingPathBatchSolver:
        name = "failing_path_batch"

        def __init__(self):
            self.initial_policies_batch = None

        def solve_batch(self, q_tensors, epsilon, **kwargs):
            del epsilon
            self.initial_policies_batch = kwargs.get("initial_policies_batch")
            results = []
            for q_tensor in q_tensors:
                q_tensor = np.asarray(q_tensor)
                results.append(
                    SreSolveResult(
                        policies=[],
                        solutions=[],
                        utilities_sr=[],
                        utilities_nominal=[],
                        success=False,
                        message="PATH failed",
                        metadata={
                            "path_failed": True,
                            "action_sizes": list(q_tensor.shape[:-1]),
                        },
                    )
                )
            return results

        def close(self):
            pass

    solver = FailingPathBatchSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=3,
            num_actions=3,
            epsilon_explore=0.0,
            learning_starts=999,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
            sre_uniform_fallback_enabled=False,
        )
    )
    try:
        q_batch = torch.zeros((1, 3, 3, 3, 3), dtype=torch.float32)
        q_batch[0, 2, :, :, 0] = 10.0
        q_batch[0, :, 1, :, 1] = 5.0
        q_batch[0, :, :, 0, 2] = 3.0
        policies = agent._solve_sre_batch_masked(
            q_batch,
            [
                [
                    np.array([True, False, True]),
                    np.array([True, True, False]),
                    np.array([True, True, False]),
                ]
            ],
        )[0]

        assert solver.initial_policies_batch is not None
        assert np.allclose(policies[0], [0.0, 0.0, 1.0])
        assert np.allclose(policies[1], [0.0, 1.0, 0.0])
        assert np.allclose(policies[2], [1.0, 0.0, 0.0])
        summary = agent.get_sre_cache_summary()
        assert summary["solver_failure_warm_start_reuses"] == 1
        assert summary["uniform_fallbacks"] == 0
    finally:
        agent.close()


def test_dueling_double_dqn_masked_targets_exclude_illegal_profiles():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_robust=0.5,
            use_gpu=False,
            sre_solver=_FakeSolver(),
        )
    )
    try:
        q_tensor = np.zeros((2, 2, 2), dtype=np.float32)
        q_tensor[0, 0, 0] = 10.0
        q_tensor[0, 1, 0] = -100.0
        policies = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ]
        masked = agent._sre_target_values_batch_masked(
            np.asarray([q_tensor]),
            [policies],
            [[np.array([True, False]), np.array([True, False])]],
        )
        unmasked = agent._sre_target_values_batch(np.asarray([q_tensor]), [policies])
        assert masked[0, 0] == pytest.approx(10.0)
        assert unmasked[0, 0] < masked[0, 0]
    finally:
        agent.close()


def test_sre_batch_key_separates_solver_names():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    solver = _FakeSolver(name="solver_a")
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            agent_id=0,
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=solver,
        )
    )
    try:
        q_tensor = np.zeros((2, 2, 2), dtype=np.float32)
        key_a = agent._sre_batch_key(q_tensor)
        solver.name = "solver_b"
        key_b = agent._sre_batch_key(q_tensor)
        assert key_a != key_b
    finally:
        agent.close()


def test_dueling_double_dqn_reuses_policy_cache_for_repeated_state():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    solver = _FakeSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            agent_id=0,
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_explore=0.0,
            learning_starts=999,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
        )
    )
    try:
        state = np.zeros(4, dtype=np.float32)
        assert agent.act(state, agent_id=0) == 0
        assert agent.act(state, agent_id=0) == 0
        summary = agent.get_sre_cache_summary()
        assert solver.calls == 1
        assert summary["misses"] == 1
        assert summary["exact_hits"] == 1
        assert summary["path_solves_avoided"] == 1
    finally:
        agent.close()


def test_dueling_double_dqn_act_joint_solves_once_for_all_agents():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeNPlayerSolver:
        name = "fake_nplayer"

        def __init__(self):
            self.calls = 0

        def solve(self, q_tensor, epsilon, *, num_repeats=20, round_digits=4):
            del epsilon, num_repeats, round_digits
            self.calls += 1
            assert np.asarray(q_tensor).shape == (2, 2, 2, 3)
            return SreSolveResult(
                policies=[
                    np.array([1.0, 0.0]),
                    np.array([0.0, 1.0]),
                    np.array([1.0, 0.0]),
                ],
                solutions=[],
                utilities_sr=[],
                utilities_nominal=[],
                success=True,
            )

        def close(self):
            pass

    solver = FakeNPlayerSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            agent_id=0,
            obs_dim=5,
            num_agents=3,
            num_actions=2,
            epsilon_explore=0.0,
            learning_starts=99,
            use_gpu=False,
            sre_solver=solver,
        )
    )
    try:
        actions = agent.act_joint(np.zeros(5, dtype=np.float32))
        assert actions == [0, 1, 0]
        assert solver.calls == 1
    finally:
        agent.close()


def test_dueling_double_dqn_accepts_approximate_sre_candidate():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=_FakeSolver(),
            sre_approx_accept_tol=1e-2,
            sre_exploitability_filter_enabled=True,
            sre_uniform_fallback_enabled=True,
        )
    )
    try:
        result = SreSolveResult(
            policies=[
                np.array([0.8, 0.2], dtype=np.float64),
                np.array([0.3, 0.7], dtype=np.float64),
            ],
            solutions=[],
            utilities_sr=[],
            utilities_nominal=[],
            success=False,
            metadata={"robust_exploitability": 5e-3},
        )
        policies, cacheable = agent._policies_from_sre_result(result)
        assert cacheable is True
        assert np.allclose(policies[0], [0.8, 0.2])
        summary = agent.get_sre_cache_summary()
        assert summary["candidate_returned"] == 1
        assert summary["uniform_fallbacks"] == 0
    finally:
        agent.close()


def test_dueling_double_dqn_rejects_large_gap_sre_candidate():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=_FakeSolver(),
            sre_approx_accept_tol=1e-2,
            sre_exploitability_filter_enabled=True,
            sre_uniform_fallback_enabled=True,
        )
    )
    try:
        result = SreSolveResult(
            policies=[
                np.array([1.0, 0.0], dtype=np.float64),
                np.array([1.0, 0.0], dtype=np.float64),
            ],
            solutions=[],
            utilities_sr=[],
            utilities_nominal=[],
            success=False,
            metadata={"robust_exploitability": 5e-2},
        )
        policies, cacheable = agent._policies_from_sre_result(result)
        assert cacheable is False
        assert np.allclose(policies[0], [0.5, 0.5])
        summary = agent.get_sre_cache_summary()
        assert summary["uniform_fallbacks"] == 1
    finally:
        agent.close()


def test_dueling_double_dqn_raises_when_uniform_fallback_disabled():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=_FakeSolver(),
            sre_approx_accept_tol=1e-2,
            sre_exploitability_filter_enabled=True,
        )
    )
    try:
        result = SreSolveResult(
            policies=[
                np.array([1.0, 0.0], dtype=np.float64),
                np.array([1.0, 0.0], dtype=np.float64),
            ],
            solutions=[],
            utilities_sr=[],
            utilities_nominal=[],
            success=False,
            metadata={"robust_exploitability": 5e-2},
        )
        with pytest.raises(RuntimeError, match="uniform fallback is disabled"):
            agent._policies_from_sre_result(result)
        summary = agent.get_sre_cache_summary()
        assert summary["uniform_fallback_enabled"] is False
        assert summary["uniform_fallbacks"] == 0
    finally:
        agent.close()


def test_dueling_double_dqn_reuses_warm_policy_when_path_returns_no_candidate():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=_FakeSolver(),
            sre_uniform_fallback_enabled=False,
        )
    )
    try:
        warm_policies = [
            np.array([0.9, 0.1], dtype=np.float64),
            np.array([0.25, 0.75], dtype=np.float64),
        ]
        result = SreSolveResult(
            policies=[
                np.array([0.5, 0.5], dtype=np.float64),
                np.array([0.5, 0.5], dtype=np.float64),
            ],
            solutions=[],
            utilities_sr=[],
            utilities_nominal=[],
            success=False,
            message="PATH MCP failed to return a candidate.",
            metadata={"path_failed": True, "robust_exploitability": 5e-2},
        )
        policies, cacheable = agent._policies_from_sre_result(
            result,
            warm_policies=warm_policies,
        )
        assert cacheable is False
        assert np.allclose(policies[0], warm_policies[0])
        assert np.allclose(policies[1], warm_policies[1])
        summary = agent.get_sre_cache_summary()
        assert summary["solver_failure_warm_start_reuses"] == 1
        assert summary["uniform_fallbacks"] == 0
    finally:
        agent.close()


def test_dueling_double_dqn_can_disable_exploitability_filter():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=_FakeSolver(),
            sre_approx_accept_tol=1e-2,
            sre_exploitability_filter_enabled=False,
        )
    )
    try:
        result = SreSolveResult(
            policies=[
                np.array([1.0, 0.0], dtype=np.float64),
                np.array([1.0, 0.0], dtype=np.float64),
            ],
            solutions=[],
            utilities_sr=[],
            utilities_nominal=[],
            success=False,
            metadata={"robust_exploitability": 5e-2},
        )
        policies, cacheable = agent._policies_from_sre_result(result)
        assert cacheable is True
        assert np.allclose(policies[0], [1.0, 0.0])
        summary = agent.get_sre_cache_summary()
        assert summary["candidate_returned"] == 1
        assert summary["uniform_fallbacks"] == 0
    finally:
        agent.close()


def test_nplayer_path_candidate_selection_can_prefer_joint_welfare():
    candidates = [
        {
            "robust_exploitability": 1e-4,
            "joint_nominal_welfare": 1.0,
        },
        {
            "robust_exploitability": 5e-2,
            "joint_nominal_welfare": 10.0,
        },
    ]

    by_gap = _sort_sre_candidates(candidates, "robust_exploitability")
    by_welfare = _sort_sre_candidates(candidates, "joint_nominal_welfare")

    assert by_gap[0]["joint_nominal_welfare"] == pytest.approx(1.0)
    assert by_welfare[0]["joint_nominal_welfare"] == pytest.approx(10.0)


def test_dueling_double_dqn_robust_values_are_used_for_targets():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_robust=0.5,
            use_gpu=False,
            sre_solver=_FakeSolver(),
        )
    )
    try:
        q_tensor = np.array(
            [
                [[10.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        policies = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ]
        nominal = agent._sre_expected_values(q_tensor, policies)
        robust = agent._sre_robust_values(q_tensor, policies)
        assert nominal[0] == pytest.approx(10.0)
        assert robust[0] == pytest.approx(5.0)
        assert robust[1] == pytest.approx(0.0)
    finally:
        agent.close()


def test_dueling_double_dqn_target_value_mode_switches_between_robust_and_nominal():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    q_tensor = np.zeros((2, 2, 2), dtype=np.float32)
    q_tensor[:, :, 0] = np.array([[0.0, 3.0], [1.0, 1.0]], dtype=np.float32)
    policies = [
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([0.5, 0.5], dtype=np.float32),
    ]

    robust_agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_robust=0.5,
            use_gpu=False,
            sre_solver=_FakeSolver(),
            sre_target_value_mode="robust",
        )
    )
    nominal_agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_robust=0.5,
            use_gpu=False,
            sre_solver=_FakeSolver(),
            sre_target_value_mode="nominal",
        )
    )
    try:
        robust = robust_agent._sre_target_values_batch([q_tensor], [policies])
        nominal = nominal_agent._sre_target_values_batch([q_tensor], [policies])
        assert robust[0, 0] != pytest.approx(nominal[0, 0])
    finally:
        robust_agent.close()
        nominal_agent.close()


def test_dueling_double_dqn_skips_sre_solves_for_terminal_targets():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FailingBatchSolver:
        name = "failing_batch_solver"

        def solve_batch(self, *args, **kwargs):
            raise AssertionError("terminal target rows should not be solved")

        def close(self):
            pass

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            lr=0.0,
            batch_size=1,
            learning_starts=1,
            train_every=1,
            use_gpu=False,
            sre_solver=FailingBatchSolver(),
            sre_policy_cache_enabled=False,
        )
    )
    try:
        loss = agent.update(
            np.zeros(4, dtype=np.float32),
            [0, 1],
            [1.0, 0.0],
            np.ones(4, dtype=np.float32),
            done=np.ones(2, dtype=np.float32),
            batch_size=1,
        )
        assert loss is not None
        assert agent.get_sre_solve_time_summary()["count"] == 0
    finally:
        agent.close()


def test_target_equilibrium_frequency_reuses_cache_between_refreshes():
    pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeBatchSolver:
        name = "fake_batch_solver"

        def __init__(self):
            self.batch_calls = 0
            self.batch_sizes = []
            self.initial_policies_batches = []

        def solve_batch(
            self,
            q_tensors,
            epsilon,
            *,
            num_repeats=20,
            include_pure_starts=True,
            initial_policies_batch=None,
            exploitability_tol=1e-4,
            early_exit=True,
        ):
            del epsilon, num_repeats, include_pure_starts, exploitability_tol, early_exit
            self.batch_calls += 1
            self.batch_sizes.append(len(q_tensors))
            self.initial_policies_batches.append(initial_policies_batch)
            return [
                SreSolveResult(
                    policies=[
                        np.array([1.0, 0.0], dtype=np.float64),
                        np.array([0.0, 1.0], dtype=np.float64),
                    ],
                    solutions=[],
                    utilities_sr=[],
                    utilities_nominal=[],
                    success=True,
                )
                for _ in q_tensors
            ]

        def close(self):
            pass

    solver = FakeBatchSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            lr=0.0,
            batch_size=1,
            learning_starts=1,
            train_every=1,
            target_equilibrium_update_steps=4,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
            sre_approx_cache_enabled=False,
        )
    )
    try:
        state = np.zeros(4, dtype=np.float32)
        next_state = np.ones(4, dtype=np.float32)
        for _ in range(5):
            loss = agent.update(
                state,
                [0, 1],
                [1.0, 0.0],
                next_state,
                done=np.zeros(2, dtype=np.float32),
                batch_size=1,
            )
            assert loss is not None

        summary = agent.get_sre_cache_summary()
        assert solver.batch_calls == 2
        assert solver.batch_sizes == [1, 1]
        assert solver.initial_policies_batches[0][0] is None
        assert solver.initial_policies_batches[1][0] is not None
        assert summary["target_equilibrium_update_steps"] == 4
        assert summary["exact_hits"] == 3
    finally:
        agent.close()


def test_nfg_style_solver_bypasses_deep_srq_policy_cache_keying():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeTorchBatchSolver:
        name = "nfg_transformer_sre"
        bypass_deep_srq_policy_cache = True

        def __init__(self):
            self.batch_calls = 0

        def solve_batch_torch(
            self,
            q_tensors,
            epsilon,
            *,
            num_repeats=20,
            include_pure_starts=True,
            initial_policies_batch=None,
            exploitability_tol=1e-4,
            early_exit=True,
        ):
            del epsilon, num_repeats, include_pure_starts, initial_policies_batch
            del exploitability_tol, early_exit
            self.batch_calls += 1
            assert isinstance(q_tensors, torch.Tensor)
            return [
                SreSolveResult(
                    policies=[
                        np.array([1.0, 0.0], dtype=np.float64),
                        np.array([0.0, 1.0], dtype=np.float64),
                    ],
                    solutions=[],
                    utilities_sr=[],
                    utilities_nominal=[],
                    success=True,
                )
                for _ in range(q_tensors.shape[0])
            ]

        def close(self):
            pass

    solver = FakeTorchBatchSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=solver,
        )
    )
    try:
        agent._sre_batch_key = lambda q_tensor: (_ for _ in ()).throw(
            AssertionError("cache keying should be bypassed")
        )
        q_batch = torch.zeros((3, 2, 2, 2), dtype=torch.float32)
        policies = agent._solve_sre_batch(q_batch, allow_solver=False)
        summary = agent.get_sre_cache_summary()
        assert solver.batch_calls == 1
        assert len(policies) == 3
        assert summary["enabled"] is False
        assert summary["disabled_by_solver"] is True
        assert summary["misses"] == 0
    finally:
        agent.close()


def test_replay_target_uses_vectorized_torch_batch_for_cache_bypass_solver():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeTorchTargetSolver:
        name = "nfg_transformer_sre"
        bypass_deep_srq_policy_cache = True
        trust_approximate_policies = True

        def __init__(self):
            self.batch_calls = 0
            self.batch_shapes = []
            self.solve_calls = 0

        def solve(self, *args, **kwargs):
            del args, kwargs
            self.solve_calls += 1
            raise AssertionError("replay target should use solve_batch_torch")

        def solve_batch_torch(
            self,
            q_tensors,
            epsilon,
            *,
            num_repeats=20,
            include_pure_starts=True,
            initial_policies_batch=None,
            exploitability_tol=1e-4,
            early_exit=True,
        ):
            del epsilon, num_repeats, include_pure_starts, initial_policies_batch
            del exploitability_tol, early_exit
            self.batch_calls += 1
            assert isinstance(q_tensors, torch.Tensor)
            self.batch_shapes.append(tuple(q_tensors.shape))
            return [
                SreSolveResult(
                    policies=[
                        np.array([1.0, 0.0], dtype=np.float64),
                        np.array([0.0, 1.0], dtype=np.float64),
                    ],
                    solutions=[],
                    utilities_sr=[],
                    utilities_nominal=[],
                    success=False,
                )
                for _ in range(q_tensors.shape[0])
            ]

        def close(self):
            pass

    solver = FakeTorchTargetSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_explore=0.0,
            learning_starts=3,
            train_every=1,
            target_equilibrium_update_steps=1,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
            sre_uniform_fallback_enabled=False,
        )
    )
    try:
        for index in range(3):
            state = np.full(4, index, dtype=np.float32)
            next_state = np.full(4, index + 1, dtype=np.float32)
            agent.replay_buffer.push(
                state,
                np.array([0, 1], dtype=np.int64),
                np.array([1.0, 0.0], dtype=np.float32),
                next_state,
                False,
            )

        loss = agent.train_step(batch_size=3)

        assert loss is not None
        assert solver.batch_calls == 1
        assert solver.batch_shapes == [(3, 2, 2, 2)]
        assert solver.solve_calls == 0
    finally:
        agent.close()


def test_replay_target_uses_direct_torch_policy_path_for_nfg_solver():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeDirectTorchPolicySolver:
        name = "nfg_transformer_sre"
        bypass_deep_srq_policy_cache = True
        trust_approximate_policies = True

        def __init__(self):
            self.policy_calls = 0
            self.batch_calls = 0
            self.policy_shapes = []

        def solve_policy_batch_torch(self, q_tensors, epsilon):
            del epsilon
            self.policy_calls += 1
            assert isinstance(q_tensors, torch.Tensor)
            self.policy_shapes.append(tuple(q_tensors.shape))
            batch_size = int(q_tensors.shape[0])
            return [
                torch.tensor([[1.0, 0.0]], dtype=torch.float32).expand(batch_size, -1),
                torch.tensor([[0.0, 1.0]], dtype=torch.float32).expand(batch_size, -1),
            ]

        def solve_batch_torch(self, *args, **kwargs):
            del args, kwargs
            self.batch_calls += 1
            raise AssertionError("direct policy path should skip result wrapping")

        def close(self):
            pass

    solver = FakeDirectTorchPolicySolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_explore=0.0,
            learning_starts=3,
            train_every=1,
            target_equilibrium_update_steps=1,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
            sre_uniform_fallback_enabled=False,
        )
    )
    try:
        for index in range(3):
            agent.replay_buffer.push(
                np.full(4, index, dtype=np.float32),
                np.array([0, 1], dtype=np.int64),
                np.array([1.0, 0.0], dtype=np.float32),
                np.full(4, index + 1, dtype=np.float32),
                False,
            )

        loss = agent.train_step(batch_size=3)

        assert loss is not None
        assert solver.policy_calls == 1
        assert solver.batch_calls == 0
        assert solver.policy_shapes == [(3, 2, 2, 2)]
    finally:
        agent.close()


def test_act_joint_uses_torch_batch_path_for_cache_bypass_solver():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeTorchActionSolver:
        name = "sred_gradient_sre"
        bypass_deep_srq_policy_cache = True

        def __init__(self):
            self.batch_calls = 0
            self.solve_calls = 0

        def solve(self, *args, **kwargs):
            del args, kwargs
            self.solve_calls += 1
            raise AssertionError("act_joint should use solve_batch_torch")

        def solve_batch_torch(
            self,
            q_tensors,
            epsilon,
            *,
            num_repeats=20,
            include_pure_starts=True,
            initial_policies_batch=None,
            exploitability_tol=1e-4,
            early_exit=True,
        ):
            del epsilon, num_repeats, include_pure_starts, initial_policies_batch
            del exploitability_tol, early_exit
            self.batch_calls += 1
            assert isinstance(q_tensors, torch.Tensor)
            assert tuple(q_tensors.shape) == (1, 2, 2, 2)
            return [
                SreSolveResult(
                    policies=[
                        np.array([1.0, 0.0], dtype=np.float64),
                        np.array([0.0, 1.0], dtype=np.float64),
                    ],
                    solutions=[],
                    utilities_sr=[],
                    utilities_nominal=[],
                    success=True,
                )
            ]

        def close(self):
            pass

    solver = FakeTorchActionSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_explore=0.0,
            learning_starts=99,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
        )
    )
    try:
        agent._sre_batch_key = lambda q_tensor: (_ for _ in ()).throw(
            AssertionError("act_joint torch path should bypass cache keying")
        )
        actions = agent.act_joint(np.zeros(4, dtype=np.float32))
        assert actions == [0, 1]
        assert solver.batch_calls == 1
        assert solver.solve_calls == 0
    finally:
        agent.close()


def test_act_uses_torch_batch_path_for_cache_bypass_solver():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeTorchActionSolver:
        name = "sr_adidas_sre"
        bypass_deep_srq_policy_cache = True

        def __init__(self):
            self.batch_calls = 0
            self.solve_calls = 0

        def solve(self, *args, **kwargs):
            del args, kwargs
            self.solve_calls += 1
            raise AssertionError("act should use solve_batch_torch")

        def solve_batch_torch(
            self,
            q_tensors,
            epsilon,
            *,
            num_repeats=20,
            include_pure_starts=True,
            initial_policies_batch=None,
            exploitability_tol=1e-4,
            early_exit=True,
        ):
            del epsilon, num_repeats, include_pure_starts, initial_policies_batch
            del exploitability_tol, early_exit
            self.batch_calls += 1
            assert isinstance(q_tensors, torch.Tensor)
            assert tuple(q_tensors.shape) == (1, 2, 2, 2)
            return [
                SreSolveResult(
                    policies=[
                        np.array([0.0, 1.0], dtype=np.float64),
                        np.array([1.0, 0.0], dtype=np.float64),
                    ],
                    solutions=[],
                    utilities_sr=[],
                    utilities_nominal=[],
                    success=True,
                )
            ]

        def close(self):
            pass

    solver = FakeTorchActionSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            agent_id=0,
            epsilon_explore=0.0,
            learning_starts=99,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
        )
    )
    try:
        agent._sre_batch_key = lambda q_tensor: (_ for _ in ()).throw(
            AssertionError("act torch path should bypass cache keying")
        )
        assert agent.act(np.zeros(4, dtype=np.float32)) == 1
        assert solver.batch_calls == 1
        assert solver.solve_calls == 0
    finally:
        agent.close()


def test_act_joint_uses_direct_torch_policy_path_for_nfg_solver():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeDirectTorchPolicySolver:
        name = "nfg_transformer_sre"
        bypass_deep_srq_policy_cache = True

        def __init__(self):
            self.policy_calls = 0
            self.batch_calls = 0

        def solve_policy_batch_torch(self, q_tensors, epsilon):
            del epsilon
            self.policy_calls += 1
            batch_size = int(q_tensors.shape[0])
            return [
                torch.tensor([[1.0, 0.0]], dtype=torch.float32).expand(batch_size, -1),
                torch.tensor([[0.0, 1.0]], dtype=torch.float32).expand(batch_size, -1),
            ]

        def solve_batch_torch(self, *args, **kwargs):
            del args, kwargs
            self.batch_calls += 1
            raise AssertionError("act_joint should prefer direct policy tensors")

        def close(self):
            pass

    solver = FakeDirectTorchPolicySolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_explore=0.0,
            learning_starts=99,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
        )
    )
    try:
        actions = agent.act_joint(np.zeros(4, dtype=np.float32))
        assert actions == [0, 1]
        assert solver.policy_calls == 1
        assert solver.batch_calls == 0
    finally:
        agent.close()


def test_act_joint_batch_uses_direct_torch_policy_path_for_nfg_solver():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeDirectTorchPolicySolver:
        name = "nfg_transformer_sre"
        bypass_deep_srq_policy_cache = True

        def __init__(self):
            self.policy_calls = 0
            self.batch_calls = 0
            self.policy_shapes = []

        def solve_policy_batch_torch(self, q_tensors, epsilon):
            del epsilon
            self.policy_calls += 1
            self.policy_shapes.append(tuple(q_tensors.shape))
            batch_size = int(q_tensors.shape[0])
            return [
                torch.tensor([[1.0, 0.0]], dtype=torch.float32).expand(batch_size, -1),
                torch.tensor([[0.0, 1.0]], dtype=torch.float32).expand(batch_size, -1),
            ]

        def solve_batch_torch(self, *args, **kwargs):
            del args, kwargs
            self.batch_calls += 1
            raise AssertionError("act_joint_batch should prefer direct policy tensors")

        def close(self):
            pass

    solver = FakeDirectTorchPolicySolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_explore=0.0,
            learning_starts=99,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
        )
    )
    try:
        actions = agent.act_joint_batch(
            [np.zeros(4, dtype=np.float32) for _ in range(3)]
        )
        assert actions == [[0, 1], [0, 1], [0, 1]]
        assert solver.policy_calls == 1
        assert solver.policy_shapes == [(3, 2, 2, 2)]
        assert solver.batch_calls == 0
    finally:
        agent.close()


def test_act_joint_batch_uses_torch_batch_path_for_cache_bypass_solver():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeTorchActionSolver:
        name = "sr_adidas_sre"
        bypass_deep_srq_policy_cache = True

        def __init__(self):
            self.batch_calls = 0
            self.solve_calls = 0
            self.batch_shapes = []

        def solve(self, *args, **kwargs):
            del args, kwargs
            self.solve_calls += 1
            raise AssertionError("act_joint_batch should use solve_batch_torch")

        def solve_batch_torch(self, q_tensors, epsilon, **kwargs):
            del epsilon, kwargs
            self.batch_calls += 1
            self.batch_shapes.append(tuple(q_tensors.shape))
            return [
                SreSolveResult(
                    policies=[
                        np.array([1.0, 0.0], dtype=np.float64),
                        np.array([0.0, 1.0], dtype=np.float64),
                    ],
                    solutions=[],
                    utilities_sr=[],
                    utilities_nominal=[],
                    success=True,
                )
                for _ in range(int(q_tensors.shape[0]))
            ]

        def close(self):
            pass

    solver = FakeTorchActionSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_explore=0.0,
            learning_starts=99,
            use_gpu=False,
            sre_solver=solver,
            sre_policy_cache_enabled=True,
        )
    )
    try:
        agent._sre_batch_key = lambda q_tensor: (_ for _ in ()).throw(
            AssertionError("cache-bypass torch action path should not key policies")
        )
        actions = agent.act_joint_batch(
            [np.zeros(4, dtype=np.float32) for _ in range(4)]
        )
        assert actions == [[0, 1], [0, 1], [0, 1], [0, 1]]
        assert solver.batch_calls == 1
        assert solver.batch_shapes == [(4, 2, 2, 2)]
        assert solver.solve_calls == 0
    finally:
        agent.close()


def test_act_joint_batch_all_explore_skips_solver():
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class ExplodingSolver:
        name = "exploding"
        bypass_deep_srq_policy_cache = True

        def solve_policy_batch_torch(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("all-explore action batch should not solve policies")

        def solve_batch_torch(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("all-explore action batch should not solve results")

        def close(self):
            pass

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            epsilon_explore=1.0,
            learning_starts=99,
            use_gpu=False,
            sre_solver=ExplodingSolver(),
        )
    )
    try:
        actions = agent.act_joint_batch(
            [np.zeros(4, dtype=np.float32) for _ in range(5)]
        )
        assert len(actions) == 5
        assert all(len(action) == 2 for action in actions)
        assert all(all(a in {0, 1} for a in action) for action in actions)
    finally:
        agent.close()


def test_torch_q_batch_is_copied_to_numpy_for_non_torch_solver():
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeNumpyBatchSolver:
        name = "fake_numpy_batch_solver"

        def __init__(self):
            self.seen_numpy = False

        def solve_batch(self, q_tensors, epsilon, **kwargs):
            del epsilon, kwargs
            q_tensors = list(q_tensors)
            self.seen_numpy = all(isinstance(q_tensor, np.ndarray) for q_tensor in q_tensors)
            assert self.seen_numpy
            return [
                SreSolveResult(
                    policies=[
                        np.array([1.0, 0.0], dtype=np.float64),
                        np.array([0.0, 1.0], dtype=np.float64),
                    ],
                    solutions=[],
                    utilities_sr=[],
                    utilities_nominal=[],
                    success=True,
                )
                for _ in q_tensors
            ]

        def close(self):
            pass

    solver = FakeNumpyBatchSolver()
    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=solver,
        )
    )
    try:
        q_batch = torch.zeros((3, 2, 2, 2), dtype=torch.float32)
        policies = agent._solve_sre_batch(q_batch)
        assert solver.seen_numpy is True
        assert len(policies) == 3
    finally:
        agent.close()


def test_collect_timing_stats_can_omit_episode_durations():
    timing = collect_timing_stats(
        [],
        wall_clock_seconds=1.25,
        episode_durations=[0.1, 0.2],
        include_episode_durations=False,
    )
    assert "episode_durations_seconds" not in timing
    assert timing["episode_time"]["count"] == 2
