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
        )
    )
    try:
        action = agent.act(np.zeros(4, dtype=np.float32), agent_id=0)
        assert action == 0
        assert solver.calls == 1
    finally:
        agent.close()
    assert solver.closed


def test_sre_cache_key_separates_solver_names():
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
        key_a = agent._sre_cache_key(q_tensor)
        solver.name = "solver_b"
        key_b = agent._sre_cache_key(q_tensor)
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
        assert summary["approx_candidates"] == 1
        assert summary["uniform_fallbacks"] == 0
        assert summary["candidate_robust_exploitability"]["p95"] == pytest.approx(5e-3)
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
        assert summary["rejected_candidates"] == 1
        assert summary["uniform_fallbacks"] == 1
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
        assert summary["unfiltered_candidates"] == 1
        assert summary["rejected_candidates"] == 0
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
        assert summary["target_equilibrium_refreshes"] == 2
        assert summary["target_equilibrium_cache_only_steps"] == 3
        assert summary["exact_hits"] == 3
        assert summary["forced_refreshes"] == 1
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
        assert summary["stores"] == 0
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
