import numpy as np
import pytest

from sre_solvers import (
    LemkeLcpBimatrixSreSolver,
    PathCBimatrixSreSolver,
    SreSolveResult,
)
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
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent

    solver = _FakeSolver()
    agent = DuelingDoubleDqnSreAgent(
        agent_id=0,
        obs_dim=4,
        num_agents=2,
        num_actions=2,
        epsilon_explore=0.0,
        learning_starts=999,
        use_gpu=False,
        sre_solver=solver,
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
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent

    solver = _FakeSolver(name="solver_a")
    agent = DuelingDoubleDqnSreAgent(
        agent_id=0,
        obs_dim=4,
        num_agents=2,
        num_actions=2,
        use_gpu=False,
        sre_solver=solver,
    )
    try:
        q_tensor = np.zeros((2, 2, 2), dtype=np.float32)
        key_a = agent._sre_cache_key(q_tensor)
        solver.name = "solver_b"
        key_b = agent._sre_cache_key(q_tensor)
        assert key_a != key_b
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
