import numpy as np
import pytest

from sre_solvers import (
    IterativeNPlayerSreSolver,
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


def test_make_sre_solver_exposes_nplayer_variants():
    for name in [
        "baseline_nplayer",
        "dca_bl_nplayer",
        "sbb_nplayer",
        "warm_start_nplayer",
    ]:
        solver = make_sre_solver(name, max_iter=2)
        assert solver.name in {name, "warm_start_nplayer"}


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
