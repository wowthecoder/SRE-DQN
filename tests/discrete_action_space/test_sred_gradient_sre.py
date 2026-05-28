import numpy as np
import pytest


torch = pytest.importorskip("torch")

from sre_solvers import SredGradientSreSolver, make_sre_solver
from sre_solvers.nplayer_common import robust_exploitability


def _rps_q_tensor():
    u1 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
    return np.stack([u1, -u1], axis=-1)


def _dominant_q_tensor_2p():
    q = np.zeros((2, 2, 2), dtype=np.float64)
    q[1, :, 0] = 2.0
    q[:, 1, 1] = 3.0
    return q


def _dominant_q_tensor_3p():
    q = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q[1, :, :, 0] = 2.0
    q[:, 1, :, 1] = 3.0
    q[:, :, 1, 2] = 4.0
    return q


def test_sred_gradient_solver_factory_aliases():
    for name in ("sred_gradient_sre", "sred_gd_sre", "sred_gd"):
        solver = make_sre_solver(name, max_iters=0, random_seed=7, device="cpu")
        try:
            assert isinstance(solver, SredGradientSreSolver)
            assert solver.name == "sred_gradient_sre"
        finally:
            solver.close()


def test_sred_gradient_rejects_invalid_tensor():
    solver = make_sre_solver("sred_gd_sre", max_iters=0, device="cpu")
    try:
        with pytest.raises(ValueError):
            solver.solve(np.zeros((2, 2), dtype=np.float64), epsilon=0.0)
    finally:
        solver.close()


def test_sred_gradient_returns_uniform_on_rps_at_zero_epsilon():
    solver = make_sre_solver(
        "sred_gd_sre",
        max_iters=5,
        eval_every=1,
        random_seed=1,
        device="cpu",
    )
    try:
        result = solver.solve(
            _rps_q_tensor(),
            epsilon=0.0,
            num_repeats=0,
            include_pure_starts=False,
            exploitability_tol=1e-8,
            round_digits=None,
        )
    finally:
        solver.close()

    assert result.success, result.message
    np.testing.assert_allclose(result.policies[0], np.full(3, 1 / 3), atol=1e-8)
    np.testing.assert_allclose(result.policies[1], np.full(3, 1 / 3), atol=1e-8)
    gap, _, _ = robust_exploitability(_rps_q_tensor(), result.policies, epsilon=0.0)
    assert gap <= 1e-8
    assert result.metadata["algorithm_family"] == "smoothed_sred_gradient"


def test_sred_gradient_finds_2p_dominant_actions_from_pure_starts():
    solver = make_sre_solver("sred_gd_sre", max_iters=0, random_seed=2, device="cpu")
    try:
        result = solver.solve(
            _dominant_q_tensor_2p(),
            epsilon=0.0,
            num_repeats=4,
            include_pure_starts=True,
            exploitability_tol=1e-6,
            round_digits=None,
        )
    finally:
        solver.close()

    assert result.success, result.message
    assert result.policies[0][1] > 1.0 - 1e-6
    assert result.policies[1][1] > 1.0 - 1e-6
    assert result.utilities_sr[0][0] == pytest.approx(
        result.metadata["robust_policy_values"][0]
    )


def test_sred_gradient_finds_3p_dominant_actions_from_pure_starts():
    q_tensor = _dominant_q_tensor_3p()
    solver = make_sre_solver("sred_gd_sre", max_iters=0, random_seed=3, device="cpu")
    try:
        result = solver.solve(
            q_tensor,
            epsilon=0.0,
            num_repeats=8,
            include_pure_starts=True,
            exploitability_tol=1e-6,
            round_digits=None,
        )
    finally:
        solver.close()

    assert result.success, result.message
    assert result.policies[0][1] > 1.0 - 1e-6
    assert result.policies[1][1] > 1.0 - 1e-6
    assert result.policies[2][1] > 1.0 - 1e-6


def test_sred_gradient_warm_start_does_not_worsen_exact_gap():
    q_tensor = _dominant_q_tensor_2p()
    cold_solver = make_sre_solver(
        "sred_gd_sre", max_iters=5, eval_every=1, random_seed=4, device="cpu"
    )
    warm_solver = make_sre_solver(
        "sred_gd_sre", max_iters=5, eval_every=1, random_seed=4, device="cpu"
    )
    try:
        cold = cold_solver.solve(
            q_tensor,
            epsilon=0.0,
            num_repeats=1,
            include_pure_starts=False,
            round_digits=None,
        )
        warm = warm_solver.solve(
            q_tensor,
            epsilon=0.0,
            num_repeats=1,
            include_pure_starts=False,
            initial_policies=cold.policies,
            round_digits=None,
        )
    finally:
        cold_solver.close()
        warm_solver.close()

    assert warm.metadata["robust_exploitability"] <= (
        cold.metadata["robust_exploitability"] + 1e-9
    )


def test_sred_gradient_batch_and_torch_batch_match():
    q_batch = np.stack([_rps_q_tensor(), _rps_q_tensor()], axis=0)
    kwargs = dict(max_iters=0, random_seed=5, device="cpu")
    numpy_solver = make_sre_solver("sred_gd_sre", **kwargs)
    torch_solver = make_sre_solver("sred_gd_sre", **kwargs)
    try:
        numpy_results = numpy_solver.solve_batch(
            q_batch,
            epsilon=np.zeros(2),
            num_repeats=0,
            include_pure_starts=False,
            round_digits=None,
        )
        torch_results = torch_solver.solve_batch_torch(
            torch.as_tensor(q_batch, dtype=torch.float32),
            epsilon=torch.zeros(2),
            num_repeats=0,
            include_pure_starts=False,
            round_digits=None,
        )
    finally:
        numpy_solver.close()
        torch_solver.close()

    assert len(numpy_results) == len(torch_results) == 2
    for numpy_result, torch_result in zip(numpy_results, torch_results):
        assert numpy_result.metadata["robust_exploitability"] == pytest.approx(
            torch_result.metadata["robust_exploitability"]
        )
        for lhs, rhs in zip(numpy_result.policies, torch_result.policies):
            np.testing.assert_allclose(lhs, rhs, atol=1e-8)


def test_sred_gradient_torch_batch_uses_vectorized_path(monkeypatch):
    q_batch = np.stack([_rps_q_tensor(), _rps_q_tensor() + 0.25], axis=0)
    solver = make_sre_solver(
        "sred_gd_sre",
        max_iters=1,
        eval_every=1,
        random_seed=8,
        device="cpu",
    )

    def fail_solve_batch(*args, **kwargs):
        raise AssertionError("solve_batch_torch should not delegate to solve_batch")

    monkeypatch.setattr(solver, "solve_batch", fail_solve_batch)
    try:
        results = solver.solve_batch_torch(
            torch.as_tensor(q_batch, dtype=torch.float32),
            epsilon=torch.zeros(2),
            num_repeats=0,
            include_pure_starts=False,
            round_digits=None,
        )
    finally:
        solver.close()

    assert len(results) == 2
    assert all(result.metadata["batched_torch"] for result in results)
    for result in results:
        for policy in result.policies:
            assert np.isclose(policy.sum(), 1.0)
            assert np.all(policy >= 0.0)


def test_sred_gradient_optimizes_inside_no_grad_context():
    q_tensor = np.random.default_rng(9).normal(size=(2, 2, 2)).astype(np.float32)
    solver = make_sre_solver(
        "sred_gd_sre", max_iters=1, eval_every=1, random_seed=9, device="cpu"
    )
    try:
        with torch.no_grad():
            result = solver.solve(
                q_tensor,
                epsilon=0.1,
                num_repeats=0,
                include_pure_starts=False,
                round_digits=None,
            )
    finally:
        solver.close()

    assert result.metadata["iterations"] == 1
    assert len(result.policies) == 2
    assert all(np.isclose(policy.sum(), 1.0) for policy in result.policies)


def test_deep_srq_accepts_sred_gradient_solver_for_three_agents():
    from dueling_double_dqn_sre import (
        DuelingDoubleDqnSreAgent,
        DuelingDoubleDqnSreAgentConfig,
    )

    solver = make_sre_solver("sred_gd_sre", max_iters=0, random_seed=6, device="cpu")
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
            sre_num_random_starts=8,
        )
    )
    try:
        q = agent.q_net(torch.zeros((4, 5), dtype=torch.float32))
        assert tuple(q.shape) == (4, 2, 2, 2, 3)
        actions = agent.act_joint(np.zeros(5, dtype=np.float32))
        assert len(actions) == 3
    finally:
        agent.close()


def test_lbf_helper_instantiates_sred_gradient_solver():
    from lbf_grid.deep_srq_lbf import _make_solver

    solver = _make_solver(
        "sred_gd_sre",
        {
            "sred_max_iters": 0,
            "sred_lr": 0.01,
            "sred_eval_every": 1,
            "sred_device": "cpu",
        },
        seed=7,
    )
    try:
        assert isinstance(solver, SredGradientSreSolver)
    finally:
        solver.close()
