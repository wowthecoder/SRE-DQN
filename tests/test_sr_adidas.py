"""Tests for the SR-ADIDAS stage-game solver."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DISCRETE = _ROOT / "discrete_action_space"
for _p in (str(_ROOT), str(_DISCRETE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _rps_q_tensor():
    u1 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
    return np.stack([u1, -u1], axis=-1)


def _dominant_q_tensor():
    q = np.zeros((2, 2, 2), dtype=np.float64)
    q[1, :, 0] = 2.0
    q[:, 1, 1] = 3.0
    return q


def test_robust_policy_value_differs_from_mixed_pure_action_values():
    from sre_solvers.nplayer_common import robust_action_values, robust_policy_value

    q = np.zeros((2, 2, 2), dtype=np.float64)
    q[:, :, 0] = np.array([[0.0, 2.0], [2.0, 0.0]])
    q[:, :, 1] = -q[:, :, 0]
    policies = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]

    action_values = robust_action_values(q, policies, epsilon=0.5, player_id=0)
    mixed_action_value = float(policies[0] @ action_values)
    mixed_policy_value = robust_policy_value(q, policies, epsilon=0.5, player_id=0)

    assert mixed_action_value == pytest.approx(0.0)
    assert mixed_policy_value == pytest.approx(1.0)


def test_sr_adidas_solver_factory_aliases():
    from sre_solvers import SrAdidasSreSolver, make_sre_solver

    solver = make_sre_solver("sr_adidas_sre", max_iters=5, random_seed=7)
    assert isinstance(solver, SrAdidasSreSolver)
    assert make_sre_solver("sr_adidas", max_iters=5).name == "sr_adidas_sre"


def test_sr_adidas_returns_uniform_on_rps_at_zero_epsilon():
    from sre_solvers import make_sre_solver
    from sre_solvers.nplayer_common import robust_exploitability

    solver = make_sre_solver(
        "sr_adidas_sre",
        max_iters=30,
        tau_init=1.0,
        tau_min=1e-3,
        random_seed=1,
    )
    result = solver.solve(
        _rps_q_tensor(),
        epsilon=0.0,
        num_repeats=3,
        include_pure_starts=True,
        exploitability_tol=1e-6,
        round_digits=None,
    )

    assert result.success, result.message
    np.testing.assert_allclose(result.policies[0], np.full(3, 1 / 3), atol=1e-6)
    np.testing.assert_allclose(result.policies[1], np.full(3, 1 / 3), atol=1e-6)
    gap, _, _ = robust_exploitability(_rps_q_tensor(), result.policies, 0.0)
    assert gap <= 1e-6


def test_sr_adidas_finds_dominant_actions():
    from sre_solvers import make_sre_solver

    solver = make_sre_solver(
        "sr_adidas_sre",
        max_iters=150,
        lr=0.35,
        tau_init=0.5,
        tau_min=1e-4,
        random_seed=3,
    )
    result = solver.solve(
        _dominant_q_tensor(),
        epsilon=0.0,
        num_repeats=4,
        include_pure_starts=True,
        exploitability_tol=5e-3,
        round_digits=None,
    )

    assert result.metadata["solver"] == "sr_adidas_sre"
    assert result.metadata["algorithm_family"] == "sr_adidas_full_tensor_robust_adi"
    assert result.success, result.message
    assert result.policies[0][1] > 0.99
    assert result.policies[1][1] > 0.99
    assert result.utilities_sr[0][0] == pytest.approx(result.metadata["robust_policy_values"][0])


def test_sr_adidas_batch_and_torch_batch_match():
    torch = pytest.importorskip("torch")
    from sre_solvers import make_sre_solver

    kwargs = dict(max_iters=20, tau_init=1.0, tau_min=1e-3, random_seed=11)
    q = np.stack([_rps_q_tensor(), _rps_q_tensor()], axis=0)

    numpy_results = make_sre_solver("sr_adidas_sre", **kwargs).solve_batch(
        q,
        epsilon=0.0,
        num_repeats=2,
        include_pure_starts=False,
    )
    torch_results = make_sre_solver("sr_adidas_sre", **kwargs).solve_batch_torch(
        torch.as_tensor(q, dtype=torch.float32),
        epsilon=0.0,
        num_repeats=2,
        include_pure_starts=False,
    )

    for lhs, rhs in zip(numpy_results, torch_results):
        for left_policy, right_policy in zip(lhs.policies, rhs.policies):
            np.testing.assert_allclose(left_policy, right_policy, atol=1e-6)
        assert lhs.metadata["robust_exploitability"] == pytest.approx(
            rhs.metadata["robust_exploitability"]
        )


def test_sr_adidas_torch_batch_uses_vectorized_path(monkeypatch):
    torch = pytest.importorskip("torch")
    from sre_solvers import make_sre_solver

    q = np.stack([_rps_q_tensor(), _rps_q_tensor() + 0.25], axis=0)
    solver = make_sre_solver(
        "sr_adidas_sre",
        max_iters=2,
        tau_init=1.0,
        tau_min=1e-3,
        random_seed=12,
        device="cpu",
    )

    def fail_solve_batch(*args, **kwargs):
        raise AssertionError("solve_batch_torch should not delegate to solve_batch")

    monkeypatch.setattr(solver, "solve_batch", fail_solve_batch)
    results = solver.solve_batch_torch(
        torch.as_tensor(q, dtype=torch.float32),
        epsilon=torch.zeros(2),
        num_repeats=1,
        include_pure_starts=False,
        round_digits=None,
    )

    assert len(results) == 2
    assert all(result.metadata["batched_torch"] for result in results)
    assert all(result.metadata["torch_device"] == "cpu" for result in results)


def test_warm_start_does_not_increase_iterations():
    from sre_solvers import make_sre_solver

    q = _dominant_q_tensor()
    cold = make_sre_solver(
        "sr_adidas_sre", max_iters=80, lr=0.35, tau_init=0.5, random_seed=4
    ).solve(q, epsilon=0.0, num_repeats=2, include_pure_starts=False)
    warm = make_sre_solver(
        "sr_adidas_sre", max_iters=80, lr=0.35, tau_init=0.5, random_seed=4
    ).solve(
        q,
        epsilon=0.0,
        num_repeats=2,
        include_pure_starts=False,
        initial_policies=cold.policies,
    )

    assert warm.metadata["iterations"] <= cold.metadata["iterations"]
    assert warm.metadata["robust_exploitability"] <= (
        cold.metadata["robust_exploitability"] + 1e-9
    )
