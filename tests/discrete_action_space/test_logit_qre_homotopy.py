import numpy as np
import pytest

from sre_solvers import LogitQreHomotopySreSolver, make_sre_solver
from sre_solvers.nplayer_common import robust_exploitability


def _rps_q_tensor():
    u1 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
    return np.stack([u1, -u1], axis=-1)


def _dominant_bimatrix_q_tensor():
    q = np.zeros((2, 2, 2), dtype=np.float64)
    q[1, :, 0] = 2.0
    q[:, 1, 1] = 3.0
    return q


def _dominant_three_player_q_tensor():
    q = np.zeros((2, 2, 2, 3), dtype=np.float64)
    q[1, :, :, 0] = 2.0
    q[:, 1, :, 1] = 3.0
    q[:, :, 1, 2] = 4.0
    return q


def test_logit_qre_solver_factory_aliases():
    solver = make_sre_solver("logit_qre_sre", random_seed=7)
    assert isinstance(solver, LogitQreHomotopySreSolver)
    assert make_sre_solver("qre_homotopy_sre").name == "logit_qre_sre"
    assert make_sre_solver("logit_qre").name == "logit_qre_sre"


def test_logit_qre_returns_uniform_on_rps_at_zero_epsilon():
    solver = make_sre_solver(
        "logit_qre_sre",
        precision_max=20.0,
        corrector_max_iters=20,
        random_seed=1,
    )
    result = solver.solve(
        _rps_q_tensor(),
        epsilon=0.0,
        num_repeats=0,
        include_pure_starts=False,
        round_digits=None,
    )

    assert result.success, result.message
    np.testing.assert_allclose(result.policies[0], np.full(3, 1 / 3), atol=1e-8)
    np.testing.assert_allclose(result.policies[1], np.full(3, 1 / 3), atol=1e-8)
    gap, _, _ = robust_exploitability(_rps_q_tensor(), result.policies, 0.0)
    assert result.metadata["algorithm_family"] == "logit_qre_homotopy"
    assert gap <= 1e-8


def test_logit_qre_finds_dominant_bimatrix_actions():
    solver = make_sre_solver(
        "logit_qre_sre",
        precision_max=100.0,
        corrector_max_iters=80,
        random_seed=3,
    )
    result = solver.solve(
        _dominant_bimatrix_q_tensor(),
        epsilon=0.0,
        num_repeats=0,
        include_pure_starts=False,
        round_digits=None,
    )

    assert result.success, result.message
    assert result.policies[0][1] > 0.999
    assert result.policies[1][1] > 0.999
    assert result.metadata["robust_exploitability"] <= 1e-4


def test_logit_qre_finds_dominant_three_player_actions():
    solver = make_sre_solver(
        "logit_qre_sre",
        precision_max=100.0,
        corrector_max_iters=80,
        random_seed=5,
    )
    result = solver.solve(
        _dominant_three_player_q_tensor(),
        epsilon=0.0,
        num_repeats=0,
        include_pure_starts=False,
        round_digits=None,
    )

    assert result.success, result.message
    assert result.policies[0][1] > 0.999
    assert result.policies[1][1] > 0.999
    assert result.policies[2][1] > 0.999


def test_logit_qre_reports_exact_mixed_policy_robust_gap():
    q = np.zeros((2, 2, 2), dtype=np.float64)
    q[:, :, 0] = np.array([[0.0, 2.0], [2.0, 0.0]])
    q[:, :, 1] = -q[:, :, 0]
    solver = make_sre_solver(
        "logit_qre_sre",
        precision_max=5.0,
        corrector_max_iters=25,
        random_seed=9,
    )
    result = solver.solve(
        q,
        epsilon=0.5,
        num_repeats=0,
        include_pure_starts=False,
        round_digits=None,
    )

    gap, player_gaps, _ = robust_exploitability(
        q, result.policies, 0.5, value_mode="mixed_policy"
    )
    assert result.metadata["robust_exploitability"] == pytest.approx(gap)
    assert result.metadata["player_robust_gaps"] == pytest.approx(player_gaps)


def test_logit_qre_warm_start_does_not_worsen_gap():
    q = _dominant_bimatrix_q_tensor()
    kwargs = dict(
        precision_max=60.0,
        corrector_max_iters=60,
        random_seed=11,
    )
    cold = make_sre_solver("logit_qre_sre", **kwargs).solve(
        q,
        epsilon=0.0,
        num_repeats=0,
        include_pure_starts=False,
    )
    warm = make_sre_solver("logit_qre_sre", **kwargs).solve(
        q,
        epsilon=0.0,
        num_repeats=0,
        include_pure_starts=False,
        initial_policies=cold.policies,
    )

    assert warm.metadata["robust_exploitability"] <= (
        cold.metadata["robust_exploitability"] + 1e-9
    )


def test_logit_qre_solve_batch_accepts_scalar_and_vector_epsilon():
    q = _dominant_bimatrix_q_tensor()
    solver = make_sre_solver(
        "logit_qre_sre",
        precision_max=20.0,
        corrector_max_iters=30,
        random_seed=13,
    )

    scalar_results = solver.solve_batch(
        [q, q],
        epsilon=0.0,
        num_repeats=0,
        include_pure_starts=False,
    )
    vector_results = solver.solve_batch(
        [q, q],
        epsilon=[0.0, 0.2],
        num_repeats=0,
        include_pure_starts=False,
    )

    assert len(scalar_results) == 2
    assert len(vector_results) == 2
    assert [result.metadata["epsilon"] for result in vector_results] == [0.0, 0.2]


def test_logit_qre_torch_batch_matches_numpy_batch():
    torch = pytest.importorskip("torch")

    q = np.stack([_rps_q_tensor(), _rps_q_tensor() + 0.25], axis=0)
    kwargs = dict(precision_max=20.0, corrector_max_iters=20, random_seed=17)
    numpy_results = make_sre_solver("logit_qre_sre", **kwargs).solve_batch(
        q,
        epsilon=0.0,
        num_repeats=0,
        include_pure_starts=False,
        round_digits=None,
    )
    torch_results = make_sre_solver("logit_qre_sre", **kwargs).solve_batch_torch(
        torch.as_tensor(q, dtype=torch.float32),
        epsilon=torch.zeros(2),
        num_repeats=0,
        include_pure_starts=False,
        round_digits=None,
    )

    for lhs, rhs in zip(numpy_results, torch_results):
        for left_policy, right_policy in zip(lhs.policies, rhs.policies):
            np.testing.assert_allclose(left_policy, right_policy, atol=1e-6)
        assert rhs.metadata["batched_torch"]
        assert lhs.metadata["robust_exploitability"] == pytest.approx(
            rhs.metadata["robust_exploitability"], abs=1e-6
        )


def test_logit_qre_accepts_deep_srq_solver_kwargs():
    solver = make_sre_solver("logit_qre_sre", precision_max=20.0, corrector_max_iters=20)
    result = solver.solve(
        _rps_q_tensor(),
        epsilon=0.0,
        num_repeats=0,
        include_pure_starts=False,
        exploitability_tol=1e-4,
        early_exit=True,
        candidate_selection="robust_exploitability",
    )

    assert result.success
