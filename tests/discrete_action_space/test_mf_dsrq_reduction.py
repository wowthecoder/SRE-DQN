"""Reduction-sanity tests for MF-DSRQ.

Two checks:
1. ε=0: robust Q-grid = nominal mean → MF-DSRQ Bellman target reduces to
   the vanilla mean-field Q-learning target (Yang 2018).

2. N=2 consistency: with a single neighbor, ā IS the opponent's policy.
   The MF-DSRQ robust target with ā = opponent_policy must equal the
   Deep SRQ TV worst-case target computed by robust_action_values.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_DISCRETE = _ROOT / "discrete_action_space"
for p in [str(_ROOT), str(_DISCRETE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sre_solvers.nplayer_common import _tv_worst_case_value as numpy_tv
from mean_field_dsrq.mf_robust_value import robust_q_grid, boltzmann_policy, tv_worst_case_batch


def _mfq_bellman_target(q_grid_next: torch.Tensor, next_mean_a: torch.Tensor, beta: float) -> torch.Tensor:
    """Vanilla MF-Q Bellman target (Yang 2018, no robustness).

    V(s') = Σ_a π(a|s', ā') · Q_mean(s', a)
    where Q_mean(s', a) = Σ_k ā'[k] · Q(s', a, k) and π = softmax(β·Q_mean).
    """
    q_mean = torch.einsum("bik,bk->bi", q_grid_next, next_mean_a)  # [B, A_own]
    pi = boltzmann_policy(q_mean, beta)
    return (pi * q_mean).sum(dim=-1)


def _mfdsrq_bellman_target(
    q_grid_next: torch.Tensor,
    next_mean_a: torch.Tensor,
    epsilon: float,
    beta: float,
) -> torch.Tensor:
    """MF-DSRQ Bellman target."""
    z = robust_q_grid(q_grid_next, next_mean_a, epsilon)  # [B, A_own]
    pi = boltzmann_policy(z, beta)
    return (pi * z).sum(dim=-1)


class TestEpsilonZeroReducesToMFQ:
    """With ε=0, MF-DSRQ target must equal MF-Q target exactly."""

    @pytest.mark.parametrize("A", [2, 5, 21])
    @pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
    def test_epsilon_zero_matches_mfq(self, A, beta):
        B = 50
        torch.manual_seed(7)
        q_grid = torch.randn(B, A, A)
        mean_a = torch.softmax(torch.randn(B, A), dim=-1)

        mfq_val = _mfq_bellman_target(q_grid, mean_a, beta)
        mfdsrq_val = _mfdsrq_bellman_target(q_grid, mean_a, epsilon=0.0, beta=beta)

        torch.testing.assert_close(mfdsrq_val, mfq_val, atol=1e-5, rtol=1e-5,
            msg=f"ε=0 mismatch at A={A}, β={beta}")

    def test_robust_target_leq_nominal_for_positive_epsilon(self):
        """For ε>0 the robust target ≤ nominal MF-Q target."""
        B, A, beta = 30, 10, 1.0
        torch.manual_seed(13)
        q_grid = torch.randn(B, A, A)
        mean_a = torch.softmax(torch.randn(B, A), dim=-1)

        nominal = _mfq_bellman_target(q_grid, mean_a, beta)
        for eps in [0.05, 0.1, 0.3]:
            robust = _mfdsrq_bellman_target(q_grid, mean_a, eps, beta)
            assert (robust <= nominal + 1e-4).all(), f"robust > nominal at ε={eps}"


class TestN2ConsistencyWithDeepSRQ:
    """N=2 consistency: MF-DSRQ with ā = opponent policy ≡ Deep SRQ TV robust value."""

    @pytest.mark.parametrize("epsilon", [0.0, 0.05, 0.20, 0.50])
    @pytest.mark.parametrize("A", [3, 5, 10])
    def test_n2_matches_robust_action_values(self, epsilon, A):
        """
        With a single neighbor, ā is the opponent's policy π_2.
        The MF-DSRQ robust Q for own action a_i is:
            min_{q ∈ TV_ε(π_2)} Σ_k q[k] · Q(a_i, k)
        which is exactly what _tv_worst_case_value computes.
        """
        rng = np.random.default_rng(42)
        B = 30

        q_np = rng.standard_normal((B, A, A)).astype(np.float32)
        opponent_policy_np = rng.dirichlet(np.ones(A), size=B).astype(np.float32)

        # NumPy reference: robust Q per own action per sample.
        expected = np.zeros((B, A), dtype=np.float64)
        for b in range(B):
            for a in range(A):
                expected[b, a] = numpy_tv(opponent_policy_np[b], q_np[b, a], epsilon)

        q_t = torch.as_tensor(q_np)
        ma_t = torch.as_tensor(opponent_policy_np)
        got = robust_q_grid(q_t, ma_t, epsilon).numpy().astype(np.float64)

        np.testing.assert_allclose(got, expected, atol=1e-4,
            err_msg=f"N=2 mismatch at ε={epsilon}, A={A}")

    @pytest.mark.parametrize("A", [4, 7])
    def test_bellman_target_consistency(self, A):
        """Full Bellman target comparison between PyTorch and NumPy for N=2 case."""
        B, epsilon, beta = 20, 0.15, 2.0
        rng = np.random.default_rng(99)

        q_np = rng.standard_normal((B, A, A)).astype(np.float32)
        opp_policy_np = rng.dirichlet(np.ones(A), size=B).astype(np.float32)

        # Build expected target via NumPy.
        expected_targets = np.zeros(B, dtype=np.float64)
        for b in range(B):
            z_b = np.array([numpy_tv(opp_policy_np[b], q_np[b, a], epsilon) for a in range(A)])
            pi_b = np.exp(beta * z_b)
            pi_b /= pi_b.sum()
            expected_targets[b] = float(pi_b @ z_b)

        q_t = torch.as_tensor(q_np)
        ma_t = torch.as_tensor(opp_policy_np)
        got = _mfdsrq_bellman_target(q_t, ma_t, epsilon, beta).numpy().astype(np.float64)

        np.testing.assert_allclose(got, expected_targets, atol=1e-4)
