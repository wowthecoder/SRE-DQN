"""Unit tests for mf_robust_value.py.

Verifies the vectorized PyTorch TV-worst-case op against the NumPy reference
implementation in sre_solvers/nplayer_common._tv_worst_case_value.
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
from mean_field_dsrq.mf_robust_value import tv_worst_case_batch, robust_q_grid, boltzmann_policy


class TestTvWorstCaseBatch:
    """Compare vectorized PyTorch TV with NumPy reference on random inputs."""

    @pytest.mark.parametrize("epsilon", [0.0, 0.05, 0.10, 0.25, 0.50, 1.0])
    @pytest.mark.parametrize("A", [2, 5, 13, 21])
    def test_matches_numpy_reference(self, epsilon, A):
        rng = np.random.default_rng(seed=42)
        B = 50
        p_np = rng.dirichlet(np.ones(A), size=B).astype(np.float32)
        v_np = rng.standard_normal((B, A)).astype(np.float32)

        expected = np.array([
            numpy_tv(p_np[i], v_np[i], epsilon) for i in range(B)
        ], dtype=np.float64)

        p_t = torch.as_tensor(p_np)
        v_t = torch.as_tensor(v_np)
        got = tv_worst_case_batch(p_t, v_t, epsilon).numpy().astype(np.float64)

        np.testing.assert_allclose(got, expected, atol=1e-4,
            err_msg=f"Mismatch at eps={epsilon}, A={A}")

    def test_epsilon_zero_equals_nominal(self):
        """ε=0 → worst case = nominal expectation."""
        B, A = 20, 10
        p = torch.softmax(torch.randn(B, A), dim=-1)
        v = torch.randn(B, A)
        result = tv_worst_case_batch(p, v, epsilon=0.0)
        expected = (p * v).sum(dim=-1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)

    def test_epsilon_one_returns_min_value(self):
        """ε=1 → worst case = min(v) (all mass moved to the minimum)."""
        B, A = 20, 10
        p = torch.softmax(torch.randn(B, A), dim=-1)
        v = torch.randn(B, A)
        result = tv_worst_case_batch(p, v, epsilon=1.0)
        expected = v.min(dim=-1).values
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=0)

    def test_output_is_leq_nominal(self):
        """Worst-case value ≤ nominal expected value for all ε > 0."""
        B, A = 100, 7
        p = torch.softmax(torch.randn(B, A), dim=-1)
        v = torch.randn(B, A)
        nominal = (p * v).sum(dim=-1)
        for eps in [0.0, 0.1, 0.5, 1.0]:
            wc = tv_worst_case_batch(p, v, eps)
            assert (wc <= nominal + 1e-5).all(), f"WC > nominal at eps={eps}"

    def test_zero_rows_use_uniform(self):
        """Rows summing to zero should fall back to uniform."""
        B, A = 4, 5
        p = torch.zeros(B, A)
        v = torch.arange(A, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        # Uniform p → TV_worst with ε=0 should = mean(v)
        result = tv_worst_case_batch(p, v, epsilon=0.0)
        expected = v.float().mean(dim=-1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)

    def test_single_action(self):
        """A=1: always returns v[0] regardless of epsilon."""
        B = 10
        p = torch.ones(B, 1)
        v = torch.randn(B, 1)
        result = tv_worst_case_batch(p, v, epsilon=0.5)
        torch.testing.assert_close(result, v.squeeze(-1), atol=1e-6, rtol=0)


class TestRobustQGrid:
    def test_shape(self):
        B, A_own, A_nbr = 8, 5, 7
        q_grid = torch.randn(B, A_own, A_nbr)
        mean_a = torch.softmax(torch.randn(B, A_nbr), dim=-1)
        out = robust_q_grid(q_grid, mean_a, epsilon=0.1)
        assert out.shape == (B, A_own)

    def test_epsilon_zero_equals_expected(self):
        """ε=0: robust Q = Σ_k ā[k] · Q(a_own, k) (nominal mean)."""
        B, A_own, A_nbr = 6, 4, 5
        q_grid = torch.randn(B, A_own, A_nbr)
        mean_a = torch.softmax(torch.randn(B, A_nbr), dim=-1)
        got = robust_q_grid(q_grid, mean_a, epsilon=0.0)
        expected = torch.einsum("bik,bk->bi", q_grid, mean_a)
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)

    @pytest.mark.parametrize("A", [2, 21])
    def test_matches_numpy_reference_per_row(self, A):
        """Cross-check each (own_action, sample) against NumPy TV."""
        B = 20
        rng = np.random.default_rng(99)
        q_np = rng.standard_normal((B, A, A)).astype(np.float32)
        p_np = rng.dirichlet(np.ones(A), size=B).astype(np.float32)
        eps = 0.15

        expected = np.zeros((B, A), dtype=np.float64)
        for b in range(B):
            for a in range(A):
                expected[b, a] = numpy_tv(p_np[b], q_np[b, a], eps)

        q_t = torch.as_tensor(q_np)
        p_t = torch.as_tensor(p_np)
        got = robust_q_grid(q_t, p_t, eps).numpy().astype(np.float64)
        np.testing.assert_allclose(got, expected, atol=1e-4)


class TestBoltzmannPolicy:
    def test_sums_to_one(self):
        B, A = 16, 10
        q = torch.randn(B, A)
        pi = boltzmann_policy(q, beta=2.0)
        assert pi.shape == (B, A)
        torch.testing.assert_close(pi.sum(dim=-1), torch.ones(B), atol=1e-5, rtol=0)

    def test_high_beta_concentrates(self):
        """Very high β → policy concentrates near the best action."""
        A = 5
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        pi = boltzmann_policy(q, beta=100.0)
        assert pi[0, 0] > 0.99

    def test_low_beta_is_near_uniform(self):
        """β→0 → policy approaches uniform."""
        B, A = 4, 5
        q = torch.randn(B, A)
        pi = boltzmann_policy(q, beta=1e-4)
        expected = torch.full((B, A), 1.0 / A)
        torch.testing.assert_close(pi, expected, atol=1e-3, rtol=0)
