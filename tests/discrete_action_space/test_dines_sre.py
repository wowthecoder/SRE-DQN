"""Unit and integration tests for DI-SRE-S (dines_sre module).

Run from repo root:
    pytest tests/discrete_action_space/test_dines_sre.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "discrete_action_space"))

from dines_sre.config import DinesSreConfig
from dines_sre.data import sample_bimatrix_games, sample_epsilon
from dines_sre.loss import sr_exploitability_loss, _sr_current_value, _fw_mixed_br
from dines_sre.model import DinesSreNet
from dines_sre.query import sr_utility_query, tv_worst_case_with_q_star


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_net():
    return DinesSreNet(
        num_actions=4,
        embed_dim=16,
        num_rounds=3,
        num_heads=2,
        ffn_hidden=32,
        eps_embed_dim=8,
    )


@pytest.fixture
def random_bimatrix():
    torch.manual_seed(42)
    return sample_bimatrix_games(8, 4, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# query.py
# ---------------------------------------------------------------------------

class TestTvWorstCaseWithQStar:
    def test_value_matches_batch_helper(self):
        from mean_field_dsrq.mf_robust_value import tv_worst_case_batch
        torch.manual_seed(0)
        B, A = 16, 5
        p = F.softmax(torch.randn(B, A), dim=-1)
        v = torch.randn(B, A)
        for eps in [0.0, 0.1, 0.5, 1.0]:
            val_ref = tv_worst_case_batch(p, v, eps)
            val_new, q_star = tv_worst_case_with_q_star(p, v, eps)
            assert torch.allclose(val_ref, val_new, atol=1e-5), (
                f"eps={eps}: value mismatch"
            )

    def test_q_star_is_distribution(self):
        torch.manual_seed(1)
        B, A = 8, 5
        p = F.softmax(torch.randn(B, A), dim=-1)
        v = torch.randn(B, A)
        _, q_star = tv_worst_case_with_q_star(p, v, epsilon=0.3)
        assert q_star.shape == (B, A)
        assert (q_star >= -1e-6).all(), "q_star has negative entries"
        assert torch.allclose(q_star.sum(dim=-1), torch.ones(B), atol=1e-5), \
            "q_star rows don't sum to 1"

    def test_q_star_achieves_value(self):
        torch.manual_seed(2)
        B, A = 4, 5
        p = F.softmax(torch.randn(B, A), dim=-1)
        v = torch.randn(B, A)
        val, q_star = tv_worst_case_with_q_star(p, v, epsilon=0.3)
        reconstructed = (q_star * v).sum(dim=-1)
        assert torch.allclose(val, reconstructed, atol=1e-5)

    def test_tv_zero_returns_nominal(self):
        torch.manual_seed(3)
        B, A = 4, 5
        p = F.softmax(torch.randn(B, A), dim=-1)
        v = torch.randn(B, A)
        val, q_star = tv_worst_case_with_q_star(p, v, epsilon=0.0)
        expected = (p * v).sum(dim=-1)
        assert torch.allclose(val, expected, atol=1e-5)


class TestSrUtilityQuery:
    def test_output_shape(self):
        B, T = 8, 5
        U_p = torch.randn(B, T, T)
        x_minus = F.softmax(torch.randn(B, T), dim=-1)
        u_sr = sr_utility_query(U_p, x_minus, epsilon=0.3)
        assert u_sr.shape == (B, T)

    def test_eps_zero_equals_expected_payoff(self):
        torch.manual_seed(10)
        B, T = 4, 5
        U_p = torch.randn(B, T, T)
        x_minus = F.softmax(torch.randn(B, T), dim=-1)
        u_sr = sr_utility_query(U_p, x_minus, epsilon=0.0)
        # With ε=0, no adversary moves, so u_sr[b,j] = sum_k x_minus[b,k]*U_p[b,j,k]
        expected = (U_p * x_minus.unsqueeze(1)).sum(dim=-1)
        assert torch.allclose(u_sr, expected, atol=1e-5)

    def test_sr_leq_nominal(self):
        torch.manual_seed(11)
        B, T = 8, 5
        U_p = torch.randn(B, T, T)
        x_minus = F.softmax(torch.randn(B, T), dim=-1)
        u_nominal = (U_p * x_minus.unsqueeze(1)).sum(dim=-1)
        u_sr = sr_utility_query(U_p, x_minus, epsilon=0.4)
        assert (u_sr <= u_nominal + 1e-5).all(), \
            "SR utility should be <= nominal utility"


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------

class TestDinesSreNet:
    def test_output_shape(self, small_net, random_bimatrix):
        eps = torch.full((8,), 0.3)
        x_hat = small_net(random_bimatrix, eps)
        assert x_hat.shape == (8, 2, 4), f"Expected (8,2,4), got {x_hat.shape}"

    def test_output_is_probability_simplex(self, small_net, random_bimatrix):
        eps = torch.full((8,), 0.3)
        x_hat = small_net(random_bimatrix, eps)
        assert (x_hat >= -1e-6).all(), "Negative probabilities"
        sums = x_hat.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Rows don't sum to 1"

    def test_scalar_epsilon_accepted(self, small_net, random_bimatrix):
        x_hat = small_net(random_bimatrix, 0.5)
        assert x_hat.shape == (8, 2, 4)

    def test_permutation_equivariance_players(self, small_net):
        """Swapping players should swap output policies."""
        torch.manual_seed(99)
        B, T = 4, 4
        U = sample_bimatrix_games(B, T, torch.device("cpu"))
        eps = torch.full((B,), 0.3)

        # Swap player 0 and player 1:
        # U[b, j, k, 0] = player-0 payoff → becomes player-1 payoff in swapped game.
        # Swapped game: U_swap[b, j, k, 0] = U[b, k, j, 1], U_swap[b, j, k, 1] = U[b, k, j, 0]
        U_swap = torch.stack([
            U[..., 1].transpose(1, 2),
            U[..., 0].transpose(1, 2),
        ], dim=-1)

        results = []
        for _ in range(5):
            torch.manual_seed(77)
            x1 = small_net(U, eps)
            torch.manual_seed(77)
            x2 = small_net(U_swap, eps)
            # x2[:, 0, :] should ≈ x1[:, 1, :] and x2[:, 1, :] ≈ x1[:, 0, :]
            results.append(
                torch.allclose(x2[:, 0, :], x1[:, 1, :], atol=0.05) and
                torch.allclose(x2[:, 1, :], x1[:, 0, :], atol=0.05)
            )
        # Equivariance holds for most random seeds (randomized init, so allow some tolerance)
        assert sum(results) >= 3, "Player permutation equivariance fails too often"

    def test_eps_zero_and_one_run_without_error(self, small_net, random_bimatrix):
        for eps_val in [0.0, 1.0]:
            eps = torch.full((8,), eps_val)
            x_hat = small_net(random_bimatrix, eps)
            assert x_hat.shape == (8, 2, 4)
            assert not torch.isnan(x_hat).any(), f"NaN at eps={eps_val}"


# ---------------------------------------------------------------------------
# loss.py
# ---------------------------------------------------------------------------

class TestSrExploitabilityLoss:
    def test_loss_non_negative(self, small_net, random_bimatrix):
        eps = torch.full((8,), 0.3)
        x_hat = small_net(random_bimatrix, eps)
        loss = sr_exploitability_loss(random_bimatrix, x_hat, epsilon=0.3)
        assert loss.item() >= -1e-6, "Loss should be non-negative"

    def test_loss_is_scalar(self, small_net, random_bimatrix):
        eps = torch.full((8,), 0.3)
        x_hat = small_net(random_bimatrix, eps)
        loss = sr_exploitability_loss(random_bimatrix, x_hat, epsilon=0.3)
        assert loss.dim() == 0

    def test_loss_has_gradient(self, small_net, random_bimatrix):
        eps = torch.full((8,), 0.3)
        x_hat = small_net(random_bimatrix, eps)
        loss = sr_exploitability_loss(random_bimatrix, x_hat, epsilon=0.3)
        loss.backward()
        # Check that at least one parameter has a non-None gradient.
        grads = [p.grad for p in small_net.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"

    def test_current_value_leq_br_value(self):
        torch.manual_seed(5)
        B, T = 16, 5
        U_p = torch.randn(B, T, T)
        x_i = F.softmax(torch.randn(B, T), dim=-1)
        x_opp = F.softmax(torch.randn(B, T), dim=-1)
        for eps in [0.0, 0.2, 0.5, 1.0]:
            cur = _sr_current_value(U_p, x_i, x_opp, eps)
            br = _fw_mixed_br(U_p, x_opp, eps, fw_iters=20)
            assert (br + 1e-4 >= cur).all(), \
                f"BR value should >= current value at eps={eps}"

    def test_br_value_geq_pure_action_max(self):
        torch.manual_seed(6)
        B, T = 16, 5
        U_p = torch.randn(B, T, T)
        x_opp = F.softmax(torch.randn(B, T), dim=-1)
        from dines_sre.query import sr_utility_query
        for eps in [0.1, 0.4]:
            br = _fw_mixed_br(U_p, x_opp, eps, fw_iters=20)
            u_sr_pure = sr_utility_query(U_p, x_opp, eps)   # [B, T]
            pure_max = u_sr_pure.max(dim=-1).values
            # Mixed BR must be >= best pure action
            assert (br + 1e-3 >= pure_max).all(), \
                f"Mixed BR should >= pure-action max at eps={eps}"


# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------

class TestDataSampler:
    def test_bimatrix_shape(self):
        U = sample_bimatrix_games(32, 5, torch.device("cpu"))
        assert U.shape == (32, 5, 5, 2)

    def test_bimatrix_range(self):
        U = sample_bimatrix_games(256, 5, torch.device("cpu"))
        assert U.min().item() >= -1.0 - 1e-6
        assert U.max().item() <= 1.0 + 1e-6

    def test_epsilon_range(self):
        eps = sample_epsilon(256, torch.device("cpu"), torch.float32)
        assert eps.min().item() >= 0.0
        assert eps.max().item() <= 1.0
