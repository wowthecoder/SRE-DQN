import numpy as np
import torch

from sre_solvers import NfgTransformerConfig, NfgTransformerSreNet, make_sre_solver
from sre_solvers.nfg_transformer.torch_utils import (
    normalize_payoffs,
    robust_exploitability_torch,
)
from sre_solvers.nfg_transformer.train import sample_q_tensor_torch
from sre_solvers.nfg_transformer.train import train_checkpoint
from sre_solvers.nplayer_common import robust_exploitability


def _tiny_model(num_players=3, num_actions=3):
    config = NfgTransformerConfig(
        num_players=num_players,
        num_actions=num_actions,
        embed_dim=16,
        num_blocks=2,
        num_heads=4,
    )
    torch.manual_seed(7)
    return NfgTransformerSreNet(config)


def test_nfg_transformer_outputs_simplex_policies():
    model = _tiny_model()
    q = torch.randn(5, 3, 3, 3, 3)
    policies = model(q, torch.zeros(5))

    assert len(policies) == 3
    assert all(policy.shape == (5, 3) for policy in policies)
    assert all(torch.all(policy >= 0.0) for policy in policies)
    assert all(
        torch.allclose(policy.sum(dim=-1), torch.ones(5), atol=1e-6)
        for policy in policies
    )


def test_nfg_transformer_supports_rectangular_games_with_one_checkpoint():
    model = _tiny_model()
    model.eval()

    with torch.no_grad():
        policies_2p = model(torch.randn(4, 2, 3, 2), epsilon=0.25)
        policies_3p = model(torch.randn(4, 2, 3, 4, 3), epsilon=torch.ones(4) * 0.5)

    assert [policy.shape for policy in policies_2p] == [(4, 2), (4, 3)]
    assert [policy.shape for policy in policies_3p] == [(4, 2), (4, 3), (4, 4)]
    assert all(torch.allclose(policy.sum(-1), torch.ones(4), atol=1e-6) for policy in policies_2p)
    assert all(torch.allclose(policy.sum(-1), torch.ones(4), atol=1e-6) for policy in policies_3p)


def test_nfg_transformer_conditions_on_epsilon():
    model = _tiny_model()
    q = torch.randn(3, 3, 3, 3, 3)

    with torch.no_grad():
        low_eps = model(q, torch.zeros(3))
        high_eps = model(q, torch.ones(3))

    diffs = [(a - b).abs().max().item() for a, b in zip(low_eps, high_eps)]
    assert max(diffs) > 1e-8


def test_nfg_transformer_is_equivariant_to_action_permutation():
    model = _tiny_model()
    model.eval()
    q = torch.randn(2, 3, 3, 3, 3)
    perm = torch.tensor([2, 0, 1])

    with torch.no_grad():
        base = model(q, torch.zeros(2))
        q_perm = q.index_select(1, perm)
        permuted = model(q_perm, torch.zeros(2))

    assert torch.allclose(permuted[0], base[0].index_select(1, perm), atol=1e-5)
    assert torch.allclose(permuted[1], base[1], atol=1e-5)
    assert torch.allclose(permuted[2], base[2], atol=1e-5)


def test_torch_robust_exploitability_matches_numpy():
    q_np = np.random.default_rng(5).normal(size=(2, 2, 2, 3)).astype(np.float32)
    policies_np = np.asarray(
        [
            [0.25, 0.75],
            [0.4, 0.6],
            [0.7, 0.3],
        ],
        dtype=np.float32,
    )

    q = torch.as_tensor(q_np).unsqueeze(0)
    policies = [torch.as_tensor(policy).unsqueeze(0) for policy in policies_np]
    gap_t, _, _ = robust_exploitability_torch(q, policies, epsilon=0.2)
    gap_np, _, _ = robust_exploitability(q_np, policies_np, epsilon=0.2)

    assert np.allclose(gap_t.detach().numpy()[0], gap_np, atol=1e-6)


def test_torch_robust_exploitability_matches_numpy_rectangular():
    q_np = np.random.default_rng(6).normal(size=(2, 3, 2)).astype(np.float32)
    policies_np = [
        np.asarray([0.25, 0.75], dtype=np.float32),
        np.asarray([0.2, 0.3, 0.5], dtype=np.float32),
    ]

    q = torch.as_tensor(q_np).unsqueeze(0)
    policies = [torch.as_tensor(policy).unsqueeze(0) for policy in policies_np]
    gap_t, _, _ = robust_exploitability_torch(q, policies, epsilon=0.35)
    gap_np, _, _ = robust_exploitability(q_np, policies_np, epsilon=0.35)

    assert np.allclose(gap_t.detach().numpy()[0], gap_np, atol=1e-6)


def test_synthetic_training_sampler_uses_global_payoff_normalization():
    torch.manual_seed(13)
    sampled = sample_q_tensor_torch(4, (2, 3, 2), device="cpu")

    torch.manual_seed(13)
    raw = torch.randn(4, 2, 3, 2, 3)
    expected = normalize_payoffs(raw)

    assert torch.allclose(sampled, expected)
    action_dims = tuple(range(1, sampled.ndim - 1))
    assert torch.allclose(
        sampled.mean(dim=action_dims),
        torch.zeros(4, 3),
        atol=1e-6,
    )


def test_train_checkpoint_saves_best_final_and_resumes(tmp_path):
    best_checkpoint = tmp_path / "nfg_sre_best.pt"
    final_checkpoint = tmp_path / "nfg_sre_final.pt"

    train_checkpoint(
        output=best_checkpoint,
        final_output=final_checkpoint,
        num_iterations=2,
        log_every=1,
        batch_size=2,
        lr=1e-4,
        embed_dim=16,
        num_blocks=2,
        num_heads=4,
        game_shapes=((2, 2),),
        seed=17,
        use_gpu=False,
    )

    assert best_checkpoint.exists()
    assert final_checkpoint.exists()
    best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    final_payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    assert best_payload["iteration"] <= 2
    assert final_payload["iteration"] == 2
    assert "optimizer_state_dict" in final_payload
    assert "rng_state" in final_payload
    assert "model_state_dict" in final_payload

    resumed_final = tmp_path / "nfg_sre_resumed_final.pt"
    train_checkpoint(
        output=best_checkpoint,
        final_output=resumed_final,
        resume_from=final_checkpoint,
        num_iterations=3,
        log_every=1,
        batch_size=2,
        lr=1e-4,
        embed_dim=16,
        num_blocks=2,
        num_heads=4,
        game_shapes=((2, 2),),
        seed=17,
        use_gpu=False,
    )

    resumed_payload = torch.load(resumed_final, map_location="cpu", weights_only=False)
    assert resumed_payload["iteration"] == 3
    assert resumed_payload["num_iterations"] == 3
    assert "optimizer_state_dict" in resumed_payload
    assert "rng_state" in resumed_payload


def test_nfg_transformer_solver_factory_with_tiny_checkpoint(tmp_path):
    model = _tiny_model(num_players=3, num_actions=2)
    checkpoint = tmp_path / "nfg_sre.pt"
    torch.save(
        {
            "config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    solver = make_sre_solver(
        "nfg_transformer_sre",
        checkpoint_path=checkpoint,
        device="cpu",
        fallback_enabled=False,
    )
    try:
        q = np.zeros((2, 2, 2, 3), dtype=np.float32)
        result = solver.solve(q, epsilon=0.0, round_digits=None)
    finally:
        solver.close()

    assert result.metadata["solver"] == "nfg_transformer_sre"
    assert result.metadata["used_fallback"] is False
    assert len(result.policies) == 3
    assert all(np.allclose(policy.sum(), 1.0) for policy in result.policies)


def test_nfg_transformer_solver_torch_batch_matches_numpy_path(tmp_path):
    model = _tiny_model(num_players=3, num_actions=2)
    checkpoint = tmp_path / "nfg_sre.pt"
    torch.save(
        {
            "config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    solver = make_sre_solver(
        "nfg_transformer_sre",
        checkpoint_path=checkpoint,
        device="cpu",
        fallback_enabled=False,
    )
    try:
        q = torch.randn(4, 2, 2, 2, 3)
        torch_results = solver.solve_batch_torch(
            q, epsilon=torch.full((4,), 0.2), round_digits=None
        )
        numpy_results = solver.solve_batch(
            q.detach().numpy(), epsilon=np.full(4, 0.2), round_digits=None
        )
    finally:
        solver.close()

    assert len(torch_results) == len(numpy_results) == 4
    for torch_result, numpy_result in zip(torch_results, numpy_results):
        assert torch_result.metadata["used_fallback"] is False
        assert np.isclose(
            torch_result.metadata["neural_robust_exploitability"],
            numpy_result.metadata["neural_robust_exploitability"],
            atol=1e-6,
        )
        for torch_policy, numpy_policy in zip(torch_result.policies, numpy_result.policies):
            assert np.allclose(torch_policy, numpy_policy, atol=1e-6)
