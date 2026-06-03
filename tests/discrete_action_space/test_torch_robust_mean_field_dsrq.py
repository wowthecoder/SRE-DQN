"""Tests for the Torch robust-action mean-field SRQ variant."""

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

from mean_field_dsrq.solver_free_mean_field_dsrq import _tv_worst_case_value  # noqa: E402
from mean_field_dsrq.torch_robust_mean_field_dsrq import (  # noqa: E402
    TorchRobustMFDsrqAgent,
    torch_tv_worst_case_values,
)
from mean_field_dsrq.train_mf_dsrq import make_mfdsrq_agent  # noqa: E402


def _make_torch_agent(**overrides):
    defaults = dict(
        type_id=0,
        obs_channels=2,
        obs_height=3,
        obs_width=3,
        n_own_actions=4,
        n_nbr_actions=4,
        batch_size=4,
        buffer_capacity=32,
        learning_starts=4,
        train_every=1,
        epsilon_explore=0.0,
        device=torch.device("cpu"),
    )
    defaults.update(overrides)
    return TorchRobustMFDsrqAgent(**defaults)


def test_torch_tv_worst_case_values_match_numpy_scalar_operator():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    payoff = rng.normal(size=(3, 4, 5)).astype(np.float32)
    mean = rng.random(size=(3, 5)).astype(np.float32)
    mean = mean / mean.sum(axis=-1, keepdims=True)
    epsilon = 0.35

    actual = torch_tv_worst_case_values(
        torch.as_tensor(mean),
        torch.as_tensor(payoff),
        epsilon,
    ).numpy()

    expected = np.empty((3, 4), dtype=np.float32)
    for batch_idx in range(3):
        for action_idx in range(4):
            expected[batch_idx, action_idx] = _tv_worst_case_value(
                mean[batch_idx],
                payoff[batch_idx, action_idx],
                epsilon,
            )
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_torch_tv_epsilon_zero_matches_mean_weighted_payoffs():
    payoff = torch.tensor(
        [[[1.0, 3.0], [2.0, -1.0]], [[0.5, 1.5], [4.0, 0.0]]],
        dtype=torch.float32,
    )
    mean = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    actual = torch_tv_worst_case_values(mean, payoff, epsilon=0.0)
    expected = torch.bmm(payoff, mean.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(actual, expected)


def test_torch_robust_agent_samples_from_batched_robust_policy():
    agent = _make_torch_agent(robust_policy_temperature=0.01)
    obs = np.random.randn(6, 2, 3, 3).astype(np.float32)
    mean = np.zeros((6, 4), dtype=np.float32)
    mean[:, 0] = 1.0

    class FixedNet(torch.nn.Module):
        def payoff_matrix(self, obs_t, feature_t=None):
            del feature_t
            batch = obs_t.shape[0]
            matrix = torch.zeros(batch, 4, 4, device=obs_t.device)
            matrix[:, 2, :] = 2.0
            matrix[:, 1, 0] = 10.0
            matrix[:, 1, 1:] = -10.0
            return matrix

        def forward(self, obs_t, mean_t, feature=None):
            return torch.bmm(self.payoff_matrix(obs_t, feature), mean_t.unsqueeze(-1)).squeeze(-1)

    agent.q_net = FixedNet()
    agent.epsilon_robust = 1.0

    actions = agent.act_batch(obs, mean)

    assert actions.tolist() == [2] * 6
    assert agent.robust_torch_operator_calls == 6


def test_torch_robust_agent_sanitizes_invalid_policy_values():
    agent = _make_torch_agent(robust_policy_temperature=0.1)
    obs = np.random.randn(5, 2, 3, 3).astype(np.float32)
    mean = np.full((5, 4), 0.25, dtype=np.float32)

    class InvalidNet(torch.nn.Module):
        def payoff_matrix(self, obs_t, feature_t=None):
            del feature_t
            batch = obs_t.shape[0]
            matrix = torch.zeros(batch, 4, 4, device=obs_t.device)
            matrix[:, 0, :] = float("nan")
            matrix[:, 1, :] = float("inf")
            matrix[:, 2, :] = -float("inf")
            matrix[:, 3, :] = 1.0
            return matrix

        def forward(self, obs_t, mean_t, feature=None):
            return torch.bmm(self.payoff_matrix(obs_t, feature), mean_t.unsqueeze(-1)).squeeze(-1)

    agent.q_net = InvalidNet()
    actions = agent.act_batch(obs, mean)

    assert actions.shape == (5,)
    assert np.all((0 <= actions) & (actions < 4))


def test_torch_robust_train_step_updates_from_replay():
    torch.manual_seed(1)
    np.random.seed(1)
    agent = _make_torch_agent()
    before = [p.detach().clone() for p in agent.q_net.parameters()]
    mean = np.full(4, 0.25, dtype=np.float32)

    for i in range(6):
        obs = np.random.randn(2, 3, 3).astype(np.float32)
        next_obs = np.random.randn(2, 3, 3).astype(np.float32)
        agent.push(obs, i % 4, float(i), next_obs, mean, mean, done=False)

    loss = agent.train_step()

    assert isinstance(loss, float)
    after = list(agent.q_net.parameters())
    assert any(not torch.allclose(old, new) for old, new in zip(before, after))
    assert agent.robust_torch_operator_calls > 0


def test_torch_robust_target_uses_online_greedy_action_not_soft_expected_policy():
    agent = _make_torch_agent(
        gamma=1.0,
        lr=0.0,
        robust_policy_temperature=100.0,
    )
    mean = np.full(4, 0.25, dtype=np.float32)

    for i in range(4):
        obs = np.zeros((2, 3, 3), dtype=np.float32)
        next_obs = np.full((2, 3, 3), i + 1, dtype=np.float32)
        agent.push(obs, 0, 0.0, next_obs, mean, mean, done=False)

    class ZeroCurrentQNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def payoff_matrix(self, obs_t, feature_t=None):
            del feature_t
            return torch.zeros(obs_t.shape[0], 4, 4, device=obs_t.device) + self.weight

        def forward(self, obs_t, mean_t, feature=None):
            del feature
            return torch.zeros(obs_t.shape[0], 4, device=obs_t.device) + self.weight

    class TargetNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def payoff_matrix(self, obs_t, feature_t=None):
            del feature_t
            return torch.zeros(obs_t.shape[0], 4, 4, device=obs_t.device) + self.weight

    calls = []

    def fake_robust_action_values(payoff_matrices, mean_actions):
        del mean_actions
        calls.append(payoff_matrices.shape[0])
        batch = payoff_matrices.shape[0]
        if len(calls) == 1:
            values = [0.0, 1.0, 0.0, 0.0]
        else:
            values = [0.0, 10.0, 20.0, 30.0]
        return torch.tensor([values] * batch, dtype=torch.float32, device=payoff_matrices.device)

    agent.q_net = ZeroCurrentQNet()
    agent.target_net = TargetNet()
    agent.opt = torch.optim.Adam(agent.q_net.parameters(), lr=0.0)
    agent._robust_action_values = fake_robust_action_values

    loss = agent.train_step()

    assert loss == pytest.approx(100.0)


def test_torch_robust_agent_passes_features_to_payoff_network():
    agent = _make_torch_agent(feature_dim=3)
    obs = np.random.randn(4, 2, 3, 3).astype(np.float32)
    mean = np.full((4, 4), 0.25, dtype=np.float32)
    feature = np.arange(12, dtype=np.float32).reshape(4, 3)

    class FeatureCheckingNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_feature = None

        def payoff_matrix(self, obs_t, feature_t=None):
            del obs_t
            self.seen_feature = feature_t.detach().cpu().numpy()
            batch = feature_t.shape[0]
            matrix = torch.zeros(batch, 4, 4, device=feature_t.device)
            matrix[:, 1, :] = feature_t[:, :1]
            matrix[:, 2, :] = 1.0
            return matrix

    agent.q_net = FeatureCheckingNet()

    actions = agent.act_batch(obs, mean, feature)

    assert actions.shape == (4,)
    np.testing.assert_allclose(agent.q_net.seen_feature, feature)


def test_training_factory_selects_torch_robust_agent():
    cfg = {
        "algorithm": "mf_srq_torch",
        "type_prefixes": {"red": "red_"},
        "epsilon_robust_start": 0.1,
    }
    agent = make_mfdsrq_agent(
        cfg,
        type_id=0,
        obs_shape=(2, 3, 3),
        n_own_actions=4,
        n_nbr_actions=4,
        device=torch.device("cpu"),
        feature_dim=3,
    )

    assert isinstance(agent, TorchRobustMFDsrqAgent)
    assert agent.algorithm_name == "mf_srq_torch"
    assert agent.feature_dim == 3
    assert not hasattr(agent, "sre_solver")
