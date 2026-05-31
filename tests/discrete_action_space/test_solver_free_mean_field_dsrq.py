"""Tests for the solver-free mean-field SRQ implementation."""

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

from mean_field_dsrq.magent_env_wrapper import MAgentMFWrapper  # noqa: E402
from mean_field_dsrq.solver_free_mean_field_dsrq import (  # noqa: E402
    PairwiseMeanFieldQNetwork,
    RobustMeanFieldResult,
    RobustMeanFieldSreOperator,
    SolverFreeMFDsrqAgent,
    _tv_worst_case_value,
)
from mean_field_dsrq.train_mf_dsrq import make_mfdsrq_agent  # noqa: E402


def test_robust_operator_epsilon_zero_matches_mean_field_greedy_value():
    matrix = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    mean = np.array([0.25, 0.75], dtype=np.float64)
    op = RobustMeanFieldSreOperator(num_actions=2, epsilon=0.0)

    result = op.solve(matrix, mean)

    assert result.success is True
    assert result.value == pytest.approx(1.5)
    np.testing.assert_allclose(result.policy, [0.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(result.worst_mean, mean, atol=1e-7)


def test_robust_operator_value_is_nonincreasing_with_epsilon():
    matrix = np.array([[3.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    mean = np.array([1.0, 0.0], dtype=np.float64)

    nominal = RobustMeanFieldSreOperator(num_actions=2, epsilon=0.0).solve(matrix, mean)
    robust = RobustMeanFieldSreOperator(num_actions=2, epsilon=1.0).solve(matrix, mean)

    assert robust.value <= nominal.value + 1e-7
    assert robust.value == pytest.approx(1.0)


def test_robust_operator_matches_bruteforce_grid_for_two_actions():
    matrix = np.array([[2.0, -1.0], [0.5, 1.2]], dtype=np.float64)
    mean = np.array([0.7, 0.3], dtype=np.float64)
    epsilon = 0.25
    op = RobustMeanFieldSreOperator(num_actions=2, epsilon=epsilon)

    result = op.solve(matrix, mean)

    grid_values = []
    for p0 in np.linspace(0.0, 1.0, 2001):
        policy = np.array([p0, 1.0 - p0], dtype=np.float64)
        values = policy @ matrix
        grid_values.append(_tv_worst_case_value(mean, values, epsilon))
    assert result.value == pytest.approx(max(grid_values), abs=2e-3)


def test_pairwise_network_forward_is_mean_weighted_payoff_sum():
    torch.manual_seed(0)
    net = PairwiseMeanFieldQNetwork(2, 3, 3, n_own_actions=3, n_mean_actions=4)
    obs = torch.randn(5, 2, 3, 3)
    mean = torch.softmax(torch.randn(5, 4), dim=-1)

    payoff = net.payoff_matrix(obs)
    out = net(obs, mean)

    expected = torch.bmm(payoff, mean.unsqueeze(-1)).squeeze(-1)
    assert out.shape == (5, 3)
    torch.testing.assert_close(out, expected)


class FakeRobustOperator:
    def __init__(self, policy):
        self.policy = np.asarray(policy, dtype=np.float32)
        self.calls = 0

    def solve(self, payoff_matrix, mean_action, epsilon=None):
        del payoff_matrix, mean_action, epsilon
        self.calls += 1
        return RobustMeanFieldResult(
            policy=self.policy,
            value=0.5,
            worst_mean=np.full(self.policy.size, 1.0 / self.policy.size, dtype=np.float32),
            lambda_value=0.0,
            success=True,
        )

    def worst_case_mean(self, values, mean_action, epsilon=None):
        del values, epsilon
        return np.asarray(mean_action, dtype=np.float32)


def _make_agent(**overrides):
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
    return SolverFreeMFDsrqAgent(**defaults)


def test_action_selection_uses_solver_free_robust_policy():
    agent = _make_agent()
    agent.robust_operator = FakeRobustOperator([0.0, 0.0, 1.0, 0.0])
    obs = np.random.randn(6, 2, 3, 3).astype(np.float32)
    mean = np.full((6, 4), 0.25, dtype=np.float32)

    actions = agent.act_batch(obs, mean)

    assert actions.tolist() == [2] * 6
    assert agent.robust_operator.calls == 6
    assert not hasattr(agent, "sre_solver")


def test_train_step_updates_from_replay_without_path_solver():
    torch.manual_seed(1)
    np.random.seed(1)
    agent = _make_agent()
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


def test_checkpoint_round_trip(tmp_path):
    agent = _make_agent()
    ckpt = tmp_path / "agent.pt"
    agent.epsilon_robust = 0.33
    agent.epsilon_explore = 0.12
    agent.save_checkpoint(ckpt)

    loaded = _make_agent()
    loaded.load_checkpoint(ckpt)

    assert loaded.epsilon_robust == pytest.approx(0.33)
    assert loaded.epsilon_explore == pytest.approx(0.12)
    for p_saved, p_loaded in zip(agent.q_net.parameters(), loaded.q_net.parameters()):
        torch.testing.assert_close(p_saved, p_loaded)


def test_training_factory_defaults_to_solver_free_agent():
    cfg = {
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
    )

    assert isinstance(agent, SolverFreeMFDsrqAgent)
    assert not hasattr(agent, "sre_solver")


class _Space:
    def __init__(self, n=None, shape=None):
        self.n = n
        self.shape = shape


class _FakeParallelEnv:
    def __init__(self):
        self.agents = ["red_0", "red_1"]

    def reset(self):
        self.agents = ["red_0", "red_1"]
        return {aid: np.zeros((2, 2, 1), dtype=np.float32) for aid in self.agents}, {}

    def action_space(self, agent):
        del agent
        return _Space(n=2)

    def observation_space(self, agent):
        del agent
        return _Space(shape=(2, 2, 1))

    def step(self, actions):
        obs = {aid: np.ones((2, 2, 1), dtype=np.float32) for aid in self.agents}
        rewards = {aid: 0.0 for aid in self.agents}
        terms = {aid: False for aid in self.agents}
        truncs = {aid: False for aid in self.agents}
        return obs, rewards, terms, truncs, {}


def test_magent_wrapper_default_mean_action_is_exact_previous_histogram():
    wrapper = MAgentMFWrapper(
        _FakeParallelEnv,
        {"red": "red_"},
    )
    wrapper.reset()

    _, _, _, mean_t, mean_tp1, _ = wrapper.step({"red_0": 1, "red_1": 1})

    np.testing.assert_allclose(mean_t["red_0"], [0.5, 0.5])
    np.testing.assert_allclose(mean_tp1["red_0"], [0.0, 1.0])
    np.testing.assert_allclose(mean_tp1["red_1"], [0.0, 1.0])

