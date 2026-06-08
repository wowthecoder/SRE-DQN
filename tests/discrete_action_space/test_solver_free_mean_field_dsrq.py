"""Tests for the torch-only mean-field SRQ implementation."""

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
from mean_field_dsrq.torch_robust_mean_field_dsrq import (  # noqa: E402
    MeanFieldReplayBuffer,
    PairwiseMeanFieldQNetwork,
    TorchRobustMFDsrqAgent,
)
from mean_field_dsrq.train_mf_dsrq import (  # noqa: E402
    conditioning_group_idx,
    make_mfdsrq_agent,
    normalize_training_config,
    resolve_mean_field_source,
)


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
        epsilon_explore=0.0,
        device=torch.device("cpu"),
    )
    defaults.update(overrides)
    return TorchRobustMFDsrqAgent(**defaults)


def test_pairwise_network_forward_is_mean_weighted_payoff_sum():
    torch.manual_seed(0)
    net = PairwiseMeanFieldQNetwork(2, 3, 3, n_own_actions=3, n_mean_actions=4, feature_dim=5)
    obs = torch.randn(5, 2, 3, 3)
    feature = torch.randn(5, 5)
    mean = torch.softmax(torch.randn(5, 4), dim=-1)

    payoff = net.payoff_matrix(obs, feature)
    out = net(obs, mean, feature=feature)

    expected = torch.bmm(payoff, mean.unsqueeze(-1)).squeeze(-1)
    assert out.shape == (5, 3)
    torch.testing.assert_close(out, expected)


def test_pairwise_network_conditions_payoffs_on_feature_vector():
    torch.manual_seed(0)
    net = PairwiseMeanFieldQNetwork(2, 3, 3, n_own_actions=3, n_mean_actions=4, feature_dim=2)
    obs = torch.randn(1, 2, 3, 3).repeat(2, 1, 1, 1)
    feature = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)

    payoff = net.payoff_matrix(obs, feature)

    assert payoff.shape == (2, 3, 4)
    assert not torch.allclose(payoff[0], payoff[1])


def test_replay_buffer_samples_feature_vectors_for_training():
    agent = _make_agent(feature_dim=3)
    mean = np.full(4, 0.25, dtype=np.float32)
    feature = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    next_feature = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    for i in range(4):
        obs = np.random.randn(2, 3, 3).astype(np.float32)
        next_obs = np.random.randn(2, 3, 3).astype(np.float32)
        agent.push(
            obs,
            i % 4,
            float(i),
            next_obs,
            mean,
            mean,
            done=False,
            feature=feature,
            next_feature=next_feature,
        )

    batch = agent.buffer.sample(4, torch.device("cpu"))

    assert batch["feature"].shape == (4, 3)
    assert batch["next_feature"].shape == (4, 3)
    torch.testing.assert_close(batch["feature"][0], torch.as_tensor(feature))
    torch.testing.assert_close(batch["next_feature"][0], torch.as_tensor(next_feature))


def test_replay_buffer_owns_pushed_arrays():
    buffer = MeanFieldReplayBuffer(capacity=4)
    obs = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    next_obs = obs + 100.0
    mean = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    next_mean = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    feature = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    next_feature = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    expected = tuple(
        value.copy()
        for value in (obs, feature, next_obs, next_feature, mean, next_mean)
    )

    buffer.push(
        obs,
        1,
        0.5,
        next_obs,
        mean,
        next_mean,
        done=False,
        feature=feature,
        next_feature=next_feature,
    )
    for value in (obs, next_obs, mean, next_mean, feature, next_feature):
        value.fill(-999.0)

    stored = buffer.buffer[0]

    np.testing.assert_allclose(stored[0], expected[0])
    np.testing.assert_allclose(stored[1], expected[1])
    np.testing.assert_allclose(stored[4], expected[2])
    np.testing.assert_allclose(stored[5], expected[3])
    np.testing.assert_allclose(stored[6], expected[4])
    np.testing.assert_allclose(stored[7], expected[5])


def test_checkpoint_round_trip(tmp_path):
    agent = _make_agent(robust_policy_temperature=0.25)
    ckpt = tmp_path / "agent.pt"
    agent.epsilon_robust = 0.33
    agent.epsilon_explore = 0.12
    agent.save_checkpoint(ckpt)

    loaded = _make_agent()
    loaded.load_checkpoint(ckpt)

    assert loaded.epsilon_robust == pytest.approx(0.33)
    assert loaded.epsilon_explore == pytest.approx(0.12)
    assert loaded.robust_policy_temperature == pytest.approx(0.25)
    for p_saved, p_loaded in zip(agent.q_net.parameters(), loaded.q_net.parameters()):
        torch.testing.assert_close(p_saved, p_loaded)


def test_training_factory_defaults_to_torch_agent():
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

    assert isinstance(agent, TorchRobustMFDsrqAgent)
    assert agent.algorithm_name == "mf_srq_torch"


def test_training_factory_rejects_deleted_algorithms():
    cfg = {
        "algorithm": "mf_srq_lp",
        "type_prefixes": {"red": "red_"},
        "epsilon_robust_start": 0.1,
    }
    with pytest.raises(ValueError, match="mf_srq_torch"):
        make_mfdsrq_agent(
            cfg,
            type_id=0,
            obs_shape=(2, 3, 3),
            n_own_actions=4,
            n_nbr_actions=4,
            device=torch.device("cpu"),
        )


def test_normalize_config_uses_constant_epsilon_when_end_missing():
    cfg = normalize_training_config(
        {
            "algorithm": "mf_srq_torch",
            "env_name": "battle_v4",
            "epsilon_robust_start": 0.1,
        },
        derive_schedule_output_dir=True,
    )

    assert cfg["epsilon_robust_end"] == pytest.approx(0.1)
    assert cfg["output_dir"].endswith("mf_srq_torch_epsilon_training/eps_0_1")


def test_normalize_config_derives_decay_and_ramp_output_dirs():
    decay = normalize_training_config(
        {
            "algorithm": "mf_srq_torch",
            "env_name": "battle_v4",
            "epsilon_robust_start": 0.5,
            "epsilon_robust_end": 0.0,
        },
        derive_schedule_output_dir=True,
    )
    ramp = normalize_training_config(
        {
            "algorithm": "mf_srq_torch",
            "env_name": "battle_v4",
            "epsilon_robust_start": 0.01,
            "epsilon_robust_end": 1.0,
        },
        derive_schedule_output_dir=True,
    )

    assert decay["output_dir"].endswith("mf_srq_torch_epsilon_decay_to_zero/start_0_5_to_0")
    assert ramp["output_dir"].endswith("mf_srq_torch_epsilon_ramp_up/start_0_01_to_1")


def test_mean_field_source_defaults_to_opponent_conditioning():
    assert resolve_mean_field_source({}) == "opponent"
    assert conditioning_group_idx(0, "opponent") == 1
    assert conditioning_group_idx(1, "opponent") == 0
    assert conditioning_group_idx(0, "same_team") == 0
    assert conditioning_group_idx(1, "self") == 1
    with pytest.raises(ValueError, match="mean_field_source"):
        resolve_mean_field_source({"mean_field_source": "nearby"})


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


class _FakeTwoTeamParallelEnv:
    def __init__(self):
        self.agents = ["red_0", "red_1", "blue_0", "blue_1"]

    def reset(self):
        self.agents = ["red_0", "red_1", "blue_0", "blue_1"]
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


def test_magent_wrapper_same_team_mean_action_is_exact_previous_histogram():
    wrapper = MAgentMFWrapper(
        _FakeParallelEnv,
        {"red": "red_"},
        mean_field_source="same_team",
    )
    wrapper.reset()

    _, _, _, mean_t, mean_tp1, _ = wrapper.step({"red_0": 1, "red_1": 1})

    np.testing.assert_allclose(mean_t["red_0"], [0.5, 0.5])
    np.testing.assert_allclose(mean_tp1["red_0"], [0.0, 1.0])
    np.testing.assert_allclose(mean_tp1["red_1"], [0.0, 1.0])


def test_magent_wrapper_default_mean_action_uses_opponent_histogram():
    wrapper = MAgentMFWrapper(
        _FakeTwoTeamParallelEnv,
        {"red": "red_", "blue": "blue_"},
    )
    wrapper.reset()

    _, _, _, mean_t, mean_tp1, _ = wrapper.step(
        {
            "red_0": 1,
            "red_1": 1,
            "blue_0": 0,
            "blue_1": 0,
        }
    )

    np.testing.assert_allclose(mean_t["red_0"], [0.5, 0.5])
    np.testing.assert_allclose(mean_t["blue_0"], [0.5, 0.5])
    np.testing.assert_allclose(mean_tp1["red_0"], [1.0, 0.0])
    np.testing.assert_allclose(mean_tp1["red_1"], [1.0, 0.0])
    np.testing.assert_allclose(mean_tp1["blue_0"], [0.0, 1.0])
    np.testing.assert_allclose(mean_tp1["blue_1"], [0.0, 1.0])
