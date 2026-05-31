import numpy as np
import pytest

pytest.importorskip("lbforaging")

from lbf_grid.pz_wrapper import make_pz_env
from lbf_grid.deep_srq_lbf import _print_lbf_evaluation_metrics
from lbf_grid.epymarl_lbf_env import EPYMARL_LBF_SCENARIOS, ExactLevelForagingEnv
from lbf_grid.notebook_eval import (
    best_joint_reward_rollout_index,
    capture_lbf_rollout_frames_from_actions,
    sample_lbf_rollouts_vectorized,
)
from lbf_grid.robust_notebook_utils import scenario_to_lbf_config


def _food_positions(env):
    return sorted(map(tuple, np.argwhere(env._inner.field > 0)))


def _food_levels_from_field(field):
    return sorted(int(value) for value in field[field > 0])


def _place_player(env, player_id, position, level=None):
    player = env._inner.players[player_id]
    player.position = tuple(position)
    if level is not None:
        player.level = int(level)


def _prepare_manual_field(env, foods):
    env._inner.field[:] = 0
    for row, col, level in foods:
        env._inner.field[row, col] = int(level)
    env._inner._food_spawned = float(env._inner.field.sum())
    env._inner._game_over = False
    env._inner._gen_valid_moves()


def _make_invalid_load_metric_env():
    env = make_pz_env(
        players=2,
        player_levels=[1, 1],
        field_size=(6, 6),
        max_food=1,
        food_levels=[3],
        normalize_reward=False,
        penalty=0.0,
        empty_load_penalty=0.01,
    )
    original_reset = env.reset

    def reset_with_layout(seed=None, options=None):
        obs, infos = original_reset(seed=seed, options=options)
        _prepare_manual_field(env, [(3, 3, 3)])
        _place_player(env, 0, (3, 2), level=1)
        _place_player(env, 1, (1, 1), level=1)
        env._inner._gen_valid_moves()
        obs_list = env._inner._make_gym_obs()
        obs = {
            agent: obs_list[index]
            for index, agent in enumerate(env.possible_agents)
        }
        metrics = infos["player_0"]["lbf_metrics"]
        metrics["initial_agent_positions"] = [
            {"agent": "player_0", "agent_id": 0, "row": 3, "col": 2, "level": 1},
            {"agent": "player_1", "agent_id": 1, "row": 1, "col": 1, "level": 1},
        ]
        metrics["initial_foods"] = [{"row": 3, "col": 3, "level": 3}]
        env._inner._lbf_metrics["initial_agent_positions"] = metrics[
            "initial_agent_positions"
        ]
        env._inner._lbf_metrics["initial_foods"] = metrics["initial_foods"]
        infos = {agent: {"lbf_metrics": metrics} for agent in env.possible_agents}
        return obs, infos

    env.reset = reset_with_layout
    return env


def test_basic_lbf_defaults_are_three_agent_full_observation():
    env = make_pz_env()
    try:
        obs, _ = env.reset(seed=0)

        assert env.possible_agents == ["player_0", "player_1", "player_2"]
        assert env._inner.field_size == (10, 10)
        assert env._inner.sight == 10
        assert len(obs) == 3
        assert len(_food_positions(env)) == 3
    finally:
        env.close()


def test_seed_controls_random_food_positions():
    env_a = make_pz_env(max_food=3, food_levels=[1, 2, 3])
    env_b = make_pz_env(max_food=3, food_levels=[1, 2, 3])
    env_c = make_pz_env(max_food=3, food_levels=[1, 2, 3])
    try:
        env_a.reset(seed=123)
        env_b.reset(seed=123)
        env_c.reset(seed=124)

        assert np.array_equal(env_a._inner.field, env_b._inner.field)
        assert not np.array_equal(env_a._inner.field, env_c._inner.field)
    finally:
        env_a.close()
        env_b.close()
        env_c.close()


def test_player_levels_can_be_fixed_per_agent():
    env = make_pz_env(players=3, player_levels=[1, 2, 3])
    try:
        env.reset(seed=7)

        assert [player.level for player in env._inner.players] == [1, 2, 3]
    finally:
        env.close()


def test_food_count_and_exact_food_levels_are_tunable():
    env = make_pz_env(max_food=4, food_levels=[1, 1, 2, 3])
    try:
        env.reset(seed=9)
        spawned_levels = sorted(int(value) for value in env._inner.field[env._inner.field > 0])

        assert len(_food_positions(env)) == 4
        assert spawned_levels == [1, 1, 2, 3]
    finally:
        env.close()


def test_dense_epymarl_scenarios_spawn_requested_food_multisets():
    expected_levels = {
        "lbf_8x8_2p_2f_levels12": [1, 1, 1, 1, 1, 2, 2, 3, 3, 3],
        "lbf_8x8_2p_2f_force_coop": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "lbf_10x10_3p_8f_levels123": [
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3,
            4,
            4,
            4,
            5,
            5,
            5,
            6,
            6,
            6,
        ],
    }

    for key, scenario in EPYMARL_LBF_SCENARIOS.items():
        config = scenario_to_lbf_config(scenario)
        for seed in range(5):
            env = make_pz_env(**config)
            try:
                _, infos = env.reset(seed=seed)
                levels = _food_levels_from_field(env._inner.field)

                assert config["normalize_reward"] is False
                assert config["simple_food_rewards"] is True
                assert config["penalty"] == pytest.approx(0.0)
                assert config["empty_load_penalty"] == pytest.approx(0.01)
                assert levels == expected_levels[key]
                assert len(_food_positions(env)) == len(expected_levels[key])
                metrics = infos["player_0"]["lbf_metrics"]
                assert sorted(food["level"] for food in metrics["initial_foods"]) == expected_levels[key]
            finally:
                env.close()


def test_dense_epymarl_gym_env_spawns_requested_food_count():
    scenario = EPYMARL_LBF_SCENARIOS["lbf_10x10_3p_8f_levels123"]
    assert scenario.kwargs["normalize_reward"] is False
    assert scenario.kwargs["simple_food_rewards"] is True
    env = ExactLevelForagingEnv(**scenario.kwargs)
    try:
        _, info = env.reset(seed=0)

        assert len(_food_records := info["lbf_metrics"]["initial_foods"]) == 18
        assert sorted(food["level"] for food in _food_records) == [
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3,
            4,
            4,
            4,
            5,
            5,
            5,
            6,
            6,
            6,
        ]
    finally:
        env.close()


def test_simple_food_reward_grants_food_level_to_solo_collector():
    env = make_pz_env(
        players=2,
        player_levels=[3, 1],
        field_size=(6, 6),
        max_food=1,
        food_levels=[3],
        normalize_reward=False,
        penalty=0.0,
        empty_load_penalty=0.01,
        simple_food_rewards=True,
    )
    try:
        env.reset(seed=0)
        _prepare_manual_field(env, [(3, 3, 3)])
        _place_player(env, 0, (3, 2), level=3)
        _place_player(env, 1, (1, 1), level=1)
        env._inner._gen_valid_moves()

        _, rewards, _, _, infos = env.step({"player_0": 5, "player_1": 0})
        metrics = infos["player_0"]["lbf_metrics"]

        assert rewards["player_0"] == pytest.approx(3.0)
        assert rewards["player_1"] == pytest.approx(0.0)
        assert metrics["foods_collected_total"] == 1
        assert metrics["foods_collected_per_agent"] == {"agent_0": 1, "agent_1": 0}
    finally:
        env.close()


def test_empty_load_penalty_only_applies_to_empty_load():
    env = make_pz_env(
        players=2,
        player_levels=[1, 1],
        field_size=(6, 6),
        max_food=1,
        food_levels=[1],
        normalize_reward=False,
        penalty=0.0,
        empty_load_penalty=0.01,
        simple_food_rewards=True,
    )
    try:
        env.reset(seed=0)
        _prepare_manual_field(env, [(4, 4, 1)])
        _place_player(env, 0, (1, 1), level=1)
        _place_player(env, 1, (1, 2), level=1)
        env._inner._gen_valid_moves()

        _, rewards, _, _, infos = env.step({"player_0": 5, "player_1": 0})
        metrics = infos["player_0"]["lbf_metrics"]

        assert rewards["player_0"] == pytest.approx(-0.01)
        assert rewards["player_1"] == pytest.approx(0.0)
        assert metrics["empty_loads_total"] == 1
        assert metrics["empty_loads_per_agent"]["agent_0"] == 1
        assert metrics["invalid_loads_total"] == 0
    finally:
        env.close()


def test_invalid_load_is_recorded_without_reward_penalty():
    env = make_pz_env(
        players=2,
        player_levels=[1, 1],
        field_size=(6, 6),
        max_food=1,
        food_levels=[3],
        normalize_reward=False,
        penalty=0.0,
        empty_load_penalty=0.01,
        simple_food_rewards=True,
    )
    try:
        env.reset(seed=0)
        _prepare_manual_field(env, [(3, 3, 3)])
        _place_player(env, 0, (3, 2), level=1)
        _place_player(env, 1, (1, 1), level=1)
        env._inner._gen_valid_moves()

        _, rewards, _, _, infos = env.step({"player_0": 5, "player_1": 0})
        metrics = infos["player_0"]["lbf_metrics"]

        assert rewards["player_0"] == pytest.approx(0.0)
        assert metrics["invalid_loads_total"] == 1
        assert metrics["invalid_loads_per_agent"]["agent_0"] == 1
        assert metrics["empty_loads_total"] == 0
    finally:
        env.close()


def test_successful_collection_records_food_per_participating_agent():
    env = make_pz_env(
        players=2,
        player_levels=[1, 1],
        field_size=(6, 6),
        max_food=1,
        food_levels=[2],
        normalize_reward=False,
        penalty=0.0,
        empty_load_penalty=0.01,
        simple_food_rewards=True,
    )
    try:
        env.reset(seed=0)
        _prepare_manual_field(env, [(3, 3, 2)])
        _place_player(env, 0, (3, 2), level=1)
        _place_player(env, 1, (3, 4), level=1)
        env._inner._gen_valid_moves()

        _, rewards, _, _, infos = env.step({"player_0": 5, "player_1": 5})
        metrics = infos["player_0"]["lbf_metrics"]

        assert rewards["player_0"] == pytest.approx(1.0)
        assert rewards["player_1"] == pytest.approx(1.0)
        assert metrics["foods_collected_total"] == 1
        assert metrics["foods_collected_per_agent"] == {"agent_0": 1, "agent_1": 1}
        assert metrics["foods_collected_by_agent"]["agent_0"] == [
            {"step": 1, "row": 3, "col": 3, "level": 2}
        ]
        assert metrics["invalid_loads_total"] == 0
    finally:
        env.close()


def test_vectorized_rollout_records_lbf_episode_metrics():
    rollouts = sample_lbf_rollouts_vectorized(
        make_env=_make_invalid_load_metric_env,
        policy_batch_fn=lambda contexts: [[5, 0] for _ in contexts],
        seed=0,
        n_episodes=2,
        max_steps=1,
        num_envs=2,
        show_progress=False,
        capture_first_episode_frames=False,
    )

    assert rollouts["episode_lengths"] == [1, 1]
    assert len(rollouts["episode_metrics"]) == 2
    assert [metric["invalid_loads_total"] for metric in rollouts["episode_metrics"]] == [1, 1]
    assert rollouts["metric_totals"]["invalid_loads_total"] == 2


def test_best_joint_reward_rollout_can_be_replayed_for_frames():
    reward_by_seed = {10: 0.0, 11: 5.0, 12: 2.0}
    instances = []

    class FakeEnv:
        possible_agents = ["player_0", "player_1"]

        def __init__(self):
            self.agents = list(self.possible_agents)
            self.seed = None
            self.steps = 0
            self.actions = []
            instances.append(self)

        def reset(self, seed=None, options=None):
            self.seed = int(seed)
            self.steps = 0
            self.actions = []
            self.agents = list(self.possible_agents)
            return {}, {}

        def step(self, action_dict):
            self.actions.append(dict(action_dict))
            reward = float(reward_by_seed[self.seed])
            self.steps += 1
            self.agents = []
            done = {agent: True for agent in self.possible_agents}
            truncated = {agent: False for agent in self.possible_agents}
            rewards = {"player_0": reward, "player_1": 0.0}
            return {}, rewards, done, truncated, {}

        def render(self, mode="rgb_array"):
            return np.full((2, 2, 3), self.seed * 10 + self.steps, dtype=np.uint8)

        def close(self):
            pass

    rollouts = sample_lbf_rollouts_vectorized(
        make_env=FakeEnv,
        policy_batch_fn=lambda contexts: [
            [context["episode_idx"], context["step"]] for context in contexts
        ],
        seed=10,
        n_episodes=3,
        max_steps=1,
        num_envs=3,
        show_progress=False,
        capture_first_episode_frames=False,
    )

    best_idx = best_joint_reward_rollout_index(rollouts)
    assert best_idx == 1
    assert rollouts["joint_rewards"] == [0.0, 5.0, 2.0]

    replay = capture_lbf_rollout_frames_from_actions(
        make_env=FakeEnv,
        actions=rollouts["rollouts"][best_idx]["actions"],
        seed=10,
        episode_idx=best_idx,
        max_steps=1,
    )

    assert replay["render_error"] is None
    assert [int(frame[0, 0, 0]) for frame in replay["frames"]] == [110, 111]
    assert instances[-1].actions == [{"player_0": 1, "player_1": 0}]


def test_training_summary_prints_latest_lbf_eval_metrics(capsys):
    metric = {
        "initial_agent_positions": [
            {"agent": "player_0", "agent_id": 0, "row": 1, "col": 2, "level": 1},
            {"agent": "player_1", "agent_id": 1, "row": 3, "col": 4, "level": 2},
        ],
        "initial_foods": [{"row": 5, "col": 6, "level": 3}],
        "episode_length": 7,
        "foods_collected_total": 1,
        "foods_collected_per_agent": {"agent_0": 1, "agent_1": 0},
        "foods_collected_by_agent": {
            "agent_0": [{"step": 3, "row": 5, "col": 6, "level": 3}],
            "agent_1": [],
        },
        "empty_loads_total": 2,
        "empty_loads_per_agent": {"agent_0": 1, "agent_1": 1},
        "invalid_loads_total": 1,
        "invalid_loads_per_agent": {"agent_0": 0, "agent_1": 1},
    }
    stats = {
        "num_agents": 2,
        "periodic_eval": [
            {
                "episode": 100,
                "global_step": 5000,
                "episode_metrics": [metric],
                "metric_totals": {
                    "episode_count": 1,
                    "episode_lengths": [7],
                    "foods_collected_total": 1,
                    "foods_collected_per_agent": {"agent_0": 1, "agent_1": 0},
                    "empty_loads_total": 2,
                    "empty_loads_per_agent": {"agent_0": 1, "agent_1": 1},
                    "invalid_loads_total": 1,
                    "invalid_loads_per_agent": {"agent_0": 0, "agent_1": 1},
                },
            }
        ],
    }

    _print_lbf_evaluation_metrics(stats)
    output = capsys.readouterr().out

    assert "LBF Evaluation Metrics" in output
    assert "Agent starting coordinates: Agent 1: (1, 2) L1, Agent 2: (3, 4) L2" in output
    assert "Food coordinates: (5, 6) L3" in output
    assert "Episode length: 7" in output
    assert "Foods collected total: 1" in output
    assert "Agent 1: (5, 6) L3" in output
    assert "Empty loads per agent: Agent 1: 1, Agent 2: 1" in output
    assert "Invalid loads per agent: Agent 1: 0, Agent 2: 1" in output


def test_pettingzoo_wrapper_uses_basic_lbf_defaults():
    env = make_pz_env()
    try:
        obs, infos = env.reset(seed=5)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        next_obs, rewards, terms, truncs, step_infos = env.step(actions)

        assert list(obs) == ["player_0", "player_1", "player_2"]
        assert set(infos) == set(obs)
        assert set(next_obs) == set(obs)
        assert set(rewards) == set(obs)
        assert set(terms) == set(obs)
        assert set(truncs) == set(obs)
        assert set(step_infos) == set(obs)
    finally:
        env.close()
