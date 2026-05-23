import numpy as np
import pytest

pytest.importorskip("lbforaging")

from lbf_grid.pz_wrapper import make_pz_env
from lbf_grid.epymarl_lbf_env import EPYMARL_LBF_SCENARIOS, ExactLevelForagingEnv
from lbf_grid.notebook_eval import sample_lbf_rollouts, sample_lbf_rollouts_vectorized
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

                assert levels == expected_levels[key]
                assert len(_food_positions(env)) == len(expected_levels[key])
                metrics = infos["player_0"]["lbf_metrics"]
                assert sorted(food["level"] for food in metrics["initial_foods"]) == expected_levels[key]
            finally:
                env.close()


def test_dense_epymarl_gym_env_spawns_requested_food_count():
    scenario = EPYMARL_LBF_SCENARIOS["lbf_10x10_3p_8f_levels123"]
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
    )
    try:
        env.reset(seed=0)
        _prepare_manual_field(env, [(3, 3, 2)])
        _place_player(env, 0, (3, 2), level=1)
        _place_player(env, 1, (3, 4), level=1)
        env._inner._gen_valid_moves()

        _, rewards, _, _, infos = env.step({"player_0": 5, "player_1": 5})
        metrics = infos["player_0"]["lbf_metrics"]

        assert rewards["player_0"] == pytest.approx(2.0)
        assert rewards["player_1"] == pytest.approx(2.0)
        assert metrics["foods_collected_total"] == 1
        assert metrics["foods_collected_per_agent"] == {"agent_0": 1, "agent_1": 1}
        assert metrics["foods_collected_by_agent"]["agent_0"] == [
            {"step": 1, "row": 3, "col": 3, "level": 2}
        ]
        assert metrics["invalid_loads_total"] == 0
    finally:
        env.close()


def test_single_rollout_records_lbf_episode_metrics():
    rollouts = sample_lbf_rollouts(
        make_env=_make_invalid_load_metric_env,
        policy_fn=lambda **_: [5, 0],
        seed=0,
        n_episodes=1,
        max_steps=1,
        show_progress=False,
        capture_first_episode_frames=False,
    )

    metric = rollouts["episode_metrics"][0]
    assert rollouts["episode_lengths"] == [1]
    assert metric["initial_agent_positions"][0]["row"] == 3
    assert metric["initial_foods"] == [{"row": 3, "col": 3, "level": 3}]
    assert metric["episode_length"] == 1
    assert metric["invalid_loads_total"] == 1
    assert rollouts["metric_totals"]["invalid_loads_total"] == 1


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
