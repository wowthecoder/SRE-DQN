import numpy as np
import pytest

pytest.importorskip("lbforaging")

from lbf_grid.pz_wrapper import make_pz_env


def _food_positions(env):
    return sorted(map(tuple, np.argwhere(env._inner.field > 0)))


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
