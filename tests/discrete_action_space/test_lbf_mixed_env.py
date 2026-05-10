import pytest

pytest.importorskip("lbforaging")

from lbf_grid.env import CustomForagingEnv


def _positions(env):
    return [p.position for p in env._inner.players]


def test_wall_blocks_movement_without_double_move():
    env = CustomForagingEnv(
        players=1,
        field_size=(5, 5),
        sight=5,
        max_food=1,
        max_episode_steps=5,
        start_positions=[(1, 1)],
        food_positions=[(3, 3)],
        food_levels=[1],
        wall_positions=[(1, 2)],
        collision_penalty=0.0,
        trap_penalty=0.0,
    )
    try:
        env.reset(seed=0)
        _, rewards, _, _, info = env.step([4])

        assert _positions(env) == [(1, 1)]
        assert rewards == [0.0]
        assert info["wall_block_agents"] == [0]
    finally:
        env.close()


def test_collision_penalties_can_distinguish_movers_and_blockers():
    env = CustomForagingEnv(
        players=2,
        field_size=(5, 5),
        sight=5,
        max_food=1,
        max_episode_steps=5,
        start_positions=[(1, 1), (1, 3)],
        food_positions=[(3, 3)],
        food_levels=[1],
        collision_penalty=-2.0,
        collision_mover_penalty=-0.5,
        trap_penalty=0.0,
    )
    try:
        env.reset(seed=1)
        _, rewards, _, _, info = env.step([4, 3])

        assert _positions(env) == [(1, 1), (1, 3)]
        assert rewards == [-2.5, -2.5]
        assert info["collision_agents"] == [0, 1]
    finally:
        env.close()


def test_trap_penalty_applies_on_entry_only_by_default():
    env = CustomForagingEnv(
        players=1,
        field_size=(5, 5),
        sight=5,
        max_food=1,
        max_episode_steps=5,
        start_positions=[(1, 1)],
        food_positions=[(3, 3)],
        food_levels=[1],
        trap_positions=[(1, 2)],
        collision_penalty=0.0,
        trap_penalty=-7.0,
    )
    try:
        env.reset(seed=2)
        _, rewards, _, _, info = env.step([4])
        assert _positions(env) == [(1, 2)]
        assert rewards == [-7.0]
        assert info["trap_agents"] == [0]

        _, rewards, _, _, info = env.step([0])
        assert _positions(env) == [(1, 2)]
        assert rewards == [0.0]
        assert info["trap_agents"] == []
    finally:
        env.close()


def test_mixed_food_rewards_stack_on_successful_cooperative_load():
    env = CustomForagingEnv(
        players=2,
        field_size=(5, 5),
        sight=5,
        max_food=1,
        max_episode_steps=5,
        start_positions=[(1, 1), (1, 3)],
        player_levels=[1, 1],
        food_positions=[(1, 2)],
        food_levels=[2],
        food_types=["coop"],
        collision_penalty=0.0,
        trap_penalty=0.0,
        team_food_reward=3.0,
        personal_food_rewards=[1.0, 2.0],
        preferred_food_bonus=[{"coop": 0.5}, {"coop": 1.5}],
        last_loader_bonus=[0.25, 0.75],
    )
    try:
        env.reset(seed=3)
        _, rewards, done, _, info = env.step([5, 5])

        assert bool(done) is True
        assert env._inner.field[1, 2] == 0
        assert rewards == pytest.approx([5.25, 7.75])
        assert info["loaded_foods"] == [
            {
                "position": (1, 2),
                "level": 2,
                "food_type": "coop",
                "participants": [0, 1],
            }
        ]
    finally:
        env.close()


def test_failed_cooperative_load_does_not_apply_food_shaping():
    env = CustomForagingEnv(
        players=2,
        field_size=(5, 5),
        sight=5,
        max_food=1,
        max_episode_steps=5,
        start_positions=[(1, 1), (1, 3)],
        player_levels=[1, 1],
        food_positions=[(1, 2)],
        food_levels=[3],
        food_types=["coop"],
        collision_penalty=0.0,
        trap_penalty=0.0,
        team_food_reward=3.0,
        personal_food_rewards=[1.0, 2.0],
        preferred_food_bonus=[{"coop": 0.5}, {"coop": 1.5}],
        last_loader_bonus=[0.25, 0.75],
    )
    try:
        env.reset(seed=4)
        _, rewards, done, _, info = env.step([5, 5])

        assert bool(done) is False
        assert env._inner.field[1, 2] == 3
        assert rewards == [0.0, 0.0]
        assert info["loaded_foods"] == []
    finally:
        env.close()
