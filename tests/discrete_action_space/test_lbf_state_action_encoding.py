import numpy as np

from lbforaging.foraging.environment import Action

from lbf_grid.pz_wrapper import make_pz_env


def _set_player(env, player_id, row, col):
    env._inner.players[player_id].position = (int(row), int(col))


def test_lbf_action_masks_match_deep_srq_reduction_rules():
    env = make_pz_env(
        players=1,
        field_size=(3, 3),
        sight=3,
        max_food=1,
        player_levels=[1],
        food_levels=[1],
        normalize_reward=False,
    )
    try:
        env.reset(seed=7)
        _set_player(env, 0, 0, 0)
        env._inner.field[:] = 0

        mask = env.action_masks(["player_0"])[0]
        assert not mask[Action.NONE.value]
        assert not mask[Action.LOAD.value]
        assert not mask[Action.NORTH.value]
        assert not mask[Action.WEST.value]
        assert mask[Action.SOUTH.value]
        assert mask[Action.EAST.value]

        env._inner.field[:] = 0
        env._inner.field[0, 1] = 1
        mask = env.action_masks(["player_0"])[0]
        assert mask[Action.NONE.value]
        assert mask[Action.LOAD.value]
        assert not mask[Action.NORTH.value]
        assert not mask[Action.WEST.value]
        assert mask[Action.SOUTH.value]
        assert mask[Action.EAST.value]
    finally:
        env.close()


def test_lbf_canonical_state_uses_live_global_positions_and_pads_foods():
    env = make_pz_env(
        players=2,
        field_size=(4, 4),
        sight=4,
        max_food=3,
        player_levels=[1, 2],
        food_levels=[1, 2, 3],
        normalize_reward=False,
    )
    try:
        env.reset(seed=11)
        _set_player(env, 0, 1, 2)
        _set_player(env, 1, 3, 0)
        env._inner.field[:] = 0
        env._inner.field[0, 3] = 2
        env._inner.field[2, 1] = 1

        state = env.global_state(["player_0", "player_1"])
        expected = np.array(
            [
                1,
                2,
                1,
                3,
                0,
                2,
                0,
                3,
                2,
                2,
                1,
                1,
                -1,
                -1,
                0,
            ],
            dtype=np.float32,
        )
        assert state.dtype == np.float32
        np.testing.assert_array_equal(state, expected)
    finally:
        env.close()
