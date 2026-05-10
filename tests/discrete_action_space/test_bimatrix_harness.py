import pytest

pytest.importorskip("pygambit")

from bimatrix_game.GridWorld import GridWorldEnv
from bimatrix_game.experiment_harness import _per_agent_done_mask


def test_serial_done_mask_marks_all_agents_terminal_on_timeout():
    env = GridWorldEnv(max_steps=1)
    env.reset()

    _, _, done, _ = env.step([0, 0])
    assert done is False

    _, _, done, _ = env.step([0, 0])
    assert done is True
    assert _per_agent_done_mask(env, num_agents=2, done=done) == [True, True]


def test_serial_done_mask_preserves_partial_finished_state_before_episode_done():
    env = GridWorldEnv()
    env.reset()
    env.agents_finished = [True, False]

    assert _per_agent_done_mask(env, num_agents=2, done=False) == [True, False]
