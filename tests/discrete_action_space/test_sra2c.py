import numpy as np
import pytest

from sre_solvers import SreSolveResult


class _FakeBatchSolver:
    name = "fake_sra2c_solver"

    def __init__(self):
        self.calls = 0
        self.shapes = []
        self.closed = False

    def solve_batch(self, q_tensors, epsilon, **_kwargs):
        del epsilon
        self.calls += 1
        results = []
        for q_tensor in q_tensors:
            q_tensor = np.asarray(q_tensor)
            self.shapes.append(q_tensor.shape)
            results.append(
                SreSolveResult(
                    policies=[
                        np.array([1.0, 0.0], dtype=np.float64),
                        np.array([0.0, 1.0], dtype=np.float64),
                    ],
                    solutions=[],
                    utilities_sr=[],
                    utilities_nominal=[],
                    success=True,
                )
            )
        return results

    def close(self):
        self.closed = True


def test_sra2c_action_sampling_respects_masks():
    pytest.importorskip("torch")
    from sra2c import Sra2cAgent, Sra2cConfig

    agent = Sra2cAgent(
        Sra2cConfig(
            state_dim=3,
            actor_obs_dim=2,
            num_agents=2,
            num_actions=2,
            epsilon_explore=1.0,
            use_gpu=False,
            sre_solver=_FakeBatchSolver(),
        )
    )
    try:
        for _ in range(20):
            actions = agent.act_joint(
                np.zeros(3, dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                action_masks=[
                    np.array([True, False]),
                    np.array([False, True]),
                ],
            )
            assert actions == [0, 1]
    finally:
        agent.close()


def test_sra2c_builds_critic_payoff_tensor_from_query_critic():
    pytest.importorskip("torch")
    from sra2c import Sra2cAgent, Sra2cConfig

    agent = Sra2cAgent(
        Sra2cConfig(
            state_dim=3,
            actor_obs_dim=2,
            num_agents=2,
            num_actions=3,
            use_gpu=False,
            sre_solver=_FakeBatchSolver(),
        )
    )
    try:
        q_tensor, action_indices = agent._payoff_game_from_critic(
            agent.critic,
            np.zeros(3, dtype=np.float32),
            [
                np.array([True, False, True]),
                np.array([False, True, True]),
            ],
        )
        assert q_tensor.shape == (2, 2, 2)
        assert [indices.tolist() for indices in action_indices] == [[0, 2], [1, 2]]
    finally:
        agent.close()


def test_sra2c_train_step_updates_critic_value_and_actor():
    pytest.importorskip("torch")
    from sra2c import Sra2cAgent, Sra2cConfig

    solver = _FakeBatchSolver()
    agent = Sra2cAgent(
        Sra2cConfig(
            state_dim=3,
            actor_obs_dim=2,
            num_agents=2,
            num_actions=2,
            batch_size=1,
            learning_starts=1,
            train_every=1,
            rollout_steps=1,
            actor_update_every=1,
            epsilon_explore=0.0,
            use_gpu=False,
            sre_solver=solver,
            sre_target_value_mode="nominal",
        )
    )
    try:
        result = agent.update(
            state=np.zeros(3, dtype=np.float32),
            local_obs=np.zeros((2, 2), dtype=np.float32),
            joint_actions=[0, 1],
            joint_rewards=[1.0, 2.0],
            next_state=np.ones(3, dtype=np.float32),
            next_local_obs=np.ones((2, 2), dtype=np.float32),
            done=np.ones(2, dtype=np.float32),
            action_masks=[
                np.array([True, True]),
                np.array([True, True]),
            ],
            next_action_masks=None,
        )
        assert result is not None
        assert result["critic_loss"] is not None
        assert result["value_loss"] is not None
        assert result["actor_loss"] is not None
        assert result["entropy"] is not None
        assert result["sre_imitation_loss"] is not None
        assert result["valid_critic_rows"] == 1
        assert result["valid_value_rows"] == 1
        assert result["valid_actor_rows"] == 1
        assert solver.calls >= 1
    finally:
        agent.close()
    assert solver.closed is True


def test_sra2c_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from sra2c import Sra2cAgent, Sra2cConfig

    del torch
    agent = Sra2cAgent(
        Sra2cConfig(
            state_dim=3,
            actor_obs_dim=2,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=_FakeBatchSolver(),
        )
    )
    try:
        checkpoint = tmp_path / "sra2c.pt"
        agent.save_checkpoint(checkpoint, include_replay_buffer=True)
        restored = Sra2cAgent(
            Sra2cConfig(
                state_dim=3,
                actor_obs_dim=2,
                num_agents=2,
                num_actions=2,
                use_gpu=False,
                sre_solver=_FakeBatchSolver(),
            )
        )
        try:
            restored.load_checkpoint(checkpoint, map_location="cpu")
            assert restored.get_usage_summary()["algorithm"] == "sra2c"
        finally:
            restored.close()
    finally:
        agent.close()
