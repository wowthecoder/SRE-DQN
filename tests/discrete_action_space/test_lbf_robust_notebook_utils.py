import ast
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
DISCRETE = ROOT / "discrete_action_space"
if str(DISCRETE) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(DISCRETE))


def test_robust_artifact_paths_use_requested_layout(tmp_path):
    from lbf_grid.robust_notebook_utils import (
        deepsrq_path_mcp_pool_evaluation_dir,
        deepsrq_path_mcp_pool_training_dir,
        sra2c_evaluation_dir,
        sra2c_training_dir,
        srac_evaluation_dir,
        srac_training_dir,
    )

    assert deepsrq_path_mcp_pool_training_dir("scenario_a", 0.01, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/deepsrq_path_mcp_nplayer_pool/training/scenario_a/0.01"
    )
    assert deepsrq_path_mcp_pool_evaluation_dir("scenario_a", 1.0, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/deepsrq_path_mcp_nplayer_pool/evaluation/scenario_a/1.0"
    )
    assert srac_training_dir("scenario_c", 0.5, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/srac/training/scenario_c/0.5"
    )
    assert srac_evaluation_dir("scenario_c", 1.0, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/srac/evaluation/scenario_c/1.0"
    )
    assert sra2c_training_dir("scenario_d", 0.1, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/sra2c/training/scenario_d/0.1"
    )
    assert sra2c_evaluation_dir("scenario_d", 0.01, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/sra2c/evaluation/scenario_d/0.01"
    )


def test_evaluation_agent_labels_show_fixed_algorithms():
    from lbf_grid.robust_notebook_utils import _evaluation_agent_labels

    labels, counts, note = _evaluation_agent_labels(
        primary_label="DeepSRQ",
        opponent_label="IQL",
        total_episodes=4,
        num_agents=3,
    )

    assert labels == [
        "Agent 1\nDeepSRQ",
        "Agent 2\nIQL",
        "Agent 3\nIQL",
    ]
    assert counts == [
        {"agent": 1, "DeepSRQ": 4},
        {"agent": 2, "IQL": 4},
        {"agent": 3, "IQL": 4},
    ]
    assert "Agent 1 uses DeepSRQ" in note


def test_deepsrq_evaluation_baselines_exclude_random():
    from lbf_grid.robust_notebook_utils import BASELINE_ALGORITHMS

    assert BASELINE_ALGORITHMS == ("iql", "ippo", "mappo", "maa2c")


def test_recent_mean_joint_reward_uses_interval_window():
    from lbf_grid.deep_srq_lbf import _recent_mean_joint_reward

    rewards_history = [
        [1.0, 3.0, 5.0, 7.0],
        [2.0, 4.0, 6.0, 8.0],
    ]

    assert _recent_mean_joint_reward(rewards_history, 2) == pytest.approx(13.0)
    assert _recent_mean_joint_reward(rewards_history, 100) == pytest.approx(9.0)
    assert _recent_mean_joint_reward([[], []], 2) is None


def test_deepsrq_full_resume_checkpoint_contains_replay(tmp_path):
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeSolver:
        name = "fake"

        def close(self):
            pass

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            buffer_size=10,
            learning_starts=2,
            sre_solver=FakeSolver(),
            use_gpu=False,
        )
    )
    agent.replay_buffer.push([0, 0, 0, 0], [0, 1], [1.0, 0.0], [0, 0, 0, 1], False)
    path = tmp_path / "deepsrq.pt"
    agent.save_checkpoint(path, include_replay_buffer=True)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "replay_buffer" in payload
    assert payload["sre_solver_name"] == "fake"
    agent.close()


def test_deepsrq_load_checkpoint_rejects_payload_without_config(tmp_path):
    torch = pytest.importorskip("torch")
    from dueling_double_dqn_sre import DuelingDoubleDqnSreAgent, DuelingDoubleDqnSreAgentConfig

    class FakeSolver:
        name = "fake"

        def close(self):
            pass

    agent = DuelingDoubleDqnSreAgent(
        DuelingDoubleDqnSreAgentConfig(
            obs_dim=4,
            num_agents=2,
            num_actions=2,
            sre_solver=FakeSolver(),
            use_gpu=False,
        )
    )
    path = tmp_path / "old_deepsrq.pt"
    torch.save({"q_net": agent.q_net.state_dict()}, path)

    with pytest.raises(ValueError, match="missing config metadata"):
        agent.load_checkpoint(path, map_location="cpu")

    agent.close()


def test_deepsrq_policy_adapter_batches_action_selection():
    from lbf_grid.robust_notebook_utils import DeepSrqPolicyAdapter

    class FakeAgent:
        def __init__(self):
            self.seen_states = None
            self.closed = False

        def act_joint_batch(self, states):
            self.seen_states = list(states)
            return [[idx, idx + 1] for idx, _ in enumerate(states)]

        def close(self):
            self.closed = True

    agent = FakeAgent()
    adapter = DeepSrqPolicyAdapter(agent)
    contexts = [
        {"state": [1.0, 0.0]},
        {"state": [0.0, 1.0]},
        {"state": [1.0, 1.0]},
    ]

    actions = adapter.act_all_batch(contexts)

    assert actions == [[0, 1], [1, 2], [2, 3]]
    assert agent.seen_states == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    adapter.close()
    assert agent.closed is True


def test_lbf_policy_actions_forward_action_masks():
    from lbf_grid.robust_notebook_utils import _policy_actions

    class FakePolicy:
        def __init__(self):
            self.seen_action_masks = None

        def act_all(self, **kwargs):
            self.seen_action_masks = kwargs.get("action_masks")
            return [0, 1]

    masks = [
        np.array([True, False]),
        np.array([False, True]),
    ]
    policy = FakePolicy()

    actions = _policy_actions(
        policy,
        state=np.zeros(2, dtype=np.float32),
        obs_dict={},
        agent_order=["agent_0", "agent_1"],
        env=None,
        step=0,
        action_masks=masks,
    )

    assert actions == [0, 1]
    assert policy.seen_action_masks is masks


@pytest.mark.parametrize(
    ("available_checkpoints", "expected_checkpoint"),
    [
        (("shared_deepsrq_final.pt", "shared_deepsrq_best.pt"), "shared_deepsrq_final.pt"),
        (("shared_deepsrq_best.pt",), "shared_deepsrq_best.pt"),
    ],
)
def test_load_deepsrq_path_mcp_pool_policy_prefers_final_then_best(
    tmp_path,
    monkeypatch,
    available_checkpoints,
    expected_checkpoint,
):
    torch = pytest.importorskip("torch")
    from lbf_grid import robust_notebook_utils as utils

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_stats.json").write_text(
        json.dumps(
            {
                "obs_dim": 4,
                "num_agents": 2,
                "num_actions": 3,
                "seed": 123,
                "solver_name": "fake_solver",
                "hyperparameters": {"agent": {}},
            }
        ),
        encoding="utf-8",
    )
    for filename in available_checkpoints:
        torch.save(
            {
                "config": {
                    "obs_dim": 4,
                    "num_agents": 2,
                    "num_actions": 3,
                    "network_type": "joint_output",
                    "q_hidden_dims": (128, 128),
                },
            },
            run_dir / filename,
        )

    class FakeSolver:
        name = "fake_solver"

        def close(self):
            pass

    class FakeAgent:
        instances = []

        def __init__(self, config):
            self.config = config
            self.loaded_checkpoint = None
            self.map_location = None
            self.closed = False
            self.instances.append(self)

        def load_checkpoint(self, path, map_location=None):
            self.loaded_checkpoint = Path(path)
            self.map_location = map_location
            self.config.sre_solver_name = "stale_checkpoint_solver"
            self.config.sre_solver_workers = 1

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        utils,
        "deepsrq_path_mcp_pool_training_dir",
        lambda *args, **kwargs: run_dir,
    )
    monkeypatch.setattr(utils, "make_sre_solver", lambda *args, **kwargs: FakeSolver())
    monkeypatch.setattr(utils, "DuelingDoubleDqnSreAgent", FakeAgent)

    scenario = utils.LbfNotebookScenario(
        key="scenario_a",
        name="Scenario A",
        gym_id="fake",
        time_limit=1,
        config={"players": 2},
    )

    adapter = utils.load_deepsrq_path_mcp_pool_policy(
        scenario,
        0.5,
        repo_root=tmp_path,
        use_gpu=False,
    )

    try:
        assert adapter.agent.loaded_checkpoint == run_dir / expected_checkpoint
        assert adapter.agent.map_location == "cpu"
        assert adapter.agent.config.sre_solver_name == "path_c_pool"
        assert adapter.agent.config.sre_solver_workers == 8
        assert adapter.agent.config.epsilon_explore == 0.0
        assert adapter.agent.config.epsilon_robust == 0.5
    finally:
        adapter.close()


def test_load_deepsrq_path_mcp_pool_policy_skips_incompatible_final(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from lbf_grid import robust_notebook_utils as utils

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_stats.json").write_text(
        json.dumps(
            {
                "obs_dim": 36,
                "num_agents": 2,
                "num_actions": 6,
                "seed": 123,
                "solver_name": "fake_solver",
                "hyperparameters": {
                    "agent": {
                        "network_type": "joint_output",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    checkpoint_payload = {
        "config": {
            "obs_dim": 36,
            "num_agents": 2,
            "num_actions": 6,
            "network_type": "joint_output",
            "q_hidden_dims": (128, 128),
        },
    }
    torch.save(checkpoint_payload, run_dir / "shared_deepsrq_final.pt")
    torch.save(checkpoint_payload, run_dir / "shared_deepsrq_best.pt")

    class FakeSolver:
        name = "fake_solver"

        def close(self):
            pass

    class FakeAgent:
        instances = []

        def __init__(self, config):
            self.config = config
            self.loaded_checkpoint = None
            self.closed = False
            self.instances.append(self)

        def load_checkpoint(self, path, map_location=None):
            del map_location
            if Path(path).name == "shared_deepsrq_final.pt":
                raise RuntimeError("size mismatch for feature.0.weight")
            self.loaded_checkpoint = Path(path)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        utils,
        "deepsrq_path_mcp_pool_training_dir",
        lambda *args, **kwargs: run_dir,
    )
    monkeypatch.setattr(utils, "make_sre_solver", lambda *args, **kwargs: FakeSolver())
    monkeypatch.setattr(utils, "DuelingDoubleDqnSreAgent", FakeAgent)

    scenario = utils.LbfNotebookScenario(
        key="scenario_a",
        name="Scenario A",
        gym_id="fake",
        time_limit=1,
        config={"players": 2},
    )

    adapter = utils.load_deepsrq_path_mcp_pool_policy(
        scenario,
        0.5,
        repo_root=tmp_path,
        use_gpu=False,
    )

    try:
        assert adapter.agent.loaded_checkpoint == run_dir / "shared_deepsrq_best.pt"
        assert FakeAgent.instances[0].closed is True
        assert FakeAgent.instances[1] is adapter.agent
        assert adapter.agent.closed is False
    finally:
        adapter.close()


def test_load_deepsrq_path_mcp_pool_policy_uses_checkpoint_model_config(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from lbf_grid import robust_notebook_utils as utils

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_stats.json").write_text(
        json.dumps(
            {
                "obs_dim": 99,
                "num_agents": 3,
                "num_actions": 6,
                "seed": 123,
                "solver_name": "fake_solver",
                "hyperparameters": {},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "config": {
                "obs_dim": 63,
                "num_agents": 3,
                "num_actions": 6,
                "network_type": "joint_output",
                "q_hidden_dims": (128, 128),
            },
        },
        run_dir / "shared_deepsrq_best.pt",
    )

    class FakeSolver:
        name = "fake_solver"

        def close(self):
            pass

    class FakeAgent:
        def __init__(self, config):
            self.config = config
            self.loaded_checkpoint = None

        def load_checkpoint(self, path, map_location=None):
            del map_location
            self.loaded_checkpoint = Path(path)

        def close(self):
            pass

    monkeypatch.setattr(
        utils,
        "deepsrq_path_mcp_pool_training_dir",
        lambda *args, **kwargs: run_dir,
    )
    monkeypatch.setattr(utils, "make_sre_solver", lambda *args, **kwargs: FakeSolver())
    monkeypatch.setattr(utils, "DuelingDoubleDqnSreAgent", FakeAgent)

    scenario = utils.LbfNotebookScenario(
        key="scenario_a",
        name="Scenario A",
        gym_id="fake",
        time_limit=1,
        config={"players": 3},
    )

    adapter = utils.load_deepsrq_path_mcp_pool_policy(
        scenario,
        0.1,
        repo_root=tmp_path,
        use_gpu=False,
    )

    try:
        assert adapter.agent.loaded_checkpoint == run_dir / "shared_deepsrq_best.pt"
        assert adapter.agent.config.obs_dim == 63
        assert adapter.agent.config.num_agents == 3
        assert adapter.agent.config.network_type == "joint_output"
    finally:
        adapter.close()


def test_srac_policy_adapter_batches_actor_action_selection():
    from lbf_grid.robust_notebook_utils import SracPolicyAdapter

    class FakeAgent:
        def __init__(self):
            self.seen_states = None
            self.seen_local_obs = None
            self.closed = False

        def act_joint_batch(self, states, local_obs_batch, action_masks_batch=None):
            del action_masks_batch
            self.seen_states = list(states)
            self.seen_local_obs = np.asarray(local_obs_batch)
            return [[idx, idx + 1] for idx, _ in enumerate(states)]

        def close(self):
            self.closed = True

    agent = FakeAgent()
    adapter = SracPolicyAdapter(agent)
    contexts = [
        {
            "state": [1.0, 0.0],
            "obs_dict": {
                "agent_0": np.array([0.0, 0.1]),
                "agent_1": np.array([1.0, 1.1]),
            },
            "agent_order": ["agent_0", "agent_1"],
        },
        {
            "state": [0.0, 1.0],
            "obs_dict": {
                "agent_0": np.array([2.0, 2.1]),
                "agent_1": np.array([3.0, 3.1]),
            },
            "agent_order": ["agent_0", "agent_1"],
        },
    ]

    actions = adapter.act_all_batch(contexts)

    assert actions == [[0, 1], [1, 2]]
    assert agent.seen_states == [[1.0, 0.0], [0.0, 1.0]]
    assert agent.seen_local_obs.shape == (2, 2, 2)
    adapter.close()
    assert agent.closed is True


def test_sra2c_policy_adapter_batches_actor_action_selection():
    from lbf_grid.robust_notebook_utils import Sra2cPolicyAdapter

    class FakeAgent:
        def __init__(self):
            self.seen_states = None
            self.seen_local_obs = None
            self.closed = False

        def act_joint_batch(self, states, local_obs_batch, action_masks_batch=None):
            del action_masks_batch
            self.seen_states = list(states)
            self.seen_local_obs = np.asarray(local_obs_batch)
            return [[idx, idx + 1] for idx, _ in enumerate(states)]

        def close(self):
            self.closed = True

    agent = FakeAgent()
    adapter = Sra2cPolicyAdapter(agent)
    contexts = [
        {
            "state": [1.0, 0.0],
            "obs_dict": {
                "agent_0": np.array([0.0, 0.1]),
                "agent_1": np.array([1.0, 1.1]),
            },
            "agent_order": ["agent_0", "agent_1"],
        },
        {
            "state": [0.0, 1.0],
            "obs_dict": {
                "agent_0": np.array([2.0, 2.1]),
                "agent_1": np.array([3.0, 3.1]),
            },
            "agent_order": ["agent_0", "agent_1"],
        },
    ]

    actions = adapter.act_all_batch(contexts)

    assert actions == [[0, 1], [1, 2]]
    assert agent.seen_states == [[1.0, 0.0], [0.0, 1.0]]
    assert agent.seen_local_obs.shape == (2, 2, 2)
    adapter.close()
    assert agent.closed is True


def test_load_srac_policy_uses_actor_only_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from lbf_grid.robust_notebook_utils import (
        LbfNotebookScenario,
        load_srac_policy,
    )
    from srac import SracAgent, SracConfig

    class FakeSolver:
        name = "fake_solver"

        def close(self):
            pass

    agent = SracAgent(
        SracConfig(
            state_dim=2,
            actor_obs_dim=2,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=FakeSolver(),
        )
    )
    try:
        checkpoint = tmp_path / "shared_srac_best.pt"
        agent.save_checkpoint(checkpoint)
    finally:
        agent.close()
    (tmp_path / "training_stats.json").write_text("{}", encoding="utf-8")
    scenario = LbfNotebookScenario(
        key="scenario_a",
        name="Scenario A",
        gym_id="fake",
        time_limit=1,
        config={},
    )

    adapter = load_srac_policy(
        scenario,
        0.5,
        run_dir_override=tmp_path,
        use_gpu=False,
    )
    try:
        actions = adapter.act_all(
            state=np.zeros(2, dtype=np.float32),
            obs_dict={
                "agent_0": np.zeros(2, dtype=np.float32),
                "agent_1": np.ones(2, dtype=np.float32),
            },
            agent_order=["agent_0", "agent_1"],
            action_masks=[
                np.array([True, False]),
                np.array([False, True]),
            ],
        )
        assert actions == [0, 1]
    finally:
        adapter.close()


def test_load_sra2c_policy_uses_actor_only_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from lbf_grid.robust_notebook_utils import (
        LbfNotebookScenario,
        load_sra2c_policy,
    )
    from sra2c import Sra2cAgent, Sra2cConfig

    class FakeSolver:
        name = "fake_solver"

        def close(self):
            pass

    agent = Sra2cAgent(
        Sra2cConfig(
            state_dim=2,
            actor_obs_dim=2,
            num_agents=2,
            num_actions=2,
            use_gpu=False,
            sre_solver=FakeSolver(),
        )
    )
    try:
        checkpoint = tmp_path / "shared_sra2c_best.pt"
        agent.save_checkpoint(checkpoint)
    finally:
        agent.close()
    (tmp_path / "training_stats.json").write_text("{}", encoding="utf-8")
    scenario = LbfNotebookScenario(
        key="scenario_a",
        name="Scenario A",
        gym_id="fake",
        time_limit=1,
        config={},
    )

    adapter = load_sra2c_policy(
        scenario,
        0.5,
        run_dir_override=tmp_path,
        use_gpu=False,
    )
    try:
        actions = adapter.act_all(
            state=np.zeros(2, dtype=np.float32),
            obs_dict={
                "agent_0": np.zeros(2, dtype=np.float32),
                "agent_1": np.ones(2, dtype=np.float32),
            },
            agent_order=["agent_0", "agent_1"],
            action_masks=[
                np.array([True, False]),
                np.array([False, True]),
            ],
        )
        assert actions == [0, 1]
    finally:
        adapter.close()


def test_vectorized_lbf_deepsrq_trainer_smoke(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from lbf_grid import deep_srq_lbf

    class FakeActionSpace:
        n = 2

        def sample(self):
            return 0

    class FakeEnv:
        possible_agents = ["agent_0", "agent_1"]

        def __init__(self):
            self.agents = list(self.possible_agents)
            self.step_count = 0

        def reset(self, seed=None):
            del seed
            self.agents = list(self.possible_agents)
            self.step_count = 0
            return {
                "agent_0": np.array([0.0, 0.0], dtype=np.float32),
                "agent_1": np.array([1.0, 1.0], dtype=np.float32),
            }, {}

        def action_space(self, agent):
            del agent
            return FakeActionSpace()

        def step(self, action_dict):
            del action_dict
            self.step_count += 1
            self.agents = []
            obs = {
                "agent_0": np.array([float(self.step_count), 0.0], dtype=np.float32),
                "agent_1": np.array([1.0, float(self.step_count)], dtype=np.float32),
            }
            rewards = {"agent_0": 1.0, "agent_1": 2.0}
            terms = {"agent_0": True, "agent_1": True}
            truncs = {"agent_0": False, "agent_1": False}
            return obs, rewards, terms, truncs, {}

        def close(self):
            pass

    class FakeBatchSolver:
        name = "fake_batch"

        def solve_batch(self, q_tensors, **kwargs):
            del kwargs
            from sre_solvers import SreSolveResult

            batch_size = int(len(q_tensors))
            return [
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
                for _ in range(batch_size)
            ]

        def close(self):
            pass

    monkeypatch.setattr(deep_srq_lbf, "LBFParallelEnv", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        deep_srq_lbf,
        "_make_solver",
        lambda solver_name, hp, seed: FakeBatchSolver(),
    )

    stats = deep_srq_lbf.train_lbf_deep_srq_vectorized(
        n_episodes=2,
        num_envs=2,
        solver_name="path_mcp_nplayer_pool",
        epsilon_robust_initial=0.1,
        epsilon_schedule="constant",
        seed=123,
        run_dir=tmp_path,
        lbf_config_overrides={"max_episode_steps": 1},
        hyperparameter_overrides={
            "agent": {
                "learning_starts": 99,
                "batch_size": 2,
                "action_epsilon_start": 0.0,
                "action_epsilon_end": 0.0,
            },
        },
        use_gpu=False,
        write_plots=False,
        include_replay_buffer=True,
        eval_interval=None,
        print_full_stats=False,
    )

    assert stats["training_mode"] == "vectorized"
    assert stats["num_envs"] == 2
    assert stats["completed_episodes"] == 2
    assert stats["total_environment_steps"] == 2
    assert stats["rewards"] == [[1.0, 1.0], [2.0, 2.0]]
    assert (tmp_path / "shared_deepsrq_best.pt").exists()
    assert (tmp_path / "shared_deepsrq_final.pt").exists()
    assert (tmp_path / "training_stats.json").exists()
    saved = json.loads((tmp_path / "training_stats.json").read_text())
    assert "rewards" not in saved
    assert "episode_lengths" not in saved
    assert saved["reward_summary"]["episodes"] == 2


def test_vectorized_lbf_sra2c_trainer_smoke(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from lbf_grid import sra2c_lbf

    class FakeActionSpace:
        n = 2

        def sample(self):
            return 0

    class FakeEnv:
        possible_agents = ["agent_0", "agent_1"]

        def __init__(self):
            self.agents = list(self.possible_agents)
            self.step_count = 0

        def reset(self, seed=None):
            del seed
            self.agents = list(self.possible_agents)
            self.step_count = 0
            return {
                "agent_0": np.array([0.0, 0.0], dtype=np.float32),
                "agent_1": np.array([1.0, 1.0], dtype=np.float32),
            }, {}

        def action_space(self, agent):
            del agent
            return FakeActionSpace()

        def step(self, action_dict):
            del action_dict
            self.step_count += 1
            self.agents = []
            obs = {
                "agent_0": np.array([float(self.step_count), 0.0], dtype=np.float32),
                "agent_1": np.array([1.0, float(self.step_count)], dtype=np.float32),
            }
            rewards = {"agent_0": 1.0, "agent_1": 2.0}
            terms = {"agent_0": True, "agent_1": True}
            truncs = {"agent_0": False, "agent_1": False}
            return obs, rewards, terms, truncs, {}

        def close(self):
            pass

    class FakeSolver:
        name = "fake_sra2c_solver"

        def close(self):
            pass

    monkeypatch.setattr(sra2c_lbf, "LBFParallelEnv", lambda **kwargs: FakeEnv())

    stats = sra2c_lbf.train_lbf_sra2c_vectorized(
        n_episodes=2,
        num_envs=2,
        solver_name="path_mcp_nplayer_pool",
        epsilon_robust_initial=0.1,
        epsilon_schedule="constant",
        seed=123,
        run_dir=tmp_path,
        lbf_config_overrides={"max_episode_steps": 1},
        hyperparameter_overrides={
            "agent": {
                "learning_starts": 99,
                "batch_size": 2,
                "action_epsilon_start": 0.0,
                "action_epsilon_end": 0.0,
                "sre_solver": FakeSolver(),
            },
        },
        use_gpu=False,
        write_plots=False,
        include_replay_buffer=True,
        eval_interval=None,
        print_full_stats=False,
    )

    assert stats["algorithm"] == "sra2c"
    assert stats["training_mode"] == "vectorized"
    assert stats["num_envs"] == 2
    assert stats["completed_episodes"] == 2
    assert stats["total_environment_steps"] == 2
    assert stats["rewards"] == [[1.0, 1.0], [2.0, 2.0]]
    assert (tmp_path / "shared_sra2c_best.pt").exists()
    assert (tmp_path / "shared_sra2c_final.pt").exists()
    assert (tmp_path / "training_stats.json").exists()
    saved = json.loads((tmp_path / "training_stats.json").read_text())
    assert "rewards" not in saved
    assert "episode_lengths" not in saved
    assert saved["reward_summary"]["episodes"] == 2


def test_new_lbf_notebooks_have_parseable_code_cells():
    notebook_paths = [
        ROOT / "discrete_action_space/lbf_grid/deepsrq_path_pool_training.ipynb",
        ROOT / "discrete_action_space/lbf_grid/deepsrq_path_pool_evaluation.ipynb",
        ROOT / "discrete_action_space/lbf_grid/srac_training.ipynb",
        ROOT / "discrete_action_space/lbf_grid/srac_evaluation.ipynb",
        ROOT / "discrete_action_space/lbf_grid/sra2c_training.ipynb",
        ROOT / "discrete_action_space/lbf_grid/sra2c_evaluation.ipynb",
    ]
    for path in notebook_paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{path}:{index}")
