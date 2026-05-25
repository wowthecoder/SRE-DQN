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
        deepsrq_evaluation_dir,
        deepsrq_path_mcp_pool_evaluation_dir,
        deepsrq_path_mcp_pool_training_dir,
        deepsrq_training_dir,
        sr_adidas_evaluation_dir,
        sr_adidas_training_dir,
    )

    assert deepsrq_training_dir("scenario_a", 0.01, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/deepsrq_nfgtransformer/training/scenario_a/0.01"
    )
    assert deepsrq_evaluation_dir("scenario_a", 1.0, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/deepsrq_nfgtransformer/evaluation/scenario_a/1.0"
    )
    assert deepsrq_path_mcp_pool_training_dir("scenario_a", 0.01, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/deepsrq_path_mcp_nplayer_pool/training/scenario_a/0.01"
    )
    assert deepsrq_path_mcp_pool_evaluation_dir("scenario_a", 1.0, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/deepsrq_path_mcp_nplayer_pool/evaluation/scenario_a/1.0"
    )
    assert sr_adidas_training_dir("scenario_b", 0.5, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/sr_adidas/training/scenario_b/0.5"
    )
    assert sr_adidas_evaluation_dir("scenario_b", 0.1, repo_root=tmp_path) == (
        tmp_path / "discrete_action_space/lbf_grid/sr_adidas/evaluation/scenario_b/0.1"
    )


def test_constant_epsilon_schedules_stay_constant():
    from lbf_grid.deep_srq_lbf import robust_epsilon_value

    assert robust_epsilon_value(0.5, "constant", 50, 100) == 0.5


def test_rotated_episode_counts_allocate_all_episodes():
    from lbf_grid.robust_notebook_utils import rotated_episode_counts

    assert rotated_episode_counts(500, 3) == [167, 167, 166]
    assert rotated_episode_counts(500, 2) == [250, 250]
    assert sum(rotated_episode_counts(7, 3)) == 7


def test_nfg_transformer_usage_summary_reports_fallback_rate():
    from sre_solvers.nfg_transformer.solver import NfgTransformerSreSolver

    solver = NfgTransformerSreSolver.__new__(NfgTransformerSreSolver)
    solver.checkpoint_path = None
    solver.fallback_enabled = True
    solver.neural_accept_count = 3
    solver.fallback_count = 1
    solver.neural_gap_sum = 1.0
    solver.neural_gap_sumsq = 0.5
    solver.neural_gap_min = 0.1
    solver.neural_gap_max = 0.4

    summary = solver.get_usage_summary()

    assert summary["total_decisions"] == 4
    assert summary["neural_accept_rate"] == pytest.approx(0.75)
    assert summary["fallback_rate"] == pytest.approx(0.25)
    assert summary["neural_robust_exploitability"]["count"] == 4


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

    class FakeDirectPolicySolver:
        name = "nfg_transformer_sre"
        bypass_deep_srq_policy_cache = True

        def solve_policy_batch_torch(self, q_tensors, epsilon):
            del epsilon
            batch_size = int(q_tensors.shape[0])
            return [
                torch.tensor([[1.0, 0.0]], dtype=torch.float32).expand(batch_size, -1),
                torch.tensor([[0.0, 1.0]], dtype=torch.float32).expand(batch_size, -1),
            ]

        def close(self):
            pass

    monkeypatch.setattr(deep_srq_lbf, "make_pz_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        deep_srq_lbf,
        "_make_solver",
        lambda solver_name, hp, seed: FakeDirectPolicySolver(),
    )

    stats = deep_srq_lbf.train_lbf_deep_srq_vectorized_experiment(
        n_episodes=2,
        num_envs=2,
        solver_name="nfg_transformer_sre",
        epsilon_robust_initial=0.1,
        epsilon_schedule="constant",
        seed=123,
        run_dir=tmp_path,
        lbf_config_overrides={"max_episode_steps": 1},
        hyperparameter_overrides={
            "learning_starts": 99,
            "batch_size": 2,
            "action_epsilon_start": 0.0,
            "action_epsilon_end": 0.0,
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


def test_new_lbf_notebooks_have_parseable_code_cells():
    notebook_paths = [
        ROOT / "discrete_action_space/lbf_grid/deepsrq_nfgtransformer_training.ipynb",
        ROOT / "discrete_action_space/lbf_grid/deepsrq_nfgtransformer_evaluation.ipynb",
        ROOT / "discrete_action_space/lbf_grid/deepsrq_path_pool_training.ipynb",
        ROOT / "discrete_action_space/lbf_grid/deepsrq_path_pool_evaluation.ipynb",
        ROOT / "discrete_action_space/lbf_grid/sr_adidas_training.ipynb",
        ROOT / "discrete_action_space/lbf_grid/sr_adidas_evaluation.ipynb",
    ]
    for path in notebook_paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{path}:{index}")
