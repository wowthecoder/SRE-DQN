import ast
import json
from pathlib import Path

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
    from sr_adidas.schedules import EpsilonSchedule

    assert robust_epsilon_value(0.5, "constant", 50, 100) == 0.5

    schedule = EpsilonSchedule(
        start=0.5,
        end=0.5,
        decay_fraction=1.0,
        total_steps=100,
        mode="linear",
    )
    values = []
    for _ in range(25):
        schedule.step()
        values.append(schedule.value())
    assert values == [0.5] * 25


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


def test_sr_adidas_full_resume_checkpoint_contains_replay_and_state(tmp_path):
    torch = pytest.importorskip("torch")
    from sr_adidas.sr_adidas_agent import SrAdidasAgent

    agent = SrAdidasAgent(
        obs_dim=4,
        num_agents=2,
        num_actions=2,
        buffer_size=10,
        batch_size=2,
        learning_starts=2,
        total_steps=10,
        use_gpu=False,
    )
    agent.push([0, 0, 0, 0], [0, 1], [1.0, 0.0], [0, 0, 0, 1], False)
    path = tmp_path / "sr_adidas.pt"
    agent.save_checkpoint(path, include_replay_buffer=True, metadata={"scenario_key": "tiny"})

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "replay_buffer" in payload
    assert "y_cache" in payload
    assert "update_calls" in payload
    assert "train_step_calls" in payload
    assert payload["metadata"]["scenario_key"] == "tiny"


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


def test_new_lbf_notebooks_have_parseable_code_cells():
    notebook_paths = [
        ROOT / "discrete_action_space/lbf_grid/deepsrq_nfgtransformer_training.ipynb",
        ROOT / "discrete_action_space/lbf_grid/deepsrq_nfgtransformer_evaluation.ipynb",
        ROOT / "discrete_action_space/lbf_grid/deepsrq_path_mcp_nplayer_pool_training.ipynb",
        ROOT / "discrete_action_space/lbf_grid/deepsrq_path_mcp_nplayer_pool_evaluation.ipynb",
        ROOT / "discrete_action_space/lbf_grid/sr_adidas_training.ipynb",
        ROOT / "discrete_action_space/lbf_grid/sr_adidas_evaluation.ipynb",
    ]
    for path in notebook_paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{path}:{index}")
