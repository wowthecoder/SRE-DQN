"""Notebook wiring tests for mean_field_dsrq MAgent2 experiments."""

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_DISCRETE = _ROOT / "discrete_action_space"
for p in [str(_ROOT), str(_DISCRETE)]:
    if p not in sys.path:
        sys.path.insert(0, p)


_MF_DIR = _ROOT / "discrete_action_space" / "mean_field_dsrq"


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_baseline_notebook_has_separate_algorithm_cells():
    path = _MF_DIR / "magent2_mfrl_baselines.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]

    for algorithm in ["iql", "ac", "mfq"]:
        matching_cells = [
            source for source in code_cells
            if f'train_mfrl_baseline("{algorithm}"' in source
        ]
        assert len(matching_cells) == 1, f"{algorithm} should have exactly one run cell"

        video_cells = [
            source for source in code_cells
            if "baseline_rollout_video_from_notebook(" in source
            and f"{algorithm}_result" in source
        ]
        assert len(video_cells) == 1, f"{algorithm} should have exactly one video cell"

        plot_cells = [
            source for source in code_cells
            if "plot_mfrl_baseline_training_curves(" in source
            and f'BASELINE_RUN_DIRS["{algorithm}"]' in source
        ]
        assert len(plot_cells) == 1, f"{algorithm} should have exactly one training-curve cell"


def test_notebooks_use_shared_helpers_not_cli_scripts():
    baseline_source = _notebook_source(_MF_DIR / "magent2_mfrl_baselines.ipynb")
    mfdsrq_source = "\n".join(
        _notebook_source(_MF_DIR / name)
        for name in [
            "magent2_mf_dsrq_torch_training.ipynb",
            "magent2_mf_dsrq_evaluation.ipynb",
        ]
    )

    assert "def find_repo_root" in baseline_source
    assert "def find_repo_root" in mfdsrq_source
    assert "train_mfrl_baseline" in baseline_source
    assert "find_latest_mfrl_run" in baseline_source
    assert "BASELINE_RUN_DIRS" in baseline_source
    assert "plot_mfrl_baseline_training_curves" in baseline_source
    assert "baseline_rollout_video_from_notebook" in baseline_source
    assert "train_mfdsrq_from_notebook" in mfdsrq_source
    assert "evaluate_mfdsrq_torch_epsilon_fixed_side_tournament" in mfdsrq_source
    assert "plot_fixed_side_tournament_bars" in mfdsrq_source
    assert "EVALUATE_BOTH_SIDES" not in mfdsrq_source
    assert "subprocess" not in baseline_source + mfdsrq_source
    assert "python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq" not in mfdsrq_source
    old_framework_name = "bench" + "marl"
    assert old_framework_name not in baseline_source.lower()


def test_evaluation_notebook_has_torch_model_cells_for_training_configs():
    notebook = json.loads((_MF_DIR / "magent2_mf_dsrq_evaluation.ipynb").read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]

    model_cells = [
        source for source in code_cells
        if "evaluate_and_plot_model(" in source
        and "def evaluate_and_plot_model" not in source
    ]
    expected_model_keys = {
        "fixed_0_01",
        "fixed_0_1",
        "fixed_0_5",
        "fixed_1_0",
        "decay_0_5_to_0",
        "decay_0_75_to_0",
        "decay_1_0_to_0",
        "ramp_0_01_to_0_5",
        "ramp_0_01_to_1_0",
        "ramp_0_1_to_0_5",
        "ramp_0_1_to_1_0",
    }

    assert len(model_cells) == len(expected_model_keys)
    for model_key in expected_model_keys:
        assert any(f'evaluate_and_plot_model("{model_key}")' in source for source in model_cells)

    history_cells = [
        source for source in code_cells
        if "plot_training_history_group(" in source
        and "def plot_training_history_group" not in source
    ]
    assert len(history_cells) == len(expected_model_keys)
    for model_key in expected_model_keys:
        assert any(f'("{model_key}",)' in source for source in history_cells)
    notebook_source = "\n".join(code_cells)
    assert "TRAINING_PLOT_SMOOTHING_WINDOW = 50" in notebook_source
    assert "baseline_training_history_frame" in notebook_source
    assert "for algorithm in BASELINE_ALGORITHMS" in notebook_source
    assert 'df["reward_main"].rolling(smoothing_window' in notebook_source
    assert 'df["reward_opponent"].rolling(smoothing_window' in notebook_source


def test_mfdsrq_mfrl_comparison_uses_low_level_battle_shapes():
    from mean_field_dsrq import eval_mf_dsrq

    source = inspect.getsource(eval_mf_dsrq._evaluate_mfdsrq_vs_mfrl_assignment)

    assert "LowLevelBattleEnv" in source
    assert "MAgentMFWrapper" not in source


def test_fixed_side_tournament_uses_main_vs_opponent_roles():
    from mean_field_dsrq import eval_mf_dsrq

    source = inspect.getsource(eval_mf_dsrq._evaluate_fixed_side_tournament_matchup)

    assert 'role="main"' in source
    assert 'role="opponent"' in source
    assert "main_algorithm" in source
    assert "opponent_algorithm" in source


def test_mfrl_helper_imports_without_magent2():
    from mean_field_dsrq.magent2_env import DEFAULT_TASK_CONFIG
    from mean_field_dsrq.mfrl_baselines import (
        BASELINE_ALGORITHMS,
        DEFAULT_MFRL_TASK_CONFIG,
        find_latest_mfrl_baseline_run,
    )

    assert set(BASELINE_ALGORITHMS) == {"iql", "ac", "mfq"}
    assert DEFAULT_TASK_CONFIG["extra_features"] is False
    assert DEFAULT_MFRL_TASK_CONFIG["extra_features"] is True

    with pytest.raises(FileNotFoundError):
        find_latest_mfrl_baseline_run("iql", _ROOT / "tmp_missing_mfrl_runs")


def test_value_baseline_actions_sample_boltzmann_policy_in_training(monkeypatch):
    from mean_field_dsrq.mfrl_baselines import ValueNet

    model = ValueNet((5, 5, 3), 4, 3)
    for param in model.eval_net.parameters():
        param.data.zero_()
    model.eval_net["final_linear"][-1].bias.data.copy_(torch.tensor([0.0, 1.0, 2.0]))

    obs = torch.zeros(4, 3, 5, 5)
    feature = torch.zeros(4, 4)

    deterministic_actions = model.act(obs, feature, deterministic=True)
    assert deterministic_actions.tolist() == [2, 2, 2, 2]

    def fake_sample(distribution):
        return torch.tensor([0, 1, 0, 1], device=distribution.probs.device)

    monkeypatch.setattr(torch.distributions.Categorical, "sample", fake_sample)

    sampled_actions = model.act(obs, feature, eps=1.0, deterministic=False)
    assert sampled_actions.tolist() == [0, 1, 0, 1]


def test_mfrl_training_update_caps_default_to_reference_unbounded():
    from mean_field_dsrq.mfrl_baselines import train_mfrl_baseline

    signature = inspect.signature(train_mfrl_baseline)

    assert signature.parameters["max_train_batches_per_update"].default is None
    assert signature.parameters["max_policy_samples_per_update"].default is None


def test_mfrl_run_finder_returns_newest_stats_folder(tmp_path):
    from mean_field_dsrq.mfrl_baselines import find_latest_mfrl_baseline_run

    old = tmp_path / "iql_battle_v4_seed42_old"
    new = tmp_path / "iql_battle_v4_seed42_new"
    old.mkdir()
    new.mkdir()
    (old / "training_stats.json").write_text("{}", encoding="utf-8")
    (new / "training_stats.json").write_text("{}", encoding="utf-8")

    assert find_latest_mfrl_baseline_run("iql", tmp_path) == new


def test_mfrl_eval_mode_does_not_call_custom_train_method():
    from mean_field_dsrq.mfrl_baselines import ActorCritic, _set_mfrl_model_eval_mode

    model = ActorCritic((3, 3, 2), 4, 5)

    _set_mfrl_model_eval_mode(model)

    assert model.net.training is False


def test_magent2_env_factory_imports_without_magent2(monkeypatch):
    from mean_field_dsrq.magent2_env import (
        DEFAULT_TASK_CONFIG,
        make_magent2_parallel_env_factory,
    )

    assert "use_mask" not in DEFAULT_TASK_CONFIG


def test_magent2_task_smoke_when_installed():
    pytest.importorskip("magent2")
    from mean_field_dsrq.mfrl_baselines import LowLevelBattleEnv

    env = LowLevelBattleEnv({"map_size": 12, "max_cycles": 2})
    meta = env.meta()
    assert meta.view_space[-1] >= 5
    assert meta.feature_space > 0
