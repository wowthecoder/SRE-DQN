"""Notebook wiring tests for mean_field_dsrq MAgent2 experiments."""

import json
import sys
from pathlib import Path

import pytest

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
    path = _MF_DIR / "magent2_benchmarl_baselines.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]

    for algorithm in ["mappo", "ippo", "qmix", "vdn", "iql"]:
        matching_cells = [
            source for source in code_cells
            if f'run_baseline_from_notebook("{algorithm}"' in source
        ]
        assert len(matching_cells) == 1, f"{algorithm} should have exactly one run cell"

        video_cells = [
            source for source in code_cells
            if "baseline_rollout_video_from_notebook(" in source
            and f"{algorithm}_result" in source
        ]
        assert len(video_cells) == 1, f"{algorithm} should have exactly one video cell"


def test_notebooks_use_shared_helpers_not_cli_scripts():
    baseline_source = _notebook_source(_MF_DIR / "magent2_benchmarl_baselines.ipynb")
    mfdsrq_source = _notebook_source(_MF_DIR / "magent2_mf_dsrq.ipynb")

    assert "def find_repo_root" in baseline_source
    assert "def find_repo_root" in mfdsrq_source
    assert "run_baseline_from_notebook" in baseline_source
    assert "baseline_rollout_video_from_notebook" in baseline_source
    assert "train_mfdsrq_from_notebook" in mfdsrq_source
    assert "evaluate_mfdsrq_from_notebook" in mfdsrq_source
    assert "subprocess" not in baseline_source + mfdsrq_source
    assert "python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq" not in mfdsrq_source


def test_benchmarl_helper_imports_without_magent2():
    from mean_field_dsrq.benchmarl_magent2 import (
        ALGORITHM_NAMES,
        latest_checkpoint,
        make_experiment_config,
    )

    assert set(ALGORITHM_NAMES) == {"mappo", "ippo", "qmix", "vdn", "iql"}
    cfg = make_experiment_config("iql", total_frames=100, frames_per_batch=50)
    assert cfg.prefer_continuous_actions is False
    assert cfg.off_policy_collected_frames_per_batch == 50

    checkpoint_dir = _ROOT / "tmp_missing_checkpoints"
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(checkpoint_dir)


def test_benchmarl_experiment_config_creates_nested_save_folder(tmp_path):
    from mean_field_dsrq.benchmarl_magent2 import make_experiment_config

    save_folder = tmp_path / "runs" / "benchmarl_magent2_notebooks"
    assert not save_folder.exists()
    cfg = make_experiment_config("mappo", save_folder=save_folder)
    assert save_folder.is_dir()
    assert cfg.save_folder == str(save_folder)


def test_parallel_env_api_compat_normalizes_reset_and_step_tuple_shapes():
    from mean_field_dsrq.benchmarl_magent2 import (
        ParallelEnvApiCompat,
        normalize_pettingzoo_parallel_api,
    )

    class FakeEnv:
        def reset(self):
            return {"agent_0": 1}, {"agent_0": {}}, "extra"

        def step(self, actions):
            assert actions == {"agent_0": 0}
            return (
                {"agent_0": 2},
                {"agent_0": 1.0},
                {"agent_0": False},
                {"agent_0": False},
                {"agent_0": {}},
                "extra",
            )

    env = ParallelEnvApiCompat(FakeEnv())
    obs, infos = env.reset()
    assert obs == {"agent_0": 1}
    assert infos == {"agent_0": {}}

    step = env.step({"agent_0": 0})
    assert len(step) == 5
    assert step[0] == {"agent_0": 2}

    fake = FakeEnv()
    patched = normalize_pettingzoo_parallel_api(fake)
    assert patched is fake
    obs, infos = patched.reset()
    assert obs == {"agent_0": 1}
    assert infos == {"agent_0": {}}
    assert len(patched.step({"agent_0": 0})) == 5


def test_magent2_task_smoke_when_installed():
    pytest.importorskip("magent2")
    from mean_field_dsrq.benchmarl_magent2 import make_magent_task

    task = make_magent_task({"map_size": 8, "max_cycles": 5})
    assert task.name == "ADVERSARIAL_PURSUIT"
    assert task.config["map_size"] == 8
