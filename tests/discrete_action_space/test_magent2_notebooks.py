"""Notebook wiring tests for mean_field_dsrq MAgent2 experiments."""

import json
import sys
import types
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

    for algorithm in ["mappo", "ippo", "iql"]:
        matching_cells = [
            source for source in code_cells
            if f'run_benchmarl_algorithm("{algorithm}"' in source
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
    assert "run_benchmarl_algorithm" in baseline_source
    assert "baseline_rollout_video_from_notebook" in baseline_source
    assert "train_mfdsrq_from_notebook" in mfdsrq_source
    assert "evaluate_mfdsrq_from_notebook" in mfdsrq_source
    assert "subprocess" not in baseline_source + mfdsrq_source
    assert "python -m discrete_action_space.mean_field_dsrq.train_mf_dsrq" not in mfdsrq_source


def test_benchmarl_helper_imports_without_magent2():
    from mean_field_dsrq.benchmarl_magent2 import (
        ALGORITHM_NAMES,
        DEFAULT_TASK_CONFIG,
        DEFAULT_USE_MASK,
        latest_checkpoint,
        make_experiment_config,
    )

    assert set(ALGORITHM_NAMES) == {"mappo", "ippo", "qmix", "vdn", "iql"}
    assert "use_mask" not in DEFAULT_TASK_CONFIG
    assert DEFAULT_USE_MASK is True
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


def test_magent2_wrapper_uses_mask_without_passing_it_to_env(monkeypatch):
    from mean_field_dsrq.benchmarl_magent2 import (
        _make_compatible_magent_env,
        make_magent_task,
    )

    captured = {}

    class FakeParallelEnv:
        def reset(self):
            return {}, {}

        def step(self, actions):
            return {}, {}, {}, {}, {}

    def parallel_env(**kwargs):
        captured["env_kwargs"] = kwargs
        return FakeParallelEnv()

    class FakePettingZooWrapper:
        def __init__(self, **kwargs):
            captured["wrapper_kwargs"] = kwargs

    magent2_mod = types.ModuleType("magent2")
    environments_mod = types.ModuleType("magent2.environments")
    battle_mod = types.ModuleType("magent2.environments.battle_v4")
    battle_mod.parallel_env = parallel_env
    torchrl_mod = types.ModuleType("torchrl")
    torchrl_envs_mod = types.ModuleType("torchrl.envs")
    torchrl_envs_mod.PettingZooWrapper = FakePettingZooWrapper

    monkeypatch.setitem(sys.modules, "magent2", magent2_mod)
    monkeypatch.setitem(sys.modules, "magent2.environments", environments_mod)
    monkeypatch.setitem(sys.modules, "magent2.environments.battle_v4", battle_mod)
    monkeypatch.setitem(sys.modules, "torchrl", torchrl_mod)
    monkeypatch.setitem(sys.modules, "torchrl.envs", torchrl_envs_mod)

    _make_compatible_magent_env(
        {
            "env_name": "battle_v4",
            "map_size": 40,
            "max_cycles": 5,
            "use_mask": True,
        },
        seed=0,
        device="cpu",
    )

    assert "use_mask" not in captured["env_kwargs"]
    assert captured["wrapper_kwargs"]["use_mask"] is True
    assert captured["wrapper_kwargs"]["done_on_any"] is False

    task = make_magent_task({"env_name": "battle_v4", "use_mask": True})
    assert task.use_mask is True
    assert "use_mask" not in task.config


def test_magent2_task_smoke_when_installed():
    pytest.importorskip("magent2")
    from mean_field_dsrq.benchmarl_magent2 import make_magent_task

    task = make_magent_task({"map_size": 40, "max_cycles": 5})
    assert task.name == "BATTLE"
    assert task.config["map_size"] == 40
