from pathlib import Path

from lbf_grid.epymarl_baselines import (
    EPYMARL_ALGORITHMS,
    SREDQN_TRAIN_UPDATES_PER_ENV_BATCH,
    build_epymarl_command,
    full_parallel_training_overrides,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_epymarl_algorithms_exclude_qmix():
    assert EPYMARL_ALGORITHMS == ("iql", "ippo", "mappo", "maa2c")


def test_qmix_is_rejected(tmp_path):
    epymarl_root = tmp_path / "epymarl"
    (epymarl_root / "src").mkdir(parents=True)
    (epymarl_root / "src" / "main.py").write_text("")

    try:
        build_epymarl_command(
            epymarl_root,
            "qmix",
            "lbf_8x8_2p_2f_levels12",
            output_root=tmp_path / "runs",
        )
    except ValueError as exc:
        assert "Unsupported EPyMARL algorithm: qmix" in str(exc)
    else:
        raise AssertionError("qmix should no longer be an active EPyMARL baseline")


def test_full_parallel_training_overrides_enable_full_batch_updates():
    overrides = full_parallel_training_overrides(
        "iql",
        batch_size_run=128,
        minibatch_size=32,
        use_cuda=False,
    )

    assert overrides["runner"] == "parallel"
    assert overrides["batch_size_run"] == 128
    assert overrides["batch_size"] == 32
    assert overrides[SREDQN_TRAIN_UPDATES_PER_ENV_BATCH] == "auto"
    assert overrides["use_cuda"] is False


def test_full_parallel_training_overrides_keep_pg_buffer_from_overwriting_batch():
    overrides = full_parallel_training_overrides(
        "mappo",
        batch_size_run=128,
        minibatch_size=32,
    )

    assert overrides["buffer_size"] == 128
    assert overrides["batch_size"] == 32
    assert overrides[SREDQN_TRAIN_UPDATES_PER_ENV_BATCH] == "auto"


def test_build_epymarl_command_passes_full_training_patch(tmp_path):
    epymarl_root = tmp_path / "epymarl"
    (epymarl_root / "src").mkdir(parents=True)
    (epymarl_root / "src" / "main.py").write_text("")

    command = build_epymarl_command(
        epymarl_root,
        "iql",
        "lbf_8x8_2p_2f_levels12",
        output_root=tmp_path / "runs",
        config_overrides=full_parallel_training_overrides("iql"),
        model_token="scenario/2025/iql",
    )

    bootstrap = command[2]
    argv_part = bootstrap.split("sys.argv = ['main.py'] + ", 1)[1]
    assert f"run_module.{SREDQN_TRAIN_UPDATES_PER_ENV_BATCH} = 'auto'" in bootstrap
    assert SREDQN_TRAIN_UPDATES_PER_ENV_BATCH not in argv_part
    assert "Could not patch EPyMARL train-update block" in bootstrap
    assert "episode_batch.batch_size" in bootstrap


def test_lbf_epymarl_notebook_prefers_best_checkpoint():
    notebook_path = (
        REPO_ROOT / "discrete_action_space" / "lbf_grid" / "lbf_epymarl_baselines.ipynb"
    )
    source = notebook_path.read_text()

    assert 'Runs IQL, IPPO, MAPPO, QMIX, and MAA2C' not in source
    assert 'for kind in (\\"best\\", \\"final\\"):' in source
    assert 'for kind in (\\"final\\", \\"best\\"):' not in source
