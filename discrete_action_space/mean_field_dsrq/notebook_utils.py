"""Notebook-facing training and evaluation helpers for mean-field DSRQ."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .benchmarl_magent2 import (
    DEFAULT_TASK_CONFIG,
    ALGORITHM_NAMES,
    run_benchmarl_algorithm,
    sample_benchmarl_rollout_video,
)
from .eval_mf_dsrq import evaluate
from .train_mf_dsrq import train


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "battle_v4.yaml"
)
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def load_mfdsrq_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a MF-DSRQ YAML config and apply notebook overrides."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg.update(overrides)
    cfg.setdefault("env_backend", "magent2")
    return cfg


def notebook_mfdsrq_config(
    *,
    total_steps: int = 20_000,
    num_envs: int = 2,
    map_size: int = DEFAULT_TASK_CONFIG["map_size"],
    max_cycles: int = DEFAULT_TASK_CONFIG["max_cycles"],
    output_dir: str | Path = RUNS_DIR / "mean_field_dsrq_notebooks",
    seed: int = 42,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Small MAgent2 battle config for interactive notebook runs."""
    overrides: dict[str, Any] = {
        "env_name": "battle_v4",
        "env_backend": "magent2",
        "map_size": int(map_size),
        "max_cycles": int(max_cycles),
        "total_steps": int(total_steps),
        "num_envs": int(num_envs),
        "seed": int(seed),
        "output_dir": str(output_dir),
        "log_interval": max(int(num_envs), min(1_000, int(total_steps))),
        "save_interval": max(int(num_envs), int(total_steps)),
        "learning_starts": min(1_000, max(128, int(total_steps) // 10)),
        "buffer_capacity": max(10_000, int(total_steps) * 10),
        "batch_size": 64,
        "use_gpu": True,
    }
    if extra_overrides:
        overrides.update(extra_overrides)
    return load_mfdsrq_config(overrides=overrides)


def train_mfdsrq_from_notebook(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run MF-DSRQ training from a notebook cell."""
    return train(dict(cfg))


def evaluate_mfdsrq_from_notebook(
    cfg: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    num_episodes: int = 5,
    obs_noise_sigmas: list[float] | None = None,
) -> dict[str, Any]:
    """Run greedy MF-DSRQ evaluation and robustness/noise sweeps."""
    return evaluate(
        dict(cfg),
        str(checkpoint_dir),
        num_episodes=int(num_episodes),
        obs_noise_sigmas=obs_noise_sigmas or [0.0],
    )


def run_baseline_from_notebook(
    algorithm_name: str,
    *,
    task_config: dict[str, Any] | None = None,
    seed: int = 0,
    total_frames: int = 20_000,
    frames_per_batch: int = 1_000,
    n_envs_per_worker: int = 1,
    save_folder: str | Path = RUNS_DIR / "benchmarl_magent2_notebooks",
    sampling_device: str = "cpu",
    train_device: str = "cpu",
    buffer_device: str = "cpu",
    parallel_collection: bool = False,
) -> dict[str, Any]:
    """Run one BenchMARL baseline from a notebook cell."""
    if algorithm_name.lower() not in ALGORITHM_NAMES:
        raise ValueError(f"Unknown baseline {algorithm_name!r}; choose from {ALGORITHM_NAMES}.")
    return run_benchmarl_algorithm(
        algorithm_name,
        task_config=task_config,
        seed=seed,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
        n_envs_per_worker=n_envs_per_worker,
        save_folder=save_folder,
        sampling_device=sampling_device,
        train_device=train_device,
        buffer_device=buffer_device,
        parallel_collection=parallel_collection,
    )


def baseline_rollout_video_from_notebook(
    result_or_folder: dict[str, Any] | str | Path,
    *,
    max_steps: int = 50,
    fps: int = 8,
    deterministic: bool = True,
    title: str | None = None,
):
    """Reload a trained BenchMARL baseline and display one evaluation rollout."""
    return sample_benchmarl_rollout_video(
        result_or_folder,
        max_steps=max_steps,
        fps=fps,
        deterministic=deterministic,
        title=title,
    )
