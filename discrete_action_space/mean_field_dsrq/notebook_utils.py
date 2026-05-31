"""Notebook-facing training and evaluation helpers for mean-field DSRQ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .benchmarl_magent2 import (
    DEFAULT_TASK_CONFIG,
    ALGORITHM_NAMES,
    run_benchmarl_algorithm,
    sample_benchmarl_rollout_video,
)
from .eval_mf_dsrq import evaluate, evaluate_mfdsrq_vs_benchmarl
from .train_mf_dsrq import train


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "battle_v4.yaml"
)
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _epsilon_slug(epsilon: float) -> str:
    text = f"{float(epsilon):g}".replace(".", "_").replace("-", "neg_")
    return f"eps_{text}"


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
        "buffer_capacity": 80_000,
        "batch_size": 64,
        "use_gpu": True,
    }
    if extra_overrides:
        overrides.update(extra_overrides)
    return load_mfdsrq_config(overrides=overrides)


def train_mfdsrq_from_notebook(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run MF-DSRQ training from a notebook cell."""
    return train(dict(cfg))


def mfdsrq_epsilon_config(
    base_cfg: dict[str, Any],
    epsilon: float,
    output_root: str | Path,
) -> dict[str, Any]:
    """Return a fixed-robust-epsilon config with an epsilon-specific output dir."""
    cfg = dict(base_cfg)
    cfg["epsilon_robust_start"] = float(epsilon)
    cfg["epsilon_robust_end"] = float(epsilon)
    cfg["epsilon_robust_decay_frac"] = 1.0
    cfg["output_dir"] = str(Path(output_root) / _epsilon_slug(epsilon))
    return cfg


def _load_training_stats(result_or_stats_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(result_or_stats_path, dict):
        if "episode_records" in result_or_stats_path and "summary" in result_or_stats_path:
            return result_or_stats_path
        stats_path = result_or_stats_path.get("stats_path")
        if not stats_path:
            raise ValueError("Training result dict does not contain stats_path.")
    else:
        stats_path = result_or_stats_path
    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


def plot_mfdsrq_training_curves(
    result_or_stats_path: dict[str, Any] | str | Path,
    *,
    smoothing_window: int = 20,
    save: bool = True,
):
    """Plot team reward curves and cumulative kill-rule wins from training stats."""
    import matplotlib.pyplot as plt
    import pandas as pd

    stats = _load_training_stats(result_or_stats_path)
    records = stats.get("episode_records", [])
    if not records:
        raise ValueError("No completed episode records found in training stats.")

    type_names = stats.get("type_names") or list(records[0]["rewards"].keys())
    rows = []
    for record in records:
        row = {
            "episode": record["episode"],
            "global_step": record["global_step"],
            "env_idx": record["env_idx"],
        }
        for type_name in type_names:
            row[f"reward_{type_name}"] = record["rewards"].get(type_name, 0.0)
            row[f"win_{type_name}"] = record["wins"].get(type_name, 0)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["episode", "env_idx"]).reset_index(drop=True)
    df["episode_index"] = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    window = max(int(smoothing_window), 1)
    for type_name in type_names:
        reward_col = f"reward_{type_name}"
        series = df[reward_col].rolling(window, min_periods=1).mean()
        axes[0].plot(df["episode_index"], series, label=type_name)
        axes[1].plot(df["episode_index"], df[f"win_{type_name}"].cumsum(), label=type_name)

    axes[0].set_title(f"Training Rewards, rolling mean {window}")
    axes[0].set_xlabel("Completed episode")
    axes[0].set_ylabel("Team reward")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Cumulative Team Wins")
    axes[1].set_xlabel("Completed episode")
    axes[1].set_ylabel("Wins")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    if save:
        run_dir = Path(stats["run_dir"])
        out_path = run_dir / "training_reward_and_wins.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {out_path}")
    return fig


def print_mfdsrq_training_win_rates(result_or_stats_path: dict[str, Any] | str | Path) -> dict[str, float]:
    """Print and return overall kill-rule team win rates from training stats."""
    stats = _load_training_stats(result_or_stats_path)
    summary = stats.get("summary", {})
    win_rates = summary.get("win_rates", {})
    episodes = int(summary.get("episodes", stats.get("completed_episodes", 0)))
    tie_rate = float(summary.get("tie_rate", 0.0))
    print(f"Completed episodes: {episodes}")
    for team, rate in win_rates.items():
        print(f"{team} win rate: {rate:.3f}")
    print(f"Tie rate: {tie_rate:.3f}")
    return win_rates


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


def find_latest_benchmarl_run(
    algorithm: str,
    baseline_root: str | Path = RUNS_DIR / "benchmarl_magent2_notebooks",
    *,
    task_name: str | None = None,
) -> Path:
    """Return the newest BenchMARL run folder for an algorithm with a checkpoint."""
    baseline_root = Path(baseline_root)
    candidates = []
    for run_dir in baseline_root.glob(f"{algorithm.lower()}_*"):
        if task_name and task_name.lower() not in run_dir.name.lower():
            continue
        checkpoint_dir = run_dir / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            candidates.append((max(p.stat().st_mtime for p in checkpoints), run_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No BenchMARL checkpoints found for {algorithm!r} under {baseline_root}"
        )
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def evaluate_mfdsrq_against_baselines(
    cfg: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    baseline_root: str | Path = RUNS_DIR / "benchmarl_magent2_notebooks",
    algorithms: tuple[str, ...] = ("mappo", "ippo", "iql"),
    baseline_folders: dict[str, str | Path] | None = None,
    num_episodes: int = 20,
    max_steps: int | None = None,
    evaluate_both_sides: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Evaluate MF-DSRQ head-to-head against BenchMARL baseline checkpoints."""
    rows = []
    results = {}
    baseline_folders = baseline_folders or {}
    task_name = str(cfg.get("env_name", "")).split("_v", 1)[0]
    for algorithm in algorithms:
        if algorithm in baseline_folders:
            baseline_folder = Path(baseline_folders[algorithm])
        else:
            baseline_folder = find_latest_benchmarl_run(
                algorithm,
                baseline_root,
                task_name=task_name or None,
            )
        result = evaluate_mfdsrq_vs_benchmarl(
            dict(cfg),
            checkpoint_dir,
            baseline_folder,
            baseline_name=algorithm,
            num_episodes=int(num_episodes),
            max_steps=max_steps,
            evaluate_both_sides=evaluate_both_sides,
        )
        summary = result["summary"]
        rows.append(
            {
                "baseline": algorithm,
                "mfdsrq_win_rate": summary["mfdsrq_win_rate"],
                "baseline_win_rate": summary["baseline_win_rate"],
                "tie_rate": summary["tie_rate"],
                "mean_mfdsrq_kills": summary["mean_mfdsrq_kills"],
                "mean_baseline_kills": summary["mean_baseline_kills"],
                "mean_mfdsrq_reward": summary["mean_mfdsrq_reward"],
                "mean_baseline_reward": summary["mean_baseline_reward"],
                "episodes": summary["episodes"],
                "baseline_folder": str(baseline_folder),
            }
        )
        results[algorithm] = result

    payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "algorithms": list(algorithms),
        "rows": rows,
        "results": results,
    }
    if save:
        out_path = Path(checkpoint_dir) / "head_to_head_vs_benchmarl.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        payload["results_path"] = str(out_path)
        print(f"Saved head-to-head comparison to {out_path}")
    return payload


def plot_mfdsrq_baseline_win_rates(comparison: dict[str, Any], *, save: bool = True):
    """Plot grouped MF-DSRQ/baseline/tie win rates from a comparison payload."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(comparison["rows"])
    if df.empty:
        raise ValueError("No comparison rows to plot.")
    x = range(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width for i in x], df["mfdsrq_win_rate"], width, label="MF-DSRQ")
    ax.bar(list(x), df["baseline_win_rate"], width, label="Baseline")
    ax.bar([i + width for i in x], df["tie_rate"], width, label="Tie")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["baseline"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Head-to-Head Kill-Rule Win Rates")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if save:
        out_path = Path(comparison["checkpoint_dir"]) / "head_to_head_win_rates.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved win-rate plot to {out_path}")
    return fig


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
