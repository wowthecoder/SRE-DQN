"""Generate report reward curves for torch MF-DSRQ epsilon sweeps.

The default paths cover the v4 same-team and opponent mean-field-source runs:

    source venv/bin/activate
    python discrete_action_space/mean_field_dsrq/plot_torch_epsilon_training_rewards.py

Figures are written to ``discrete_action_space/mean_field_dsrq/report_graphs``.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sre-dqn")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "report_graphs"
DEFAULT_ENV_NAME = "battle_v4"
DEFAULT_SEED_DIR = "seed42"
DEFAULT_SMOOTHING_WINDOW = 100


@dataclass(frozen=True)
class ScheduleSpec:
    label: str
    root_name: str
    schedule_dir: str

    def stats_path(self, runs_dir: Path, env_name: str, seed_dir: str) -> Path:
        return (
            runs_dir
            / self.root_name
            / self.schedule_dir
            / env_name
            / seed_dir
            / "training_stats.json"
        )


@dataclass(frozen=True)
class VariantSpec:
    label: str
    file_stem: str
    schedules: tuple[ScheduleSpec, ...]


def _variant_specs() -> tuple[VariantSpec, ...]:
    fixed_same_team = "mf_srq_torch_epsilon_training_v4_same_team"
    decay_same_team = "mf_srq_torch_epsilon_decay_to_zero_v4_same_team"
    fixed_opponent = "mf_srq_torch_epsilon_training_v4_opp"
    decay_opponent = "mf_srq_torch_epsilon_decay_to_zero_v4_opp"
    schedule_pairs = (
        ("fixed_0.01", "eps_0_01", fixed_same_team, fixed_opponent),
        ("fixed_0.1", "eps_0_1", fixed_same_team, fixed_opponent),
        ("fixed_0.5", "eps_0_5", fixed_same_team, fixed_opponent),
        ("decay_0.5", "start_0_5_to_0", decay_same_team, decay_opponent),
        ("decay_0.75", "start_0_75_to_0", decay_same_team, decay_opponent),
        ("decay_1.0", "start_1_to_0", decay_same_team, decay_opponent),
    )
    return (
        VariantSpec(
            label='mean_field_source="same_team"',
            file_stem="same_team",
            schedules=tuple(
                ScheduleSpec(label, same_team_root, schedule_dir)
                for label, schedule_dir, same_team_root, _ in schedule_pairs
            ),
        ),
        VariantSpec(
            label='mean_field_source="opponent"',
            file_stem="opponent",
            schedules=tuple(
                ScheduleSpec(label, opponent_root, schedule_dir)
                for label, schedule_dir, _, opponent_root in schedule_pairs
            ),
        ),
    )


def _load_training_stats(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _episode_records(stats: dict) -> list[dict]:
    records = stats.get("episode_records") or stats.get("records") or []
    if not records:
        raise ValueError("No episode records found.")
    return records


def _reward_series(stats: dict, role: str) -> tuple[np.ndarray, np.ndarray]:
    records = _episode_records(stats)
    x = np.arange(1, len(records) + 1, dtype=np.int64)
    y = np.asarray(
        [record.get("rewards", {}).get(role, np.nan) for record in records],
        dtype=np.float64,
    )
    return x, y


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    values = np.asarray(values, dtype=np.float64)
    if window == 1 or values.size == 0:
        return values.copy()

    valid = np.isfinite(values)
    safe_values = np.where(valid, values, 0.0)
    counts = np.convolve(valid.astype(np.float64), np.ones(window), mode="full")[: values.size]
    totals = np.convolve(safe_values, np.ones(window), mode="full")[: values.size]
    out = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(totals, counts, out=out, where=counts > 0)
    return out


def _load_variant_runs(
    variant: VariantSpec,
    *,
    runs_dir: Path,
    env_name: str,
    seed_dir: str,
) -> list[tuple[ScheduleSpec, Path, dict]]:
    loaded = []
    for schedule in variant.schedules:
        stats_path = schedule.stats_path(runs_dir, env_name, seed_dir)
        if not stats_path.exists():
            warnings.warn(f"Skipping missing stats: {stats_path}", stacklevel=2)
            continue
        try:
            loaded.append((schedule, stats_path, _load_training_stats(stats_path)))
        except Exception as exc:
            warnings.warn(f"Skipping unreadable stats {stats_path}: {exc}", stacklevel=2)
    return loaded


def _plot_role_overlay(
    ax,
    runs: Iterable[tuple[ScheduleSpec, Path, dict]],
    *,
    role: str,
    smoothing_window: int,
) -> None:
    for schedule, _, stats in runs:
        x, rewards = _reward_series(stats, role)
        ax.plot(x, _rolling_mean(rewards, smoothing_window), linewidth=1.6, label=schedule.label)
    ax.set_title(f"{role.capitalize()} model rewards")
    ax.set_xlabel("Completed episode")
    ax.set_ylabel(f"Reward, rolling mean {smoothing_window}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)


def save_variant_overlay(
    variant: VariantSpec,
    runs: list[tuple[ScheduleSpec, Path, dict]],
    *,
    output_dir: Path,
    smoothing_window: int,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=False, sharey=False)
    for ax, role in zip(axes, ("main", "opponent"), strict=True):
        _plot_role_overlay(ax, runs, role=role, smoothing_window=smoothing_window)
    fig.suptitle(f"Torch MF-DSRQ reward curves: {variant.label}", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / f"{variant.file_stem}_training_reward_curves.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _runs_by_label(runs: list[tuple[ScheduleSpec, Path, dict]]) -> dict[str, dict]:
    return {schedule.label: stats for schedule, _, stats in runs}


def save_combined_schedule_grid(
    variants_and_runs: list[tuple[VariantSpec, list[tuple[ScheduleSpec, Path, dict]]]],
    *,
    output_dir: Path,
    smoothing_window: int,
    dpi: int,
) -> Path:
    if not variants_and_runs:
        raise ValueError("At least one variant is required for the combined schedule grid.")

    schedules = variants_and_runs[0][0].schedules
    run_maps = {
        variant.file_stem: _runs_by_label(runs)
        for variant, runs in variants_and_runs
    }

    fig, axes = plt.subplots(
        len(schedules),
        2,
        figsize=(14, 20),
        sharex=False,
        sharey=False,
    )
    for row_idx, schedule in enumerate(schedules):
        for col_idx, role in enumerate(("main", "opponent")):
            ax = axes[row_idx, col_idx]
            for variant, _ in variants_and_runs:
                stats = run_maps[variant.file_stem].get(schedule.label)
                if stats is None:
                    continue
                x, rewards = _reward_series(stats, role)
                ax.plot(
                    x,
                    _rolling_mean(rewards, smoothing_window),
                    linewidth=1.4,
                    label=variant.file_stem,
                )
            if not ax.lines:
                ax.text(
                    0.5,
                    0.5,
                    "missing training_stats.json",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_title(f"{schedule.label} - {role}")
            ax.set_xlabel("Completed episode")
            ax.set_ylabel(f"Reward, rolling mean {smoothing_window}")
            ax.grid(alpha=0.25)
            if ax.lines:
                ax.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle("Torch MF-DSRQ reward grid: same_team vs opponent mean-field source", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "same_team_vs_opponent_training_reward_grid_6x2.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save torch MF-DSRQ epsilon-sweep reward figures from training_stats.json files."
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--seed-dir", default=DEFAULT_SEED_DIR)
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    variants_and_runs = []
    for variant in _variant_specs():
        runs = _load_variant_runs(
            variant,
            runs_dir=args.runs_dir,
            env_name=args.env_name,
            seed_dir=args.seed_dir,
        )
        if not runs:
            warnings.warn(f"No stats found for {variant.label}; skipping figures.", stacklevel=2)
            continue
        variants_and_runs.append((variant, runs))
        saved_paths.append(
            save_variant_overlay(
                variant,
                runs,
                output_dir=output_dir,
                smoothing_window=args.smoothing_window,
                dpi=args.dpi,
            )
        )

    if variants_and_runs:
        saved_paths.append(
            save_combined_schedule_grid(
                variants_and_runs,
                output_dir=output_dir,
                smoothing_window=args.smoothing_window,
                dpi=args.dpi,
            )
        )

    for path in saved_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
