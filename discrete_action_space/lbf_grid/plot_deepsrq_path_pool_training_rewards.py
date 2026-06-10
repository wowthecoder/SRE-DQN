"""Plot Deep SRQ PATH-pool LBF training reward comparisons."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TRAINING_ROOT = (
    Path(__file__).resolve().parent / "deepsrq_path_lcp_pool" / "training"
)
DEFAULT_OUTPUT_DIR = DEFAULT_TRAINING_ROOT / "training_reward_summary_plots"
DEFAULT_BASELINE_ROOT = Path(__file__).resolve().parent / "baseline_runs" / "epymarl"
DEFAULT_RUN_KEYS = ("0.01", "0.1", "0.5", "decay_0.5_to_0", "decay_1_to_0")
SCENARIO_FILENAME_SLUGS = {
    "lbf_8x8_2p_2f_levels12": "levels12",
    "lbf_8x8_2p_2f_force_coop": "force_coop",
}
BASELINE_ALGORITHM_ORDER = ("ippo", "iql", "maa2c", "mappo")
METRIC_NAMES = ("agent_1", "agent_2", "joint")
METRIC_TITLES = {
    "agent_1": "Agent 1 Training Reward",
    "agent_2": "Agent 2 Training Reward",
    "joint": "Joint Training Reward",
}


@dataclass(frozen=True)
class RewardRun:
    scenario_key: str
    run_key: str
    label: str
    path: Path
    rewards: np.ndarray

    @property
    def n_episodes(self) -> int:
        return int(self.rewards.shape[1])

    @property
    def series(self) -> dict[str, np.ndarray]:
        return {
            "agent_1": self.rewards[0],
            "agent_2": self.rewards[1],
            "joint": self.rewards[0] + self.rewards[1],
        }


@dataclass(frozen=True)
class BaselineRun:
    algorithm: str
    path: Path
    episodes: np.ndarray
    curves: dict[str, np.ndarray]

    @property
    def series(self) -> dict[str, np.ndarray]:
        series: dict[str, np.ndarray] = {}
        if "agent_0" in self.curves:
            series["agent_1"] = self.curves["agent_0"]
        if "agent_1" in self.curves:
            series["agent_2"] = self.curves["agent_1"]

        for key in ("joint", "total", "shared"):
            if key in self.curves:
                series["joint"] = self.curves[key]
                break
        else:
            agent_curves = [
                self.curves[key]
                for key in sorted(self.curves)
                if key.startswith("agent_")
            ]
            if agent_curves:
                length = min(curve.size for curve in agent_curves)
                series["joint"] = np.stack(
                    [curve[:length] for curve in agent_curves],
                    axis=0,
                ).sum(axis=0)

        return series


def _run_sort_key(run_key: str) -> tuple[int, float, str]:
    try:
        return (0, float(run_key), run_key)
    except ValueError:
        match = re.search(r"decay_([0-9.]+)_to_0", run_key)
        if match:
            return (1, float(match.group(1)), run_key)
        return (2, float("inf"), run_key)


def _run_label(run_key: str) -> str:
    try:
        return f"eps={float(run_key):g}"
    except ValueError:
        match = re.fullmatch(r"decay_([0-9.]+)_to_0", run_key)
        if match:
            return f"eps {float(match.group(1)):g} -> 0"
        return run_key.replace("_", " ")


def _number_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _run_filename_slug(run_key: str) -> str:
    try:
        return f"constant_eps_{_number_slug(float(run_key))}"
    except ValueError:
        match = re.fullmatch(r"decay_([0-9.]+)_to_0", run_key)
        if match:
            return f"decay_eps_{_number_slug(float(match.group(1)))}_to_0"
        return _safe_slug(run_key)


def _scenario_filename_slug(scenario_key: str) -> str:
    return SCENARIO_FILENAME_SLUGS.get(str(scenario_key), _safe_slug(scenario_key))


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = max(1, int(window))
    if window <= 1 or values.size == 0:
        return values.copy()
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    means = np.empty_like(values, dtype=np.float64)
    for idx in range(values.size):
        start = max(0, idx + 1 - window)
        means[idx] = (cumsum[idx + 1] - cumsum[start]) / float(idx + 1 - start)
    return means


def _load_reward_run(path: Path) -> RewardRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rewards = payload.get("rewards")
    if rewards is not None:
        arr = np.asarray(rewards, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 2:
            per_agent = arr[:2]
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            per_agent = arr[:, :2].T
        else:
            raise ValueError(f"Unsupported rewards shape in {path}: {arr.shape}")
    else:
        episode_rewards = payload.get("episode_rewards")
        arr = np.asarray(episode_rewards, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Unsupported episode_rewards shape in {path}: {arr.shape}")
        per_agent = arr[:, :2].T

    run_dir = path.parent
    scenario_key = run_dir.parent.name
    run_key = run_dir.name
    return RewardRun(
        scenario_key=scenario_key,
        run_key=run_key,
        label=_run_label(run_key),
        path=path,
        rewards=per_agent,
    )


def _load_baseline_run(path: Path) -> BaselineRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    curves_payload = payload.get("reward_curve") or {}
    episodes = np.asarray(curves_payload.get("episodes") or [], dtype=np.float64)
    curves = {
        str(key): np.asarray(values, dtype=np.float64)
        for key, values in curves_payload.items()
        if key != "episodes" and values
    }
    if episodes.size == 0 and curves:
        first = next(iter(curves.values()))
        episodes = np.arange(1, first.size + 1, dtype=np.float64)

    return BaselineRun(
        algorithm=str(payload.get("algorithm") or path.parent.name).lower(),
        path=path,
        episodes=episodes,
        curves=curves,
    )


def discover_runs(training_root: Path) -> dict[str, dict[str, RewardRun]]:
    runs: dict[str, dict[str, RewardRun]] = {}
    for rewards_path in sorted(training_root.glob("*/*/training_rewards.json")):
        run = _load_reward_run(rewards_path)
        runs.setdefault(run.scenario_key, {})[run.run_key] = run
    return runs


def discover_baselines(baseline_root: Path) -> dict[str, dict[str, BaselineRun]]:
    baselines: dict[str, dict[str, BaselineRun]] = {}
    for stats_path in sorted(baseline_root.glob("*/*/reward_stats.json")):
        scenario_key = stats_path.parent.parent.name
        baseline = _load_baseline_run(stats_path)
        baselines.setdefault(scenario_key, {})[baseline.algorithm] = baseline
    return baselines


def _ordered_run_keys(runs_by_scenario: dict[str, dict[str, RewardRun]]) -> list[str]:
    keys = set()
    for scenario_runs in runs_by_scenario.values():
        keys.update(scenario_runs)
    return sorted(keys, key=_run_sort_key)


def _default_run_keys(runs_by_scenario: dict[str, dict[str, RewardRun]]) -> list[str]:
    available = set(_ordered_run_keys(runs_by_scenario))
    selected = [run_key for run_key in DEFAULT_RUN_KEYS if run_key in available]
    if selected:
        return selected
    return _ordered_run_keys(runs_by_scenario)


def _ordered_baseline_algorithms(scenario_baselines: dict[str, BaselineRun]) -> list[str]:
    known = [
        algorithm
        for algorithm in BASELINE_ALGORITHM_ORDER
        if algorithm in scenario_baselines
    ]
    extras = sorted(
        algorithm
        for algorithm in scenario_baselines
        if algorithm not in BASELINE_ALGORITHM_ORDER
    )
    return known + extras


def plot_scenario_comparison(
    *,
    scenario_key: str,
    scenario_runs: dict[str, RewardRun],
    run_keys: list[str],
    output_dir: Path,
    smoothing_window: int,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharex=True)
    colors = plt.get_cmap("tab10")

    for metric_idx, metric in enumerate(METRIC_NAMES):
        ax = axes[metric_idx]
        for line_idx, run_key in enumerate(run_keys):
            run = scenario_runs.get(run_key)
            if run is None:
                continue
            values = _rolling_mean(run.series[metric], smoothing_window)
            episodes = np.arange(1, values.size + 1)
            ax.plot(
                episodes,
                values,
                linewidth=1.8,
                color=colors(line_idx % 10),
                label=run.label,
            )
        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(alpha=0.25)
        if metric_idx == 2:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"{scenario_key} - Deep SRQ PATH-pool training rewards "
        f"(rolling mean window={int(smoothing_window)})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path = (
        output_dir
        / f"{_scenario_filename_slug(scenario_key)}__schedule_comparison.png"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_schedule_with_baselines(
    *,
    scenario_key: str,
    run_key: str,
    run: RewardRun,
    scenario_baselines: dict[str, BaselineRun],
    output_dir: Path,
    smoothing_window: int,
) -> Path | None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharex=True)
    baseline_colors = plt.get_cmap("tab10")
    baseline_algorithms = _ordered_baseline_algorithms(scenario_baselines)

    for metric in METRIC_NAMES:
        ax = axes[METRIC_NAMES.index(metric)]
        deep_values = _rolling_mean(run.series[metric], smoothing_window)
        deep_episodes = np.arange(1, deep_values.size + 1)
        ax.plot(
            deep_episodes,
            deep_values,
            color="#111111",
            linewidth=2.2,
            label=f"Deep SRQ {_run_label(run_key)}",
        )
        for idx, algorithm in enumerate(baseline_algorithms):
            baseline = scenario_baselines[algorithm]
            values = baseline.series.get(metric)
            if values is None or values.size == 0 or baseline.episodes.size == 0:
                continue
            length = min(values.size, baseline.episodes.size)
            ax.plot(
                baseline.episodes[:length],
                values[:length],
                color=baseline_colors(idx % 10),
                linewidth=1.7,
                marker="o",
                markersize=3,
                label=algorithm.upper(),
            )

        ax.set_title(METRIC_TITLES[metric])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(alpha=0.25)
        if metric == "joint":
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"{scenario_key} - {_run_label(run_key)} vs EPyMARL baselines "
        f"(Deep SRQ rolling window={int(smoothing_window)})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path = (
        output_dir
        / f"{_scenario_filename_slug(scenario_key)}__{_run_filename_slug(run_key)}__vs_baselines.png"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_plots(
    *,
    training_root: Path,
    baseline_root: Path,
    output_dir: Path,
    smoothing_window: int,
    run_keys_filter: set[str] | None = None,
) -> list[Path]:
    runs_by_scenario = discover_runs(training_root)
    if not runs_by_scenario:
        raise FileNotFoundError(
            f"No training_rewards.json files found under {training_root}"
        )

    baselines_by_scenario = discover_baselines(baseline_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_keys = _default_run_keys(runs_by_scenario)
    if run_keys_filter is not None:
        all_run_keys = _ordered_run_keys(runs_by_scenario)
        run_keys = [run_key for run_key in all_run_keys if run_key in run_keys_filter]
        if not run_keys:
            raise ValueError(
                "None of the requested run keys were found. "
                f"Requested: {sorted(run_keys_filter)}"
            )

    saved_paths: list[Path] = []
    for scenario_key in sorted(runs_by_scenario):
        saved_paths.append(
            plot_scenario_comparison(
                scenario_key=scenario_key,
                scenario_runs=runs_by_scenario[scenario_key],
                run_keys=run_keys,
                output_dir=output_dir,
                smoothing_window=smoothing_window,
            )
        )

        scenario_baselines = baselines_by_scenario.get(scenario_key, {})
        for run_key in run_keys:
            run = runs_by_scenario[scenario_key].get(run_key)
            if run is None:
                continue
            output_path = plot_schedule_with_baselines(
                scenario_key=scenario_key,
                run_key=run_key,
                run=run,
                scenario_baselines=scenario_baselines,
                output_dir=output_dir,
                smoothing_window=smoothing_window,
            )
            if output_path is not None:
                saved_paths.append(output_path)
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Deep SRQ PATH-pool LBF training rewards from training_rewards.json."
        )
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
        help=f"Training artifact root. Default: {DEFAULT_TRAINING_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved figures. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help=f"EPyMARL baseline artifact root. Default: {DEFAULT_BASELINE_ROOT}",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=100,
        help="Rolling mean window for plotted curves. Default: 100 episodes.",
    )
    parser.add_argument(
        "--run-key",
        action="append",
        default=None,
        help=(
            "Optional schedule/run directory name to include, such as 0.5 or "
            "decay_0.5_to_0. Repeat this flag to plot a subset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = write_plots(
        training_root=args.training_root,
        baseline_root=args.baseline_root,
        output_dir=args.output_dir,
        smoothing_window=args.smoothing_window,
        run_keys_filter=set(args.run_key) if args.run_key else None,
    )
    print(f"Saved {len(saved_paths)} plot(s):")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
