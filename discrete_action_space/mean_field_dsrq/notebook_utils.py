"""Notebook-facing training and evaluation helpers for mean-field DSRQ."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from queue import Empty
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is optional in non-notebook contexts
    _tqdm = None

from .magent2_env import (
    DEFAULT_TASK_CONFIG,
)
from .mfrl_baselines import (
    BASELINE_ALGORITHMS,
    DEFAULT_BASELINE_RUNS_DIR,
    DEFAULT_MFRL_TASK_CONFIG,
    find_latest_mfrl_baseline_run,
    sample_mfrl_rollout_video,
    train_mfrl_baseline,
)
from .eval_mf_dsrq import (
    _evaluate_fixed_side_tournament_matchup,
    _evaluate_mfdsrq_vs_mfrl_assignment,
    _resolve_torch_device,
    _summarize_fixed_side_records,
    _summarize_matchup_records,
    evaluate,
    evaluate_mfdsrq_vs_mfrl_baseline,
)
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
    target_episodes: int = 50,
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
        "target_episodes": int(target_episodes),
        "num_envs": int(num_envs),
        "seed": int(seed),
        "output_dir": str(output_dir),
        "reward_log_interval": max(int(num_envs), min(100, int(target_episodes))),
        "save_every": 400,
        "learning_starts": min(5_000, max(128, int(target_episodes) * int(max_cycles) // 10)),
        "buffer_capacity": 80_000,
        "batch_size": 64,
        "self_play_tau": 0.01,
        "max_train_batches_per_update": None,
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
        if "episode_records" in result_or_stats_path or "records" in result_or_stats_path:
            return result_or_stats_path
        stats_path = result_or_stats_path.get("stats_path")
        if not stats_path:
            raise ValueError("Training result dict does not contain stats_path.")
    else:
        stats_path = Path(result_or_stats_path)
        if stats_path.is_dir():
            stats_path = stats_path / "training_stats.json"
    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


def _strict_kill_wins(record: dict[str, Any], type_names: list[str]) -> tuple[dict[str, int], int]:
    if "winner" in record:
        winner = record.get("winner")
        return {t: int(winner == t) for t in type_names}, int(winner == "tie")
    kills = {t: int(record.get("kills", {}).get(t, 0)) for t in type_names}
    if not kills:
        return {t: 0 for t in type_names}, 1
    max_kill = max(kills.values())
    winner_count = sum(int(kills[t] == max_kill) for t in type_names)
    wins = {
        t: int(winner_count == 1 and kills[t] == max_kill)
        for t in type_names
    }
    return wins, int(winner_count != 1)


def _strict_training_summary(stats: dict[str, Any], type_names: list[str]) -> dict[str, Any]:
    records = stats.get("episode_records") or stats.get("records", [])
    n = len(records)
    win_counts = {t: 0 for t in type_names}
    tie_count = 0
    for record in records:
        wins, tie = _strict_kill_wins(record, type_names)
        tie_count += tie
        for type_name in type_names:
            win_counts[type_name] += wins.get(type_name, 0)
    return {
        "episodes": n,
        "win_counts": win_counts,
        "win_rates": {t: (win_counts[t] / n if n else 0.0) for t in type_names},
        "tie_count": tie_count,
        "tie_rate": (tie_count / n if n else 0.0),
    }


def plot_mfdsrq_training_curves(
    result_or_stats_path: dict[str, Any] | str | Path,
    *,
    smoothing_window: int = 20,
    save: bool = True,
):
    """Plot team rewards, strict kill-rule wins, and kills from training stats."""
    import matplotlib.pyplot as plt
    import pandas as pd

    stats = _load_training_stats(result_or_stats_path)
    records = stats.get("episode_records") or stats.get("records", [])
    if not records:
        raise ValueError("No completed episode records found in training stats.")

    type_names = stats.get("type_names") or list(records[0]["rewards"].keys())
    rows = []
    for record in records:
        row = {
            "episode": record["episode"],
            "global_step": record.get("global_step", record.get("env_steps", record["episode"])),
            "env_idx": record["env_idx"],
        }
        for type_name in type_names:
            strict_wins, _ = _strict_kill_wins(record, type_names)
            row[f"reward_{type_name}"] = record["rewards"].get(type_name, 0.0)
            row[f"win_{type_name}"] = strict_wins.get(type_name, 0)
            row[f"kills_{type_name}"] = record.get("kills", {}).get(type_name, 0)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["episode", "env_idx"]).reset_index(drop=True)
    df["episode_index"] = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    window = max(int(smoothing_window), 1)
    for type_name in type_names:
        reward_col = f"reward_{type_name}"
        series = df[reward_col].rolling(window, min_periods=1).mean()
        axes[0].plot(df["episode_index"], series, label=type_name)
        axes[1].plot(df["episode_index"], df[f"win_{type_name}"].cumsum(), label=type_name)
        axes[2].plot(df["episode_index"], df[f"kills_{type_name}"], label=type_name)

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

    axes[2].set_title("Kills Per Episode")
    axes[2].set_xlabel("Completed episode")
    axes[2].set_ylabel("Kills")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    if save:
        run_dir = Path(stats["run_dir"])
        out_path = run_dir / "training_reward_wins_and_kills.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {out_path}")
    return fig


def plot_mfrl_baseline_training_curves(
    result_or_stats_path: dict[str, Any] | str | Path,
    *,
    smoothing_window: int = 20,
    save: bool = True,
):
    """Plot baseline team rewards, cumulative wins, and kills from training stats."""
    import matplotlib.pyplot as plt
    import pandas as pd

    stats = _load_training_stats(result_or_stats_path)
    records = stats.get("records", [])
    if not records:
        raise ValueError("No completed baseline records found in training stats.")

    type_names = ["main", "opponent"]
    rows = []
    for record in records:
        row = {
            "episode": record["episode"],
            "env_idx": record.get("env_idx", 0),
        }
        for type_name in type_names:
            row[f"reward_{type_name}"] = record["rewards"].get(type_name, 0.0)
            row[f"win_{type_name}"] = int(record.get("winner") == type_name)
            row[f"kills_{type_name}"] = record.get("kills", {}).get(type_name, 0)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["episode", "env_idx"]).reset_index(drop=True)
    df["episode_index"] = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    window = max(int(smoothing_window), 1)
    for type_name in type_names:
        reward_col = f"reward_{type_name}"
        series = df[reward_col].rolling(window, min_periods=1).mean()
        axes[0].plot(df["episode_index"], series, label=type_name)
        axes[1].plot(df["episode_index"], df[f"win_{type_name}"].cumsum(), label=type_name)
        axes[2].plot(df["episode_index"], df[f"kills_{type_name}"], label=type_name)

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

    axes[2].set_title("Kills Per Episode")
    axes[2].set_xlabel("Completed episode")
    axes[2].set_ylabel("Kills")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    if save:
        run_dir = Path(stats["run_dir"])
        out_path = run_dir / "training_reward_wins_and_kills.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {out_path}")
    return fig


def print_mfdsrq_training_win_rates(result_or_stats_path: dict[str, Any] | str | Path) -> dict[str, float]:
    """Print and return overall kill-rule team win rates from training stats."""
    stats = _load_training_stats(result_or_stats_path)
    records = stats.get("episode_records") or stats.get("records", [])
    type_names = stats.get("type_names") or (list(records[0]["rewards"].keys()) if records else [])
    summary = _strict_training_summary(stats, type_names) if records else stats.get("summary", {})
    win_rates = summary.get("win_rates")
    if win_rates is None:
        win_rates = {
            "main": float(summary.get("main_win_rate", 0.0)),
            "opponent": float(summary.get("opponent_win_rate", 0.0)),
        }
    episodes = int(summary.get("episodes", stats.get("completed_episodes", 0)))
    tie_rate = float(summary.get("tie_rate", 0.0))
    print(f"Completed episodes: {episodes}")
    for team, rate in win_rates.items():
        print(f"{team} win rate: {rate:.3f}")
    print(f"Tie rate: {tie_rate:.3f}")
    return win_rates


def evaluate_mfdsrq_from_notebook(
    cfg: dict[str, Any],
    checkpoint_dir: str | Path | Mapping[str, str | Path],
    *,
    num_episodes: int = 5,
    obs_noise_sigmas: list[float] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run greedy MF-DSRQ evaluation and robustness/noise sweeps."""
    return evaluate(
        dict(cfg),
        checkpoint_dir,
        num_episodes=int(num_episodes),
        obs_noise_sigmas=obs_noise_sigmas or [0.0],
        device=device,
    )


def find_latest_mfrl_run(
    algorithm: str,
    baseline_root: str | Path = DEFAULT_BASELINE_RUNS_DIR,
) -> Path:
    """Return the newest PyTorch MFRL baseline run folder for an algorithm."""
    return find_latest_mfrl_baseline_run(algorithm, baseline_root)


def _checkpoint_result_dir(checkpoint_source: str | Path | Mapping[str, str | Path]) -> Path:
    if not isinstance(checkpoint_source, Mapping):
        return Path(checkpoint_source)
    parents = [Path(path).parent for path in checkpoint_source.values()]
    if not parents:
        raise ValueError("checkpoint_source mapping must contain at least one checkpoint path.")
    return Path(os.path.commonpath([str(parent) for parent in parents]))


def _checkpoint_source_payload(checkpoint_source: str | Path | Mapping[str, str | Path]):
    if isinstance(checkpoint_source, Mapping):
        return {team: str(path) for team, path in checkpoint_source.items()}
    return str(checkpoint_source)


def _split_episode_chunks(num_episodes: int, num_chunks: int) -> list[tuple[int, int]]:
    num_episodes = int(num_episodes)
    num_chunks = max(int(num_chunks), 1)
    if num_episodes <= 0:
        return []
    num_chunks = min(num_chunks, num_episodes)
    base = num_episodes // num_chunks
    remainder = num_episodes % num_chunks
    chunks = []
    start = 0
    for chunk_idx in range(num_chunks):
        count = base + int(chunk_idx < remainder)
        chunks.append((start, count))
        start += count
    return chunks


def _make_eval_progress_bar(total_episodes: int, *, epsilon: float, enabled: bool = True):
    if not enabled or _tqdm is None or int(total_episodes) <= 0:
        return None
    return _tqdm(
        total=int(total_episodes),
        desc=f"MF-DSRQ eps={epsilon:g} eval episodes",
        unit="ep",
        dynamic_ncols=True,
        leave=True,
    )


class _LocalProgressQueue:
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def put(self, value: int):
        if self.progress_bar is not None:
            self.progress_bar.update(int(value))


def _drain_progress_queue(progress_queue, progress_bar) -> None:
    if progress_queue is None or progress_bar is None:
        return
    while True:
        try:
            progress_bar.update(int(progress_queue.get_nowait()))
        except Empty:
            break


def _evaluate_mfdsrq_assignment_worker(task: dict[str, Any]) -> dict[str, Any]:
    return _evaluate_mfdsrq_vs_mfrl_assignment(
        task["cfg"],
        task["checkpoint_paths"],
        task["baseline_folder"],
        baseline_name=task["baseline"],
        mfdsrq_team=task["mfdsrq_team"],
        baseline_team=task["baseline_team"],
        num_episodes=task["num_episodes"],
        max_steps=task.get("max_steps"),
        episode_offset=task.get("episode_offset", 0),
        progress_queue=task.get("progress_queue"),
        device=task.get("device"),
    )


def _evaluate_fixed_side_tournament_worker(task: dict[str, Any]) -> dict[str, Any]:
    return _evaluate_fixed_side_tournament_matchup(
        task["cfg"],
        task["checkpoint_paths"],
        task["baseline_folders"],
        main_algorithm=task["main_algorithm"],
        opponent_algorithm=task["opponent_algorithm"],
        num_episodes=task["num_episodes"],
        max_steps=task.get("max_steps"),
        episode_offset=task.get("episode_offset", 0),
        progress_queue=task.get("progress_queue"),
        device=task.get("device"),
    )


def _eval_multiprocessing_context(device) -> mp.context.BaseContext:
    if _resolve_torch_device(device).type == "cuda":
        return mp.get_context("spawn")
    return mp.get_context()


def _comparison_payload_from_worker_results(
    *,
    epsilon: float,
    checkpoint_paths: Mapping[str, str | Path],
    algorithms: tuple[str, ...],
    baseline_folders: dict[str, Path],
    worker_results: list[dict[str, Any]],
    num_episodes_per_side: int,
    evaluate_both_sides: bool,
    workers: int,
    device: str = "cpu",
) -> dict[str, Any]:
    rows = []
    results = {}
    result_dir = _checkpoint_result_dir(checkpoint_paths)
    for algorithm in algorithms:
        baseline_records = [
            record
            for result in worker_results
            if result["baseline"] == algorithm
            for record in result["records"]
        ]
        baseline_records.sort(key=lambda record: (record.get("assignment", ""), int(record.get("episode", 0))))
        assignment_summaries = {}
        for record in baseline_records:
            assignment_summaries.setdefault(record["assignment"], []).append(record)
        assignment_summaries = {
            assignment: _summarize_matchup_records(records)
            for assignment, records in assignment_summaries.items()
        }
        summary = _summarize_matchup_records(baseline_records)
        baseline_checkpoint = next(
            (
                result.get("baseline_checkpoint")
                for result in worker_results
                if result["baseline"] == algorithm and result.get("baseline_checkpoint") is not None
            ),
            None,
        )
        results[algorithm] = {
            "baseline": algorithm,
            "baseline_checkpoint": baseline_checkpoint,
            "checkpoint_dir": _checkpoint_source_payload(checkpoint_paths),
            "num_episodes_per_assignment": int(num_episodes_per_side),
            "records": baseline_records,
            "assignments": assignment_summaries,
            "summary": summary,
        }
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
                "baseline_folder": str(baseline_folders[algorithm]),
            }
        )
    return {
        "epsilon": float(epsilon),
        "checkpoint_dir": str(result_dir),
        "checkpoint_source": _checkpoint_source_payload(checkpoint_paths),
        "algorithms": list(algorithms),
        "num_episodes_per_assignment": int(num_episodes_per_side),
        "evaluate_both_sides": bool(evaluate_both_sides),
        "workers": int(workers),
        "device": str(device),
        "rows": rows,
        "results": results,
    }


def evaluate_mfdsrq_torch_epsilon_against_baselines(
    cfg: dict[str, Any],
    epsilon: float,
    checkpoint_paths: Mapping[str, str | Path],
    *,
    baseline_root: str | Path = DEFAULT_BASELINE_RUNS_DIR,
    algorithms: tuple[str, ...] = BASELINE_ALGORITHMS,
    baseline_folders: dict[str, str | Path] | None = None,
    num_episodes_per_side: int = 500,
    max_steps: int | None = None,
    evaluate_both_sides: bool = True,
    workers: int = 1,
    episode_chunk_size: int | None = None,
    show_progress: bool = True,
    save: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate one MF-DSRQ torch epsilon against MFRL baselines, optionally in worker processes."""
    algorithms = tuple(algorithms)
    eval_device = _resolve_torch_device(
        device if device is not None else cfg.get("device"),
        use_gpu=cfg.get("use_gpu", True),
    )
    baseline_folders = {
        algorithm: Path(folder)
        for algorithm, folder in (baseline_folders or {}).items()
    }
    for algorithm in algorithms:
        baseline_folders.setdefault(algorithm, find_latest_mfrl_run(algorithm, baseline_root))

    type_names = list(cfg["type_prefixes"].keys())
    if len(type_names) != 2:
        raise ValueError("Head-to-head MFRL comparison expects exactly two teams.")
    assignments = [(type_names[0], type_names[1])]
    if evaluate_both_sides:
        assignments.append((type_names[1], type_names[0]))

    workers = max(int(workers), 1)
    base_units = max(len(algorithms) * len(assignments), 1)
    if episode_chunk_size is None:
        chunks_per_assignment = max(1, min(int(num_episodes_per_side), (workers + base_units - 1) // base_units))
    else:
        chunk_size = max(int(episode_chunk_size), 1)
        chunks_per_assignment = max(1, min(int(num_episodes_per_side), (int(num_episodes_per_side) + chunk_size - 1) // chunk_size))
    chunks = _split_episode_chunks(num_episodes_per_side, chunks_per_assignment)
    tasks: list[dict[str, Any]] = []
    for algorithm in algorithms:
        for mfdsrq_team, baseline_team in assignments:
            for episode_offset, chunk_size in chunks:
                tasks.append(
                    {
                        "cfg": dict(cfg),
                        "checkpoint_paths": {team: str(path) for team, path in checkpoint_paths.items()},
                        "baseline_folder": str(baseline_folders[algorithm]),
                        "baseline": algorithm,
                        "mfdsrq_team": mfdsrq_team,
                        "baseline_team": baseline_team,
                        "num_episodes": chunk_size,
                        "episode_offset": episode_offset,
                        "max_steps": max_steps,
                        "device": str(eval_device),
                    }
                )

    if not tasks:
        worker_results = []
    elif workers == 1 or len(tasks) == 1:
        total_episodes = sum(int(task["num_episodes"]) for task in tasks)
        progress = _make_eval_progress_bar(total_episodes, epsilon=float(epsilon), enabled=show_progress)
        worker_results = []
        try:
            progress_queue = _LocalProgressQueue(progress)
            for task in tasks:
                task_with_progress = dict(task)
                task_with_progress["progress_queue"] = progress_queue
                result = _evaluate_mfdsrq_assignment_worker(task_with_progress)
                worker_results.append(result)
        finally:
            if progress is not None:
                progress.close()
    else:
        worker_results = []
        total_episodes = sum(int(task["num_episodes"]) for task in tasks)
        progress = _make_eval_progress_bar(total_episodes, epsilon=float(epsilon), enabled=show_progress)
        mp_context = _eval_multiprocessing_context(eval_device)
        with mp_context.Manager() as manager:
            progress_queue = manager.Queue()
            tasks_with_progress = []
            for task in tasks:
                task_with_progress = dict(task)
                task_with_progress["progress_queue"] = progress_queue
                tasks_with_progress.append(task_with_progress)
            with ProcessPoolExecutor(
                max_workers=min(workers, len(tasks)),
                mp_context=mp_context,
            ) as executor:
                pending = {
                    executor.submit(_evaluate_mfdsrq_assignment_worker, task)
                    for task in tasks_with_progress
                }
                try:
                    while pending:
                        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        _drain_progress_queue(progress_queue, progress)
                        for future in done:
                            worker_results.append(future.result())
                    _drain_progress_queue(progress_queue, progress)
                finally:
                    _drain_progress_queue(progress_queue, progress)
                    if progress is not None:
                        progress.close()

    payload = _comparison_payload_from_worker_results(
        epsilon=float(epsilon),
        checkpoint_paths=checkpoint_paths,
        algorithms=algorithms,
        baseline_folders=baseline_folders,
        worker_results=worker_results,
        num_episodes_per_side=int(num_episodes_per_side),
        evaluate_both_sides=evaluate_both_sides,
        workers=workers,
        device=str(eval_device),
    )
    if save:
        out_path = Path(payload["checkpoint_dir"]) / "head_to_head_vs_mfrl_baselines.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        payload["results_path"] = str(out_path)
        print(f"Saved head-to-head comparison to {out_path}")
    return payload


def _fixed_side_tournament_payload_from_worker_results(
    *,
    epsilon: float,
    checkpoint_paths: Mapping[str, str | Path],
    algorithms: tuple[str, ...],
    matchup_pairs: tuple[tuple[str, str], ...],
    baseline_folders: dict[str, Path],
    worker_results: list[dict[str, Any]],
    num_episodes_per_matchup: int,
    workers: int,
    device: str = "cpu",
    experiment_label: str | None = None,
) -> dict[str, Any]:
    rows = []
    matchups = {}
    result_dir = _checkpoint_result_dir(checkpoint_paths)
    for main_algorithm, opponent_algorithm in matchup_pairs:
        matchups.setdefault(main_algorithm, {})
        matchup_records = [
            record
            for result in worker_results
            if result["main_algorithm"] == main_algorithm
            and result["opponent_algorithm"] == opponent_algorithm
            for record in result["records"]
        ]
        matchup_records.sort(key=lambda record: int(record.get("episode", 0)))
        summary = _summarize_fixed_side_records(matchup_records)
        result_for_pair = next(
            (
                result
                for result in worker_results
                if result["main_algorithm"] == main_algorithm
                and result["opponent_algorithm"] == opponent_algorithm
            ),
            {},
        )
        matchups[main_algorithm][opponent_algorithm] = {
            "main_algorithm": main_algorithm,
            "opponent_algorithm": opponent_algorithm,
            "main_checkpoint": result_for_pair.get("main_checkpoint"),
            "opponent_checkpoint": result_for_pair.get("opponent_checkpoint"),
            "num_episodes": summary["episodes"],
            "records": matchup_records,
            "summary": summary,
        }
        rows.append(
            {
                "main_algorithm": main_algorithm,
                "opponent_algorithm": opponent_algorithm,
                "main_win_rate": summary["main_win_rate"],
                "opponent_win_rate": summary["opponent_win_rate"],
                "tie_rate": summary["tie_rate"],
                "mean_main_kills": summary["mean_main_kills"],
                "mean_opponent_kills": summary["mean_opponent_kills"],
                "mean_main_reward": summary["mean_main_reward"],
                "mean_opponent_reward": summary["mean_opponent_reward"],
                "episodes": summary["episodes"],
            }
        )
    return {
        "epsilon": float(epsilon),
        "experiment_label": experiment_label,
        "checkpoint_dir": str(result_dir),
        "checkpoint_source": _checkpoint_source_payload(checkpoint_paths),
        "algorithms": list(algorithms),
        "matchup_pairs": [list(pair) for pair in matchup_pairs],
        "baseline_folders": {algorithm: str(path) for algorithm, path in baseline_folders.items()},
        "num_episodes_per_matchup": int(num_episodes_per_matchup),
        "evaluate_both_sides": False,
        "workers": int(workers),
        "device": str(device),
        "rows": rows,
        "matchups": matchups,
    }


def evaluate_mfdsrq_torch_epsilon_fixed_side_tournament(
    cfg: dict[str, Any],
    epsilon: float,
    checkpoint_paths: Mapping[str, str | Path],
    *,
    baseline_root: str | Path = DEFAULT_BASELINE_RUNS_DIR,
    algorithms: tuple[str, ...] = ("mfdsrq", *BASELINE_ALGORITHMS),
    matchup_pairs: tuple[tuple[str, str], ...] | None = None,
    baseline_folders: dict[str, str | Path] | None = None,
    num_episodes_per_matchup: int = 25,
    max_steps: int | None = None,
    workers: int = 1,
    episode_chunk_size: int | None = None,
    show_progress: bool = True,
    save: bool = True,
    device: str | None = None,
    experiment_label: str | None = None,
) -> dict[str, Any]:
    """Evaluate fixed-side main-vs-opponent tournament for one MF-DSRQ epsilon."""
    algorithms = tuple(str(algorithm).lower() for algorithm in algorithms)
    if matchup_pairs is None:
        resolved_matchup_pairs = tuple(
            (main_algorithm, opponent_algorithm)
            for main_algorithm in algorithms
            for opponent_algorithm in algorithms
        )
    else:
        resolved_matchup_pairs = tuple(
            (str(main_algorithm).lower(), str(opponent_algorithm).lower())
            for main_algorithm, opponent_algorithm in matchup_pairs
        )
    if not resolved_matchup_pairs:
        raise ValueError("matchup_pairs must contain at least one matchup.")

    algorithm_set = set(algorithms)
    missing_algorithms = sorted(
        {
            algorithm
            for pair in resolved_matchup_pairs
            for algorithm in pair
            if algorithm not in algorithm_set
        }
    )
    if missing_algorithms:
        raise ValueError(
            "matchup_pairs contains algorithms not listed in algorithms: "
            + ", ".join(missing_algorithms)
        )

    eval_device = _resolve_torch_device(
        device if device is not None else cfg.get("device"),
        use_gpu=cfg.get("use_gpu", True),
    )
    baseline_folders = {
        algorithm: Path(folder)
        for algorithm, folder in (baseline_folders or {}).items()
    }
    for algorithm in algorithms:
        if algorithm == "mfdsrq":
            continue
        baseline_folders.setdefault(algorithm, find_latest_mfrl_run(algorithm, baseline_root))

    workers = max(int(workers), 1)
    base_units = max(len(resolved_matchup_pairs), 1)
    if episode_chunk_size is None:
        chunks_per_matchup = max(1, min(int(num_episodes_per_matchup), (workers + base_units - 1) // base_units))
    else:
        chunk_size = max(int(episode_chunk_size), 1)
        chunks_per_matchup = max(
            1,
            min(
                int(num_episodes_per_matchup),
                (int(num_episodes_per_matchup) + chunk_size - 1) // chunk_size,
            ),
        )
    chunks = _split_episode_chunks(num_episodes_per_matchup, chunks_per_matchup)

    tasks: list[dict[str, Any]] = []
    for main_algorithm, opponent_algorithm in resolved_matchup_pairs:
        for episode_offset, chunk_size in chunks:
            tasks.append(
                {
                    "cfg": dict(cfg),
                    "checkpoint_paths": {team: str(path) for team, path in checkpoint_paths.items()},
                    "baseline_folders": {algorithm: str(path) for algorithm, path in baseline_folders.items()},
                    "main_algorithm": main_algorithm,
                    "opponent_algorithm": opponent_algorithm,
                    "num_episodes": chunk_size,
                    "episode_offset": episode_offset,
                    "max_steps": max_steps,
                    "device": str(eval_device),
                }
            )

    if not tasks:
        worker_results = []
    elif workers == 1 or len(tasks) == 1:
        total_episodes = sum(int(task["num_episodes"]) for task in tasks)
        progress = _make_eval_progress_bar(total_episodes, epsilon=float(epsilon), enabled=show_progress)
        worker_results = []
        try:
            progress_queue = _LocalProgressQueue(progress)
            for task in tasks:
                task_with_progress = dict(task)
                task_with_progress["progress_queue"] = progress_queue
                worker_results.append(_evaluate_fixed_side_tournament_worker(task_with_progress))
        finally:
            if progress is not None:
                progress.close()
    else:
        worker_results = []
        total_episodes = sum(int(task["num_episodes"]) for task in tasks)
        progress = _make_eval_progress_bar(total_episodes, epsilon=float(epsilon), enabled=show_progress)
        mp_context = _eval_multiprocessing_context(eval_device)
        with mp_context.Manager() as manager:
            progress_queue = manager.Queue()
            tasks_with_progress = []
            for task in tasks:
                task_with_progress = dict(task)
                task_with_progress["progress_queue"] = progress_queue
                tasks_with_progress.append(task_with_progress)
            with ProcessPoolExecutor(
                max_workers=min(workers, len(tasks)),
                mp_context=mp_context,
            ) as executor:
                pending = {
                    executor.submit(_evaluate_fixed_side_tournament_worker, task)
                    for task in tasks_with_progress
                }
                try:
                    while pending:
                        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        _drain_progress_queue(progress_queue, progress)
                        for future in done:
                            worker_results.append(future.result())
                    _drain_progress_queue(progress_queue, progress)
                finally:
                    _drain_progress_queue(progress_queue, progress)
                    if progress is not None:
                        progress.close()

    payload = _fixed_side_tournament_payload_from_worker_results(
        epsilon=float(epsilon),
        checkpoint_paths=checkpoint_paths,
        algorithms=algorithms,
        matchup_pairs=resolved_matchup_pairs,
        baseline_folders=baseline_folders,
        worker_results=worker_results,
        num_episodes_per_matchup=int(num_episodes_per_matchup),
        workers=workers,
        device=str(eval_device),
        experiment_label=experiment_label,
    )
    if save:
        out_path = Path(payload["checkpoint_dir"]) / "fixed_side_tournament.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        payload["results_path"] = str(out_path)
        print(f"Saved fixed-side tournament to {out_path}")
    return payload


def evaluate_mfdsrq_against_baselines(
    cfg: dict[str, Any],
    checkpoint_dir: str | Path | Mapping[str, str | Path],
    *,
    baseline_root: str | Path = DEFAULT_BASELINE_RUNS_DIR,
    algorithms: tuple[str, ...] = BASELINE_ALGORITHMS,
    baseline_folders: dict[str, str | Path] | None = None,
    num_episodes: int = 20,
    max_steps: int | None = None,
    evaluate_both_sides: bool = True,
    save: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate MF-DSRQ head-to-head against PyTorch MFRL baseline checkpoints."""
    rows = []
    results = {}
    baseline_folders = baseline_folders or {}
    for algorithm in algorithms:
        if algorithm in baseline_folders:
            baseline_folder = Path(baseline_folders[algorithm])
        else:
            baseline_folder = find_latest_mfrl_run(algorithm, baseline_root)
        result = evaluate_mfdsrq_vs_mfrl_baseline(
            dict(cfg),
            checkpoint_dir,
            baseline_folder,
            baseline_name=algorithm,
            num_episodes=int(num_episodes),
            max_steps=max_steps,
            evaluate_both_sides=evaluate_both_sides,
            device=device,
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

    result_dir = _checkpoint_result_dir(checkpoint_dir)
    payload = {
        "checkpoint_dir": str(result_dir),
        "checkpoint_source": _checkpoint_source_payload(checkpoint_dir),
        "algorithms": list(algorithms),
        "device": str(
            _resolve_torch_device(
                device if device is not None else cfg.get("device"),
                use_gpu=cfg.get("use_gpu", True),
            )
        ),
        "rows": rows,
        "results": results,
    }
    if save:
        out_path = result_dir / "head_to_head_vs_mfrl_baselines.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        payload["results_path"] = str(out_path)
        print(f"Saved head-to-head comparison to {out_path}")
    return payload


def _algorithm_display_name(algorithm: str) -> str:
    labels = {
        "mfdsrq": "MF-DSRQ",
        "iql": "IQL",
        "ac": "AC",
        "mfq": "MFQ",
    }
    return labels.get(str(algorithm).lower(), str(algorithm).upper())


def _uses_selected_fixed_side_matchups(tournament: dict[str, Any]) -> bool:
    matchup_pairs = tournament.get("matchup_pairs")
    if not matchup_pairs:
        return False
    algorithms = [str(algorithm).lower() for algorithm in tournament["algorithms"]]
    full_pairs = {
        (main_algorithm, opponent_algorithm)
        for main_algorithm in algorithms
        for opponent_algorithm in algorithms
    }
    actual_pairs = {
        (str(main_algorithm).lower(), str(opponent_algorithm).lower())
        for main_algorithm, opponent_algorithm in matchup_pairs
    }
    return actual_pairs != full_pairs


def _plot_fixed_side_tournament_metric(
    tournament: dict[str, Any],
    *,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    save: bool = True,
):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(tournament["rows"])
    if df.empty:
        raise ValueError("No tournament rows to plot.")
    algorithms = [str(algorithm).lower() for algorithm in tournament["algorithms"]]
    if _uses_selected_fixed_side_matchups(tournament):
        pair_order = {
            (str(main_algorithm).lower(), str(opponent_algorithm).lower()): idx
            for idx, (main_algorithm, opponent_algorithm) in enumerate(tournament["matchup_pairs"])
        }
        df = df.assign(
            _order=[
                pair_order.get((str(row.main_algorithm).lower(), str(row.opponent_algorithm).lower()), idx)
                for idx, row in enumerate(df.itertuples(index=False))
            ]
        ).sort_values("_order")
        labels = [
            f"{_algorithm_display_name(row.main_algorithm)} vs {_algorithm_display_name(row.opponent_algorithm)}"
            for row in df.itertuples(index=False)
        ]
        x = list(range(len(df)))
        fig_width = max(7, 2.2 * len(df) + 2)
        fig, ax = plt.subplots(figsize=(fig_width, 4.8))
        ax.bar(x, df[metric].to_numpy(dtype=float), width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if metric.endswith("win_rate"):
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        if save:
            out_path = Path(tournament["checkpoint_dir"]) / filename
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved {title.lower()} plot to {out_path}")
        return fig

    matrix = (
        df.pivot(index="main_algorithm", columns="opponent_algorithm", values=metric)
        .reindex(index=algorithms, columns=algorithms)
        .fillna(0.0)
    )
    labels = [_algorithm_display_name(algorithm) for algorithm in algorithms]
    x = list(range(len(algorithms)))
    width = min(0.18, 0.75 / max(len(algorithms), 1))
    offsets = [
        (idx - (len(algorithms) - 1) / 2.0) * width
        for idx in range(len(algorithms))
    ]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for col_idx, opponent_algorithm in enumerate(algorithms):
        values = matrix[opponent_algorithm].to_numpy(dtype=float)
        ax.bar(
            [pos + offsets[col_idx] for pos in x],
            values,
            width,
            label=f"vs {_algorithm_display_name(opponent_algorithm)}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metric.endswith("win_rate"):
        ax.set_ylim(0, 1.05)
    ax.legend(ncols=min(len(algorithms), 4), fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if save:
        out_path = Path(tournament["checkpoint_dir"]) / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {title.lower()} plot to {out_path}")
    return fig


def plot_fixed_side_tournament_win_rates(tournament: dict[str, Any], *, save: bool = True):
    """Plot fixed-side tournament main win rates grouped by main algorithm."""
    epsilon = float(tournament.get("epsilon", 0.0))
    label = tournament.get("experiment_label") or f"eps={epsilon:g}"
    return _plot_fixed_side_tournament_metric(
        tournament,
        metric="main_win_rate",
        ylabel="Main-side win rate",
        title=f"Fixed-side tournament win rates, {label}",
        filename="fixed_side_tournament_win_rates.png",
        save=save,
    )


def plot_fixed_side_tournament_rewards(tournament: dict[str, Any], *, save: bool = True):
    """Plot fixed-side tournament main average rewards grouped by main algorithm."""
    epsilon = float(tournament.get("epsilon", 0.0))
    label = tournament.get("experiment_label") or f"eps={epsilon:g}"
    return _plot_fixed_side_tournament_metric(
        tournament,
        metric="mean_main_reward",
        ylabel="Main-side average total reward",
        title=f"Fixed-side tournament rewards, {label}",
        filename="fixed_side_tournament_rewards.png",
        save=save,
    )


def plot_fixed_side_tournament_bars(tournament: dict[str, Any], *, save: bool = True):
    """Plot fixed-side tournament win-rate and reward bar charts."""
    return (
        plot_fixed_side_tournament_win_rates(tournament, save=save),
        plot_fixed_side_tournament_rewards(tournament, save=save),
    )


def plot_mfdsrq_torch_baseline_bars(
    comparison: dict[str, Any],
    *,
    epsilon: float | None = None,
    save: bool = True,
):
    """Plot MF-DSRQ torch versus baseline win rates and rewards for one epsilon."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(comparison["rows"])
    if df.empty:
        raise ValueError("No comparison rows to plot.")
    epsilon = float(comparison.get("epsilon", epsilon if epsilon is not None else 0.0))
    x = list(range(len(df)))
    width = 0.36
    labels = [f"vs {name.upper()}" for name in df["baseline"]]
    mf_label = f"MF-DSRQ torch eps={epsilon:g}"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar([i - width / 2 for i in x], df["mfdsrq_win_rate"], width, label=mf_label)
    axes[0].bar([i + width / 2 for i in x], df["baseline_win_rate"], width, label="Baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Average win rate")
    axes[0].set_title("Head-to-head win rate")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([i - width / 2 for i in x], df["mean_mfdsrq_reward"], width, label=mf_label)
    axes[1].bar([i + width / 2 for i in x], df["mean_baseline_reward"], width, label="Baseline")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Average total reward")
    axes[1].set_title("Head-to-head total reward")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    if save:
        out_path = Path(comparison["checkpoint_dir"]) / "head_to_head_win_rate_and_reward.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved win-rate/reward plot to {out_path}")
    return fig


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
    """Reload a trained PyTorch MFRL baseline and display one evaluation rollout."""
    return sample_mfrl_rollout_video(
        result_or_folder,
        max_steps=max_steps,
        fps=fps,
        deterministic=deterministic,
        title=title,
    )
