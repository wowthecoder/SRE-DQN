import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint


def summarize_durations(durations):
    durations = np.asarray(durations, dtype=float)
    if durations.size == 0:
        return {
            "count": 0,
            "mean_milliseconds": None,
            "min_milliseconds": None,
            "max_milliseconds": None,
            "std_milliseconds": None,
        }
    return {
        "count": int(durations.size),
        "mean_milliseconds": float(np.mean(durations) * 1_000.0),
        "min_milliseconds": float(np.min(durations) * 1_000.0),
        "max_milliseconds": float(np.max(durations) * 1_000.0),
        "std_milliseconds": float(np.std(durations) * 1_000.0),
    }


def combine_duration_summaries(summaries):
    count = sum(int(summary.get("count", 0) or 0) for summary in summaries)
    if count == 0:
        return summarize_durations([])

    weighted_sum = 0.0
    weighted_sumsq = 0.0
    mins = []
    maxes = []
    for summary in summaries:
        summary_count = int(summary.get("count", 0) or 0)
        if summary_count == 0:
            continue
        mean = _duration_summary_value_seconds(summary, "mean")
        std = _duration_summary_value_seconds(summary, "std")
        weighted_sum += mean * summary_count
        weighted_sumsq += (std * std + mean * mean) * summary_count
        mins.append(_duration_summary_value_seconds(summary, "min"))
        maxes.append(_duration_summary_value_seconds(summary, "max"))

    mean = weighted_sum / count
    variance = max(weighted_sumsq / count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    return {
        "count": int(count),
        "mean_milliseconds": float(mean * 1_000.0),
        "min_milliseconds": float(min(mins) * 1_000.0),
        "max_milliseconds": float(max(maxes) * 1_000.0),
        "std_milliseconds": float(std * 1_000.0),
    }


def _duration_summary_value_seconds(summary, kind):
    milliseconds_key = f"{kind}_milliseconds"
    if milliseconds_key in summary:
        return float(summary[milliseconds_key]) / 1_000.0
    seconds_key = f"{kind}_seconds"
    if seconds_key in summary:
        return float(summary[seconds_key])
    microseconds_key = f"{kind}_microseconds"
    if microseconds_key in summary:
        return float(summary[microseconds_key]) / 1_000_000.0
    raise KeyError(f"Duration summary missing {kind} duration field.")


def collect_timing_stats(
    agents,
    wall_clock_seconds=None,
    episode_durations=None,
    include_episode_durations=True,
):
    sre_solve_summaries = []
    backend_solve_summaries = []
    update_times_by_agent = []
    seen_agent_objects = set()

    for idx, agent_wrapper in enumerate(agents):
        agent = getattr(agent_wrapper, "agent", agent_wrapper)
        agent_object_id = id(agent)
        if agent_object_id in seen_agent_objects:
            continue
        seen_agent_objects.add(agent_object_id)
        if hasattr(agent, "get_sre_solve_time_summary"):
            agent_sre_solve_summary = agent.get_sre_solve_time_summary()
        else:
            agent_sre_solve_summary = summarize_durations([])
        solver = getattr(agent, "sre_solver", None) or getattr(agent, "path_solver", None)
        if solver is not None and hasattr(solver, "get_solve_time_summary"):
            agent_backend_solve_summary = solver.get_solve_time_summary()
        else:
            agent_backend_solve_summary = summarize_durations([])
        if hasattr(agent, "get_sre_cache_summary"):
            agent_sre_cache_summary = agent.get_sre_cache_summary()
        else:
            agent_sre_cache_summary = {}
        agent_update_times = list(getattr(agent, "update_times", []) or [])
        sre_solve_summaries.append(agent_sre_solve_summary)
        backend_solve_summaries.append(agent_backend_solve_summary)
        update_times_by_agent.append(
            {
                "agent_index": idx,
                "algorithm": getattr(agent_wrapper, "algorithm", type(agent).__name__),
                "sre_solve_time": agent_sre_solve_summary,
                "backend_solve_time": agent_backend_solve_summary,
                "sre_policy_cache": agent_sre_cache_summary,
                "update_time": summarize_durations(agent_update_times),
            }
        )

    episode_durations = list(episode_durations or [])
    timing = {
        "wall_clock_seconds": (
            None if wall_clock_seconds is None else float(wall_clock_seconds)
        ),
        "episode_time": summarize_durations(episode_durations),
        "sre_solve_time": combine_duration_summaries(sre_solve_summaries),
        "backend_solve_time": combine_duration_summaries(backend_solve_summaries),
        "agents": update_times_by_agent,
    }
    cache_summaries = [
        agent.get("sre_policy_cache", {})
        for agent in update_times_by_agent
        if agent.get("sre_policy_cache")
    ]
    if cache_summaries:
        timing["sre_policy_cache"] = {
            "requests": int(sum(summary.get("requests", 0) or 0 for summary in cache_summaries)),
            "exact_hits": int(sum(summary.get("exact_hits", 0) or 0 for summary in cache_summaries)),
            "approx_hits": int(sum(summary.get("approx_hits", 0) or 0 for summary in cache_summaries)),
            "misses": int(sum(summary.get("misses", 0) or 0 for summary in cache_summaries)),
            "path_solves_avoided": int(sum(summary.get("path_solves_avoided", 0) or 0 for summary in cache_summaries)),
            "evictions": int(sum(summary.get("evictions", 0) or 0 for summary in cache_summaries)),
        }
        requests = timing["sre_policy_cache"]["requests"]
        avoided = timing["sre_policy_cache"]["path_solves_avoided"]
        timing["sre_policy_cache"]["hit_rate"] = (
            None if requests == 0 else float(avoided / requests)
        )
    if include_episode_durations:
        timing["episode_durations_seconds"] = [
            float(duration) for duration in episode_durations
        ]
    return timing


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _reward_summary(rows, *, agent_first, labels=None):
    try:
        values = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.size == 0:
        return None
    if values.ndim == 1:
        values = values.reshape(1, -1) if agent_first else values.reshape(-1, 1)
    if values.ndim != 2:
        return None
    agent_values = values if agent_first else values.T
    if labels is None or len(labels) != agent_values.shape[0]:
        labels = [f"Agent {idx + 1}" for idx in range(agent_values.shape[0])]
    joint = agent_values.sum(axis=0)
    return {
        "episodes": int(agent_values.shape[1]),
        "per_agent": [
            {
                "agent": str(labels[idx]),
                "mean": float(np.mean(agent_values[idx])),
                "std": float(np.std(agent_values[idx])),
                "min": float(np.min(agent_values[idx])),
                "max": float(np.max(agent_values[idx])),
            }
            for idx in range(agent_values.shape[0])
        ],
        "joint": {
            "mean": float(np.mean(joint)),
            "std": float(np.std(joint)),
            "min": float(np.min(joint)),
            "max": float(np.max(joint)),
        },
    }


def compact_training_stats(
    stats,
    *,
    drop_reward_histories=False,
    drop_lbf_episode_details=False,
    drop_episode_lengths=False,
):
    """Return a copy with bulky per-episode histories replaced by summaries."""

    def compact(value):
        if isinstance(value, dict):
            result = {}
            labels = value.get("agent_labels")
            if drop_reward_histories and "rewards" in value:
                summary = _reward_summary(value["rewards"], agent_first=True, labels=labels)
                if summary is not None:
                    result["reward_summary"] = summary
            if drop_reward_histories and "episode_rewards" in value:
                summary = _reward_summary(
                    value["episode_rewards"],
                    agent_first=False,
                    labels=labels,
                )
                if summary is not None:
                    result["episode_reward_summary"] = summary
                    means = [row["mean"] for row in summary["per_agent"]]
                    result.setdefault("mean_agent_rewards", means)
                    result.setdefault("mean_joint_reward", summary["joint"]["mean"])

            for key, child in value.items():
                if drop_reward_histories and key in {
                    "rewards",
                    "episode_rewards",
                    "joint_rewards",
                }:
                    continue
                if drop_lbf_episode_details and key in {
                    "episode_metrics",
                    "metric_totals",
                }:
                    continue
                if drop_episode_lengths and key == "episode_lengths":
                    continue
                result[str(key)] = compact(child)
            return result
        if isinstance(value, list):
            return [compact(item) for item in value]
        if isinstance(value, tuple):
            return [compact(item) for item in value]
        return value

    return compact(stats)


def save_training_stats(
    stats_path,
    stats,
    *,
    drop_reward_histories=False,
    drop_lbf_episode_details=False,
    drop_episode_lengths=False,
):
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_to_save = compact_training_stats(
        stats,
        drop_reward_histories=drop_reward_histories,
        drop_lbf_episode_details=drop_lbf_episode_details,
        drop_episode_lengths=drop_episode_lengths,
    )
    if stats_path.suffix == ".pkl":
        with open(stats_path, "wb") as f:
            pickle.dump(stats_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        return stats_path

    serialized = _to_jsonable(stats_to_save)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, sort_keys=False)
        f.write("\n")
    return stats_path


def load_training_stats(stats_path):
    stats_path = Path(stats_path)
    if stats_path.suffix == ".pkl":
        with open(stats_path, "rb") as f:
            return pickle.load(f)
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_rewards(stats):
    rewards = stats["rewards"]
    last_n = min(stats.get("last_n", 1000), len(rewards[0]))
    rows = []
    agent_labels = stats.get("agent_labels") or [
        f"Agent {agent_id}" for agent_id in range(1, len(rewards) + 1)
    ]

    for agent_id, agent_rewards in enumerate(rewards, start=1):
        data = np.asarray(agent_rewards, dtype=float)
        last_rewards = data[-last_n:] if last_n else data
        row = {
            "Scenario": stats.get("scenario_name", "Scenario"),
            "Agent": agent_labels[agent_id - 1],
            "Mean": float(np.mean(data)),
            "Std": float(np.std(data)),
            "MeanLastN": float(np.mean(last_rewards)),
            "StdLastN": float(np.std(last_rewards)),
            "LastN": int(last_n),
            "Episodes": int(len(data)),
        }
        if stats.get("pair_label"):
            row["Pair"] = stats["pair_label"]
        rows.append(row)

    return rows


def print_summary_table(rows):
    last_n = rows[0]["LastN"]
    headers = ["Scenario"]
    if "Pair" in rows[0]:
        headers.append("Pair")
    headers.extend(
        [
            "Agent",
            "Mean",
            "Std",
            f"Mean (Last {last_n})",
            f"Std (Last {last_n})",
            "Episodes",
        ]
    )

    table_rows = []
    for row in rows:
        values = [row["Scenario"]]
        if "Pair" in row:
            values.append(row["Pair"])
        values.extend(
            [
                row["Agent"],
                f'{row["Mean"]:.2f}',
                f'{row["Std"]:.2f}',
                f'{row["MeanLastN"]:.2f}',
                f'{row["StdLastN"]:.2f}',
                str(row["Episodes"]),
            ]
        )
        table_rows.append(values)

    widths = [len(header) for header in headers]
    for row in table_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    header_line = " | ".join(
        header.ljust(width) for header, width in zip(headers, widths)
    )
    separator_line = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(separator_line)
    for row in table_rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def plot_training_stats(stats, out_path=None):
    rewards = stats["rewards"]
    window_size = stats.get("window_size", 25)
    separator = stats.get("separator", 10)
    last_n = min(stats.get("last_n", 1000), len(rewards[0]))
    agent_labels = stats.get("agent_labels") or [
        f"Agent {agent_id}" for agent_id in range(1, len(rewards) + 1)
    ]
    num_agents = len(rewards)
    fig_height = max(5, 4 * num_agents)
    fig, axes = plt.subplots(num_agents, 1, figsize=(15, fig_height), sharex=True)
    axes = np.atleast_1d(axes)

    title_parts = [stats.get("scenario_name", "Scenario")]
    if stats.get("pair_label"):
        title_parts.append(stats["pair_label"])

    for agent_id, ax in enumerate(axes):
        data = np.asarray(rewards[agent_id], dtype=float)
        episodes = np.arange(len(data))
        moving_avg = [
            float(np.mean(data[i : i + window_size]))
            for i in range(0, len(data), window_size)
        ]
        episodes_moving_avg = episodes[: len(moving_avg) * window_size : window_size]
        mean_val = float(np.mean(data))
        mean_last_n = float(np.mean(data[-last_n:]))
        std_val = float(np.std(data))
        std_last_n = float(np.std(data[-last_n:]))
        label = agent_labels[agent_id]

        ax.plot(
            episodes[::separator],
            data[::separator],
            label=label,
            marker="o",
            linestyle="None",
            markersize=3,
        )
        ax.plot(
            episodes_moving_avg,
            moving_avg,
            label=f"{label} Rolling Average",
            color="blue",
            linestyle="-",
        )
        ax.axhline(
            mean_val, color="red", linestyle="--", label=f"{label} Mean ({mean_val:.2f})"
        )
        ax.plot([], [], " ", label=f"{label} Mean (Last {last_n} episodes): {mean_last_n:.2f}")
        ax.plot([], [], " ", label=f"{label} Std Dev ({std_val:.2f})")
        ax.plot([], [], " ", label=f"{label} Std Dev (Last {last_n} episodes): {std_last_n:.2f}")
        ax.set_title(" - ".join(title_parts + [label]))
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def _fmt_duration(summary):
    if not summary or not summary.get("count"):
        return "count=0"
    mean_ms = _duration_summary_value_seconds(summary, "mean") * 1_000.0
    min_ms = _duration_summary_value_seconds(summary, "min") * 1_000.0
    max_ms = _duration_summary_value_seconds(summary, "max") * 1_000.0
    std_ms = _duration_summary_value_seconds(summary, "std") * 1_000.0
    return (
        f"count={summary['count']}, "
        f"mean={mean_ms:.3f} ms, "
        f"min={min_ms:.3f} ms, "
        f"max={max_ms:.3f} ms, "
        f"std={std_ms:.3f} ms"
    )


def print_timing_stats(stats):
    timing = stats.get("timing")
    if not timing:
        print("Timing: not recorded in this stats file.")
        return

    wall_clock = timing.get("wall_clock_seconds")
    if wall_clock is None:
        print("Wall clock: not recorded")
    else:
        print(f"Wall clock: {wall_clock:.3f}s ({wall_clock / 60.0:.2f} min)")

    print(f"Episode time:   {_fmt_duration(timing.get('episode_time'))}")
    print(f"SRE solve time: {_fmt_duration(timing.get('sre_solve_time'))}")
    backend_summary = timing.get("backend_solve_time") or timing.get("path_solve_time")
    print(f"Backend solve:  {_fmt_duration(backend_summary)}")

    for agent in timing.get("agents", []):
        label = agent.get("algorithm") or agent.get("agent_type") or f"Agent {agent.get('agent_index')}"
        print(f"  Agent {agent.get('agent_index')} ({label})")
        print(f"    SRE solve:  {_fmt_duration(agent.get('sre_solve_time'))}")
        agent_backend = agent.get("backend_solve_time") or agent.get("path_solve_time")
        print(f"    Backend:    {_fmt_duration(agent_backend)}")
        print(f"    Update:     {_fmt_duration(agent.get('update_time'))}")


def print_stats_payload(stats, title=None):
    if title:
        print(title)
        print("=" * len(title))
    pprint(stats, sort_dicts=False, width=120)
    print("\nTiming Summary")
    print_timing_stats(stats)


def discover_stats_files(root="."):
    root = Path(root)
    patterns = [
        "training_stats_scenario*.pkl",
        "training_stats_scenario*.txt",
        "deep_srq_epsilon_runs/*/*/training_stats.pkl",
        "deep_srq_epsilon_runs/*/*/training_stats.txt",
        "comparison_runs/*/*/training_stats.pkl",
        "comparison_runs/*/*/training_stats.txt",
        "scenario_runs/*/*/training_stats.txt",
    ]
    paths = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted(set(paths))
