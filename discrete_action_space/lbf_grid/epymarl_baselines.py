"""Run LBF baselines through EPyMARL."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .epymarl_lbf_env import (
    get_epymarl_lbf_scenario,
    register_epymarl_lbf_envs,
)


EPYMARL_ALGORITHMS = ("random", "iql", "ippo", "mappo", "qmix", "maa2c")
UNSUPPORTED_EPYMARL_OVERRIDES = frozenset({"env_args.disable_env_checker"})


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _format_epymarl_override_value(value) -> str:
    """Format a Python value for EPyMARL's ``with key=value`` CLI overrides."""
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return "null"
    return str(value)


def _epymarl_model_token(scenario_key: str, seed: int, algorithm: str) -> str:
    """Return the deterministic EPyMARL model directory token."""
    return f"{scenario_key}/{int(seed)}/{str(algorithm).lower()}"


def _load_episode_metrics(metrics_dir: str | Path) -> list[dict]:
    metrics_dir = Path(metrics_dir)
    if not metrics_dir.exists():
        return []

    records = []
    sequence = 0
    for path in sorted(metrics_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload["_source_file"] = path.name
                payload["_sequence"] = sequence
                sequence += 1
                records.append(payload)
    return sorted(
        records,
        key=lambda item: (
            str(item.get("phase", "")),
            int(item.get("episode_index", 0)),
            int(item.get("pid", 0)),
            int(item.get("_sequence", 0)),
        ),
    )


def _episode_metric_totals(records: list[dict]) -> dict:
    num_agents = 0
    for record in records:
        for key in (
            "foods_collected_per_agent",
            "empty_loads_per_agent",
            "invalid_loads_per_agent",
        ):
            values = record.get(key)
            if isinstance(values, dict):
                for agent_key in values:
                    try:
                        num_agents = max(num_agents, int(str(agent_key).split("_")[-1]) + 1)
                    except ValueError:
                        pass

    per_agent = lambda: {f"agent_{idx}": 0 for idx in range(num_agents)}
    totals = {
        "episode_count": len(records),
        "episode_lengths": [int(record.get("episode_length", 0)) for record in records],
        "foods_collected_total": 0,
        "foods_collected_per_agent": per_agent(),
        "empty_loads_total": 0,
        "empty_loads_per_agent": per_agent(),
        "invalid_loads_total": 0,
        "invalid_loads_per_agent": per_agent(),
    }

    for record in records:
        totals["foods_collected_total"] += int(record.get("foods_collected_total", 0))
        totals["empty_loads_total"] += int(record.get("empty_loads_total", 0))
        totals["invalid_loads_total"] += int(record.get("invalid_loads_total", 0))
        for src_key, dst_key in (
            ("foods_collected_per_agent", "foods_collected_per_agent"),
            ("empty_loads_per_agent", "empty_loads_per_agent"),
            ("invalid_loads_per_agent", "invalid_loads_per_agent"),
        ):
            for agent, value in (record.get(src_key) or {}).items():
                totals[dst_key][agent] = totals[dst_key].get(agent, 0) + int(value)
    return totals


def _normalize_config_overrides(config_overrides):
    if isinstance(config_overrides, tuple) and len(config_overrides) == 1:
        config_overrides = config_overrides[0]
    if config_overrides is None:
        return {}
    if not isinstance(config_overrides, dict):
        raise TypeError(
            "config_overrides must be a dict of EPyMARL CLI overrides; "
            f"got {type(config_overrides).__name__}"
        )
    return {
        key: value
        for key, value in config_overrides.items()
        if key not in UNSUPPORTED_EPYMARL_OVERRIDES
    }


def _series_summary(values) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "final": None,
            "last_100_mean": None,
            "last_100_std": None,
        }
    tail = arr[-min(100, arr.size):]
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "final": float(arr[-1]),
        "last_100_mean": float(tail.mean()),
        "last_100_std": float(tail.std()),
    }


def _metric_values(metrics: dict, key: str) -> list[float]:
    return [float(v) for v in metrics.get(key, {}).get("values", [])]


def _metric_steps(metrics: dict, key: str) -> list[float]:
    return [float(v) for v in metrics.get(key, {}).get("steps", [])]


def _training_return_keys(metrics: dict) -> list[tuple[str, str, str | None]]:
    keys: list[tuple[str, str, str | None]] = []
    for key in sorted(k for k in metrics if re.match(r"^agent_\d+_return_mean$", k)):
        agent_id = re.search(r"\d+", key).group()
        keys.append((f"agent_{agent_id}", key, f"agent_{agent_id}_return_std"))
    if "total_return_mean" in metrics:
        keys.append(("total", "total_return_mean", "total_return_std"))
    if "return_mean" in metrics:
        keys.append(("shared", "return_mean", "return_std"))
    return keys


def _episode_axis(steps, *, n_episodes: int, t_max: int) -> list[float]:
    steps_arr = np.asarray(steps, dtype=np.float64)
    if steps_arr.size == 0:
        return []

    # EPyMARL's logged "episode" counter is runner/batch-oriented for parallel
    # runs, so it can undercount the user-requested scenario episode budget.
    # The training budget we set is t_max = n_episodes * scenario_time_limit;
    # use that invariant for notebook plots.
    return (steps_arr / max(float(t_max) / max(int(n_episodes), 1), 1.0)).tolist()


def _align_curve_to_episode_budget(
    steps,
    values,
    *,
    n_episodes: int,
    t_max: int,
) -> tuple[list[float], list[float], bool]:
    episodes = _episode_axis(steps, n_episodes=n_episodes, t_max=t_max)
    values = list(values)
    appended_final_budget_point = False
    if episodes and values and episodes[-1] < float(n_episodes):
        episodes = [*episodes, float(n_episodes)]
        values = [*values, values[-1]]
        appended_final_budget_point = True
    return episodes, values, appended_final_budget_point


def _save_reward_curve(curves: dict[str, list[float]], out_path: Path, *, title: str) -> None:
    if not curves:
        return

    import matplotlib.pyplot as plt

    episodes = curves.get("episodes")
    if not episodes:
        return

    aggregate_labels = {"joint", "total", "shared"}
    has_agent_curves = any(
        label.startswith("agent_") and values
        for label, values in curves.items()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, values in curves.items():
        if label == "episodes":
            continue
        if has_agent_curves and label in aggregate_labels:
            continue
        ax.plot(episodes, values, linewidth=1.8, marker="", label=label.replace("_", " ").title())
    ax.set_xlabel("Episode")
    ax.set_ylabel("Training reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _build_random_reward_artifacts(stats: dict) -> tuple[dict, dict[str, list[float]]]:
    rewards = np.asarray(stats["rewards"], dtype=np.float64)
    joint = rewards.sum(axis=0)
    curves = {
        "episodes": list(range(1, int(stats["n_episodes"]) + 1)),
        **{f"agent_{i}": rewards[i].tolist() for i in range(rewards.shape[0])},
        "joint": joint.tolist(),
    }
    reward_stats = {
        "algorithm": "random",
        "seed": int(stats["seed"]),
        "t_max": int(stats["t_max"]),
        "n_episodes": int(stats["n_episodes"]),
        "reward_statistics": {
            **{
                f"agent_{i}": _series_summary(rewards[i])
                for i in range(rewards.shape[0])
            },
            "joint": _series_summary(joint),
        },
        "reward_curve": curves,
    }
    return reward_stats, curves


def _build_epymarl_reward_artifacts(
    metrics: dict,
    *,
    algorithm: str,
    seed: int,
    t_max: int,
    n_episodes: int,
) -> tuple[dict, dict[str, list[float]]]:
    curves: dict[str, list[float]] = {}
    reward_statistics: dict[str, dict] = {}

    for label, mean_key, std_key in _training_return_keys(metrics):
        values = _metric_values(metrics, mean_key)
        steps = _metric_steps(metrics, mean_key)
        if not values:
            continue
        episodes, aligned_values, appended_final_budget_point = (
            _align_curve_to_episode_budget(
                steps,
                values,
                n_episodes=n_episodes,
                t_max=t_max,
            )
        )
        if "episodes" not in curves:
            curves["episodes"] = episodes
        curves[label] = aligned_values
        reward_statistics[label] = {
            **_series_summary(aligned_values),
            "logged_steps": [int(step) for step in steps],
            "episode_axis_source": "env_step_scaled_to_requested_episode_budget",
        }
        if appended_final_budget_point:
            reward_statistics[label]["final_budget_point"] = (
                "last_logged_reward_repeated_at_requested_n_episodes"
            )
        std_values = [] if std_key is None else _metric_values(metrics, std_key)
        if std_values:
            reward_statistics[label]["logged_std"] = std_values

    reward_stats = {
        "algorithm": str(algorithm).lower(),
        "seed": int(seed),
        "t_max": int(t_max),
        "n_episodes": int(n_episodes),
        "reward_statistics": reward_statistics,
        "reward_curve": curves,
    }
    return reward_stats, curves


def n_frames_for_episodes(scenario_key: str, n_episodes: int = 1000) -> int:
    """Convert an episode budget to EPyMARL environment timesteps."""
    scenario = get_epymarl_lbf_scenario(scenario_key)
    return int(n_episodes) * int(scenario.time_limit)


def run_random_policy_baseline(
    scenario_key: str,
    *,
    n_episodes: int = 1000,
    seed: int = 2025,
    output_root: str | Path = "lbf_epymarl_baseline_runs",
):
    """Run a random policy directly in Gymnasium LBF for reward-scale baselines."""
    import gymnasium as gym

    register_epymarl_lbf_envs()
    scenario = get_epymarl_lbf_scenario(scenario_key)
    rewards_by_agent = [[] for _ in range(scenario.n_agents)]
    episode_lengths = []
    out_dir = Path(output_root) / scenario.key / "random"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "episode_metrics" / "training"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for path in metrics_dir.glob("*.jsonl"):
        path.unlink()

    old_metrics_env = {
        key: os.environ.get(key)
        for key in (
            "SREDQN_LBF_METRICS_DIR",
            "SREDQN_LBF_METRICS_RUN_ID",
            "SREDQN_LBF_METRICS_PHASE",
        )
    }
    os.environ["SREDQN_LBF_METRICS_DIR"] = str(metrics_dir)
    os.environ["SREDQN_LBF_METRICS_RUN_ID"] = f"{scenario.key}_random_seed{int(seed)}"
    os.environ["SREDQN_LBF_METRICS_PHASE"] = "training"

    try:
        for episode in range(int(n_episodes)):
            env = gym.make(scenario.gym_id, disable_env_checker=True)
            try:
                env.reset(seed=int(seed) + episode)
                done = False
                truncated = False
                ep_rewards = np.zeros(scenario.n_agents, dtype=np.float64)
                ep_steps = 0
                while not (done or truncated):
                    _, rewards, done, truncated, _ = env.step(env.action_space.sample())
                    ep_rewards += np.asarray(rewards, dtype=np.float64)
                    ep_steps += 1
            finally:
                env.close()

            for agent_id, reward in enumerate(ep_rewards):
                rewards_by_agent[agent_id].append(float(reward))
            episode_lengths.append(int(ep_steps))
    finally:
        for key, value in old_metrics_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    stats = {
        "algorithm": "random",
        "scenario_key": scenario.key,
        "scenario_name": scenario.description,
        "gym_id": scenario.gym_id,
        "n_episodes": int(n_episodes),
        "t_max": n_frames_for_episodes(scenario.key, n_episodes),
        "seed": int(seed),
        "rewards": rewards_by_agent,
        "episode_lengths": episode_lengths,
        "mean_joint_reward": float(np.sum(rewards_by_agent, axis=0).mean()),
        "mean_last_100_joint_reward": float(np.sum(rewards_by_agent, axis=0)[-100:].mean()),
    }

    reward_stats, curves = _build_random_reward_artifacts(stats)
    episode_metrics = _load_episode_metrics(metrics_dir)
    episode_metrics_path = out_dir / "training_episode_metrics.json"
    episode_metrics_path.write_text(json.dumps(_json_safe(episode_metrics), indent=2))
    reward_stats["episode_metrics"] = episode_metrics
    reward_stats["episode_metric_totals"] = _episode_metric_totals(episode_metrics)
    reward_stats["episode_metrics_path"] = str(episode_metrics_path)
    stats_path = out_dir / "reward_stats.json"
    curve_path = out_dir / "reward_curve.png"
    stats_path.write_text(json.dumps(_json_safe(reward_stats), indent=2))
    _save_reward_curve(curves, curve_path, title=f"Random - {scenario.key}")
    stats["reward_stats_path"] = str(stats_path)
    stats["reward_curve_path"] = str(curve_path)
    stats["episode_metrics_path"] = str(episode_metrics_path)
    return stats


def build_epymarl_command(
    epymarl_root: str | Path,
    algorithm: str,
    scenario_key: str,
    *,
    n_episodes: int = 1000,
    seed: int = 2025,
    output_root: str | Path = "lbf_epymarl_baseline_runs",
    common_reward: bool = False,
    config_overrides: dict[str, object] | None = None,
    model_token: str | None = None,
) -> list[str]:
    """Build a Python command that registers local LBF IDs then runs EPyMARL."""
    algorithm = str(algorithm).lower()
    if algorithm not in ("iql", "ippo", "mappo", "qmix", "maa2c"):
        raise ValueError(f"Unsupported EPyMARL algorithm: {algorithm}")

    epymarl_root = Path(epymarl_root).expanduser().resolve()
    main_py = epymarl_root / "src" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Could not find EPyMARL main.py at {main_py}")

    scenario = get_epymarl_lbf_scenario(scenario_key)
    t_max = n_frames_for_episodes(scenario_key, n_episodes)
    result_root = Path(output_root).expanduser().resolve()
    run_name = f"{scenario.key}_{algorithm}_seed{int(seed)}"

    overrides = [
        f"--config={algorithm}",
        "--env-config=gymma",
        "with",
        f"env_args.time_limit={scenario.time_limit}",
        f"env_args.key={scenario.gym_id}",
        f"t_max={t_max}",
        f"seed={int(seed)}",
        f"name={run_name}",
        f"common_reward={str(bool(common_reward))}",
        f"local_results_path={str(result_root)}",
    ]
    config_overrides = _normalize_config_overrides(config_overrides)
    for key, value in config_overrides.items():
        overrides.append(f"{key}={_format_epymarl_override_value(value)}")

    model_token = "" if model_token is None else str(model_token)
    bootstrap = (
        "import runpy, sys, types; "
        "from pathlib import Path; "
        f"epymarl_root = Path({str(epymarl_root)!r}); "
        "sys.path.insert(0, str(epymarl_root / 'src')); "
        f"model_token = {model_token!r}; "
        "\nif model_token:\n"
        "    run_path = epymarl_root / 'src' / 'run.py'\n"
        "    run_source = run_path.read_text()\n"
        "    old = \"unique_token = (\\n        f\\\"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}\\\"\\n    )\"\n"
        "    new = \"unique_token = \" + repr(model_token)\n"
        "    if old not in run_source:\n"
        "        raise RuntimeError('Could not patch EPyMARL unique_token assignment')\n"
        "    run_source = run_source.replace(old, new)\n"
        "    run_module = types.ModuleType('run')\n"
        "    run_module.__file__ = str(run_path)\n"
        "    sys.modules['run'] = run_module\n"
        "    exec(compile(run_source, str(run_path), 'exec'), run_module.__dict__)\n"
        "from discrete_action_space.lbf_grid.epymarl_lbf_env import "
        "register_epymarl_lbf_envs; "
        "register_epymarl_lbf_envs(); "
        f"sys.argv = ['main.py'] + {overrides!r}; "
        f"runpy.run_path({str(main_py)!r}, run_name='__main__')"
    )
    return [sys.executable, "-c", bootstrap]


def run_epymarl_baseline(
    epymarl_root: str | Path,
    algorithm: str,
    scenario_key: str,
    *,
    n_episodes: int = 1000,
    seed: int = 2025,
    output_root: str | Path = "lbf_epymarl_baseline_runs",
    common_reward: bool = False,
    config_overrides: dict[str, object] | None = None,
    check: bool = True,
    show_progress: bool = False,
):
    """Run one baseline (random or EPyMARL) for a single scenario."""
    algorithm = str(algorithm).lower()
    if str(algorithm).lower() == "random":
        return run_random_policy_baseline(
            scenario_key,
            n_episodes=n_episodes,
            seed=seed,
            output_root=output_root,
        )

    command = build_epymarl_command(
        epymarl_root,
        algorithm,
        scenario_key,
        n_episodes=n_episodes,
        seed=seed,
        output_root=output_root,
        common_reward=common_reward,
        config_overrides=config_overrides,
        model_token=_epymarl_model_token(scenario_key, seed, algorithm),
    )

    repo_root = Path(__file__).resolve().parents[2]
    epymarl_root = Path(epymarl_root).expanduser().resolve()
    env = os.environ.copy()
    pythonpath = [str(repo_root), str(epymarl_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    scenario = get_epymarl_lbf_scenario(scenario_key)
    run_dir = Path(output_root) / scenario.key / algorithm
    run_dir.mkdir(parents=True, exist_ok=True)
    model_root = (
        Path(output_root).expanduser().resolve()
        / "models"
        / scenario.key
        / str(int(seed))
        / algorithm
    )
    if model_root.exists():
        shutil.rmtree(model_root)
    metrics_dir = run_dir / "episode_metrics" / "training"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for path in metrics_dir.glob("*.jsonl"):
        path.unlink()
    env["SREDQN_LBF_METRICS_DIR"] = str(metrics_dir)
    env["SREDQN_LBF_METRICS_RUN_ID"] = f"{scenario.key}_{algorithm}_seed{int(seed)}"
    env["SREDQN_LBF_METRICS_PHASE"] = "training"
    t_max = n_frames_for_episodes(scenario_key, n_episodes)
    sacred_base = epymarl_root / "results" / "sacred" / algorithm / scenario.gym_id

    start_time = time.time()
    proc = subprocess.Popen(
        command,
        cwd=str(epymarl_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Drain stdout/stderr in background threads to avoid pipe-buffer deadlock.
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _drain(stream, buf: list[str]) -> None:
        buf.append(stream.read())

    t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_chunks), daemon=True)
    t_out.start()
    t_err.start()

    if show_progress:
        # Sacred redirects the subprocess's FDs to cout.txt almost immediately.
        # Poll that file for "t_env: <N>" lines to drive a step-level progress bar.
        from tqdm.auto import tqdm as _tqdm

        bar = _tqdm(
            total=t_max,
            unit="step",
            desc=f"{str(algorithm).upper()} {scenario.key}",
        )
        last_t = 0
        cout_path: Path | None = None
        find_deadline = start_time + 30  # give Sacred 30 s to create the file

        try:
            while proc.poll() is None:
                if cout_path is None and time.time() < find_deadline and sacred_base.exists():
                    candidates = [
                        p for p in sacred_base.glob("*/cout.txt")
                        if p.stat().st_mtime >= start_time
                    ]
                    if candidates:
                        cout_path = max(candidates, key=lambda p: p.stat().st_mtime)

                if cout_path is not None and cout_path.exists():
                    matches = re.findall(r"t_env[:\s]+(\d+)", cout_path.read_text())
                    if matches:
                        t = min(int(matches[-1]), t_max)
                        if t > last_t:
                            bar.update(t - last_t)
                            last_t = t

                time.sleep(0.5)
        finally:
            bar.update(max(0, t_max - last_t))
            bar.close()
    else:
        proc.wait()

    t_out.join()
    t_err.join()

    # Locate the Sacred metrics file written during this run.
    sacred_metrics_path = None
    if sacred_base.exists():
        candidates = [
            p for p in sacred_base.glob("*/metrics.json")
            if p.stat().st_mtime >= start_time
        ]
        if candidates:
            sacred_metrics_path = str(max(candidates, key=lambda p: p.stat().st_mtime))

    reward_stats_path = run_dir / "reward_stats.json"
    reward_curve_path = run_dir / "reward_curve.png"

    if sacred_metrics_path and Path(sacred_metrics_path).exists():
        metrics = json.loads(Path(sacred_metrics_path).read_text())
        reward_stats, curves = _build_epymarl_reward_artifacts(
            metrics,
            algorithm=algorithm,
            seed=seed,
            t_max=t_max,
            n_episodes=n_episodes,
        )
    else:
        reward_stats = {
            "algorithm": algorithm,
            "seed": int(seed),
            "t_max": int(t_max),
            "n_episodes": int(n_episodes),
            "reward_statistics": {},
            "reward_curve": {},
        }
        curves = {}

    episode_metrics = _load_episode_metrics(metrics_dir)
    episode_metrics_path = run_dir / "training_episode_metrics.json"
    episode_metrics_path.write_text(json.dumps(_json_safe(episode_metrics), indent=2))
    reward_stats["episode_metrics"] = episode_metrics
    reward_stats["episode_metric_totals"] = _episode_metric_totals(episode_metrics)
    reward_stats["episode_metrics_path"] = str(episode_metrics_path)

    reward_stats_path.write_text(json.dumps(_json_safe(reward_stats), indent=2))
    _save_reward_curve(
        curves,
        reward_curve_path,
        title=f"{str(algorithm).upper()} - {scenario.key}",
    )

    record = {
        "algorithm": algorithm,
        "scenario_key": scenario.key,
        "scenario_name": scenario.description,
        "gym_id": scenario.gym_id,
        "n_episodes": int(n_episodes),
        "t_max": t_max,
        "seed": int(seed),
        "returncode": int(proc.returncode),
        "config_overrides": _normalize_config_overrides(config_overrides),
        "reward_stats_path": str(reward_stats_path),
        "reward_curve_path": str(reward_curve_path),
        "episode_metrics_path": str(episode_metrics_path),
        "model_root": str(model_root),
        "sacred_metrics_path": sacred_metrics_path,
    }
    if check and proc.returncode != 0:
        stderr_tail = "".join(stderr_chunks)[-4000:]
        raise RuntimeError(
            f"EPyMARL {algorithm} on {scenario.key} failed with code "
            f"{proc.returncode}.\n{stderr_tail}"
        )
    return record
