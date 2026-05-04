from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import HTML, display
from matplotlib import animation
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig


REWARD_METRIC_PATTERN = re.compile(
    r"^(collection|eval)_(adversary_reward|agent_reward|reward)_episode_reward_(min|mean|max)$"
)


def make_run_root(base_dir: Path, algorithm_label: str) -> Path:
    run_root = (base_dir / "benchmarl_runs" / f"vmas_simple_tag_{algorithm_label}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def configure_simple_tag_task(
    max_steps: int,
    num_good_agents: int,
    num_adversaries: int,
    num_landmarks: int,
    **task_overrides: Any,
):
    task = VmasTask.SIMPLE_TAG.get_from_yaml()
    task.config.update(
        max_steps=max_steps,
        num_good_agents=num_good_agents,
        num_adversaries=num_adversaries,
        num_landmarks=num_landmarks,
        **task_overrides,
    )
    return task


def build_off_policy_experiment_config(
    *,
    device: str,
    share_policy_params: bool,
    continuous_actions: bool,
    lr: float,
    max_n_iters: int,
    num_envs: int,
    frames_per_batch: int,
    train_batch_size: int,
    optimizer_steps: int,
    replay_buffer_size: int,
    evaluation_interval: int,
    evaluation_episodes: int,
    project_name: str,
    run_root: Path,
    checkpoint_interval: int,
    gamma: float = 0.99,
    exploration_eps_init: float = 0.8,
    exploration_eps_end: float = 0.05,
    soft_target_update: bool = False,
    polyak_tau: float | None = None,
    hard_target_update_frequency: int | None = None,
    render: bool = False,
):
    config = ExperimentConfig.get_from_yaml()
    config.sampling_device = device
    config.train_device = device
    config.buffer_device = device
    config.share_policy_params = share_policy_params
    config.prefer_continuous_actions = continuous_actions
    config.gamma = gamma
    config.lr = lr
    config.soft_target_update = soft_target_update
    if soft_target_update and polyak_tau is not None:
        config.polyak_tau = polyak_tau
    if not soft_target_update and hard_target_update_frequency is not None:
        config.hard_target_update_frequency = hard_target_update_frequency
    config.exploration_eps_init = exploration_eps_init
    config.exploration_eps_end = exploration_eps_end
    config.max_n_iters = max_n_iters
    config.off_policy_n_envs_per_worker = num_envs
    config.off_policy_collected_frames_per_batch = frames_per_batch
    config.off_policy_train_batch_size = train_batch_size
    config.off_policy_n_optimizer_steps = optimizer_steps
    config.off_policy_memory_size = replay_buffer_size
    config.evaluation = True
    config.render = render
    config.evaluation_interval = evaluation_interval
    config.evaluation_episodes = evaluation_episodes
    config.loggers = ["csv"]
    config.project_name = project_name
    config.create_json = True
    config.save_folder = str(run_root)
    config.checkpoint_interval = checkpoint_interval
    config.checkpoint_at_end = True
    return config


def print_run_settings(title: str, run_root: Path, settings: dict[str, Any]) -> None:
    print(f"{title} run root:", run_root)
    print(f"{title} settings:")
    for key, value in settings.items():
        print(f"  {key}={value}")


def find_latest_checkpoint(root: Path):
    candidates = sorted(
        root.rglob("checkpoint_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def plot_reward_metrics_for_run(
    run_root: Path,
    algorithm_label: str,
    *,
    save: bool = True,
):
    csv_files = sorted(run_root.rglob("*.csv"), key=lambda path: path.stat().st_mtime)
    print(f"CSV files found for {algorithm_label}: {len(csv_files)}")

    metrics = {}
    for path in csv_files:
        match = REWARD_METRIC_PATTERN.match(path.stem)
        if not match:
            continue
        phase, group, stat = match.groups()
        df = pd.read_csv(path, header=None, names=["step", "value"])
        if df.empty:
            continue
        metrics.setdefault((phase, group), {})[stat] = df

    if not metrics:
        print(f"No reward metrics found for {algorithm_label}.")
        return []

    save_dir = run_root / "figures"
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for phase in sorted({phase for phase, _ in metrics}):
        present_groups = [
            group
            for group in ("agent_reward", "adversary_reward")
            if (phase, group) in metrics
        ]
        if not present_groups:
            continue

        fig, axes = plt.subplots(
            1,
            len(present_groups),
            figsize=(6 * len(present_groups), 4),
            squeeze=False,
        )
        for ax, group in zip(axes[0], present_groups):
            for stat in ("min", "mean", "max"):
                df = metrics[(phase, group)].get(stat)
                if df is not None:
                    ax.plot(df["step"], df["value"], label=stat)
            ax.set_title(f"{phase}: {group}")
            ax.set_xlabel("step")
            ax.set_ylabel("episode reward")
            ax.grid(alpha=0.3)
            ax.legend()

        fig.suptitle(
            f"VMAS Simple Tag - {algorithm_label.upper().replace('_', '-')} ({phase})",
            y=1.03,
        )
        plt.tight_layout()

        if save:
            out_path = save_dir / f"{algorithm_label}_{phase}_reward_curves.png"
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            saved_paths.append(out_path)
            print("Saved figure to:", out_path)
        plt.show()
        plt.close(fig)

    return saved_paths


def to_hwc_uint8(frame):
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def collect_eval_rollouts(exp, episodes: int, max_steps: int):
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        if exp.test_env.batch_size == ():
            return [
                exp.test_env.rollout(
                    max_steps=max_steps,
                    policy=exp.policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                for _ in range(episodes)
            ]

        rollout_td = exp.test_env.rollout(
            max_steps=max_steps,
            policy=exp.policy,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )
        return list(rollout_td.unbind(0))


def summarize_simple_tag_rollouts(rollouts, max_steps: int):
    episode_rows = []
    for episode_idx, rollout in enumerate(rollouts):
        done = rollout.get(("next", "done")).reshape(rollout.shape[0], -1).any(-1)
        done_idx = torch.nonzero(done, as_tuple=False)
        episode_length = int(done_idx[0].item() + 1) if done_idx.numel() else max_steps
        adv_obs = rollout.get(("next", "adversary", "observation"))
        prey_rel = adv_obs[..., 12:14]
        predator_prey_dist = torch.linalg.vector_norm(prey_rel, dim=-1)

        episode_rows.append(
            {
                "episode": episode_idx,
                "captured": float(episode_length < max_steps),
                "episode_length": episode_length,
                "prey_survival_time": episode_length,
                "min_predator_prey_distance": float(predator_prey_dist.min().item()),
                "mean_min_predator_prey_distance": float(
                    predator_prey_dist.min(dim=-1).values.mean().item()
                ),
            }
        )

    episode_df = pd.DataFrame(episode_rows)
    summary = {
        "capture_rate": float(episode_df["captured"].mean()),
        "mean_episode_length": float(episode_df["episode_length"].mean()),
        "mean_prey_survival_time": float(episode_df["prey_survival_time"].mean()),
        "mean_min_predator_prey_distance": float(
            episode_df["min_predator_prey_distance"].mean()
        ),
    }
    return episode_df, summary


def save_simple_tag_eval_diagnostics(
    run_root: Path,
    algorithm_label: str,
    episode_df: pd.DataFrame,
    summary: dict[str, float],
):
    metrics_dir = run_root / "metrics"
    figures_dir = run_root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / f"{algorithm_label}_eval_episode_metrics.csv"
    json_path = metrics_dir / f"{algorithm_label}_eval_summary.json"
    fig_path = figures_dir / f"{algorithm_label}_eval_diagnostics.png"

    episode_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(
        episode_df["episode_length"],
        bins=min(len(episode_df), 10),
        color="#3a6ea5",
        alpha=0.85,
    )
    axes[0].set_title("Episode Length")
    axes[0].set_xlabel("steps")
    axes[0].set_ylabel("count")

    axes[1].hist(
        episode_df["min_predator_prey_distance"],
        bins=min(len(episode_df), 10),
        color="#c84c31",
        alpha=0.85,
    )
    axes[1].set_title("Min Predator-Prey Distance")
    axes[1].set_xlabel("distance")

    summary_items = [
        ("capture_rate", summary["capture_rate"]),
        ("mean_episode_length", summary["mean_episode_length"]),
        ("prey_survival_time", summary["mean_prey_survival_time"]),
    ]
    axes[2].bar(
        [item[0] for item in summary_items],
        [item[1] for item in summary_items],
        color=["#4c956c", "#6c5b7b", "#f4a259"],
    )
    axes[2].set_title("Evaluation Summary")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle(f"{algorithm_label.upper()} Simple Tag Evaluation Diagnostics")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("Saved diagnostics CSV to:", csv_path)
    print("Saved diagnostics JSON to:", json_path)
    print("Saved diagnostics figure to:", fig_path)
    return csv_path, json_path, fig_path


def render_latest_policy_video(
    run_root: Path,
    algorithm_label: str,
    max_steps: int,
    *,
    title_label: str | None = None,
):
    latest_ckpt = find_latest_checkpoint(run_root)
    if latest_ckpt is None:
        raise RuntimeError(f"No checkpoint found to render for {algorithm_label}.")

    viz_exp = Experiment.reload_from_file(
        str(latest_ckpt),
        experiment_patch={"render": False, "evaluation": False},
    )

    frames = []
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        try:
            viz_exp.test_env.rollout(
                max_steps=max_steps,
                policy=viz_exp.policy,
                auto_cast_to_device=True,
                break_when_any_done=False,
                callback=lambda env, td: frames.append(
                    viz_exp.task.render_callback(viz_exp, env, td)
                ),
            )
        except ImportError as err:
            print("Rendering needs OpenGL/GLU on this machine.")
            print("On Ubuntu, install: sudo apt-get install python3-opengl libglu1-mesa xvfb")
            print(f"Original error: {err}")
            return None

    if not frames:
        print(f"No frames captured for {algorithm_label}.")
        return None

    print(f"Captured {len(frames)} frames.")
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(to_hwc_uint8(frames[0]))
    ax.set_title(f"Simple Tag - {(title_label or algorithm_label).upper().replace('_', '-')}")
    ax.axis("off")

    ani = animation.FuncAnimation(
        fig,
        lambda i: [im.set_data(to_hwc_uint8(frames[i])) or im],
        frames=len(frames),
        interval=60,
        blit=True,
    )

    out_dir = run_root / "manual_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"{algorithm_label}_eval.mp4"
    gif_path = out_dir / f"{algorithm_label}_eval.gif"

    try:
        ani.save(mp4_path, writer="ffmpeg", fps=20, dpi=120)
        saved_path = mp4_path
    except Exception as err:
        print("MP4 save failed, falling back to GIF:", err)
        ani.save(gif_path, writer="pillow", fps=20)
        saved_path = gif_path

    print("Saved video to:", saved_path)
    plt.close(fig)
    display(HTML(ani.to_jshtml()))
    return saved_path
