"""Notebook helpers for LBF rollout evaluation and reward plots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


def central_state(obs_dict, agent_order):
    return np.concatenate(
        [np.asarray(obs_dict[agent], dtype=np.float32).reshape(-1) for agent in agent_order]
    ).astype(np.float32, copy=False)


def sample_lbf_rollout(
    *,
    make_env: Callable,
    policy_fn: Callable,
    seed: int = 0,
    max_steps: int | None = None,
):
    """Run one evaluation episode, returning frames and reward diagnostics."""
    env = make_env()
    frames = []
    actions = []
    render_failed = False

    def _try_render():
        nonlocal render_failed
        if render_failed:
            return None
        try:
            return env.render()
        except Exception as exc:  # pragma: no cover - depends on local display backend
            render_failed = True
            print(f"[rollout render disabled: {type(exc).__name__}: {exc}]")
            return None

    try:
        obs_dict, _ = env.reset(seed=seed)
        agent_order = list(env.possible_agents)
        total_rewards = np.zeros(len(agent_order), dtype=np.float64)
        step_rewards = []
        steps = 0

        frame = _try_render()
        if frame is not None:
            frames.append(np.asarray(frame))

        while env.agents and (max_steps is None or steps < int(max_steps)):
            state = central_state(obs_dict, agent_order)
            action_list = policy_fn(
                state=state,
                obs_dict=obs_dict,
                agent_order=agent_order,
                env=env,
                step=steps,
            )
            action_list = [int(action) for action in action_list]
            action_dict = {
                agent: action_list[index] for index, agent in enumerate(agent_order)
            }
            obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)
            rewards = np.asarray(
                [reward_dict.get(agent, 0.0) for agent in agent_order],
                dtype=np.float64,
            )
            total_rewards += rewards
            step_rewards.append(rewards.tolist())
            actions.append(action_list)
            steps += 1

            frame = _try_render()
            if frame is not None:
                frames.append(np.asarray(frame))

            if all(
                bool(term_dict.get(agent, False)) or bool(trunc_dict.get(agent, False))
                for agent in agent_order
            ):
                break
    finally:
        env.close()

    return {
        "frames": frames,
        "actions": actions,
        "step_rewards": step_rewards,
        "total_rewards": total_rewards.tolist(),
        "joint_reward": float(total_rewards.sum()),
        "steps": int(steps),
    }


def rollout_video_html(frames, *, fps: int = 4, title: str = "Evaluation rollout"):
    """Create an IPython HTML animation for captured rgb_array frames."""
    if not frames:
        raise ValueError("No rollout frames were captured.")

    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML

    interval_ms = int(1000 / max(1, int(fps)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    image = ax.imshow(frames[0])
    ax.set_title(title)

    def _update(frame):
        image.set_data(frame)
        return (image,)

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=interval_ms,
        blit=True,
    )
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


def _record_label(record: dict) -> str:
    return (
        record.get("run_key")
        or "__".join(
            part
            for part in [
                record.get("scenario_key"),
                record.get("ablation_variant") or record.get("algorithm"),
            ]
            if part
        )
        or str(record.get("algorithm", "run"))
    )


def _load_reward_payload(record: dict) -> dict:
    stats_path = record.get("reward_stats_path") or record.get("stats_path")
    if stats_path and Path(stats_path).exists():
        try:
            return json.loads(Path(stats_path).read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def reward_series(record: dict) -> np.ndarray:
    """Return a joint/aggregate reward series for either repo stats format."""
    rewards = record.get("rewards")
    if rewards is not None:
        arr = np.asarray(rewards, dtype=np.float64)
        if arr.ndim == 2 and arr.size:
            return arr.sum(axis=0)
        if arr.ndim == 1:
            return arr

    payload = _load_reward_payload(record)
    curves = payload.get("reward_curve", {})
    for key in ("joint", "total", "shared"):
        values = curves.get(key)
        if values:
            return np.asarray(values, dtype=np.float64)
    agent_curves = [
        np.asarray(values, dtype=np.float64)
        for label, values in curves.items()
        if label.startswith("agent_") and values
    ]
    if agent_curves:
        length = min(series.size for series in agent_curves)
        return np.stack([series[:length] for series in agent_curves], axis=0).sum(axis=0)
    return np.asarray([], dtype=np.float64)


def reward_summary_rows(records: Iterable[dict], *, tail: int = 100) -> list[dict]:
    rows = []
    for record in records:
        values = reward_series(record)
        if values.size == 0:
            continue
        tail_values = values[-min(int(tail), values.size):]
        rows.append(
            {
                "run": _record_label(record),
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "final": float(values[-1]),
                f"last_{int(tail)}_mean": float(tail_values.mean()),
            }
        )
    return rows


def plot_reward_statistics(records: Iterable[dict], *, title: str, tail: int = 100):
    """Plot aggregate reward distributions and tail means for notebook display."""
    import matplotlib.pyplot as plt

    prepared = [
        (_record_label(record), reward_series(record))
        for record in records
    ]
    prepared = [(label, values) for label, values in prepared if values.size]
    if not prepared:
        print("[no reward series available for plotting]")
        return None

    labels = [label for label, _ in prepared]
    series = [values for _, values in prepared]
    tail_means = [
        float(values[-min(int(tail), values.size):].mean()) for values in series
    ]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.6 * len(labels)), 4.5))
    axes[0].boxplot(series, labels=labels, showmeans=True)
    axes[0].set_title("Joint reward distribution")
    axes[0].set_ylabel("Joint reward")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, tail_means)
    axes[1].set_title(f"Mean joint reward over last {int(tail)} points")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
