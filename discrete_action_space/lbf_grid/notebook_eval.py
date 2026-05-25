"""Notebook helpers for LBF rollout evaluation and reward plots."""
from __future__ import annotations

import json
import inspect
from importlib import resources
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import matplotlib.pyplot as plt

try:
    from .instrumented_env import aggregate_lbf_episode_metrics, extract_lbf_metrics
except ImportError:  # Script/notebook import from the lbf_grid directory
    from instrumented_env import aggregate_lbf_episode_metrics, extract_lbf_metrics

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional in plain scripts.
    tqdm = None


def central_state(obs_dict, agent_order):
    return np.concatenate(
        [np.asarray(obs_dict[agent], dtype=np.float32).reshape(-1) for agent in agent_order]
    ).astype(np.float32, copy=False)


def global_state(env, obs_dict, agent_order):
    if hasattr(env, "global_state"):
        return env.global_state(agent_order)
    inner = getattr(env, "_inner", env)
    if getattr(inner, "field", None) is not None and getattr(inner, "players", None) is not None:
        try:
            from .state_action_encoding import canonical_lbf_state
        except ImportError:
            from state_action_encoding import canonical_lbf_state

        return canonical_lbf_state(env, agent_order)
    return central_state(obs_dict, agent_order)


def action_masks(env, agent_order):
    if hasattr(env, "action_masks"):
        return env.action_masks(agent_order)
    inner = getattr(env, "_inner", env)
    if getattr(inner, "field", None) is not None and getattr(inner, "players", None) is not None:
        try:
            from .state_action_encoding import lbf_action_masks
        except ImportError:
            from state_action_encoding import lbf_action_masks

        return lbf_action_masks(env, agent_order)
    return None


def _unwrap_lbf_env(env):
    current = env
    seen = set()
    for _ in range(12):
        if current is None:
            return None
        ident = id(current)
        if ident in seen:
            return None
        seen.add(ident)
        if getattr(current, "field", None) is not None and getattr(current, "players", None) is not None:
            return current
        for attr in ("_inner", "_env", "unwrapped", "env"):
            try:
                child = getattr(current, attr, None)
            except Exception:
                child = None
            if child is not None and child is not current:
                current = child
                break
        else:
            return None
    return None


def _load_lbf_icon(name: str, size: int):
    from PIL import Image

    try:
        icon_path = resources.files("lbforaging.foraging").joinpath("icons", name)
        image = Image.open(icon_path)
    except Exception:
        return None
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.convert("RGBA").resize((size, size), resampling)


def _draw_badge(draw, *, center_x: float, center_y: float, radius: float, text: str):
    from PIL import ImageFont

    x0 = center_x - radius
    y0 = center_y - radius
    x1 = center_x + radius
    y1 = center_y + radius
    draw.ellipse((x0, y0, x1, y1), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(10, int(radius * 1.25)))
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        (center_x - text_w / 2, center_y - text_h / 2 - 1),
        text,
        fill=(0, 0, 0),
        font=font,
    )


def lbf_state_frame(env, *, grid_size: int = 50):
    """Render an LBF grid frame from environment state without opening a GUI."""
    from PIL import Image, ImageDraw

    inner = _unwrap_lbf_env(env)
    if inner is None:
        return None
    field = getattr(inner, "field", None)
    players = getattr(inner, "players", None)
    if field is None or players is None:
        return None

    field = np.asarray(field)
    if field.ndim != 2:
        return None
    rows, cols = field.shape
    stride = grid_size + 1
    width = 1 + cols * stride
    height = 1 + rows * stride
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for row in range(rows + 1):
        y = row * stride + 1
        draw.line((0, y, stride * cols, y), fill=(0, 0, 0), width=1)
    for col in range(cols + 1):
        x = col * stride + 1
        draw.line((x, 0, x, stride * rows), fill=(0, 0, 0), width=1)

    apple = _load_lbf_icon("apple.png", grid_size)
    agent_icon = _load_lbf_icon("agent.png", grid_size)

    food_rows, food_cols = np.nonzero(field > 0)
    for row, col in zip(food_rows, food_cols):
        x = int(col) * stride
        y = int(row) * stride
        if apple is not None:
            image.paste(apple, (x, y), apple)
        else:
            draw.ellipse((x + 6, y + 6, x + grid_size - 6, y + grid_size - 6), fill=(255, 0, 0))
        _draw_badge(
            draw,
            center_x=x + 0.75 * stride,
            center_y=y + 0.75 * stride,
            radius=grid_size / 5,
            text=str(field[row, col]),
        )

    for player in players:
        position = getattr(player, "position", None)
        if position is None:
            continue
        row, col = position
        if 0 <= int(row) < rows and 0 <= int(col) < cols:
            x = int(col) * stride
            y = int(row) * stride
            if agent_icon is not None:
                image.paste(agent_icon, (x, y), agent_icon)
            else:
                draw.ellipse((x + 10, y + 8, x + grid_size - 10, y + grid_size - 2), fill=(0, 0, 0))
            _draw_badge(
                draw,
                center_x=x + 0.75 * stride,
                center_y=y + 0.75 * stride,
                radius=grid_size / 5,
                text=str(getattr(player, "level", "")),
            )

    return np.asarray(image, dtype=np.uint8)


def sample_lbf_rollout(
    *,
    make_env: Callable,
    policy_fn: Callable,
    seed: int = 0,
    max_steps: int | None = None,
    capture_frames: bool = True,
):
    """Run one evaluation episode, returning frames and reward diagnostics."""
    env = make_env()
    frames = []
    actions = []
    render_failed = False
    render_error = None

    def _try_render():
        nonlocal render_failed, render_error
        if not capture_frames:
            return None
        if render_failed:
            return None
        frame = lbf_state_frame(env)
        if frame is not None:
            return frame
        try:  # Last-resort array render for non-standard envs; never requests human mode.
            return env.render(mode="rgb_array")
        except Exception as exc:  # pragma: no cover - depends on local render backend
            render_failed = True
            render_error = f"{type(exc).__name__}: {exc}"
            print(f"[rollout render disabled: {render_error}]")
            return None

    try:
        obs_dict, reset_info = env.reset(seed=seed)
        agent_order = list(env.possible_agents)
        total_rewards = np.zeros(len(agent_order), dtype=np.float64)
        step_rewards = []
        steps = 0
        episode_metrics = extract_lbf_metrics(reset_info)

        frame = _try_render()
        if frame is not None:
            frames.append(np.asarray(frame))

        while env.agents and (max_steps is None or steps < int(max_steps)):
            state = global_state(env, obs_dict, agent_order)
            action_list = policy_fn(
                state=state,
                obs_dict=obs_dict,
                agent_order=agent_order,
                env=env,
                step=steps,
                action_masks=action_masks(env, agent_order),
            )
            action_list = [int(action) for action in action_list]
            action_dict = {
                agent: action_list[index] for index, agent in enumerate(agent_order)
            }
            obs_dict, reward_dict, term_dict, trunc_dict, step_info = env.step(action_dict)
            episode_metrics = extract_lbf_metrics(step_info) or episode_metrics
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
        "episode_metrics": episode_metrics,
        "render_error": render_error,
    }


def _make_episode_env(make_env: Callable, capture_frames: bool, accepts_capture: bool):
    if accepts_capture:
        return make_env(capture_frames=capture_frames)
    return make_env()


def _make_env_accepts_capture_arg(make_env: Callable) -> bool:
    try:
        make_env_signature = inspect.signature(make_env)
    except (TypeError, ValueError):
        return False
    return (
        "capture_frames" in make_env_signature.parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in make_env_signature.parameters.values()
        )
    )


def sample_lbf_rollouts_vectorized(
    *,
    make_env: Callable,
    policy_batch_fn: Callable,
    policy_fn: Callable | None = None,
    seed: int = 0,
    n_episodes: int = 1,
    max_steps: int | None = None,
    num_envs: int = 16,
    progress_label: str | None = None,
    show_progress: bool = True,
    capture_first_episode_frames: bool = True,
):
    """Run multiple LBF evaluation episodes with several live envs at once."""
    episode_count = max(1, int(n_episodes))
    num_envs = max(1, min(int(num_envs), episode_count))
    make_env_accepts_capture = _make_env_accepts_capture_arg(make_env)
    rollouts = [None] * episode_count
    active = []
    next_episode_idx = 0

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(
            total=episode_count,
            desc=progress_label or "LBF evaluation",
            unit="episode",
            leave=True,
        )

    def start_episode(episode_idx):
        capture_frames = bool(capture_first_episode_frames and episode_idx == 0)
        env = _make_episode_env(make_env, capture_frames, make_env_accepts_capture)
        obs_dict, reset_info = env.reset(seed=int(seed) + episode_idx)
        agent_order = list(env.possible_agents)
        slot = {
            "episode_idx": int(episode_idx),
            "env": env,
            "obs_dict": obs_dict,
            "agent_order": agent_order,
            "total_rewards": np.zeros(len(agent_order), dtype=np.float64),
            "step_rewards": [],
            "actions": [],
            "steps": 0,
            "frames": [],
            "episode_metrics": extract_lbf_metrics(reset_info),
            "capture_frames": capture_frames,
            "render_failed": False,
            "render_error": None,
        }
        if capture_frames:
            frame = lbf_state_frame(env)
            if frame is not None:
                slot["frames"].append(np.asarray(frame))
            else:
                try:
                    frame = env.render(mode="rgb_array")
                except Exception as exc:  # pragma: no cover - render backend specific
                    slot["render_failed"] = True
                    slot["render_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"[rollout render disabled: {slot['render_error']}]")
                else:
                    if frame is not None:
                        slot["frames"].append(np.asarray(frame))
        return slot

    try:
        while next_episode_idx < episode_count and len(active) < num_envs:
            active.append(start_episode(next_episode_idx))
            next_episode_idx += 1

        while active:
            contexts = []
            context_slots = []
            finished_indices = []
            for slot_index, slot in enumerate(active):
                env = slot["env"]
                if not env.agents or (
                    max_steps is not None and slot["steps"] >= int(max_steps)
                ):
                    finished_indices.append(slot_index)
                    continue
                state = global_state(env, slot["obs_dict"], slot["agent_order"])
                contexts.append(
                    {
                        "state": state,
                        "obs_dict": slot["obs_dict"],
                        "agent_order": slot["agent_order"],
                        "env": env,
                        "step": slot["steps"],
                        "episode_idx": slot["episode_idx"],
                        "action_masks": action_masks(env, slot["agent_order"]),
                    }
                )
                context_slots.append(slot_index)

            if contexts:
                if policy_batch_fn is not None:
                    batch_actions = policy_batch_fn(contexts)
                elif policy_fn is not None:
                    batch_actions = [policy_fn(**context) for context in contexts]
                else:
                    raise ValueError("Either policy_batch_fn or policy_fn is required.")

                for slot_index, action_list in zip(context_slots, batch_actions):
                    slot = active[slot_index]
                    env = slot["env"]
                    agent_order = slot["agent_order"]
                    action_list = [int(action) for action in action_list]
                    action_dict = {
                        agent: action_list[index]
                        for index, agent in enumerate(agent_order)
                    }
                    obs_dict, reward_dict, term_dict, trunc_dict, step_info = env.step(action_dict)
                    slot["episode_metrics"] = (
                        extract_lbf_metrics(step_info) or slot["episode_metrics"]
                    )
                    rewards = np.asarray(
                        [reward_dict.get(agent, 0.0) for agent in agent_order],
                        dtype=np.float64,
                    )
                    slot["obs_dict"] = obs_dict
                    slot["total_rewards"] += rewards
                    slot["step_rewards"].append(rewards.tolist())
                    slot["actions"].append(action_list)
                    slot["steps"] += 1

                    if slot["capture_frames"] and not slot["render_failed"]:
                        frame = lbf_state_frame(env)
                        if frame is None:
                            try:
                                frame = env.render(mode="rgb_array")
                            except Exception as exc:  # pragma: no cover
                                slot["render_failed"] = True
                                slot["render_error"] = f"{type(exc).__name__}: {exc}"
                                print(f"[rollout render disabled: {slot['render_error']}]")
                        if frame is not None:
                            slot["frames"].append(np.asarray(frame))

                    if all(
                        bool(term_dict.get(agent, False))
                        or bool(trunc_dict.get(agent, False))
                        for agent in agent_order
                    ):
                        finished_indices.append(slot_index)

            for slot_index in sorted(set(finished_indices), reverse=True):
                slot = active.pop(slot_index)
                slot["env"].close()
                episode_idx = slot["episode_idx"]
                rollouts[episode_idx] = {
                    "frames": slot["frames"],
                    "actions": slot["actions"],
                    "step_rewards": slot["step_rewards"],
                    "total_rewards": slot["total_rewards"].tolist(),
                    "joint_reward": float(slot["total_rewards"].sum()),
                    "steps": int(slot["steps"]),
                    "episode_metrics": slot["episode_metrics"],
                    "render_error": slot["render_error"],
                }
                if progress is not None:
                    progress.update(1)
                while next_episode_idx < episode_count and len(active) < num_envs:
                    active.append(start_episode(next_episode_idx))
                    next_episode_idx += 1
    finally:
        for slot in active:
            slot["env"].close()
        if progress is not None:
            progress.close()

    rollouts = [rollout for rollout in rollouts if rollout is not None]
    return {
        "rollouts": rollouts,
        "frames": rollouts[0]["frames"] if rollouts else [],
        "render_error": rollouts[0].get("render_error") if rollouts else None,
        "episode_rewards": [rollout["total_rewards"] for rollout in rollouts],
        "joint_rewards": [rollout["joint_reward"] for rollout in rollouts],
        "steps": [rollout["steps"] for rollout in rollouts],
        "episode_lengths": [rollout["steps"] for rollout in rollouts],
        "episode_metrics": [rollout.get("episode_metrics") for rollout in rollouts],
        "metric_totals": aggregate_lbf_episode_metrics(
            [rollout.get("episode_metrics") for rollout in rollouts],
            len(rollouts[0]["total_rewards"]) if rollouts else None,
        ),
    }


def _rollout_animation(frames, *, fps: int = 4, title: str = "Evaluation rollout"):
    if not frames:
        raise ValueError("No rollout frames were captured.")

    import matplotlib.pyplot as plt
    from matplotlib import animation

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
    return fig, anim


def rollout_video_html(frames, *, fps: int = 4, title: str = "Evaluation rollout"):
    """Create an IPython HTML animation for captured rgb_array frames."""
    import matplotlib.pyplot as plt
    from IPython.display import HTML

    fig, anim = _rollout_animation(frames, fps=fps, title=title)
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


def save_rollout_video(
    frames,
    out_path,
    *,
    fps: int = 4,
    title: str = "Evaluation rollout",
):
    """Save captured rgb_array rollout frames as an animation file."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, anim = _rollout_animation(frames, fps=fps, title=title)
    try:
        suffix = out_path.suffix.lower()
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=max(1, int(fps)))
            anim.save(out_path, writer=writer)
        else:
            anim.save(out_path, fps=max(1, int(fps)))
    finally:
        plt.close(fig)
    return out_path


def display_rollout_video(
    frames,
    *,
    fps: int = 4,
    title: str = "Evaluation rollout",
    render_error: str | None = None,
    output_path=None,
):
    """Save and link a rollout animation when frames are available."""
    if not frames:
        reason = f" ({render_error})" if render_error else ""
        print(f"[rollout video skipped: no render frames captured{reason}]")
        return None

    from IPython.display import FileLink, display

    try:
        if output_path is None:
            html = rollout_video_html(frames, fps=fps, title=title)
            display(html)
            return html
        saved_path = save_rollout_video(frames, output_path, fps=fps, title=title)
    except Exception as exc:  # pragma: no cover - depends on notebook display backend
        print(f"[rollout video skipped: {type(exc).__name__}: {exc}]")
        return None
    display(FileLink(str(saved_path)))
    print(f"[rollout video saved: {saved_path}]")
    return saved_path


def sample_lbf_rollouts(
    *,
    make_env: Callable,
    policy_fn: Callable,
    seed: int = 0,
    n_episodes: int = 1,
    max_steps: int | None = None,
    progress_label: str | None = None,
    show_progress: bool = True,
    capture_first_episode_frames: bool = True,
):
    """Run multiple evaluation episodes and keep the first episode's frames."""
    make_env_accepts_capture = _make_env_accepts_capture_arg(make_env)

    rollouts = []
    episode_count = max(1, int(n_episodes))
    episode_iter = range(episode_count)
    if show_progress and tqdm is not None:
        episode_iter = tqdm(
            episode_iter,
            total=episode_count,
            desc=progress_label or "LBF evaluation",
            unit="episode",
            leave=True,
        )
    for episode_idx in episode_iter:
        capture_frames = bool(capture_first_episode_frames and episode_idx == 0)
        rollout = sample_lbf_rollout(
            make_env=lambda capture_frames=capture_frames: _make_episode_env(
                make_env, capture_frames, make_env_accepts_capture
            ),
            policy_fn=policy_fn,
            seed=int(seed) + episode_idx,
            max_steps=max_steps,
            capture_frames=capture_frames,
        )
        rollouts.append(rollout)

    return {
        "rollouts": rollouts,
        "frames": rollouts[0]["frames"] if rollouts else [],
        "render_error": rollouts[0].get("render_error") if rollouts else None,
        "episode_rewards": [rollout["total_rewards"] for rollout in rollouts],
        "joint_rewards": [rollout["joint_reward"] for rollout in rollouts],
        "steps": [rollout["steps"] for rollout in rollouts],
        "episode_lengths": [rollout["steps"] for rollout in rollouts],
        "episode_metrics": [rollout.get("episode_metrics") for rollout in rollouts],
        "metric_totals": aggregate_lbf_episode_metrics(
            [rollout.get("episode_metrics") for rollout in rollouts],
            len(rollouts[0]["total_rewards"]) if rollouts else None,
        ),
    }


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


def _episode_axis_from_payload(payload: dict) -> list[float] | None:
    t_max = payload.get("t_max")
    n_episodes = payload.get("n_episodes")
    if not t_max or not n_episodes:
        return None

    for stats in payload.get("reward_statistics", {}).values():
        logged_steps = stats.get("logged_steps")
        if logged_steps:
            scale = float(t_max) / max(int(n_episodes), 1)
            episodes = [float(step) / max(scale, 1.0) for step in logged_steps]
            if episodes and episodes[-1] < float(n_episodes):
                episodes.append(float(n_episodes))
            return episodes
    return None


def _align_values_to_axis(values, episodes) -> list[float]:
    aligned = list(values)
    if episodes and len(episodes) == len(aligned) + 1 and aligned:
        aligned.append(aligned[-1])
    return aligned


def _is_nonflat_series(values, *, atol: float = 1e-12) -> bool:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return False
    return bool(np.ptp(values) > float(atol))


def agent_training_reward_series(record: dict) -> tuple[list[float], list[tuple[str, np.ndarray]]]:
    """Return per-agent training reward curves from ablation or EPyMARL artifacts."""
    rewards = record.get("rewards")
    if rewards is not None:
        arr = np.asarray(rewards, dtype=np.float64)
        if arr.ndim == 2 and arr.size:
            episodes = list(range(1, arr.shape[1] + 1))
            labels = record.get("agent_labels") or [
                f"Agent {idx + 1}" for idx in range(arr.shape[0])
            ]
            series = [
                (str(labels[idx]), np.asarray(arr[idx], dtype=np.float64))
                for idx in range(arr.shape[0])
                if _is_nonflat_series(arr[idx])
            ]
            return episodes, series

    payload = _load_reward_payload(record)
    curves = payload.get("reward_curve", {})
    agent_labels = sorted(
        label for label, values in curves.items()
        if label.startswith("agent_") and values
    )
    if not agent_labels:
        shared = curves.get("shared") or curves.get("total") or curves.get("joint")
        if not shared:
            return [], []
        agent_labels = ["shared"]

    episodes = _episode_axis_from_payload(payload) or curves.get("episodes")
    if not episodes:
        first_values = curves.get(agent_labels[0], [])
        episodes = list(range(1, len(first_values) + 1))

    series = []
    for label in agent_labels:
        values = curves.get(label)
        if values:
            aligned = np.asarray(_align_values_to_axis(values, episodes), dtype=np.float64)
            if not _is_nonflat_series(aligned):
                continue
            display_label = label.replace("_", " ").title()
            series.append((display_label, aligned))
    return list(episodes), series


def plot_individual_agent_training_rewards(record: dict, *, title_prefix: str | None = None):
    """Plot one line/mean/std training reward figure per agent."""
    import matplotlib.pyplot as plt

    episodes, series = agent_training_reward_series(record)
    if not episodes or not series:
        print(f"[no per-agent training rewards for {_record_label(record)}]")
        return []

    run_label = title_prefix or _record_label(record)
    figs = []
    for label, values in series:
        x = np.asarray(episodes[: values.size], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std())
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(x, values, linewidth=1.3, alpha=0.8, marker="", label=label)
        ax.axhline(mean, color="black", linestyle=":", linewidth=1.4)
        ax.text(
            0.99,
            0.95,
            f"mean={mean:.4f}\nstd={std:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
        )
        ax.set_title(f"{run_label} - {label}")
        ax.set_xlabel("Scenario episode")
        ax.set_ylabel("Training reward")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        figs.append(fig)
    return figs


def plot_combined_agent_training_rewards(record: dict, *, title: str | None = None):
    """Plot one clean line chart comparing agent reward trajectories."""
    import matplotlib.pyplot as plt

    episodes, series = agent_training_reward_series(record)
    if not episodes or not series:
        print(f"[no per-agent training rewards for {_record_label(record)}]")
        return None

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for label, values in series:
        x = np.asarray(episodes[: values.size], dtype=np.float64)
        ax.plot(x, values, linewidth=1.6, marker="", label=label)
    ax.set_title(title or f"{_record_label(record)} - agent reward comparison")
    ax.set_xlabel("Scenario episode")
    ax.set_ylabel("Training reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def display_training_reward_plots(records: Iterable[dict]):
    """Display the requested per-agent and combined training plots for each run."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    displayed = []
    for record in records:
        run_label = _record_label(record)
        for fig in plot_individual_agent_training_rewards(record, title_prefix=run_label):
            display(fig)
            plt.close(fig)
            displayed.append(fig)
        combined = plot_combined_agent_training_rewards(
            record,
            title=f"{run_label} - agent reward comparison",
        )
        if combined is not None:
            display(combined)
            plt.close(combined)
            displayed.append(combined)
    return displayed


def _evaluation_reward_matrix(eval_record: dict) -> tuple[list[str], list[np.ndarray]]:
    rewards = eval_record.get("episode_rewards")
    if rewards is None and "total_rewards" in eval_record:
        rewards = [eval_record["total_rewards"]]
    if rewards is None:
        return [], []

    arr = np.asarray(rewards, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    labels = eval_record.get("agent_labels")
    if labels and arr.ndim == 2 and len(labels) == arr.shape[0]:
        agent_first = arr
    else:
        agent_first = arr.T

    if labels is None or len(labels) != agent_first.shape[0]:
        if agent_first.shape[0] == 1:
            labels = ["Shared reward"]
        else:
            labels = [f"Agent {idx + 1}" for idx in range(agent_first.shape[0])]

    return [str(label) for label in labels], [
        np.asarray(agent_first[idx], dtype=np.float64)
        for idx in range(agent_first.shape[0])
    ]


def plot_evaluation_agent_reward_boxplot(eval_record: dict, *, title: str):
    """Plot one evaluation reward boxplot with one box per agent."""
    import matplotlib.pyplot as plt

    labels, series = _evaluation_reward_matrix(eval_record)
    series = [values for values in series if values.size]
    if not labels or not series:
        print(f"[no evaluation reward series for {title}]")
        return None

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(series)), 4))
    try:
        ax.boxplot(series, tick_labels=labels[: len(series)], showmeans=True)
    except TypeError:  # matplotlib<3.9
        ax.boxplot(series, labels=labels[: len(series)], showmeans=True)
    ax.set_title(title)
    ax.set_ylabel("Evaluation episode reward")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


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
