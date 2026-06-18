"""Render one MF-DSRQ-vs-baseline Battle evaluation episode to an MP4.

Example:
    python -m discrete_action_space.mean_field_dsrq.render_mfdsrq_vs_baseline_video

The script intentionally records only the rollout video. It reuses the existing
low-level policy loaders and action helpers from eval_mf_dsrq.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
if str(_DISCRETE_DIR) not in sys.path:
    sys.path.insert(0, str(_DISCRETE_DIR))

from mean_field_dsrq.eval_mf_dsrq import (
    _close_low_level_tournament_policy,
    _load_low_level_tournament_policy,
    _low_level_tournament_actions,
    _resolve_torch_device,
)
from mean_field_dsrq.magent2_env import DEFAULT_TASK_CONFIG, LowLevelBattleEnv
from mean_field_dsrq.mfrl_baselines import DEFAULT_BASELINE_RUNS_DIR, find_latest_mfrl_baseline_run
from mean_field_dsrq.train_mf_dsrq import _actions_to_mean, _load_config


DEFAULT_CONFIG = _THIS_DIR / "configs" / "battle_v4.yaml"
DEFAULT_BASELINE = "mfq"
DEFAULT_MFDSRQ_RUN = (
    _THIS_DIR
    / "runs"
    / "mf_srq_torch_epsilon_training_v4_opp"
    / "eps_0_01"
    / "battle_v4"
    / "seed42"
)
DEFAULT_OUTPUT = _THIS_DIR / "runs" / "presentation_videos" / "mfdsrq_red_vs_mfq_blue_win.mp4"


def _latest_mfdsrq_run(root: Path = _THIS_DIR / "runs") -> Path:
    candidates = [
        path.parent
        for path in root.glob("**/fixed_side_tournament.json")
        if (path.parent / "ckpt_main_best.pt").exists()
    ]
    if not candidates:
        candidates = [
            path.parent
            for path in root.glob("**/ckpt_main_best.pt")
            if (path.parent / "ckpt_opponent_best.pt").exists()
        ]
    if not candidates:
        raise FileNotFoundError(f"No MF-DSRQ run with ckpt_main_best.pt found under {root}.")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def _ensure_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Expected RGB frame with shape [H, W, C], got {frame.shape}.")
    frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(frame)) <= 1.0 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    height, width = frame.shape[:2]
    if height % 2 or width % 2:
        padded = np.zeros((height + height % 2, width + width % 2, 3), dtype=np.uint8)
        padded[:height, :width] = frame
        frame = padded
    return np.ascontiguousarray(frame)


def _write_mp4(frames: list[np.ndarray], output_path: Path, *, fps: int) -> None:
    if not frames:
        raise ValueError("Cannot write a video with no frames.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH; install ffmpeg or pass frames to another writer.")

    first = _ensure_rgb_uint8(frames[0])
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            frame = _ensure_rgb_uint8(frame)
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame size changed from {(height, width)} to {frame.shape[:2]}."
                )
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}:\n{stderr}")


def _load_eval_config(config_path: Path, mfdsrq_run: Path) -> dict:
    cfg = _load_config(str(config_path))
    run_config = mfdsrq_run / "config.json"
    if run_config.exists():
        with open(run_config, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def _team_counts(env: LowLevelBattleEnv) -> list[int]:
    return [int(env.get_num(0)), int(env.get_num(1))]


def _win_summary(initial_counts: list[int], final_counts: list[int]) -> dict:
    main_kills = int(initial_counts[1] - final_counts[1])
    opponent_kills = int(initial_counts[0] - final_counts[0])
    if main_kills > opponent_kills:
        winner = "main"
    elif opponent_kills > main_kills:
        winner = "opponent"
    else:
        winner = "tie"
    return {
        "winner": winner,
        "main_kills": main_kills,
        "opponent_kills": opponent_kills,
        "initial_counts": {"main": int(initial_counts[0]), "opponent": int(initial_counts[1])},
        "final_counts": {"main": int(final_counts[0]), "opponent": int(final_counts[1])},
    }


def _team_rewards(env: LowLevelBattleEnv) -> list[np.ndarray]:
    return [
        np.asarray(env.grid.get_reward(env.handles[0]), dtype=np.float32),
        np.asarray(env.grid.get_reward(env.handles[1]), dtype=np.float32),
    ]


def render_episode(
    *,
    config_path: Path,
    mfdsrq_run: Path,
    baseline_run: Path,
    baseline_algorithm: str,
    output_path: Path,
    max_steps: int | None,
    fps: int,
    frame_stride: int,
    device: str | None,
    randomize_handles: bool,
    require_main_win: bool,
    require_negative_main_reward: bool,
    max_attempts: int,
) -> Path:
    cfg = _load_eval_config(config_path, mfdsrq_run)
    cfg["randomize_handles_on_reset"] = bool(randomize_handles)
    baseline_algorithm = str(baseline_algorithm).lower()
    task_config = {**DEFAULT_TASK_CONFIG, **cfg}
    env = LowLevelBattleEnv(task_config)
    meta = env.meta()
    device_obj = _resolve_torch_device(device if device is not None else cfg.get("device"), use_gpu=cfg.get("use_gpu", True))
    max_steps = int(max_steps or cfg.get("max_cycles", env.max_steps))
    frame_stride = max(1, int(frame_stride))

    checkpoint_paths = {
        "main": mfdsrq_run / "ckpt_main_best.pt",
        "opponent": mfdsrq_run / "ckpt_opponent_best.pt",
    }
    baseline_folders = {baseline_algorithm: baseline_run}
    main_policy = None
    opponent_policy = None

    try:
        main_policy = _load_low_level_tournament_policy(
            algorithm="mfdsrq",
            role="main",
            cfg=cfg,
            checkpoint_paths=checkpoint_paths,
            baseline_folders=baseline_folders,
            env=env,
            device=device_obj,
        )
        opponent_policy = _load_low_level_tournament_policy(
            algorithm=baseline_algorithm,
            role="opponent",
            cfg=cfg,
            checkpoint_paths=checkpoint_paths,
            baseline_folders=baseline_folders,
            env=env,
            device=device_obj,
        )

        best_summary = None
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            env.reset()
            initial_counts = _team_counts(env)
            episode_rewards = [0.0, 0.0]
            frames = [env.render()]
            former_prob = [
                np.zeros((1, int(meta.num_actions)), dtype=np.float32),
                np.zeros((1, int(meta.num_actions)), dtype=np.float32),
            ]

            for step_idx in range(max_steps):
                actions = [
                    _low_level_tournament_actions(
                        policy=main_policy,
                        cfg=cfg,
                        env=env,
                        group_idx=0,
                        former_prob=former_prob,
                    ),
                    _low_level_tournament_actions(
                        policy=opponent_policy,
                        cfg=cfg,
                        env=env,
                        group_idx=1,
                        former_prob=former_prob,
                    ),
                ]
                if len(actions[0]) == 0 and len(actions[1]) == 0:
                    break

                env.set_action(0, actions[0])
                env.set_action(1, actions[1])
                done = env.step()
                rewards = _team_rewards(env)
                episode_rewards[0] += float(np.sum(rewards[0]))
                episode_rewards[1] += float(np.sum(rewards[1]))
                if (step_idx + 1) % frame_stride == 0 or bool(done):
                    frames.append(env.render())
                former_prob = [
                    _actions_to_mean(actions[0], int(meta.num_actions)),
                    _actions_to_mean(actions[1], int(meta.num_actions)),
                ]
                env.clear_dead()
                if bool(done):
                    break

            summary = _win_summary(initial_counts, _team_counts(env))
            summary["attempt"] = int(attempt)
            summary["frames"] = int(len(frames))
            summary["main_reward"] = float(episode_rewards[0])
            summary["opponent_reward"] = float(episode_rewards[1])
            best_summary = summary
            print(
                f"Attempt {attempt}: winner={summary['winner']} "
                f"main_kills={summary['main_kills']} "
                f"opponent_kills={summary['opponent_kills']} "
                f"main_reward={summary['main_reward']:.3f} "
                f"opponent_reward={summary['opponent_reward']:.3f}"
            )
            main_win_ok = not require_main_win or summary["winner"] == "main"
            main_reward_ok = (
                not require_negative_main_reward or summary["main_reward"] < 0.0
            )
            if main_win_ok and main_reward_ok:
                _write_mp4(frames, output_path, fps=fps)
                print(f"Wrote {len(frames)} frames to {output_path}")
                return output_path

        raise RuntimeError(
            "No rollout satisfying the requested MF-DSRQ conditions found. "
            f"Last attempt summary: {best_summary}"
        )
        return output_path
    finally:
        _close_low_level_tournament_policy(main_policy)
        _close_low_level_tournament_policy(opponent_policy)
        close = getattr(env.env, "close", None)
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one MF-DSRQ main vs baseline opponent episode.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mfdsrq-run", type=Path, default=None)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, choices=("iql", "ac", "mfq"))
    parser.add_argument("--baseline-run", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-non-main-win",
        action="store_true",
        help="Write the first rollout even if MF-DSRQ main does not win.",
    )
    parser.add_argument(
        "--require-negative-main-reward",
        action="store_true",
        help="Only write a rollout where MF-DSRQ main has negative total reward.",
    )
    parser.add_argument(
        "--randomize-handles",
        action="store_true",
        help="Allow Battle red/blue handle randomization on reset. Disabled by default for stable presentation videos.",
    )
    args = parser.parse_args()

    mfdsrq_run = args.mfdsrq_run or (DEFAULT_MFDSRQ_RUN if DEFAULT_MFDSRQ_RUN.exists() else _latest_mfdsrq_run())
    baseline_run = args.baseline_run or find_latest_mfrl_baseline_run(args.baseline, DEFAULT_BASELINE_RUNS_DIR)
    print(f"MF-DSRQ run: {mfdsrq_run}")
    print(f"{args.baseline.upper()} run:      {baseline_run}")

    render_episode(
        config_path=args.config,
        mfdsrq_run=mfdsrq_run,
        baseline_run=baseline_run,
        baseline_algorithm=args.baseline,
        output_path=args.output,
        max_steps=args.max_steps,
        fps=args.fps,
        frame_stride=args.frame_stride,
        device=args.device,
        randomize_handles=args.randomize_handles,
        require_main_win=not args.allow_non_main_win,
        require_negative_main_reward=args.require_negative_main_reward,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
