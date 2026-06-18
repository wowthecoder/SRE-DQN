"""Render one MF-DSRQ opponent-source epsilon-0.01 vs IQL Battle episode.

This is a presentation preset around ``render_mfdsrq_vs_baseline_video.py``.
It leaves the generic renderer and existing presentation videos untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DISCRETE_DIR = _THIS_DIR.parent
if str(_DISCRETE_DIR) not in sys.path:
    sys.path.insert(0, str(_DISCRETE_DIR))

from mean_field_dsrq.mfrl_baselines import DEFAULT_BASELINE_RUNS_DIR, find_latest_mfrl_baseline_run
from mean_field_dsrq.render_mfdsrq_vs_baseline_video import DEFAULT_CONFIG, render_episode


DEFAULT_MFDSRQ_RUN = (
    _THIS_DIR
    / "runs"
    / "mf_srq_torch_epsilon_training_v4_opp"
    / "eps_0_01"
    / "battle_v4"
    / "seed42"
)
DEFAULT_OUTPUT = (
    _THIS_DIR
    / "runs"
    / "presentation_videos"
    / "mfdsrq_opp_eps001_red_vs_iql_blue.mp4"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render MF-DSRQ opponent-source epsilon-0.01 main vs IQL opponent."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mfdsrq-run", type=Path, default=DEFAULT_MFDSRQ_RUN)
    parser.add_argument("--iql-run", type=Path, default=None)
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
        "--allow-non-negative-main-reward",
        action="store_true",
        help="Write the first MF-DSRQ win even if MF-DSRQ main reward is not negative.",
    )
    parser.add_argument(
        "--randomize-handles",
        action="store_true",
        help="Allow Battle red/blue handle randomization. Disabled by default.",
    )
    args = parser.parse_args()

    if not args.mfdsrq_run.exists():
        raise FileNotFoundError(f"MF-DSRQ run not found: {args.mfdsrq_run}")

    iql_run = args.iql_run or find_latest_mfrl_baseline_run("iql", DEFAULT_BASELINE_RUNS_DIR)
    print(f"MF-DSRQ opponent-source eps=0.01 run: {args.mfdsrq_run}")
    print(f"IQL run:                                  {iql_run}")
    print(f"Output:                                   {args.output}")

    render_episode(
        config_path=args.config,
        mfdsrq_run=args.mfdsrq_run,
        baseline_run=iql_run,
        baseline_algorithm="iql",
        output_path=args.output,
        max_steps=args.max_steps,
        fps=args.fps,
        frame_stride=args.frame_stride,
        device=args.device,
        randomize_handles=args.randomize_handles,
        require_main_win=not args.allow_non_main_win,
        require_negative_main_reward=not args.allow_non_negative_main_reward,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
