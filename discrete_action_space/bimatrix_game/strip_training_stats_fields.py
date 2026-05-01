#!/usr/bin/env python3
"""Remove bulky history fields from saved training_stats.txt files."""

import argparse
import json
from pathlib import Path


TOP_LEVEL_KEYS = ("epsilon_history", "action_epsilon_history")
TIMING_KEYS = ("episode_durations_seconds",)


def strip_stats_file(path, *, dry_run=False):
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    removed = []
    for key in TOP_LEVEL_KEYS:
        if key in stats:
            del stats[key]
            removed.append(key)

    timing = stats.get("timing")
    if isinstance(timing, dict):
        for key in TIMING_KEYS:
            if key in timing:
                del timing[key]
                removed.append(f"timing.{key}")

    if removed and not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=False)
            f.write("\n")

    return removed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove epsilon_history, action_epsilon_history, and "
            "timing.episode_durations_seconds from training_stats.txt files."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parent / "joint_q_network",
        type=Path,
        help="Directory to search. Defaults to ./joint_q_network next to this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would change without rewriting them.",
    )
    args = parser.parse_args()

    stats_paths = sorted(args.root.glob("**/training_stats.txt"))
    changed = 0

    for path in stats_paths:
        removed = strip_stats_file(path, dry_run=args.dry_run)
        if removed:
            changed += 1
            action = "Would update" if args.dry_run else "Updated"
            print(f"{action} {path}: removed {', '.join(removed)}")

    print(
        f"{'Would update' if args.dry_run else 'Updated'} "
        f"{changed} of {len(stats_paths)} training_stats.txt files."
    )


if __name__ == "__main__":
    main()
