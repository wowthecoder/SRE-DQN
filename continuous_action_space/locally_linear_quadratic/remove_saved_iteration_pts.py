#!/usr/bin/env python3
"""
Remove periodic network checkpoint .pt files saved during training.

This targets files of the form:
    Action_Net_<iter>.pt
    Value_Net_<iter>.pt
    SRE_Action_Net_<iter>.pt
    SRE_Value_Net_<iter>.pt
    ...and the same patterns with any additional prefix before Action/Value.

It intentionally does NOT remove:
    - best_checkpoint/checkpoint.pt
    - final weights like Action_Net.pt / Value_Net.pt
    - *_best.pt files
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ITERATION_CHECKPOINT_RE = re.compile(
    r"^(?:.*_)?(?:SRE_)?(?:Action|Value)_Net_\d+\.pt$"
)


def find_iteration_checkpoints(root: Path) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*.pt"):
        if not path.is_file():
            continue
        if path.name == "checkpoint.pt":
            continue
        if ITERATION_CHECKPOINT_RE.match(path.name):
            matches.append(path)
    return sorted(matches)


def main() -> int:
    default_root = Path(__file__).resolve().parent / "pt_files"
    parser = argparse.ArgumentParser(
        description="Remove periodic network checkpoint .pt files saved every 1000 iterations."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=str(default_root),
        help="Root directory to scan. Defaults to continous_action_space/pt_files next to this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be removed without deleting them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the summary line.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root directory does not exist: {root}")
        return 1

    matches = find_iteration_checkpoints(root)

    if not args.quiet:
        for path in matches:
            prefix = "Would remove" if args.dry_run else "Removing"
            print(f"{prefix}: {path}")

    if not args.dry_run:
        for path in matches:
            path.unlink()

    print(
        f"{'Would remove' if args.dry_run else 'Removed'} {len(matches)} iteration checkpoint file(s) under {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
