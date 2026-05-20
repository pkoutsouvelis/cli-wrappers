"""Command-line entry point for ``hd-bet-wrapper``."""

from __future__ import annotations

import argparse
from pathlib import Path

from hdbet_wrapper.config import load_config
from hdbet_wrapper.runner import HDBETRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hd-bet-wrapper",
        description="Batch-friendly HD-BET driven by a YAML config + nifti-finder discovery.",
    )
    parser.add_argument(
        "-c", "--config", required=True, type=Path, help="Path to YAML config."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover inputs and print planned outputs without running HD-BET.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if set(cfg.keys()) != {"input", "output", "hdbet", "logging"}:
        raise ValueError(
            f"Internal error: config did not contain all expected keys; "
            f"expected: 'input', 'output', 'hdbet', and 'logging'; "
            f"got: {set(cfg.keys())}"
        )
    runner = HDBETRunner(**cfg["input"], **cfg["output"], **cfg["logging"])
    runner(**cfg["hdbet"], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
