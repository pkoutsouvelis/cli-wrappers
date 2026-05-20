"""Command-line entry point for ``synthseg-wrapper``."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthseg_wrapper.config import load_config
from synthseg_wrapper.runner import SynthSegRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthseg-wrapper",
        description="Batch-friendly Photo-SynthSeg driver with YAML config + nifti-finder discovery.",
    )
    parser.add_argument(
        "-c", "--config", required=True, type=Path, help="Path to YAML config."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover inputs and print planned outputs without running SynthSeg.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    expected = {"input", "output", "synthseg_init", "synthseg_call", "logging"}
    if set(cfg.keys()) != expected:
        raise ValueError(
            f"Internal error: config did not contain all expected keys; "
            f"expected: {sorted(expected)}; got: {set(cfg.keys())}"
        )
    runner = SynthSegRunner(
        **cfg["input"],
        **cfg["output"],
        **cfg["synthseg_init"],
        **cfg["logging"],
    )
    runner(**cfg["synthseg_call"], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
