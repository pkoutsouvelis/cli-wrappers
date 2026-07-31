"""Reusable YAML config helpers.

Tool-level config modules build on these primitives:

- :func:`load_yaml` reads and minimally validates the top-level mapping.
- :func:`require_keys` / :func:`reject_unknown_keys` are small key-set guards.
- :func:`parse_input` enforces the shared ``input.{files,from_file,dataset}`` shape.
- :func:`parse_logging` enforces the shared ``logging`` section.

Each tool implements its own ``parse_<tool>`` and ``parse_output`` (where the
output knobs vary per tool) and stitches the four sections together in its CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and assert the root is a mapping."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    return raw


def require_keys(section_name: str, raw: dict[str, Any], required: set[str]) -> None:
    """Raise ``ValueError`` if any of ``required`` is missing from ``raw``."""
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(
            f"`{section_name}` is missing required keys: {sorted(missing)}"
        )


def reject_unknown_keys(
    section_name: str, raw: dict[str, Any], allowed: set[str]
) -> None:
    """Raise ``ValueError`` if ``raw`` contains keys outside ``allowed``."""
    extra = set(raw.keys()) - allowed
    if extra:
        raise ValueError(
            f"`{section_name}` cannot contain keys other than {sorted(allowed)}; "
            f"got unexpected keys {sorted(extra)}"
        )


def parse_input(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse the shared ``input`` section.

    Accepts exactly one of:
      - ``files: [...]``
      - ``from_file: /path/to/paths.txt`` (one filepath per line)
      - ``dataset: {root, patterns, levels?, filters?}``

    Optional ``root`` may be set alongside ``files`` or ``from_file`` so
    outputs mirror each path's ``root``-relative parent under ``output_dir``.
    Every listed path must lie under that root. ``root`` is invalid with
    ``dataset`` (use ``dataset.root`` instead).

    Returns ``{"data": <payload>}`` which downstream runners take as their
    ``data`` constructor argument.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"input must be a mapping, got {type(raw).__name__}")

    reject_unknown_keys("input", raw, {"files", "from_file", "dataset", "root"})

    has_files = "files" in raw and raw["files"] is not None
    has_from_file = "from_file" in raw and raw["from_file"] is not None
    has_dataset = "dataset" in raw and raw["dataset"] is not None
    n_modes = sum((has_files, has_from_file, has_dataset))
    if n_modes != 1:
        raise ValueError(
            "input: specify exactly one of 'files', 'from_file', or 'dataset'"
        )

    root = raw.get("root")
    if root is not None and not isinstance(root, (str, Path)):
        raise ValueError(f"input.root must be a path string, got {type(root).__name__}")

    if has_dataset:
        if root is not None:
            raise ValueError(
                "input.root cannot be used with input.dataset; "
                "set dataset.root instead"
            )
        return {"data": raw["dataset"]}

    if has_files:
        if root is None:
            return {"data": raw["files"]}
        return {"data": {"files": raw["files"], "root": root}}

    from_file = raw["from_file"]
    if not isinstance(from_file, (str, Path)):
        raise ValueError(
            "input.from_file must be a path string, " f"got {type(from_file).__name__}"
        )
    payload: dict[str, Any] = {"from_file": from_file}
    if root is not None:
        payload["root"] = root
    return {"data": payload}


def parse_logging(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Parse the shared ``logging`` section.

    Returns a dict with keys ``log_dir`` and ``log_level`` matching the runner
    constructor signature.
    """
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"logging settings must be a mapping, got {type(raw).__name__}"
        )

    reject_unknown_keys("logging", raw, {"log_dir", "level"})

    level = raw.get("level", "INFO")
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ValueError(
            f"logging.level must be DEBUG|INFO|WARNING|ERROR, got {level!r}"
        )

    return {
        "log_dir": raw.get("log_dir"),
        "log_level": level,
    }
