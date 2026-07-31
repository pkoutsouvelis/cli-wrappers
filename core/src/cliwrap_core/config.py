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

    Returns ``{"data": <payload>}`` which downstream runners take as their
    ``data`` constructor argument. For ``from_file``, the payload is
    ``{"from_file": <path>}`` so :class:`~cliwrap_core.BaseRunner` can load
    the paths (same semantics as an explicit ``files`` list).
    """
    if not isinstance(raw, dict):
        raise ValueError(f"input must be a mapping, got {type(raw).__name__}")

    has_files = "files" in raw and raw["files"] is not None
    has_from_file = "from_file" in raw and raw["from_file"] is not None
    has_dataset = "dataset" in raw and raw["dataset"] is not None
    n_modes = sum((has_files, has_from_file, has_dataset))
    if n_modes != 1:
        raise ValueError(
            "input: specify exactly one of 'files', 'from_file', or 'dataset'"
        )

    if has_files:
        return {"data": raw["files"]}
    if has_from_file:
        from_file = raw["from_file"]
        if not isinstance(from_file, (str, Path)):
            raise ValueError(
                "input.from_file must be a path string, "
                f"got {type(from_file).__name__}"
            )
        return {"data": {"from_file": from_file}}
    return {"data": raw["dataset"]}


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
