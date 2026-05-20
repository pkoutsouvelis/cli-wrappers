"""YAML config loader for ``hd-bet-wrapper``.

Defines the HD-BET-specific ``output`` and ``hdbet`` sections; the ``input``
and ``logging`` sections are parsed by the shared helpers in
:mod:`cliwrap_core.config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cliwrap_core.config import (
    load_yaml,
    parse_input,
    parse_logging,
    reject_unknown_keys,
)


_OUTPUT_KEYS = {
    "output_dir",
    "save_mask",
    "save_bet_image",
    "mask_suffix",
    "bet_image_suffix",
    "overwrite",
}
_HDBET_KEYS = {
    "use_tta",
    "device",
    "num_processes_preprocessing",
    "num_processes_segmentation_export",
    "verbose",
}
_TOP_LEVEL_KEYS = {"input", "output", "hdbet", "logging"}


def _parse_output(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"output must be a mapping, got {type(raw).__name__}")
    reject_unknown_keys("output", raw, _OUTPUT_KEYS)
    return raw


def _parse_hdbet(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"hdbet settings must be a mapping, got {type(raw).__name__}")
    reject_unknown_keys("hdbet settings", raw, _HDBET_KEYS)
    return raw


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML config file for ``hd-bet-wrapper``."""
    raw = load_yaml(path)
    if "input" not in raw:
        raise ValueError("Config must contain an 'input' key")
    extra = set(raw.keys()) - _TOP_LEVEL_KEYS
    if extra:
        raise ValueError(
            f"Config cannot contain keys other than {sorted(_TOP_LEVEL_KEYS)}; "
            f"got unexpected keys {sorted(extra)}"
        )
    return {
        "input": parse_input(raw["input"]),
        "output": _parse_output(raw.get("output")),
        "hdbet": _parse_hdbet(raw.get("hdbet")),
        "logging": parse_logging(raw.get("logging")),
    }
