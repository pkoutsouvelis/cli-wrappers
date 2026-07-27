"""YAML config loader for ``synthseg-wrapper``.

Defines the SynthSeg-specific ``output`` and ``synthseg`` sections; ``input``
and ``logging`` are parsed by the shared helpers in :mod:`cliwrap_core.config`.
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
    "save_segmentation",
    "segmentation_suffix",
    "save_posteriors",
    "posteriors_suffix",
    "save_resampled",
    "resampled_suffix",
    "save_volumes",
    "volumes_suffix",
    "save_qc",
    "qc_suffix",
    "overwrite",
}

_SYNTHSEG_KEYS = {
    "synthseg_home",
    "python_executable",
    "parc",
    "robust",
    "fast",
    "ct",
    "cpu",
    "threads",
    "crop",
    "v1",
    "num_parts",
    "part_idx",
}

_TOP_LEVEL_KEYS = {"input", "output", "synthseg", "logging"}


def _parse_output(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"output must be a mapping, got {type(raw).__name__}")
    reject_unknown_keys("output", raw, _OUTPUT_KEYS)
    return raw


def _parse_synthseg(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"synthseg settings must be a mapping, got {type(raw).__name__}"
        )
    reject_unknown_keys("synthseg settings", raw, _SYNTHSEG_KEYS)
    return raw


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML config file for ``synthseg-wrapper``."""
    raw = load_yaml(path)
    if "input" not in raw:
        raise ValueError("Config must contain an 'input' key")
    extra = set(raw.keys()) - _TOP_LEVEL_KEYS
    if extra:
        raise ValueError(
            f"Config cannot contain keys other than {sorted(_TOP_LEVEL_KEYS)}; "
            f"got unexpected keys {sorted(extra)}"
        )

    synthseg_cfg = _parse_synthseg(raw.get("synthseg"))
    output_cfg = _parse_output(raw.get("output"))

    # ``synthseg_home`` is required for the runner; surface a clear error here.
    if "synthseg_home" not in synthseg_cfg:
        raise ValueError(
            "synthseg.synthseg_home is required (path to a Photo-SynthSeg clone). "
            "Run `synthseg-setup` to provision one."
        )

    # Per-input volumes/QC CSVs mirror the segmentation layout under output_dir.
    save_volumes = output_cfg.get("save_volumes", False)
    save_qc = output_cfg.get("save_qc", False)
    if (save_volumes or save_qc) and "output_dir" not in output_cfg:
        raise ValueError(
            "output.output_dir is required when save_volumes or save_qc is true "
            "(per-input CSV files are written under the mirrored output tree)."
        )

    # Configuration of `runner` constructor (split between init and call kwargs).
    init_kwargs = {
        "synthseg_home": synthseg_cfg["synthseg_home"],
        "python_executable": synthseg_cfg.get("python_executable"),
    }
    call_kwargs = {
        k: synthseg_cfg[k]
        for k in (
            "parc",
            "robust",
            "fast",
            "ct",
            "cpu",
            "threads",
            "crop",
            "v1",
            "num_parts",
            "part_idx",
        )
        if k in synthseg_cfg
    }

    return {
        "input": parse_input(raw["input"]),
        "output": output_cfg,
        "synthseg_init": init_kwargs,
        "synthseg_call": call_kwargs,
        "logging": parse_logging(raw.get("logging")),
    }
