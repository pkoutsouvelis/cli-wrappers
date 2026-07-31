"""Tests for :func:`synthseg_wrapper.config.load_config`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from synthseg_wrapper.config import load_config


def _write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_minimal_dataset_config(tmp_path: Path):
    cfg = load_config(
        _write(
            tmp_path / "cfg.yaml",
            {
                "input": {
                    "dataset": {
                        "root": str(tmp_path / "bids"),
                        "patterns": "**/*.nii.gz",
                    }
                },
                "synthseg": {"synthseg_home": "/opt/Photo-SynthSeg"},
            },
        )
    )
    assert set(cfg.keys()) == {
        "input",
        "output",
        "synthseg_init",
        "synthseg_call",
        "logging",
    }
    assert cfg["synthseg_init"] == {
        "synthseg_home": "/opt/Photo-SynthSeg",
        "python_executable": None,
    }
    assert cfg["synthseg_call"] == {}


def test_from_file_mode_config(tmp_path: Path):
    cfg = load_config(
        _write(
            tmp_path / "cfg.yaml",
            {
                "input": {
                    "from_file": "/data/paths.txt",
                    "root": "/data/FOMO300k",
                },
                "synthseg": {"synthseg_home": "/opt/Photo-SynthSeg"},
            },
        )
    )
    assert cfg["input"]["data"] == {
        "from_file": "/data/paths.txt",
        "root": "/data/FOMO300k",
    }


def test_missing_synthseg_home(tmp_path: Path):
    with pytest.raises(ValueError, match="synthseg.synthseg_home is required"):
        load_config(
            _write(
                tmp_path / "cfg.yaml",
                {"input": {"files": ["/a.nii.gz"]}},
            )
        )


def test_volumes_without_output_dir_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="output.output_dir is required"):
        load_config(
            _write(
                tmp_path / "cfg.yaml",
                {
                    "input": {"files": ["/a.nii.gz"]},
                    "output": {"save_volumes": True},
                    "synthseg": {"synthseg_home": "/opt/Photo-SynthSeg"},
                },
            )
        )


def test_unknown_top_level_key(tmp_path: Path):
    with pytest.raises(ValueError, match="Config cannot contain keys"):
        load_config(
            _write(
                tmp_path / "cfg.yaml",
                {
                    "input": {"files": ["/a.nii.gz"]},
                    "synthseg": {"synthseg_home": "/opt"},
                    "bogus": 1,
                },
            )
        )


def test_unknown_synthseg_key(tmp_path: Path):
    with pytest.raises(ValueError, match="`synthseg settings` cannot contain"):
        load_config(
            _write(
                tmp_path / "cfg.yaml",
                {
                    "input": {"files": ["/a.nii.gz"]},
                    "synthseg": {"synthseg_home": "/opt", "bogus": True},
                },
            )
        )


def test_unknown_output_key(tmp_path: Path):
    with pytest.raises(ValueError, match="`output` cannot contain"):
        load_config(
            _write(
                tmp_path / "cfg.yaml",
                {
                    "input": {"files": ["/a.nii.gz"]},
                    "output": {"unknown": 1},
                    "synthseg": {"synthseg_home": "/opt"},
                },
            )
        )


def test_call_kwargs_pass_through(tmp_path: Path):
    cfg = load_config(
        _write(
            tmp_path / "cfg.yaml",
            {
                "input": {"files": ["/a.nii.gz"]},
                "synthseg": {
                    "synthseg_home": "/opt/Photo-SynthSeg",
                    "parc": True,
                    "robust": True,
                    "threads": 8,
                    "crop": [192, 192, 192],
                },
            },
        )
    )
    assert cfg["synthseg_call"] == {
        "parc": True,
        "robust": True,
        "threads": 8,
        "crop": [192, 192, 192],
    }


def test_part_keys_pass_through_to_call_kwargs(tmp_path: Path):
    cfg = load_config(
        _write(
            tmp_path / "cfg.yaml",
            {
                "input": {"files": ["/a.nii.gz"]},
                "synthseg": {
                    "synthseg_home": "/opt/Photo-SynthSeg",
                    "num_parts": 4,
                    "part_idx": [0, 2],
                },
            },
        )
    )
    assert cfg["synthseg_call"] == {"num_parts": 4, "part_idx": [0, 2]}
