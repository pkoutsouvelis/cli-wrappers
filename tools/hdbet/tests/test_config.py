"""Tests for :func:`hdbet_wrapper.config.load_config`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from hdbet_wrapper.config import load_config


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
                }
            },
        )
    )
    assert set(cfg.keys()) == {"input", "output", "hdbet", "logging"}
    assert cfg["input"]["data"] == {
        "root": str(tmp_path / "bids"),
        "patterns": "**/*.nii.gz",
    }
    assert cfg["logging"] == {"log_dir": None, "log_level": "INFO"}


def test_files_mode_config(tmp_path: Path):
    cfg = load_config(
        _write(
            tmp_path / "cfg.yaml",
            {"input": {"files": ["/a.nii.gz", "/b.nii.gz"]}},
        )
    )
    assert cfg["input"]["data"] == ["/a.nii.gz", "/b.nii.gz"]
    assert cfg["input"]["resolve_explicit_filepaths"] is True


def test_from_file_mode_config(tmp_path: Path):
    cfg = load_config(
        _write(
            tmp_path / "cfg.yaml",
            {
                "input": {
                    "from_file": "/data/paths.txt",
                    "root": "/data/FOMO300k",
                    "resolve_explicit_filepaths": False,
                }
            },
        )
    )
    assert cfg["input"]["data"] == {
        "from_file": "/data/paths.txt",
        "root": "/data/FOMO300k",
    }
    assert cfg["input"]["resolve_explicit_filepaths"] is False


def test_input_mutual_exclusion_both(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        load_config(
            _write(
                tmp_path / "cfg.yaml",
                {
                    "input": {
                        "files": ["/a.nii.gz"],
                        "dataset": {"root": "/r", "patterns": "*.nii.gz"},
                    }
                },
            )
        )


def test_input_mutual_exclusion_from_file_and_files(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        load_config(
            _write(
                tmp_path / "cfg.yaml",
                {
                    "input": {
                        "files": ["/a.nii.gz"],
                        "from_file": "/data/paths.txt",
                    }
                },
            )
        )


def test_input_mutual_exclusion_neither(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        load_config(_write(tmp_path / "cfg.yaml", {"input": {}}))


def test_unknown_top_level_key(tmp_path: Path):
    with pytest.raises(ValueError, match="Config cannot contain keys"):
        load_config(
            _write(
                tmp_path / "cfg.yaml",
                {
                    "input": {"files": ["/a.nii.gz"]},
                    "bogus": 1,
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
                },
            )
        )


def test_unknown_hdbet_key(tmp_path: Path):
    with pytest.raises(ValueError, match="`hdbet settings` cannot contain"):
        load_config(
            _write(
                tmp_path / "cfg.yaml",
                {
                    "input": {"files": ["/a.nii.gz"]},
                    "hdbet": {"bogus": True},
                },
            )
        )


def test_bad_logging_level(tmp_path: Path):
    with pytest.raises(ValueError, match="logging.level"):
        load_config(
            _write(
                tmp_path / "cfg.yaml",
                {
                    "input": {"files": ["/a.nii.gz"]},
                    "logging": {"level": "VERBOSE"},
                },
            )
        )


def test_logging_block_renames_level_to_log_level(tmp_path: Path):
    cfg = load_config(
        _write(
            tmp_path / "cfg.yaml",
            {
                "input": {"files": ["/a.nii.gz"]},
                "logging": {"level": "WARNING", "log_dir": str(tmp_path / "logs")},
            },
        )
    )
    assert cfg["logging"] == {
        "log_dir": str(tmp_path / "logs"),
        "log_level": "WARNING",
    }


def test_root_must_be_mapping(tmp_path: Path):
    bad = tmp_path / "cfg.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="Config root must be a mapping"):
        load_config(bad)


def test_hdbet_part_keys_are_accepted(tmp_path: Path):
    cfg = load_config(
        _write(
            tmp_path / "cfg.yaml",
            {
                "input": {"files": ["/a.nii.gz"]},
                "hdbet": {"num_parts": 4, "part_idx": [0, 2]},
            },
        )
    )
    assert cfg["hdbet"] == {"num_parts": 4, "part_idx": [0, 2]}
