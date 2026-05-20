"""Tests for :mod:`synthseg_wrapper.cli`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from synthseg_wrapper.cli import main


def _write_cfg(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_cli_dry_run_with_dataset(
    bids_tree: Path,
    tmp_path: Path,
    fake_photo_synthseg: Path,
    patch_synthseg,
    capsys,
):
    cfg_path = _write_cfg(
        tmp_path / "cfg.yaml",
        {
            "input": {
                "dataset": {
                    "root": str(bids_tree),
                    "patterns": "**/anat/*T1w.nii*",
                }
            },
            "synthseg": {"synthseg_home": str(fake_photo_synthseg)},
        },
    )

    rc = main(["-c", str(cfg_path), "--dry-run"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "sub-01_ses-01_T1w_synthseg.nii.gz" in out
    assert "sub-02_ses-01_T1w_synthseg.nii.gz" in out
    assert patch_synthseg.calls == []


def test_cli_runs_with_files(
    single_nifti: Path,
    tmp_path: Path,
    fake_photo_synthseg: Path,
    patch_synthseg,
):
    cfg_path = _write_cfg(
        tmp_path / "cfg.yaml",
        {
            "input": {"files": str(single_nifti)},
            "synthseg": {
                "synthseg_home": str(fake_photo_synthseg),
                "threads": 2,
            },
        },
    )

    rc = main(["-c", str(cfg_path)])
    assert rc == 0
    assert len(patch_synthseg.calls) == 1
    assert (single_nifti.parent / "scan_T1w_synthseg.nii.gz").exists()


def test_cli_missing_config_arg():
    with pytest.raises(SystemExit):
        main([])


def test_cli_runs_with_logging(
    single_nifti: Path,
    tmp_path: Path,
    fake_photo_synthseg: Path,
    patch_synthseg,
):
    log_dir = tmp_path / "logs"
    cfg_path = _write_cfg(
        tmp_path / "cfg.yaml",
        {
            "input": {"files": str(single_nifti)},
            "synthseg": {"synthseg_home": str(fake_photo_synthseg)},
            "logging": {"log_dir": str(log_dir), "level": "INFO"},
        },
    )

    rc = main(["-c", str(cfg_path)])
    assert rc == 0
    [log_file] = log_dir.glob("synthseg_run_*.log")
    assert "Done." in log_file.read_text()
