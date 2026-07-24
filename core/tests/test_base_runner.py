"""Tests for :class:`cliwrap_core.base_runner.BaseRunner`."""

from __future__ import annotations

from pathlib import Path

import pytest

from cliwrap_core.base_runner import BaseRunner, OutputSpec


class _DummyRunner(BaseRunner):
    """Minimal subclass exposing the internals for testing."""

    def __init__(self, **kwargs):
        kwargs.setdefault("logger_name", "cliwrap_dummy")
        kwargs.setdefault("logger_label", "DUMMY")
        super().__init__(**kwargs)


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def test_invalid_data_type():
    with pytest.raises(ValueError, match="data must be"):
        _DummyRunner(data=42)  # type: ignore[arg-type]


def test_bad_log_level(tmp_path: Path):
    p = _touch(tmp_path / "x.nii.gz")
    with pytest.raises(ValueError, match="log_level"):
        _DummyRunner(data=p, log_level="VERBOSE")  # type: ignore[arg-type]


def test_list_must_contain_paths_or_strings():
    with pytest.raises(ValueError, match="Path or str"):
        _DummyRunner(data=[1, 2])  # type: ignore[list-item]


def test_dataset_requires_root_and_patterns(tmp_path: Path):
    with pytest.raises(ValueError, match="`root` key"):
        _DummyRunner(data={"patterns": "*.nii.gz"})

    with pytest.raises(ValueError, match="`patterns` key"):
        _DummyRunner(data={"root": str(tmp_path)})


def test_plan_outputs_single_file(tmp_path: Path):
    p = _touch(tmp_path / "scan.nii.gz")
    runner = _DummyRunner(data=p)
    plans = runner._plan_outputs(
        [
            OutputSpec("seg", "synthseg", save=True, required=True),
            OutputSpec("post", "post", save=False),
        ]
    )
    assert plans["__inputs__"] == [str(p)]
    assert plans["seg"] == [str(tmp_path / "scan_synthseg.nii.gz")]
    assert plans["post"] == [str(tmp_path / "scan_post.nii.gz")]


def test_plan_outputs_mirrors_under_root(tmp_path: Path):
    root = tmp_path / "ds"
    p = _touch(root / "sub-01" / "anat" / "T1w.nii.gz")
    out_dir = tmp_path / "out"
    runner = _DummyRunner(
        data={"root": str(root), "patterns": "**/*.nii.gz"},
        output_dir=out_dir,
    )
    plans = runner._plan_outputs(
        [OutputSpec("seg", "synthseg", save=True, required=True)]
    )
    assert plans["__inputs__"] == [str(p)]
    assert plans["seg"] == [str(out_dir / "sub-01" / "anat" / "T1w_synthseg.nii.gz")]


def test_dataset_levels_are_forwarded(tmp_path: Path):
    root = tmp_path / "ds"
    keep = _touch(root / "PT001" / "a_T1w.nii.gz")
    _touch(root / "other" / "b_T1w.nii.gz")
    runner = _DummyRunner(
        data={
            "root": str(root),
            "patterns": "*T1w.nii*",
            "levels": {"dataset": "PT*"},
        }
    )
    assert runner._input_files == [keep]


def test_plan_outputs_skips_when_required_output_exists(tmp_path: Path):
    p = _touch(tmp_path / "scan.nii.gz")
    _touch(tmp_path / "scan_synthseg.nii.gz")  # pretend output exists
    runner = _DummyRunner(data=p, overwrite=False)
    plans = runner._plan_outputs(
        [OutputSpec("seg", "synthseg", save=True, required=True)]
    )
    assert plans["__inputs__"] == []
    assert plans["seg"] == []


def test_plan_outputs_does_not_skip_when_only_optional_exists(tmp_path: Path):
    p = _touch(tmp_path / "scan.nii.gz")
    _touch(tmp_path / "scan_post.nii.gz")  # optional output
    runner = _DummyRunner(data=p, overwrite=False)
    plans = runner._plan_outputs(
        [
            OutputSpec("seg", "synthseg", save=True, required=True),
            OutputSpec("post", "post", save=False, required=False),
        ]
    )
    assert plans["__inputs__"] == [str(p)]
