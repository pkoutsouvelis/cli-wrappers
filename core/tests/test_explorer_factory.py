"""Tests for :func:`cliwrap_core.explorer_factory.get_data_explorer`."""

from __future__ import annotations

from pathlib import Path

import pytest
from nifti_finder.explorers import FileFinder

from cliwrap_core.explorer_factory import _build_filter, get_data_explorer


def test_get_data_explorer_returns_file_finder():
    explorer = get_data_explorer("**/*.nii.gz")
    assert isinstance(explorer, FileFinder)


def test_get_data_explorer_with_levels():
    explorer = get_data_explorer(
        patterns="*T1w.nii*",
        levels={"dataset": "PT*"},
    )
    assert isinstance(explorer, FileFinder)


def test_get_data_explorer_rejects_empty_levels():
    with pytest.raises(ValueError, match="must not be empty"):
        get_data_explorer(patterns="*.nii*", levels={})


def test_build_filter_none():
    assert _build_filter(None) is None


def test_build_filter_single():
    flt = _build_filter({"name": "ExcludeFileSuffix", "kwargs": {"suffix": "_bet"}})
    assert flt is not None
    assert flt.__class__.__name__ == "ExcludeFileSuffix"


def test_build_filter_compose_and():
    flt = _build_filter(
        {
            "name": "ComposeFilter",
            "kwargs": {
                "logic": "AND",
                "filters": [
                    {"name": "ExcludeFileSuffix", "kwargs": {"suffix": "_bet"}},
                    {"name": "IncludeExtension", "kwargs": {"extension": ".nii.gz"}},
                ],
            },
        }
    )
    assert flt is not None
    assert flt.__class__.__name__ == "ComposeFilter"


def test_build_filter_missing_name():
    with pytest.raises(ValueError, match="`name` key is required"):
        _build_filter({"kwargs": {"suffix": "x"}})


def test_build_filter_missing_kwargs():
    with pytest.raises(ValueError, match="`kwargs` key is required"):
        _build_filter({"name": "ExcludeFileSuffix"})


def test_build_filter_unknown_filter():
    with pytest.raises(ImportError, match="Filter 'NotARealFilter' not found"):
        _build_filter({"name": "NotARealFilter", "kwargs": {}})


def test_explorer_scans_real_tree(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.nii.gz").write_bytes(b"")
    (tmp_path / "a" / "y_bet.nii.gz").write_bytes(b"")
    (tmp_path / "a" / "z.txt").write_bytes(b"")

    explorer = get_data_explorer(
        patterns="**/*.nii.gz",
        filters={
            "name": "ExcludeFileSuffix",
            "kwargs": {"suffix": "_bet"},
        },
    )
    paths = explorer.list(str(tmp_path), sort=True, unique=True)
    names = [Path(p).name for p in paths]
    assert names == ["x.nii.gz"]


def test_explorer_respects_levels(tmp_path: Path):
    keep = tmp_path / "PT001"
    skip = tmp_path / "other"
    keep.mkdir()
    skip.mkdir()
    (keep / "a_T1w.nii.gz").write_bytes(b"")
    (skip / "b_T1w.nii.gz").write_bytes(b"")

    explorer = get_data_explorer(
        patterns="*T1w.nii*",
        levels={"dataset": "PT*"},
    )
    paths = explorer.list(str(tmp_path), sort=True, unique=True)
    names = [Path(p).name for p in paths]
    assert names == ["a_T1w.nii.gz"]
