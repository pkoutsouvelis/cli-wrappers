"""Tests for :func:`cliwrap_core.explorer_factory.get_data_explorer`."""

from __future__ import annotations

from pathlib import Path

import pytest
from nifti_finder.explorers import AllPurposeFileExplorer

from cliwrap_core.explorer_factory import _build_filter, get_data_explorer


def test_get_data_explorer_returns_all_purpose_explorer():
    explorer = get_data_explorer("**/*.nii.gz", filter_kwargs=None)
    assert isinstance(explorer, AllPurposeFileExplorer)


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
        "**/*.nii.gz",
        filter_kwargs={
            "name": "ExcludeFileSuffix",
            "kwargs": {"suffix": "_bet"},
        },
    )
    paths = explorer.list(str(tmp_path), sort=True, unique=True)
    names = [Path(p).name for p in paths]
    assert names == ["x.nii.gz"]
