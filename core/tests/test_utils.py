"""Tests for :mod:`cliwrap_core.utils`."""

from __future__ import annotations

from pathlib import Path

import pytest

from cliwrap_core.utils import (
    get_ext,
    normalize_part_indices,
    resolve_path,
    slice_by_parts,
)


def test_resolve_path_expands_and_resolves(tmp_path: Path):
    p = resolve_path(tmp_path / "x")
    assert p.is_absolute()


def test_get_ext_double_extension():
    assert get_ext("foo/bar.nii.gz") == ".nii.gz"


def test_get_ext_single_extension():
    assert get_ext("foo/bar.nii") == ".nii"


def test_normalize_part_indices_int():
    assert normalize_part_indices(2, 4) == [2]


def test_normalize_part_indices_list_dedupes_and_sorts():
    assert normalize_part_indices([3, 1, 3], 4) == [1, 3]


def test_normalize_part_indices_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        normalize_part_indices(3, 3)


def test_normalize_part_indices_rejects_bad_num_parts():
    with pytest.raises(ValueError, match="num_parts"):
        normalize_part_indices(0, 0)


def test_slice_by_parts_contiguous_chunks():
    items = list(range(10))
    assert slice_by_parts(items, 4, [0]) == [0, 1]
    assert slice_by_parts(items, 4, [1]) == [2, 3, 4]
    assert slice_by_parts(items, 4, [2]) == [5, 6]
    assert slice_by_parts(items, 4, [3]) == [7, 8, 9]
    assert slice_by_parts(items, 4, [0, 2]) == [0, 1, 5, 6]


def test_slice_by_parts_noop_for_single_part():
    items = ["a", "b", "c"]
    assert slice_by_parts(items, 1, [0]) == items
