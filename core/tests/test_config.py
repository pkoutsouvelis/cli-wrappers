"""Tests for :mod:`cliwrap_core.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from cliwrap_core.config import parse_input


def test_parse_input_files():
    assert parse_input({"files": ["/a.nii.gz"]}) == {"data": ["/a.nii.gz"]}


def test_parse_input_from_file():
    assert parse_input({"from_file": "/data/paths.txt"}) == {
        "data": {"from_file": "/data/paths.txt"}
    }


def test_parse_input_from_file_path_object():
    p = Path("/data/paths.txt")
    assert parse_input({"from_file": p}) == {"data": {"from_file": p}}


def test_parse_input_dataset():
    ds = {"root": "/data", "patterns": "*.nii.gz"}
    assert parse_input({"dataset": ds}) == {"data": ds}


def test_parse_input_rejects_multiple_modes():
    with pytest.raises(ValueError, match="exactly one"):
        parse_input({"files": ["/a.nii.gz"], "from_file": "/paths.txt"})


def test_parse_input_rejects_neither():
    with pytest.raises(ValueError, match="exactly one"):
        parse_input({})


def test_parse_input_rejects_bad_from_file_type():
    with pytest.raises(ValueError, match="from_file must be a path"):
        parse_input({"from_file": ["/a.txt", "/b.txt"]})
