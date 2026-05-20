"""Tests for :mod:`hdbet_wrapper.utils` (citation) and re-exported core utils."""

from __future__ import annotations

from pathlib import Path

from cliwrap_core import get_ext, resolve_path

from hdbet_wrapper.utils import print_hd_bet_citation


def test_resolve_path_expands_and_resolves(tmp_path: Path):
    p = resolve_path(tmp_path / "x")
    assert p.is_absolute()


def test_get_ext_double_extension():
    assert get_ext("foo/bar.nii.gz") == ".nii.gz"


def test_get_ext_single_extension():
    assert get_ext("foo/bar.nii") == ".nii"


def test_print_hd_bet_citation_default_print(capsys):
    print_hd_bet_citation()
    out = capsys.readouterr().out
    assert "HD-BET" in out
    assert "nnU-Net" in out
    assert out.count("########################") == 2


def test_print_hd_bet_citation_with_custom_emit():
    lines: list[str] = []
    print_hd_bet_citation(emit=lines.append)
    assert lines[0] == "########################"
    assert lines[-1] == "########################"
    assert any("HD-BET" in line for line in lines)
