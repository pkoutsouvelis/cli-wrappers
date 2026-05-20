"""Tests for :mod:`synthseg_wrapper.setup`."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from synthseg_wrapper.setup import install_weights


def _make_weights_zip(zip_path: Path, *, top_level_dir: str = "SynthSeg_models") -> Path:
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{top_level_dir}/synthseg_2.0.h5", b"weights")
        zf.writestr(f"{top_level_dir}/synthseg_parc_2.0.h5", b"weights")
    return zip_path


def test_install_weights_renames_top_level_dir(tmp_path: Path):
    home = tmp_path / "Photo-SynthSeg"
    home.mkdir()
    zip_path = _make_weights_zip(tmp_path / "SynthSeg_models.zip")

    install_weights(home, zip_path)

    assert (home / "models" / "synthseg_2.0.h5").exists()
    assert not (home / "SynthSeg_models").exists()


def test_install_weights_overwrites_existing_models_dir(tmp_path: Path):
    home = tmp_path / "Photo-SynthSeg"
    (home / "models").mkdir(parents=True)
    (home / "models" / "stale.h5").write_text("stale")
    zip_path = _make_weights_zip(tmp_path / "SynthSeg_models.zip")

    install_weights(home, zip_path)

    assert (home / "models" / "synthseg_2.0.h5").exists()
    assert not (home / "models" / "stale.h5").exists()


def test_install_weights_strips_macosx_junk(tmp_path: Path):
    home = tmp_path / "Photo-SynthSeg"
    home.mkdir()
    zip_path = tmp_path / "SynthSeg_models.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("SynthSeg_models/synthseg_2.0.h5", b"weights")
        zf.writestr("__MACOSX/SynthSeg_models/._junk", b"")

    install_weights(home, zip_path)
    assert not (home / "__MACOSX").exists()


def test_install_weights_missing_zip(tmp_path: Path):
    home = tmp_path / "Photo-SynthSeg"
    home.mkdir()
    with pytest.raises(FileNotFoundError, match="Weights archive not found"):
        install_weights(home, tmp_path / "nope.zip")
