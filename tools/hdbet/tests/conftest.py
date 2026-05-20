"""Shared pytest fixtures for hd-bet-wrapper tests.

The HD-BET model is never loaded: ``patch_hdbet`` replaces the symbols
imported by :mod:`hdbet_wrapper.runner` with lightweight fakes that write
valid (empty) NIfTI outputs. Multiprocessing in ``_apply_masks`` is bypassed
by patching the bound method to a synchronous implementation, so tests don't
need to pickle anything across processes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

_TINY_ARR = np.zeros((4, 4, 4), dtype=np.int16)


def _write_nifti(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(_TINY_ARR, affine=np.eye(4)), str(path))


@pytest.fixture()
def bids_tree(tmp_path: Path) -> Path:
    """Two-subject BIDS-style tree with T1w files and `_bet` noise files."""
    root = tmp_path / "bids"
    for sub in ("sub-01", "sub-02"):
        anat = root / sub / "ses-01" / "anat"
        _write_nifti(anat / f"{sub}_ses-01_T1w.nii.gz")
        _write_nifti(anat / f"{sub}_ses-01_T1w_bet.nii.gz")
    return root


@pytest.fixture()
def single_nifti(tmp_path: Path) -> Path:
    p = tmp_path / "alpha" / "scan_T1w.nii.gz"
    _write_nifti(p)
    return p


class FakePredictor:
    """Stand-in for the nnUNetPredictor that HD-BET initializes.

    ``predict_from_files`` records its call and writes a tiny NIfTI at each
    requested mask path, mimicking HD-BET's first-stage output.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict_from_files(
        self,
        inputs: list[list[str]],
        outputs_mask: list[str],
        **kwargs: Any,
    ) -> None:
        self.calls.append({"inputs": inputs, "outputs_mask": outputs_mask, **kwargs})
        for mask_path in outputs_mask:
            _write_nifti(Path(mask_path))


@pytest.fixture()
def patch_hdbet(monkeypatch: pytest.MonkeyPatch) -> FakePredictor:
    """Patch heavy HD-BET / torch symbols at their import sites in the runner."""
    from hdbet_wrapper import runner as runner_mod

    predictor = FakePredictor()

    monkeypatch.setattr(runner_mod, "maybe_download_parameters", lambda: None)

    def _get_predictor(use_tta: bool, device: Any, verbose: bool) -> FakePredictor:
        predictor.init_kwargs = {  # type: ignore[attr-defined]
            "use_tta": use_tta,
            "device": str(device),
            "verbose": verbose,
        }
        return predictor

    monkeypatch.setattr(runner_mod, "get_hdbet_predictor", _get_predictor)

    monkeypatch.setattr(runner_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner_mod.torch.backends.mps, "is_available", lambda: False)

    def _apply_masks(
        self,
        inputs: list[list[str]],
        outputs_mask: list[str],
        outputs_bet: list[str],
        num_processes: int,
    ) -> None:
        for _im, _bet, out in zip(inputs, outputs_mask, outputs_bet):
            _write_nifti(Path(out))

    monkeypatch.setattr(runner_mod.HDBETRunner, "_apply_masks", _apply_masks)

    return predictor
