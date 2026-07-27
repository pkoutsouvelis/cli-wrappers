"""Shared pytest fixtures for synthseg-wrapper tests.

The real Photo-SynthSeg predict script is never invoked. Instead, the runner's
``_run_subprocess`` hook is replaced with a fake that:

1. Parses the command line just enough to recover the ``--i``, ``--o``,
   ``--post``, ``--resample``, ``--vol``, and ``--qc`` arguments.
2. Reads the list files and writes tiny NIfTIs at each output path; for
   ``--vol`` / ``--qc`` it writes a minimal CSV.
3. Captures the call for assertions.

A ``fake_photo_synthseg`` fixture also fabricates a minimal Photo-SynthSeg
layout (``scripts/commands/SynthSeg_predict.py``) so the runner's constructor
file-existence check passes.
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
def fake_photo_synthseg(tmp_path: Path) -> Path:
    """Create a minimal Photo-SynthSeg directory layout for path checks."""
    home = tmp_path / "Photo-SynthSeg"
    (home / "scripts" / "commands").mkdir(parents=True)
    (home / "scripts" / "commands" / "SynthSeg_predict.py").write_text(
        "# stub for tests\n"
    )
    (home / "models").mkdir()
    return home


@pytest.fixture()
def bids_tree(tmp_path: Path) -> Path:
    """Two-subject BIDS-style tree with T1w files."""
    root = tmp_path / "bids"
    for sub in ("sub-01", "sub-02"):
        anat = root / sub / "ses-01" / "anat"
        _write_nifti(anat / f"{sub}_ses-01_T1w.nii.gz")
    return root


@pytest.fixture()
def single_nifti(tmp_path: Path) -> Path:
    p = tmp_path / "alpha" / "scan_T1w.nii.gz"
    _write_nifti(p)
    return p


class FakeSubprocess:
    """Replacement for ``SynthSegRunner._run_subprocess``.

    Mimics what SynthSeg_predict.py would do: read the input list, write
    segmentations (and optional posteriors / resampled NIfTIs / volumes /
    qc) at the paths the runner already planned for it. The parsed list
    contents are snapshotted on each call so tests can inspect them after
    the runner's tempdir has been cleaned up.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        # Per-call snapshots, indexed in lock-step with ``calls``.
        self.inputs: list[list[str]] = []
        self.segmentations: list[list[str]] = []
        self.posteriors: list[list[str]] = []
        self.resampled: list[list[str]] = []
        self.volumes: list[list[str]] = []
        self.qc: list[list[str]] = []

    @staticmethod
    def _arg(cmd: list[str], flag: str) -> str | None:
        try:
            i = cmd.index(flag)
        except ValueError:
            return None
        return cmd[i + 1]

    @staticmethod
    def _read_list(path: str) -> list[str]:
        return [line for line in Path(path).read_text().splitlines() if line]

    def __call__(self, cmd: list[str]) -> None:
        self.calls.append(list(cmd))

        i_path = self._arg(cmd, "--i")
        self.inputs.append(self._read_list(i_path) if i_path else [])

        seg_list = self._arg(cmd, "--o")
        seg_paths = self._read_list(seg_list) if seg_list else []
        self.segmentations.append(seg_paths)
        for p in seg_paths:
            _write_nifti(Path(p))

        for flag, sink in [("--post", self.posteriors), ("--resample", self.resampled)]:
            list_path = self._arg(cmd, flag)
            paths = self._read_list(list_path) if list_path else []
            sink.append(paths)
            for p in paths:
                _write_nifti(Path(p))

        for flag in ("--vol", "--qc"):
            list_path = self._arg(cmd, flag)
            if not list_path:
                continue
            paths = self._read_list(list_path)
            if flag == "--vol":
                self.volumes.append(paths)
            else:
                self.qc.append(paths)
            for csv_path in paths:
                p = Path(csv_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("region,volume\n")


@pytest.fixture()
def patch_synthseg(monkeypatch: pytest.MonkeyPatch) -> FakeSubprocess:
    """Patch the runner's subprocess hook with a writing fake."""
    from synthseg_wrapper import runner as runner_mod

    fake = FakeSubprocess()

    def _run(self: Any, cmd: list[str]) -> None:
        fake(cmd)

    monkeypatch.setattr(runner_mod.SynthSegRunner, "_run_subprocess", _run)
    return fake
