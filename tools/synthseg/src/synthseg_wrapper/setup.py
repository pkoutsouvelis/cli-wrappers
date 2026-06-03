"""Provision a Photo-SynthSeg clone and its model weights.

Mirrors the work done in the project's reference Dockerfile, sans the OS-level
``apt-get`` step:

1. ``git clone --branch synthseg_tf2.15 --single-branch`` Photo-SynthSeg
2. ``unzip`` the user-supplied ``SynthSeg_models.zip`` into the clone
3. Rename the unzipped folder to ``models/`` (matches ``mv SynthSeg_models models``)

The weights archive must be downloaded manually first (the SharePoint URL the
authors host is not directly ``wget``-able):

    https://mitprod-my.sharepoint.com/personal/bbillot_mit_edu/...
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_REPO_URL = "https://github.com/MGH-LEMoN/Photo-SynthSeg/tree/synthseg_tf2.15"
_BRANCH = "synthseg_tf2.15"
_WEIGHTS_DIR_IN_ZIP = "SynthSeg_models"


def _log(msg: str) -> None:
    print(f"[synthseg-setup] {msg}", flush=True)


def _resolve(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def clone_photo_synthseg(synthseg_home: Path, force: bool = False) -> None:
    if synthseg_home.exists():
        if not force:
            _log(
                f"Clone target already exists: {synthseg_home}. "
                "Pass --force to wipe and re-clone."
            )
            return
        _log(f"Removing existing {synthseg_home}")
        shutil.rmtree(synthseg_home)

    synthseg_home.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "clone",
        "--branch",
        _BRANCH,
        "--single-branch",
        _REPO_URL,
        str(synthseg_home),
    ]
    _log(" ".join(cmd))
    subprocess.run(cmd, check=True)


def install_weights(synthseg_home: Path, weights_zip: Path) -> None:
    if not weights_zip.exists():
        raise FileNotFoundError(
            f"Weights archive not found: {weights_zip}. Download "
            "SynthSeg_models.zip from the SharePoint link in the SynthSeg "
            "documentation first."
        )

    models_dir = synthseg_home / "models"
    if models_dir.exists():
        _log(f"Removing existing {models_dir}")
        shutil.rmtree(models_dir)

    _log(f"Unzipping {weights_zip} -> {synthseg_home}")
    with zipfile.ZipFile(weights_zip, "r") as zf:
        zf.extractall(synthseg_home)

    # Match the Dockerfile's `mv SynthSeg_models models`.
    extracted = synthseg_home / _WEIGHTS_DIR_IN_ZIP
    if extracted.exists():
        extracted.rename(models_dir)
    elif not models_dir.exists():
        raise RuntimeError(
            f"Expected unzipped folder '{_WEIGHTS_DIR_IN_ZIP}' or 'models' "
            f"inside {synthseg_home}; found neither."
        )

    # Drop the mac archive noise the Dockerfile also strips.
    macosx_junk = synthseg_home / "__MACOSX"
    if macosx_junk.exists():
        shutil.rmtree(macosx_junk)

    _log(f"Models installed at {models_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthseg-setup",
        description=(
            "Clone Photo-SynthSeg and unpack model weights. Mirrors the project's "
            "reference Dockerfile, sans OS-level packages."
        ),
    )
    parser.add_argument(
        "--synthseg-home",
        type=Path,
        required=True,
        help="Destination directory for the Photo-SynthSeg clone.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=False,
        help="Path to a local SynthSeg_models.zip. If omitted, only the clone is performed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe an existing clone target before cloning.",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip git cloning and only (re)install weights into an existing clone.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    synthseg_home = _resolve(args.synthseg_home)

    if not args.skip_clone:
        clone_photo_synthseg(synthseg_home, force=args.force)

    if args.weights is not None:
        install_weights(synthseg_home, _resolve(args.weights))
    else:
        _log("No --weights provided; skipping weight installation.")

    _log("Setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
