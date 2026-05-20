"""Path utilities shared across wrapper tools."""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: Path | str) -> Path:
    """Expand user (``~``) and resolve to an absolute path."""
    return Path(path).expanduser().resolve()


def get_ext(path: Path | str) -> str:
    """Return the full multi-part file extension with leading dots.

    Examples:
        ``"scan.nii.gz"`` -> ``".nii.gz"``
        ``"scan.nii"``    -> ``".nii"``
    """
    return "".join(Path(path).suffixes)
