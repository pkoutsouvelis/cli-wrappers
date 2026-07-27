"""Path and part-slicing utilities shared across wrapper tools."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, TypeVar

T = TypeVar("T")


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


def normalize_part_indices(
    part_idx: int | Sequence[int],
    num_parts: int,
) -> list[int]:
    """Validate and normalize ``part_idx`` for a ``num_parts`` split.

    Returns a sorted, deduplicated list of part indices in ``[0, num_parts)``.
    """
    if not isinstance(num_parts, int) or isinstance(num_parts, bool) or num_parts < 1:
        raise ValueError(f"num_parts must be an integer >= 1, got {num_parts!r}")

    if isinstance(part_idx, bool) or not isinstance(part_idx, (int, Sequence)):
        raise ValueError(
            "part_idx must be an int or a sequence of ints, "
            f"got {type(part_idx).__name__}"
        )

    if isinstance(part_idx, int):
        indices = [part_idx]
    else:
        if isinstance(part_idx, (str, bytes, bytearray)):
            raise ValueError(
                "part_idx must be an int or a sequence of ints, "
                f"got {type(part_idx).__name__}"
            )
        indices = list(part_idx)
        if not indices:
            raise ValueError("part_idx sequence must not be empty")
        if any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
            raise ValueError("part_idx sequence must contain only integers")

    invalid = [i for i in indices if i < 0 or i >= num_parts]
    if invalid:
        raise ValueError(
            f"part_idx values must be in [0, {num_parts}); got invalid {invalid}"
        )

    return sorted(set(indices))


def slice_by_parts(
    items: list[T],
    num_parts: int,
    part_indices: Sequence[int],
) -> list[T]:
    """Select contiguous chunks of ``items`` for the given part indices.

    The sequence is partitioned into ``num_parts`` contiguous slices of nearly
    equal length (sizes differ by at most one). Selected parts are concatenated
    in ascending ``part_indices`` order.
    """
    n = len(items)
    selected: list[T] = []
    for part_id in part_indices:
        start = (part_id * n) // num_parts
        end = ((part_id + 1) * n) // num_parts
        selected.extend(items[start:end])
    return selected
