"""Path and part-slicing utilities shared across wrapper tools."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def resolve_path(path: Path | str) -> Path:
    """Expand user (``~``) and resolve to an absolute path."""
    return Path(path).expanduser().resolve()


def coerce_path(path: Path | str, *, resolve: bool = True) -> Path:
    """Expand ``~``; optionally ``resolve()`` (filesystem realpath)."""
    p = Path(path).expanduser()
    return p.resolve() if resolve else p


def ensure_under_root(paths: Sequence[Path], root: Path) -> None:
    """Raise ``ValueError`` if any path is not under ``root``.

    Uses ``Path.relative_to`` only (no per-path ``resolve()``), so ``paths``
    must already be in a form compatible with ``root`` (typically both
    absolute and, if ``root`` was resolved, already realpath'd).
    """
    for path in paths:
        try:
            path.relative_to(root)
        except ValueError as e:
            raise ValueError(f"Input path {path} is not under root {root}") from e


def maximal_directories(dirs: Iterable[Path]) -> list[Path]:
    """Return directories that are not strict ancestors of another in ``dirs``.

    ``Path.mkdir(parents=True)`` on these still creates any skipped ancestors,
    so fewer ``mkdir`` calls are needed when both a parent and a child appear.
    """
    unique = list({Path(d) for d in dirs})
    unique.sort(key=lambda p: len(p.parts), reverse=True)
    kept: list[Path] = []
    for d in unique:
        if any(_is_strict_descendant(k, d) for k in kept):
            continue
        kept.append(d)
    return kept


def _is_strict_descendant(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return child != parent


def read_path_list(path: Path | str, *, resolve_paths: bool = True) -> list[Path]:
    """Read absolute/relative filepaths from a text file (one path per line).

    Blank lines and lines whose first non-whitespace character is ``#`` are
    ignored. Each remaining line is stripped and coerced via
    :func:`coerce_path`. The list file itself is always resolved so it can be
    opened reliably.

    Args:
        path: Path to the text file.
        resolve_paths: If True (default), ``resolve()`` each listed filepath.
            Set False to only expand ``~`` (much faster on large lists / slow
            filesystems when paths are already absolute).
    """
    list_path = resolve_path(path)
    if not list_path.is_file():
        raise FileNotFoundError(f"Path list file not found: {list_path}")

    paths: list[Path] = []
    with list_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(coerce_path(line, resolve=resolve_paths))

    if not paths:
        raise ValueError(f"Path list file is empty (no paths): {list_path}")

    return paths


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
