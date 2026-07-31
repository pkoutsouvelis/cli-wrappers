"""Base runner for CLI-tool wrappers.

:class:`BaseRunner` absorbs the parts every tool wrapper repeats:

- Input resolution: a single file, a list of files, a text file listing
  paths (one per line), or a dataset mapping with ``root`` + ``patterns``
  (+ optional ``levels`` / ``filters``) handed to ``nifti-finder``.
- Output planning: given a list of :class:`OutputSpec` items, build the parallel
  input/output path lists (mirrored under ``output_dir`` when in dataset mode)
  with overwrite semantics.
- Optional contiguous part slicing of a planned path mapping via
  :meth:`BaseRunner._slice_plan`.
- A configured logger.

Subclasses implement ``__call__`` to drive the tool-specific inference, using
the planned paths from :meth:`BaseRunner._plan_outputs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TypeAlias

from cliwrap_core.explorer_factory import get_data_explorer
from cliwrap_core.logging_utils import LogLevel, setup_logger
from cliwrap_core.utils import (
    ensure_under_root,
    get_ext,
    normalize_part_indices,
    read_path_list,
    resolve_path,
    slice_by_parts,
)

InputData: TypeAlias = Path | str | list[Path] | list[str] | dict[str, Any]
PartIdx: TypeAlias = int | Sequence[int]


@dataclass(frozen=True)
class OutputSpec:
    """Declarative description of one output stream a tool writes per input.

    Attributes:
        name: A short identifier (e.g. ``"mask"``, ``"segmentation"``). Used as
            the dict key in the plan returned by :meth:`BaseRunner._plan_outputs`.
        suffix: The suffix appended to the input stem; empty string keeps the
            stem as-is. A separating ``_`` is inserted automatically iff
            ``suffix`` is non-empty.
        save: Whether the user asked to keep this output. Outputs with
            ``save=False`` are still produced (some tools always write them)
            but cleanup is the subclass's responsibility.
        required: If True, the runner skips an input whenever this output
            already exists and ``overwrite`` is False. Tools that *always*
            produce a given output should mark it required.
    """

    name: str
    suffix: str
    save: bool = True
    required: bool = False


class BaseRunner:
    """Shared scaffolding for tool-specific runners.

    Args:
        data: Input spec: a single Path/str, a list of Paths/strs, an explicit
            path mapping ``{"files": [...], "root":?}`` or
            ``{"from_file": "/path/to/paths.txt", "root":?}``, or a dataset
            mapping ``{"root": ..., "patterns": ..., "levels": ..., "filters": ...}``.
            ``levels`` and ``filters`` are optional; omit ``levels`` for a flat
            recursive scan. ``from_file`` is a text file with one filepath per
            line (blank lines and ``#`` comments ignored). Optional ``root`` on
            ``files`` / ``from_file`` mirrors each path's root-relative parent
            under ``output_dir`` and requires every path to lie under ``root``.
        output_dir: Root output directory. When ``None`` and ``data`` is a
            single file, list, or from_file without ``root``, outputs land next
            to each input; when ``None`` and a ``root`` is set (explicit or
            dataset), outputs land next to inputs under that root.
        overwrite: Whether to re-process inputs whose required outputs exist.
        log_dir: Optional directory for a timestamped run log.
        log_level: Minimum log level.
        logger_name: Python logger name (e.g. ``"hdbet_wrapper"``).
        logger_label: Bracketed log prefix (e.g. ``"HD-BET WRAPPER"``).
        log_file_prefix: Filename stem for the per-run log file.
    """

    def __init__(
        self,
        data: InputData,
        output_dir: Path | str | None = None,
        overwrite: bool = True,
        log_dir: Path | str | None = None,
        log_level: LogLevel = "INFO",
        *,
        logger_name: str,
        logger_label: str,
        log_file_prefix: str = "run",
    ) -> None:
        if not isinstance(data, (Path, str, list, dict)):
            raise ValueError(
                f"data must be a Path, str, list, or dict, got {type(data).__name__}"
            )

        if not isinstance(output_dir, (Path, str, type(None))):
            raise ValueError(
                f"output_dir must be a Path, str, or None, got {type(output_dir).__name__}"
            )

        if not isinstance(overwrite, bool):
            raise ValueError(
                f"overwrite must be a boolean, got {type(overwrite).__name__}"
            )
        self._overwrite = overwrite

        if log_dir is not None and not isinstance(log_dir, (Path, str)):
            raise ValueError(
                f"log_dir must be a Path, str, or None, got {type(log_dir).__name__}"
            )
        if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError(
                f"log_level must be one of 'DEBUG', 'INFO', 'WARNING', or 'ERROR'; "
                f"got {log_level!r}"
            )
        self._logger = setup_logger(
            log_dir=log_dir,
            level=log_level,
            name=logger_name,
            label=logger_label,
            file_prefix=log_file_prefix,
        )

        self._root, self._input_files = self._resolve_data(data)
        if output_dir is not None:
            self._output_dir: Path | None = resolve_path(output_dir)
        elif self._root is not None:
            self._output_dir = self._root
        else:
            self._output_dir = None

    def _resolve_data(self, data: InputData) -> tuple[Path | None, list[Path]]:
        if isinstance(data, (Path, str)):
            return None, [resolve_path(data)]

        if isinstance(data, list):
            if any(not isinstance(p, (Path, str)) for p in data):
                raise ValueError("Input list must contain only Path or str objects")
            return None, [resolve_path(p) for p in data]

        if isinstance(data, dict):
            if "from_file" in data or "files" in data:
                return self._resolve_explicit_paths(data)

            self._logger.info("Instantiating data explorer...")

            if "root" not in data:
                raise ValueError("`root` key is required in `data` dictionary")
            if not isinstance(data["root"], (Path, str)):
                raise ValueError(
                    f"`root` must be a Path or str object, got {type(data['root']).__name__}"
                )
            if "patterns" not in data:
                raise ValueError("`patterns` key is required in `data` dictionary")

            try:
                explorer = get_data_explorer(
                    patterns=data["patterns"],
                    levels=data.get("levels"),
                    filters=data.get("filters"),
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to bind arguments to data explorer; see "
                    "``nifti_finder``'s documentation of ``FileFinder`` for "
                    "more details."
                ) from e

            self._logger.info("Data explorer instantiated successfully.")
            self._logger.info("Finding files...")
            root = resolve_path(data["root"])
            files = explorer.list(root, sort=True, unique=True)
            self._logger.info("Found %d unique files.", len(files))
            return root, files

        raise AssertionError("unreachable")  # pragma: no cover

    def _resolve_explicit_paths(
        self, data: dict[str, Any]
    ) -> tuple[Path | None, list[Path]]:
        """Resolve ``files`` / ``from_file`` payloads with optional ``root``."""
        if "from_file" in data and "files" in data:
            raise ValueError("data cannot contain both 'from_file' and 'files'")
        if "patterns" in data:
            raise ValueError(
                "data cannot mix explicit paths ('files' / 'from_file') with "
                "'patterns'"
            )

        unknown = set(data.keys()) - {"from_file", "files", "root"}
        if unknown:
            raise ValueError(
                "explicit path data cannot contain keys other than "
                f"['files', 'from_file', 'root']; got unexpected keys {sorted(unknown)}"
            )

        if "from_file" in data:
            from_file = data["from_file"]
            if not isinstance(from_file, (Path, str)):
                raise ValueError(
                    "`from_file` must be a Path or str, "
                    f"got {type(from_file).__name__}"
                )
            self._logger.info("Reading paths from: %s", from_file)
            files = read_path_list(from_file)
            self._logger.info("Loaded %d path(s) from %s.", len(files), from_file)
        else:
            raw_files = data["files"]
            if not isinstance(raw_files, list):
                raise ValueError(
                    "`files` must be a list of paths, "
                    f"got {type(raw_files).__name__}"
                )
            if any(not isinstance(p, (Path, str)) for p in raw_files):
                raise ValueError("Input list must contain only Path or str objects")
            files = [resolve_path(p) for p in raw_files]

        root_raw = data.get("root")
        if root_raw is None:
            return None, files

        if not isinstance(root_raw, (Path, str)):
            raise ValueError(
                f"`root` must be a Path or str object, got {type(root_raw).__name__}"
            )
        root = resolve_path(root_raw)
        ensure_under_root(files, root)
        self._logger.info(
            "Using root %s for output mirroring (%d path(s)).", root, len(files)
        )
        return root, files

    def _out_parent_for(self, p: Path) -> Path:
        """Return the directory in which outputs for ``p`` should be written."""
        if self._output_dir is None:
            return p.parent
        if self._root is not None:
            return self._output_dir / p.parent.relative_to(self._root)
        return self._output_dir

    def _stem_and_ext(self, p: Path) -> tuple[str, str]:
        ext = get_ext(p)
        return p.name[: -len(ext)] if ext else p.name, ext

    def _output_path(self, p: Path, suffix: str) -> Path:
        stem, ext = self._stem_and_ext(p)
        sep = "_" if suffix else ""
        return self._out_parent_for(p) / f"{stem}{sep}{suffix}{ext}"

    def _plan_outputs(self, specs: list[OutputSpec]) -> dict[str, list[str]]:
        """Compute parallel input/output path lists.

        For each input file in ``self._input_files``, an output path is built
        per :class:`OutputSpec`. Files whose required outputs already exist are
        skipped when ``overwrite`` is False; the skip is enforced consistently
        across all required outputs.

        Returns:
            A mapping with:
              - ``"__inputs__"``: ``list[str]`` of kept input paths
              - ``<spec.name>``: ``list[str]`` of output paths per spec, in
                lock-step with ``__inputs__``
            All values are strings (most underlying CLIs prefer strings).
        """
        plans: dict[str, list[str]] = {"__inputs__": []}
        for spec in specs:
            plans[spec.name] = []

        for p in self._input_files:
            out_parent = self._out_parent_for(p)
            out_parent.mkdir(parents=True, exist_ok=True)

            candidate_outputs = {
                spec.name: self._output_path(p, spec.suffix) for spec in specs
            }

            if not self._overwrite and any(
                spec.required and candidate_outputs[spec.name].exists()
                for spec in specs
            ):
                continue

            plans["__inputs__"].append(str(p))
            for spec in specs:
                plans[spec.name].append(str(candidate_outputs[spec.name]))

        return plans

    def _slice_plan(
        self,
        plans: dict[str, list[Any]],
        *,
        num_parts: int = 1,
        part_idx: PartIdx = 0,
    ) -> dict[str, list[Any]]:
        """Keep only the planned pairs belonging to ``part_idx`` of ``num_parts``.

        Every list value in ``plans`` is sliced with the same contiguous
        partition so parallel input/output streams stay aligned. Logs the
        count before and after slicing.
        """
        if not plans:
            return plans

        lengths = {key: len(values) for key, values in plans.items()}
        if len(set(lengths.values())) > 1:
            raise RuntimeError(
                "Internal error: planned path lists have unequal lengths: " f"{lengths}"
            )

        part_indices = normalize_part_indices(part_idx, num_parts)
        n_before = next(iter(lengths.values()), 0)
        self._logger.info(
            "Planned %d input/output pair(s) before part slicing "
            "(num_parts=%d, part_idx=%s).",
            n_before,
            num_parts,
            part_indices if len(part_indices) > 1 else part_indices[0],
        )

        if num_parts == 1 and part_indices == [0]:
            self._logger.info(
                "Part slicing is a no-op; keeping all %d pair(s).", n_before
            )
            return plans

        sliced = {
            key: slice_by_parts(values, num_parts, part_indices)
            for key, values in plans.items()
        }
        self._logger.info(
            "Kept %d of %d input/output pair(s) after part slicing.",
            len(next(iter(sliced.values()), [])),
            n_before,
        )
        return sliced

    def __call__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise NotImplementedError("Subclasses must implement __call__")
