"""Shared building blocks for CLI-tool wrappers.

This package collects the parts that every wrapper tends to need:

- File discovery via ``nifti-finder`` (:mod:`cliwrap_core.explorer_factory`)
- A small YAML config helper with reusable section parsers
  (:mod:`cliwrap_core.config`)
- A stdout + optional file logger (:mod:`cliwrap_core.logging_utils`)
- A :class:`~cliwrap_core.base_runner.BaseRunner` that absorbs input resolution
  and output-path planning so each tool only writes its own ``__call__``.
"""

from cliwrap_core.base_runner import BaseRunner, OutputSpec
from cliwrap_core.config import (
    load_yaml,
    parse_input,
    parse_logging,
    require_keys,
    reject_unknown_keys,
)
from cliwrap_core.explorer_factory import get_data_explorer
from cliwrap_core.logging_utils import LogLevel, get_logger, setup_logger
from cliwrap_core.utils import (
    ensure_under_root,
    get_ext,
    normalize_part_indices,
    read_path_list,
    resolve_path,
    slice_by_parts,
)

__all__ = [
    "BaseRunner",
    "OutputSpec",
    "LogLevel",
    "ensure_under_root",
    "get_data_explorer",
    "get_ext",
    "get_logger",
    "load_yaml",
    "normalize_part_indices",
    "parse_input",
    "parse_logging",
    "read_path_list",
    "reject_unknown_keys",
    "require_keys",
    "resolve_path",
    "setup_logger",
    "slice_by_parts",
]
__version__ = "0.1.0"
