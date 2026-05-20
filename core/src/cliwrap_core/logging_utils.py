"""Logger setup shared by wrapper tools.

Builds a named logger that always streams to stdout and optionally also writes
to a timestamped file under ``log_dir``. Each tool picks its own ``name`` and
``label`` so log lines are visibly attributable.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from cliwrap_core.utils import resolve_path

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def _format_for(label: str) -> str:
    return f"%(asctime)s [{label}] [%(levelname)s] %(message)s"


def setup_logger(
    log_dir: Path | str | None = None,
    level: LogLevel = "INFO",
    *,
    name: str,
    label: str,
    file_prefix: str = "run",
) -> logging.Logger:
    """Configure and return a named logger.

    A ``StreamHandler`` to stdout is always attached. When ``log_dir`` is given,
    a ``FileHandler`` writing ``<file_prefix>_<timestamp>.log`` is added as well.

    Args:
        log_dir: Directory to write a timestamped run log. No file handler when None.
        level: Minimum log level.
        name: Logger name (e.g. ``"hdbet_wrapper"`` or ``"synthseg_wrapper"``).
        label: Bracketed prefix shown in every log line (e.g. ``"HD-BET WRAPPER"``).
        file_prefix: Filename stem for the per-run log file.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter(_format_for(label))

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        log_dir_p = resolve_path(log_dir)
        log_dir_p.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir_p / f"{file_prefix}_{ts}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger by name without (re)configuring it."""
    return logging.getLogger(name)
