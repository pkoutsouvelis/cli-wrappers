"""Tests for :mod:`cliwrap_core.logging_utils`."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cliwrap_core.logging_utils import get_logger, setup_logger

_LOGGER_NAME = "cliwrap_test_logger"
_LOGGER_LABEL = "TEST WRAPPER"


@pytest.fixture(autouse=True)
def _reset_logger():
    yield
    logger = logging.getLogger(_LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def test_setup_logger_stream_only_by_default():
    logger = setup_logger(name=_LOGGER_NAME, label=_LOGGER_LABEL)
    assert logger.name == _LOGGER_NAME
    handler_types = {type(h).__name__ for h in logger.handlers}
    assert handler_types == {"StreamHandler"}
    assert logger.level == logging.INFO


def test_setup_logger_with_log_dir_creates_file_handler(tmp_path: Path):
    log_dir = tmp_path / "logs"
    logger = setup_logger(
        log_dir=log_dir,
        level="DEBUG",
        name=_LOGGER_NAME,
        label=_LOGGER_LABEL,
        file_prefix="testrun",
    )

    assert logger.level == logging.DEBUG
    handler_types = {type(h).__name__ for h in logger.handlers}
    assert handler_types == {"StreamHandler", "FileHandler"}

    log_files = list(log_dir.glob("testrun_*.log"))
    assert len(log_files) == 1


def test_setup_logger_resets_handlers_on_repeat_call(tmp_path: Path):
    setup_logger(
        log_dir=tmp_path / "a", level="INFO", name=_LOGGER_NAME, label=_LOGGER_LABEL
    )
    logger = setup_logger(
        log_dir=tmp_path / "b", level="INFO", name=_LOGGER_NAME, label=_LOGGER_LABEL
    )

    handler_types = [type(h).__name__ for h in logger.handlers]
    assert handler_types.count("StreamHandler") == 1
    assert handler_types.count("FileHandler") == 1


def test_setup_logger_writes_log_lines(tmp_path: Path):
    logger = setup_logger(
        log_dir=tmp_path / "logs",
        level="INFO",
        name=_LOGGER_NAME,
        label="HD-BET WRAPPER",
        file_prefix="hdbet_run",
    )
    logger.info("hello world")

    [log_file] = (tmp_path / "logs").glob("hdbet_run_*.log")
    contents = log_file.read_text()
    assert "[HD-BET WRAPPER]" in contents
    assert "[INFO]" in contents
    assert "hello world" in contents


def test_get_logger_returns_same_named_logger():
    setup_logger(name=_LOGGER_NAME, label=_LOGGER_LABEL)
    assert get_logger(_LOGGER_NAME) is logging.getLogger(_LOGGER_NAME)
