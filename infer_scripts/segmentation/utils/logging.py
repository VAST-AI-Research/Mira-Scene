import logging
import os
import sys

_ROOT_NAME = "mira_stage1"
_FMT = logging.Formatter("%(message)s")


def get_logger(stage_name=None):
    """Return a logger that writes ``%(message)s`` to stdout.

    Multiple calls with the same ``stage_name`` return the same logger and
    avoid duplicating the stdout handler.
    """
    name = _ROOT_NAME if stage_name is None else f"{_ROOT_NAME}.{stage_name}"
    logger = logging.getLogger(name)
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
               for h in logger.handlers):
        logger.setLevel(logging.INFO)
        logger.propagate = False
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(_FMT)
        logger.addHandler(ch)
    return logger


def attach_file_handler(logger, log_path):
    """Attach a FileHandler that mirrors the logger output to ``log_path``.

    Returns the handler so the caller can detach it later via
    ``logger.removeHandler(h); h.close()``.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(_FMT)
    logger.addHandler(fh)
    return fh
