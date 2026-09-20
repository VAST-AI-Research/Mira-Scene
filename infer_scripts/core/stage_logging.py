"""Persistent stdout/stderr logging for standalone stage entry points.

The pipeline additionally publishes per-case logs.  This module guarantees
that invoking any stage directly still leaves a timestamped diagnostic log
under the selected output root.
"""

from __future__ import annotations

import os
import shlex
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class _Tee:
    def __init__(self, terminal, log_handle):
        self.terminal = terminal
        self.log_handle = log_handle

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log_handle.write(value)
        self.log_handle.flush()
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")


def _argument_value(flags: tuple[str, ...]) -> str | None:
    """Read a path option without taking ownership of the stage's argparse."""
    found = None
    for index, token in enumerate(sys.argv[1:]):
        for flag in flags:
            if token == flag and index + 2 <= len(sys.argv) - 1:
                found = sys.argv[index + 2]
            elif token.startswith(flag + "="):
                found = token.split("=", 1)[1]
    return found


def _sharded_log_name(log_name: str) -> str:
    """Keep standalone stage logs separate when several workers share a root."""
    raw_count = _argument_value(("--num_shards",))
    raw_id = _argument_value(("--shard_id",))
    try:
        count = int(raw_count) if raw_count is not None else 1
        shard_id = int(raw_id) if raw_id is not None else 0
    except ValueError:
        return log_name
    if count <= 1:
        return log_name
    path = Path(log_name)
    return f"{path.stem}_shard_{shard_id:02d}{path.suffix}"


def run_logged(
    main: Callable[[], None],
    log_name: str,
    *,
    primary_root_flags: tuple[str, ...],
    fallback_root_flags: tuple[str, ...] = (),
) -> None:
    """Run ``main`` while teeing both output streams to ``logs/log_name``."""
    # Keep introspection side-effect free: all stage CLIs must expose --help
    # even on machines without writable model/output directories.
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        main()
        return
    raw_root = _argument_value(primary_root_flags) or _argument_value(fallback_root_flags)
    root = Path(raw_root).expanduser().resolve() if raw_root else Path.cwd()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / _sharded_log_name(log_name)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## {datetime.now(timezone.utc).isoformat()}\n"
            f"cwd: {os.getcwd()}\n"
            f"command: {shlex.join(sys.argv)}\n"
        )
        handle.flush()
        with redirect_stdout(_Tee(sys.stdout, handle)), redirect_stderr(_Tee(sys.stderr, handle)):
            try:
                main()
            except BaseException:
                # The interpreter prints an uncaught exception only after the
                # redirect context has unwound, so persist it explicitly.
                traceback.print_exc(file=handle)
                handle.flush()
                raise
