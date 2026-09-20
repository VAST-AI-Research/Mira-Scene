"""Small, dependency-light filesystem helpers shared by inference stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        # mkstemp deliberately creates 0600 files.  These artifacts live on a
        # shared PFS and inference is commonly launched as root inside a pod,
        # so retaining that mode makes JSON results unreadable to the host
        # user. Preserve an existing artifact's mode, otherwise publish 0644.
        previous = path.stat() if path.exists() else None
        os.fchmod(fd, (previous.st_mode & 0o777) if previous else 0o644)
        # If a privileged pod updates a case directory owned by the PFS user,
        # keep the replacement file under that same ownership.  os.replace()
        # otherwise swaps in the root-owned temporary inode.
        if hasattr(os, "fchown") and os.geteuid() == 0:
            owner = previous or path.parent.stat()
            os.fchown(fd, owner.st_uid, owner.st_gid)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def files_complete(case_dir: Path, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if not any(path.is_file() and path.stat().st_size > 0 for path in case_dir.glob(pattern)):
            return False
    return True
