"""Append-only, hash-chained JSON-lines event ledger."""

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


GENESIS = "0" * 64


def canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


class AppendOnlyHashLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError("append-only log path must be new")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self._head = GENESIS

    @property
    def head(self) -> str:
        return self._head

    def append(self, event: Mapping[str, object]) -> str:
        envelope = {
            "record_sequence": self._sequence,
            "previous_record_hash": self._head,
            "event": dict(event),
        }
        digest = hashlib.sha256(canonical_bytes(envelope)).hexdigest()
        record = dict(envelope, record_sha256=digest)
        line = canonical_bytes(record) + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(self.path, flags, 0o600)
        try:
            view = memoryview(line)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("append-only log write made no progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        self._sequence += 1
        self._head = digest
        return digest


def verify(path: Path) -> str:
    expected_sequence = 0
    previous = GENESIS
    with Path(path).open("rb") as handle:
        for raw in handle:
            record = json.loads(raw)
            digest = record.pop("record_sha256")
            if record.get("record_sequence") != expected_sequence:
                raise ValueError("non-monotone record sequence")
            if record.get("previous_record_hash") != previous:
                raise ValueError("broken previous-record hash")
            actual = hashlib.sha256(canonical_bytes(record)).hexdigest()
            if actual != digest:
                raise ValueError("record hash mismatch")
            previous = digest
            expected_sequence += 1
    return previous
