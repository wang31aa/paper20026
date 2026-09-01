#!/usr/bin/env python3
"""Shared, standard-library-only helpers for the prospective physical contract."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

PROTOCOL_ID = "PHYS-E2E-OCT-R34"
ARMS = ("observer_loop", "oracle_target", "no_target")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ContractError(ValueError):
    """A fail-closed evidence-contract violation."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            if not header:
                raise ContractError(f"CSV has no header: {path}")
            if len(header) != len(set(header)):
                raise ContractError(f"CSV has duplicate columns: {path}")
            return header, list(reader)
    except OSError as exc:
        raise ContractError(f"cannot read CSV {path}: {exc}") from exc


def require_columns(header: Iterable[str], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(header))
    if missing:
        raise ContractError(f"{label} missing required columns: {missing}")


def require_sha256(value: str, label: str) -> None:
    if not SHA256_RE.fullmatch(value or ""):
        raise ContractError(f"{label} is not lowercase SHA-256")


def require_safe_id(value: str, label: str) -> None:
    if not SAFE_ID_RE.fullmatch(value or ""):
        raise ContractError(f"{label} is not a safe identifier: {value!r}")


def parse_bool(value: str, label: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ContractError(f"{label} must be true or false, got {value!r}")


def parse_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be an integer, got {value!r}") from exc
    return parsed


def parse_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be numeric, got {value!r}") from exc
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise ContractError(f"{label} must be finite")
    return parsed


def resolve_beneath(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ContractError(f"unsafe relative path: {relative!r}")
    root_resolved = root.resolve()
    candidate = (root / rel).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ContractError(f"path escapes raw root: {relative!r}")
    return candidate
