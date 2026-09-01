#!/usr/bin/env python3
"""Fail-closed validator for PHYS-E2E-OCT-R34 preregistration parameters.

Exit status 0 means that a completed parameter file is eligible to start the
registered acquisition. It never means that a physical experiment succeeded.
Draft files are blocked unless --draft-lint is explicitly requested.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = ROOT / "PARAMETER_SCHEMA.json"
PROTOCOL = ROOT / "PROTOCOL_MATH.md"
COMPLETED_STATUS = "preregistration_parameters_no_results"
DRAFT_STATUS = "draft"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
PLACEHOLDER_RE = re.compile(
    r"(?:\bTBD\b|\bTODO\b|REPLACE|PLACEHOLDER|EXAMPLE_ONLY|FILL_ME)",
    re.IGNORECASE,
)
FORBIDDEN_OUTCOME_KEYS = {
    "result",
    "results",
    "physical_result",
    "physical_results",
    "observed_outcome",
    "observed_outcomes",
    "observed_effect",
    "effect_estimate",
    "p_value",
    "confidence_interval",
    "primary_success",
    "physical_validation_passed",
    "manuscript_score",
}
SUPPORTED_SCHEMA_KEYWORDS = {
    "$schema",
    "$id",
    "$ref",
    "$defs",
    "title",
    "description",
    "type",
    "additionalProperties",
    "required",
    "properties",
    "const",
    "enum",
    "pattern",
    "format",
    "minLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
    "uniqueItems",
    "items",
    "anyOf",
    "x-semanticChecks",
}


class ValidationFailure(RuntimeError):
    """A fail-closed validation error."""


@dataclass
class Report:
    checks: list[str] = field(default_factory=list)

    def passed(self, message: str) -> None:
        self.checks.append(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationFailure("top-level JSON value must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationFailure(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")


def key_paths(value: Any, path: str = "$") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, key, child
            yield from key_paths(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from key_paths(child, f"{path}[{index}]")


def reject_outcome_fields(value: dict[str, Any]) -> None:
    bad = [
        path
        for path, key, _ in key_paths(value)
        if key.lower() in FORBIDDEN_OUTCOME_KEYS
    ]
    if bad:
        raise ValidationFailure(
            "physical outcome fields are prohibited in a parameter file: "
            + ", ".join(bad)
        )


def reject_placeholders_and_nulls(value: dict[str, Any]) -> None:
    errors: list[str] = []
    for path, item in walk(value):
        if item is None:
            errors.append(f"{path}=null")
        elif isinstance(item, str) and PLACEHOLDER_RE.search(item):
            errors.append(f"{path} contains placeholder text")
    if errors:
        raise ValidationFailure(
            "completed file contains unresolved values: " + "; ".join(errors[:20])
        )


def validate_hash_fields(value: dict[str, Any]) -> None:
    errors: list[str] = []
    for path, key, item in key_paths(value):
        if key.endswith("_sha256"):
            if not isinstance(item, str) or not SHA256_RE.fullmatch(item):
                errors.append(f"{path} is not a lowercase SHA-256")
            elif item == "0" * 64:
                errors.append(f"{path} is an all-zero SHA-256")
        elif key.endswith("_md5"):
            if not isinstance(item, str) or not MD5_RE.fullmatch(item):
                errors.append(f"{path} is not a lowercase MD5")
            elif item == "0" * 32:
                errors.append(f"{path} is an all-zero MD5")
    if errors:
        raise ValidationFailure("; ".join(errors))


def canonical_parameter_payload_sha256(value: dict[str, Any]) -> str:
    payload = copy.deepcopy(value)
    try:
        payload["freeze"].pop("canonical_parameter_payload_sha256")
    except (KeyError, TypeError) as exc:
        raise ValidationFailure(
            "freeze.canonical_parameter_payload_sha256 is required"
        ) from exc
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def resolve_local_ref(root_schema: dict[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise ValidationFailure(f"only local JSON Schema references are allowed: {reference}")
    current: Any = root_schema
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            raise ValidationFailure(f"unresolved JSON Schema reference: {reference}")
        current = current[token]
    return current


def validate_schema_document(schema: dict[str, Any]) -> None:
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValidationFailure("PARAMETER_SCHEMA must declare Draft 2020-12")
    if not isinstance(schema.get("$defs"), dict):
        raise ValidationFailure("PARAMETER_SCHEMA must define $defs")

    def inspect(node: Any, path: str) -> None:
        if isinstance(node, bool):
            return
        if not isinstance(node, dict):
            raise ValidationFailure(f"schema node {path} must be an object or boolean")
        unknown = set(node) - SUPPORTED_SCHEMA_KEYWORDS
        if unknown:
            raise ValidationFailure(
                f"unsupported schema keyword(s) at {path}: {sorted(unknown)}"
            )
        if "$ref" in node:
            resolve_local_ref(schema, node["$ref"])
        for key in ("properties", "$defs"):
            mapping = node.get(key, {})
            if not isinstance(mapping, dict):
                raise ValidationFailure(f"{path}.{key} must be an object")
            for name, child in mapping.items():
                inspect(child, f"{path}.{key}.{name}")
        if "items" in node:
            inspect(node["items"], f"{path}.items")
        for index, child in enumerate(node.get("anyOf", [])):
            inspect(child, f"{path}.anyOf[{index}]")

    inspect(schema, "$")


def matches_type(value: Any, declared: str) -> bool:
    if declared == "object":
        return isinstance(value, dict)
    if declared == "array":
        return isinstance(value, list)
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared == "null":
        return value is None
    raise ValidationFailure(f"unsupported JSON Schema type: {declared}")


def validate_format(value: str, declared: str, path: str) -> None:
    if declared == "date-time":
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationFailure(f"{path}: invalid date-time") from exc
        if parsed.tzinfo is None:
            raise ValidationFailure(f"{path}: date-time must include timezone")
    elif declared == "uri":
        parsed = urlparse(value)
        if not parsed.scheme or (
            parsed.scheme in {"http", "https"} and not parsed.netloc
        ):
            raise ValidationFailure(f"{path}: invalid absolute URI")
    else:
        raise ValidationFailure(f"{path}: unsupported format {declared!r}")


def validate_instance_against_schema(
    value: Any,
    node: Any,
    root_schema: dict[str, Any],
    path: str = "$",
) -> None:
    if node is True:
        return
    if node is False:
        raise ValidationFailure(f"{path}: schema is false")
    if "$ref" in node:
        validate_instance_against_schema(
            value, resolve_local_ref(root_schema, node["$ref"]), root_schema, path
        )
        return
    if "anyOf" in node:
        failures = []
        for option in node["anyOf"]:
            try:
                validate_instance_against_schema(value, option, root_schema, path)
                break
            except ValidationFailure as exc:
                failures.append(str(exc))
        else:
            raise ValidationFailure(
                f"{path}: no anyOf branch matched ({' | '.join(failures)})"
            )
    if "const" in node and value != node["const"]:
        raise ValidationFailure(f"{path}: value does not equal required const")
    if "enum" in node and value not in node["enum"]:
        raise ValidationFailure(f"{path}: value is not in enum")
    declared_type = node.get("type")
    if declared_type is not None and not matches_type(value, declared_type):
        raise ValidationFailure(f"{path}: expected {declared_type}")

    if isinstance(value, dict):
        required = node.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValidationFailure(f"{path}: missing required keys {missing}")
        properties = node.get("properties", {})
        if node.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise ValidationFailure(
                    f"{path}: additional properties prohibited: {sorted(extras)}"
                )
        for key, child in value.items():
            if key in properties:
                validate_instance_against_schema(
                    child, properties[key], root_schema, f"{path}.{key}"
                )

    if isinstance(value, list):
        if "minItems" in node and len(value) < node["minItems"]:
            raise ValidationFailure(f"{path}: fewer than minItems")
        if "maxItems" in node and len(value) > node["maxItems"]:
            raise ValidationFailure(f"{path}: more than maxItems")
        if node.get("uniqueItems"):
            encoded = [
                json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
                for item in value
            ]
            if len(encoded) != len(set(encoded)):
                raise ValidationFailure(f"{path}: array items are not unique")
        if "items" in node:
            for index, item in enumerate(value):
                validate_instance_against_schema(
                    item, node["items"], root_schema, f"{path}[{index}]"
                )

    if isinstance(value, str):
        if "minLength" in node and len(value) < node["minLength"]:
            raise ValidationFailure(f"{path}: string is shorter than minLength")
        if "pattern" in node and re.search(node["pattern"], value) is None:
            raise ValidationFailure(f"{path}: string does not match pattern")
        if "format" in node:
            validate_format(value, node["format"], path)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValidationFailure(f"{path}: number is not finite")
        if "minimum" in node and value < node["minimum"]:
            raise ValidationFailure(f"{path}: number is below minimum")
        if "maximum" in node and value > node["maximum"]:
            raise ValidationFailure(f"{path}: number is above maximum")
        if "exclusiveMinimum" in node and value <= node["exclusiveMinimum"]:
            raise ValidationFailure(f"{path}: number is not above exclusiveMinimum")
        if "exclusiveMaximum" in node and value >= node["exclusiveMaximum"]:
            raise ValidationFailure(f"{path}: number is not below exclusiveMaximum")


def validate_schema(value: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        validate_instance_against_schema(value, schema, schema)
    except ValidationFailure as exc:
        raise ValidationFailure(f"schema validation failed: {exc}") from exc


def validate_freeze(value: dict[str, Any], parameter_path: Path) -> None:
    if value.get("status") != COMPLETED_STATUS:
        raise ValidationFailure(
            f"completed validation requires status={COMPLETED_STATUS!r}"
        )
    freeze = value["freeze"]
    try:
        frozen_at = datetime.fromisoformat(freeze["frozen_at_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationFailure("freeze.frozen_at_utc is not a valid ISO timestamp") from exc
    if frozen_at.tzinfo is None:
        raise ValidationFailure("freeze.frozen_at_utc must include a timezone")
    uri = freeze["registration_uri"].lower()
    if any(token in uri for token in ("example.", "localhost", "placeholder", "tbd")):
        raise ValidationFailure("registration_uri is a placeholder or local URI")
    expected_protocol = sha256_file(PROTOCOL)
    if freeze["protocol_sha256"] != expected_protocol:
        raise ValidationFailure(
            "protocol_sha256 does not match the local frozen PROTOCOL_MATH.md"
        )
    expected_payload = canonical_parameter_payload_sha256(value)
    if freeze["canonical_parameter_payload_sha256"] != expected_payload:
        raise ValidationFailure(
            "canonical parameter payload hash mismatch; compute it after removing "
            "only freeze.canonical_parameter_payload_sha256"
        )
    if parameter_path.resolve() == DEFAULT_SCHEMA.resolve():
        raise ValidationFailure("the schema itself cannot be validated as parameters")


def as_finite_matrix(value: Any, name: str, order: int | None = None) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValidationFailure(f"{name} must be a non-empty matrix")
    rows: list[list[float]] = []
    width: int | None = None
    for index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ValidationFailure(f"{name}[{index}] must be a non-empty row")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValidationFailure(f"{name} has ragged rows")
        numeric_row: list[float] = []
        for column, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValidationFailure(f"{name}[{index}][{column}] is not numeric")
            number = float(item)
            if not math.isfinite(number):
                raise ValidationFailure(f"{name}[{index}][{column}] is not finite")
            numeric_row.append(number)
        rows.append(numeric_row)
    if order is not None and (len(rows) != order or width != order):
        raise ValidationFailure(f"{name} must have shape {order} by {order}")
    return rows


def fraction_matrix(value: Sequence[Sequence[float]]) -> list[list[Fraction]]:
    return [[Fraction(str(item)) for item in row] for row in value]


def identity(order: int) -> list[list[Fraction]]:
    return [
        [Fraction(int(row == column)) for column in range(order)]
        for row in range(order)
    ]


def transpose(matrix: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    return [list(column) for column in zip(*matrix)]


def matmul(
    left: Sequence[Sequence[Fraction]],
    right: Sequence[Sequence[Fraction]],
) -> list[list[Fraction]]:
    right_t = transpose(right)
    return [
        [sum(a * b for a, b in zip(row, column)) for column in right_t]
        for row in left
    ]


def matsub(
    left: Sequence[Sequence[Fraction]],
    right: Sequence[Sequence[Fraction]],
) -> list[list[Fraction]]:
    return [
        [a - b for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def scale(
    scalar: Fraction,
    matrix: Sequence[Sequence[Fraction]],
) -> list[list[Fraction]]:
    return [[scalar * item for item in row] for row in matrix]


def is_symmetric(matrix: Sequence[Sequence[Fraction]]) -> bool:
    return all(
        matrix[row][column] == matrix[column][row]
        for row in range(len(matrix))
        for column in range(len(matrix))
    )


def is_positive_semidefinite_exact(
    matrix: Sequence[Sequence[Fraction]],
) -> bool:
    """Exact Schur-complement test for a symmetric rational PSD matrix."""
    work = [list(row) for row in matrix]
    if not is_symmetric(work):
        return False
    while work:
        order = len(work)
        if any(work[index][index] < 0 for index in range(order)):
            return False
        pivot = next(
            (index for index in range(order) if work[index][index] > 0),
            None,
        )
        if pivot is None:
            return all(item == 0 for row in work for item in row)
        if pivot != 0:
            work[0], work[pivot] = work[pivot], work[0]
            for row in work:
                row[0], row[pivot] = row[pivot], row[0]
        diagonal = work[0][0]
        vector = [work[index][0] for index in range(1, order)]
        work = [
            [
                work[row + 1][column + 1]
                - vector[row] * vector[column] / diagonal
                for column in range(order - 1)
            ]
            for row in range(order - 1)
        ]
    return True


def validate_hardware(value: dict[str, Any]) -> None:
    apparatus = value["apparatus"]
    nodes = apparatus["nodes"]
    if apparatus["physical_motor_count"] != 3 or len(nodes) != 3:
        raise ValidationFailure("exactly three physical motors/nodes are required")
    node_ids = [node["node_id"] for node in nodes]
    if sorted(node_ids) != [0, 1, 2] or len(set(node_ids)) != 3:
        raise ValidationFailure("node_id set must be exactly {0,1,2}")
    leaders = [node for node in nodes if node["role"] == "physical_leader"]
    followers = [node for node in nodes if node["role"] == "physical_follower"]
    if len(leaders) != 1 or leaders[0]["node_id"] != 0 or len(followers) != 2:
        raise ValidationFailure("node 0 must be the only leader; nodes 1 and 2 followers")
    for identity_key in ("hardware_serial", "motor_serial", "encoder_serial"):
        identities = [node[identity_key] for node in nodes]
        if len(set(identities)) != len(identities):
            raise ValidationFailure(f"{identity_key} values must be unique")


def validate_topology(value: dict[str, Any]) -> None:
    topology = value["topology"]
    adjacency = as_finite_matrix(
        topology["follower_adjacency"], "topology.follower_adjacency", 2
    )
    pinning = topology["pinning"]
    if len(pinning) != 2:
        raise ValidationFailure("topology.pinning must have length 2")
    if any(weight < 0 for row in adjacency for weight in row):
        raise ValidationFailure("adjacency weights must be nonnegative")
    if any(adjacency[index][index] != 0 for index in range(2)):
        raise ValidationFailure("adjacency diagonal must be zero")
    expected = []
    for row in range(2):
        degree = sum(adjacency[row])
        expected.append(
            [
                (degree if row == column else 0.0)
                - adjacency[row][column]
                + (float(pinning[row]) if row == column else 0.0)
                for column in range(2)
            ]
        )
    declared = as_finite_matrix(
        topology["pinned_laplacian"], "topology.pinned_laplacian", 2
    )
    if any(
        abs(expected[row][column] - declared[row][column]) > 1e-12
        for row in range(2)
        for column in range(2)
    ):
        raise ValidationFailure("pinned_laplacian is inconsistent with A and pinning")

    reached = {index for index, weight in enumerate(pinning) if weight > 0}
    changed = True
    while changed:
        changed = False
        for receiver in range(2):
            if receiver in reached:
                continue
            if any(
                adjacency[receiver][sender] > 0 and sender in reached
                for sender in range(2)
            ):
                reached.add(receiver)
                changed = True
    if reached != {0, 1}:
        raise ValidationFailure("directed follower graph is not rooted at a pinned node")


def validate_partitions(value: dict[str, Any]) -> None:
    partitions = value["data_partitions"]
    names = (
        "calibration_run_ids",
        "identification_run_ids",
        "confirmation_run_ids",
        "primary_run_ids",
    )
    owners: dict[str, str] = {}
    overlaps: list[str] = []
    for name in names:
        run_ids = partitions[name]
        if len(run_ids) != len(set(run_ids)):
            raise ValidationFailure(f"{name} contains duplicate run IDs")
        for run_id in run_ids:
            previous = owners.setdefault(run_id, name)
            if previous != name:
                overlaps.append(f"{run_id}: {previous} and {name}")
    if overlaps:
        raise ValidationFailure("data partitions overlap: " + "; ".join(overlaps))


def validate_model_identity(value: dict[str, Any]) -> None:
    models = value["physical_model"]["nodes"]
    nominals = value["observer_controller"]["nominal_parameters"]
    if sorted(model["node_id"] for model in models) != [0, 1, 2]:
        raise ValidationFailure("physical_model node IDs must be exactly {0,1,2}")
    if sorted(model["node_id"] for model in nominals) != [0, 1, 2]:
        raise ValidationFailure("nominal model node IDs must be exactly {0,1,2}")
    nominal_by_id = {model["node_id"]: model for model in nominals}
    for model in models:
        node_id = model["node_id"]
        nominal = nominal_by_id[node_id]
        for name in ("a", "b", "c"):
            lower, upper = model[f"{name}_interval"]
            if lower > upper:
                raise ValidationFailure(f"node {node_id} {name} interval is reversed")
            if not lower <= nominal[f"{name}_nominal"] <= upper:
                raise ValidationFailure(
                    f"node {node_id} nominal {name} lies outside its interval"
                )
        if model["b_interval"][0] <= 0:
            raise ValidationFailure(f"node {node_id} b interval includes nonpositive gain")


def validate_certificate(value: dict[str, Any]) -> None:
    certificate = value["certificate"]
    order = certificate["augmented_state_order"]
    tolerance = Fraction(str(certificate["verification_tolerance"]))
    p_float = as_finite_matrix(certificate["P"], "certificate.P", order)
    p = fraction_matrix(p_float)
    if not is_symmetric(p):
        raise ValidationFailure("certificate.P is not exactly symmetric")
    if sum(p[index][index] for index in range(order)) != Fraction(1):
        raise ValidationFailure("certificate.P must have exact trace 1")

    p_lower = Fraction(str(certificate["P_min_eigenvalue_lower_bound"]))
    p_margin_matrix = matsub(p, scale(p_lower, identity(order)))
    if not is_positive_semidefinite_exact(p_margin_matrix):
        raise ValidationFailure(
            "P - P_min_eigenvalue_lower_bound * I is not positive semidefinite"
        )

    mode_ids = certificate["mode_ids"]
    if len(mode_ids) != certificate["mode_count"] or len(set(mode_ids)) != len(mode_ids):
        raise ValidationFailure("mode_count and unique mode_ids disagree")
    matrices = certificate["system_matrices"]
    expected_count = certificate["vertex_count"] * certificate["mode_count"]
    if len(matrices) != expected_count:
        raise ValidationFailure(
            f"expected {expected_count} system matrices, found {len(matrices)}"
        )
    pairs: set[tuple[str, str]] = set()
    seen_modes: set[str] = set()
    lam = Fraction(str(certificate["lambda"]))
    contraction_margin = Fraction(
        str(certificate["contraction_margin_lower_bound"])
    )
    contraction_margin = max(Fraction(0), contraction_margin - tolerance)
    for index, item in enumerate(matrices):
        pair = (item["mode_id"], item["vertex_id"])
        if pair in pairs:
            raise ValidationFailure(f"duplicate system matrix pair {pair}")
        pairs.add(pair)
        if item["mode_id"] not in mode_ids:
            raise ValidationFailure(f"unknown mode_id {item['mode_id']!r}")
        seen_modes.add(item["mode_id"])
        f_float = as_finite_matrix(item["F"], f"system_matrices[{index}].F", order)
        f = fraction_matrix(f_float)
        slack = matsub(
            scale(lam * lam, p),
            matmul(transpose(f), matmul(p, f)),
        )
        rigorous_slack = matsub(
            slack,
            scale(contraction_margin, identity(order)),
        )
        if not is_positive_semidefinite_exact(rigorous_slack):
            raise ValidationFailure(
                f"contraction inequality fails for mode={pair[0]}, vertex={pair[1]}"
            )
    if seen_modes != set(mode_ids):
        raise ValidationFailure("one or more declared modes have no system matrix")

    delta = Fraction(str(certificate["disturbance_P_norm_bound"]))
    radius = Fraction(str(certificate["invariant_radius_P_norm"]))
    if lam * radius + delta > radius + tolerance:
        raise ValidationFailure("lambda * invariant radius + delta exceeds radius")


def validate_completed(
    value: dict[str, Any],
    schema: dict[str, Any],
    parameter_path: Path,
) -> Report:
    report = Report()
    reject_outcome_fields(value)
    report.passed("no physical outcome fields")
    reject_placeholders_and_nulls(value)
    report.passed("no placeholders or nulls")
    validate_schema(value, schema)
    report.passed("JSON Schema")
    validate_hash_fields(value)
    report.passed("hash formats")
    validate_freeze(value, parameter_path)
    report.passed("freeze and canonical hashes")
    validate_hardware(value)
    report.passed("three-node physical apparatus")
    validate_topology(value)
    report.passed("receiver-row topology and target rooting")
    validate_model_identity(value)
    report.passed("model intervals and nominal parameters")
    validate_partitions(value)
    report.passed("pairwise-disjoint data partitions")
    validate_certificate(value)
    report.passed("exact-rational quadratic certificate screening")
    return report


def lint_draft(value: dict[str, Any]) -> Report:
    if value.get("status") != DRAFT_STATUS:
        raise ValidationFailure("draft lint requires status='draft'")
    if value.get("protocol_id") != "PHYS-E2E-OCT-R34":
        raise ValidationFailure("draft template has the wrong protocol_id")
    if value.get("classification") != "design_only_not_executed_not_result":
        raise ValidationFailure("draft classification is missing or unsafe")
    reject_outcome_fields(value)
    report = Report()
    report.passed("draft identity")
    report.passed("no physical outcome fields")
    report.passed("draft is explicitly non-executable and non-evidentiary")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--draft-lint",
        action="store_true",
        help="lint an explicit draft template; never marks it acquisition-ready",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = load_json(args.parameters)
        schema = load_json(args.schema)
        validate_schema_document(schema)
        if value.get("status") == DRAFT_STATUS:
            if not args.draft_lint:
                raise ValidationFailure(
                    "DRAFT_BLOCKED: design-only parameters cannot pass completed "
                    "validation; use --draft-lint only to inspect the template"
                )
            report = lint_draft(value)
            print(
                json.dumps(
                    {
                        "status": "DRAFT_TEMPLATE_OK",
                        "eligible_for_primary_acquisition": False,
                        "physical_result": False,
                        "manuscript_evidence": False,
                        "checks": report.checks,
                    },
                    indent=2,
                )
            )
            return 0
        if args.draft_lint:
            raise ValidationFailure("--draft-lint cannot validate a completed file")
        report = validate_completed(value, schema, args.parameters)
        print(
            json.dumps(
                {
                    "status": "PREREGISTRATION_PARAMETERS_VALID",
                    "eligible_for_primary_acquisition": True,
                    "physical_result": False,
                    "manuscript_evidence": False,
                    "message": (
                        "This validates frozen design parameters only. It does not "
                        "validate hardware execution or any physical outcome."
                    ),
                    "checks": report.checks,
                },
                indent=2,
            )
        )
        return 0
    except ValidationFailure as exc:
        print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
        return 2 if "DRAFT_BLOCKED" in str(exc) else 1
    except Exception as exc:  # pragma: no cover - final fail-closed barrier
        print(f"VALIDATOR_INTERNAL_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
