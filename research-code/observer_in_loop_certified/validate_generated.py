#!/usr/bin/env python3
"""Validate a clean regeneration against frozen, independently checked R3 data."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    actual = args.results.resolve()
    frozen = ROOT / 'results_r3'
    freeze = json.loads((ROOT / 'SOURCE_FREEZE_R3.json').read_text())
    validation = json.loads((frozen / 'validation_schemafix.json').read_text())
    source_ok = all(sha256(ROOT / name) == digest
                    for name, digest in freeze['sha256'].items())
    completion = json.loads((actual / 'RUN_COMPLETION.json').read_text())
    metadata = json.loads((actual / 'metadata.json').read_text())
    provenance_ok = (
        completion.get('protocol_id') == 'CERT-OIL-R3'
        and metadata.get('protocol_id') == 'CERT-OIL-R3'
        and completion.get('source_freeze_sha256') == sha256(ROOT / 'SOURCE_FREEZE_R3.json')
        and metadata.get('source_freeze_sha256') == sha256(ROOT / 'SOURCE_FREEZE_R3.json'))
    command = [sys.executable, str(ROOT / 'compare_results.py'),
               '--expected', str(frozen), '--actual', str(actual)]
    comparison = subprocess.run(command, check=False, text=True,
                                capture_output=True)
    report = {
        'passed': bool(source_ok and validation.get('passed') and provenance_ok
                       and comparison.returncode == 0),
        'frozen_source_hashes_match': source_ok,
        'frozen_independent_validation_passed': bool(validation.get('passed')),
        'generated_completion_and_freeze_provenance_match': provenance_ok,
        'generated_arrays_and_tables_match_frozen': comparison.returncode == 0,
        'comparison': json.loads(comparison.stdout) if comparison.stdout else comparison.stderr,
    }
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
