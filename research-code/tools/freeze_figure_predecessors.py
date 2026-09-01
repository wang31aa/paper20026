#!/usr/bin/env python3
"""Freeze every pre-Round37 figure export before any layout rebuild."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = FIG / "originals_round37"
EXTENSIONS = (".svg", ".pdf", ".png", ".tiff")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite immutable predecessor snapshot: {OUT}")
    files = sorted(p for p in FIG.iterdir()
                   if p.is_file() and p.name.startswith("Fig") and p.suffix.lower() in EXTENSIONS)
    stems = {p.stem for p in files}
    if len(stems) != 12 or len(files) != 48:
        raise SystemExit(f"expected 12 stems/48 exports, found {len(stems)}/{len(files)}")
    OUT.mkdir()
    records = []
    for src in files:
        dst = OUT / src.name
        shutil.copy2(src, dst)
        records.append({"path": src.name, "bytes": src.stat().st_size,
                        "sha256": digest(src)})
        if digest(dst) != records[-1]["sha256"]:
            raise RuntimeError(f"copy verification failed: {src.name}")
    manifest = {"schema_version": 1, "status": "immutable pre-Round37 figure snapshot",
                "file_count": len(records), "records": records}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS: froze {len(records)} figure exports at {OUT}")


if __name__ == "__main__":
    main()
