#!/usr/bin/env python3
"""Create the journal-facing SI source without disabled internal history blocks."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript" / "NC_Rebuilt_SI.tex"
OUTPUT = ROOT / "manuscript" / "NC_Rebuilt_SI_Submission.tex"


def strip_disabled_blocks(text: str) -> str:
    kept: list[str] = []
    depth = 0
    for line in text.splitlines(keepends=True):
        token = line.strip()
        if token == r"\iffalse":
            depth += 1
            continue
        if token == r"\fi" and depth:
            depth -= 1
            continue
        if depth == 0:
            kept.append(line)
    if depth:
        raise ValueError("Unclosed \\iffalse block in SI source")
    return "".join(kept)


def main() -> None:
    source_text = SOURCE.read_text(encoding="utf-8")
    clean_text = strip_disabled_blocks(source_text)
    if r"\iffalse" in clean_text:
        raise ValueError("Disabled block marker remains in clean SI")
    OUTPUT.write_text(clean_text, encoding="utf-8")
    print(f"WROTE: {OUTPUT}")


if __name__ == "__main__":
    main()
