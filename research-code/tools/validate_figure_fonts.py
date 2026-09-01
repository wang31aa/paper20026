#!/usr/bin/env python3
"""Fail if any editable figure's base SVG text is below 7 pt-equivalent.

MathText creates smaller tspan children for subscripts and superscripts. Those
children are intentionally excluded; the gate applies to the parent text
object that sets the final-size label or annotation.
"""
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
failures = []
checked = 0
for path in sorted((ROOT / "figures").glob("Fig*.svg")):
    tree = ET.parse(path)
    sizes = []
    for element in tree.iter():
        if element.tag.rsplit("}", 1)[-1] != "text":
            continue
        style = element.get("style", "")
        match = re.search(r"font-size:\s*([0-9.]+)px", style)
        if not match:
            match = re.search(r"(?:^|;)\s*font:\s*([0-9.]+)px(?:\s|$)", style)
        if match:
            sizes.append(float(match.group(1)))
    if not sizes:
        # TeX/TikZ is also an editable vector source. dvisvgm may outline its
        # fonts, so validate the source declarations instead of pretending the
        # outlined SVG contains editable text.
        tex = path.with_suffix(".tex")
        if tex.exists():
            source = tex.read_text()
            tex_sizes = [float(x) for x in re.findall(r"\\fontsize\{([0-9.]+)\}", source)]
            if not tex_sizes:
                failures.append(f"{path.name}: outlined SVG has no declared TeX font sizes")
                continue
            checked += 1
            if min(tex_sizes) < 7.0:
                failures.append(f"{tex.name}: minimum {min(tex_sizes):.6g} < 7")
            continue
        failures.append(f"{path.name}: no editable SVG text sizes or TeX vector source found")
        continue
    checked += 1
    if min(sizes) < 7.0:
        failures.append(f"{path.name}: minimum {min(sizes):.6g} < 7")

if failures:
    print("FAIL: final-size figure typography")
    print("\n".join(failures))
    sys.exit(1)
print(f"PASS: {checked} editable figures have minimum base SVG text size >= 7")
