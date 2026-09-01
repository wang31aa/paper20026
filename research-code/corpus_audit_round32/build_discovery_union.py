#!/usr/bin/env python3
"""Normalize and deduplicate every frozen OpenAlex and DBLP discovery hit."""
from __future__ import annotations

import csv
import glob
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def norm_doi(value):
    if not value:
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", value)
    return value.rstrip(" .;,)")


def norm_title(value):
    value = unicodedata.normalize("NFKD", value or "").lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def oa_abstract(inv):
    if not inv:
        return ""
    slots = []
    for word, positions in inv.items():
        for pos in positions:
            slots.append((pos, word))
    return " ".join(word for _, word in sorted(slots))


rows = []
for path in sorted(glob.glob(str(RAW / "openalex_*.json"))):
    payload = json.loads(Path(path).read_text())
    qid = re.search(r"openalex_(Q\d)_", Path(path).name).group(1)
    for item in payload.get("results", []):
        rows.append({
            "source": "OpenAlex", "query_id": qid,
            "source_id": item.get("id", ""), "doi": norm_doi(item.get("doi")),
            "title": item.get("title") or item.get("display_name") or "",
            "year": item.get("publication_year") or "", "type": item.get("type") or "",
            "abstract": oa_abstract(item.get("abstract_inverted_index")),
            "url": (item.get("primary_location") or {}).get("landing_page_url") or "",
        })
for path in sorted(glob.glob(str(RAW / "dblp_*.json"))):
    payload = json.loads(Path(path).read_text())
    qid = re.search(r"dblp_(Q\d)_", Path(path).name).group(1)
    for hit in payload.get("result", {}).get("hits", {}).get("hit", []):
        info = hit.get("info", {})
        title = re.sub(r"<[^>]+>", "", info.get("title", ""))
        rows.append({
            "source": "DBLP", "query_id": qid, "source_id": info.get("key", ""),
            "doi": norm_doi(info.get("doi")), "title": title,
            "year": info.get("year", ""), "type": info.get("type", ""),
            "abstract": "", "url": info.get("url", ""),
        })

merged = {}
for row in rows:
    key = "doi:" + row["doi"] if row["doi"] else "title:" + norm_title(row["title"])
    out = merged.setdefault(key, {
        "dedup_key": key, "doi": row["doi"], "title": row["title"],
        "year": row["year"], "types": set(), "sources": set(), "queries": set(),
        "source_ids": set(), "abstract": row["abstract"], "urls": set(),
    })
    out["sources"].add(row["source"]); out["queries"].add(row["query_id"])
    out["types"].add(str(row["type"])); out["source_ids"].add(row["source_id"])
    if row["url"]: out["urls"].add(row["url"])
    if len(row["abstract"]) > len(out["abstract"]): out["abstract"] = row["abstract"]

fields = ["dedup_key", "doi", "title", "year", "types", "sources", "queries",
          "source_ids", "abstract", "urls"]
with (ROOT / "discovery_union.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for out in sorted(merged.values(), key=lambda x: (str(x["year"]), x["title"])):
        for field in ("types", "sources", "queries", "source_ids", "urls"):
            out[field] = ";".join(sorted(out[field]))
        w.writerow(out)

by_source = defaultdict(set); by_query = defaultdict(set)
for key, out in merged.items():
    for s in out["sources"] if isinstance(out["sources"], set) else out["sources"].split(";"):
        by_source[s].add(key)
    for q in out["queries"] if isinstance(out["queries"], set) else out["queries"].split(";"):
        by_query[q].add(key)
summary = {
    "raw_query_memberships": len(rows), "deduplicated_union": len(merged),
    "unique_by_source": {k: len(v) for k, v in sorted(by_source.items())},
    "unique_by_query_family": {k: len(v) for k, v in sorted(by_query.items())},
    "both_sources": sum(1 for x in merged.values() if "OpenAlex" in x["sources"] and "DBLP" in x["sources"]),
    "with_doi": sum(1 for x in merged.values() if x["doi"]),
    "with_openalex_abstract": sum(1 for x in merged.values() if x["abstract"]),
    "claim_level_status": "HOLD: API metadata/abstracts cannot establish the five method interfaces",
}
(ROOT / "discovery_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
