#!/usr/bin/env python3
"""Rebuild complete per-query request/count logs from frozen raw pages."""
import glob, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
queries = {
    "Q1": "quasi-synchronization", "Q2": "quasi synchronization",
    "Q3": "synchronization errors estimation", "Q4": "distributed error estimation",
}
summary = {"OpenAlex": {}, "DBLP": {}, "excluded_pilots": {}}
for source in ("openalex", "dblp"):
    for qid, query in queries.items():
        files = sorted(glob.glob(str(RAW / f"{source}_{qid}_*.json")))
        if not files:
            continue
        if source == "openalex":
            pages = [json.loads(Path(p).read_text()) for p in files]
            returned = sum(len(p.get("results", [])) for p in pages)
            reported = int(pages[0].get("meta", {}).get("count", 0))
        else:
            pages = [json.loads(Path(p).read_text()) for p in files]
            returned = sum(len(p["result"]["hits"].get("hit", [])) for p in pages)
            reported = int(pages[0]["result"]["hits"].get("@total", 0))
        summary["OpenAlex" if source == "openalex" else "DBLP"][qid] = {
            "query": query, "pages": len(files), "reported_total": reported,
            "returned_total": returned, "complete": returned == reported,
            "raw_files": [str(Path(p).relative_to(ROOT)) for p in files],
        }

pilot_oa = sorted((ROOT / "raw_pilot_broad_query").glob("*.json"))
if pilot_oa:
    p = json.loads(pilot_oa[0].read_text())
    summary["excluded_pilots"]["openalex_full_metadata_search_Q1"] = {
        "reported_total": p["meta"]["count"], "pages_preserved": len(pilot_oa),
        "reason": "full-metadata relevance expansion outside frozen title-topic boundary",
    }
pilot_cr = sorted(RAW.glob("crossref_Q1_*.json"))
if pilot_cr:
    pages = [json.loads(p.read_text()) for p in pilot_cr]
    summary["excluded_pilots"]["crossref_query_title_Q1"] = {
        "reported_total": pages[0]["message"]["total-results"],
        "records_preserved": sum(len(p["message"].get("items", [])) for p in pages),
        "reason": "relevance expansion too broad for complete bounded topic corpus",
    }
(ROOT / "retrieval_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
