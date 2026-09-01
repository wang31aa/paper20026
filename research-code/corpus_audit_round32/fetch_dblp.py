#!/usr/bin/env python3
"""Fetch complete DBLP API results for the frozen title-query families."""
import datetime as dt
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
QUERIES = {
    "Q1": "quasi-synchronization",
    "Q2": "quasi synchronization",
    "Q3": "synchronization errors estimation",
    "Q4": "distributed error estimation",
}

log = []
all_hits = []
for qid, query in QUERIES.items():
    offset = 0
    page = 0
    query_hits = []
    total = None
    while total is None or offset < total:
        params = {"q": query, "format": "json", "h": 100, "f": offset}
        url = "https://dblp.org/search/publ/api?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "nature-corpus-audit/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = json.load(response)
        result = payload.get("result", {})
        total = int(result.get("hits", {}).get("@total", 0))
        hits = result.get("hits", {}).get("hit", [])
        page += 1
        (RAW / f"dblp_{qid}_{page:03d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        query_hits.extend(hits)
        offset += len(hits)
        if not hits:
            break
    if len(query_hits) != total:
        raise RuntimeError(f"DBLP query {qid} not completely returned: {len(query_hits)}/{total}")
    for hit in query_hits:
        hit["_query_id"] = qid
        all_hits.append(hit)
    log.append({"source": "dblp", "query_id": qid, "url": url,
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "total": total, "sent": len(query_hits)})

(ROOT / "dblp_requests.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
(ROOT / "dblp_records.json").write_text(json.dumps(all_hits, indent=2), encoding="utf-8")
