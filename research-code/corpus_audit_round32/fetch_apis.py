#!/usr/bin/env python3
"""Fetch every page for the frozen Round-32 OpenAlex and Crossref queries."""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
RAW.mkdir(exist_ok=True)
QUERIES = {
    "Q1": "quasi-synchronization",
    "Q2": "quasi synchronization",
    "Q3": "synchronization errors estimation",
    "Q4": "distributed error estimation",
}


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "nature-corpus-audit/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def openalex() -> list[dict]:
    records = []
    log = []
    for qid, query in QUERIES.items():
        if os.environ.get("ONLY_QID") not in (None, "", qid):
            continue
        cursor = "*"
        page = 0
        while cursor:
            params = {
                "filter": f"from_publication_date:2000-01-01,to_publication_date:2026-07-30,title.search:{query}",
                "per-page": 200,
                "cursor": cursor,
                "select": "id,doi,title,display_name,publication_year,publication_date,type,authorships,primary_location,abstract_inverted_index,open_access,ids",
            }
            url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
            payload = get_json(url)
            page += 1
            stamp = dt.datetime.now(dt.timezone.utc).isoformat()
            (RAW / f"openalex_{qid}_{page:03d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log.append({"source": "openalex", "query_id": qid, "page": page, "url": url, "retrieved_utc": stamp, "count": len(payload.get("results", []))})
            for item in payload.get("results", []):
                item["_query_id"] = qid
                records.append(item)
            cursor = payload.get("meta", {}).get("next_cursor")
            if not payload.get("results"):
                break
            time.sleep(0.25)
        (ROOT / f"openalex_records_{qid}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (ROOT / "openalex_requests.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return records


def crossref() -> list[dict]:
    records = []
    log = []
    for qid, query in QUERIES.items():
        if os.environ.get("ONLY_QID") not in (None, "", qid):
            continue
        cursor = "*"
        page = 0
        while cursor:
            params = {
                "query.title": query,
                "filter": "from-pub-date:2000-01-01,until-pub-date:2026-07-30",
                "rows": 1000,
                "cursor": cursor,
                "select": "DOI,title,abstract,type,published-print,published-online,author,URL,container-title,link",
            }
            url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
            payload = get_json(url)
            page += 1
            stamp = dt.datetime.now(dt.timezone.utc).isoformat()
            (RAW / f"crossref_{qid}_{page:03d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            items = payload.get("message", {}).get("items", [])
            log.append({"source": "crossref", "query_id": qid, "page": page, "url": url, "retrieved_utc": stamp, "count": len(items)})
            for item in items:
                item["_query_id"] = qid
                records.append(item)
            nxt = payload.get("message", {}).get("next-cursor")
            cursor = nxt if items and nxt and len(items) == 1000 else None
            time.sleep(0.25)
        (ROOT / f"crossref_records_{qid}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (ROOT / "crossref_requests.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return records


def main() -> None:
    source = os.environ.get("ONLY_SOURCE", "both")
    oa = openalex() if source in ("both", "openalex") else []
    cr = crossref() if source in ("both", "crossref") else []
    (ROOT / "openalex_records.json").write_text(json.dumps(oa, indent=2), encoding="utf-8")
    (ROOT / "crossref_records.json").write_text(json.dumps(cr, indent=2), encoding="utf-8")
    with (ROOT / "retrieval_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "record_query_memberships"])
        w.writerow(["OpenAlex", len(oa)])
        w.writerow(["Crossref", len(cr)])


if __name__ == "__main__":
    main()
