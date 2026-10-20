#!/usr/bin/env python3
"""Minimal multi-source literature search (Semantic Scholar + arXiv).

The literature-search skill's helper scripts live under a `deep-research` skill
that is not installed here, so this hits the two APIs directly with `requests`.
Results are merged on normalised title and written as JSONL.
"""
from __future__ import annotations

import argparse, json, re, sys, time, urllib.parse
import requests

S2 = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = ("title,year,venue,citationCount,authors,abstract,externalIds,"
             "publicationTypes,openAccessPdf")
ARXIV = "http://export.arxiv.org/api/query"


def norm(t):
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:90]


def s2(query, limit=20, retries=4):
    for a in range(retries):
        try:
            r = requests.get(S2, params={"query": query, "limit": limit,
                                         "fields": S2_FIELDS}, timeout=40)
            if r.status_code == 429:
                time.sleep(6 * (a + 1)); continue
            r.raise_for_status()
            out = []
            for p in r.json().get("data", []) or []:
                out.append({
                    "title": p.get("title"), "year": p.get("year"),
                    "venue": p.get("venue") or "", "cites": p.get("citationCount", 0),
                    "authors": [x["name"] for x in (p.get("authors") or [])][:6],
                    "abstract": p.get("abstract") or "",
                    "arxiv": (p.get("externalIds") or {}).get("ArXiv"),
                    "doi": (p.get("externalIds") or {}).get("DOI"),
                    "src": "s2", "query": query,
                })
            return out
        except Exception as e:
            if a == retries - 1:
                print(f"  [s2 fail] {query[:50]}: {type(e).__name__}", file=sys.stderr)
            time.sleep(4 * (a + 1))
    return []


def arxiv(query, limit=10):
    try:
        q = urllib.parse.urlencode({"search_query": f"all:{query}",
                                    "max_results": limit,
                                    "sortBy": "relevance"})
        r = requests.get(f"{ARXIV}?{q}", timeout=40)
        r.raise_for_status()
        import xml.etree.ElementTree as ET
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in ET.fromstring(r.text).findall("a:entry", ns):
            t = (e.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
            out.append({
                "title": re.sub(r"\s+", " ", t),
                "year": int((e.findtext("a:published", "", ns) or "0")[:4] or 0),
                "venue": "arXiv", "cites": 0,
                "authors": [x.findtext("a:name", "", ns)
                            for x in e.findall("a:author", ns)][:6],
                "abstract": re.sub(r"\s+", " ",
                                   (e.findtext("a:summary", "", ns) or "")).strip(),
                "arxiv": (e.findtext("a:id", "", ns) or "").split("/abs/")[-1],
                "doi": None, "src": "arxiv", "query": query,
            })
        return out
    except Exception as e:
        print(f"  [arxiv fail] {query[:50]}: {type(e).__name__}", file=sys.stderr)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="file, one query per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=20)
    args = ap.parse_args()

    qs = [l.strip() for l in open(args.queries) if l.strip()
          and not l.startswith("#")]
    seen, rows = {}, []
    for q in qs:
        print(f"[query] {q}", flush=True)
        got = s2(q, args.per_source) + arxiv(q, max(6, args.per_source // 2))
        new = 0
        for p in got:
            k = norm(p["title"])
            if not k:
                continue
            if k in seen:
                seen[k]["queries"] = sorted(set(seen[k].get("queries", []) + [q]))
                # prefer the record with a citation count / venue
                if p["cites"] > seen[k]["cites"]:
                    seen[k].update({kk: p[kk] for kk in ("cites", "venue", "year")})
                continue
            p["queries"] = [q]
            seen[k] = p
            rows.append(p)
            new += 1
        print(f"   +{new} new  ({len(rows)} total)", flush=True)
        time.sleep(3.5)

    rows.sort(key=lambda p: (-(p["cites"] or 0), -(p["year"] or 0)))
    with open(args.out, "w") as f:
        for p in rows:
            f.write(json.dumps(p) + "\n")
    print(f"\nwrote {len(rows)} unique papers -> {args.out}")


if __name__ == "__main__":
    main()
