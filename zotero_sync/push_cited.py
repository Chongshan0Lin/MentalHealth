#!/usr/bin/env python3
"""Push the paper's cited references into Zotero, with pinned BBT citation keys.

Reads paper/refs.bib, keeps only entries actually \cite'd in paper/main.tex, maps
each to a Zotero item, and writes `Citation Key: <key>` into the item's Extra
field. Extra syncs, so a pinned key survives the round trip -- Better BibTeX's
unpinned keys live only in the local BBT database and would come back different.

Idempotent: an item whose Extra already carries the same citation key is skipped,
so re-running after adding a reference only uploads the new ones.

Read-only by default. Pass --write to actually POST.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
import requests

ROOT = "/data/cl195/NeurIPS_workshop"
API = "https://api.zotero.org"
UID = 14436928

TYPE_MAP = {"inproceedings": "conferencePaper", "article": "journalArticle",
            "misc": "preprint", "software": "computerProgram",
            "book": "book", "incollection": "bookSection",
            "phdthesis": "thesis", "techreport": "report"}


def parse_bib(path):
    s = open(path).read()
    out = {}
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,]+),(.*?)\n\}\s*\n', s + "\n", re.S):
        typ, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        f = {}
        for fm in re.finditer(r'(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\s*$)',
                              body, re.S):
            f[fm.group(1).lower()] = re.sub(r'\s+', ' ', fm.group(2)).strip()
        out[key] = (typ, f)
    return out


ACCENTS = {"'": {"a":"\u00e1","e":"\u00e9","i":"\u00ed","o":"\u00f3","u":"\u00fa",
                 "n":"\u0144","c":"\u0107","s":"\u015b","y":"\u00fd","z":"\u017a",
                 "A":"\u00c1","E":"\u00c9","I":"\u00cd","O":"\u00d3","U":"\u00da"},
           '`': {"a":"\u00e0","e":"\u00e8","i":"\u00ec","o":"\u00f2","u":"\u00f9"},
           '"': {"a":"\u00e4","e":"\u00eb","i":"\u00ef","o":"\u00f6","u":"\u00fc",
                 "A":"\u00c4","O":"\u00d6","U":"\u00dc"},
           '^': {"a":"\u00e2","e":"\u00ea","i":"\u00ee","o":"\u00f4","u":"\u00fb"},
           '~': {"a":"\u00e3","n":"\u00f1","o":"\u00f5","N":"\u00d1"},
           'c': {"c":"\u00e7","C":"\u00c7"},
           'v': {"c":"\u010d","s":"\u0161","z":"\u017e","r":"\u0159","e":"\u011b"}}


def delatex(s):
    """BibTeX accent commands -> Unicode. Zotero stores plain UTF-8, so leaving
    {\'i} in a name would put the literal backslash into the library."""
    if not s:
        return s
    for acc, tbl in ACCENTS.items():
        a = re.escape(acc)
        for base, ch in tbl.items():
            for pat in (r"\\" + a + r"\{" + base + r"\}",
                        r"\{\\" + a + base + r"\}",
                        r"\{\\" + a + r"\{" + base + r"\}\}",
                        r"\\" + a + base + r"(?![a-zA-Z])"):
                s = re.sub(pat, ch, s)
    s = s.replace("\\ss", "\u00df").replace("\\&", "&").replace("--", "\u2013")
    return re.sub(r"[{}]", "", s)


def authors(raw):
    """BibTeX author string -> Zotero creators. Handles 'Last, First' and
    braced corporate names like {VERIFY AUTHORS} / {The Team}."""
    if not raw:
        return []
    cs = []
    for a in re.split(r'\s+and\s+', raw):
        a = a.strip()
        if not a:
            continue
        if a.startswith("{") and a.endswith("}"):
            cs.append({"creatorType": "author", "name": delatex(a[1:-1])})
        elif "," in a:
            last, first = a.split(",", 1)
            cs.append({"creatorType": "author",
                       "firstName": delatex(first.strip()),
                       "lastName": delatex(last.strip())})
        else:
            parts = a.split()
            cs.append({"creatorType": "author",
                       "firstName": delatex(" ".join(parts[:-1])),
                       "lastName": delatex(parts[-1])}
                      if len(parts) > 1 else
                      {"creatorType": "author", "name": delatex(a)})
    return cs


def to_zotero(key, typ, f):
    zt = TYPE_MAP.get(typ, "document")
    # `journal = {arXiv preprint arXiv:NNNN}` is not a journal. Zotero has a
    # first-class preprint type; using it keeps the repository/archiveID fields
    # meaningful and stops the fake journal name reaching the bibliography.
    if zt == "journalArticle" and re.match(r"\s*arxiv preprint",
                                           f.get("journal", ""), re.I):
        zt = "preprint"
        m = re.search(r"(\d{4}\.\d{4,5})", f.get("journal", ""))
        if m and not f.get("eprint"):
            f = dict(f, eprint=m.group(1))
    # strip the {braces} BibTeX uses to protect capitalisation
    title = delatex(f.get("title", ""))
    it = {"itemType": zt, "title": title, "creators": authors(f.get("author", "")),
          "date": f.get("year", ""), "extra": f"Citation Key: {key}"}
    if f.get("doi"):
        it["DOI"] = f["doi"]
    if f.get("url"):
        it["url"] = f["url"]
    if f.get("pages"):
        it["pages"] = f["pages"]
    if zt == "conferencePaper":
        it["proceedingsTitle"] = delatex(f.get("booktitle", ""))
    elif zt == "journalArticle":
        it["publicationTitle"] = delatex(f.get("journal", ""))
        if f.get("volume"):
            it["volume"] = f["volume"]
    elif zt == "preprint":
        it["repository"] = f.get("archiveprefix", "arXiv")
        if f.get("eprint"):
            it["archiveID"] = f"arXiv:{f['eprint']}"
    elif zt == "computerProgram":
        it.pop("DOI", None)
    if f.get("eprint") and zt != "preprint":
        it["extra"] += f"\narXiv:{f['eprint']}"
    if f.get("note"):
        it["extra"] += f"\n{delatex(f['note'])}"
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="actually POST to Zotero")
    ap.add_argument("--all", action="store_true",
                    help="push every refs.bib entry, not only the cited ones")
    args = ap.parse_args()

    kf = os.environ.get("ZOTERO_API_KEY")
    if not kf:
        sys.exit("ZOTERO_API_KEY not set -- source /data/cl195/.secrets/zotero.env")
    H = {"Zotero-API-Key": kf, "Content-Type": "application/json"}

    bib = parse_bib(f"{ROOT}/paper/refs.bib")
    tex = open(f"{ROOT}/paper/main.tex").read()
    cited = set()
    for m in re.finditer(r'\\cite[a-zA-Z]*\{([^}]*)\}', tex):
        cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    keys = sorted(bib) if args.all else sorted(k for k in bib if k in cited)
    print(f"refs.bib: {len(bib)} entries; cited in main.tex: {len(cited)}; "
          f"pushing: {len(keys)}")

    # what is already up there, by pinned citation key
    have = {}
    start = 0
    while True:
        r = requests.get(f"{API}/users/{UID}/items",
                         headers={"Zotero-API-Key": kf},
                         params={"limit": 100, "start": start}, timeout=60)
        r.raise_for_status()
        page = r.json()
        for it in page:
            for line in (it["data"].get("extra") or "").split("\n"):
                if line.lower().startswith("citation key"):
                    have[line.split(":", 1)[1].strip()] = it["key"]
        if len(page) < 100:
            break
        start += 100
    print(f"already pinned in Zotero: {len(have)}")

    todo = [k for k in keys if k not in have]
    skip = [k for k in keys if k in have]
    if skip:
        print(f"skipping {len(skip)} already present: {', '.join(skip)}")
    if not todo:
        print("nothing to do")
        return

    items = [to_zotero(k, *bib[k]) for k in todo]
    print(f"\n{'key':28s} {'zotero type':18s} title")
    for k, it in zip(todo, items):
        print(f"{k:28s} {it['itemType']:18s} {it['title'][:52]}")

    if not args.write:
        json.dump(items, open(f"{ROOT}/zotero_sync/payload_preview.json", "w"),
                  indent=1)
        print(f"\nDRY RUN. payload -> zotero_sync/payload_preview.json")
        print("re-run with --write to upload")
        return

    for i in range(0, len(items), 50):          # API caps a write at 50 objects
        batch = items[i:i + 50]
        r = requests.post(f"{API}/users/{UID}/items", headers=H,
                          data=json.dumps(batch), timeout=90)
        print(f"\nPOST {len(batch)} items -> HTTP {r.status_code}")
        if r.status_code not in (200, 201):
            print(r.text[:800]); sys.exit(1)
        res = r.json()
        for idx, ok in (res.get("success") or {}).items():
            print(f"  ok      {todo[i + int(idx)]:28s} -> {ok}")
        for idx, fail in (res.get("failed") or {}).items():
            print(f"  FAILED  {todo[i + int(idx)]:28s} -> {fail}")
        for idx, un in (res.get("unchanged") or {}).items():
            print(f"  same    {todo[i + int(idx)]:28s}")
        time.sleep(1)


if __name__ == "__main__":
    main()
