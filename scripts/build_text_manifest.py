#!/usr/bin/env python3
"""Index the deployed text datasets into one JSONL.

Covers the IMHI test sets shipped in the MentaLLaMA repo plus the standalone
SDCNL corpus.  Labels in IMHI are not a column -- they are the opening clause of
the `gpt-3.5-turbo` gold explanation, so we parse them out and record the raw
clause alongside, so nothing is lost if a parse is wrong.
"""
import json, os, re, glob
from collections import Counter

import pandas as pd

ROOT = "/data/cl195/NeurIPS_workshop"
IMHI = f"{ROOT}/repos/MentaLLaMA/test_data/test_instruction"
OUT = f"{ROOT}/data/processed"

BINARY = {"DR", "Irf", "MultiWD", "dreaddit", "loneliness"}
# opening-clause patterns for the multi-class sets
MULTI = {
    "CAMS": (r"this post shows causes of depression related to (.+?)[,.]",
             r"this post shows no causes of depression"),
    # SAD's catch-all class is phrased "other stress causes" rather than
    # "related to <cause>", so it is matched by the negative pattern.
    "SAD": (r"this post shows the stress cause relat\w* to (.+?)[,.]",
            r"this post shows other stress causes"),
    "swmh": (r"this post shows mental disorder symptoms relat\w* to (.+?)[,.]",
             r"this post shows no mental disorder symptoms"),
    "t-sid": (r"this post shows (.+?)[,.]",
              r"this post shows no mental disorders"),
}


def parse_label(name, gold):
    g = str(gold).strip().lower()
    if name in BINARY:
        head = re.split(r"[,.]", g)[0].strip()
        if head.startswith("yes"):
            return "yes"
        if head.startswith("no"):
            return "no"
        return None
    pos, neg = MULTI[name]
    if neg and re.match(neg, g):
        return "other" if name == "SAD" else "none"
    m = re.match(pos, g)
    return m.group(1).strip() if m else None


def split_query(q):
    """IMHI queries are 'Consider this post: "<post>" Question: <q>'."""
    m = re.match(r'.*?post:\s*"(.*)"\s*(Question:.*)$', str(q), re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return str(q), ""


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, summary = [], []

    for f in sorted(glob.glob(f"{IMHI}/*.csv")):
        name = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        labs = Counter()
        for i, r in d.iterrows():
            post, question = split_query(r["query"])
            lab = parse_label(name, r["gpt-3.5-turbo"])
            labs[lab] += 1
            rows.append({
                "uid": f"imhi/{name}/test/{i}",
                "source_dataset": name,
                "corpus": "IMHI",
                "split": "test",
                "task_type": "binary" if name in BINARY else "multiclass",
                "text": post,
                "question": question,
                "label": lab,
                "gold_explanation": str(r["gpt-3.5-turbo"]),
                "n_words": len(post.split()),
            })
        unparsed = labs.pop(None, 0)
        summary.append((name, len(d), "binary" if name in BINARY else "multi",
                        len(labs), unparsed, dict(labs.most_common(5))))

    # SDCNL: has a real label column
    s = pd.read_csv(f"{ROOT}/repos/SDCNL/data/combined-set.csv")
    for i, r in s.iterrows():
        text = f"{r['title']} {r['selftext']}".strip()
        rows.append({
            "uid": f"sdcnl/all/{i}",
            "source_dataset": "SDCNL",
            "corpus": "SDCNL",
            "split": "all",
            "task_type": "binary",
            "text": text,
            "question": "Does the post indicate suicidal ideation?",
            "label": "yes" if int(r["is_suicide"]) == 1 else "no",
            "gold_explanation": None,
            "n_words": len(text.split()),
        })
    summary.append(("SDCNL", len(s), "binary", 2, 0,
                    s["is_suicide"].value_counts().to_dict()))

    path = f"{OUT}/text_manifest.jsonl"
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"wrote {path}  ({len(rows)} rows)\n")
    print("%-12s %7s %-10s %6s %9s  %s" %
          ("dataset", "rows", "task", "labels", "unparsed", "distribution"))
    for name, n, t, nl, unp, dist in summary:
        flag = "  <-- CHECK" if unp else ""
        print("%-12s %7d %-10s %6d %9d  %s%s" % (name, n, t, nl, unp, dist, flag))


if __name__ == "__main__":
    main()
