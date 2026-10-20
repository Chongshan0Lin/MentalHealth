#!/usr/bin/env python3
"""Derive an authoritative label space per HBA (dataset, task).

The prompt text is NOT a reliable source: `meld_emotion` lists
"anger, disgust, fear, happy, surprise, neutral, sad" but its gold
`class_label` uses `joy` (402) and `sadness` (208) -- 610 of 2610 rows, 23%,
whose gold answer is absent from the option list.  Scoring naively against the
prompt options marks all of them wrong and skews every ablation comparison.

So: take `class_label` as ground truth, and keep a surface-form alias map for
the wording the prompt actually uses.
"""
import glob, json, collections

import pyarrow.parquet as pq

SHARDS = "/data/cl195/NeurIPS_workshop/data/raw/HBA-test/test-*.parquet"
OUT = "/data/cl195/NeurIPS_workshop/data/processed/hba_label_spaces.json"

# gold class_label -> surface form used in the prompt's option list
ALIASES = {
    "meld_emotion": {"joy": "happy", "sadness": "sad"},
}


def main():
    per = collections.defaultdict(collections.Counter)
    for f in sorted(glob.glob(SHARDS)):
        t = pq.ParquetFile(f).read(
            columns=["dataset", "task", "class_label"]).to_pydict()
        for ds, tk, cl in zip(t["dataset"], t["task"], t["class_label"]):
            per[(ds, tk)][cl] += 1

    # A closed label space needs few, reused values. mimeqa / siq2 store the
    # free-text gold answer in class_label, so they are open-ended QA and fall
    # outside the closed-label scoring path entirely.
    MAX_LABELS = 20

    out = {}
    print("%-20s %-16s %6s %7s  %s" % ("dataset", "task", "n", "|L|", "label space"))
    for (ds, tk), c in sorted(per.items()):
        labels = sorted(c)
        n = sum(c.values())
        closed = len(labels) <= MAX_LABELS
        alias = ALIASES.get(ds, {})
        out[ds] = {
            "task": tk,
            "closed_label_space": closed,
            "labels": labels if closed else None,               # gold values
            "surface": [alias.get(l, l) for l in labels] if closed else None,
            "alias": alias,
            "counts": dict(c) if closed else None,
            "n": n,
            "n_distinct": len(labels),
            "majority": c.most_common(1)[0][0] if closed else None,
            "majority_rate": round(c.most_common(1)[0][1] / n, 4) if closed else None,
            "chance": round(1 / len(labels), 4) if closed else None,
        }
        if closed:
            flag = "  <- alias applied" if alias else ""
            print("%-20s %-16s %6d %7d  %s%s" % (ds, tk, n, len(labels), labels, flag))
        else:
            print("%-20s %-16s %6d %7d  OPEN-ENDED (free-text gold) -- excluded "
                  "from label scoring" % (ds, tk, n, len(labels)))

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    closed = {k: v for k, v in out.items() if v["closed_label_space"]}
    print("\nclosed-label datasets: %d of %d\n" % (len(closed), len(out)))
    print("%-20s %9s %9s   majority class" % ("dataset", "chance", "majority"))
    for ds, v in sorted(closed.items(), key=lambda x: -x[1]["majority_rate"]):
        print("%-20s %9.3f %9.3f   %s"
              % (ds, v["chance"], v["majority_rate"], v["majority"]))


if __name__ == "__main__":
    main()
