#!/usr/bin/env python3
"""Experiment C1 analysis -- template keying and source-prior leakage.

Two contrasts, both paired within item and bootstrapped over items:

  Δ_template = Acc(native) - Acc(normalized)
      how much accuracy the source's own instruction wording is worth while the
      post is still present. A drop concentrated in particular sources means the
      model was keying on template identity for those sources.

  Δ_prior    = Acc(native_nopost) - Acc(normalized_nopost)
      the same contrast with the post removed. Both sides are content-free, so
      any gap is the wording alone buying label-prior information. This is the
      direct measurement of "recognize the dataset, then guess its prior".

`Acc(native_nopost)` on its own is read against the source's majority class, not
against chance: a content-free prompt that lands on the majority label has
learned the corpus prior, which is the whole shortcut.

Every source is reported separately. Pooling across sources would average a
source that leaks heavily with nine that do not, and the design's hypothesis is
specifically about concentration.
"""
from __future__ import annotations

import json, sys, collections
import numpy as np

ROOT = "/data/cl195/NeurIPS_workshop"
N_BOOT = 2000
RNG = np.random.default_rng(0)
CONDS = ["native", "normalized", "native_nopost", "normalized_nopost"]


def load(path):
    d = collections.defaultdict(dict)          # (source, cond) -> uid -> row
    for l in open(path):
        r = json.loads(l)
        d[(r["source_dataset"], r["condition"])][r["uid"]] = r
    return d


def bal_acc(gold, pred, space):
    per = []
    for c in space:
        m = [i for i, g in enumerate(gold) if g == c]
        if m:
            per.append(np.mean([pred[i] == c for i in m]))
    return float(np.mean(per)) if per else float("nan")


def paired_delta(rows_a, rows_b, uids, n=N_BOOT):
    """Acc(a) - Acc(b), paired on uid, with a percentile CI."""
    a = np.array([rows_a[u]["correct"] for u in uids], dtype=float)
    b = np.array([rows_b[u]["correct"] for u in uids], dtype=float)
    obs = a.mean() - b.mean()
    d = []
    for _ in range(n):
        i = RNG.integers(0, len(uids), len(uids))
        d.append(a[i].mean() - b[i].mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi


def main():
    path = sys.argv[1] if len(sys.argv) > 1 \
        else f"{ROOT}/runs/c1_MentaLLaMA-chat-7B.jsonl"
    d = load(path)
    model = json.loads(open(path).readline())["model"]
    sources = sorted({s for s, _ in d})
    print(f"loaded {path}   model={model}   sources={len(sources)}")

    # ---------------- per-source accuracy table -------------------------
    print(f"\n{'='*112}")
    print("C1 -- ACCURACY BY SOURCE AND CONDITION")
    print("=" * 112)
    print(f"{'source':12s} {'n':>5s} {'|L|':>4s} {'major':>6s} "
          f"{'native':>8s} {'normal':>8s} {'nat_np':>8s} {'norm_np':>8s}   "
          f"{'bal:native':>11s} {'bal:normal':>11s}")
    print("-" * 112)
    summary = {}
    for s in sources:
        uids = sorted(d[(s, "native")])
        gold = [d[(s, "native")][u]["gold"] for u in uids]
        space = d[(s, "native")][uids[0]]["label_space"]
        maj = max(collections.Counter(gold).values()) / len(gold)
        acc, bal = {}, {}
        for c in CONDS:
            pred = [d[(s, c)][u]["pred"] for u in uids]
            acc[c] = float(np.mean([p == g for p, g in zip(pred, gold)]))
            bal[c] = bal_acc(gold, pred, space)
        summary[s] = dict(n=len(uids), space=space, maj=maj, acc=acc, bal=bal,
                          uids=uids, gold=gold)
        print(f"{s:12s} {len(uids):5d} {len(space):4d} {maj:6.3f} "
              f"{acc['native']:8.3f} {acc['normalized']:8.3f} "
              f"{acc['native_nopost']:8.3f} {acc['normalized_nopost']:8.3f}   "
              f"{bal['native']:11.3f} {bal['normalized']:11.3f}")

    # ---------------- Δ_template ----------------------------------------
    print(f"\n{'='*112}")
    print("Δ_template = Acc(native) - Acc(normalized)   [post present, paired "
          "bootstrap over items]")
    print("=" * 112)
    for s in sources:
        u = summary[s]["uids"]
        obs, lo, hi = paired_delta(d[(s, "native")], d[(s, "normalized")], u)
        flag = "TEMPLATE-KEYED" if lo > 0 else ("inverted" if hi < 0 else "n.s.")
        print(f"  {s:12s} {obs:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {flag}")

    # ---------------- Δ_prior -------------------------------------------
    print(f"\n{'='*112}")
    print("Δ_prior = Acc(native_nopost) - Acc(normalized_nopost)   [content-free "
          "both sides]")
    print("=" * 112)
    for s in sources:
        u = summary[s]["uids"]
        obs, lo, hi = paired_delta(d[(s, "native_nopost")],
                                   d[(s, "normalized_nopost")], u)
        flag = "WORDING CARRIES PRIOR" if lo > 0 else \
               ("inverted" if hi < 0 else "n.s.")
        print(f"  {s:12s} {obs:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {flag}")

    # ---------------- what does a content-free prompt answer? ------------
    print(f"\n{'='*112}")
    print("CONTENT-FREE PREDICTION -- what the template alone makes the model say")
    print("=" * 112)
    print(f"{'source':12s} {'majority label':26s} {'native_nopost pred':26s} "
          f"{'acc':>6s} {'= majority?':>12s}")
    print("-" * 112)
    for s in sources:
        u = summary[s]["uids"]
        gold = summary[s]["gold"]
        majlab = collections.Counter(gold).most_common(1)[0][0]
        preds = collections.Counter(d[(s, "native_nopost")][x]["pred"] for x in u)
        top = preds.most_common(1)[0]
        acc = summary[s]["acc"]["native_nopost"]
        hit = "YES" if top[0] == majlab else "no"
        n_distinct = f"({len(preds)} distinct)" if len(preds) > 1 else ""
        print(f"{s:12s} {str(majlab)[:25]:26s} "
              f"{(str(top[0])[:18] + ' ' + n_distinct):26s} {acc:6.3f} "
              f"{hit:>12s}")

    # ---------------- content value -------------------------------------
    print(f"\n{'='*112}")
    print("CONTENT VALUE = Acc(native) - Acc(native_nopost)   [is the post used "
          "at all?]")
    print("=" * 112)
    for s in sources:
        u = summary[s]["uids"]
        obs, lo, hi = paired_delta(d[(s, "native")], d[(s, "native_nopost")], u)
        flag = "post helps" if lo > 0 else ("post HURTS" if hi < 0 else
                                            "POST ADDS NOTHING")
        print(f"  {s:12s} {obs:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {flag}   "
              f"(native {summary[s]['acc']['native']:.3f} vs nopost "
              f"{summary[s]['acc']['native_nopost']:.3f}, majority "
              f"{summary[s]['maj']:.3f})")


if __name__ == "__main__":
    main()
