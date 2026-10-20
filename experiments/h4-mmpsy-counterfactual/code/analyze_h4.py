#!/usr/bin/env python3
"""H4 analysis -- does the segment-count artifact move the prediction?

Every arm shares byte-identical content characters; only the number of
whitespace boundaries differs. So any shift in p(depression) between arms is
attributable to the structural feature H1 found (0.772 balanced accuracy from
answer-segment count alone, against 0.507 from the transcript's content).

Paired within item throughout; the bootstrap resamples items.

The targeted subgroup is the one that matters. `seg_plus` only pushes an item
onto the depressed group's modal count if it started at 7, and `seg_minus` only
pulls it onto the control group's modal count if it started at 8. Items far
from the 7/8 boundary are moved between counts that carry no class information,
so pooling them dilutes the contrast.
"""
from __future__ import annotations

import collections, glob, json, sys
import numpy as np

ROOT = "/data/cl195/NeurIPS_workshop"
RES = f"{ROOT}/experiments/h4-mmpsy-counterfactual/results"
STAGES = ["SFT", "RL", "HARPO"]
ARMS = ["seg_minus", "native", "seg_plus"]
N_BOOT = 4000
RNG = np.random.default_rng(0)


def p_dep(lp):
    v = np.array([lp["depression"], lp["no_depression"]], dtype=np.float64)
    e = np.exp(v - v.max())
    return float(e[0] / e.sum())


def boot(x, n=N_BOOT):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    d = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), *np.percentile(d, [2.5, 97.5])


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum()
                  + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def main():
    data = {}
    for st in STAGES:
        f = f"{RES}/h4_mmpsy_depression_{st}.jsonl"
        if not glob.glob(f):
            continue
        by = collections.defaultdict(dict)
        for l in open(f):
            r = json.loads(l)
            by[r["arm"]][r["idx"]] = r
        data[st] = by
    if not data:
        print("no results yet")
        return

    common = None
    for st, by in data.items():
        for a in ARMS:
            ids = set(by.get(a, {}))
            common = ids if common is None else (common & ids)
    common = sorted(common)
    any_st = data[list(data)[0]]
    gold = np.array([1 if any_st["native"][i]["gold"] == "depression" else 0
                     for i in common])
    nseg = np.array([any_st["native"][i]["n_segments_native"] for i in common])
    print(f"stages: {list(data)}   items common to all: {len(common)}   "
          f"positives: {int(gold.sum())}  majority: {max(gold.mean(), 1-gold.mean()):.3f}")
    print(f"native segment counts: {dict(sorted(collections.Counter(nseg.tolist()).items()))}\n")

    print("=" * 104)
    print("PER-ARM  (content characters byte-identical across arms)")
    print("=" * 104)
    print(f"{'stage':7s} {'arm':11s} {'mean p(dep)':>12s} {'[95% CI]':>20s} "
          f"{'acc':>7s} {'AUC':>7s} {'P(pred=dep)':>12s}")
    print("-" * 104)
    P = {}
    for st in [s for s in STAGES if s in data]:
        for a in ARMS:
            v = np.array([p_dep(data[st][a][i]["logprobs"]) for i in common])
            P[(st, a)] = v
            m, lo, hi = boot(v)
            acc = np.mean([(v[k] > .5) == gold[k] for k in range(len(common))])
            print(f"{st:7s} {a:11s} {m:12.4f} [{lo:.4f}, {hi:.4f}] {acc:7.3f} "
                  f"{auc(gold, v):7.3f} {float((v > .5).mean()):12.3f}")
        print("-" * 104)

    print("\n" + "=" * 104)
    print("CSS = p(dep | +1 segment) - p(dep | -1 segment), paired within item")
    print("=" * 104)
    print(f"{'stage':7s} {'group':28s} {'n':>4s} {'CSS':>9s} {'[95% CI]':>20s} "
          f"{'flips':>7s}")
    print("-" * 104)
    groups = [("all items", np.ones(len(common), bool)),
              ("native 7 seg (-> 8 = dep mode)", nseg == 7),
              ("native 8 seg (-> 7 = ctrl mode)", nseg == 8)]
    for st in [s for s in STAGES if s in data]:
        for name, mask in groups:
            if mask.sum() < 3:
                continue
            d = P[(st, "seg_plus")][mask] - P[(st, "seg_minus")][mask]
            m, lo, hi = boot(d)
            fl = float(np.mean((P[(st, "seg_plus")][mask] > .5)
                               != (P[(st, "seg_minus")][mask] > .5)))
            sig = "***" if (lo > 0 or hi < 0) else "n.s."
            print(f"{st:7s} {name:28s} {int(mask.sum()):4d} {m:+9.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] {fl:7.3f}  {sig}")
        print("-" * 104)

    print("\n" + "=" * 104)
    print("DIRECTION CHECK -- each arm against native")
    print("=" * 104)
    for st in [s for s in STAGES if s in data]:
        row = []
        for a in ["seg_plus", "seg_minus"]:
            d = P[(st, a)] - P[(st, "native")]
            m, lo, hi = boot(d)
            row.append(f"{a}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"  {st:7s} " + "   ".join(row))

    print("\n" + "=" * 104)
    print("ACROSS TRAINING STAGES")
    print("=" * 104)
    have = [s for s in STAGES if s in data]
    if len(have) > 1:
        css = [float((P[(s, "seg_plus")] - P[(s, "seg_minus")]).mean()) for s in have]
        a_ = [auc(gold, P[(s, "native")]) for s in have]
        print("  CSS          " + "  ".join(f"{s}={v:+.4f}" for s, v in zip(have, css))
              + f"   Δ(last−first) = {css[-1]-css[0]:+.4f}")
        print("  native AUC   " + "  ".join(f"{s}={v:.3f}" for s, v in zip(have, a_)))

    out = f"{RES}/h4_summary.json"
    json.dump({f"{s}|{a}": {"mean_p": float(P[(s, a)].mean())}
               for s in [x for x in STAGES if x in data] for a in ARMS},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
