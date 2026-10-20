#!/usr/bin/env python3
"""H5 analysis -- CSS for the register shortcut, item transcript held fixed."""
from __future__ import annotations
import collections, glob, json
import numpy as np

RES = "/data/cl195/NeurIPS_workshop/experiments/h5-ptsd-register-cf/results"
STAGES, ARMS = ["SFT", "RL", "HARPO"], ["ctx_vlog", "native", "ctx_testimony"]
RNG = np.random.default_rng(0)


def p_pos(lp):
    v = np.array([lp["ptsd"], lp["no_ptsd"]], float)
    e = np.exp(v - v.max()); return float(e[0] / e.sum())


def boot(x, n=4000):
    x = np.asarray(x, float)
    if len(x) < 3: return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan)
    d = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), *np.percentile(d, [2.5, 97.5])


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    a, b = p[y == 1], p[y == 0]
    if not len(a) or not len(b): return np.nan
    return float(((a[:, None] > b[None, :]).sum()
                  + .5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def main():
    data = {}
    for st in STAGES:
        f = f"{RES}/h5_ptsd_{st}.jsonl"
        if not glob.glob(f): continue
        by = collections.defaultdict(dict)
        for l in open(f):
            r = json.loads(l); by[r["arm"]][r["idx"]] = r
        data[st] = by
    if not data: print("no results"); return

    common = None
    for st, by in data.items():
        for a in ARMS:
            s = set(by.get(a, {}))
            common = s if common is None else (common & s)
    common = sorted(common)
    ref = data[list(data)[0]]
    gold = np.array([1 if ref["native"][i]["gold"] == "ptsd" else 0 for i in common])
    print(f"stages: {list(data)}   items common to all: {len(common)}   "
          f"positives: {int(gold.sum())}")
    print(f"truncated: {sum(ref['native'][i]['truncated'] for i in common)}/{len(common)}\n")

    P = {}
    print("=" * 96)
    print("PER-ARM  (each item's own transcript byte-identical across arms)")
    print("=" * 96)
    print(f"{'stage':7s} {'arm':15s} {'mean p(ptsd)':>13s} {'[95% CI]':>20s} "
          f"{'acc':>7s} {'AUC':>7s}")
    print("-" * 96)
    for st in [s for s in STAGES if s in data]:
        for a in ARMS:
            v = np.array([p_pos(data[st][a][i]["logprobs"]) for i in common])
            P[(st, a)] = v
            m, lo, hi = boot(v)
            acc = np.mean([(v[k] > .5) == gold[k] for k in range(len(common))])
            print(f"{st:7s} {a:15s} {m:13.4f} [{lo:.4f}, {hi:.4f}] {acc:7.3f} "
                  f"{auc(gold, v):7.3f}")
        print("-" * 96)

    print("\n" + "=" * 96)
    print("CSS = p(ptsd | testimony register) - p(ptsd | vlog register), paired")
    print("=" * 96)
    for st in [s for s in STAGES if s in data]:
        d = P[(st, "ctx_testimony")] - P[(st, "ctx_vlog")]
        m, lo, hi = boot(d)
        fl = float(np.mean((P[(st, "ctx_testimony")] > .5)
                           != (P[(st, "ctx_vlog")] > .5)))
        sig = "***" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {st:7s} CSS = {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"flips {fl:.3f}   {sig}")
        for nm, msk in [("  on true positives", gold == 1),
                        ("  on true negatives", gold == 0)]:
            mm, l2, h2 = boot(d[msk])
            print(f"    {nm:20s} n={int(msk.sum()):3d}  {mm:+.4f} [{l2:+.4f}, {h2:+.4f}]")

    print("\n" + "=" * 96)
    print("DIRECTION vs native")
    print("=" * 96)
    for st in [s for s in STAGES if s in data]:
        parts = []
        for a in ["ctx_testimony", "ctx_vlog"]:
            m, lo, hi = boot(P[(st, a)] - P[(st, "native")])
            parts.append(f"{a}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"  {st:7s} " + "   ".join(parts))

    have = [s for s in STAGES if s in data]
    if len(have) > 1:
        print("\n" + "=" * 96); print("ACROSS STAGES"); print("=" * 96)
        css = [float((P[(s, "ctx_testimony")] - P[(s, "ctx_vlog")]).mean()) for s in have]
        print("  CSS         " + "  ".join(f"{s}={v:+.4f}" for s, v in zip(have, css))
              + f"   Δ = {css[-1]-css[0]:+.4f}")
        print("  native AUC  " + "  ".join(
            f"{s}={auc(gold, P[(s,'native')]):.3f}" for s in have))


if __name__ == "__main__":
    main()
