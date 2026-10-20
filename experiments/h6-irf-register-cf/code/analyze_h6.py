#!/usr/bin/env python3
"""H6 analysis -- register CSS on Irf, with a signal-free prefix as the zero point."""
from __future__ import annotations
import collections, glob, json, os
import numpy as np

RES = "/data/cl195/NeurIPS_workshop/experiments/h6-irf-register-cf/results"
ARMS = ["ctx_no", "native", "ctx_neutral", "ctx_yes"]
RNG = np.random.default_rng(0)


def p_yes(lp):
    v = np.array([lp["yes"], lp["no"]], float)
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
    for f in sorted(glob.glob(f"{RES}/h6_irf_*.jsonl")):
        by = collections.defaultdict(dict)
        for l in open(f):
            r = json.loads(l); by[r["arm"]][r["uid"]] = r
        model = json.loads(open(f).readline())["model"]
        uids = sorted(set.intersection(*[set(by[a]) for a in ARMS]))
        gold = np.array([1 if by["native"][u]["gold"] == "yes" else 0 for u in uids])
        q = np.array([("belong" if "belonging" in by["native"][u]["question"]
                       else "burden") for u in uids])
        print("=" * 100)
        print(f"{model}   n={len(uids)}   yes={int(gold.sum())} "
              f"({gold.mean():.3f})   truncated="
              f"{sum(by['native'][u]['truncated'] for u in uids)}")
        print("=" * 100)
        P = {a: np.array([p_yes(by[a][u]["logprobs"]) for u in uids]) for a in ARMS}

        print(f"{'arm':13s} {'mean p(yes)':>12s} {'[95% CI]':>20s} {'acc':>7s} "
              f"{'AUC':>7s} {'P(pred=yes)':>12s}")
        print("-" * 100)
        for a in ARMS:
            m, lo, hi = boot(P[a])
            acc = float(np.mean((P[a] > .5) == gold))
            print(f"{a:13s} {m:12.4f} [{lo:.4f}, {hi:.4f}] {acc:7.3f} "
                  f"{auc(gold, P[a]):7.3f} {float((P[a] > .5).mean()):12.3f}")

        print(f"\n{'contrast':34s} {'Δ p(yes)':>10s} {'[95% CI]':>20s} {'flips':>7s}")
        print("-" * 100)
        for a, b, lab in [("ctx_yes", "ctx_no", "CSS: yes-register − no-register"),
                          ("ctx_yes", "ctx_neutral", "yes-register − signal-free"),
                          ("ctx_no", "ctx_neutral", "no-register − signal-free"),
                          ("ctx_neutral", "native", "signal-free prefix − native")]:
            d = P[a] - P[b]
            m, lo, hi = boot(d)
            fl = float(np.mean((P[a] > .5) != (P[b] > .5)))
            sig = "***" if (lo > 0 or hi < 0) else "n.s."
            print(f"{lab:34s} {m:+10.4f} [{lo:+.4f}, {hi:+.4f}] {fl:7.3f}  {sig}")

        d = P["ctx_yes"] - P["ctx_no"]
        print("\n  CSS by true label:")
        for nm, msk in [("gold = yes", gold == 1), ("gold = no ", gold == 0)]:
            m, lo, hi = boot(d[msk])
            print(f"    {nm}  n={int(msk.sum()):3d}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print("  CSS by question:")
        for nm in ["belong", "burden"]:
            msk = q == nm
            m, lo, hi = boot(d[msk])
            print(f"    {nm:9s}  n={int(msk.sum()):3d}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print()


if __name__ == "__main__":
    main()
