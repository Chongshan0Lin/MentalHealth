#!/usr/bin/env python3
"""A2b analysis -- CF-Sensitivity with the participant's evidence held fixed.

Every arm shares byte-identical participant text, so a shift in p(depression)
between arms can only come from the interviewer lines. That is what makes this
a causal read where A1's ablation was not: ablation removed the interviewer
turns and any legitimate signal entangled with them at the same time.

Paired throughout -- each participant contributes one difference, and the
bootstrap resamples participants, not observations.
"""
from __future__ import annotations

import json, sys, collections
import numpy as np

ROOT = "/data/cl195/NeurIPS_workshop"
N_BOOT = 4000
RNG = np.random.default_rng(0)
ARMS = ["cf_none", "cf_neutral", "cf_depression", "cf_empirical"]


def p_yes(lp):
    v = np.array([lp["Yes"], lp["No"]], dtype=np.float64)
    e = np.exp(v - v.max())
    return float(e[0] / e.sum())


def boot_mean(x, n=N_BOOT):
    x = np.asarray(x, dtype=float)
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
    path = sys.argv[1] if len(sys.argv) > 1 else \
        f"{ROOT}/experiments/a2-ellie-counterfactual/results/a2b_MentaLLaMA-chat-7B.jsonl"
    by = collections.defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        by[r["arm"]][r["uid"]] = r
    model = json.loads(open(path).readline())["model"]
    uids = sorted(by["cf_none"])
    gold = np.array([by["cf_none"][u]["phq8_binary"] for u in uids])
    print(f"loaded {path}\nmodel={model}  n={len(uids)}  "
          f"positives={int(gold.sum())}  truncated C: "
          f"{sum(by['cf_none'][u]['c_truncated'] for u in uids)}/{len(uids)}")

    P = {a: np.array([p_yes(by[a][u]["logprobs"]) for u in uids]) for a in ARMS}

    print(f"\n{'='*100}\nPER-ARM (participant evidence byte-identical "
          f"throughout)\n{'='*100}")
    print(f"{'arm':16s} {'mean p(dep)':>12s} {'[95% CI]':>18s} "
          f"{'acc':>7s} {'AUC':>7s} {'P(pred=Yes)':>12s}")
    print("-" * 100)
    for a in ARMS:
        m, lo, hi = boot_mean(P[a])
        acc = np.mean([(P[a][i] > .5) == gold[i] for i in range(len(uids))])
        pr = float((P[a] > .5).mean())
        print(f"{a:16s} {m:12.4f} [{lo:.4f}, {hi:.4f}] {acc:7.3f} "
              f"{auc(gold, P[a]):7.3f} {pr:12.3f}")

    print(f"\n{'='*100}\nCF-SENSITIVITY -- paired within participant\n{'='*100}")
    contrasts = [
        ("cf_depression", "cf_neutral", "CONFIRMATORY: clinical vs biographical"),
        ("cf_empirical",  "cf_neutral", "EXPLORATORY:  A2a-derived vs biographical"),
        ("cf_depression", "cf_none",    "clinical vs no interviewer"),
        ("cf_neutral",    "cf_none",    "biographical vs no interviewer"),
        ("cf_empirical",  "cf_none",    "A2a-derived vs no interviewer"),
    ]
    for a, b, label in contrasts:
        d = P[a] - P[b]
        m, lo, hi = boot_mean(d)
        flip = float(np.mean((P[a] > .5) != (P[b] > .5)))
        sig = "***" if lo > 0 or hi < 0 else "n.s."
        print(f"  {label}")
        print(f"    Δp(dep) = {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig:>5s}   "
              f"|Δ| mean {np.abs(d).mean():.4f}   pred flips {flip:.3f}")

    # Does the effect concentrate where the corpus effect is?  A2a found the
    # association only in the 123 ID-carrying transcripts; the models never see
    # the IDs, so there is no reason for a split unless something else differs.
    print(f"\n{'='*100}\nBY TRUE LABEL -- does interviewer context move the two "
          f"groups differently?\n{'='*100}")
    for a, b, label in contrasts[:2]:
        d = P[a] - P[b]
        print(f"  {label}")
        for name, mask in [("depressed", gold == 1), ("not depressed", gold == 0)]:
            m, lo, hi = boot_mean(d[mask])
            print(f"    {name:15s} n={int(mask.sum()):3d}  Δ = {m:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
