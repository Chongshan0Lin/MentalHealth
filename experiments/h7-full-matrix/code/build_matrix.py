#!/usr/bin/env python3
"""Per-dataset model tables, in the same shape as the 8-model DAIC sweep.

Reads the original H4/H5/H6 results and the H7 fill-in runs together, so every
model appears once per dataset regardless of which experiment produced it.
"""
from __future__ import annotations
import collections, glob, json, os, sys
import numpy as np

R = "/data/cl195/NeurIPS_workshop"
RNG = np.random.default_rng(0)
MODELS = ["MentaLLaMA", "Menta", "qwen3_base", "OSim-8B",
          "OmniSapiens-SFT", "OmniSapiens-RL", "OmniSapiens2.0", "HumanOmniV2"]

SPEC = {
    "mmpsy": dict(arms=["seg_minus", "native", "seg_plus"], pos="depression",
                  neg="no_depression", hi="seg_plus", lo="seg_minus",
                  base="native", key="idx"),
    "ptsd":  dict(arms=["ctx_vlog", "native", "ctx_testimony"], pos="ptsd",
                  neg="no_ptsd", hi="ctx_testimony", lo="ctx_vlog",
                  base="native", key="idx"),
    "irf":   dict(arms=["ctx_no", "native", "ctx_neutral", "ctx_yes"], pos="yes",
                  neg="no", hi="ctx_yes", lo="ctx_no", base="native", key="uid"),
}
# where each model's rows live: H7 fill-ins, plus the original experiments
LEGACY = {
    ("mmpsy", "OmniSapiens-SFT"): "experiments/h4-mmpsy-counterfactual/results/h4_mmpsy_depression_SFT.jsonl",
    ("mmpsy", "OmniSapiens-RL"):  "experiments/h4-mmpsy-counterfactual/results/h4_mmpsy_depression_RL.jsonl",
    ("mmpsy", "OmniSapiens2.0"):  "experiments/h4-mmpsy-counterfactual/results/h4_mmpsy_depression_HARPO.jsonl",
    ("ptsd",  "OmniSapiens-SFT"): "experiments/h5-ptsd-register-cf/results/h5_ptsd_SFT.jsonl",
    ("ptsd",  "OmniSapiens-RL"):  "experiments/h5-ptsd-register-cf/results/h5_ptsd_RL.jsonl",
    ("ptsd",  "OmniSapiens2.0"):  "experiments/h5-ptsd-register-cf/results/h5_ptsd_HARPO.jsonl",
    ("irf",   "MentaLLaMA"):      "experiments/h6-irf-register-cf/results/h6_irf_MentaLLaMA-chat-7B.jsonl",
    ("irf",   "Menta"):           "experiments/h6-irf-register-cf/results/h6_irf_Menta.jsonl",
}


def path(task, model):
    p = f"{R}/experiments/h7-full-matrix/results/{task}_{model}.jsonl"
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    q = LEGACY.get((task, model))
    if q and os.path.exists(f"{R}/{q}"):
        return f"{R}/{q}"
    return None


def boot(x, n=3000):
    x = np.asarray(x, float)
    if len(x) < 3: return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan)
    d = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), *np.percentile(d, [2.5, 97.5])


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    a, b = p[y == 1], p[y == 0]
    if not len(a) or not len(b): return float("nan")
    return float(((a[:, None] > b[None, :]).sum()
                  + .5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def row(task, model):
    sp = SPEC[task]; p = path(task, model)
    if not p: return None
    by = collections.defaultdict(dict)
    for l in open(p):
        r = json.loads(l); by[r["arm"]][r[sp["key"]]] = r
    if not all(a in by for a in sp["arms"]): return None
    ids = sorted(set.intersection(*[set(by[a]) for a in sp["arms"]]))
    if len(ids) < 10: return None
    gold = np.array([1 if by[sp["base"]][i]["gold"] == sp["pos"] else 0 for i in ids])

    def pv(a):
        o = []
        for i in ids:
            lp = by[a][i]["logprobs"]
            v = np.array([lp[sp["pos"]], lp[sp["neg"]]], float)
            e = np.exp(v - v.max()); o.append(e[0] / e.sum())
        return np.array(o)
    P = {a: pv(a) for a in sp["arms"]}
    css, lo, hi = boot(P[sp["hi"]] - P[sp["lo"]])
    flips = float(np.mean((P[sp["hi"]] > .5) != (P[sp["lo"]] > .5)))
    # construct sensitivity proxy: spread of p across arms, per item
    cs = float(np.mean(np.ptp(np.vstack([P[a] for a in sp["arms"]]), axis=0)))
    base_auc = auc(gold, P[sp["base"]])
    worst = min(float(np.mean((P[a] > .5) == gold)) for a in sp["arms"])
    return dict(n=len(ids), css=css, lo=lo, hi=hi, flips=flips, cs=cs,
                base_auc=base_auc, base_acc=float(np.mean((P[sp["base"]] > .5) == gold)),
                worst_acc=worst,
                src="H7" if "h7-full-matrix" in p else "orig")


def verdict(css, flips):
    if abs(css) < 0.02: return "**null**"
    if css < 0: return "**wrong sign**"
    if css < 0.10: return "weak"
    if css < 0.25: return "moderate"
    return "**strong**"


TITLE = {"mmpsy": "`mmpsy_depression` — structural shortcut (answer-segment count)",
         "ptsd": "`ptsd_in_the_wild` — register shortcut, cross-source",
         "irf": "`Irf` — register shortcut, same-source"}
RATE = {"mmpsy": "segment count 0.772 (content 0.507, chance 0.500)",
        "ptsd": "text 0.948, stopwords only 0.880 (chance 0.500)",
        "irf": "text 0.643, stopwords only 0.622 (chance 0.500)"}

for task in ["mmpsy", "ptsd", "irf"]:
    print(f"\n### {TITLE[task]}\n")
    print(f"Shortcut rate (model-free): **{RATE[task]}**\n")
    print("| model | n | CSS | 95% CI | flips | native AUC | native acc | worst-arm acc | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    got = 0
    for m in MODELS:
        r = row(task, m)
        if not r:
            print(f"| {m} | — | — | — | — | — | — | — | *not run* |")
            continue
        got += 1
        print(f"| {m} | {r['n']} | **{r['css']:+.4f}** | [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
              f"{r['flips']:.3f} | {r['base_auc']:.3f} | {r['base_acc']:.3f} | "
              f"{r['worst_acc']:.3f} | {verdict(r['css'], r['flips'])} |")
    print(f"\n({got}/8 models)")
