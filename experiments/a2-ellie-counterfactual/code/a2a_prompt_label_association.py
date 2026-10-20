#!/usr/bin/env python3
"""A2a -- is *which questions the interviewer asked* label-associated?

This is the substrate test for the reframed shortcut pathway. The old framing
(side signal -> identify corpus -> apply its prior) is capped at the majority
baseline and explains nothing. The replacement claims the side signal carries
*other* label-correlated information. In DAIC the obvious candidate is question
selection: the interview is adaptive, so which scripted prompts Ellie chose to
ask is itself a clinical judgement.

No model is involved. This asks only whether the correlation exists in the
corpus -- if it does not, there is nothing for a model to exploit and A2b's
counterfactual has no substrate to detect.

Per scripted prompt ID: depression rate among participants who were asked it vs
those who were not, Fisher exact, Benjamini-Hochberg across the tested IDs.
Only IDs asked to 20-103 of the 123 ID-carrying transcripts are tested -- an ID
asked to everyone (or almost no one) carries no contrast.
"""
from __future__ import annotations

import json, re, collections
import numpy as np
from scipy.stats import fisher_exact

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
PAT = re.compile(r"^Ellie: ([A-Za-z][A-Za-z0-9_]*) \((.*)\)\s*$")
MIN_N, MAX_FRAC = 20, 0.84      # keep IDs with real variation in who was asked


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, p[i] * n / (n - rank + 1))
        adj[i] = prev
    return adj


def main():
    rows = [json.loads(l) for l in open(COND)]
    # restrict to the transcripts that carry scripted IDs at all
    subs = []
    for r in rows:
        ids = {m.group(1) for m in
               (PAT.match(l) for l in (r["text_ellie_only"] or "").split("\n"))
               if m}
        if ids:
            subs.append({"uid": r["uid"], "y": r["phq8_binary"], "ids": ids,
                         "text": {m.group(1): m.group(2) for m in
                                  (PAT.match(l) for l in
                                   (r["text_ellie_only"] or "").split("\n")) if m}})
    N = len(subs)
    base = np.mean([s["y"] for s in subs])
    print(f"transcripts with scripted IDs: {N}   depression rate: {base:.3f}")

    all_ids = collections.Counter(i for s in subs for i in s["ids"])
    tested = [i for i, n in all_ids.items() if MIN_N <= n <= MAX_FRAC * N]
    print(f"distinct IDs: {len(all_ids)}   tested (asked to {MIN_N}-"
          f"{int(MAX_FRAC*N)} participants): {len(tested)}\n")

    recs = []
    for pid in tested:
        asked = [s["y"] for s in subs if pid in s["ids"]]
        notas = [s["y"] for s in subs if pid not in s["ids"]]
        tab = [[sum(asked), len(asked) - sum(asked)],
               [sum(notas), len(notas) - sum(notas)]]
        orr, p = fisher_exact(tab)
        txt = collections.Counter(
            s["text"][pid] for s in subs if pid in s["text"]).most_common(1)[0][0]
        recs.append({"id": pid, "n_asked": len(asked),
                     "rate_asked": float(np.mean(asked)),
                     "rate_not": float(np.mean(notas)) if notas else float("nan"),
                     "odds_ratio": float(orr), "p": float(p), "text": txt})

    for r, q in zip(recs, bh([r["p"] for r in recs])):
        r["q"] = float(q)
    recs.sort(key=lambda r: r["p"])

    sig = [r for r in recs if r["q"] < 0.05]
    print(f"{'='*112}")
    print(f"SIGNIFICANT AFTER BH CORRECTION: {len(sig)} / {len(recs)}")
    print(f"{'='*112}")
    print(f"{'prompt id':26s} {'n':>4s} {'dep|asked':>10s} {'dep|not':>8s} "
          f"{'OR':>6s} {'p':>9s} {'q':>9s}  question")
    print("-" * 112)
    for r in (sig if sig else recs[:15]):
        print(f"{r['id'][:25]:26s} {r['n_asked']:4d} {r['rate_asked']:10.3f} "
              f"{r['rate_not']:8.3f} {r['odds_ratio']:6.2f} {r['p']:9.2e} "
              f"{r['q']:9.2e}  {r['text'][:44]}")
    if not sig:
        print("(none significant; showing the 15 smallest raw p-values)")

    print(f"\n{'='*112}\nSTRONGEST ASSOCIATIONS BY EFFECT SIZE (raw p < .05, "
          f"either direction)\n{'='*112}")
    strong = sorted([r for r in recs if r["p"] < .05],
                    key=lambda r: -abs(np.log(max(r["odds_ratio"], 1e-9))))
    for r in strong[:12]:
        d = "MORE depressed" if r["odds_ratio"] > 1 else "LESS depressed"
        print(f"  {r['id'][:25]:26s} OR={r['odds_ratio']:5.2f} ({d})  "
              f"{r['rate_asked']:.2f} vs {r['rate_not']:.2f}   {r['text'][:40]}")

    out = f"{ROOT}/experiments/a2-ellie-counterfactual/results/a2a_prompt_label_association.json"
    with open(out, "w") as f:
        json.dump({"n_transcripts": N, "base_rate": float(base),
                   "n_tested": len(recs), "n_significant_bh": len(sig),
                   "records": recs}, f, indent=2)
    print(f"\nwrote {out}")

    # How much of the label can question-selection alone predict?  A binary
    # ask/not-ask matrix over the tested IDs -- no text, no counts, just which
    # prompts were asked. This is the tightest possible statement of "the
    # interviewer's choices carry the label".
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    X = np.array([[1 if pid in s["ids"] else 0 for pid in tested] for s in subs])
    y = np.array([s["y"] for s in subs])
    # `cross_val_predict` requires a partition, so repeats are averaged over
    # separate StratifiedKFold seeds rather than via RepeatedStratifiedKFold.
    aucs, shufs = [], []
    for seed in range(10):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        p = cross_val_predict(
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
            X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
        rng = np.random.default_rng(seed)
        shufs.append(np.mean([roc_auc_score(rng.permutation(y), p)
                              for _ in range(100)]))
    print(f"\n{'='*112}")
    print("QUESTION-SELECTION-ONLY PREDICTOR (binary asked/not-asked matrix, "
          "no text at all)")
    print(f"{'='*112}")
    print(f"  features: {X.shape[1]} prompt IDs   n={len(y)}   "
          f"majority baseline = {max(base, 1-base):.3f}")
    print(f"  CV AUC = {np.mean(aucs):.3f}  (sd {np.std(aucs):.3f} over 10 "
          f"CV seeds;  shuffled {np.mean(shufs):.3f})")
    with open(f"{ROOT}/experiments/a2-ellie-counterfactual/results/"
              f"a2a_question_selection_auc.json", "w") as f:
        json.dump({"auc_mean": float(np.mean(aucs)),
                   "auc_sd": float(np.std(aucs)),
                   "shuffled": float(np.mean(shufs)),
                   "n_features": int(X.shape[1]), "n": int(len(y))}, f, indent=2)


if __name__ == "__main__":
    main()
