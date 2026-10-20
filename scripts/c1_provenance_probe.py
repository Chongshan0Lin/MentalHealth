#!/usr/bin/env python3
"""C1b -- is the source corpus recoverable from the post text alone?

The provenance shortcut has two ingredients, and they must be shown separately:

  (i)  source identity is recoverable from the input   <- this script
  (ii) source identity is worth knowing, i.e. sources have different label
       priors                                          <- this script
  (iii) the model actually encodes it                  <- c1_hidden_probe.py

A TF-IDF + logistic regression probe over post text answers (i) at the data
level: it sees nothing but surface lexical statistics, so whatever it recovers
is available to any model. (ii) is a property of the label distributions and
needs no model at all -- but without it, (i) is harmless, because knowing the
source buys nothing if every source has the same prior.

Reads the uid list straight out of a C1 run so the probe is fitted on exactly
the matched subset the models were scored on, with no re-derivation of the
sampling.

The sharpest single contrast is **swmh vs t-sid**: near-identical questions,
overlapping label spaces, different priors. If a bag of words separates those
two corpora, "recognize the dataset, then guess its prior" is available.
"""
from __future__ import annotations

import json, sys, collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.pipeline import make_pipeline

ROOT = "/data/cl195/NeurIPS_workshop"
MANIFEST = f"{ROOT}/data/processed/text_manifest.jsonl"
SEED = 0


def pipe():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                        max_features=50000),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))


def probe(X, y, n_splits=5, label="probe"):
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pred = cross_val_predict(pipe(), X, y, cv=cv)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    rng = np.random.default_rng(SEED)
    shuf = np.mean([accuracy_score(rng.permutation(y), pred) for _ in range(200)])
    print(f"  {label:34s} n={len(y):5d}  classes={len(set(y)):2d}  "
          f"acc={acc:.3f}  macroF1={f1:.3f}  chance={1/len(set(y)):.3f}  "
          f"shuffled={shuf:.3f}")
    return pred, acc


def main():
    run = sys.argv[1] if len(sys.argv) > 1 \
        else f"{ROOT}/runs/c1_MentaLLaMA-chat-7B.jsonl"
    uids = {json.loads(l)["uid"] for l in open(run)}
    rows = [json.loads(l) for l in open(MANIFEST)]
    rows = [r for r in rows if r["uid"] in uids]
    print(f"matched subset from {run}: {len(rows)} rows\n")

    X = [r["text"] for r in rows]
    src = [r["source_dataset"] for r in rows]

    # ---- (ii) is source identity worth knowing? -------------------------
    print("=" * 92)
    print("(ii) LABEL PRIOR BY SOURCE -- knowing the corpus is only useful if "
          "priors differ")
    print("=" * 92)
    for s in sorted(set(src)):
        labs = collections.Counter(r["label"] for r in rows
                                   if r["source_dataset"] == s)
        n = sum(labs.values())
        top, topn = labs.most_common(1)[0]
        p = np.array(list(labs.values())) / n
        H = -(p * np.log2(p)).sum()
        print(f"  {s:12s} n={n:4d}  |L|={len(labs)}  majority={topn/n:.3f} "
              f"({top})  H={H:.2f} bits  (max {np.log2(len(labs)):.2f})")

    # ---- (i) is source recoverable from text? ---------------------------
    print(f"\n{'='*92}")
    print("(i) SOURCE-IDENTITY PROBE -- TF-IDF + LogReg on post text")
    print("=" * 92)
    pred, _ = probe(X, src, label="10-way source identity")

    print("\n  per-source recall:")
    labs = sorted(set(src))
    cm = confusion_matrix(src, pred, labels=labs)
    for i, s in enumerate(labs):
        rec = cm[i, i] / cm[i].sum() if cm[i].sum() else float("nan")
        conf = sorted(((cm[i, j], labs[j]) for j in range(len(labs)) if j != i),
                      reverse=True)[0]
        print(f"    {s:12s} recall={rec:.3f}   most confused with "
              f"{conf[1]} ({conf[0]})")

    # ---- the sharp pair --------------------------------------------------
    print(f"\n{'='*92}")
    print("SHARPEST PAIR -- swmh vs t-sid (near-identical task, different prior)")
    print("=" * 92)
    idx = [i for i, s in enumerate(src) if s in ("swmh", "t-sid")]
    if idx:
        XX = [X[i] for i in idx]
        yy = np.array([1 if src[i] == "swmh" else 0 for i in idx])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        p = cross_val_predict(pipe(), XX, yy, cv=cv, method="predict_proba")[:, 1]
        print(f"  binary corpus probe   n={len(yy)}  "
              f"AUC={roc_auc_score(yy, p):.3f}  "
              f"acc={accuracy_score(yy, (p > .5).astype(int)):.3f}")

    # ---- pairwise separability ------------------------------------------
    print(f"\n{'='*92}")
    print("PAIRWISE CORPUS SEPARABILITY (AUC) -- how distinguishable is each pair?")
    print("=" * 92)
    labs = sorted(set(src))
    print("      " + "".join(f"{s[:6]:>8s}" for s in labs))
    for a in labs:
        cells = []
        for b in labs:
            if a >= b:
                cells.append(f"{'':>8s}")
                continue
            idx = [i for i, s in enumerate(src) if s in (a, b)]
            yy = np.array([1 if src[i] == a else 0 for i in idx])
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            p = cross_val_predict(pipe(), [X[i] for i in idx], yy, cv=cv,
                                  method="predict_proba")[:, 1]
            cells.append(f"{roc_auc_score(yy, p):8.3f}")
        print(f"{a[:6]:>6s}" + "".join(cells))


if __name__ == "__main__":
    main()
