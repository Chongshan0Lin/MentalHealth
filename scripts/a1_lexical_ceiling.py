#!/usr/bin/env python3
"""A1 control -- is the interviewer-prompt shortcut present in the data at all?

A1 returned a null: neither MentaLLaMA nor Menta ranks depression above the
label-shuffle floor from Ellie's turns alone. Three things could produce that,
and the design's falsifiability clause only names the first:

  1. the scoring pipeline is broken,
  2. the shortcut is in the data but these models do not use it,
  3. the shortcut is not recoverable from this text at all.

A supervised linear probe on the same four text conditions separates them. It
is the "shortcut ceiling" of SHORTCUT_STUDY_PLAN Part 1: TF-IDF + logistic
regression sees nothing but surface lexical statistics, so whatever it achieves
is an upper bound on what a lexical shortcut could deliver -- and unlike the
zero-shot LLMs, it is *fitted to DAIC*, so it will find the shortcut if one
exists.

Note the asymmetry this creates, which is the point: the LLMs are zero-shot and
never saw DAIC training data, while the probe is trained on it. A shortcut that
the probe finds and the LLMs miss is case 2, not case 1.

Reported with repeated stratified CV and a label-shuffle floor, matching the
controls used for the model runs.
"""
from __future__ import annotations

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
N_SPLITS, N_REPEATS, SEED = 5, 10, 0

import sys
sys.path.insert(0, ROOT)
from scripts.run_a1 import strip_prompt_ids


def cv_auc(X, y, shuffle=False, seed=SEED):
    """Repeated stratified CV AUC. -> (mean, lo, hi) over folds."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if shuffle:
        y = rng.permutation(y)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=seed)
    aucs = []
    for tr, te in cv.split(X, y):
        clf = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                            max_features=20000),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
        clf.fit([X[i] for i in tr], y[tr])
        p = clf.predict_proba([X[i] for i in te])[:, 1]
        if len(set(y[te].tolist())) > 1:
            aucs.append(roc_auc_score(y[te], p))
    a = np.array(aucs)
    return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)


def main():
    rows = [json.loads(l) for l in open(COND)]
    y = [r["phq8_binary"] for r in rows]
    import re
    ID_RE = re.compile(r"^Ellie: [A-Za-z][A-Za-z0-9_]* \(", re.M)
    has_ids = [bool(ID_RE.search(r["text_full"])) for r in rows]

    conds = {
        "full":                [r["text_full"] for r in rows],
        "full_stripped":       [strip_prompt_ids(r["text_full"]) for r in rows],
        "participant_only":    [r["text_participant_only"] for r in rows],
        "ellie_only":          [r["text_ellie_only"] or "" for r in rows],
        "ellie_only_stripped": [strip_prompt_ids(r["text_ellie_only"] or "")
                                for r in rows],
        "ellie_masked":        [r["text_ellie_masked"] for r in rows],
    }

    print(f"n={len(rows)}  positives={sum(y)}  majority={max(np.mean(y), 1-np.mean(y)):.3f}")
    print(f"{N_SPLITS}-fold x {N_REPEATS} repeats, TF-IDF(1-2gram) + LogReg(balanced)\n")
    print(f"{'condition':22s} {'CV AUC':>8s} {'[2.5, 97.5] over folds':>26s} "
          f"{'shuffled':>10s}")
    print("-" * 74)
    for c, X in conds.items():
        m, lo, hi = cv_auc(X, y)
        sm, _, _ = cv_auc(X, y, shuffle=True)
        print(f"{c:22s} {m:8.3f} {'[%.3f, %.3f]' % (lo, hi):>26s} {sm:10.3f}")

    # The decisive contrast: Ellie's turns alone, split by whether the
    # transcript carries the prompt-ID annotation artifact.
    print(f"\n{'-'*74}\nellie_only by prompt-ID presence (F5)\n{'-'*74}")
    for name, want in [("with IDs", True), ("without IDs", False)]:
        idx = [i for i, h in enumerate(has_ids) if h == want]
        yy = [y[i] for i in idx]
        if len(set(yy)) < 2:
            continue
        for variant in ["ellie_only", "ellie_only_stripped"]:
            XX = [conds[variant][i] for i in idx]
            m, lo, hi = cv_auc(XX, yy)
            print(f"  {name:12s} {variant:22s} n={len(idx):3d}  "
                  f"AUC={m:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
