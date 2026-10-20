#!/usr/bin/env python3
"""I7 Phase 1b -- train the biographical-vs-clinical probe, with both controls.

Reports, for every layer and pooling:
  probe        linear probe on hidden states
  control      identical probe on RANDOMLY SHUFFLED labels (Elazar et al. 2021)
               -- matched in dimensionality, sample size and training procedure,
               so it measures what removing this many degrees of freedom costs
               by itself. This is the baseline Phase 3's causal claim rests on.
  lexical      TF-IDF + logistic regression on the raw prompt text

The lexical ceiling is the gate. Decoding the arm from surface tokens is close
to trivial -- the word "diagnosed" appears in one arm and not the other -- so a
hidden-state probe that merely matches TF-IDF has demonstrated nothing about
the representation. C1's final-layer margin over TF-IDF was exactly zero and the
first draft omitted that; it is reported here at every layer regardless of
outcome.

Cross-validation is grouped by participant. Each participant contributes one
item to EACH class, so ungrouped CV would place the same person in train and
test and inflate everything. We use grouped 10-fold rather than the
leave-one-participant-out named in PLAN.md: with 188 participants LOO means 188
PCA+fit cycles per cell, and grouped 10-fold gives the same leakage guarantee at
a twentieth of the cost. Deviation recorded here rather than silently taken.

PCA to 256 dimensions is applied inside each fold before the probe. It is a
linear map, so linear decodability is preserved; without it lbfgs on 4096
standardised dimensions does not converge (a lesson from C1).
"""
from __future__ import annotations

import importlib.util, json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = "/data/cl195/NeurIPS_workshop"
RES = f"{ROOT}/experiments/i7-causal-probing/results"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
ARMS = ["cf_neutral", "cf_depression"]
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')
rng = np.random.default_rng(0)

_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
POOLS, interleave = _a2b.POOLS, _a2b.interleave


def cv_auc(X, y, groups, n_splits=10, pca=256):
    """Grouped CV; returns out-of-fold AUC and accuracy."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        k = min(pca, len(tr) - 1, X.shape[1])
        pipe = make_pipeline(StandardScaler(), PCA(n_components=k,
                                                   random_state=0),
                             LogisticRegression(max_iter=3000, C=0.1))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof), float(((oof > 0.5).astype(int) == y).mean())


def boot_ci(y, s, B=2000):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def main():
    d = np.load(f"{RES}/daic_hidden.npz", allow_pickle=True)
    H, layers = d["H"], d["layers"]
    meta = json.load(open(f"{RES}/daic_hidden_meta.json"))
    n_p = len(meta)
    print(f"hidden states: {H.shape}  ({n_p} participants x {len(ARMS)} arms x "
          f"{len(layers)} layers x 2 pools)\n")

    # flatten: each participant contributes one item per arm
    groups = np.repeat([m["participant_id"] for m in meta], len(ARMS))
    y = np.tile([0, 1], n_p)                    # 0=neutral, 1=clinical
    y_ctrl = rng.permutation(y)                 # control task

    # ---- lexical ceiling ---------------------------------------------------
    rows = [json.loads(l) for l in open(COND)]
    texts = []
    for r in rows:
        for arm in ARMS:
            texts.append(TEMPLATE.format(
                text=interleave(r["text_participant_only"], POOLS[arm])))
    Xt = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                         sublinear_tf=True).fit_transform(texts).toarray()
    lex_auc, lex_acc = cv_auc(Xt, y, groups, pca=256)
    print(f"LEXICAL CEILING (TF-IDF on raw text): AUC {lex_auc:.3f}  "
          f"acc {lex_acc:.3f}\n")

    print(f"{'layer':>5s} {'pool':>5s} {'probe AUC':>10s} {'[95% CI]':>16s} "
          f"{'acc':>6s} {'control':>8s} {'vs lexical':>11s}")
    print("-" * 78)
    out = {"lexical": {"auc": lex_auc, "acc": lex_acc}, "cells": []}
    for li, L in enumerate(layers):
        for pi, pool in enumerate(["mean", "last"]):
            X = H[:, :, li, pi, :].reshape(n_p * len(ARMS), -1)
            a, acc = cv_auc(X, y, groups)
            ca, _ = cv_auc(X, y_ctrl, groups)
            # CI needs out-of-fold scores; recompute once for the CI
            oof = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=10).split(X, y, groups):
                pipe = make_pipeline(StandardScaler(),
                                     PCA(n_components=256, random_state=0),
                                     LogisticRegression(max_iter=3000, C=0.1))
                pipe.fit(X[tr], y[tr]); oof[te] = pipe.predict_proba(X[te])[:, 1]
            lo, hi = boot_ci(y, oof)
            beats = "YES" if lo > lex_auc else "no"
            print(f"{int(L):5d} {pool:>5s} {a:10.3f} [{lo:6.3f},{hi:6.3f}] "
                  f"{acc:6.3f} {ca:8.3f} {beats:>11s}")
            out["cells"].append(dict(layer=int(L), pool=pool, auc=a, acc=acc,
                                     ci=[lo, hi], control_auc=ca,
                                     beats_lexical=bool(lo > lex_auc)))

    json.dump(out, open(f"{RES}/phase1_probes.json", "w"), indent=2)
    any_beat = any(c["beats_lexical"] for c in out["cells"])
    print("\n" + "=" * 78)
    print("PHASE 1 GATE")
    print("=" * 78)
    if any_beat:
        best = max((c for c in out["cells"] if c["beats_lexical"]),
                   key=lambda c: c["auc"])
        print(f"  PASS -- layer {best['layer']} / {best['pool']} exceeds the "
              f"lexical ceiling ({best['auc']:.3f} vs {lex_auc:.3f}, "
              f"CI lower {best['ci'][0]:.3f})")
        print("  -> proceed to Phase 2 (Delta-h vs Delta-p correlation)")
    else:
        print(f"  FAIL -- no layer exceeds the lexical ceiling ({lex_auc:.3f}) "
              f"with a CI excluding it.")
        print("  -> the arm signal is not linearly separable beyond surface")
        print("     vocabulary. Report as a negative result and DELETE §4.4.")
    print(f"\nwrote {RES}/phase1_probes.json")


if __name__ == "__main__":
    main()
