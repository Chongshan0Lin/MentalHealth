#!/usr/bin/env python3
"""I7 Phase 1b (v2) -- probe the PARTICIPANT representation only.

Question: does the interviewer's context change how the model represents the
patient's own words? The participant text is byte-identical across arms, so a
lexical baseline on it is exactly chance and the gate is a real test rather than
the degenerate one v1 produced (TF-IDF on the full prompt reached AUC 1.000,
leaving no headroom).

Gate: does the probe exceed 0.500 with a bootstrap CI excluding it?

Contrasts reported:
  neutral vs clinical          -- matches the headline behavioural effect
  neutral_matched vs clinical  -- length-matched, controls the positional
                                  confound (interviewer pools of 38 vs 38 words
                                  rather than 50 vs 38, so participant tokens
                                  sit at nearly the same absolute positions)

Controls, as in v1: a control-task probe on shuffled labels (Elazar et al.
2021), and grouped CV by participant since each participant contributes one item
to each class.
"""
from __future__ import annotations

import json
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
rng = np.random.default_rng(0)


def oof_scores(X, y, groups, n_splits=10, pca=256):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        k = min(pca, len(tr) - 1, X.shape[1])
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=k, random_state=0),
                             LogisticRegression(max_iter=3000, C=0.1))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def boot_ci(y, s, B=2000):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def main():
    d = np.load(f"{RES}/daic_hidden_participant.npz", allow_pickle=True)
    H, layers, arms = d["H"], d["layers"], [str(a) for a in d["arms"]]
    meta = json.load(open(f"{RES}/daic_hidden_participant_meta.json"))
    n_p = len(meta)
    print(f"hidden states: {H.shape}  arms={arms}\n")

    pid = np.array([m["participant_id"] for m in meta])

    # ---- lexical baseline on the PARTICIPANT text (identical across arms) ---
    rows = [json.loads(l) for l in open(COND)]
    texts = [r["text_participant_only"] for r in rows] * 2
    y2 = np.concatenate([np.zeros(n_p), np.ones(n_p)]).astype(int)
    g2 = np.concatenate([pid, pid])
    Xt = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                         sublinear_tf=True).fit_transform(texts).toarray()
    lex = roc_auc_score(y2, oof_scores(Xt, y2, g2))
    print(f"LEXICAL BASELINE (participant text only, identical across arms): "
          f"AUC {lex:.3f}")
    print("  -- must be ~0.500 by construction; anything else is a bug\n")

    out = {"lexical": lex, "contrasts": []}
    for a0, a1 in [("cf_neutral", "cf_depression"),
                   ("cf_neutral_matched", "cf_depression")]:
        i0, i1 = arms.index(a0), arms.index(a1)
        print("=" * 74)
        print(f"CONTRAST: {a0}  vs  {a1}")
        print("=" * 74)
        print(f"{'layer':>5s} {'probe AUC':>10s} {'[95% CI]':>16s} "
              f"{'control':>8s} {'> 0.500?':>9s}")
        print("-" * 74)
        y = np.concatenate([np.zeros(n_p), np.ones(n_p)]).astype(int)
        g = np.concatenate([pid, pid])
        y_ctrl = rng.permutation(y)
        for li, L in enumerate(layers):
            X = np.concatenate([H[:, i0, li, :], H[:, i1, li, :]])
            oof = oof_scores(X, y, g)
            a = roc_auc_score(y, oof)
            lo, hi = boot_ci(y, oof)
            ca = roc_auc_score(y_ctrl, oof_scores(X, y_ctrl, g))
            sig = "YES" if lo > 0.5 else "no"
            print(f"{int(L):5d} {a:10.3f} [{lo:6.3f},{hi:6.3f}] {ca:8.3f} "
                  f"{sig:>9s}")
            out["contrasts"].append(dict(a0=a0, a1=a1, layer=int(L), auc=a,
                                         ci=[lo, hi], control_auc=ca,
                                         separable=bool(lo > 0.5)))
        print()

    json.dump(out, open(f"{RES}/phase1_probes_participant.json", "w"), indent=2)
    prim = [c for c in out["contrasts"]
            if c["a0"] == "cf_neutral" and c["separable"]]
    print("=" * 74)
    print("PHASE 1 GATE (v2)")
    print("=" * 74)
    if prim:
        b = max(prim, key=lambda c: c["auc"])
        print(f"  PASS -- layer {b['layer']}: AUC {b['auc']:.3f} "
              f"[{b['ci'][0]:.3f}, {b['ci'][1]:.3f}], control {b['control_auc']:.3f}")
        print("  The interviewer's context DOES change how the model represents")
        print("  the patient's own words. -> proceed to Phase 2.")
    else:
        print("  FAIL -- no layer separates the arms from the participant")
        print("  representation alone.")
        print("  The patient's representation is UNCHANGED by interviewer context;")
        print("  the effect must arise at readout, not in the evidence encoding.")
        print("  This is consistent with the threshold-shift mechanism (ranking")
        print("  preserved, operating point moved) and is a reportable positive")
        print("  finding -- but it means §4.4 should be replaced, not kept.")
    print(f"\nwrote {RES}/phase1_probes_participant.json")


if __name__ == "__main__":
    main()
