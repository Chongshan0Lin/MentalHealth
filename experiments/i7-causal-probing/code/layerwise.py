#!/usr/bin/env python3
"""I7 -- layer-by-layer: where does the shortcut signal live, and where does it
predict behaviour?

Two questions, following the reconnaissance framing (the point of a layer sweep
is to choose WHERE to intervene, not to look thorough):

  Q1  At which layer is the biographical-vs-clinical signal strongest?
  Q2  Is that the same layer at which the signal predicts the behavioural
      shift dp?

If Q1 and Q2 peak at different depths, that is a sharper statement of
"decodable != used" than "every layer decodes it".

WHY NOT AUC. Phase 1 found probe AUC = 1.000 at every layer tested: the measure
saturates and has no peak to find. We therefore report CROSS-FITTED COHEN'S d
along the discriminant direction as the strength measure. The direction is fitted
on the training fold and the effect size evaluated on the held-out fold, so it is
not the circular quantity you get by measuring separation along the direction
that was chosen to maximise it. AUC is still printed, to document the saturation.

MESSAGE PASSING. We also track how the contamination accumulates with depth:
relative displacement ||dh|| / ||h|| per layer, and the layer-to-layer increment.
"""
from __future__ import annotations

import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = "/data/cl195/NeurIPS_workshop"
RES = f"{ROOT}/experiments/i7-causal-probing/results"
A2B = f"{ROOT}/experiments/a2-ellie-counterfactual/results/a2b_MentaLLaMA-chat-7B.jsonl"
rng = np.random.default_rng(0)
sig = lambda x: 1 / (1 + np.exp(-x))


def partial_corr(x, y, z):
    Z = np.vstack([np.ones_like(z), z]).T
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)


def boot_partial(x, y, z, B=2000):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        try:
            v.append(partial_corr(x[i], y[i], z[i])[0])
        except Exception:
            pass
    return np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)


def bar(v, lo, hi, w=22):
    if not np.isfinite(v):
        return " " * w
    n = int(round(w * (v - lo) / (hi - lo + 1e-12)))
    return "#" * max(0, min(w, n))


def main():
    d = np.load(f"{RES}/daic_hidden_participant_alllayers.npz", allow_pickle=True)
    H, layers, arms = d["H"], d["layers"], [str(a) for a in d["arms"]]
    meta = json.load(open(f"{RES}/daic_hidden_participant_alllayers_meta.json"))
    pid = np.array([m["participant_id"] for m in meta])
    i0, i1 = arms.index("cf_neutral"), arms.index("cf_depression")
    n = len(pid)

    lp = {}
    for line in open(A2B):
        r = json.loads(line)
        lp.setdefault(r["arm"], {})[r["participant_id"]] = (
            r["logprobs"]["Yes"] - r["logprobs"]["No"])
    dp = np.array([sig(lp["cf_depression"][p]) - sig(lp["cf_neutral"][p])
                   for p in pid])

    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    g = np.concatenate([pid, pid])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(2 * n), y, g))

    rows = []
    for li, L in enumerate(layers):
        A, B = H[:, i0, li, :], H[:, i1, li, :]
        D = B - A
        nrm = np.linalg.norm(D, axis=1)
        rel = float((nrm / (np.linalg.norm(A, axis=1) + 1e-9)).mean())

        # cross-fitted Cohen's d: direction from train fold, effect on test fold
        ds, aucs, oof = [], [], np.zeros(2 * n)
        X = np.concatenate([A, B])
        for tr, te in folds:
            mu = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
            u = mu / (np.linalg.norm(mu) + 1e-9)
            p_te = X[te] @ u
            a_, b_ = p_te[y[te] == 0], p_te[y[te] == 1]
            sd = np.sqrt(((a_.var(ddof=1) + b_.var(ddof=1)) / 2)) + 1e-12
            ds.append((b_.mean() - a_.mean()) / sd)
            oof[te] = p_te
        auc = roc_auc_score(y, oof)

        # behavioural prediction, along the mean-shift direction + random control
        mu = D.mean(0); u = mu / np.linalg.norm(mu)
        pr = partial_corr(D @ u, dp, nrm)
        lo, hi = boot_partial(D @ u, dp, nrm)
        rv = rng.normal(size=D.shape[1]); rv /= np.linalg.norm(rv)
        pr_r = partial_corr(D @ rv, dp, nrm)[0]

        rows.append(dict(layer=int(L), cohen_d=float(np.mean(ds)),
                         auc=float(auc), rel_disp=rel,
                         cos_mean=float(((D @ u) / (nrm + 1e-9)).mean()),
                         partial_r=float(pr[0]), ci=[float(lo), float(hi)],
                         random_r=float(pr_r)))

    dmax = max(r["cohen_d"] for r in rows)
    rmax = max(abs(r["partial_r"]) for r in rows)
    print("=" * 104)
    print("LAYER-BY-LAYER: signal strength vs behavioural prediction")
    print("=" * 104)
    print(f"{'L':>3s} {'Cohen d':>8s} {'strength':<24s} {'AUC':>6s} "
          f"{'rel disp':>9s} {'cos':>5s} {'partial r':>10s} {'[95% CI]':>16s} "
          f"{'rand':>6s}")
    print("-" * 104)
    for r in rows:
        star = "*" if r["ci"][0] > 0 or r["ci"][1] < 0 else " "
        print(f"{r['layer']:3d} {r['cohen_d']:8.2f} {bar(r['cohen_d'],0,dmax):<24s} "
              f"{r['auc']:6.3f} {r['rel_disp']:9.4f} {r['cos_mean']:5.2f} "
              f"{r['partial_r']:9.3f}{star} "
              f"[{r['ci'][0]:6.3f},{r['ci'][1]:6.3f}] {r['random_r']:6.3f}")

    peak_d = max(rows, key=lambda r: r["cohen_d"])
    peak_r = max(rows, key=lambda r: abs(r["partial_r"]))
    sig_rows = [r for r in rows if r["ci"][0] > 0 or r["ci"][1] < 0]
    rand_sig = sum(1 for r in rows if abs(r["random_r"]) > abs(r["partial_r"]))

    print("\n" + "=" * 104)
    print("Q1 -- where is the signal strongest?")
    print("=" * 104)
    print(f"  peak Cohen's d = {peak_d['cohen_d']:.2f} at layer {peak_d['layer']}"
          f"   (AUC there = {peak_d['auc']:.3f})")
    print(f"  AUC range across all layers: "
          f"{min(r['auc'] for r in rows):.3f} - {max(r['auc'] for r in rows):.3f}"
          f"   <- saturated, which is why Cohen's d is the strength measure")

    print("\n" + "=" * 104)
    print("Q2 -- where does the signal predict behaviour?")
    print("=" * 104)
    print(f"  peak |partial r| = {peak_r['partial_r']:+.3f} at layer "
          f"{peak_r['layer']}  (random control there: {peak_r['random_r']:+.3f})")
    print(f"  layers with CI excluding zero: "
          f"{[r['layer'] for r in sig_rows]}")
    print(f"  layers where a RANDOM direction beat the real one: {rand_sig}/{len(rows)}")

    print("\n" + "=" * 104)
    print("Q1 vs Q2")
    print("=" * 104)
    print(f"  strength peaks at layer {peak_d['layer']}, "
          f"behavioural prediction peaks at layer {peak_r['layer']}")
    if peak_d["layer"] != peak_r["layer"]:
        print("  -> DIFFERENT layers. Where the signal is most decodable is not")
        print("     where it most relates to behaviour: a sharper form of")
        print("     'decodable != used' than a per-layer decodability table.")
    else:
        print("  -> SAME layer; strength and behavioural relevance coincide.")
    print(f"\n  If Phase 3 were run, the reconnaissance says to erase at layer "
          f"{peak_r['layer']}\n  (behavioural relevance), NOT layer {peak_d['layer']} "
          f"(raw decodability).")

    json.dump(rows, open(f"{RES}/layerwise.json", "w"), indent=2)
    print(f"\nwrote {RES}/layerwise.json")


if __name__ == "__main__":
    main()
