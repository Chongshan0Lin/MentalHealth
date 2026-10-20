#!/usr/bin/env python3
"""I7 across models -- is the shortcut written into the PATIENT representation
in every model, or only in the domain-fine-tuned ones?

`layerwise.py` answered this for MentaLLaMA and is left untouched as the frozen
single-model analysis. This script runs the same two questions over every model
whose hidden states have been extracted, so the probing evidence covers the same
model set as the behavioural evidence.

  Q1  How strongly is `cf_depression` vs `cf_neutral` written into the
      participant tokens, layer by layer?  (cross-fitted Cohen's d)
  Q2  Where does that signal predict the behavioural shift dp?  (partial r,
      controlling for displacement norm)

The controlled contrast the model set was chosen for:

    qwen3_base   Qwen3-4B-base, no adaptation
    menta        the SAME base + the Menta clinical LoRA

Backbone, tokenizer and prompt are identical, so any difference between those
two rows is attributable to the clinical fine-tuning and nothing else.

DEPTH IS REPORTED RELATIVELY. MentaLLaMA has 33 extraction points and the Qwen3
models have 37; "layer 13" is not the same place in the two networks. Bands are
therefore reported as a fraction of depth, and only relative bands are compared
across models.

TWO CONTROLS, both required before any row is read as signal:
  - **control task** (Elazar et al. 2021): the arm labels are shuffled *within
    participant* and the whole cross-fitted pipeline re-run. A probe that scores
    on shuffled labels is reading capacity, not the concept.
  - **random direction**: for Q2, dp is correlated against a random unit
    direction in the same displacement space.
"""
from __future__ import annotations

import argparse, json, os

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = "/data/cl195/NeurIPS_workshop"
RES = f"{ROOT}/experiments/i7-causal-probing/results"
A2B = f"{ROOT}/experiments/a2-ellie-counterfactual/results"
rng = np.random.default_rng(0)
sig = lambda x: 1 / (1 + np.exp(-x))

# tag -> (hidden-state npz stem, a2b per-participant file stem or None)
MODELS = {
    "mentallama":  ("daic_hidden_participant_alllayers", "a2b_MentaLLaMA-chat-7B"),
    "menta":       ("hidden_menta",                      "a2b_Menta"),
    "qwen3_base":  ("hidden_qwen3_base",                  "a2b_qwen3_base"),
    "osim8b":      ("hidden_osim8b",                      "a2b_osim8b"),
}


def partial_corr(x, y, z):
    Z = np.vstack([np.ones_like(z), z]).T
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)


def boot_partial(x, y, z, B=1000):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        try:
            v.append(partial_corr(x[i], y[i], z[i])[0])
        except Exception:
            pass
    return np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)


def xfit_d(X, y, folds):
    """Cross-fitted Cohen's d: direction from the train fold, effect size on the
    held-out fold. Measuring separation along the direction that was fitted to
    maximise it on the same data is circular and always large."""
    ds, oof = [], np.zeros(len(y))
    for tr, te in folds:
        mu = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        u = mu / (np.linalg.norm(mu) + 1e-9)
        p = X[te] @ u
        a, b = p[y[te] == 0], p[y[te] == 1]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
        ds.append((b.mean() - a.mean()) / sd)
        oof[te] = p
    return float(np.mean(ds)), float(roc_auc_score(y, oof))


def run_model(tag, stem, a2b_stem):
    d = np.load(f"{RES}/{stem}.npz", allow_pickle=True)
    H, layers = d["H"], [int(x) for x in d["layers"]]
    arms = [str(a) for a in d["arms"]]
    meta = json.load(open(f"{RES}/{stem}_meta.json"))
    pid = np.array([m["participant_id"] for m in meta])
    i0, i1 = arms.index("cf_neutral"), arms.index("cf_depression")
    n = len(pid)

    dp = None
    p_a2b = f"{A2B}/{a2b_stem}.jsonl" if a2b_stem else None
    if p_a2b and os.path.exists(p_a2b):
        lp = {}
        for line in open(p_a2b):
            r = json.loads(line)
            lp.setdefault(r["arm"], {})[r["participant_id"]] = (
                r["logprobs"]["Yes"] - r["logprobs"]["No"])
        # Require every participant in both arms. A run still in progress leaves
        # a partial file, and silently probing the prefix would compare a
        # complete representation against a truncated behavioural sample.
        have = all(a in lp and set(pid) <= set(lp[a])
                   for a in ("cf_depression", "cf_neutral"))
        if have:
            dp = np.array([sig(lp["cf_depression"][p]) - sig(lp["cf_neutral"][p])
                           for p in pid])
        else:
            print(f"  [{tag}] a2b incomplete -> Q1 only", flush=True)

    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    g = np.concatenate([pid, pid])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(2 * n), y, g))
    # Control task: flip the arm assignment for a random half of participants.
    # Shuffling within participant keeps the group structure and the class
    # balance, so only the concept-label link is destroyed.
    flip = rng.random(n) < 0.5
    y_ctl = np.concatenate([flip.astype(int), 1 - flip.astype(int)])

    rows = []
    for li, L in enumerate(layers):
        A, B = H[:, i0, li, :], H[:, i1, li, :]
        X = np.concatenate([A, B])
        dd, auc = xfit_d(X, y, folds)
        d_ctl, _ = xfit_d(X, y_ctl, folds)

        D = B - A
        nrm = np.linalg.norm(D, axis=1)
        r = dict(layer=L, depth=li / (len(layers) - 1), cohen_d=dd,
                 cohen_d_control=d_ctl, auc=auc,
                 rel_disp=float((nrm / (np.linalg.norm(A, axis=1) + 1e-9)).mean()))
        if dp is not None:
            mu = D.mean(0); u = mu / np.linalg.norm(mu)
            pr = partial_corr(D @ u, dp, nrm)
            lo, hi = boot_partial(D @ u, dp, nrm)
            rv = rng.normal(size=D.shape[1]); rv /= np.linalg.norm(rv)
            r.update(partial_r=float(pr[0]), ci=[float(lo), float(hi)],
                     random_r=float(partial_corr(D @ rv, dp, nrm)[0]))
        rows.append(r)
    return rows, dp is not None


def bar(v, hi, w=18):
    return "#" * max(0, min(w, int(round(w * v / (hi + 1e-12)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=list(MODELS))
    args = ap.parse_args()

    out = {}
    for tag in args.tags:
        stem, a2b_stem = MODELS[tag]
        if not os.path.exists(f"{RES}/{stem}.npz"):
            print(f"[skip {tag}] no hidden states")
            continue
        rows, has_dp = run_model(tag, stem, a2b_stem)
        out[tag] = rows
        dmax = max(r["cohen_d"] for r in rows)

        print("\n" + "=" * 104)
        print(f"{tag}   ({len(rows)} extraction points, "
              f"{'with' if has_dp else 'NO'} behavioural dp)")
        print("=" * 104)
        hdr = (f"{'L':>3s} {'depth':>6s} {'Cohen d':>8s} {'ctl':>6s} "
               f"{'strength':<18s} {'AUC':>6s} {'rel disp':>9s}")
        if has_dp:
            hdr += f" {'partial r':>10s} {'[95% CI]':>16s} {'rand':>6s}"
        print(hdr); print("-" * 104)
        for r in rows:
            line = (f"{r['layer']:3d} {r['depth']:6.2f} {r['cohen_d']:8.2f} "
                    f"{r['cohen_d_control']:6.2f} {bar(r['cohen_d'],dmax):<18s} "
                    f"{r['auc']:6.3f} {r['rel_disp']:9.4f}")
            if has_dp:
                star = "*" if r["ci"][0] > 0 or r["ci"][1] < 0 else " "
                line += (f" {r['partial_r']:9.3f}{star} "
                         f"[{r['ci'][0]:6.3f},{r['ci'][1]:6.3f}] "
                         f"{r['random_r']:6.3f}")
            print(line)

    # ---- cross-model comparison on a common relative-depth axis ------------
    print("\n" + "=" * 104)
    print("CROSS-MODEL -- Q1: is the shortcut written into the patient "
          "representation at all?")
    print("=" * 104)
    print(f"{'model':12s} {'peak d':>7s} {'at depth':>9s} {'mean d':>7s} "
          f"{'max ctl d':>10s} {'frac layers d>1':>16s}")
    print("-" * 104)
    for tag, rows in out.items():
        body = rows[1:]                    # layer 0 is the embedding, d == 0
        pk = max(body, key=lambda r: r["cohen_d"])
        print(f"{tag:12s} {pk['cohen_d']:7.2f} {pk['depth']:9.2f} "
              f"{np.mean([r['cohen_d'] for r in body]):7.2f} "
              f"{max(r['cohen_d_control'] for r in body):10.2f} "
              f"{np.mean([r['cohen_d'] > 1 for r in body]):16.2f}")
    print("\n  ctl d is the same pipeline on shuffled arm labels; it must sit "
          "near 0.\n  depth is layer index / (n_layers - 1), comparable across "
          "models of\n  different depth. The signal is not confined to a band: "
          "it is large at\n  essentially every layer in every model, which is "
          "why the localisation\n  question belongs to Q2, not Q1.")

    if {"menta", "qwen3_base"} <= set(out):
        print("\n" + "=" * 104)
        print("CONTROLLED CONTRAST -- same backbone, clinical LoRA the only "
              "variable")
        print("=" * 104)
        a = max(out["qwen3_base"], key=lambda r: r["cohen_d"])
        b = max(out["menta"], key=lambda r: r["cohen_d"])
        print(f"  qwen3_base (no adaptation): peak d {a['cohen_d']:.2f} "
              f"at depth {a['depth']:.2f}")
        print(f"  menta      (clinical LoRA): peak d {b['cohen_d']:.2f} "
              f"at depth {b['depth']:.2f}")
        print(f"  -> fine-tuning changes peak signal strength by "
              f"{b['cohen_d']-a['cohen_d']:+.2f} "
              f"({b['cohen_d']/max(a['cohen_d'],1e-9):.2f}x)")
        print("  The signal being present in the unadapted base would mean "
              "fine-tuning\n  amplifies a representation the backbone already "
              "carries, rather than\n  creating one.")

    if any("partial_r" in r for rows in out.values() for r in rows):
        print("\n" + "=" * 104)
        print("CROSS-MODEL -- Q2: where does the signal predict behaviour?")
        print("=" * 104)
        print(f"{'model':12s} {'peak |r|':>9s} {'at depth':>9s} {'sig/tot':>9s} "
              f"{'rand>real':>10s} {'mean|r|':>8s} {'mean|rand|':>11s} {'verdict':>10s}")
        print("-" * 104)
        for tag, rows in out.items():
            if "partial_r" not in rows[0]:
                print(f"{tag:12s} {'--':>9s}  (no per-participant dp yet)")
                continue
            # Layer 0 is the embedding: the participant text is byte-identical
            # across arms, so D is exactly 0 there and partial_r is nan. Drop
            # non-finite rows before taking the peak, or max() returns the nan.
            fin = [r for r in rows if np.isfinite(r["partial_r"])]
            pk = max(fin, key=lambda r: abs(r["partial_r"]))
            ns = sum(1 for r in fin if r["ci"][0] > 0 or r["ci"][1] < 0)
            # A significant-layer count on its own over-claims: if dp has any
            # structure a projection can pick up, arbitrary directions correlate
            # with it too and the bootstrap CI is anti-conservative. The count of
            # layers where a RANDOM direction beats the real one is the check --
            # near 50% means the significant layers are an artifact.
            beat = sum(1 for r in fin
                       if abs(r["random_r"]) > abs(r["partial_r"]))
            mr = np.mean([abs(r["partial_r"]) for r in fin])
            mrand = np.mean([abs(r["random_r"]) for r in fin])
            ok = beat < 0.25 * len(fin) and mr > 1.5 * mrand
            print(f"{tag:12s} {pk['partial_r']:+9.3f} {pk['depth']:9.2f} "
                  f"{ns:6d}/{len(fin):<3d} {beat:6d}/{len(fin):<3d} "
                  f"{mr:8.3f} {mrand:11.3f} "
                  f"{'coupled' if ok else 'no/artifact':>10s}")
        print("\n  Under no effect: sig ~ 5% of layers and rand>real ~ 50%.\n"
              "  'coupled' requires a random direction to win in under a quarter "
              "of layers\n  AND the real direction to carry 1.5x the mean "
              "correlation.")

    json.dump(out, open(f"{RES}/layerwise_multi.json", "w"), indent=2)
    print(f"\nwrote {RES}/layerwise_multi.json")


if __name__ == "__main__":
    main()
