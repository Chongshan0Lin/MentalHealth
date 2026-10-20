#!/usr/bin/env python3
"""I1 -- LLM-arm inference-time interventions, with split discipline.

Step 1 of INTERVENTION_DESIGN.md §7. Re-scores logprobs already on disk; no
GPU, no model load. Nothing here is a new measurement of the model -- it is a
new *decision rule* applied to existing measurements.

The §0.5 ceiling numbers pooled all 188 participants and picked the threshold
against the labels being scored. That bounds what a threshold correction can
buy; it is not a method. Everything below fits on dev (n=34) and reports on
test (n=47), which is what makes these methods rather than ceilings.

Methods
  P0     no intervention                     -- A1 `full`
  D1     delete interviewer turns            -- A1 `participant_only`
  D1m    mask interviewer words, keep turns  -- A1 `ellie_masked`
  P3-a   oracle threshold (ceiling only)     -- fit on test, reported as a bound
  P3-b   dev-fit per-arm shift               -- delta estimated on dev
  P3-c   two-pass label-free correction      -- reduces to the S-free score

Scoring convention follows EXPERIMENT_DESIGN.md §0: the decision variable is
the label log-odds  l = log p(Yes) - log p(No), and p = sigmoid(l). AUC is
computed on l, which is monotone in p, so the two agree.
"""
from __future__ import annotations

import json
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as _sk_auc


def roc_auc_score(y, s):
    """Mann-Whitney AUC with average ranks for ties.

    sklearn's version costs ~1 ms once validation is counted, and the bootstrap
    below calls this ~70k times. Verified against sklearn in __main__.
    """
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

ROOT = "/data/cl195/NeurIPS_workshop"
A1 = f"{ROOT}/runs/a1_MentaLLaMA-chat-7B.jsonl"
A2B = f"{ROOT}/experiments/a2-ellie-counterfactual/results/a2b_MentaLLaMA-chat-7B.jsonl"
OUT = f"{ROOT}/experiments/i1-llm-inference-interventions/results"
B = 2000
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- loading

def load(path, keyfield):
    """-> {key: {split: (pid, logodds, y)}}. A1 stores logprobs as dicts with
    a 'mean' field, A2b as bare floats; both are handled."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        lp = r["logprobs"]
        g = lambda k: lp[k]["mean"] if isinstance(lp[k], dict) else lp[k]
        out.setdefault(r[keyfield], {}).setdefault(r["split"], []).append(
            (r["participant_id"], g("Yes") - g("No"), r["phq8_binary"]))
    return {k: {s: (np.array([a for a, _, _ in v]),
                    np.array([b for _, b, _ in v]),
                    np.array([c for _, _, c in v]))
                for s, v in d.items()} for k, d in out.items()}


def pool(d, splits):
    """Concatenate several splits of one condition."""
    p = np.concatenate([d[s][0] for s in splits])
    l = np.concatenate([d[s][1] for s in splits])
    y = np.concatenate([d[s][2] for s in splits])
    return p, l, y


# ---------------------------------------------------------------- metrics

def best_threshold(l, y):
    """Accuracy-maximising threshold. Ties broken toward the midpoint of the
    score range so the choice does not depend on sort order."""
    cand = np.unique(np.concatenate([l, [l.min() - 1, l.max() + 1]]))
    accs = np.array([((l > t).astype(int) == y).mean() for t in cand])
    best = cand[accs == accs.max()]
    return float(best[np.argmin(np.abs(best - np.median(l)))])


def acc_at(l, y, t):
    return float(((l > t).astype(int) == y).mean())


def boot_ci(fn, *arrs, need_both=True):
    n = len(arrs[0])
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if need_both and len(np.unique(arrs[-1][i])) < 2:
            continue
        vals.append(fn(*[a[i] for a in arrs]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def shuffle_floor(l, y, fn):
    """95th percentile of the metric under label permutation."""
    vals = []
    for _ in range(B):
        vals.append(fn(l, rng.permutation(y)))
    return float(np.percentile(vals, 95))


def report(name, l_dev, y_dev, l_test, y_test):
    """Fit the operating point on dev, report everything on test."""
    t = best_threshold(l_dev, y_dev)
    auc = roc_auc_score(y_test, l_test)
    lo, hi = boot_ci(lambda a, b: roc_auc_score(b, a), l_test, y_test)
    acc = acc_at(l_test, y_test, t)
    alo, ahi = boot_ci(lambda a, b: acc_at(a, b, t), l_test, y_test,
                       need_both=False)
    floor = shuffle_floor(l_test, y_test, lambda a, b: roc_auc_score(b, a))
    return dict(method=name, n=len(y_test), thr_dev=t,
                auc=auc, auc_lo=lo, auc_hi=hi, auc_floor=floor,
                clears_floor=bool(lo > floor),
                acc=acc, acc_lo=alo, acc_hi=ahi,
                p_yes=float((l_test > t).mean()))


def fmt(r):
    star = "*" if r["clears_floor"] else " "
    return (f"{r['method']:34s} {r['auc']:.3f}{star} "
            f"[{r['auc_lo']:.3f},{r['auc_hi']:.3f}] "
            f"{r['acc']:.3f} [{r['acc_lo']:.3f},{r['acc_hi']:.3f}] "
            f"{r['p_yes']:.3f}  {r['auc_floor']:.3f}")


HDR = (f"{'method':34s} {'AUC':>6s} {'[95% CI]':>16s} "
       f"{'acc':>5s} {'[95% CI]':>15s} {'P(Y)':>5s}  {'floor':>5s}")


# ---------------------------------------------------------------- main

def _selfcheck():
    """The fast AUC must agree with sklearn, including on ties."""
    g = np.random.default_rng(7)
    for _ in range(200):
        n = g.integers(10, 80)
        y = g.integers(0, 2, n)
        if len(np.unique(y)) < 2:
            continue
        s = g.normal(size=n).round(1)          # rounding forces ties
        assert abs(roc_auc_score(y, s) - _sk_auc(y, s)) < 1e-9
    print("selfcheck: fast AUC == sklearn (200 cases, ties included)")


def main():
    _selfcheck()
    a1 = load(A1, "condition")
    a2 = load(A2B, "arm")
    rows = []

    print("=" * 100)
    print("I1 -- LLM inference-time interventions | MentaLLaMA-chat-7B")
    print("threshold fitted on DEV (n=34), all numbers reported on TEST (n=47)")
    print("=" * 100)
    maj = a1["full"]["test"][2].mean()
    maj = max(maj, 1 - maj)
    print(f"test majority-class accuracy = {maj:.3f} | "
          f"* = AUC lower bound clears the label-shuffle floor\n")

    # ---- A. input-level interventions, real transcripts -----------------
    print("A. INPUT-LEVEL INTERVENTIONS (real transcripts, A1 conditions)")
    print(HDR)
    for cond, name in [("full", "P0  no intervention"),
                       ("participant_only", "D1  delete interviewer turns"),
                       ("ellie_masked", "D1m mask interviewer words"),
                       ("ellie_only", "--  shortcut only (reference)")]:
        d = a1[cond]
        r = report(name, d["dev"][1], d["dev"][2], d["test"][1], d["test"][2])
        r["family"] = "input"
        rows.append(r)
        print(fmt(r))

    # ---- B. counterfactual arms, uncorrected ----------------------------
    print("\nB. COUNTERFACTUAL ARMS, NO CORRECTION (P0 under adversarial context)")
    print(HDR)
    for arm in ["cf_neutral", "cf_none", "cf_empirical", "cf_depression"]:
        d = a2[arm]
        r = report(f"P0  {arm}", d["dev"][1], d["dev"][2],
                   d["test"][1], d["test"][2])
        r["family"] = "cf_uncorrected"
        rows.append(r)
        print(fmt(r))

    # ---- C. threshold corrections ---------------------------------------
    # P3-b: delta estimated on dev as the mean shift from the S-free anchor,
    # then subtracted at test. P3-c: per-patient two-pass correction, which is
    # algebraically the S-free score itself -- stated explicitly rather than
    # reimplemented, so the identity is auditable.
    print("\nC. THRESHOLD CORRECTIONS (fit on dev, applied to test)")
    print(HDR)
    ref = a2["cf_none"]
    for arm in ["cf_neutral", "cf_empirical", "cf_depression"]:
        d = a2[arm]
        # dev participants are aligned across arms (same file, same order)
        assert (d["dev"][0] == ref["dev"][0]).all()
        assert (d["test"][0] == ref["test"][0]).all()
        delta = float(np.mean(d["dev"][1] - ref["dev"][1]))
        lb = d["test"][1] - delta
        r = report(f"P3-b {arm} (d={delta:+.3f})",
                   d["dev"][1] - delta, d["dev"][2], lb, d["test"][2])
        r["family"] = "P3b"
        r["delta_dev"] = delta
        rows.append(r)
        print(fmt(r))

    for arm in ["cf_neutral", "cf_empirical", "cf_depression"]:
        d = a2[arm]
        lc = ref["test"][1]          # two-pass correction == S-free score
        r = report(f"P3-c {arm}", ref["dev"][1], ref["dev"][2], lc, d["test"][2])
        r["family"] = "P3c"
        rows.append(r)
        print(fmt(r))

    # ---- D. E2b constructed-conflict set --------------------------------
    # Each test participant appears once, under the arm that points AWAY from
    # their label: negatives get the clinical pool, positives the biographical.
    print("\nD. E2b CONSTRUCTED-CONFLICT SET (interviewer context vs label)")
    print(HDR)
    yt = ref["test"][2]
    neg, pos = yt == 0, yt == 1
    print(f"  cells: gold=No + cf_depression n={neg.sum()} | "
          f"gold=Yes + cf_neutral n={pos.sum()}")

    def conflict(scores_by_arm, dev_l, dev_y):
        l = np.where(neg, scores_by_arm["cf_depression"],
                     scores_by_arm["cf_neutral"])
        return report("", dev_l, dev_y, l, yt)

    raw = {a: a2[a]["test"][1] for a in ["cf_depression", "cf_neutral"]}
    devraw = np.where(ref["dev"][2] == 0, a2["cf_depression"]["dev"][1],
                      a2["cf_neutral"]["dev"][1])
    r = conflict(raw, devraw, ref["dev"][2]); r["method"] = "P0  on E2b"
    r["family"] = "E2b"; rows.append(r); print(fmt(r))

    corr = {}
    for a in ["cf_depression", "cf_neutral"]:
        dl = float(np.mean(a2[a]["dev"][1] - ref["dev"][1]))
        corr[a] = a2[a]["test"][1] - dl
    devcorr = np.where(
        ref["dev"][2] == 0,
        a2["cf_depression"]["dev"][1] - float(np.mean(a2["cf_depression"]["dev"][1] - ref["dev"][1])),
        a2["cf_neutral"]["dev"][1] - float(np.mean(a2["cf_neutral"]["dev"][1] - ref["dev"][1])))
    r = conflict(corr, devcorr, ref["dev"][2]); r["method"] = "P3-b on E2b"
    r["family"] = "E2b"; rows.append(r); print(fmt(r))

    r = report("P3-c on E2b", ref["dev"][1], ref["dev"][2], ref["test"][1], yt)
    r["family"] = "E2b"; rows.append(r); print(fmt(r))

    # ---- E. CF-Sensitivity ----------------------------------------------
    print("\nE. CF-SENSITIVITY  (mean |p(Yes|C,S1) - p(Yes|C,S2)|, test split)")
    sig = lambda x: 1 / (1 + np.exp(-x))
    d_dep, d_neu = a2["cf_depression"]["test"][1], a2["cf_neutral"]["test"][1]
    dl_dep = float(np.mean(a2["cf_depression"]["dev"][1] - ref["dev"][1]))
    dl_neu = float(np.mean(a2["cf_neutral"]["dev"][1] - ref["dev"][1]))
    cfs = {
        "P0  uncorrected": np.abs(sig(d_dep) - sig(d_neu)),
        "P3-b dev-fit shift": np.abs(sig(d_dep - dl_dep) - sig(d_neu - dl_neu)),
        "P3-c two-pass": np.abs(sig(ref["test"][1]) - sig(ref["test"][1])),
    }
    for k, v in cfs.items():
        lo, hi = boot_ci(lambda a: float(a.mean()), v, need_both=False)
        print(f"  {k:22s} {v.mean():.4f}  [{lo:.4f}, {hi:.4f}]")
        rows.append(dict(method=k, family="cf_sens", cf_sens=float(v.mean()),
                         cf_lo=lo, cf_hi=hi))

    # ---- F. leave-one-out cross-fit over all 188 -------------------------
    # The test split (n=47) cannot separate these methods: every CI in section
    # A spans ~0.3 AUC and none clears the shuffle floor. That is a power
    # limit, not a null.
    #
    # The model is zero-shot -- it never trained on any DAIC split -- so the
    # ONLY quantity being fitted anywhere in this file is a scalar threshold.
    # Cross-fitting that scalar leave-one-out lets every one of the 188
    # participants be evaluated with a threshold that never saw them, which is
    # honest and 4x better powered than the 47-participant split.
    print("\nF. LEAVE-ONE-OUT CROSS-FIT THRESHOLD, all 188 participants")
    print("   (threshold for participant i fitted on the other 187; no oracle)")
    print(HDR)

    def loo(l, y, name):
        pred = np.empty(len(y), dtype=int)
        for i in range(len(y)):
            m = np.ones(len(y), bool); m[i] = False
            pred[i] = int(l[i] > best_threshold(l[m], y[m]))
        auc = roc_auc_score(y, l)
        alo, ahi = boot_ci(lambda a, b: float((a == b).mean()), pred, y,
                           need_both=False)
        lo, hi = boot_ci(lambda a, b: roc_auc_score(b, a), l, y)
        floor = shuffle_floor(l, y, lambda a, b: roc_auc_score(b, a))
        return dict(method=name, family="loo", n=len(y), auc=auc,
                    auc_lo=lo, auc_hi=hi, auc_floor=floor,
                    clears_floor=bool(lo > floor),
                    acc=float((pred == y).mean()), acc_lo=alo, acc_hi=ahi,
                    p_yes=float(pred.mean()))

    for cond, name in [("full", "P0  no intervention"),
                       ("participant_only", "D1  delete interviewer turns"),
                       ("ellie_masked", "D1m mask interviewer words"),
                       ("ellie_only", "--  shortcut only (reference)")]:
        p, l, y = pool(a1[cond], ["train", "dev", "test"])
        r = loo(l, y, name); rows.append(r); print(fmt(r))

    # paired D1-minus-P0 AUC difference, same participants
    _, lf, yf = pool(a1["full"], ["train", "dev", "test"])
    _, lp, yp = pool(a1["participant_only"], ["train", "dev", "test"])
    assert (yf == yp).all()
    diffs = []
    for _ in range(B):
        i = rng.integers(0, len(yf), len(yf))
        if len(np.unique(yf[i])) < 2:
            continue
        diffs.append(roc_auc_score(yf[i], lp[i]) - roc_auc_score(yf[i], lf[i]))
    diffs = np.array(diffs)
    print(f"\n  paired D1 - P0 AUC difference: {diffs.mean():+.3f} "
          f"[{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]"
          f"   excludes zero: {'YES' if np.percentile(diffs, 2.5) > 0 else 'NO'}")
    rows.append(dict(method="D1 - P0 paired AUC diff", family="loo_paired",
                     delta=float(diffs.mean()),
                     lo=float(np.percentile(diffs, 2.5)),
                     hi=float(np.percentile(diffs, 97.5))))

    # E2b over all 188, LOO-thresholded
    print("\n  E2b conflict set, all 188:")
    yA = a2["cf_none"]["train"][2]
    pA, lA, yA = pool(a2["cf_none"], ["train", "dev", "test"])
    _, ld, _ = pool(a2["cf_depression"], ["train", "dev", "test"])
    _, ln, _ = pool(a2["cf_neutral"], ["train", "dev", "test"])
    l_conf = np.where(yA == 0, ld, ln)
    for l, nm in [(l_conf, "P0  on E2b"), (lA, "P3-c on E2b")]:
        r = loo(l, yA, nm); rows.append(r); print(fmt(r))

    with open(f"{OUT}/i1_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT}/i1_summary.json  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
