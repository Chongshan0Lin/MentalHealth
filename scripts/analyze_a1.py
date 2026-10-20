#!/usr/bin/env python3
"""Experiment A1 analysis -- ablation metrics with the mandated controls.

Reads the run JSONL and reports, per condition: accuracy, balanced accuracy,
macro-F1, AUC, and Shortcut Sufficiency, each with a bootstrap CI over
participants and each against a label-shuffle noise floor.

Two readings are produced for every metric, raw and PMI-normalized
(`log p(y|x) - log p(y|null)`); the gap between them is the label-prior effect.

Headline metric is **balanced accuracy**, not accuracy: F3 records that the
DAIC label split is 133/55, so a constant `No` predictor scores .707 and raw
accuracy is close to uninformative. Balanced accuracy also gives SS a clean
chance point of exactly .5.

    SS = (bal_acc(condition) - .5) / (bal_acc(full) - .5)

SS_S = SS(ellie_only) is the number A1 exists to produce: the interviewer's
turns contain no participant evidence at all, so SS_S >> 0 means the label is
recoverable from the interviewer's side of the conversation.
"""
from __future__ import annotations

import json, sys, collections
import numpy as np

ROOT = "/data/cl195/NeurIPS_workshop"
N_BOOT = 2000
RNG = np.random.default_rng(0)

ORDER = ["full", "full_stripped", "participant_only", "ellie_only",
         "ellie_only_stripped", "ellie_masked", "empty"]


def p_yes(row, pmi):
    """P(Yes) from the two mean token log-probs, optionally PMI-normalized."""
    lp = row["logprobs"]
    v = np.array([lp["Yes"]["mean"], lp["No"]["mean"]], dtype=np.float64)
    if pmi:
        nl = row["null_logprobs"]
        v = v - np.array([nl["Yes"]["mean"], nl["No"]["mean"]])
    e = np.exp(v - v.max())
    return float(e[0] / e.sum())


def metrics(y, p):
    """y: 0/1 gold, p: P(Yes). -> dict of scalar metrics."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p > 0.5).astype(int)
    out = {"acc": float((pred == y).mean())}

    recs = []
    for c in (0, 1):
        m = y == c
        recs.append(float((pred[m] == c).mean()) if m.any() else np.nan)
    out["bal_acc"] = float(np.nanmean(recs))
    out["recall_0"], out["recall_1"] = recs

    f1s = []
    for c in (0, 1):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    out["macro_f1"] = float(np.mean(f1s))

    pos, neg = p[y == 1], p[y == 0]
    if len(pos) and len(neg):
        # Mann-Whitney AUC with ties at 0.5
        out["auc"] = float((( pos[:, None] > neg[None, :]).sum()
                            + 0.5 * (pos[:, None] == neg[None, :]).sum())
                           / (len(pos) * len(neg)))
    else:
        out["auc"] = float("nan")
    out["pred_pos_rate"] = float(pred.mean())
    return out


def boot_ci(y, p, fn, n=N_BOOT):
    """Percentile CI by resampling participants."""
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) < 2:      # degenerate resample
            continue
        vals.append(fn(y[i], p[i]))
    v = np.array([x for x in vals if np.isfinite(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) \
        if len(v) else (float("nan"), float("nan"))


def shuffle_floor(y, p, fn, n=N_BOOT):
    """Noise floor: the metric under permuted labels. -> (mean, p95)."""
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        vals.append(fn(RNG.permutation(y), p))
    v = np.array([x for x in vals if np.isfinite(x)])
    return float(v.mean()), float(np.percentile(v, 95))


def load(path):
    by_cond = collections.defaultdict(dict)
    for l in open(path):
        d = json.loads(l)
        by_cond[d["condition"]][d["uid"]] = d
    return by_cond


def table(by_cond, uids, pmi, title):
    print(f"\n{'='*104}\n{title}\n{'='*104}")
    gold = [by_cond["full"][u]["phq8_binary"] for u in uids]
    maj = max(np.mean(gold), 1 - np.mean(gold))
    print(f"n={len(uids)}  positives={int(sum(gold))}  "
          f"majority-class acc={maj:.3f}  chance bal_acc=0.500")

    ref = None
    print(f"\n{'condition':22s} {'acc':>6s} {'bal_acc':>8s} "
          f"{'[95% CI]':>16s} {'floor95':>8s} {'macroF1':>8s} {'AUC':>6s} "
          f"{'P(pred=Yes)':>12s} {'SS':>7s}")
    print("-" * 104)
    rows_out = {}
    for c in ORDER:
        if c not in by_cond:
            continue
        rs = [by_cond[c][u] for u in uids]
        y = [r["phq8_binary"] for r in rs]
        p = [p_yes(r, pmi) for r in rs]
        m = metrics(y, p)
        lo, hi = boot_ci(y, p, lambda a, b: metrics(a, b)["bal_acc"])
        _, f95 = shuffle_floor(y, p, lambda a, b: metrics(a, b)["bal_acc"])
        if c == "full":
            ref = m["bal_acc"]
        ss = (m["bal_acc"] - .5) / (ref - .5) if ref and abs(ref - .5) > 1e-9 else float("nan")
        m.update(bal_ci=(lo, hi), floor95=f95, ss=ss)
        rows_out[c] = m
        star = "*" if lo > f95 else " "
        print(f"{c:22s} {m['acc']:6.3f} {m['bal_acc']:8.3f}{star}"
              f" [{lo:.3f}, {hi:.3f}] {f95:8.3f} {m['macro_f1']:8.3f} "
              f"{m['auc']:6.3f} {m['pred_pos_rate']:12.3f} {ss:7.3f}")
    print("-" * 104)
    print("* = bootstrap lower bound exceeds the label-shuffle 95th-percentile floor")
    return rows_out


def auc_table(by_cond, uids, title, restrict_ids=None):
    """AUC with bootstrap CI and label-shuffle floor.

    Necessary because both A1 models turn out to predict a near-constant label
    in most conditions (P(pred=Yes) is 0.016 for MentaLLaMA and 1.000 for Menta
    under several ablations). Accuracy, balanced accuracy and any SS built on
    them then describe the model's operating point rather than how much of the
    label is recoverable from the input, and a degenerate condition scores an
    automatic .500 bal_acc that is indistinguishable from genuine chance.

    AUC is threshold-free, so it separates "no information" from "information
    the model refuses to act on at this threshold". It is also invariant to PMI
    normalization -- the null term is a per-label constant, hence a monotone
    shift of the score -- which is why the raw and PMI tables above print
    identical AUCs. That identity is a free correctness check on the scorer.
    """
    if restrict_ids is not None:
        uids = [u for u in uids if by_cond["full"][u]["has_prompt_ids"] == restrict_ids]
    print(f"\n{'='*104}\n{title}  (n={len(uids)})\n{'='*104}")
    print(f"{'condition':22s} {'AUC':>6s} {'[95% CI]':>16s} {'floor95':>8s} "
          f"{'SS_auc':>8s}")
    print("-" * 104)
    fn = lambda a, b: metrics(a, b)["auc"]
    ref = None
    out = {}
    for c in ORDER:
        if c not in by_cond:
            continue
        rs = [by_cond[c][u] for u in uids]
        y = [r["phq8_binary"] for r in rs]
        p = [p_yes(r, False) for r in rs]
        a = metrics(y, p)["auc"]
        lo, hi = boot_ci(y, p, fn)
        _, f95 = shuffle_floor(y, p, fn)
        if c == "full":
            ref = a
        ss = (a - .5) / (ref - .5) if ref and abs(ref - .5) > 1e-9 else float("nan")
        out[c] = {"auc": a, "ci": (lo, hi), "floor95": f95, "ss": ss}
        star = "*" if lo > f95 else " "
        print(f"{c:22s} {a:6.3f}{star} [{lo:.3f}, {hi:.3f}] {f95:8.3f} {ss:8.3f}")
    print("-" * 104)
    print("* = bootstrap lower bound exceeds the label-shuffle 95th-percentile floor")
    return out


def paired_strip_test(by_cond, uids, cond, n=N_BOOT):
    """F5: paired bootstrap on AUC(unstripped) - AUC(stripped).

    Restricted to the transcripts that actually carry prompt IDs, and paired on
    participant, because the two variants differ only by the annotation
    artifact. A CI excluding 0 means the ellie-side signal is the artifact.
    """
    sub = [u for u in uids if by_cond["full"][u]["has_prompt_ids"]]
    y = np.array([by_cond[cond][u]["phq8_binary"] for u in sub])
    pa = np.array([p_yes(by_cond[cond][u], False) for u in sub])
    pb = np.array([p_yes(by_cond[cond + "_stripped"][u], False) for u in sub])
    obs = metrics(y, pa)["auc"] - metrics(y, pb)["auc"]
    diffs = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) < 2:
            continue
        d = metrics(y[i], pa[i])["auc"] - metrics(y[i], pb[i])["auc"]
        if np.isfinite(d):
            diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  {cond:20s} AUC {metrics(y, pa)['auc']:.3f} (with IDs) - "
          f"{metrics(y, pb)['auc']:.3f} (stripped) = {obs:+.3f}  "
          f"[{lo:+.3f}, {hi:+.3f}]  {'ARTIFACT-DRIVEN' if lo > 0 else 'not significant'}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 \
        else f"{ROOT}/runs/a1_MentaLLaMA-chat-7B.jsonl"
    by_cond = load(path)
    uids = sorted(by_cond["full"])
    print(f"loaded {path}")
    print(f"conditions: {sorted(by_cond)}")

    # ---- coverage, per D1: reported, never silently equalised -------------
    print(f"\n{'='*104}\nCOVERAGE (conditions cover unequal shares of the interview -- "
          f"carry as covariate)\n{'='*104}")
    print(f"{'condition':22s} {'median tok':>11s} {'median words':>13s} "
          f"{'median turns':>13s} {'truncated':>10s}")
    for c in ORDER:
        if c not in by_cond:
            continue
        rs = [by_cond[c][u] for u in uids]
        print(f"{c:22s} {np.median([r['n_tok'] for r in rs]):11.0f} "
              f"{np.median([r['n_words'] for r in rs]):13.0f} "
              f"{np.median([r['n_turns'] for r in rs]):13.0f} "
              f"{sum(r['truncated'] for r in rs):7d}/{len(rs)}")

    raw = table(by_cond, uids, False, "A1 -- RAW label log-likelihood")
    pmi = table(by_cond, uids, True, "A1 -- PMI-normalized  (raw vs PMI gap = label-prior effect)")

    print(f"\n{'='*104}\nPRIOR GAP  (bal_acc PMI - bal_acc raw)\n{'='*104}")
    for c in ORDER:
        if c in raw:
            print(f"  {c:22s} raw {raw[c]['bal_acc']:.3f}  ->  "
                  f"PMI {pmi[c]['bal_acc']:.3f}   Δ={pmi[c]['bal_acc']-raw[c]['bal_acc']:+.3f}")

    auc_table(by_cond, uids, "A1 -- AUC (threshold-free; the informative readout)")
    auc_table(by_cond, uids, "A1 -- AUC, transcripts WITH prompt IDs", restrict_ids=True)
    auc_table(by_cond, uids, "A1 -- AUC, transcripts WITHOUT prompt IDs", restrict_ids=False)

    print(f"\n{'='*104}\nF5 -- paired bootstrap: is the ellie-side signal the "
          f"annotation artifact?\n{'='*104}")
    paired_strip_test(by_cond, uids, "ellie_only")
    paired_strip_test(by_cond, uids, "full")

    # ---- F5: does the prompt-ID artifact inflate SS_S? -------------------
    with_ids = [u for u in uids if by_cond["full"][u]["has_prompt_ids"]]
    print(f"\n{'='*104}\nF5 -- PROMPT-ID STRIP, restricted to the {len(with_ids)} "
          f"transcripts that carry IDs\n{'='*104}")
    for c in ["full", "ellie_only"]:
        for variant in [c, c + "_stripped"]:
            rs = [by_cond[variant][u] for u in with_ids]
            y = [r["phq8_binary"] for r in rs]
            p = [p_yes(r, False) for r in rs]
            m = metrics(y, p)
            lo, hi = boot_ci(y, p, lambda a, b: metrics(a, b)["bal_acc"])
            print(f"  {variant:22s} bal_acc={m['bal_acc']:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"acc={m['acc']:.3f}  AUC={m['auc']:.3f}")
        print()

    # ---- F6: format is confounded with the label ------------------------
    print(f"{'='*104}\nF6 -- stratified by prompt-ID presence (format confound)\n{'='*104}")
    for name, sub in [("has IDs", with_ids),
                      ("no IDs", [u for u in uids if u not in set(with_ids)])]:
        g = [by_cond["full"][u]["phq8_binary"] for u in sub]
        print(f"\n  {name}: n={len(sub)}  depression rate={np.mean(g):.3f}")
        for c in ["full", "participant_only", "ellie_only", "ellie_masked"]:
            rs = [by_cond[c][u] for u in sub]
            y = [r["phq8_binary"] for r in rs]
            p = [p_yes(r, False) for r in rs]
            m = metrics(y, p)
            print(f"    {c:22s} bal_acc={m['bal_acc']:.3f}  acc={m['acc']:.3f}  "
                  f"AUC={m['auc']:.3f}")

    # ---- split breakdown -------------------------------------------------
    print(f"\n{'='*104}\nBY SPLIT (inference-only; no split was trained on here)\n{'='*104}")
    for sp in ["train", "dev", "test"]:
        sub = [u for u in uids if by_cond["full"][u]["split"] == sp]
        if not sub:
            continue
        print(f"\n  {sp} (n={len(sub)})")
        for c in ["full", "participant_only", "ellie_only", "ellie_masked", "empty"]:
            rs = [by_cond[c][u] for u in sub]
            y = [r["phq8_binary"] for r in rs]
            p = [p_yes(r, False) for r in rs]
            m = metrics(y, p)
            print(f"    {c:22s} bal_acc={m['bal_acc']:.3f}  acc={m['acc']:.3f}  "
                  f"AUC={m['auc']:.3f}")


if __name__ == "__main__":
    main()
