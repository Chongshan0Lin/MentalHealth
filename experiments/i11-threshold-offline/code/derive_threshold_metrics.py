#!/usr/bin/env python3
"""I11 -- two offline derivations for the threshold-correction arm. No GPU.

Both close gaps identified in the 2026-09-05 verification audit
(`to_human/VERIFICATION_gaps.md`, items 3 and 5).

S1 -- CSS under threshold correction, derived from stored scores.
    `run_full_suite.py:189` implements the intervention as a post-hoc shift in
    LOG-ODDS space:

        S_thresh = S_full - (co[0] + co[1] * feat)

    with `feat` a regex clinical-word density over the interviewer's turns
    (`run_full_suite.py:179-183`) and `co` stored per model as
    `threshold_fit.coef`. Nothing about it touches the model, so applying the
    same rule to the two counterfactual arms needs no inference: the arms'
    per-participant log-odds are already stored in `scores.cf_{depression,
    neutral}`, and each arm's interviewer text is the FIXED pool from
    `run_a2b.POOLS`, so its `feat` is one constant per arm.

    This is the one intervention whose CSS can be recovered without re-running
    the model, and it was recorded as `None`.

S2 -- test-split-only AUC.
    `run_full_suite.py:186-189` fits `co` on train+dev (n=141) but then applies
    the correction to ALL 188 rows, and `interventions.threshold_proxy.auc`
    scores all 188. 141 of those 188 were used to fit the correction being
    evaluated, so the stored number is in-sample contaminated. This recomputes
    every arm on the 47 held-out test participants only.

Menta and Qwen3-4B-base are excluded throughout: they were run before the
threshold arm existed and carry neither `scores` nor `threshold_fit`.
"""
from __future__ import annotations

import importlib.util, json, os, re, sys
import numpy as np
from scipy.stats import rankdata

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
RES = f"{ROOT}/experiments/i10-model-sweep/results"
OUT = f"{ROOT}/experiments/i11-threshold-offline/results"

MODELS = [("mentallama_suite", "MentaLLaMA"), ("osim8b", "OSim-8B"),
          ("omnisapiens_sft", "OmniSap-SFT"), ("omnisapiens_rl", "OmniSap-RL"),
          ("omnisapiens_20", "OmniSap-2.0"), ("humanomniv2", "HumanOmniV2")]

# Verbatim from run_full_suite.py:179-181. Do not "improve" it -- the point is
# to reproduce the fitted intervention exactly, not a better version of it.
CLIN = re.compile(r"\b(diagnos\w*|therap\w*|depress\w*|p_t_s_d|ptsd|sleep|"
                  r"medicat\w*|counsel\w*|symptom\w*|feel\w*)\b", re.I)

sig = lambda x: 1 / (1 + np.exp(-x))


def auc(y, s):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0 or np.allclose(s, s[0]):
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(y, s, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        v.append(auc(y[i], np.asarray(s)[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v \
        else (float("nan"), float("nan"))


def clin_density(text: str) -> float:
    """feat, exactly as run_full_suite.py:182-183 computes it."""
    return len(CLIN.findall(text or "")) / max(1, len((text or "").split()))


def pool_ellie_only(prompts) -> str:
    """The `text_ellie_only` analogue for a counterfactual arm.

    `interleave` (run_a2b.py:68) emits every pool prompt as its own
    `Ellie: <prompt>` line for every participant -- prompts that do not fit the
    even-interval positions are appended -- so the arm's interviewer text is the
    same string for all 188 participants, and `feat` is a constant.
    """
    return "\n".join(f"Ellie: {p}" for p in prompts)


def main():
    rows = [json.loads(l) for l in open(COND)]
    y = np.array([r["phq8_binary"] for r in rows])
    split = np.array([r["split"] for r in rows])
    te = split == "test"
    feat_real = np.array([clin_density(r["text_ellie_only"]) for r in rows])

    spec = importlib.util.spec_from_file_location(
        "a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
    a2b = importlib.util.module_from_spec(spec); sys.modules["a2b"] = a2b
    spec.loader.exec_module(a2b)
    feat_arm = {a: clin_density(pool_ellie_only(p)) for a, p in a2b.POOLS.items()}

    print(f"n = {len(rows)}   test = {int(te.sum())}   train+dev = {int((~te).sum())}")
    print(f"\nclinical-word density of each fixed interviewer pool (constant "
          f"across participants):")
    for a, f in sorted(feat_arm.items()):
        print(f"  {a:16s} {f:.6f}   ({len(a2b.POOLS[a])} prompts, "
              f"{len(pool_ellie_only(a2b.POOLS[a]).split())} words)")
    print(f"  {'real transcripts':16s} {feat_real.mean():.6f}   "
          f"(mean over 188; range {feat_real.min():.4f}-{feat_real.max():.4f})")

    out = {"n": len(rows), "n_test": int(te.sum()),
           "feat_pool": feat_arm, "feat_real_mean": float(feat_real.mean()),
           "models": {}}

    # ---------------------------------------------------------------- S1
    print(f"\n{'='*104}\nS1 -- CSS under threshold correction, derived offline "
          f"(no inference)\n{'='*104}")
    print(f"{'model':14s} {'co0':>9s} {'co1':>8s} {'shift_dep':>10s} "
          f"{'shift_neu':>10s} | {'CSS base':>9s} {'CSS thr':>9s} {'Δ':>8s} | "
          f"{'signed base':>12s} {'signed thr':>11s}")
    print("-" * 104)
    for tag, name in MODELS:
        d = json.load(open(f"{RES}/suite_{tag}.json"))
        co = d["threshold_fit"]["coef"]
        S = {k: np.array(v, float) for k, v in d["scores"].items()}
        sh = {a: co[0] + co[1] * feat_arm[a] for a in ("cf_depression", "cf_neutral")}
        p_dep_b, p_neu_b = sig(S["cf_depression"]), sig(S["cf_neutral"])
        p_dep_t = sig(S["cf_depression"] - sh["cf_depression"])
        p_neu_t = sig(S["cf_neutral"] - sh["cf_neutral"])
        css_b, css_t = np.abs(p_dep_b - p_neu_b).mean(), np.abs(p_dep_t - p_neu_t).mean()
        sgn_b, sgn_t = (p_dep_b - p_neu_b).mean(), (p_dep_t - p_neu_t).mean()
        fl_b = float(np.mean((p_dep_b > .5) != (p_neu_b > .5)))
        fl_t = float(np.mean((p_dep_t > .5) != (p_neu_t > .5)))
        print(f"{name:14s} {co[0]:9.4f} {co[1]:8.4f} {sh['cf_depression']:10.4f} "
              f"{sh['cf_neutral']:10.4f} | {css_b:9.4f} {css_t:9.4f} "
              f"{css_t-css_b:+8.4f} | {sgn_b:+12.4f} {sgn_t:+11.4f}")
        out["models"][name] = dict(
            coef=co, shift_dep=sh["cf_depression"], shift_neu=sh["cf_neutral"],
            css_base_abs=float(css_b), css_thresh_abs=float(css_t),
            css_base_signed=float(sgn_b), css_thresh_signed=float(sgn_t),
            flips_base=fl_b, flips_thresh=fl_t)

    print("\nCS (construct sensitivity) under threshold correction:")
    print("  The E5 construct pair holds the interviewer's turns FIXED and varies")
    print("  only the participant's content, so `feat` -- a function of the")
    print("  interviewer text alone -- is IDENTICAL for the two members of every")
    print("  pair. The correction subtracts the same constant from both log-odds,")
    print("  so the paired contrast is EXACTLY invariant in log-odds space.")
    print("  In probability space it changes only through the sigmoid's curvature.")
    print("  That residual cannot be computed here: run_full_suite.py keeps the E5")
    print("  arrays in a local `E` and stores only the scalar `cs`, so the")
    print("  per-item E5 scores are not on disk. Numeric CS needs 2 x 188 re-scores")
    print("  per model; the invariance result does not.")

    # ---------------------------------------------------------------- S2
    print(f"\n{'='*104}\nS2 -- AUC on the 47 held-out test participants only\n"
          f"{'='*104}")
    print("`threshold_fit.coef` is fitted on train+dev (n=141) but the stored")
    print("`interventions.threshold_proxy.auc` scores all 188, so 141 of the rows")
    print("scored were used to fit the correction. Test-only is the clean read.\n")
    print(f"{'model':14s} {'full(188)':>10s} {'thr(188)':>9s} {'Δ188':>8s} | "
          f"{'full(47)':>9s} {'thr(47)':>8s} {'Δ47':>8s} {'95% CI on Δ47':>22s}")
    print("-" * 104)
    for tag, name in MODELS:
        d = json.load(open(f"{RES}/suite_{tag}.json"))
        S = {k: np.array(v, float) for k, v in d["scores"].items()}
        thr = S["full"] - (d["threshold_fit"]["coef"][0]
                           + d["threshold_fit"]["coef"][1] * feat_real)
        a_f188, a_t188 = auc(y, S["full"]), auc(y, thr)
        a_f47, a_t47 = auc(y[te], S["full"][te]), auc(y[te], thr[te])
        rng = np.random.default_rng(0)
        dl = []
        for _ in range(2000):
            i = rng.integers(0, int(te.sum()), int(te.sum()))
            yy = y[te][i]
            if len(set(yy)) < 2:
                continue
            dl.append(auc(yy, thr[te][i]) - auc(yy, S["full"][te][i]))
        lo, hi = np.percentile(dl, [2.5, 97.5])
        print(f"{name:14s} {a_f188:10.4f} {a_t188:9.4f} {a_t188-a_f188:+8.4f} | "
              f"{a_f47:9.4f} {a_t47:8.4f} {a_t47-a_f47:+8.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]".rjust(22))
        out["models"][name].update(
            auc_full_188=a_f188, auc_thresh_188=a_t188,
            auc_full_47=a_f47, auc_thresh_47=a_t47,
            d_auc_47_ci=[float(lo), float(hi)])

    # every arm on test only, for context
    print(f"\nEvery stored arm, test split only (n=47):")
    arms = ["full", "participant_only", "ellie_masked", "ellie_only", "cf_none",
            "cf_neutral", "cf_depression", "targeted_removal", "threshold_proxy"]
    print(f"{'model':14s} " + "".join(f"{a[:11]:>12s}" for a in arms))
    for tag, name in MODELS:
        d = json.load(open(f"{RES}/suite_{tag}.json"))
        S = {k: np.array(v, float) for k, v in d["scores"].items()}
        print(f"{name:14s} " + "".join(f"{auc(y[te], S[a][te]):12.4f}" for a in arms))
        out["models"][name]["test_only_auc"] = {
            a: auc(y[te], S[a][te]) for a in arms}

    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/threshold_offline.json", "w"), indent=1)
    print(f"\nwrote {OUT}/threshold_offline.json")


if __name__ == "__main__":
    main()
