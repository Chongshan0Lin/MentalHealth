#!/usr/bin/env python3
"""I10 -- the whole detection + intervention suite, for one model, in one load.

Every headline number in this project has rested on a single model per arm:
MentaLLaMA for the LLM interventions, MentalBERT for the encoder ones. This
runs the full battery on any text model so the table can be read across models
instead of down one column.

The controlled comparison this enables. `Qwen3-4B-Instruct-2507` is exactly the
base of `Menta` (verified: the LoRA's base_model_name_or_path). Running both
isolates one variable -- mental-health domain fine-tuning -- and asks the
question the paper's premise depends on:

    is the shortcut introduced by domain fine-tuning, or already in the base?

Conditions are scored once and reused, so marginalisation, threshold correction
and the conflict set cost nothing extra. Chain-of-thought needs generation and
is opt-in (--cot).
"""
from __future__ import annotations

import argparse, importlib.util, json, os, re, sys, time
import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer, OmniScorer

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
E5 = f"{ROOT}/experiments/suite/results/e5_construct_pairs.jsonl"
OUT = f"{ROOT}/experiments/i10-model-sweep/results"
LABELS = ["Yes", "No"]
BASE = ('Consider this post: "{text}" '
        'Question: Does the poster suffer from depression?')
P1 = ("Base your prediction only on clinically relevant evidence in the "
      "participant's own words. Do not infer depression from the interviewer's "
      "choice of questions, from sentiment, or from writing style. ")
sig = lambda x: 1 / (1 + np.exp(-x))
rng = np.random.default_rng(0)

_s = importlib.util.spec_from_file_location(
    "a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
a2b = importlib.util.module_from_spec(_s); sys.modules["a2b"] = a2b
_s.loader.exec_module(a2b)
POOLS = dict(a2b.POOLS); POOLS["cf_none"] = []; interleave = a2b.interleave
ARMS = ["cf_none", "cf_neutral", "cf_depression", "cf_empirical"]


def auc(y, s):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0 or np.allclose(s, s[0]):
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(y, s, B=1000):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a = auc(y[i], s[i])
        if np.isfinite(a):
            v.append(a)
    return (np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=4032)
    ap.add_argument("--scorer", default="text", choices=["text", "omni"],
                    help="omni = Qwen2.5-Omni thinker (OmniSapiens*, "
                         "HumanOmniV2). media=None makes OmniScorer text-only, "
                         "so the same suite runs on both families.")
    ap.add_argument("--cot", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.scorer == "omni":
        # prefix caching stays OFF: Qwen2.5-Omni uses M-RoPE, so a continuation
        # forwarded against a cache scores label tokens at wrong positions
        # (~11 nat drift). OmniScorer already defaults to the uncached path.
        assert args.adapter is None, "omni path has no adapter support"
        sc = OmniScorer(args.ckpt, device=args.device)
        sc.tok = sc.proc.tokenizer
    else:
        sc = TextScorer(args.ckpt, device=args.device, adapter=args.adapter)
    print(f"{args.tag}: {args.ckpt}" + (f" + {args.adapter}" if args.adapter else ""),
          flush=True)

    rows = [json.loads(l) for l in open(COND)]
    pairs = [json.loads(l) for l in open(E5)]
    y = np.array([r["phq8_binary"] for r in rows])

    longest = max(len(" ".join(p)) for p in POOLS.values())
    budget = args.budget - (longest // 3) - 64
    trunc = lambda t: (sc.tok.decode(sc.tok(t, add_special_tokens=False).input_ids[:budget])
                       if len(sc.tok(t, add_special_tokens=False).input_ids) > budget else t)
    C = [trunc(r["text_participant_only"]) for r in rows]

    def s1(prompt):
        lp = sc.score(prompt, LABELS)
        return lp["Yes"]["mean"] - lp["No"]["mean"]

    t0 = time.time()
    S = {}
    # --- detection: the A1 ladder on real transcripts ---------------------
    for k, fld in [("full", "text_full"), ("participant_only", "text_participant_only"),
                   ("ellie_masked", "text_ellie_masked"), ("ellie_only", "text_ellie_only")]:
        S[k] = np.array([s1(BASE.format(text=trunc(r[fld] or ""))) for r in rows])
        print(f"  {k} done ({time.time()-t0:.0f}s)", flush=True)
    # --- counterfactual arms ----------------------------------------------
    for a in ARMS:
        S[a] = np.array([s1(BASE.format(text=interleave(c, POOLS[a]))) for c in C])
        print(f"  {a} done ({time.time()-t0:.0f}s)", flush=True)
    # --- targeted removal: only the clinical questions, not the channel ----
    import re as _re
    PAT = _re.compile(r"^Ellie: ([A-Za-z][A-Za-z0-9_]*) \((.*)\)\s*$")
    CL = set(json.load(open(f"{ROOT}/experiments/suite/results/"
                            f"e4a_prompt_clusters.json"))["clusters"]["clinical_care"])
    def strip_clinical(t):
        return "\n".join(l for l in (t or "").split("\n")
                          if not (PAT.match(l) and PAT.match(l).group(1) in CL))
    S["targeted_removal"] = np.array(
        [s1(BASE.format(text=trunc(strip_clinical(r["text_full"])))) for r in rows])
    print(f"  targeted_removal done ({time.time()-t0:.0f}s)", flush=True)

    # --- P1 instruction ----------------------------------------------------
    for a in ["cf_none", "cf_neutral", "cf_depression"]:
        S["p1_" + a] = np.array([s1(P1 + BASE.format(text=interleave(c, POOLS[a])))
                                 for c in C])
    print(f"  P1 done ({time.time()-t0:.0f}s)", flush=True)
    # --- E5 construct pairs, three input variants --------------------------
    cond = {r["uid"]: r for r in rows}
    E = {}
    for v in ["base", "d1", "d1m"]:
        pos, neg = [], []
        for p in pairs:
            for side, acc in (("pos", pos), ("neg", neg)):
                u = p[side]["uid"]
                if v == "base":
                    txt = p[side]["text"]
                else:
                    ell = [l for l in (cond[p["context_uid"]]["text_ellie_only"] or ""
                                       ).split("\n") if l.strip()]
                    lines = ["Ellie: [Q]"] * len(ell) if v == "d1m" else []
                    txt = interleave(cond[u]["text_participant_only"], lines)
                acc.append(s1(BASE.format(text=trunc(txt))))
        E[v] = (np.array(pos), np.array(neg))
        print(f"  E5/{v} done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- derived metrics --------------------------------------
    css = lambda a, b: float(np.abs(sig(a) - sig(b)).mean())
    cs = lambda v: float(np.abs(sig(E[v][0]) - sig(E[v][1])).mean())
    conflict = lambda d, n: auc(y, np.where(y == 0, d, n))

    base_css = css(S["cf_depression"], S["cf_neutral"])
    R = {"tag": args.tag, "ckpt": args.ckpt, "adapter": args.adapter}
    R["detection"] = {
        "removal_ellie_only_auc": auc(y, S["ellie_only"]),
        "removal_ci": list(boot_auc(y, S["ellie_only"])),
        "full_auc": auc(y, S["full"]),
        "counterfactual_css": base_css,
        "counterfactual_flips": float(np.mean((S["cf_depression"] > 0) != (S["cf_neutral"] > 0))),
        "conflict_auc": conflict(S["cf_depression"], S["cf_neutral"]),
        "construct_sens": cs("base"),
        "ser": base_css / max(cs("base"), 1e-9),
        "p_yes": float((S["full"] > 0).mean()),
    }
    # threshold correction from an observable property of the interviewer's
    # turns (clinical-word density), fitted on train+dev, applied to test
    CLIN = __import__("re").compile(
        r"\b(diagnos\w*|therap\w*|depress\w*|p_t_s_d|ptsd|sleep|medicat\w*|"
        r"counsel\w*|symptom\w*|feel\w*)\b", __import__("re").I)
    feat = np.array([len(CLIN.findall(r["text_ellie_only"] or "")) /
                     max(1, len((r["text_ellie_only"] or "").split())) for r in rows])
    shift = S["full"] - S["participant_only"]
    sp = np.array([r["split"] for r in rows])
    tr = sp != "test"
    A = np.vstack([np.ones(tr.sum()), feat[tr]]).T
    co, *_ = np.linalg.lstsq(A, shift[tr], rcond=None)
    S["threshold_proxy"] = S["full"] - (co[0] + co[1] * feat)
    R_thresh = dict(coef=[float(co[0]), float(co[1])],
                    corr_feature_shift=float(np.corrcoef(feat[tr], shift[tr])[0, 1]))

    m2 = (S["cf_neutral"] + S["cf_depression"]) / 2
    m4 = sum(S[a] for a in ARMS) / 4
    R["interventions"] = {
        "none":              dict(auc=auc(y, S["full"]), css=base_css, cs=cs("base")),
        "delete_interviewer":dict(auc=auc(y, S["participant_only"]), css=0.0, cs=cs("d1")),
        "blank_words":       dict(auc=auc(y, S["ellie_masked"]), css=0.0, cs=cs("d1m")),
        "marginalise_2":     dict(auc=auc(y, m2), css=0.0, cs=None),
        "marginalise_4":     dict(auc=auc(y, m4), css=0.0, cs=None),
        "targeted_removal":  dict(auc=auc(y, S["targeted_removal"]), css=None, cs=None),
        "threshold_proxy":   dict(auc=auc(y, S["threshold_proxy"]), css=None, cs=None),
        "instruct_ignore":   dict(auc=auc(y, S["p1_cf_none"]),
                                  css=css(S["p1_cf_depression"], S["p1_cf_neutral"]),
                                  cs=None),
    }
    R["threshold_fit"] = R_thresh
    R["scores"] = {k: v.tolist() for k, v in S.items()}
    json.dump(R, open(f"{OUT}/suite_{args.tag}.json", "w"), indent=2)

    d = R["detection"]
    print(f"\n{'='*74}\n{args.tag}\n{'='*74}")
    print("DETECTION")
    print(f"  removal (interviewer only)  AUC {d['removal_ellie_only_auc']:.3f} "
          f"[{d['removal_ci'][0]:.3f}, {d['removal_ci'][1]:.3f}]")
    print(f"  counterfactual CSS          {d['counterfactual_css']:.4f}   "
          f"flips {d['counterfactual_flips']:.3f}")
    print(f"  construct sensitivity       {d['construct_sens']:.4f}   SER {d['ser']:.2f}")
    print(f"  IID AUC {d['full_auc']:.3f}   conflict AUC {d['conflict_auc']:.3f}   "
          f"P(Yes) {d['p_yes']:.3f}")
    print("\nINTERVENTIONS")
    print(f"  {'method':22s} {'AUC':>7s} {'dAUC':>7s} {'CSS':>8s} {'resp to patient':>16s}")
    b = R["interventions"]["none"]
    for k, v in R["interventions"].items():
        cstr = f"{v['cs']:16.4f}" if v["cs"] is not None else f"{'--':>16s}"
        xstr = f"{v['css']:8.4f}" if v["css"] is not None else f"{'--':>8s}"
        print(f"  {k:22s} {v['auc']:7.3f} {v['auc']-b['auc']:+7.3f} {xstr}{cstr}")
    print(f"\nwrote {OUT}/suite_{args.tag}.json   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
