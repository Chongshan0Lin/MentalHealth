#!/usr/bin/env python3
"""I12 -- CSS for the targeted-removal arm, the variant that is informative.

`run_full_suite.py:201` records `css=None` for `targeted_removal`: the arm was
scored once, on the real transcript with the `clinical_care` prompt cluster
stripped, so there is no counterfactual pair and no CSS.

Two ways to build one, and they answer different questions.

  (a) Strip the clinical questions from the counterfactual POOLS and re-score.
      Near-tautological: the cf_depression pool IS clinical questions, so
      stripping it collapses the two arms into each other and CSS falls toward 0
      for the same reason the delete/marginalise arms do. Not run.

  (b) Leave the pools intact and strip clinical questions from the REAL
      transcript, then measure the counterfactual contrast under that treatment.
      This asks the question the paper's "surgical removal does not work" claim
      actually needs: after partial removal, does a residual channel still move
      the prediction?  <- this script

Concretely, per participant we score two arms built from the SAME
clinical-stripped participant+interviewer transcript, with the two fixed pools
interleaved as in `run_a2b.interleave`. The participant's own words are
byte-identical between arms, exactly as in the main counterfactual protocol, so
the resulting CSS is directly comparable to `detection.counterfactual_css`.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
import numpy as np

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/i12-targeted-cf/results"
LABELS = ["Yes", "No"]
BASE = ('Consider this post: "{text}" '
        'Question: Does the poster suffer from depression?')
sig = lambda x: 1 / (1 + np.exp(-x))

sys.path.insert(0, ROOT)
from shortcut_harness.scoring import TextScorer, OmniScorer

_s = importlib.util.spec_from_file_location(
    "a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
a2b = importlib.util.module_from_spec(_s); sys.modules["a2b"] = a2b
_s.loader.exec_module(a2b)
POOLS, interleave = a2b.POOLS, a2b.interleave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=4032)
    ap.add_argument("--scorer", default="text", choices=["text", "omni"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = [json.loads(l) for l in open(COND)]
    y = np.array([r["phq8_binary"] for r in rows])

    # same cluster file and the same ID-regex membership test as
    # run_full_suite.py:124-131. Lines look like
    #   Ellie: therapy_going (do you still go to therapy now)
    # and only the parenthesised ID is matched against the cluster.
    import re as _re
    PAT = _re.compile(r"^Ellie: ([A-Za-z][A-Za-z0-9_]*) \((.*)\)\s*$")
    CL = set(json.load(open(f"{ROOT}/experiments/suite/results/"
                            f"e4a_prompt_clusters.json"))["clusters"]["clinical_care"])

    def strip_clinical(t):
        return "\n".join(l for l in (t or "").split("\n")
                          if not (PAT.match(l) and PAT.match(l).group(1) in CL))

    # Only 123 of 188 transcripts carry prompt IDs at all, so stripping is a
    # no-op on the other 65. Reporting over all 188 dilutes the effect by a
    # third; both subsets are given.
    has_id = np.array([any(PAT.match(l) for l in (r["text_full"] or "").split("\n"))
                       for r in rows])

    sc = (OmniScorer(args.ckpt, device=args.device)
          if args.scorer == "omni"
          else TextScorer(args.ckpt, adapter=args.adapter, device=args.device))
    longest = max(len(" ".join(p)) for p in POOLS.values())
    budget = args.budget - (longest // 3) - 32

    def trunc(t):
        w = (t or "").split()
        return " ".join(w[:budget]) if len(w) > budget else (t or "")

    def s1(prompt):
        sd = sc.score(prompt, LABELS)
        lp = {k: v["mean"] for k, v in sd.items()}
        return lp["Yes"] - lp["No"]

    ARMS = ["cf_neutral", "cf_depression"]
    S, t0 = {}, time.time()
    for arm in ARMS:
        acc = []
        for r in rows:
            stripped = strip_clinical(r["text_full"])
            # participant lines survive strip_clinical untouched; the pools are
            # then interleaved into what remains, same positions in both arms
            txt = interleave(trunc(stripped), POOLS[arm])
            acc.append(s1(BASE.format(text=txt)))
        S[arm] = np.array(acc)
        print(f"  tgt/{arm} done ({time.time()-t0:.0f}s)", flush=True)

    p_d, p_n = sig(S["cf_depression"]), sig(S["cf_neutral"])
    d = p_d - p_n
    def summarise(mask):
        dd = d[mask]; pd_, pn_ = p_d[mask], p_n[mask]
        return dict(n=int(mask.sum()), css_abs=float(np.abs(dd).mean()),
                    css_signed=float(dd.mean()),
                    flips=float(np.mean((pd_ > .5) != (pn_ > .5))),
                    frac_positive=float(np.mean(dd > 0)))

    res = dict(tag=args.tag, ckpt=args.ckpt, adapter=args.adapter,
               all_188=summarise(np.ones(len(rows), bool)),
               id_bearing_123=summarise(has_id),
               no_id_65=summarise(~has_id),
               n_with_id=int(has_id.sum()),
               scores={k: v.tolist() for k, v in S.items()})
    res.update(res["all_188"])
    json.dump(res, open(f"{OUT}/targeted_cf_{args.tag}.json", "w"), indent=1)
    for k in ("all_188", "id_bearing_123", "no_id_65"):
        v = res[k]
        print(f"  {k:16s} n={v['n']:3d}  |CSS| {v['css_abs']:.4f}  "
              f"signed {v['css_signed']:+.4f}  flips {v['flips']:.3f}")
    print(f"  ({time.time()-t0:.0f}s)")
    print(f"wrote {OUT}/targeted_cf_{args.tag}.json")


if __name__ == "__main__":
    main()
