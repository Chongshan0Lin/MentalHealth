#!/usr/bin/env python3
"""E5 under counterfactual marginalisation, to complete the SER table.

SER = (responsiveness to interviewer) / (responsiveness to patient). For
marginalisation the numerator is zero by construction -- the average over a
fixed set of interviewer contexts cannot depend on which single context was
presented -- but the DENOMINATOR is a real measurement and is what makes the
ratio meaningful rather than tautological. Without it we cannot distinguish
"ignores the clinician and still hears the patient" from "ignores everything".

For each E5 pair we replace the interviewer's turns with each of k fixed pools,
score, and average. Both sides of a pair receive the identical set of pools, so
the pair still differs only in which patient's words are present.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
PAIRS = f"{ROOT}/experiments/suite/results/e5_construct_pairs.jsonl"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/i3-e5-construct/results"
LABELS = ["Yes", "No"]
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')

_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
POOLS, interleave = _a2b.POOLS, _a2b.interleave
CONTEXTS = ["cf_none", "cf_neutral", "cf_depression", "cf_empirical"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=4032)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cond = {json.loads(l)["uid"]: json.loads(l) for l in open(COND)}
    pairs = [json.loads(l) for l in open(PAIRS)]
    sc = TextScorer(f"{ROOT}/checkpoints/{args.model}", device=args.device)
    print(f"loaded {args.model} | {len(pairs)} pairs x 2 sides x "
          f"{len(CONTEXTS)} contexts", flush=True)

    out_path = f"{OUT}/e5_marginalise_{args.model}.jsonl"
    n, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, p in enumerate(pairs):
            for side in ("pos", "neg"):
                d = p[side]
                pt = cond[d["uid"]]["text_participant_only"]
                ids = sc.tok(pt, add_special_tokens=False).input_ids
                trunc = len(ids) > args.budget - 64
                C = sc.tok.decode(ids[:args.budget - 64]) if trunc else pt
                for ctx in CONTEXTS:
                    text = interleave(C, POOLS.get(ctx, []))
                    scored = sc.score(TEMPLATE.format(text=text), LABELS)
                    lp = {k: v["mean"] for k, v in scored.items()}
                    f.write(json.dumps({
                        "pair": i, "side": side, "context": ctx,
                        "uid": d["uid"], "split": d["split"],
                        "gold": d["gold"], "logprobs": lp,
                        "truncated": trunc,
                    }) + "\n")
                    n += 1
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(pairs)} pairs | {n} scored | "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"\nwrote {out_path} ({n} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
