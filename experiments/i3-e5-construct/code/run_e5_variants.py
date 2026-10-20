#!/usr/bin/env python3
"""E5 under the input-level interventions -- closing the F-A gap.

I3 evaluated only P1 against the construct-preservation criterion. D1 is the
project's STRONGEST intervention (+0.126 AUC) and its construct cost has never
been measured, so it has never faced F-A at all. This scores the same 200 E5
pairs under:

  d1   interviewer turns deleted entirely      (= A1 participant_only)
  d1m  interviewer words replaced with [Q]     (= A1 ellie_masked)

P3-c is NOT scored separately: I1 §4 established it reduces algebraically to the
participant-only score, so its E5 numbers are d1's by construction. Spending GPU
to recompute an identity would be waste, and reporting it as an independent
result would be double-counting.

Note what D1 does to this measurement. E5 varies the participant with the
interviewer held fixed; D1 removes the interviewer. The pair then differs ONLY
in participant, which is exactly the quantity construct sensitivity is defined
on -- so d1's construct sensitivity is the model's evidence-sensitivity with the
side channel gone. That is the right comparison, but it means d1 changes the
stimulus rather than only the decision rule, and the write-up must say so.
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
PAIRS = f"{ROOT}/experiments/suite/results/e5_construct_pairs.jsonl"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/i3-e5-construct/results"
LABELS = ["Yes", "No"]
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')


def interleave_lines(participant_text, lines):
    """Same insertion geometry as run_a2b.interleave, but taking whole lines."""
    plines = [l for l in participant_text.split("\n") if l.strip()]
    if not lines:
        return "\n".join(plines)
    step = max(1, len(plines) // (len(lines) + 1))
    out, pi = [], 0
    for i, l in enumerate(plines):
        if pi < len(lines) and i == pi * step:
            out.append(lines[pi]); pi += 1
        out.append(l)
    out.extend(lines[pi:])
    return "\n".join(out)


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
    print(f"loaded {args.model} | {len(pairs)} pairs", flush=True)

    out_path = f"{OUT}/e5_variants_{args.model}.jsonl"
    n, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, p in enumerate(pairs):
            ell = [l for l in (cond[p["context_uid"]]["text_ellie_only"] or ""
                               ).split("\n") if l.strip()]
            for side in ("pos", "neg"):
                d = p[side]
                pt = cond[d["uid"]]["text_participant_only"]
                variants = {
                    "d1":  interleave_lines(pt, []),                    # no S
                    "d1m": interleave_lines(pt, ["Ellie: [Q]"] * len(ell)),
                }
                for vname, text in variants.items():
                    ids = sc.tok(text, add_special_tokens=False).input_ids
                    trunc = len(ids) > args.budget
                    if trunc:
                        text = sc.tok.decode(ids[:args.budget])
                    scored = sc.score(TEMPLATE.format(text=text), LABELS)
                    lp = {k: v["mean"] for k, v in scored.items()}
                    f.write(json.dumps({
                        "pair": i, "side": side, "variant": vname,
                        "uid": d["uid"], "split": d["split"],
                        "context_uid": p["context_uid"], "gold": d["gold"],
                        "logprobs": lp, "pred": max(lp, key=lp.get),
                        "truncated": trunc,
                    }) + "\n")
                    n += 1
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(pairs)} pairs | {n} scored | "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"\nwrote {out_path} ({n} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
