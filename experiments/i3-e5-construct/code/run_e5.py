#!/usr/bin/env python3
"""E5 -- construct-preservation scoring.

The RQ3 instrument, and the one metric the whole intervention design hangs on
(INTERVENTION_DESIGN.md §6 F-A). A2 fixed the participant and swapped the
interviewer; E5 does the mirror -- fixes the interviewer byte-identically and
swaps *which participant's words* sit inside it.

  construct sensitivity := mean |p(Y | C_i, S_j) - p(Y | C_i', S_j)|

An intervention that drives CF-Sens down while leaving this flat has debiased
the model. One that drives both down has merely made the model less responsive
to its input, which is the failure mode the design pre-commits to calling a
failure. Without this number, "shortcut reliance went down" is uninterpretable.

Pairs are built in ../../suite/code/build_suite.py: 200 pairs, each sharing one
interviewer context, one depressed and one non-depressed participant matched on
turn count (median |delta| = 7 turns). Gold follows the participant.

Also scores the same participants' unpaired text as an anchor, so construct
sensitivity can be read against the model's ordinary discrimination.
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
PAIRS = f"{ROOT}/experiments/suite/results/e5_construct_pairs.jsonl"
OUT = f"{ROOT}/experiments/i3-e5-construct/results"
LABELS = ["Yes", "No"]
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')

P1_INSTRUCTION = (
    "Base your prediction only on clinically relevant evidence in the "
    "participant's own words. Do not infer depression from the interviewer's "
    "choice of questions, from sentiment, or from writing style. ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=4032)
    ap.add_argument("--with-p1", action="store_true",
                    help="also score under the P1 instruction, so construct "
                         "preservation can be evaluated FOR an intervention")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pairs = [json.loads(l) for l in open(PAIRS)]
    sc = TextScorer(f"{ROOT}/checkpoints/{args.model}", device=args.device)
    print(f"loaded {args.model} | {len(pairs)} pairs", flush=True)

    variants = [("base", False)] + ([("p1", True)] if args.with_p1 else [])
    out_path = f"{OUT}/e5_{args.model}.jsonl"
    n, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, p in enumerate(pairs):
            for side in ("pos", "neg"):
                d = p[side]
                ids = sc.tok(d["text"], add_special_tokens=False).input_ids
                trunc = len(ids) > args.budget
                text = sc.tok.decode(ids[:args.budget]) if trunc else d["text"]
                for vname, instructed in variants:
                    prompt = TEMPLATE.format(text=text)
                    if instructed:
                        prompt = P1_INSTRUCTION + prompt
                    scored = sc.score(prompt, LABELS)
                    lp = {k: v["mean"] for k, v in scored.items()}
                    f.write(json.dumps({
                        "pair": i, "side": side, "variant": vname,
                        "uid": d["uid"], "split": d["split"],
                        "context_uid": p["context_uid"],
                        "gold": d["gold"], "logprobs": lp,
                        "pred": max(lp, key=lp.get),
                        "truncated": trunc,
                    }) + "\n")
                    n += 1
            f.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(pairs)} pairs | {n} scored | {el:.0f}s",
                      flush=True)
    print(f"\nwrote {out_path} ({n} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
