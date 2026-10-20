#!/usr/bin/env python3
"""E5 scored under chain-of-thought, so P2 faces F-A on the same footing.

P2 was rejected under F-B (IID utility down 0.198 AUC, P(pred=Yes) >= 0.95 in
every arm). That rejection is sound but uses the weaker of the two criteria.
Every other intervention was judged on F-A -- the paired shortcut-vs-construct
test -- and P2 should be too, otherwise the comparison table mixes standards.

The prediction from I5 is specific and worth stating before the run: CoT pins
p(depression) near 0.97 for everyone, so construct sensitivity should collapse
alongside CF-Sens. If it does, P2 fails F-A as well as F-B and the flattening
diagnosis is confirmed on the criterion designed to detect it. If construct
sensitivity somehow survives, the I5 reading is wrong and CoT is doing something
more interesting than saturating.

Same 200 pairs, same generation settings as run_p2.py (greedy, 128 new tokens),
label scored teacher-forced conditioned on the generated reasoning.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import torch

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
PAIRS = f"{ROOT}/experiments/suite/results/e5_construct_pairs.jsonl"
OUT = f"{ROOT}/experiments/i5-p2-cot/results"
LABELS = ["Yes", "No"]
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')
COT_TRIGGER = " Let's think step by step about the clinical evidence."
COT_ANSWER = " Therefore, the answer is"


@torch.no_grad()
def generate(sc, prompt, max_new_tokens=128):
    rendered = sc._render(prompt)
    ids = sc.tok(rendered, return_tensors="pt").input_ids.to(sc.model.device)
    out = sc.model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=sc.tok.pad_token_id or sc.tok.eos_token_id)
    gen = out[0, ids.shape[1]:]
    return sc.tok.decode(gen, skip_special_tokens=True), int(gen.shape[0])


def degenerate(text):
    t = text.strip()
    if len(t) < 10:
        return True
    w = t.split()
    return len(w) >= 8 and len(set(w)) / len(w) < 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=3600)
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pairs = [json.loads(l) for l in open(PAIRS)]
    sc = TextScorer(f"{ROOT}/checkpoints/{args.model}", device=args.device)
    print(f"loaded {args.model} | {len(pairs)} pairs", flush=True)

    out_path = f"{OUT}/e5_cot_{args.model}.jsonl"
    n, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, p in enumerate(pairs):
            for side in ("pos", "neg"):
                d = p[side]
                ids = sc.tok(d["text"], add_special_tokens=False).input_ids
                trunc = len(ids) > args.budget
                text = sc.tok.decode(ids[:args.budget]) if trunc else d["text"]
                base = TEMPLATE.format(text=text)
                reasoning, n_tok = generate(sc, base + COT_TRIGGER, args.max_new)
                full = base + COT_TRIGGER + " " + reasoning.strip() + COT_ANSWER
                scored = sc.score(full, LABELS)
                lp = {k: v["mean"] for k, v in scored.items()}
                f.write(json.dumps({
                    "pair": i, "side": side, "variant": "cot",
                    "uid": d["uid"], "split": d["split"],
                    "context_uid": p["context_uid"], "gold": d["gold"],
                    "logprobs": lp, "pred": max(lp, key=lp.get),
                    "n_gen_tokens": n_tok, "degenerate": degenerate(reasoning),
                    "reasoning": reasoning.strip()[:400], "truncated": trunc,
                }) + "\n")
                n += 1
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(pairs)} pairs | {n} scored | "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"\nwrote {out_path} ({n} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
