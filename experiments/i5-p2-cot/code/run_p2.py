#!/usr/bin/env python3
"""P2 -- chain-of-thought prompting as an anti-shortcut intervention.

The last untested intervention in INTERVENTION_DESIGN.md. Shortcut Suite
(EMNLP 2024) reports CoT mitigating several shortcut types, so it is a cheap
baseline worth having; the design's own note is that if a prompt fixes this,
retraining is unnecessary.

THE METHODOLOGICAL PROBLEM, AND HOW IT IS HANDLED
CoT needs free generation, which collides with the project's scoring rule:
parse-failure rate correlates with perturbation strength, so parsing a generated
answer manufactures fake accuracy drops in exactly the arms under test
(EXPERIMENT_DESIGN.md §0). Resolution, fixed in the design before running:

  1. generate the reasoning (greedy, so deterministic -- no seeds needed),
  2. score the label by teacher-forced log-likelihood CONDITIONED on the
     generated reasoning.

The answer is therefore never parsed. What can still go wrong is degenerate
generation (empty, truncated, or repetitive), so generation length and
degeneracy rate are logged per arm. If those correlate with arm -- i.e. the
clinical arm degenerates more than the biographical one -- the comparison is
confounded and the arm is void. That check is the point of recording them.

Run on the same three arms as P1 so the two prompting interventions are
directly comparable.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
import torch

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/i5-p2-cot/results"
LABELS = ["Yes", "No"]
LABEL_OF = {1: "Yes", 0: "No"}

TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')
COT_TRIGGER = " Let's think step by step about the clinical evidence."
COT_ANSWER = " Therefore, the answer is"

_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
POOLS, interleave = _a2b.POOLS, _a2b.interleave

ARMS = ["cf_none", "cf_neutral", "cf_depression"]


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
    """Empty, or a single token/phrase repeated -- the failure modes that would
    silently void an arm."""
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

    rows = [json.loads(l) for l in open(COND)]
    sc = TextScorer(f"{ROOT}/checkpoints/{args.model}", device=args.device)
    print(f"loaded {args.model}", flush=True)

    longest = max(len(" ".join(p)) for p in POOLS.values())
    budget = args.budget - (longest // 3) - 64

    out_path = f"{OUT}/p2_{args.model}.jsonl"
    n, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            gold = LABEL_OF[r["phq8_binary"]]
            ids = sc.tok(r["text_participant_only"],
                         add_special_tokens=False).input_ids
            trunc = len(ids) > budget
            C = sc.tok.decode(ids[:budget]) if trunc else r["text_participant_only"]

            for arm in ARMS:
                text = interleave(C, POOLS.get(arm, []))
                base = TEMPLATE.format(text=text)
                reasoning, n_tok = generate(sc, base + COT_TRIGGER,
                                            args.max_new)
                # label scored teacher-forced, conditioned on the reasoning --
                # never parsed out of it
                full = base + COT_TRIGGER + " " + reasoning.strip() + COT_ANSWER
                scored = sc.score(full, LABELS)
                lp = {k: v["mean"] for k, v in scored.items()}
                f.write(json.dumps({
                    "uid": r["uid"], "participant_id": r["participant_id"],
                    "split": r["split"], "model": args.model, "arm": arm,
                    "gold": gold, "phq8_binary": r["phq8_binary"],
                    "logprobs": lp, "pred": max(lp, key=lp.get),
                    "n_gen_tokens": n_tok,
                    "degenerate": degenerate(reasoning),
                    "reasoning": reasoning.strip()[:600],
                    "c_truncated": trunc,
                }) + "\n")
                n += 1
            f.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)} | {n} scored | {el:.0f}s "
                      f"({el/(i+1):.1f}s/participant)", flush=True)
    print(f"\nwrote {out_path} ({n} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
