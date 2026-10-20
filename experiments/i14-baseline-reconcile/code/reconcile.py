#!/usr/bin/env python3
"""I14 -- why is the GRPO baseline CSS 0.2864 when every other MentaLLaMA
baseline in the paper is 0.205?

The two pipelines share a byte-identical prompt template and the same
counterfactual pools. They differ in exactly one thing that could matter:

  i10 shared sweep   truncate `text_participant_only` to ~3868 TOKENS
                     (`run_full_suite.py:103-106`), which truncates 17/188
  i8  GRPO eval      truncate to 1800 WORDS
                     (`eval_grpo.py --budget-words`, default 1800), 61/188

RL needed short inputs so 16 rollouts fit one 48 GB card, and the evaluation
inherited that budget. Less patient evidence leaves the fixed five-question
interviewer pool a larger share of the prompt, which should raise CSS.

This holds the scorer fixed and varies only the budget, so whatever gap remains
is not truncation.
"""
from __future__ import annotations
import importlib.util, json, os, sys, time
import numpy as np
import torch

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/i14-baseline-reconcile/results"
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')
sig = lambda x: 1 / (1 + np.exp(-x))

_s = importlib.util.spec_from_file_location(
    "a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
a2b = importlib.util.module_from_spec(_s); sys.modules["a2b"] = a2b
_s.loader.exec_module(a2b)
POOLS, interleave = a2b.POOLS, a2b.interleave


@torch.no_grad()
def score(model, tok, user):
    """Verbatim from eval_grpo.py:score -- bare-label teacher forcing."""
    base = tok(user, return_tensors="pt").input_ids.to(model.device)
    out = []
    for lab in ["Yes", "No"]:
        full = tok(user + " " + lab, return_tensors="pt").input_ids.to(model.device)
        lg = model(input_ids=full).logits[0, :-1].log_softmax(-1)
        n = full.shape[1] - base.shape[1]
        out.append(float(lg[-n:].gather(1, full[0, 1:][-n:, None]).mean()))
    return out[0] - out[1]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ck = f"{ROOT}/checkpoints/MentaLLaMA-chat-7B"
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForCausalLM.from_pretrained(
        ck, dtype=torch.bfloat16, device_map="cuda:0").eval()
    rows = [json.loads(l) for l in open(COND)]
    longest = max(len(" ".join(p)) for p in POOLS.values())
    tok_budget = 4032 - (longest // 3) - 64          # i10's number, verbatim

    def cut_words(t, n):
        w = t.split()
        return " ".join(w[:n]) if len(w) > n else t

    def cut_tokens(t, n):
        ids = tok(t, add_special_tokens=False).input_ids
        return tok.decode(ids[:n]) if len(ids) > n else t

    CONDS = [("i8  1800 words", lambda t: cut_words(t, 1800)),
             ("i10 %d tokens" % tok_budget, lambda t: cut_tokens(t, tok_budget)),
             ("no truncation", lambda t: t)]
    res = {}
    for name, cut in CONDS:
        t0 = time.time(); S = {}
        for arm in ("cf_neutral", "cf_depression"):
            S[arm] = np.array([score(model, tok, TEMPLATE.format(
                text=interleave(cut(r["text_participant_only"]), POOLS[arm])))
                for r in rows])
        p_hi, p_lo = sig(S["cf_depression"]), sig(S["cf_neutral"])
        d = p_hi - p_lo
        n_cut = sum(1 for r in rows
                    if cut(r["text_participant_only"]) != r["text_participant_only"])
        res[name] = dict(css_abs=float(np.abs(d).mean()), css_signed=float(d.mean()),
                         flips=float(np.mean((p_hi > .5) != (p_lo > .5))),
                         n_truncated=n_cut, secs=round(time.time() - t0))
        print(f"{name:18s} |CSS|={res[name]['css_abs']:.4f}  "
              f"signed={res[name]['css_signed']:+.4f}  "
              f"flips={res[name]['flips']:.3f}  truncated={n_cut}/188  "
              f"({res[name]['secs']}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/truncation_sweep.json", "w"), indent=1)
    print(f"\nreference: i10 stored 0.2050 | i7 LEACE 0.2057 | i8 GRPO eval 0.2864")


if __name__ == "__main__":
    main()
