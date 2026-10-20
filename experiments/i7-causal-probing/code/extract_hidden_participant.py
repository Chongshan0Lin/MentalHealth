#!/usr/bin/env python3
"""I7 Phase 1a (v2) -- hidden states pooled over PARTICIPANT tokens only.

WHY v1 FAILED. The first version pooled over the whole prompt, interviewer
tokens included. The two arms differ by literal word presence ("diagnosed"
appears in one and not the other), so a TF-IDF baseline on the raw text
separates them at AUC 1.000. With the ceiling at 1.0 no probe can exceed it and
the gate is unpassable by construction. That run answered nothing.

WHAT v2 ASKS INSTEAD. Pool only over the token positions belonging to the
PARTICIPANT's own turns, excluding every interviewer token. The participant text
is byte-identical across arms -- it is truncated once and reused, exactly as in
run_a2b.py -- so a lexical baseline on it is exactly chance. The question
becomes:

    does the interviewer's context change how the model represents the
    PATIENT'S OWN WORDS?

Both outcomes are informative, which is what v1 lacked:
  separable   -> the context contaminates the evidence representation, which
                 would explain the behavioural shift
  inseparable -> the patient's representation is untouched and the effect
                 happens at readout, consistent with our finding that the
                 shortcut moves the decision threshold and not the ranking

POSITIONAL CONFOUND, AND THE CONTROL FOR IT. Interviewer pools differ in length
(cf_neutral 50 words, cf_depression 38), so participant tokens sit at different
absolute positions across arms and could differ through position alone rather
than context content. We therefore also extract `cf_neutral_matched` (38 words,
built in I2 to match cf_depression to within one word). Comparing against that
arm removes almost all of the positional difference.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, re, sys, time
import numpy as np
import torch

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
E3 = f"{ROOT}/experiments/i2-e3-matched-and-prompting/results/e3_pools.json"
OUT = f"{ROOT}/experiments/i7-causal-probing/results"
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')

_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
POOLS, interleave = dict(_a2b.POOLS), _a2b.interleave
POOLS.update({k: v for k, v in json.load(open(E3)).items()
              if k not in POOLS})
ARMS = ["cf_neutral", "cf_depression", "cf_neutral_matched"]


def participant_spans(prompt, body):
    """Character spans of the participant's lines inside the full prompt."""
    base = prompt.index(body)
    spans, off = [], 0
    for line in body.split("\n"):
        if line.startswith("Participant:"):
            spans.append((base + off, base + off + len(line)))
        off += len(line) + 1
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="absolute checkpoint path")
    ap.add_argument("--adapter", default=None, help="LoRA dir (e.g. Menta)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--scorer", default="text", choices=["text", "omni"])
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=4032)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="default: every layer 0..N (embeddings first)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    ckpt = args.ckpt or f"{ROOT}/checkpoints/{args.model}"
    tag = args.tag or os.path.basename(ckpt)
    render = lambda p: p          # text models take the raw prompt
    if args.scorer == "omni":
        # Qwen2.5-Omni thinker. The chat template wraps the prompt, so the
        # participant spans must be located in the RENDERED string, not the
        # raw one -- otherwise the offsets point at the wrong characters.
        from transformers import (Qwen2_5OmniProcessor,
                                  Qwen2_5OmniThinkerForConditionalGeneration)
        proc = Qwen2_5OmniProcessor.from_pretrained(ckpt)
        tok = proc.tokenizer
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            ckpt, dtype=torch.bfloat16, device_map=args.device).eval()
        render = lambda p: proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": p}]}],
            add_generation_prompt=True, tokenize=False)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.adapter or ckpt)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, dtype=torch.bfloat16, device_map=args.device).eval()
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).eval()
    assert tok.is_fast, "need a fast tokenizer for offset mapping"
    D = model.config.hidden_size
    if args.layers is None:
        args.layers = list(range(model.config.num_hidden_layers + 1))
    print(f"loaded {tag}: {model.config.num_hidden_layers} layers, "
          f"hidden {D}", flush=True)

    rows = [json.loads(l) for l in open(COND)]
    longest = max(len(" ".join(p)) for p in POOLS.values())
    budget = args.budget - (longest // 3) - 32

    H = np.zeros((len(rows), len(ARMS), len(args.layers), D), dtype=np.float32)
    meta, ntok, tokstats = [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        ids = tok(r["text_participant_only"], add_special_tokens=False).input_ids
        trunc = len(ids) > budget
        C = tok.decode(ids[:budget]) if trunc else r["text_participant_only"]
        per_arm = []
        for a, arm in enumerate(ARMS):
            body = interleave(C, POOLS[arm])
            prompt = render(TEMPLATE.format(text=body))
            enc = tok(prompt, return_tensors="pt",
                      return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(model.device) for k, v in enc.items()}
            spans = participant_spans(prompt, body)
            mask = np.array([any(s <= o0 and o1 <= e for s, e in spans)
                             and o1 > o0 for o0, o1 in offs])
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            for li, L in enumerate(args.layers):
                h = out.hidden_states[L][0].float().cpu().numpy()
                H[i, a, li] = h[mask].mean(0)
            per_arm.append(int(mask.sum()))
        meta.append({"participant_id": r["participant_id"], "uid": r["uid"],
                     "split": r["split"], "phq8_binary": r["phq8_binary"],
                     "truncated": bool(trunc),
                     "n_participant_tokens": per_arm})
        tokstats.append(per_arm)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)} | {el:.0f}s "
                  f"({el/(i+1):.1f}s/participant)", flush=True)

    ts = np.array(tokstats)
    print(f"\nparticipant tokens pooled per arm (mean): "
          f"{dict(zip(ARMS, ts.mean(0).round(1)))}")
    print(f"  identical count across arms for "
          f"{100*(ts[:,0]==ts[:,1]).mean():.0f}% of participants "
          f"(neutral vs clinical)")
    np.savez_compressed(f"{OUT}/hidden_{tag}.npz", H=H,
                        layers=np.array(args.layers), arms=np.array(ARMS))
    json.dump(meta, open(f"{OUT}/hidden_{tag}_meta.json", "w"),
              indent=2)
    print(f"wrote {OUT}/hidden_{tag}.npz  shape {H.shape} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
