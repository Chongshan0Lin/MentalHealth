#!/usr/bin/env python3
"""I7 Phase 1a -- extract DAIC hidden states under the two counterfactual arms.

The existing probe (C1) was trained on IMHI/SDCNL and asks a between-corpus
question; the behavioural effect we want to explain is within-DAIC and turns on
which interviewer questions were asked. This extracts the representations the
correct probe needs.

Prompt construction is byte-identical to run_a2b.py -- same template, same
truncation budget, same interleaving positions -- so every representation here
corresponds one-to-one with a scored row already in
a2b_MentaLLaMA-chat-7B.jsonl. That correspondence is what makes Phase 2's
Δh-vs-Δp correlation well defined.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/i7-causal-probing/results"
ARMS = ["cf_neutral", "cf_depression"]
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')

_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
POOLS, interleave = _a2b.POOLS, _a2b.interleave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=4032)
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24, 32])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = f"{ROOT}/checkpoints/{args.model}"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"loaded {args.model}: {n_layers} layers, "
          f"hidden {model.config.hidden_size}", flush=True)
    assert max(args.layers) <= n_layers, f"model has only {n_layers} layers"

    rows = [json.loads(l) for l in open(COND)]
    # identical budget policy to run_a2b.py
    longest = max(len(" ".join(p)) for p in POOLS.values())
    budget = args.budget - (longest // 3) - 32

    H = np.zeros((len(rows), len(ARMS), len(args.layers), 2,
                  model.config.hidden_size), dtype=np.float32)
    meta = []
    t0 = time.time()
    for i, r in enumerate(rows):
        ids = tok(r["text_participant_only"], add_special_tokens=False).input_ids
        trunc = len(ids) > budget
        C = tok.decode(ids[:budget]) if trunc else r["text_participant_only"]
        for a, arm in enumerate(ARMS):
            text = interleave(C, POOLS[arm])
            enc = tok(TEMPLATE.format(text=text), return_tensors="pt").to(
                model.device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states          # tuple: embeddings + n_layers
            for li, L in enumerate(args.layers):
                h = hs[L][0].float()        # [T, D]
                H[i, a, li, 0] = h.mean(0).cpu().numpy()   # mean-pooled
                H[i, a, li, 1] = h[-1].cpu().numpy()       # last token
        meta.append({"participant_id": r["participant_id"], "uid": r["uid"],
                     "split": r["split"], "phq8_binary": r["phq8_binary"],
                     "truncated": bool(trunc)})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)} | {el:.0f}s "
                  f"({el/(i+1):.1f}s/participant)", flush=True)

    np.savez_compressed(f"{OUT}/daic_hidden.npz", H=H,
                        layers=np.array(args.layers), arms=np.array(ARMS))
    json.dump(meta, open(f"{OUT}/daic_hidden_meta.json", "w"), indent=2)
    print(f"\nwrote {OUT}/daic_hidden.npz  shape {H.shape} "
          f"({H.nbytes/2**20:.0f} MB uncompressed, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
