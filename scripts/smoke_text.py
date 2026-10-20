#!/usr/bin/env python3
"""Smoke test: load MentaLLaMA and run it on one real DAIC transcript."""
import json, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = "/data/cl195/NeurIPS_workshop/checkpoints/MentaLLaMA-chat-7B"
MANIFEST = "/data/cl195/NeurIPS_workshop/data/processed/daic_manifest.jsonl"


def participant_text(path, max_words=400):
    import pandas as pd
    t = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
    turns = t[t["speaker"] == "Participant"]["value"].fillna("").astype(str)
    words = " ".join(turns).split()
    return " ".join(words[:max_words])


def main():
    rec = json.loads(open(MANIFEST).readline())
    text = participant_text(rec["transcript_path"])
    print(f"sample {rec['uid']}  label={rec['phq8_binary']}  words={len(text.split())}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s | {model.config.model_type} | "
          f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B params | "
          f"dtype={next(model.parameters()).dtype}")

    prompt = (f"Consider this post: \"{text}\" Question: Does the poster suffer "
              f"from depression?")
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80, do_sample=False)
    print("\n--- generation ---")
    print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))

    # label-scoring path: log P(label | prompt) for a fixed label set
    print("\n--- label log-probs ---")
    for lab in ["Yes", "No"]:
        full = tok(prompt + " " + lab, return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(**full).logits[0, :-1].log_softmax(-1)
        tgt = full["input_ids"][0, 1:]
        n = full["input_ids"].shape[1] - ids["input_ids"].shape[1]
        print(f"  {lab:4s}  mean logprob = {lg[-n:].gather(1, tgt[-n:, None]).mean():.4f}")

    print(f"\npeak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
