#!/usr/bin/env python3
"""Smoke test OSim-8B and Menta (Qwen3-4B base + LoRA) on one IMHI sample."""
import argparse, json, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/data/cl195/NeurIPS_workshop"
MANIFEST = f"{ROOT}/data/processed/text_manifest.jsonl"


def load(model):
    if model == "Menta":
        # HF repo ships only a LoRA adapter; the base has to be attached.
        from peft import PeftModel
        base = f"{ROOT}/checkpoints/Qwen3-4B-base"
        tok = AutoTokenizer.from_pretrained(base)
        m = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, device_map="cuda:0")
        m = PeftModel.from_pretrained(m, f"{ROOT}/checkpoints/Menta-LoRA/Menta")
        return tok, m.eval()
    ck = f"{ROOT}/checkpoints/{model}"
    tok = AutoTokenizer.from_pretrained(ck)
    m = AutoModelForCausalLM.from_pretrained(
        ck, dtype=torch.bfloat16, device_map="cuda:0")
    return tok, m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["OSim-8B", "Menta"])
    args = ap.parse_args()

    rec = next(json.loads(l) for l in open(MANIFEST)
               if json.loads(l)["source_dataset"] == "dreaddit")
    print(f"sample {rec['uid']} label={rec['label']} words={rec['n_words']}")

    t0 = time.time()
    tok, model = load(args.model)
    print(f"loaded {args.model} in {time.time()-t0:.0f}s | "
          f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    text = " ".join(rec["text"].split()[:300])
    msgs = [{"role": "user",
             "content": f'Consider this post: "{text}" {rec["question"]}'}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     tokenize=False)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=100, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    print("\n--- generation ---")
    print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)[:600])
    print(f"\npeak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
