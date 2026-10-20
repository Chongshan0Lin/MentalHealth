#!/usr/bin/env python3
"""Smoke test the Qwen2.5-Omni-Thinker checkpoints on one real DAIC clip.

Both OmniSapiens2.0 and HumanOmniV2 share this architecture, so the same
loading path covers them.  Audio is truncated to --seconds because the Whisper
front-end tops out at chunk_length=300s.

Usage: python smoke_omni.py <OmniSapiens2.0|HumanOmniV2> [--seconds 60]
"""
import argparse, json, time

import librosa
import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

ROOT = "/data/cl195/NeurIPS_workshop"
MANIFEST = f"{ROOT}/data/processed/daic_manifest.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()
    ckpt = f"{ROOT}/checkpoints/{args.model}"

    rec = json.loads(open(MANIFEST).readline())
    sr = 16000
    wav, _ = librosa.load(rec["audio_path"], sr=sr, mono=True,
                          duration=args.seconds)
    print(f"sample {rec['uid']} label={rec['phq8_binary']} "
          f"audio={len(wav)/sr:.0f}s of {rec['audio_seconds']}s")

    t0 = time.time()
    proc = Qwen2_5OmniProcessor.from_pretrained(ckpt)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        ckpt, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print(f"loaded {args.model} in {time.time()-t0:.0f}s | "
          f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    conv = [
        {"role": "system", "content": [
            {"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": rec["audio_path"]},
            {"type": "text", "text": "Describe this person's emotional state "
                                     "based on the audio."}]},
    ]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    # Leave input_features in float32 -- the audio tower casts internally, and
    # forcing them to bfloat16 here silently corrupts the Whisper features and
    # the model degenerates into repeated tokens.
    inputs = proc(text=[text], audio=[wav], sampling_rate=sr,
                  return_tensors="pt", padding=True).to(model.device)

    # OmniSapiens2.0 ships an empty generation_config.json, so generate() never
    # stops at <|im_end|> and rambles on in fake "Human:" turns after a correct
    # short answer.  Pin the chat EOS/pad ids from the tokenizer for any model
    # whose generation config omits them.
    eos = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad = proc.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if model.generation_config.eos_token_id is None:
        print(f"[fix] generation_config has no eos_token_id; pinning {eos}")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                             eos_token_id=eos, pad_token_id=pad)
    gen = out[0][inputs["input_ids"].shape[1]:]
    print(f"\n--- generation ({time.time()-t0:.0f}s) ---")
    print(proc.batch_decode([gen], skip_special_tokens=True)[0])
    print(f"\npeak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
