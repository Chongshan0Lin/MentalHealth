#!/usr/bin/env python3
"""C1b (iii) -- does the model itself encode which corpus an item came from?

The TF-IDF probe shows source identity is recoverable from surface lexis, i.e.
available to any model. This script asks the stronger question: is it *linearly
present in the model's own representation* of the post?

The post is encoded alone -- no question, no template -- so the probe measures
text-borne provenance only. Any source information carried by the instruction
wording is handled separately by the native/normalized contrast in the main C1
run; mixing the two here would make a high probe score uninterpretable.

Hidden states are taken from several depths, because linear recoverability
typically peaks in the middle of the network rather than at the output layer,
and reporting only the last layer would understate the encoding.

Two pooling schemes per layer:
  last  - final-token state, what the LM actually conditions its next token on
  mean  - mean over tokens, a fairer readout of "is it in there at all"
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")

ROOT = "/data/cl195/NeurIPS_workshop"
MANIFEST = f"{ROOT}/data/processed/text_manifest.jsonl"
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=f"{ROOT}/runs/c1_MentaLLaMA-chat-7B.jsonl")
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--pca", type=int, default=256)
    args = ap.parse_args()

    ckpt = args.ckpt or f"{ROOT}/checkpoints/{args.model}"
    cache = args.cache or f"{ROOT}/runs/c1_hidden_{args.model}.npz"

    uids = {json.loads(l)["uid"] for l in open(args.run)}
    rows = [json.loads(l) for l in open(MANIFEST)]
    rows = [r for r in rows if r["uid"] in uids]
    src = np.array([r["source_dataset"] for r in rows])
    print(f"matched subset: {len(rows)} rows, {len(set(src))} sources")

    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        H, layers, src = z["H"], list(z["layers"]), z["src"]
        print(f"loaded cached states {H.shape} from {cache}")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.adapter or ckpt)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, dtype=torch.bfloat16, device_map=args.device)
        if args.adapter:                  # Menta ships a LoRA adapter only
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
        model = model.eval()
        n_layers = model.config.num_hidden_layers
        layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
        print(f"probing layers {layers} of {n_layers}")

        # H[i, layer_slot, pool_slot, :]
        H = np.zeros((len(rows), len(layers), 2, model.config.hidden_size),
                     dtype=np.float16)
        t0 = time.time()
        for i, r in enumerate(rows):
            ids = tok(r["text"], return_tensors="pt",
                      truncation=True, max_length=args.budget).input_ids
            with torch.no_grad():
                out = model(input_ids=ids.to(model.device),
                            output_hidden_states=True)
            for j, L in enumerate(layers):
                h = out.hidden_states[L][0].float()
                H[i, j, 0] = h[-1].cpu().numpy()
                H[i, j, 1] = h.mean(0).cpu().numpy()
            if (i + 1) % 250 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)}  {el:.0f}s ({el/(i+1):.2f}s/row)",
                      flush=True)
        np.savez_compressed(cache, H=H, layers=np.array(layers), src=src)
        print(f"wrote {cache}")

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, f1_score

    # PCA before the classifier. A 4096-d probe on 2,864 samples is badly
    # overparameterised -- 40k coefficients fit from 2.9k rows -- and lbfgs
    # spends hours converging on the ill-conditioned standardised matrix for a
    # fit that is mostly memorising noise. 256 components keep the linearly
    # decodable structure (which is what a probe is asking about) and make the
    # whole sweep run in minutes. The probe stays linear in the hidden state:
    # PCA is itself a linear map, so composing it with logistic regression
    # measures exactly the same "is source identity linearly recoverable"
    # quantity, just in a better-conditioned basis.
    print(f"\n{'='*92}", flush=True)
    print(f"SOURCE-IDENTITY PROBE ON HIDDEN STATES -- {args.model}")
    print(f"{'='*92}")
    print(f"{'layer':>8s} {'pool':>6s} {'acc':>7s} {'macroF1':>9s} "
          f"{'chance':>8s} {'shuffled':>9s}", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    chance = 1 / len(set(src.tolist()))
    rng = np.random.default_rng(SEED)
    for j, L in enumerate(layers):
        for k, pool in enumerate(["last", "mean"]):
            X = H[:, j, k, :].astype(np.float32)
            clf = make_pipeline(
                StandardScaler(),
                PCA(n_components=args.pca, random_state=SEED),
                LogisticRegression(max_iter=2000, class_weight="balanced"))
            pred = cross_val_predict(clf, X, src, cv=cv, n_jobs=5)
            shuf = np.mean([accuracy_score(rng.permutation(src), pred)
                            for _ in range(200)])
            print(f"{L:8d} {pool:>6s} {accuracy_score(src, pred):7.3f} "
                  f"{f1_score(src, pred, average='macro'):9.3f} {chance:8.3f} "
                  f"{shuf:9.3f}", flush=True)


if __name__ == "__main__":
    main()
