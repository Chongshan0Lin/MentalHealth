#!/usr/bin/env python3
"""I13 -- is the shortcut direction TRANSPORTED or RECOMPUTED along the stream?

The audit left one question open. Erasing the single strongest linear direction
at layer 13 removes only 6.3 % of the behavioural effect; erasing at layers 8-24
removes 23.2 %. Three explanations fit that pattern equally well:

  (a) redundancy   -- independent copies live in layers outside the erased band
  (b) rank > 1     -- the concept occupies a subspace and rank-one erasure
                      leaves the rest of it standing
  (c) recomputation-- the model REBUILDS the direction downstream from the
                      interviewer tokens, so removing it once is undone

They are not distinguishable from the erasure numbers alone, and the difference
matters: under (a) and (b) a wide enough erasure eventually works; under (c) no
single-point intervention ever can, because the quantity is regenerated.

This separates them directly. Erase at ONE layer, then measure the probe's
Cohen's d at every layer DOWNSTREAM of it:

  d stays at ~0 after the erased layer  -> transported. The information was
      carried forward, cutting it once cuts it for good, and the residual 77 %
      comes from layers outside the band. Explanations (a)/(b).
  d RECOVERS above the erased layer     -> recomputed. The model reconstructs
      the direction from the input tokens at every block. Explanation (c).

LEACE's defining guarantee is that no linear probe can recover the concept from
the erased representation, so d at the erased layer itself must be ~0. That is
the positive control: if it is not, the hook is not doing what we think.

Cohen's d is cross-fitted exactly as in `layerwise_multi.py:xfit_d` -- direction
from the train fold, effect size on the held-out fold, GroupKFold on
participant_id -- so the numbers are directly comparable to the unerased curve
in `layerwise_multi.json`.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/i13-erasure-recovery/results"
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')

_s = importlib.util.spec_from_file_location(
    "a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
a2b = importlib.util.module_from_spec(_s); sys.modules["a2b"] = a2b
_s.loader.exec_module(a2b)
POOLS, interleave = a2b.POOLS, a2b.interleave

_l = importlib.util.spec_from_file_location(
    "p3", f"{ROOT}/experiments/i7-causal-probing/code/phase3_leace.py")
p3 = importlib.util.module_from_spec(_l); sys.modules["p3"] = p3
_l.loader.exec_module(p3)
fit_leace, make_hook = p3.fit_leace, p3.make_hook


def xfit_d(X, y, folds):
    """Cross-fitted Cohen's d. Verbatim from layerwise_multi.py:xfit_d."""
    ds, oof = [], np.zeros(len(y))
    for tr, te in folds:
        mu = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        u = mu / (np.linalg.norm(mu) + 1e-9)
        p = X[te] @ u
        a, b = p[y[te] == 0], p[y[te] == 1]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
        ds.append((b.mean() - a.mean()) / sd)
        oof[te] = p
    return float(np.mean(ds)), float(roc_auc_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--erase-layer", type=int, default=13)
    ap.add_argument("--budget", type=int, default=4032)
    ap.add_argument("--tokens-per-item", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = f"{ROOT}/checkpoints/{args.model}"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.bfloat16, device_map=args.device).eval()
    L0 = args.erase_layer
    block = model.model.layers[L0 - 1]      # hidden_states[L] is the output of block L-1
    nL = model.config.num_hidden_layers
    print(f"loaded {args.model}: {nL} blocks, erasing at layer {L0}", flush=True)

    rows = [json.loads(l) for l in open(COND)]
    pid = np.array([r["participant_id"] for r in rows])
    longest = max(len(" ".join(p)) for p in POOLS.values())
    budget = args.budget - (longest // 3) - 32
    rng = np.random.default_rng(0)

    def build(r, arm):
        ids = tok(r["text_participant_only"], add_special_tokens=False).input_ids
        C = tok.decode(ids[:budget]) if len(ids) > budget else r["text_participant_only"]
        return TEMPLATE.format(text=interleave(C, POOLS[arm]))

    ARMS = ["cf_neutral", "cf_depression"]

    def sweep(tag):
        """Mean-pooled hidden state at every layer, both arms, all participants."""
        H = np.zeros((len(rows), 2, nL + 1, model.config.hidden_size), np.float32)
        t0 = time.time()
        for i, r in enumerate(rows):
            for a, arm in enumerate(ARMS):
                enc = tok(build(r, arm), return_tensors="pt").to(model.device)
                with torch.no_grad():
                    o = model(**enc, output_hidden_states=True)
                for L in range(nL + 1):
                    H[i, a, L] = o.hidden_states[L][0].float().mean(0).cpu().numpy()
            if (i + 1) % 40 == 0:
                print(f"  [{tag}] {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        return H

    # ---- Pass A: per-token states at L0, to fit the eraser -----------------
    print("pass A: fitting the eraser at the target layer", flush=True)
    Xs, zs, t0 = [], [], time.time()
    for i, r in enumerate(rows):
        for a, arm in enumerate(ARMS):
            enc = tok(build(r, arm), return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model(**enc, output_hidden_states=True)
            h = o.hidden_states[L0][0].float().cpu().numpy()
            n = rng.choice(len(h), min(args.tokens_per_item, len(h)), replace=False)
            Xs.append(h[n]); zs.append(np.full(len(n), a))
        if (i + 1) % 40 == 0:
            print(f"  [fit] {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate(Xs); z = np.concatenate(zs).astype(float)
    mu, A, B = fit_leace(X, z)
    print(f"  eraser fitted on {X.shape}, class balance {z.mean():.3f}", flush=True)

    # ---- Pass B: layer sweep with and without the hook ---------------------
    print("pass B/none: layer sweep, no erasure", flush=True)
    H_none = sweep("none")
    dt = next(model.parameters()).dtype
    hd = block.register_forward_hook(make_hook(mu, A, B, model.device, dt))
    print(f"pass B/erased: layer sweep, eraser active at layer {L0}", flush=True)
    H_er = sweep("erased")
    hd.remove()

    # ---- probe every layer in both conditions ------------------------------
    y = np.concatenate([np.zeros(len(rows)), np.ones(len(rows))]).astype(int)
    g = np.concatenate([pid, pid])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(2 * len(rows)), y, g))

    print(f"\n{'='*84}\nCohen's d by layer -- does the direction come back after "
          f"erasure at L{L0}?\n{'='*84}")
    print(f"{'L':>3s} {'d none':>8s} {'d erased':>9s} {'recovery':>9s}  "
          f"{'AUC none':>9s} {'AUC erased':>11s}")
    out = []
    for L in range(nL + 1):
        Xn = np.concatenate([H_none[:, 0, L], H_none[:, 1, L]])
        Xe = np.concatenate([H_er[:, 0, L], H_er[:, 1, L]])
        dn, an = xfit_d(Xn, y, folds)
        de, ae = xfit_d(Xe, y, folds)
        rec = de / dn if abs(dn) > 1e-6 else float("nan")
        mark = "  <- erased here" if L == L0 else ""
        print(f"{L:3d} {dn:8.3f} {de:9.3f} {rec:9.3f}  {an:9.4f} {ae:11.4f}{mark}")
        out.append(dict(layer=L, d_none=dn, d_erased=de, recovery=rec,
                        auc_none=an, auc_erased=ae))

    post = [r for r in out if r["layer"] > L0]
    at = [r for r in out if r["layer"] == L0][0]
    print(f"\nat the erased layer L{L0}: d {at['d_none']:.3f} -> {at['d_erased']:.3f} "
          f"({at['recovery']*100:.1f} % of the original)")
    if post:
        print(f"mean over the {len(post)} layers downstream: "
              f"d {np.mean([r['d_none'] for r in post]):.3f} -> "
              f"{np.mean([r['d_erased'] for r in post]):.3f} "
              f"({np.mean([r['recovery'] for r in post])*100:.1f} % recovered)")
        print(f"final layer L{nL}: d {out[-1]['d_none']:.3f} -> "
              f"{out[-1]['d_erased']:.3f} ({out[-1]['recovery']*100:.1f} %)")

    json.dump(dict(model=args.model, erase_layer=L0, n_layers=nL, rows=out),
              open(f"{OUT}/recovery_L{L0}.json", "w"), indent=1)
    print(f"\nwrote {OUT}/recovery_L{L0}.json")


if __name__ == "__main__":
    main()
