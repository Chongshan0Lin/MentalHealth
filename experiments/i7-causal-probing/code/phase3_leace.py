#!/usr/bin/env python3
"""I7 Phase 3 -- causal erasure at layer 13 (LEACE), with the control task.

Reconnaissance from the 33-layer sweep: signal STRENGTH peaks at layer 10, but
behavioural relevance peaks at layer 13 and forms a contiguous band over layers
8-21 (14/14 significant, all positive, paired Wilcoxon vs random p = 1.2e-4).
We erase where the signal relates to BEHAVIOUR, not where it is most decodable.

Method: LEACE \\[Belrose et al. 2023\\], closed-form, in the amnesic-probing
framework \\[Elazar et al. 2021\\], following \\[Dobrzeniecka et al. 2025\\] in
preferring closed-form erasure to INLP's iterative nullspace projection.

For a binary concept z the LEACE eraser is rank-one:
    x  ->  x - S^{1/2} u u^T S^{-1/2} (x - mu),      u = S^{-1/2} Sxz, normalised
with S = Cov(x) (ridge-regularised) and Sxz = Cov(x, z).

THE CONTROL TASK IS THE EXPERIMENT. An identical eraser is fitted to a
SHUFFLED-LABEL direction -- same rank, same machinery, same covariance -- and
applied the same way. Any drop in the counterfactual effect must be compared
against it, because removing a rank-one direction from the residual stream costs
something regardless of what that direction encodes. Without this comparison a
causal claim does not stand.

THIRD CONTROL, BEYOND ELAZAR'S PROTOCOL. We report SER = shortcut sensitivity /
construct sensitivity. Two methods in this project (chain-of-thought,
product-of-experts) drove shortcut sensitivity to near zero by destroying the
model. Erasure can fail the same way. Success needs the counterfactual effect to
shrink WHILE construct sensitivity and IID AUC hold.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
E5 = f"{ROOT}/experiments/suite/results/e5_construct_pairs.jsonl"
OUT = f"{ROOT}/experiments/i7-causal-probing/results"
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')
LABELS = ["Yes", "No"]
ARMS = ["cf_none", "cf_neutral", "cf_depression"]
sig = lambda x: 1 / (1 + np.exp(-x))

_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
POOLS, interleave = _a2b.POOLS, _a2b.interleave


# ------------------------------------------------------------------ LEACE

def fit_leace(X, z, ridge=1e-3):
    """Rank-one LEACE eraser for a binary concept. Returns (mu, A, B) with the
    erasure  x -> x - A (B . (x - mu))."""
    X = X.astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    S = (Xc.T @ Xc) / (len(X) - 1)
    S += ridge * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    S_half = (V * np.sqrt(w)) @ V.T
    S_ihalf = (V / np.sqrt(w)) @ V.T
    sxz = (Xc * (z - z.mean())[:, None]).mean(0)
    u = S_ihalf @ sxz
    u /= np.linalg.norm(u) + 1e-12
    return mu, S_half @ u, S_ihalf @ u


def make_hook(mu, A, B, device, dtype):
    mu_t = torch.tensor(mu, device=device, dtype=dtype)
    A_t = torch.tensor(A, device=device, dtype=dtype)
    B_t = torch.tensor(B, device=device, dtype=dtype)

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        coef = ((h - mu_t) * B_t).sum(-1, keepdim=True)
        h2 = h - coef * A_t
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return hook


# ------------------------------------------------------------------ scoring

@torch.no_grad()
def score(model, tok, prompt):
    """Teacher-forced label log-odds, matching the project's scoring rule."""
    base = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    outs = []
    for lab in LABELS:
        full = tok(prompt + " " + lab, return_tensors="pt").input_ids.to(
            model.device)
        logits = model(input_ids=full).logits
        lp = logits[0, :-1].log_softmax(-1)
        n = full.shape[1] - base.shape[1]
        tgt = full[0, 1:]
        outs.append(float(lp[-n:].gather(1, tgt[-n:, None]).mean()))
    return outs[0] - outs[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", type=int, nargs="+", default=[13],
                    help="erase at every listed layer simultaneously. Erasing "
                         "at one layer leaves the remaining blocks free to "
                         "regenerate the offset and pass it downstream, which "
                         "is the likeliest reason the single-layer erasure at "
                         "13 removed only 6.3 %% of the behavioural effect.")
    ap.add_argument("--budget", type=int, default=4032)
    ap.add_argument("--tokens-per-item", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = f"{ROOT}/checkpoints/{args.model}"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.bfloat16, device_map=args.device).eval()
    blocks = [model.model.layers[L - 1] for L in args.layers]  # hs[L] = block L-1
    print(f"loaded {args.model}; erasing at layers {args.layers}", flush=True)

    rows = [json.loads(l) for l in open(COND)]
    longest = max(len(" ".join(p)) for p in POOLS.values())
    budget = args.budget - (longest // 3) - 32
    rng = np.random.default_rng(0)

    def build(r, arm):
        ids = tok(r["text_participant_only"], add_special_tokens=False).input_ids
        C = tok.decode(ids[:budget]) if len(ids) > budget else r["text_participant_only"]
        return TEMPLATE.format(text=interleave(C, POOLS.get(arm, [])))

    # ---- Pass A: per-token states at the erasure layer ---------------------
    print("pass A: collecting per-token states for the LEACE fit", flush=True)
    Xs = {L: [] for L in args.layers}; zs, t0 = [], time.time()
    for i, r in enumerate(rows):
        for a, arm in enumerate(["cf_neutral", "cf_depression"]):
            enc = tok(build(r, arm), return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model(**enc, output_hidden_states=True)
            n = None
            for L in args.layers:
                h = o.hidden_states[L][0].float().cpu().numpy()
                if n is None:
                    n = rng.choice(len(h), min(args.tokens_per_item, len(h)),
                                   replace=False)
                Xs[L].append(h[n])
            zs.append(np.full(len(n), a))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    z = np.concatenate(zs).astype(float)
    zc = rng.permutation(z)
    # one eraser PER LAYER, each fitted on that layer's own statistics -- a
    # direction found at layer 13 is not the direction at layer 24
    erasers = {"real": {}, "control": {}}
    for L in args.layers:
        X = np.concatenate(Xs[L])
        erasers["real"][L] = fit_leace(X, z)
        erasers["control"][L] = fit_leace(X, zc)
    print(f"  fit set per layer: {np.concatenate(Xs[args.layers[0]]).shape}, "
          f"class balance {z.mean():.3f}, {len(args.layers)} erasers each",
          flush=True)

    # ---- Pass B: score under no / real / control erasure -------------------
    pairs = [json.loads(l) for l in open(E5)]
    res = {}
    for cond in ["none", "real", "control"]:
        handles = []
        if cond != "none":
            dt = next(model.parameters()).dtype
            for L, blk in zip(args.layers, blocks):
                mu, A, B = erasers[cond][L]
                handles.append(blk.register_forward_hook(
                    make_hook(mu, A, B, model.device, dt)))
        t1 = time.time()
        arm_scores = {a: [] for a in ARMS}
        for r in rows:
            for arm in ARMS:
                arm_scores[arm].append(score(model, tok, build(r, arm)))
        pos, neg = [], []
        for p in pairs:
            pos.append(score(model, tok, TEMPLATE.format(text=p["pos"]["text"])))
            neg.append(score(model, tok, TEMPLATE.format(text=p["neg"]["text"])))
        for hd in handles:
            hd.remove()
        res[cond] = {k: np.array(v) for k, v in arm_scores.items()}
        res[cond]["pos"] = np.array(pos); res[cond]["neg"] = np.array(neg)
        print(f"  {cond}: scored in {time.time()-t1:.0f}s", flush=True)

    # ---- report ------------------------------------------------------------
    from scipy.stats import rankdata
    y = np.array([r["phq8_binary"] for r in rows])

    def auc(y_, s):
        n1 = int(y_.sum()); n0 = len(y_) - n1
        r_ = rankdata(s)
        return float((r_[y_ == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    print("\n" + "=" * 92)
    print(f"PHASE 3 -- LEACE erasure at layers {args.layers}")
    print("=" * 92)
    print(f"{'condition':10s} {'dp (S-sens)':>12s} {'vs none':>9s} "
          f"{'C-sens':>8s} {'SER':>6s} {'AUC cf_none':>12s}")
    print("-" * 92)
    out = {}
    for cond in ["none", "real", "control"]:
        R = res[cond]
        s = float(np.abs(sig(R["cf_depression"]) - sig(R["cf_neutral"])).mean())
        c = float(np.abs(sig(R["pos"]) - sig(R["neg"])).mean())
        a = auc(y, R["cf_none"])
        res[cond]["_s"] = s
        print(f"{cond:10s} {s:12.4f} {s/res['none']['_s']:8.2f}x {c:8.4f} "
              f"{s/max(c,1e-9):6.2f} {a:12.3f}")
        out[cond] = dict(s_sens=s, c_sens=c, ser=s / max(c, 1e-9), auc=a)

    print("\n" + "=" * 92); print("VERDICT"); print("=" * 92)
    r_, c_, n_ = out["real"], out["control"], out["none"]
    drop_r = 1 - r_["s_sens"] / n_["s_sens"]
    drop_c = 1 - c_["s_sens"] / n_["s_sens"]
    print(f"  shortcut effect: real erasure -{drop_r:.1%}, "
          f"control erasure -{drop_c:.1%}")
    causal = drop_r > 2 * max(drop_c, 0.01)
    intact = r_["auc"] >= n_["auc"] - 0.05 and r_["c_sens"] >= 0.8 * n_["c_sens"]
    if causal and intact:
        print("  PASS -- the drop is specific to this direction AND the model")
        print("          survives (AUC and construct sensitivity preserved).")
        print("          This is causal evidence, and a representation-level")
        print("          mitigation for the taxonomy.")
    elif causal:
        print("  PARTIAL -- direction-specific, but the model degraded")
        print(f"          (AUC {n_['auc']:.3f} -> {r_['auc']:.3f}, "
              f"C-sens {n_['c_sens']:.4f} -> {r_['c_sens']:.4f}).")
        print("          Same failure mode as CoT and PoE: flattening.")
    else:
        print("  FAIL -- the drop is not distinguishable from removing an")
        print("          arbitrary rank-one direction. No causal claim.")
    json.dump(out, open(f"{OUT}/phase3_leace_L{args.layers[0]}-{args.layers[-1]}.json", "w"),
              indent=2)
    print(f"\nwrote {OUT}/phase3_leace_layer{args.layer}.json")


if __name__ == "__main__":
    main()
