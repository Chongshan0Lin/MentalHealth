#!/usr/bin/env python3
"""Pick the RL input length: the shortcut effect must survive truncation, and
16 concurrent rollouts must fit in one 48 GB card."""
import json, sys, importlib.util, numpy as np
sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer
ROOT = "/data/cl195/NeurIPS_workshop"
s = importlib.util.spec_from_file_location(
    "a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
a2b = importlib.util.module_from_spec(s); sys.modules["a2b"] = a2b
s.loader.exec_module(a2b)
POOLS = dict(a2b.POOLS); POOLS["cf_none"] = []; interleave = a2b.interleave
BASE = ('Consider this post: "{text}" '
        'Question: Does the poster suffer from depression?')
rows = [json.loads(l) for l in open(f"{ROOT}/data/processed/daic_conditions.jsonl")]
sc = TextScorer(f"{ROOT}/checkpoints/MentaLLaMA-chat-7B", device="cuda:0")
sig = lambda x: 1 / (1 + np.exp(-x))
out = {}
print(f"{'words':>6s} {'~tok':>6s} {'mean dp':>9s} {'flips':>7s} {'KV(16 seq)':>11s}",
      flush=True)
for W in [1000, 1200, 1400, 1800]:
    d, f = [], 0
    for r in rows:
        C = " ".join(r["text_participant_only"].split()[:W]); v = {}
        for arm in ["cf_neutral", "cf_depression"]:
            lp = sc.score(BASE.format(text=interleave(C, POOLS[arm])), ["Yes", "No"])
            v[arm] = lp["Yes"]["mean"] - lp["No"]["mean"]
        d.append(sig(v["cf_depression"]) - sig(v["cf_neutral"]))
        f += (v["cf_depression"] > 0) != (v["cf_neutral"] > 0)
    nt = len(sc.tok(BASE.format(text=interleave(
        " ".join(rows[0]["text_participant_only"].split()[:W]),
        POOLS["cf_depression"])), add_special_tokens=False).input_ids)
    kv = 16 * (nt + 256) * 512 / 1024**2
    out[W] = dict(tokens=nt, dp=float(np.mean(d)), flips=f/len(rows), kv_gb=kv)
    print(f"{W:6d} {nt:6d} {np.mean(d):+9.4f} {f/len(rows):7.3f} {kv:9.1f}GB",
          flush=True)
json.dump(out, open(f"{ROOT}/experiments/i8-grpo-mitigation/results/length_sweep.json","w"), indent=2)
print("\n  full-length reference: dp +0.2050, flips 0.559")
print("  budget: 48GB - 14GB weights - ~6GB activations = ~28GB available for KV")
