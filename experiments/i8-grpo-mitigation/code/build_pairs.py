#!/usr/bin/env python3
"""I8 step 1 -- build the (x_i, x_i') counterfactual pairs for GRPO.

Each pair holds symptom evidence byte-identical and changes only the
interviewer's questions, which is exactly the construction the RL design calls
for. The arms already exist and are frozen in SUITE_FREEZE.json, so no new
counterfactual data is generated here.

PAIR ADJACENCY MATTERS. R_inv compares the majority prediction of the x_i
rollouts against the majority of the x_i' rollouts, so both members must be
generated in the same batch. We emit the two members adjacently and disable
dataset shuffling in the trainer; shuffling happens here, at PAIR level, so
epochs still differ while members stay together.

AUGMENTATION, WITHIN THE n LIMIT. DAIC has 107 training participants and that
does not change. But each participant has four interviewer arms, so pairing a
participant with a different arm each epoch gives ~4x the distinct inputs
without inventing data: the label and the participant's words are untouched.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys
import numpy as np

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
E3 = f"{ROOT}/experiments/i2-e3-matched-and-prompting/results/e3_pools.json"
OUT = f"{ROOT}/experiments/i8-grpo-mitigation/data"

# The original IMHI template, byte-identical to run_a2b.py. R_format was
# dropped (see PLAN.md): evaluation teacher-forces the label and never parses,
# so the format reward served no purpose -- while ANY mention of output format
# costs 18 % of the counterfactual effect at full length and 72 % under the
# truncation RL needs (I9, logs/i9_format_ser.log). Keeping the prompt identical
# to every other experiment is what makes these numbers comparable.
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')


def make_prompt(text):
    return TEMPLATE.format(text=text)

_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
POOLS, interleave = dict(_a2b.POOLS), _a2b.interleave
POOLS.update({k: v for k, v in json.load(open(E3)).items() if k not in POOLS})
POOLS["cf_none"] = []
ARMS = ["cf_none", "cf_neutral", "cf_depression", "cf_empirical"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--budget-words", type=int, default=1200,
                    help="1200 from the length sweep: dp +0.290, the strongest "
                         "(vs +0.205 at full length); 1200 measured marginally "
                         "higher but OOMs once logits are counted, not just KV")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = [json.loads(l) for l in open(COND) if json.loads(l)["split"] == "train"]
    print(f"DAIC train participants: {len(rows)} "
          f"({sum(r['phq8_binary'] for r in rows)} positive)")

    out, pid = [], 0
    for ep in range(args.epochs):
        order = rng.permutation(len(rows))
        for k in order:
            r = rows[k]
            a, b = rng.choice(ARMS, size=2, replace=False)
            words = r["text_participant_only"].split()
            C = " ".join(words[:args.budget_words])
            for arm in (a, b):
                out.append({
                    "prompt": make_prompt(interleave(C, POOLS[arm])),
                    "pair_id": pid, "arm": arm,
                    "label": "Yes" if r["phq8_binary"] else "No",
                    "participant_id": r["participant_id"], "epoch": ep,
                })
            pid += 1

    with open(f"{OUT}/grpo_pairs.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")

    arms = {}
    for o in out:
        arms[o["arm"]] = arms.get(o["arm"], 0) + 1
    print(f"\nwrote {len(out)} rows = {pid} pairs x 2 "
          f"({args.epochs} epochs of {len(rows)} participants)")
    print(f"  arm usage: {arms}")
    print(f"  label balance: "
          f"{sum(o['label']=='Yes' for o in out)/len(out):.3f} positive")
    print(f"  pair members adjacent: "
          f"{all(out[i]['pair_id']==out[i+1]['pair_id'] for i in range(0,len(out),2))}")
    print(f"\n  -> set shuffle_dataset=False in GRPOConfig; shuffling is done "
          f"here at pair level")
    print(f"  {OUT}/grpo_pairs.jsonl")


if __name__ == "__main__":
    main()
