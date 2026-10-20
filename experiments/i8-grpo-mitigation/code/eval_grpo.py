#!/usr/bin/env python3
"""I8 -- evaluate M0 / M1 against the frozen suite.

SCORING. Every number in this project is a teacher-forced label log-likelihood,
never parsed from generation, because parse-failure rate correlates with
perturbation strength. GRPO trains the model to emit <answer>Yes</answer>, so we
keep the rule by teacher-forcing the label INSIDE the answer tag: score
"...<answer>Yes</answer>" against "...<answer>No</answer>" and take the
log-odds. That yields a continuous, parse-free score on the same scale as the
rest of the paper, so AUC is available. Parse rate is reported separately as a
diagnostic, not used for scoring.

FOUR NUMBERS, NOT TWO. The design asks for conflict accuracy and CSS. We report
both, plus two guards this project has learned it cannot do without:

  conflict AUC   -- accuracy is uninterpretable here; a constant "No" scores
                    0.702 on the conflict set against a 0.707 majority baseline
  CSS            -- the design's target (= our CF-Sens)
  construct sens -- from the frozen 200 E5 pairs; without it, a model that
                    stopped responding to anything posts a perfect CSS
  SER, AUC(IID)  -- the ratio and the utility guard
"""
from __future__ import annotations

import argparse, importlib.util, json, os, re, sys
import numpy as np
import torch
from scipy.stats import rankdata

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
E5 = f"{ROOT}/experiments/suite/results/e5_construct_pairs.jsonl"
OUT = f"{ROOT}/experiments/i8-grpo-mitigation/results"
sig = lambda x: 1 / (1 + np.exp(-x))
ANS = re.compile(r"\b(yes|no)\b", re.I)   # R_format was dropped: bare answers

_spec = importlib.util.spec_from_file_location(
    "bp", f"{ROOT}/experiments/i8-grpo-mitigation/code/build_pairs.py")
bp = importlib.util.module_from_spec(_spec); sys.modules["bp"] = bp
_spec.loader.exec_module(bp)
POOLS, interleave, make_prompt = bp.POOLS, bp.interleave, bp.make_prompt


def auc(y, s):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def score(model, tok, user, max_new=0):
    """Log-odds of Yes vs No, teacher-forced on the bare label.

    An earlier version appended a <think>...</think><answer> scaffold, left over
    from when R_format was still part of the design. That was a bug: I9 measured
    that any format scaffold suppresses the counterfactual effect (baseline CSS
    fell from 0.206 to 0.066 under it), so all four models were being compared
    in a regime where the phenomenon had largely been erased. R_format is gone,
    the models answer in MentaLLaMA's native bare style, and this now matches
    the scoring used everywhere else in the project."""
    stem = user
    base = tok(stem, return_tensors="pt").input_ids.to(model.device)
    out = []
    for lab in ["Yes", "No"]:
        full = tok(stem + " " + lab, return_tensors="pt"
                   ).input_ids.to(model.device)
        lg = model(input_ids=full).logits[0, :-1].log_softmax(-1)
        n = full.shape[1] - base.shape[1]
        out.append(float(lg[-n:].gather(1, full[0, 1:][-n:, None]).mean()))
    return out[0] - out[1]


@torch.no_grad()
def parse_rate(model, tok, users, max_new=200):
    ok = 0
    for u in users:
        ids = tok(u, return_tensors="pt").input_ids.to(model.device)
        g = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
        ok += bool(ANS.search(tok.decode(g[0, ids.shape[1]:],
                                         skip_special_tokens=True)))
    return ok / len(users)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA dir; omit for baseline")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget-words", type=int, default=1800)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = f"{ROOT}/checkpoints/{args.model}"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.bfloat16, device_map=args.device).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    print(f"{args.tag}: adapter={args.adapter or 'none (baseline)'}", flush=True)

    rows = [json.loads(l) for l in open(COND) if json.loads(l)["split"] == "test"]
    allr = [json.loads(l) for l in open(COND)]
    build = lambda r, arm: make_prompt(interleave(
        " ".join(r["text_participant_only"].split()[:args.budget_words]),
        POOLS[arm]))

    # counterfactual arms on all 188, matching every other experiment
    S = {}
    for arm in ["cf_none", "cf_neutral", "cf_depression"]:
        S[arm] = np.array([score(model, tok, build(r, arm)) for r in allr])
    y = np.array([r["phq8_binary"] for r in allr])

    css = float(np.abs(sig(S["cf_depression"]) - sig(S["cf_neutral"])).mean())
    # E2b conflict set: the arm that points AWAY from each label
    conf = np.where(y == 0, S["cf_depression"], S["cf_neutral"])
    conf_auc = auc(y, conf)
    conf_acc = float(((conf > 0).astype(int) == y).mean())
    iid_auc = auc(y, S["cf_none"])

    pairs = [json.loads(l) for l in open(E5)]
    pos = np.array([score(model, tok, make_prompt(
        " ".join(p["pos"]["text"].split()[:args.budget_words])))
        for p in pairs])
    neg = np.array([score(model, tok, make_prompt(
        " ".join(p["neg"]["text"].split()[:args.budget_words])))
        for p in pairs])
    csens = float(np.abs(sig(pos) - sig(neg)).mean())

    pr = parse_rate(model, tok, [build(r, "cf_none") for r in rows[:20]])

    res = dict(tag=args.tag, conflict_auc=conf_auc, conflict_acc=conf_acc,
               css=css, construct_sens=csens, ser=css / max(csens, 1e-9),
               iid_auc=iid_auc, p_yes=float((S["cf_none"] > 0).mean()),
               pair_acc=float((pos > neg).mean()), parse_rate=pr)
    print("\n" + "=" * 72)
    print(f"{args.tag}")
    print("=" * 72)
    for k in ["conflict_auc", "conflict_acc", "css", "construct_sens", "ser",
              "iid_auc", "p_yes", "pair_acc", "parse_rate"]:
        print(f"  {k:16s} {res[k]:.4f}")
    print("\n  baseline (untrained, from I1/I3): conflict_auc 0.329, css 0.2057,")
    print("  construct_sens 0.1154, ser 1.78, iid_auc 0.777")
    json.dump(res, open(f"{OUT}/eval_{args.tag}.json", "w"), indent=2)
    print(f"\nwrote {OUT}/eval_{args.tag}.json")


if __name__ == "__main__":
    main()
