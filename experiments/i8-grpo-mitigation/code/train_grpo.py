#!/usr/bin/env python3
"""I8 -- GRPO training, arms M0 and M1.

  M0   R_accuracy                  diagnostic: does vanilla RLVR amplify
                                   the shortcut?
  M1   R_accuracy + lambda*R_inv   mitigation

R_format was dropped by decision after I9: evaluation teacher-forces the label
and never parses, so a format reward buys nothing, while any format instruction
in the prompt costs 18 % of the counterfactual effect at full length and 72 %
under the truncation RL requires. Both arms use the original IMHI template
verbatim, so these numbers sit on the same scale as the rest of the paper.

Identical data, backbone, hyperparameters and seed between arms; only the reward
set differs, so any difference is attributable to R_inv.

THE MONITOR IS NOT OPTIONAL. R_inv rewards agreement between a participant's two
counterfactual versions, and a model that answers one class to everything agrees
with itself perfectly. With DAIC's base rate that degenerate policy collects
R_format 1.00 + R_accuracy 0.72 + R_inv 1.00 = 2.72 of 3.00 -- 91 % of the
maximum for learning nothing. This project has already measured that exact
failure three times (chain-of-thought, product-of-experts, and a small version
under LEACE), so we log P(pred=Yes) every step and pre-commit: if it exceeds
0.90 at convergence, R_inv has been hacked and M1's headline numbers are void
regardless of how good CSS looks. `--reward-inv correct` runs the fallback
M1' in which R_inv requires the pair to be consistent AND right.
"""
from __future__ import annotations

import argparse, json, os, re, collections

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True")
import numpy as np
import torch

ROOT = "/data/cl195/NeurIPS_workshop"
DATA = f"{ROOT}/experiments/i8-grpo-mitigation/data/grpo_pairs.jsonl"
OUT = f"{ROOT}/experiments/i8-grpo-mitigation/results"

# R_format dropped, so the model answers in MentaLLaMA's native IMHI style
# ("Yes, the poster suffers from...") rather than in tags. Take the first
# standalone yes/no token as the prediction.
ANS = re.compile(r"\b(yes|no)\b", re.I)
_seen = collections.defaultdict(list)          # step -> predictions, for the monitor


def _text(c):
    return c if isinstance(c, str) else c[-1]["content"]


def pred_of(c):
    m = ANS.search(_text(c))
    return m.group(1).capitalize() if m else None


# ------------------------------------------------------------------ rewards

def r_accuracy(completions, label, **kw):
    return [1.0 if pred_of(c) == y else 0.0 for c, y in zip(completions, label)]


def make_r_accuracy_balanced(p_pos):
    """Inverse-frequency weighted accuracy.

    M0 showed that plain accuracy is enough on its own to collapse the policy:
    28 % of DAIC labels are positive, so answering "No" to everything scores
    0.72 without learning anything, and GRPO found that optimum by step ~60.
    Weighting each class by 0.5 / p(class) caps the constant policy at 0.50
    while a perfect policy still scores 1.00, so the degenerate solution stops
    being attractive."""
    w = {"Yes": 0.5 / p_pos, "No": 0.5 / (1 - p_pos)}

    def r(completions, label, **kw):
        return [w[y] if pred_of(c) == y else 0.0
                for c, y in zip(completions, label)]
    r.__name__ = "r_accuracy_balanced"
    return r


def make_r_inv(mode, lam, log_path):
    """Pair-level invariance reward, per the design: take the majority
    prediction of each arm's rollouts and award the whole pair if they agree.

    `mode='consistent'` is the reward as drafted. `mode='correct'` is the
    fallback that closes the constant-prediction exploit by additionally
    requiring the agreed answer to be right."""

    def r_inv(completions, pair_id, label, arm, **kw):
        preds = [pred_of(c) for c in completions]
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for p, pid_, a in zip(preds, pair_id, arm):
            by[pid_][a].append(p)

        # monitor: which class is the policy emitting, and how often is it parseable
        step = kw.get("trainer_state").global_step if kw.get("trainer_state") else -1
        ok = [p for p in preds if p is not None]
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": step,
                "p_yes": float(np.mean([p == "Yes" for p in ok])) if ok else None,
                "parse_rate": float(len(ok) / max(len(preds), 1)),
                "n": len(preds)}) + "\n")

        award = {}
        for pid_, arms in by.items():
            if len(arms) < 2:
                award[pid_] = 0.0            # pair split across batches
                continue
            maj = []
            for a, ps in arms.items():
                ps = [p for p in ps if p]
                maj.append(collections.Counter(ps).most_common(1)[0][0] if ps
                           else None)
            consistent = maj[0] is not None and maj[0] == maj[1]
            if mode == "correct":
                y = next(l for l, q in zip(label, pair_id) if q == pid_)
                award[pid_] = lam if (consistent and maj[0] == y) else 0.0
            else:
                award[pid_] = lam if consistent else 0.0
        return [award[p] for p in pair_id]

    r_inv.__name__ = f"r_inv_{mode}"
    return r_inv


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["M0", "M1"], required=True)
    ap.add_argument("--reward-inv", choices=["consistent", "correct"],
                    default="consistent", help="'correct' = the M1' fallback")
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-rewards", default="group",
                    choices=["group", "batch", "none"],
                    help="'group' (TRL default) divides advantages by the "
                         "within-group std, which cancels any class weighting "
                         "AND zeroes the gradient for groups the policy gets "
                         "uniformly wrong -- so a collapsed policy stops "
                         "receiving signal from the minority class and locks "
                         "in. 'batch' keeps those groups' advantage negative.")
    ap.add_argument("--balanced", action="store_true",
                    help="use inverse-frequency weighted accuracy (arm B); "
                         "plain accuracy alone collapsed M0 to constant No")
    ap.add_argument("--max-completion", type=int, default=64,
                    help="the IMHI-style answer opens with Yes/No, so "
                         "64 tokens is ample and the logits tensor "
                         "(seqs x len x 32k vocab) is what actually OOMs")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = (args.arm + ("" if args.reward_inv == "consistent" else "prime")
           + ("_bal" if args.balanced else "")
           + ("" if args.scale_rewards == "group" else f"_{args.scale_rewards}"))

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    rows = [json.loads(l) for l in open(DATA)]
    ds = Dataset.from_list(rows)
    print(f"{tag}: {len(ds)} rows ({len(ds)//2} pairs), "
          f"reward_inv={args.reward_inv if args.arm=='M1' else 'OFF'}")

    mon = f"{OUT}/monitor_{tag}.jsonl"
    open(mon, "w").close()
    if args.balanced:
        p_pos = sum(r["label"] == "Yes" for r in rows) / len(rows)
        funcs = [make_r_accuracy_balanced(p_pos)]
        print(f"  balanced accuracy: p(pos)={p_pos:.3f}, "
              f"weights Yes={0.5/p_pos:.2f} No={0.5/(1-p_pos):.2f}; "
              f"a constant policy now caps at 0.50")
    else:
        funcs = [r_accuracy]
    if args.arm == "M1":
        funcs.append(make_r_inv(args.reward_inv, args.lam, mon))
    else:
        # M0 gets the monitor too, so the degeneracy check is comparable
        m = make_r_inv("consistent", 0.0, mon); m.__name__ = "monitor_only"
        funcs.append(m)

    cfg = GRPOConfig(
        output_dir=f"{OUT}/ckpt_{tag}",
        num_generations=args.num_generations,
        # Both members of a pair must be GENERATED together for R_inv, but the
        # forward/backward can be chunked: 2 x 8 accumulation = 16 = one pair's
        # worth of rollouts. Transcripts run ~2.4k tokens, so 16 sequences in
        # one backward OOMs a 48 GB card even with LoRA and checkpointing.
        generation_batch_size=args.num_generations * 2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=args.num_generations,
        scale_rewards=args.scale_rewards,
        shuffle_dataset=False,          # shuffling was done at pair level
        beta=0.0,                       # no KL penalty, per the design
        learning_rate=args.lr,
        num_train_epochs=1,             # epochs are baked into the pair file
        max_completion_length=args.max_completion,
        # The checkpoint's config declares torch_dtype=float32, so TRL loads
        # the 7B model as 28 GB unless told otherwise -- which is why the OOM
        # footprint did not move when we halved the sequence length. Every other
        # experiment in this project loads bf16 explicitly; match that.
        model_init_kwargs={"dtype": "bfloat16"},
        bf16=True, gradient_checkpointing=True,
        log_completions=True, logging_steps=1,
        save_strategy="epoch", report_to=[], seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=f"{ROOT}/checkpoints/{args.model}",
        reward_funcs=funcs, args=cfg, train_dataset=ds,
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]),
    )
    trainer.train()
    trainer.save_model(f"{OUT}/ckpt_{tag}/final")

    # ---- the pre-registered degeneracy check -------------------------------
    m = [json.loads(l) for l in open(mon)]
    tail = [x for x in m[-max(1, len(m)//5):] if x["p_yes"] is not None]
    p_yes = float(np.mean([x["p_yes"] for x in tail])) if tail else float("nan")
    parse = float(np.mean([x["parse_rate"] for x in m[-max(1, len(m)//5):]]))
    print("\n" + "=" * 70)
    print(f"{tag} DEGENERACY CHECK (final fifth of training)")
    print("=" * 70)
    print(f"  P(pred = Yes) = {p_yes:.3f}      parse rate = {parse:.3f}")
    if not np.isnan(p_yes) and (p_yes > 0.90 or p_yes < 0.10):
        print("  *** VOID -- policy collapsed to a near-constant answer.")
        if args.arm == "M1":
            print("      R_inv rewards agreement, and a constant policy agrees")
            print("      with itself perfectly -- the exploit flagged in PLAN.md.")
            print("      Fallback: --reward-inv correct")
        else:
            print("      NOTE: this arm has no R_inv. The collapse comes from")
            print("      plain accuracy alone: 72 % of labels are No, so a")
            print("      constant No scores 0.72 for learning nothing.")
            print("      Fallback: --balanced")
        print("      Headline numbers for this arm are not interpretable.")
    else:
        print("  OK -- policy is not degenerate; headline numbers stand.")
    json.dump({"arm": tag, "p_yes_final": p_yes, "parse_rate": parse},
              open(f"{OUT}/degeneracy_{tag}.json", "w"), indent=2)
    print(f"\nsaved: {OUT}/ckpt_{tag}/final")


if __name__ == "__main__":
    main()
