#!/usr/bin/env python3
"""A2b -- Ellie-prompt counterfactual: does interviewer context move the
prediction when participant evidence is held byte-identical?

Four arms. C (the participant's own turns) is truncated once and reused
verbatim in all of them, so every difference is attributable to the interviewer
lines inserted around it.

  cf_none        C alone -- the anchor
  cf_neutral     C + 5 biographical scripted prompts        (CONFIRMATORY)
  cf_depression  C + 5 clinically loaded scripted prompts   (CONFIRMATORY)
  cf_empirical   C + the 5 prompts A2a found most           (EXPLORATORY --
                 label-associated in this corpus            chosen after seeing
                                                            A2a, flagged as such)

The two confirmatory pools were fixed in protocol.md before A2a ran; the
empirical pool could only be built afterwards, so it is labelled exploratory
and reported separately rather than pooled with the pre-registered arms.

Interviewer lines are interleaved at even intervals among the participant turns
rather than stacked as a preamble: five questions in a row is not a shape any
interview transcript takes, and the models were tuned on natural text. The
insertion positions are computed from the participant turn count alone, so they
are identical across arms -- only the inserted strings differ.
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
LABELS = ["Yes", "No"]
LABEL_OF = {1: "Yes", 0: "No"}
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')

# Scripted prompts, verbatim from the corpus (F5). Pools 1 and 2 were fixed in
# protocol.md on semantic grounds before any association test was run.
POOLS = {
    "cf_depression": [          # CONFIRMATORY -- clinically loaded
        "have you been diagnosed with depression",
        "have you ever been diagnosed with p_t_s_d",
        "how have you been feeling lately",
        "how easy is it for you to get a good night's sleep",
        "how are you at controlling your temper",
    ],
    "cf_neutral": [             # CONFIRMATORY -- biographical
        "where are you from originally",
        "what did you study at school",
        "i'm sure you can tell by my shoes i'm not much of a world explorer do you travel a lot",
        "what are some things you really like about l_a",
        "what are some things you don't really like about l_a",
    ],
    "cf_empirical": [           # EXPLORATORY -- top A2a associations
        "what got you to seek help",              # OR 11.26
        "do you still go to therapy now",         # OR 10.42
        "are they triggered by something",        # OR  8.50
        "how long ago were you diagnosed",        # OR  8.24
        "when was the last time that happened",   # OR  4.00
    ],
}
ARMS = ["cf_none", "cf_neutral", "cf_depression", "cf_empirical"]


def interleave(participant_text: str, prompts: list[str]) -> str:
    """Insert `prompts` at even intervals among the participant turn lines.

    Positions depend only on the number of participant turns, so they are the
    same in every arm; C's bytes are never touched.
    """
    lines = [l for l in participant_text.split("\n") if l.strip()]
    if not prompts:
        return "\n".join(lines)
    step = max(1, len(lines) // (len(prompts) + 1))
    out, pi = [], 0
    for i, l in enumerate(lines):
        if pi < len(prompts) and i == pi * step:
            out.append(f"Ellie: {prompts[pi]}")
            pi += 1
        out.append(l)
    out.extend(f"Ellie: {p}" for p in prompts[pi:])   # any that did not fit
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=int, default=4032)
    args = ap.parse_args()

    ckpt = args.ckpt or f"{ROOT}/checkpoints/{args.model}"
    out_path = args.out or (f"{ROOT}/experiments/a2-ellie-counterfactual/"
                            f"results/a2b_{args.model}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = [json.loads(l) for l in open(COND)]
    t0 = time.time()
    sc = TextScorer(ckpt, device=args.device, adapter=args.adapter)
    print(f"loaded {args.model} in {time.time()-t0:.0f}s", flush=True)

    # C is truncated ONCE per participant and reused across arms. Truncating
    # per-arm would let the interviewer lines push participant text past the
    # budget, so the arms would differ in how much of C the model saw -- the
    # counterfactual would then measure truncation, not interviewer context.
    # The reserve leaves room for the longest pool.
    longest = max(len(" ".join(p)) for p in POOLS.values())
    budget = args.budget - (longest // 3) - 32

    n, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            gold = LABEL_OF[r["phq8_binary"]]
            ids = sc.tok(r["text_participant_only"],
                         add_special_tokens=False).input_ids
            trunc = len(ids) > budget
            C = sc.tok.decode(ids[:budget]) if trunc else r["text_participant_only"]

            for arm in ARMS:
                text = interleave(C, POOLS.get(arm, []))
                scored = sc.score(TEMPLATE.format(text=text), LABELS)
                lp = {k: v["mean"] for k, v in scored.items()}
                pred = max(lp, key=lp.get)
                f.write(json.dumps({
                    "uid": r["uid"], "participant_id": r["participant_id"],
                    "split": r["split"], "model": args.model, "arm": arm,
                    "gold": gold, "phq8_binary": r["phq8_binary"],
                    "phq8_score": r["phq8_score"],
                    "has_prompt_ids": bool("(" in (r["text_ellie_only"] or "")),
                    "logprobs": lp, "pred": pred, "correct": pred == gold,
                    "c_truncated": trunc, "c_tokens": min(len(ids), budget),
                }) + "\n")
                n += 1
            f.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)} | {n} scored | {el:.0f}s "
                      f"({el/(i+1):.1f}s/participant)", flush=True)

    print(f"\nwrote {out_path}  ({n} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
