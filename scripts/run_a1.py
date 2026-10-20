#!/usr/bin/env python3
"""Experiment A1 -- DAIC-WOZ ablation (the positive control).

Scores every participant under each text condition by teacher-forced label
log-likelihood over {Yes, No}; nothing is parsed out of free generation.

Conditions
  full                every turn, both speakers                 -> C + S
  participant_only    participant turns only                    -> C
  ellie_only          interviewer turns only                    -> S   *the shortcut*
  ellie_masked        Ellie turns replaced by [Q], order kept   -> S minus lexical content
  empty               no transcript at all                      -> Acc_null, and the PMI null

F5 requires the prompt-ID strip: 123 of 188 transcripts annotate Ellie's turns
with the scripted identifier (`depression_diagnosed (have you been ...)`), which
is an annotation artifact far more explicit than speech. `full` and `ellie_only`
are therefore run twice, stripped and unstripped. `participant_only` carries no
IDs (verified: 0 of 188) and `ellie_masked` discards Ellie's text entirely, so
neither has a stripped variant.

Truncation is the text analogue of D1: MentaLLaMA's context is 4096 tokens while
the median `full` transcript is ~2.1k words, so a budget is unavoidable. One
budget is applied identically to every condition and the head is kept (matching
D1's "first 300 s"). Conditions therefore cover unequal shares of the interview
-- that is not silently equalised; covered words/turns and a truncation flag are
logged per row so coverage can be carried as a covariate.
"""
from __future__ import annotations

import argparse, json, re, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"

LABELS = ["Yes", "No"]          # first token of the IMHI gold response
LABEL_OF = {1: "Yes", 0: "No"}

# MentaLLaMA's native IMHI template, verbatim from repos/MentaLLaMA/test_data.
# Kept unchanged rather than reworded for the interview setting: staying in the
# distribution the model was tuned on matters more here than surface fit, and
# the identical template is used for every condition, so the ablation contrast
# is unaffected by the wording.
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')

# `Ellie: some_prompt_id (what she actually said)` -> `Ellie: what she actually said`
PROMPT_ID = re.compile(r'^(Ellie): [A-Za-z][A-Za-z0-9_]* \((.*)\)\s*$')


def strip_prompt_ids(text: str) -> str:
    out = []
    for line in text.split("\n"):
        m = PROMPT_ID.match(line)
        out.append(f"{m.group(1)}: {m.group(2)}" if m else line)
    return "\n".join(out)


def conditions_for(row: dict) -> dict[str, str]:
    """-> {condition: transcript text}. `empty` is added by the caller."""
    return {
        "full":                row["text_full"],
        "full_stripped":       strip_prompt_ids(row["text_full"]),
        "participant_only":    row["text_participant_only"],
        "ellie_only":          row["text_ellie_only"] or "",
        "ellie_only_stripped": strip_prompt_ids(row["text_ellie_only"] or ""),
        "ellie_masked":        row["text_ellie_masked"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reserve", type=int, default=64,
                    help="tokens held back from the context for template+label")
    ap.add_argument("--budget", type=int, default=0,
                    help="override the transcript token budget. Set this to the "
                         "budget of the shortest-context model in the comparison "
                         "so truncation is not a cross-model confound.")
    args = ap.parse_args()

    ckpt = args.ckpt or f"{ROOT}/checkpoints/{args.model}"
    out_path = args.out or f"{ROOT}/runs/a1_{args.model}.jsonl"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = [json.loads(l) for l in open(COND)]
    if args.limit:
        rows = rows[:args.limit]

    t0 = time.time()
    sc = TextScorer(ckpt, device=args.device, adapter=args.adapter)
    ctx = sc.model.config.max_position_embeddings
    budget = args.budget or (ctx - args.reserve)
    print(f"loaded {args.model} in {time.time()-t0:.0f}s | ctx={ctx} | "
          f"transcript budget={budget} tok", flush=True)

    def truncate(text):
        """Keep the head, within `budget` tokens. -> (text, n_tok, truncated)"""
        ids = sc.tok(text, add_special_tokens=False).input_ids
        if len(ids) <= budget:
            return text, len(ids), False
        return sc.tok.decode(ids[:budget]), budget, True

    null_prompt = TEMPLATE.format(text="")
    null_scored = sc.score(null_prompt, LABELS)
    print("null (Acc_0) logprobs:",
          {k: round(v["mean"], 4) for k, v in null_scored.items()}, flush=True)

    n_done, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            gold = LABEL_OF[r["phq8_binary"]]
            has_ids = bool(re.search(r"^Ellie: [A-Za-z][A-Za-z0-9_]* \(",
                                     r["text_full"], flags=re.M))
            conds = conditions_for(r)
            conds["empty"] = ""

            for cond, text in conds.items():
                if cond == "empty":
                    scored, ntok, trunc = null_scored, 0, False
                else:
                    text, ntok, trunc = truncate(text)
                    scored = sc.score(TEMPLATE.format(text=text), LABELS)
                f.write(json.dumps({
                    "uid": r["uid"], "participant_id": r["participant_id"],
                    "split": r["split"], "model": args.model,
                    "condition": cond,
                    "gold": gold, "phq8_binary": r["phq8_binary"],
                    "phq8_score": r["phq8_score"], "gender": r["gender"],
                    "has_prompt_ids": has_ids,
                    "logprobs": {k: {"sum": v["sum"], "mean": v["mean"],
                                     "n_tok": v["n_tok"]}
                                 for k, v in scored.items()},
                    "null_logprobs": {k: {"sum": v["sum"], "mean": v["mean"]}
                                      for k, v in null_scored.items()},
                    # coverage, per D1: reported, never silently equalised
                    "n_tok": ntok, "truncated": trunc,
                    "n_words": len(text.split()),
                    "n_turns": len([l for l in text.split("\n") if l.strip()]),
                }) + "\n")
                n_done += 1
            f.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)} participants | {n_done} scored | "
                      f"{el:.0f}s ({el/(i+1):.1f}s/participant)", flush=True)

    print(f"\nwrote {out_path}  ({n_done} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
