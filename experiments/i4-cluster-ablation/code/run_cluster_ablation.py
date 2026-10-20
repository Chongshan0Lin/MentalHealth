#!/usr/bin/env python3
"""Cluster-resolved ablation -- the replacement for the dead E4a.

E4a (leave-one-cluster-out protocol shift) is not constructible from DAIC:
only 2 of 9 clusters yield a usable fold, and exposure-based splits differ in
base rate by 7x because the interview is adaptive, so protocol shift and label
shift are the same variable (see ../../suite/analysis.md).

What IS constructible is ablation at cluster resolution: strip cluster k's
prompts from every transcript and measure the change. No fold, no split, all
participants retained, coverage 93-100 % for seven of nine clusters.

This localises WHICH topical thread carries the effect. It is a sensitivity
analysis, not a generalisation test -- it does not answer RQ4, and the write-up
must not claim it does.

Pairs naturally with I1's D1/D1m result: if the damage is lexical, ablating the
clinical_care cluster should recover most of what deleting ALL interviewer
turns recovered (+0.126 AUC), while ablating biographical clusters should do
little. That is a much sharper claim than "the interviewer matters".
"""
from __future__ import annotations

import argparse, json, os, re, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
CLUST = f"{ROOT}/experiments/suite/results/e4a_prompt_clusters.json"
OUT = f"{ROOT}/experiments/i4-cluster-ablation/results"
LABELS = ["Yes", "No"]
LABEL_OF = {1: "Yes", 0: "No"}
PAT = re.compile(r"^Ellie: ([A-Za-z][A-Za-z0-9_]*) \((.*)\)\s*$")
TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')


def ablate(text_full, drop_ids):
    """Remove Ellie lines whose scripted ID is in `drop_ids`.

    Participant lines and every other Ellie line are left byte-identical, so
    the only difference between conditions is which questions are present.
    """
    out = []
    for line in (text_full or "").split("\n"):
        m = PAT.match(line)
        if m and m.group(1) in drop_ids:
            continue
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=4032)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    C = json.load(open(CLUST))
    clusters = C["clusters"]
    rows = [json.loads(l) for l in open(COND)]
    # only ID-carrying transcripts can be ablated by cluster
    rows = [r for r in rows
            if any(PAT.match(l) for l in (r["text_ellie_only"] or "").split("\n"))]
    print(f"{len(rows)} ID-carrying transcripts | {len(clusters)} clusters",
          flush=True)

    conds = [("none", set())] + [(c, set(ps)) for c, ps in clusters.items()]
    conds.append(("all_clustered", {p for ps in clusters.values() for p in ps}))

    sc = TextScorer(f"{ROOT}/checkpoints/{args.model}", device=args.device)
    print(f"loaded {args.model} | {len(conds)} conditions", flush=True)

    out_path = f"{OUT}/cluster_ablation_{args.model}.jsonl"
    n, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            gold = LABEL_OF[r["phq8_binary"]]
            for cname, drop in conds:
                text = ablate(r["text_full"], drop)
                ids = sc.tok(text, add_special_tokens=False).input_ids
                trunc = len(ids) > args.budget
                if trunc:
                    text = sc.tok.decode(ids[:args.budget])
                scored = sc.score(TEMPLATE.format(text=text), LABELS)
                lp = {k: v["mean"] for k, v in scored.items()}
                f.write(json.dumps({
                    "uid": r["uid"], "participant_id": r["participant_id"],
                    "split": r["split"], "ablated_cluster": cname,
                    "gold": gold, "phq8_binary": r["phq8_binary"],
                    "logprobs": lp, "pred": max(lp, key=lp.get),
                    "truncated": trunc, "n_tokens": min(len(ids), args.budget),
                }) + "\n")
                n += 1
            f.flush()
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)} | {n} scored | {el:.0f}s",
                      flush=True)
    print(f"\nwrote {out_path} ({n} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
