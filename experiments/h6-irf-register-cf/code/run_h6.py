#!/usr/bin/env python3
"""H6 -- do IMHI-native models use the `Irf` register shortcut?

Fourth dataset, and the purest register shortcut in the audit. H2 probed `Irf`
model-free:

    text (TF-IDF)      0.643   (chance 0.500)
    stopwords only     0.622   -- 97% of the full-text signal

Deleting every content word costs almost nothing: the classes differ in *how the
post is written*, not in a removable vocabulary. That ratio is the highest in
either audit (`mosei_emotion` is next at 92%).

Two things make this the right next test after H5:

- **Different corpus family.** DAIC, `mmpsy` and `ptsd_in_the_wild` are all HBA
  or clinical-interview data. `Irf` is IMHI, and MentaLLaMA and Menta were
  instruction-tuned on IMHI -- so this asks whether a model uses a shortcut in
  data it was actually trained on, which none of the earlier counterfactuals do.
- **Same-source classes.** In `ptsd_in_the_wild` the two classes came from
  different production sources entirely (veteran testimony vs TED talks), so
  "register" and "topic" were confounded. Here both classes are Reddit
  mental-health posts, so the contrast is closer to register alone.

## Arms

Each post's own text is byte-identical in every arm; only the prepended block
differs, and the three pools are token-matched to each other.

    native       the post as collected
    ctx_neutral  + a block from MultiWD                     <- the zero point
    ctx_no       + a block from Irf `no` posts
    ctx_yes      + a block from Irf `yes` posts

`ctx_neutral` is the control H5 did not have. MultiWD is the one dataset in
either audit whose text carries **no** label signal (H2: 0.499 against chance
0.500), so a MultiWD prefix adds length and mental-health topic without adding
class evidence. If `ctx_yes` moves the prediction much further than
`ctx_neutral` does, the effect is the register content and not merely the
presence of a prefix.

Pool source items are excluded from evaluation, so no post is ever scored
against a context drawn from itself.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
MANIFEST = f"{ROOT}/data/processed/text_manifest.jsonl"
LABELS = ["yes", "no"]
SURFACE = {"yes": "Yes", "no": "No"}
TEMPLATE = 'Consider this post: "{text}" {question}'
POOL_ITEMS, POOL_WORDS = 3, 150
ARMS = ["native", "ctx_neutral", "ctx_no", "ctx_yes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--budget", type=int, default=900,
                    help="post word budget, applied identically per arm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = args.ckpt or f"{ROOT}/checkpoints/{args.model}"
    out = (f"{ROOT}/experiments/h6-irf-register-cf/results/"
           f"h6_irf_{args.model}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = [json.loads(l) for l in open(MANIFEST)]
    irf = [r for r in rows if r["source_dataset"] == "Irf" and r.get("label")]
    mwd = [r for r in rows if r["source_dataset"] == "MultiWD" and r.get("text")]
    rng = random.Random(args.seed)

    # Pools: the longest posts per class, so a fixed-length slice stays
    # register-typical rather than being one whole short post.
    pools, pool_uids = {}, set()
    def build(cands, key):
        cands = sorted(cands, key=lambda r: -len(r["text"].split()))[:POOL_ITEMS]
        chunks = []
        for r in cands:
            pool_uids.add(r["uid"])
            chunks.append(" ".join(r["text"].split()[:POOL_WORDS // POOL_ITEMS]))
        pools[key] = " ".join(chunks)
    build([r for r in irf if r["label"] == "yes"], "ctx_yes")
    build([r for r in irf if r["label"] == "no"], "ctx_no")
    build(mwd, "ctx_neutral")

    n_tok = min(len(p.split()) for p in pools.values())
    pools = {k: " ".join(v.split()[:n_tok]) for k, v in pools.items()}
    print(f"pools token-matched at {n_tok} words; "
          f"{len(pool_uids)} source posts excluded", flush=True)

    pool_set = set(pool_uids)
    evalrows = [r for r in irf if r["uid"] not in pool_set]
    rng.shuffle(evalrows)
    evalrows = evalrows[:args.n]
    lab = {}
    for r in evalrows:
        lab[r["label"]] = lab.get(r["label"], 0) + 1
    print(f"evaluating {len(evalrows)} posts  {lab}", flush=True)

    surf = [SURFACE[l] for l in LABELS]
    back = {SURFACE[l]: l for l in LABELS}

    t0 = time.time()
    sc = TextScorer(ckpt, device=args.device, adapter=args.adapter)
    print(f"loaded {args.model} in {time.time()-t0:.0f}s", flush=True)

    ndone, t0 = 0, time.time()
    with open(out, "w") as f:
        for k, r in enumerate(evalrows):
            # truncate the post ONCE and reuse verbatim across arms
            w = r["text"].split()
            trunc = len(w) > args.budget
            own = " ".join(w[:args.budget])

            for arm in ARMS:
                text = own if arm == "native" else pools[arm] + " " + own
                scored = sc.score(TEMPLATE.format(text=text,
                                                  question=r["question"]), surf)
                lp = {back[kk]: v["mean"] for kk, v in scored.items()}
                pred = max(lp, key=lp.get)
                f.write(json.dumps({
                    "uid": r["uid"], "model": args.model, "arm": arm,
                    "question": r["question"], "gold": r["label"],
                    "pred": pred, "correct": pred == r["label"], "logprobs": lp,
                    "own_words": len(own.split()), "truncated": trunc,
                }) + "\n")
                ndone += 1
            f.flush()
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(evalrows)} | {ndone} scored | {el:.0f}s "
                      f"({el/(k+1):.2f}s/post)", flush=True)

    print(f"\nwrote {out}  ({ndone} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
