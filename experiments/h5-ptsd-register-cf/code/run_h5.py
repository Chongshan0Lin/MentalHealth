#!/usr/bin/env python3
"""H5 -- do models use the `ptsd_in_the_wild` register shortcut?

Third shortcut mechanism, after DAIC (lexical: which questions were asked) and
mmpsy (structural: how many answer segments were recorded). Here the shortcut is
**register / production source**: H1 found text-only balanced accuracy 0.948 on
this subset, and 0.880 *with every content word deleted* -- the classes differ in
how people talk, not in a removable word list. Scrubbing all PTSD and military
vocabulary raised accuracy (0.948 -> 0.953).

The design constraint has to be stated rather than worked around: **construct and
register are perfectly confounded in this corpus.** The positives are veteran
testimony and clinical scenes; the negatives are TED-style talks and lifestyle
content. There is no PTSD-positive vlog and no PTSD-negative testimony, so
register cannot be swapped *within* an item -- the item is its register.

What is available is the A2 move: hold the item's own transcript byte-identical
and prepend a fixed block carrying the other class's register.

    native          the transcript as collected
    ctx_vlog        + a fixed vlog/talk-register passage
    ctx_testimony   + a fixed testimony-register passage

If the model reads register, p(ptsd) rises under `ctx_testimony` relative to
`ctx_vlog` while not one word of the item's own transcript has changed. H1 showed
the register signal is front-loaded (first 25 words alone give 0.719), so a
prepended block sits exactly where that signal lives.

Two controls built in:

- **The two pools are token-matched** to each other, so the primary contrast is
  length-controlled by construction and cannot be read as "more text".
- **Pool source items are excluded from evaluation**, so no item is ever scored
  against a context drawn from itself.

Text-only. The video channel is confounded by register too (talk footage vs
clinical footage), so including it would muddy exactly the thing being isolated;
placeholders are stripped identically in every arm, so that choice cannot produce
the contrast.
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
import pyarrow.parquet as pq
from shortcut_harness.media import strip_placeholders
from shortcut_harness.scoring import OmniScorer

ROOT = "/data/cl195/NeurIPS_workshop"
MARK = "Above is a video"
SHARDS = [0, 1, 2, 3]
COLS = ["dataset", "problem", "answer", "class_label"]
SURFACE = {"ptsd": "ptsd", "no_ptsd": "no ptsd"}
POOL_PER_CLASS, POOL_WORDS = 2, 120


def load():
    rows = []
    for i in SHARDS:
        t = pq.read_table(f"{ROOT}/data/raw/HBA-test/test-{i:05d}-of-00060.parquet",
                          columns=COLS).to_pandas()
        rows += [r for _, r in t[t["dataset"] == "ptsd_in_the_wild"].iterrows()]
    return rows


def transcript_of(problem):
    head = problem.split(MARK)[0]
    # drop the <video>/<audio> placeholder lines; they are re-added by the
    # prompt builder so every arm carries the same markers
    return "\n".join(l for l in head.split("\n")
                     if l.strip() not in ("<video>", "<audio>")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["SFT", "RL", "HARPO"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=2400,
                    help="transcript word budget, applied identically per arm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = {"SFT": f"{ROOT}/checkpoints/OmniSapiens-SFT",
            "RL": f"{ROOT}/checkpoints/OmniSapiens-RL",
            "HARPO": f"{ROOT}/checkpoints/OmniSapiens2.0"}[args.stage]
    out = (f"{ROOT}/experiments/h5-ptsd-register-cf/results/"
           f"h5_ptsd_{args.stage}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = load()
    template = MARK + rows[0]["problem"].split(MARK, 1)[1]

    # Pool items: the longest per class, so a 120-word block is a small slice of
    # them and stays register-typical rather than being one whole short clip.
    pools, pool_ids = {}, set()
    for lab, key in [("no_ptsd", "ctx_vlog"), ("ptsd", "ctx_testimony")]:
        cand = sorted([(i, r) for i, r in enumerate(rows)
                       if r["class_label"] == lab],
                      key=lambda x: -len(transcript_of(x[1]["problem"]).split()))
        chunks = []
        for i, r in cand[:POOL_PER_CLASS]:
            pool_ids.add(i)
            chunks.append(" ".join(transcript_of(r["problem"]).split()
                                   [:POOL_WORDS // POOL_PER_CLASS]))
        pools[key] = " ".join(chunks)

    # token-match the two pools to each other by word count
    n = min(len(p.split()) for p in pools.values())
    pools = {k: " ".join(v.split()[:n]) for k, v in pools.items()}
    print(f"pools token-matched at {n} words; pool source items excluded: "
          f"{sorted(pool_ids)}", flush=True)

    eval_rows = [(i, r) for i, r in enumerate(rows) if i not in pool_ids]
    print(f"evaluating {len(eval_rows)} items "
          f"({sum(1 for _, r in eval_rows if r['class_label']=='ptsd')} positive)",
          flush=True)

    labels = ["ptsd", "no_ptsd"]
    surf = [SURFACE[l] for l in labels]
    back = {SURFACE[l]: l for l in labels}
    q = strip_placeholders(template, keep_audio=False, keep_video=False,
                           keep_text=True)

    t0 = time.time()
    sc = OmniScorer(ckpt, device=args.device)
    print(f"loaded {args.stage} in {time.time()-t0:.0f}s", flush=True)

    ndone, t0 = 0, time.time()
    with open(out, "w") as f:
        for k, (idx, r) in enumerate(eval_rows):
            # truncate ONCE and reuse verbatim, so the arms cannot differ in how
            # much of the item's own transcript the model saw
            words = transcript_of(r["problem"]).split()
            trunc = len(words) > args.budget
            own = " ".join(words[:args.budget])

            arms = {"native": own,
                    "ctx_vlog": pools["ctx_vlog"] + " " + own,
                    "ctx_testimony": pools["ctx_testimony"] + " " + own}

            for arm, text in arms.items():
                scored = sc.score(f"{text}\n{q}", surf, media=None)
                lp = {back[kk]: v["mean"] for kk, v in scored.items()}
                pred = max(lp, key=lp.get)
                f.write(json.dumps({
                    "idx": idx, "stage": args.stage, "arm": arm,
                    "gold": r["class_label"], "pred": pred,
                    "correct": pred == r["class_label"], "logprobs": lp,
                    "own_words": len(own.split()), "truncated": trunc,
                }) + "\n")
                ndone += 1
            f.flush()
            if (k + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(eval_rows)} | {ndone} scored | {el:.0f}s "
                      f"({el/(k+1):.1f}s/item)", flush=True)

    print(f"\nwrote {out}  ({ndone} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
