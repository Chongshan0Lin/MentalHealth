#!/usr/bin/env python3
"""D / B1 -- modality ablation ladder, run identically across the three
OmniSapiens training stages.

Architecture and base model are held fixed; training stage is the only
variable, which is what makes D the highest-value experiment in the design:

    SFT   HumanBehaviorAtlas/omnisapiens_sft
    GRPO  ddvd233/OmniSapiens-7B-RL
    HARPO HumanBehaviorAtlas/OmniSapiens2.0

Ladder (per item, media and placeholders kept in sync):

    full           <video> <audio> transcript
    audio_only     <audio> transcript              -- video binary withheld
    video_only     <video> transcript              -- audio binary withheld
    text_only      transcript                      -- both withheld
    question_only  neither media nor transcript    -- Acc_0, the leakage probe

Two settings inherited from earlier findings, not re-decided here:

- **336p video preset** (F4). The calibration sweep put 336p at +0.009 balanced
  accuracy against native for 6.3x throughput, while 224p cost -0.091 -- and a
  handicapped video channel manufactures apparent audio dominance, the very
  quantity B2 exists to measure. Locked for every stage and condition.
- **`joy`/`sadness` alias map** (F1). The prompt offers `happy`/`sad` while gold
  uses `joy`/`sadness` for 610 of 2610 rows; `class_label` is the authority and
  the prompt wording is a surface form.

Sampling is from the natural label distribution, not stratified. C1 showed that
per-label stratification flattens the corpus prior and thereby destroys the
asymmetry a prior-based shortcut would exploit -- accuracy per condition is only
interpretable against the real majority rate.
"""
from __future__ import annotations

import argparse, glob, json, os, random, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
import numpy as np
import pyarrow.parquet as pq

from shortcut_harness.media import load_audio, load_video, strip_placeholders
from shortcut_harness.scoring import OmniScorer

ROOT = "/data/cl195/NeurIPS_workshop"
SPACES = f"{ROOT}/data/processed/hba_label_spaces.json"
SHARDS = f"{ROOT}/data/raw/HBA-test/test-*.parquet"

CONDS = [
    ("full",          True,  True,  True),
    ("audio_only",    True,  False, True),
    ("video_only",    False, True,  True),
    ("text_only",     False, False, True),
    ("question_only", False, False, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["SFT", "RL", "HARPO"])
    ap.add_argument("--source", default="meld_emotion")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--preset", default="336p")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = {"SFT":   f"{ROOT}/checkpoints/OmniSapiens-SFT",
            "RL":    f"{ROOT}/checkpoints/OmniSapiens-RL",
            "HARPO": f"{ROOT}/checkpoints/OmniSapiens2.0"}[args.stage]
    out_path = (f"{ROOT}/experiments/d-training-stage-reliance/results/"
                f"d_b1_{args.source}_{args.stage}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    space = json.load(open(SPACES))[args.source]
    labels, surface = space["labels"], space["surface"]
    back = {s: l for l, s in zip(labels, surface)}
    print(f"{args.source}: {len(labels)} labels  majority={space['majority']} "
          f"({space['majority_rate']:.3f})  chance={space['chance']:.3f}")

    # Collect the source's rows across shards, then sample from the natural
    # distribution with a fixed seed.
    rows = []
    for s in sorted(glob.glob(SHARDS)):
        ds = pq.read_table(s, columns=["dataset"]).to_pydict()["dataset"]
        if args.source not in set(ds):
            continue
        df = pq.read_table(s).to_pandas()
        rows += [r for _, r in df[df["dataset"] == args.source].iterrows()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]
    print(f"sampled {len(rows)} rows", flush=True)

    t0 = time.time()
    sc = OmniScorer(ckpt, device=args.device)
    print(f"loaded {args.stage} in {time.time()-t0:.0f}s", flush=True)

    n, t0, skipped = 0, time.time(), 0
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            gold = r["class_label"]
            if gold not in labels:
                skipped += 1
                continue
            try:
                audio = load_audio(bytes(r["audios"][0])) if len(r["audios"]) else None
                video = (load_video(bytes(r["videos"][0]), preset=args.preset)
                         if len(r["videos"]) else None)
            except Exception as e:                    # a few HBA blobs fail to decode
                print(f"  [skip {i}] media decode: {type(e).__name__}", flush=True)
                skipped += 1
                continue

            for cond, keep_a, keep_v, keep_t in CONDS:
                prompt = strip_placeholders(r["problem"], keep_audio=keep_a,
                                            keep_video=keep_v, keep_text=keep_t)
                media = {"audio": audio if keep_a else None,
                         "video": video if keep_v else None}
                scored = sc.score(prompt, surface, media=media)
                lp = {back[k]: v["mean"] for k, v in scored.items()}
                pred = max(lp, key=lp.get)
                f.write(json.dumps({
                    "idx": i, "source": args.source, "stage": args.stage,
                    "condition": cond, "gold": gold, "pred": pred,
                    "correct": pred == gold, "logprobs": lp,
                    "preset": args.preset,
                    "has_audio": bool(len(r["audios"])),
                    "has_video": bool(len(r["videos"])),
                }) + "\n")
                n += 1
            f.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)} | {n} scored | {el:.0f}s "
                      f"({el/(i+1):.1f}s/item) | skipped {skipped}", flush=True)

    print(f"\nwrote {out_path}  ({n} rows, {skipped} skipped, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
