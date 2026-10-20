#!/usr/bin/env python3
"""B2 -- which modality wins when audio and video disagree?

B1 asks whether a channel is *usable* (ablate it, watch accuracy fall). That
cannot separate "the model reads video" from "the model reads audio and video
happens to correlate with it". The conflict set puts the two channels in direct
contradiction, so exactly one of them can be right, and the model's choice is a
direct read of modality dominance.

Per pair (a, b) with different labels, three items share one blanked prompt:

    reference       (A_a, V_a)   both channels agree -> accuracy sanity check
    audio_swapped   (A_b, V_a)   audio says y_b, video says y_a
    video_swapped   (A_a, V_b)   audio says y_a, video says y_b

Pooling the two swap conditions:

    R_A = P(pred = the audio's label)      R_V = P(pred = the video's label)
    D_mod = R_A - R_V                      R_other = 1 - R_A - R_V

D_mod > 0 is audio dominance. `R_other` is always reported: if it is large the
model is simply broken by conflict and D_mod describes noise, not a preference.

Two properties of the design that make D_mod readable:

- **The label prior cancels.** Across the two swap conditions the audio side
  carries {y_b} u {y_a} and the video side carries {y_a} u {y_b} -- identical
  marginals. A model answering purely from its label prior therefore scores
  R_A = R_V and D_mod = 0. This only holds for the *pooled* estimate, so the
  per-condition numbers below are diagnostics, not results: if they disagree in
  sign, the pooling assumption has failed and nothing should be read from D_mod.
- **The transcript is blanked** (locked decision D2), so the contest is strictly
  A vs V and no third channel can supply the answer.

Inherited, not re-decided here:

- **336p video preset** (F4). 224p costs 0.091 balanced accuracy, and a
  handicapped video channel manufactures apparent *audio* dominance -- the very
  quantity this experiment measures. Locked across every model and condition.
- **`strip=True` on video decode.** HBA MP4s carry their own AAC track. If it
  reaches the model, a swapped-audio item holds two competing audio sources and
  R_A/R_V stop meaning anything. This matters far more here than in B1.
- **Scoring against `surface`** (F1). Gold uses `joy`/`sadness`, the prompt
  offers `happy`/`sad`; the conflict set is already built in surface forms.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
import numpy as np

from shortcut_harness.media import load_audio, load_video
from shortcut_harness.scoring import OmniScorer

ROOT = "/data/cl195/NeurIPS_workshop"
DATA = f"{ROOT}/experiments/b2-modality-conflict/data"
OUT = f"{ROOT}/experiments/b2-modality-conflict/results"

CKPT = {"SFT":       f"{ROOT}/checkpoints/OmniSapiens-SFT",
        "RL":        f"{ROOT}/checkpoints/OmniSapiens-RL",
        "HARPO":     f"{ROOT}/checkpoints/OmniSapiens2.0",
        "HumanOmni": f"{ROOT}/checkpoints/HumanOmniV2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CKPT))
    ap.add_argument("--preset", default="336p")      # F4, do not lower
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)  # pairs; 0 = all
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    meta = json.load(open(f"{DATA}/conflict_meta.json"))
    labels, items = meta["labels"], meta["items"]
    blobs = np.load(f"{DATA}/conflict_media.npz")
    pairs = sorted({it["pair"] for it in items})
    if args.limit:
        pairs = pairs[:args.limit]
    by_pair = {p: [it for it in items if it["pair"] == p] for p in pairs}
    print(f"{len(pairs)} pairs x 3 conditions, preset {args.preset}", flush=True)

    t0 = time.time()
    sc = OmniScorer(CKPT[args.model], device=args.device)
    print(f"loaded {args.model} in {time.time()-t0:.0f}s", flush=True)

    out_path = f"{OUT}/conflict_{args.model}.jsonl"
    n, skipped, t0 = 0, 0, time.time()
    with open(out_path, "w") as f:
        for pi, p in enumerate(pairs):
            # Decode once per distinct blob: the three conditions reuse the same
            # four sources (a/b x audio/video), so caching on content hash cuts
            # video decoding -- by far the dominant cost -- from 6 to 4 per pair.
            cache = {}

            def get(key, kind):
                raw = blobs[key].tobytes()
                h = hashlib.md5(raw).hexdigest()
                if h not in cache:
                    cache[h] = (load_audio(raw) if kind == "a"
                                else load_video(raw, preset=args.preset))
                return cache[h]

            try:
                media = {it["cond"]: {"audio": get(it["key"] + "_a", "a"),
                                      "video": get(it["key"] + "_v", "v")}
                         for it in by_pair[p]}
            except Exception as e:
                # Drop the whole pair, never a single condition: a pair missing
                # one arm would bias R_A against R_V.
                print(f"  [skip pair {p}] media decode: {type(e).__name__}: {e}",
                      flush=True)
                skipped += 1
                continue

            for it in by_pair[p]:
                scored = sc.score(it["problem"], labels, media=media[it["cond"]])
                lp = {k: v["mean"] for k, v in scored.items()}
                pred = max(lp, key=lp.get)
                f.write(json.dumps({
                    "pair": p, "cond": it["cond"], "pred": pred,
                    "audio_label": it["audio_label"],
                    "video_label": it["video_label"], "gold": it["gold"],
                    "match_audio": pred == it["audio_label"],
                    "match_video": pred == it["video_label"],
                    "logprobs": lp, "preset": args.preset,
                    "model": args.model}) + "\n")
                n += 1
            f.flush()
            if (pi + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  {pi+1}/{len(pairs)} pairs | {n} scored | {el:.0f}s "
                      f"({el/(pi+1):.1f}s/pair) | skipped {skipped}", flush=True)

    print(f"\nwrote {out_path}  ({n} rows, {skipped} pairs skipped, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
