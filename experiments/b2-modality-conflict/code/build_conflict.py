#!/usr/bin/env python3
"""B2 -- build the audio/video conflict set.

The core experiment of the original design (EXPERIMENT_DESIGN.md B2), unblocked
on 2026-08-20 and never built. Ablation asks whether a channel is *usable*; the
conflict set asks which channel *wins* when two channels disagree, which is the
only way to measure modality dominance.

Construction. Take sample i (label y_i) and sample j (label y_j != y_i) from
meld_emotion, and emit three conditions:

    (A_i, V_i)   reference
    (A_j, V_i)   audio swapped in from a different-label sample
    (A_i, V_j)   video swapped in

Then R_A = P(prediction matches the audio's label), R_V likewise for video, and
D_mod = R_A - R_V. `R_other` (matching neither) is reported always: a high
R_other means conflict simply breaks the model rather than revealing dominance.

LOCKED DECISION D2 -- the transcript is blanked in every conflict condition.
The transcript line inside `problem` carries lexical emotion cues, so leaving it
in makes the contest three-way and D_mod uninterpretable. A text-vs-A-vs-V arm
is a legitimate extension but is reported separately, never pooled.

F1 applies here. meld_emotion's gold labels use `joy`/`sadness` while its own
option list offers `happy`/`sad` -- 610 of 2610 rows, 23 %, whose correct answer
cannot be selected. The alias map in hba_label_spaces.json is applied so the
conflict labels are drawn from the answerable space.
"""
from __future__ import annotations

import argparse, collections, glob, json, os, random, re
import numpy as np
import pyarrow.parquet as pq

ROOT = "/data/cl195/NeurIPS_workshop"
OUT = f"{ROOT}/experiments/b2-modality-conflict/data"
SRC = "meld_emotion"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SRC)
    ap.add_argument("--n-pairs", type=int, default=300)   # D3: 300-500/condition
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(args.seed)

    idx = json.load(open(f"{ROOT}/experiments/b2-modality-conflict/av_index.json"))
    rows_wanted = {tuple(x) for x in idx[args.source]}
    files = sorted(glob.glob(f"{ROOT}/data/raw/HBA-test/*.parquet"))
    ls = json.load(open(f"{ROOT}/data/processed/hba_label_spaces.json"))[args.source]
    # `labels` is the GOLD vocabulary (joy, sadness); `surface` is what the
    # prompt actually offers (happy, sad); `alias` maps gold -> surface. The
    # answerable space is `surface` -- scoring against `labels` is exactly the
    # F1 defect (610/2610 rows whose right answer is not on the menu).
    alias = ls.get("alias", {}) if isinstance(ls, dict) else {}
    labels = ls.get("surface") or ls.get("labels") if isinstance(ls, dict) else ls

    print(f"{args.source}: {len(rows_wanted)} rows with audio+video")
    print(f"  answerable label space: {labels}")
    if alias:
        print(f"  F1 alias map applied: {alias}")

    # ---- pull the rows we need, keeping blobs -----------------------------
    keep = collections.defaultdict(list)
    for fi, f in enumerate(files):
        want = sorted(r for (g, r) in rows_wanted if g == fi)
        if not want:
            continue
        pf = pq.ParquetFile(f); off = 0
        for b in pf.iter_batches(batch_size=256):
            hit = [r for r in want if off <= r < off + b.num_rows]
            if hit:
                d = b.to_pylist()
                for r in hit:
                    x = d[r - off]
                    lab = alias.get(x["class_label"], x["class_label"])
                    if lab not in labels:
                        continue                       # unanswerable, F1
                    keep[lab].append({
                        "file": fi, "row": r, "label": lab,
                        "problem": x["problem"],
                        "audio": x["audios"][0], "video": x["videos"][0]})
            off += b.num_rows
        print(f"  scanned {os.path.basename(f)} "
              f"({sum(len(v) for v in keep.values())} kept)", flush=True)

    print(f"\nusable rows by label: "
          f"{ {k: len(v) for k, v in sorted(keep.items())} }")

    # ---- pair samples with DIFFERENT labels --------------------------------
    pool = [x for v in keep.values() for x in v]
    rng.shuffle(pool)
    pairs, used = [], set()
    for a in pool:
        if len(pairs) >= args.n_pairs:
            break
        if id(a) in used:
            continue
        cand = [b for b in pool if b["label"] != a["label"] and id(b) not in used]
        if not cand:
            continue
        b = rng.choice(cand)
        used.add(id(a)); used.add(id(b))
        pairs.append((a, b))

    # Blank the transcript: keep the modality placeholders and the question, drop
    # everything in between. The question cannot be found from the tail -- meld's
    # option list is spread over one line per option ("anger, \ndisgust, ...
    # \nsad"), so the last line is an option, not the question. Anchor on the
    # question stem instead and keep everything from there on.
    STEM = "How would you describe"

    def blank(problem):
        lines = problem.split("\n")
        head = [l for l in lines if l.strip() in ("<audio>", "<video>")]
        q = next(i for i, l in enumerate(lines) if l.startswith(STEM))
        out = "\n".join(head + lines[q:])
        for lab in labels:                      # the menu must survive intact
            assert lab in out, f"option {lab} lost while blanking"
        return out

    meta, blobs = [], {}
    for k, (a, b) in enumerate(pairs):
        for cond, au, vi, al, vl in [
            ("reference",     a, a, a["label"], a["label"]),
            ("audio_swapped", b, a, b["label"], a["label"]),
            ("video_swapped", a, b, a["label"], b["label"])]:
            key = f"{k}_{cond}"
            blobs[key + "_a"] = au["audio"]; blobs[key + "_v"] = vi["video"]
            meta.append(dict(pair=k, cond=cond, key=key,
                             audio_label=al, video_label=vl,
                             gold=a["label"] if cond == "reference" else None,
                             problem=blank(a["problem"])))
    np.savez(f"{OUT}/conflict_media.npz",
             **{k: np.frombuffer(v, dtype=np.uint8) for k, v in blobs.items()})
    json.dump({"labels": labels, "source": args.source, "items": meta},
              open(f"{OUT}/conflict_meta.json", "w"), indent=2)
    sz = os.path.getsize(f"{OUT}/conflict_media.npz") / 2**20
    print(f"\nbuilt {len(pairs)} pairs x 3 conditions = {len(meta)} items "
          f"({sz:.0f} MB of media)")
    print(f"  label pairs are always discordant: "
          f"{all(a['label'] != b['label'] for a, b in pairs)}")
    print(f"  wrote {OUT}/conflict_meta.json")
    print(f"\nsample blanked problem:\n  {meta[0]['problem']!r}")


if __name__ == "__main__":
    main()
