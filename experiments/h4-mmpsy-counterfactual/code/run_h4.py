#!/usr/bin/env python3
"""H4 -- do models USE the mmpsy segment-count shortcut?

H1 established that in `mmpsy_depression` the transcript content carries nothing
(text-only balanced acc 0.507 against chance 0.500) while **how many answer
segments were recorded** carries a great deal (0.772). The depressed group has
one more recorded response than the control group:

    no_depression   7 segments: 115/156   (74%)
    depression      8 segments:  22/35    (63%)

That is availability. This experiment asks the A2 question -- does any model
actually *use* it? -- with a manipulation that is unusually clean for a
structural shortcut.

**The intervention touches whitespace only.** A segment is a whitespace-
separated chunk (`len(text.split())`, the exact feature H1 probed). To add a
segment we split one chunk at an existing punctuation mark; to remove one we
join two adjacent chunks. Every content character is preserved in both
directions -- only the count of whitespace boundaries changes. Nothing about
what the participant said differs between arms.

Arms:

    native      as collected
    seg_plus    one extra segment  (a chunk split at punctuation)
    seg_minus   one fewer segment  (two chunks joined)

`seg_plus` pushes the 7-segment items onto the depressed group's modal count and
`seg_minus` pulls the 8-segment items onto the control group's. If the model
reads the protocol artifact, p(depression) rises under `seg_plus` and falls
under `seg_minus`, with the participant's words untouched.

Run across the three OmniSapiens training stages, since HBA is their own
benchmark and D already showed post-training changes what these checkpoints do
with a content-free input.
"""
from __future__ import annotations

import argparse, glob, json, os, re, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
import pyarrow.parquet as pq
from shortcut_harness.scoring import OmniScorer

ROOT = "/data/cl195/NeurIPS_workshop"
TPL_MARK = "The above is an audio recording"
# The prompt offers "depression, no depression"; `class_label` uses the
# underscored form. Score the surface the prompt actually presents.
SURFACE = {"depression": "depression", "no_depression": "no depression"}
SPLIT_AT = re.compile(r"[，,。！!？?；;]")


def segments(t):
    return t.split()


def seg_plus(segs):
    """Split the longest chunk at an interior punctuation mark. Content chars
    preserved; one whitespace boundary added."""
    order = sorted(range(len(segs)), key=lambda i: -len(segs[i]))
    for i in order:
        s = segs[i]
        cuts = [m.end() for m in SPLIT_AT.finditer(s) if 2 <= m.end() <= len(s) - 2]
        if cuts:
            c = cuts[len(cuts) // 2]
            return segs[:i] + [s[:c], s[c:]] + segs[i + 1:]
    return None


def seg_minus(segs):
    """Join the two shortest adjacent chunks. Content chars preserved; one
    whitespace boundary removed."""
    if len(segs) < 2:
        return None
    i = min(range(len(segs) - 1), key=lambda j: len(segs[j]) + len(segs[j + 1]))
    return segs[:i] + [segs[i] + segs[i + 1]] + segs[i + 2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["SFT", "RL", "HARPO"])
    ap.add_argument("--subset", default="mmpsy_depression")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ckpt = {"SFT": f"{ROOT}/checkpoints/OmniSapiens-SFT",
            "RL": f"{ROOT}/checkpoints/OmniSapiens-RL",
            "HARPO": f"{ROOT}/checkpoints/OmniSapiens2.0"}[args.stage]
    out = (f"{ROOT}/experiments/h4-mmpsy-counterfactual/results/"
           f"h4_{args.subset}_{args.stage}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = []
    for s in sorted(glob.glob(f"{ROOT}/data/raw/HBA-test/test-*.parquet")):
        ds = pq.read_table(s, columns=["dataset"]).to_pydict()["dataset"]
        if args.subset in set(ds):
            df = pq.read_table(s).to_pandas()
            rows += [r for _, r in df[df["dataset"] == args.subset].iterrows()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{args.subset}: {len(rows)} rows", flush=True)

    labels = ["depression", "no_depression"]
    surf = [SURFACE[l] for l in labels]
    back = {SURFACE[l]: l for l in labels}

    t0 = time.time()
    sc = OmniScorer(ckpt, device=args.device)
    print(f"loaded {args.stage} in {time.time()-t0:.0f}s", flush=True)

    n, t0, skipped = 0, time.time(), 0
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            problem = r["problem"]
            if TPL_MARK not in problem:
                skipped += 1
                continue
            head, tail = problem.split(TPL_MARK, 1)
            transcript, template = head.strip(), TPL_MARK + tail
            segs = segments(transcript)

            arms = {"native": segs,
                    "seg_plus": seg_plus(list(segs)),
                    "seg_minus": seg_minus(list(segs))}
            if any(v is None for v in arms.values()):
                skipped += 1
                continue

            for arm, sv in arms.items():
                text = " ".join(sv)
                prompt = f"{text}\n{template}"
                scored = sc.score(prompt, surf, media=None)
                lp = {back[k]: v["mean"] for k, v in scored.items()}
                pred = max(lp, key=lp.get)
                f.write(json.dumps({
                    "idx": i, "subset": args.subset, "stage": args.stage,
                    "arm": arm, "gold": r["class_label"], "pred": pred,
                    "correct": pred == r["class_label"], "logprobs": lp,
                    "n_segments": len(sv),
                    "n_segments_native": len(segs),
                    "n_chars": len(text.replace(" ", "")),
                }, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)} | {n} scored | {el:.0f}s "
                      f"({el/(i+1):.2f}s/item) | skipped {skipped}", flush=True)

    print(f"\nwrote {out}  ({n} rows, {skipped} skipped, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
