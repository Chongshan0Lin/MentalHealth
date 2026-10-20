#!/usr/bin/env python3
"""B2 analysis -- modality dominance, and whether it agrees with ablation.

Reports, per model:

    ref_acc     accuracy on the (A_a, V_a) reference items. A guard, not a
                result: the transcript is blanked, so if this sits at chance the
                model cannot do the task from A+V at all and D_mod is noise.
    R_A, R_V    P(pred = audio's label) / P(pred = video's label), pooled over
                the two swap conditions so the label prior cancels.
    R_other     1 - R_A - R_V. Large values mean conflict breaks the model
                rather than revealing a preference.
    D_mod       R_A - R_V. Positive = audio dominance.

Cluster bootstrap over pairs (both swap items of a pair move together; they
share the two source clips and are not independent).

Two null checks are printed rather than assumed:

  - **prior-only null.** A model answering from its label prior alone must score
    D_mod = 0, because pooling the two swap conditions gives the audio and video
    sides identical label marginals. We verify this on the actual pairing by
    replaying each model's own modal label as a constant predictor.
  - **per-condition split.** D_mod computed inside audio_swapped alone and
    video_swapped alone. Pooling is what makes the prior cancel, so these two
    are diagnostics: agreeing signs support the pooled number, opposite signs
    mean the pooled estimate is averaging away a condition artifact.

Finally, B1's ablation ladder is loaded for the same three training stages and
placed beside D_mod. Ablation measures whether a channel is *usable*; conflict
measures which channel *wins*. They are free to disagree, and a disagreement is
the reportable finding, not a bug.
"""
from __future__ import annotations

import collections, glob, json, os, sys

import numpy as np

ROOT = "/data/cl195/NeurIPS_workshop"
RES = f"{ROOT}/experiments/b2-modality-conflict/results"
B1 = f"{ROOT}/experiments/d-training-stage-reliance/results"
ORDER = ["SFT", "RL", "HARPO", "HumanOmni"]
NBOOT = 2000
rng = np.random.default_rng(0)


def rates(rows):
    """R_A, R_V, R_other over the swap items given."""
    if not rows:
        return (np.nan,) * 3
    a = np.mean([r["match_audio"] for r in rows])
    v = np.mean([r["match_video"] for r in rows])
    return a, v, 1 - a - v


def boot_dmod(by_pair):
    """Cluster bootstrap of D_mod, resampling whole pairs."""
    keys = list(by_pair)
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        take = rng.choice(len(keys), len(keys), replace=True)
        rows = [r for i in take for r in by_pair[keys[i]]]
        a, v, _ = rates(rows)
        out[b] = a - v
    return np.percentile(out, [2.5, 97.5])


def b1_ladder(stage):
    """audio_only / video_only accuracy from the B1 ablation ladder."""
    p = f"{B1}/d_b1_meld_emotion_{stage}.jsonl"
    if not os.path.exists(p):
        return None
    acc = collections.defaultdict(list)
    for l in open(p):
        d = json.loads(l)
        acc[d["condition"]].append(d["correct"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    models = [m for m in ORDER
              if os.path.exists(f"{RES}/conflict_{m}.jsonl")]
    if not models:
        sys.exit("no results yet")

    summary = {}
    print("=" * 100)
    print("B2 -- modality conflict: which channel wins when audio and video "
          "disagree?")
    print("=" * 100)
    print(f"{'model':10s} {'pairs':>6s} {'ref_acc':>8s} {'R_A':>7s} {'R_V':>7s} "
          f"{'R_other':>8s} {'D_mod':>8s} {'95% CI':>16s}")
    print("-" * 100)

    for m in models:
        rows = [json.loads(l) for l in open(f"{RES}/conflict_{m}.jsonl")]
        ref = [r for r in rows if r["cond"] == "reference"]
        swap = [r for r in rows if r["cond"] != "reference"]
        by_pair = collections.defaultdict(list)
        for r in swap:
            by_pair[r["pair"]].append(r)

        ref_acc = float(np.mean([r["pred"] == r["gold"] for r in ref]))
        a, v, o = rates(swap)
        lo, hi = boot_dmod(by_pair)
        summary[m] = dict(n_pairs=len(by_pair), ref_acc=ref_acc,
                          R_A=a, R_V=v, R_other=o, D_mod=a - v,
                          ci=[float(lo), float(hi)])
        print(f"{m:10s} {len(by_pair):6d} {ref_acc:8.3f} {a:7.3f} {v:7.3f} "
              f"{o:8.3f} {a-v:+8.3f}   [{lo:+.3f},{hi:+.3f}]")

    print(f"\nchance under a uniform guess over 7 labels: R_A = R_V = 0.143, "
          f"D_mod = 0.000")

    # ---- diagnostics -------------------------------------------------------
    print("\n" + "=" * 100)
    print("DIAGNOSTICS -- these decide whether D_mod above is readable")
    print("=" * 100)
    for m in models:
        rows = [json.loads(l) for l in open(f"{RES}/conflict_{m}.jsonl")]
        swap = [r for r in rows if r["cond"] != "reference"]
        print(f"\n{m}")

        # per-condition split: signs must agree
        d = {}
        for c in ("audio_swapped", "video_swapped"):
            a, v, o = rates([r for r in swap if r["cond"] == c])
            d[c] = a - v
            print(f"  {c:14s} R_A {a:.3f}  R_V {v:.3f}  D {a-v:+.3f}")
        agree = np.sign(d["audio_swapped"]) == np.sign(d["video_swapped"])
        print(f"  per-condition signs agree: {agree}"
              + ("" if agree else "   <-- pooled D_mod is NOT interpretable"))

        # prior-only null: replay the model's modal answer as a constant
        modal = collections.Counter(r["pred"] for r in swap).most_common(1)[0][0]
        ca = np.mean([r["audio_label"] == modal for r in swap])
        cv = np.mean([r["video_label"] == modal for r in swap])
        print(f"  prior-only null (always '{modal}'): D {ca-cv:+.3f} "
              f"(design predicts 0.000)")

        # non-neutral subset: 'neutral' is 48% of the corpus and models carry a
        # strong prior on it, so the contest is cleanest when neither side is it
        nn = [r for r in swap
              if "neutral" not in (r["audio_label"], r["video_label"])]
        a, v, o = rates(nn)
        print(f"  neither side neutral (n={len(nn)}): R_A {a:.3f}  R_V {v:.3f}  "
              f"D {a-v:+.3f}  R_other {o:.3f}")

    # ---- does conflict agree with ablation? --------------------------------
    print("\n" + "=" * 100)
    print("B1 (is the channel USABLE?) vs B2 (which channel WINS?)")
    print("=" * 100)
    print(f"{'stage':10s} {'B1 audio_only':>14s} {'B1 video_only':>14s} "
          f"{'B1 says':>10s} {'B2 D_mod':>10s} {'B2 says':>10s} {'agree':>7s}")
    print("-" * 100)
    for m in models:
        lad = b1_ladder(m)
        if lad is None or m not in summary:
            continue
        ao, vo = lad.get("audio_only"), lad.get("video_only")
        b1_says = "audio" if ao > vo else "video"
        b2_says = "audio" if summary[m]["D_mod"] > 0 else "video"
        print(f"{m:10s} {ao:14.3f} {vo:14.3f} {b1_says:>10s} "
              f"{summary[m]['D_mod']:+10.3f} {b2_says:>10s} "
              f"{str(b1_says == b2_says):>7s}")
        summary[m]["b1"] = lad

    json.dump(summary, open(f"{RES}/conflict_summary.json", "w"), indent=2)
    print(f"\nwrote {RES}/conflict_summary.json")


if __name__ == "__main__":
    main()
