#!/usr/bin/env python3
"""Pick the video preset for Experiments B and D.

Downsampling video is a confound, not a speed knob: shrinking the video channel
handicaps it, and in a conflict experiment that manufactures apparent *audio
dominance* -- the very quantity B2 measures. So the preset has to be chosen on
evidence and then frozen across every condition and model.

Protocol: run the **video-only** condition (the channel at risk) over one fixed
sample at each preset, and compare each preset against `native`:

  balanced accuracy   does the weaker channel still carry its signal?
  agreement w/ native does the model still make the same call per item?
  mean gold logprob   sensitive to degradation that accuracy rounds away
  tokens / seconds    what the fidelity costs

Choose the smallest preset whose video-only accuracy and agreement have not yet
departed from native. Report the whole curve, not just the winner.
"""
import argparse, json, sys, time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/data/cl195/NeurIPS_workshop/shortcut_harness")
from media import VIDEO_PRESETS, load_audio, load_video, strip_placeholders  # noqa: E402
from scoring import OmniScorer, predict  # noqa: E402

ROOT = "/data/cl195/NeurIPS_workshop"
SHARD = f"{ROOT}/data/raw/HBA-test/test-00005-of-00060.parquet"


def balanced_acc(gold, pred, labels):
    per = []
    for l in labels:
        idx = [i for i, g in enumerate(gold) if g == l]
        if idx:
            per.append(sum(pred[i] == l for i in idx) / len(idx))
    return float(np.mean(per)) if per else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--model", default="OmniSapiens2.0")
    ap.add_argument("--presets", default="native,half,336p,224p,224p_sparse")
    ap.add_argument("--out", default=f"{ROOT}/data/processed/video_preset_calibration.json")
    args = ap.parse_args()

    spaces = json.load(open(f"{ROOT}/data/processed/hba_label_spaces.json"))
    L = spaces["meld_emotion"]["labels"]
    alias = spaces["meld_emotion"]["alias"]
    surface = [alias.get(l, l) for l in L]

    t = pq.ParquetFile(SHARD).read().to_pydict()
    # Restrict to clips `native` can actually be scored on. Native costs ~371
    # tokens/frame, so 34% of MELD clips exceed the 32768 position limit --
    # transformers only warns and then produces garbage, which would silently
    # poison the reference arm. See scripts/probe_native_feasible.py.
    feas = json.load(open(f"{ROOT}/data/processed/calib_rows_native_feasible.json"))
    rows = feas["rows"][: args.n]
    print(f"restricted to native-feasible clips (<= {feas['margin']} tokens): "
          f"{len(feas['rows'])} available, using {len(rows)}")
    gold = [t["class_label"][i] for i in rows]
    print(f"sample: n={len(rows)} meld_emotion rows | gold dist: "
          f"{dict(Counter(gold).most_common())}")
    print(f"label space (surface forms): {surface}\n")

    sc = OmniScorer(f"{ROOT}/checkpoints/{args.model}")
    s2g = {s: g for s, g in zip(surface, L)}

    results, native_pred = {}, None
    for preset in args.presets.split(","):
        preds, golds, gold_lp, ntok, t0 = [], [], [], [], time.time()
        for i, r in enumerate(rows):
            # keep_text=False is load-bearing: leaving the transcript in makes
            # this a video+text condition, and the text carries enough of the
            # emotion label to mask exactly the video degradation we are trying
            # to detect.
            prompt = strip_placeholders(t["problem"][r], keep_audio=False,
                                        keep_video=True, keep_text=False)
            media = {"audio": None,
                     "video": load_video(bytes(t["videos"][r][0]), preset=preset)}
            scored = sc.score(prompt, surface, media=media)
            y, p = predict(scored)
            preds.append(s2g[y])
            golds.append(t["class_label"][r])
            gold_lp.append(scored[alias.get(golds[-1], golds[-1])]["mean"])
            if i == 0:
                ntok.append(sc._pack(sc._build(prompt, media),
                                     media)["input_ids"].shape[1])
        el = time.time() - t0
        acc = sum(p == g for p, g in zip(preds, golds)) / len(preds)
        bacc = balanced_acc(golds, preds, L)
        agree = (1.0 if native_pred is None
                 else sum(a == b for a, b in zip(preds, native_pred)) / len(preds))
        if native_pred is None:
            native_pred = preds
        results[preset] = {
            "accuracy": round(acc, 4), "balanced_accuracy": round(bacc, 4),
            "agreement_with_native": round(agree, 4),
            "mean_gold_logprob": round(float(np.mean(gold_lp)), 4),
            "input_tokens_item0": ntok[0], "seconds": round(el, 1),
            "sec_per_item": round(el / len(rows), 2),
            "video_shape": list(VIDEO_PRESETS[preset]),
            "pred_dist": dict(Counter(preds).most_common()),
        }
        print("%-12s acc=%.3f  bal_acc=%.3f  agree_native=%.3f  gold_lp=%+.3f  "
              "tok=%5d  %5.1fs (%.2fs/item)"
              % (preset, acc, bacc, agree, np.mean(gold_lp), ntok[0], el,
                 el / len(rows)))

    json.dump({"model": args.model, "n": len(rows), "source": "meld_emotion",
               "condition": "video_only", "results": results},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")

    base = results[args.presets.split(",")[0]]
    print("\nrelative to native:")
    for p, r in results.items():
        print("  %-12s Δbal_acc=%+.3f  Δgold_lp=%+.3f  agree=%.3f  speedup=%.1fx"
              % (p, r["balanced_accuracy"] - base["balanced_accuracy"],
                 r["mean_gold_logprob"] - base["mean_gold_logprob"],
                 r["agreement_with_native"], base["seconds"] / r["seconds"]))


if __name__ == "__main__":
    main()
