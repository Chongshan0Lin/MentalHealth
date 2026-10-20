#!/usr/bin/env python3
"""D / B1 analysis -- does post-training change modality reliance?

Architecture, base model and evaluation are identical across the three files;
training stage is the only variable, so a difference between them is
attributable to post-training and nothing else.

Reported per stage and per condition:

  acc / bal_acc   -- bal_acc because meld_emotion is 48 % neutral (F3), so raw
                     accuracy mostly reports how often the model says neutral
  Acc_0           -- the `question_only` arm: what survives with neither media
                     nor transcript. Above chance means the benchmark leaks.
  SS              -- Shortcut Sufficiency on balanced accuracy, chance .5:
                     (bal_acc(cond) - .5) / (bal_acc(full) - .5)
  Delta_m         -- reliance on modality m = bal_acc(full) - bal_acc(without m)

The headline contrast is how these move SFT -> GRPO -> HARPO. Benchmark
accuracy and shortcut reliance are reported side by side: accuracy climbing
while reliance also climbs is the decoupling the design was built to detect.
"""
from __future__ import annotations

import glob, json, collections
import numpy as np

ROOT = "/data/cl195/NeurIPS_workshop"
RES = f"{ROOT}/experiments/d-training-stage-reliance/results"
STAGES = ["SFT", "RL", "HARPO"]
CONDS = ["full", "audio_only", "video_only", "text_only", "question_only"]
N_BOOT = 2000
RNG = np.random.default_rng(0)


def bal_acc(gold, pred, labels):
    per = []
    for c in labels:
        m = [i for i, g in enumerate(gold) if g == c]
        if m:
            per.append(np.mean([pred[i] == c for i in m]))
    return float(np.mean(per)) if per else float("nan")


def boot(gold, pred, labels, fn, n=N_BOOT):
    idx = np.arange(len(gold))
    v = []
    for _ in range(n):
        s = RNG.integers(0, len(idx), len(idx))
        g = [gold[i] for i in s]
        p = [pred[i] for i in s]
        x = fn(g, p, labels)
        if np.isfinite(x):
            v.append(x)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    space = json.load(open(f"{ROOT}/data/processed/hba_label_spaces.json"))["meld_emotion"]
    labels = space["labels"]
    data = {}
    for st in STAGES:
        f = f"{RES}/d_b1_meld_emotion_{st}.jsonl"
        if not glob.glob(f):
            continue
        rows = [json.loads(l) for l in open(f)]
        by = collections.defaultdict(dict)
        for r in rows:
            by[r["condition"]][r["idx"]] = r
        data[st] = by

    if not data:
        print("no result files yet")
        return

    # restrict to items every stage and condition scored, so the comparison is
    # paired on exactly the same items
    common = None
    for st, by in data.items():
        for c in CONDS:
            ids = set(by.get(c, {}))
            common = ids if common is None else (common & ids)
    common = sorted(common)
    print(f"stages present: {list(data)}   items common to all: {len(common)}")
    if not common:
        return

    gold = [data[list(data)[0]]["full"][i]["gold"] for i in common]
    dist = collections.Counter(gold)
    maj = max(dist.values()) / len(gold)
    print(f"gold distribution: {dict(dist.most_common())}")
    print(f"majority = {maj:.3f}   chance = {1/len(labels):.3f}\n")

    print("=" * 108)
    print("PER-STAGE LADDER")
    print("=" * 108)
    print(f"{'stage':7s} {'condition':14s} {'acc':>7s} {'bal_acc':>8s} "
          f"{'[95% CI]':>16s} {'SS':>7s}")
    print("-" * 108)
    summary = {}
    for st in [s for s in STAGES if s in data]:
        ref = None
        for c in CONDS:
            pred = [data[st][c][i]["pred"] for i in common]
            acc = float(np.mean([p == g for p, g in zip(pred, gold)]))
            ba = bal_acc(gold, pred, labels)
            lo, hi = boot(gold, pred, labels, bal_acc)
            if c == "full":
                ref = ba
            ss = (ba - .5) / (ref - .5) if ref and abs(ref - .5) > 1e-9 else float("nan")
            summary[(st, c)] = {"acc": acc, "bal_acc": ba, "ci": (lo, hi), "ss": ss}
            print(f"{st:7s} {c:14s} {acc:7.3f} {ba:8.3f} [{lo:.3f}, {hi:.3f}] "
                  f"{ss:7.3f}")
        print("-" * 108)

    print("\n" + "=" * 108)
    print("MODALITY RELIANCE  Delta_m = bal_acc(full) - bal_acc(m withheld)")
    print("=" * 108)
    print(f"{'stage':7s} {'D_video':>9s} {'D_audio':>9s} {'D_media':>9s} "
          f"{'Acc_0':>8s} {'Acc_0 - chance':>15s}")
    print("-" * 108)
    for st in [s for s in STAGES if s in data]:
        full = summary[(st, "full")]["bal_acc"]
        dv = full - summary[(st, "audio_only")]["bal_acc"]   # video withheld
        da = full - summary[(st, "video_only")]["bal_acc"]   # audio withheld
        dm = full - summary[(st, "text_only")]["bal_acc"]    # both withheld
        a0 = summary[(st, "question_only")]["bal_acc"]
        print(f"{st:7s} {dv:9.3f} {da:9.3f} {dm:9.3f} {a0:8.3f} "
              f"{a0-1/len(labels):15.3f}")

    print("\n" + "=" * 108)
    print("ACROSS TRAINING STAGES -- does post-training raise capability and "
          "reliance together?")
    print("=" * 108)
    have = [s for s in STAGES if s in data]
    if len(have) > 1:
        print(f"{'condition':14s} " + " ".join(f"{s:>10s}" for s in have)
              + f"   {'HARPO-SFT':>10s}")
        for c in CONDS:
            vals = [summary[(s, c)]["bal_acc"] for s in have]
            d = (vals[-1] - vals[0]) if len(vals) > 1 else float("nan")
            print(f"{c:14s} " + " ".join(f"{v:10.3f}" for v in vals)
                  + f"   {d:+10.3f}")
        print()
        for name, fn in [("D_media (media reliance)",
                          lambda s: summary[(s, "full")]["bal_acc"]
                                  - summary[(s, "text_only")]["bal_acc"]),
                         ("Acc_0 (benchmark leakage)",
                          lambda s: summary[(s, "question_only")]["bal_acc"])]:
            vals = [fn(s) for s in have]
            print(f"{name:28s} " + " ".join(f"{s}={v:.3f}" for s, v in zip(have, vals))
                  + f"   Δ(HARPO−SFT) = {vals[-1]-vals[0]:+.3f}")

    out = f"{RES}/d_b1_summary.json"
    with open(out, "w") as f:
        json.dump({f"{s}|{c}": v for (s, c), v in summary.items()}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
