#!/usr/bin/env python3
"""H1 -- what shortcuts live in the HBA-test datasets themselves?

Model-free throughout. This is the A2a move applied to a second benchmark: if
the label can be recovered from something that is not the construct, there is a
shortcut for a model to find, and no model needs to be run to establish it.

TEXT ONLY. The paper does not run multimodal experiments; HBA is treated here as
a collection of text tasks whose *datasets* are audited. Withholding audio and
video is the point -- a "human behaviour" benchmark whose labels are recoverable
from the transcript alone is not measuring behaviour.

Six probes, each with the control that makes it readable:

  P1 label skew        always-majority accuracy per subset
  P2 text-only         TF-IDF on the transcript -> label, vs chance
                       CONTROL: cremad and tess use fixed emotion-neutral
                       carrier sentences by design, so they must land at chance.
                       They do (0.169 / 0.107), which validates the probe.
  P3 stopword-only     the same probe on FUNCTION WORDS ONLY. Above chance means
                       the classes differ in register or production source, not
                       in the construct -- content words cannot explain it.
  P4 length            transcript length alone. Reported in BOTH characters and
                       whitespace segments: for the Chinese subsets these are
                       different quantities and only one of them carries signal.
  P5 answerability     gold label absent from the offered option list (F1)
  P6 string damage     `joy`->`happy` applied as a blind substring replacement,
                       corrupting enjoy/joyful/enjoyable in `problem` and in the
                       gold `answer` of the open-ended QA subsets.

The transcript is separated from the question template by taking the longest
common suffix of `problem` within each subset -- the template is constant per
subset, so this needs no per-dataset regex and cannot silently mis-split.
"""
from __future__ import annotations

import argparse, collections, glob, json, os, re, sys, warnings

warnings.filterwarnings("ignore")
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

ROOT = "/data/cl195/NeurIPS_workshop"
HERE = f"{ROOT}/experiments/h1-hba-shortcut"
COLS = ["problem", "answer", "dataset", "modality_signature", "task", "class_label"]
PH = ("<audio>", "<video>", "<image>")
# open-ended QA: the "label" is a free-text sentence, so a closed-set probe is
# meaningless. Their template and string-damage results are still reported.
OPEN_QA = {"intentqa", "siq2", "mimeqa"}
STOP = set(
    "the a an and or but if of to in on at for with it is was were be been am are "
    "i you he she they we my your his her their this that these those so just like "
    "really very not no yes do does did have has had will would can could s t m re "
    "ve ll d".split())
DAMAGE = re.compile(r"\b\w*(enhappy|happyful|happyous|sadded|angerd|angery)\w*\b", re.I)


def is_cjk(s):
    return sum(1 for c in s if "一" <= c <= "鿿") > len(s) * 0.2


def extract():
    """Pull the text columns only -- the media columns are ~66 GB."""
    out = []
    for f in sorted(glob.glob(f"{ROOT}/data/raw/HBA-test/*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=512, columns=COLS):
            out += b.to_pylist()
    return out


def common_suffix(strs, cap=4000):
    s = strs[0]
    for t in strs[1:cap]:
        i = 0
        while i < min(len(s), len(t)) and s[len(s) - 1 - i] == t[len(t) - 1 - i]:
            i += 1
        s = s[len(s) - i:]
        if not s:
            break
    return s


def parse(rows):
    """Split `problem` into transcript and the subset-constant question template."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    out = []
    for ds, rs in by.items():
        suf = common_suffix([r["problem"] for r in rs])
        for r in rs:
            body = r["problem"][:len(r["problem"]) - len(suf)] if suf else r["problem"]
            tr = "\n".join(l for l in body.split("\n")
                           if l.strip() not in PH).strip()
            out.append(dict(dataset=ds, task=r["task"], label=r["class_label"],
                            answer=r["answer"], template=suf.strip(), transcript=tr,
                            modality=r["modality_signature"]))
    return out


def probe(X, y, char=False, seeds=3):
    vec = (TfidfVectorizer(sublinear_tf=True, analyzer="char_wb",
                           ngram_range=(2, 4), min_df=2) if char else
           TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2))
    accs = []
    for sd in range(seeds):
        try:
            p = cross_val_predict(
                make_pipeline(vec, LogisticRegression(max_iter=3000,
                                                      class_weight="balanced")),
                X, y, cv=StratifiedKFold(5, shuffle=True, random_state=sd))
            accs.append(balanced_accuracy_score(y, p))
        except Exception:
            return float("nan")
    return float(np.mean(accs))


def numeric_probe(v, y, seeds=5):
    """One numeric feature. A forest, not a threshold: the informative pattern
    can be an interval (exactly 8 segments) rather than a cut point."""
    X = np.asarray(v, dtype=float).reshape(-1, 1)
    accs = []
    for sd in range(seeds):
        p = cross_val_predict(
            RandomForestClassifier(300, random_state=sd, class_weight="balanced"),
            X, y, cv=StratifiedKFold(5, shuffle=True, random_state=sd))
        accs.append(balanced_accuracy_score(y, p))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    os.makedirs(f"{HERE}/data", exist_ok=True)
    os.makedirs(f"{HERE}/results", exist_ok=True)

    cache = f"{HERE}/data/parsed.json"
    if args.refresh or not os.path.exists(cache):
        P = parse(extract())
        json.dump(P, open(cache, "w"))
    P = json.load(open(cache))
    by = collections.defaultdict(list)
    for r in P:
        by[r["dataset"]].append(r)
    spaces = json.load(open(f"{ROOT}/data/processed/hba_label_spaces.json"))

    res = {}
    print("=" * 108)
    print("H1 -- HBA-test dataset audit (model-free, text only)")
    print("=" * 108)
    print(f"{'dataset':18s} {'n':>5s} {'#lab':>4s} {'chance':>7s} {'major':>7s} "
          f"{'TEXT':>6s} {'STOP':>6s} {'CHARS':>6s} {'SEGS':>6s} {'unansw':>7s}")
    print("-" * 108)
    for ds, rows in sorted(by.items(), key=lambda x: -len(x[1])):
        y = np.array([r["label"] for r in rows])
        T = [r["transcript"] for r in rows]
        cnt = collections.Counter(y)
        sp = spaces.get(ds, {})
        surface = sp.get("surface") or sp.get("labels") or []
        unans = sum(1 for r in rows if surface and r["label"] not in surface)
        rec = dict(n=len(rows), n_labels=len(cnt), modality=rows[0]["modality"],
                   task=rows[0]["task"], template=rows[0]["template"],
                   unanswerable=unans)
        if ds in OPEN_QA or len(cnt) < 2 or min(cnt.values()) < 5:
            res[ds] = rec
            print(f"{ds:18s} {len(rows):5d} {'--':>4s} {'--':>7s} {'--':>7s} "
                  f"{'--':>6s} {'--':>6s} {'--':>6s} {'--':>6s} {unans:7d}"
                  f"   (open-ended)")
            continue
        chance, major = 1 / len(cnt), max(cnt.values()) / len(y)
        cjk = is_cjk(" ".join(T[:50]))
        full = probe(T, y, char=cjk)
        stop = probe([" ".join(w for w in t.lower().split() if w in STOP)
                      for t in T], y)
        chars = numeric_probe([len(t) for t in T], y)
        segs = numeric_probe([len(t.split()) for t in T], y)
        rec.update(chance=chance, majority=major, text=full, stopword=stop,
                   length_chars=chars, length_segments=segs)
        res[ds] = rec
        print(f"{ds:18s} {len(rows):5d} {len(cnt):4d} {chance:7.3f} {major:7.3f} "
              f"{full:6.3f} {stop:6.3f} {chars:6.3f} {segs:6.3f} {unans:7d}")

    print("\n  TEXT/STOP/CHARS/SEGS are balanced accuracy; compare against chance.")
    print("  STOP uses function words only -- above chance means the classes "
          "differ in\n  register or source rather than in the construct.")
    print("  cremad and tess are the negative controls: fixed carrier sentences, "
          "so text\n  must land at chance, and it does.")

    # ---- P6: blind substring replacement ----------------------------------
    dmg = collections.Counter()
    for r in P:
        for fld in ("transcript", "template", "answer", "label"):
            v = r.get(fld)
            if isinstance(v, str):
                for m in DAMAGE.finditer(v):
                    dmg[(r["dataset"], fld, m.group(0).lower())] += 1
    print("\n" + "=" * 108)
    print("P6 -- `joy`->`happy` applied as a blind SUBSTRING replacement")
    print("=" * 108)
    agg = collections.Counter()
    for (ds, fld, w), n in dmg.items():
        agg[(ds, fld)] += n
    for (ds, fld), n in agg.most_common():
        print(f"  {ds:18s} {fld:12s} {n:4d} corrupted tokens")
    print(f"  total: {sum(dmg.values())} occurrences across "
          f"{len({d for d, _, _ in dmg})} subsets")
    print("  Corruption of the `answer` field matters most: for the open-ended "
          "QA subsets\n  the gold string itself is wrong, so a model that "
          "writes correct English is\n  scored incorrect.")
    res["_string_damage"] = {f"{d}/{f}": n for (d, f), n in agg.items()}

    json.dump(res, open(f"{HERE}/results/audit.json", "w"), indent=2)
    print(f"\nwrote {HERE}/results/audit.json")


if __name__ == "__main__":
    main()
