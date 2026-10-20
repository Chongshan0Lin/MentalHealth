#!/usr/bin/env python3
"""H3 -- ESConv, the decisive negative control for the DAIC question-selection
shortcut.

ESConv is the direct structural analogue of A2a: a two-party emotional-support
conversation in which the supporter's move is annotated with one of 8 strategies,
so the 8 strategy labels play the role of Ellie's 75 prompt IDs. If "which move
the helper chose" predicts the seeker's label, the DAIC shortcut generalises. If
it sits at chance, the DAIC shortcut needs something ESConv does not have --
fixed wording, so that SELECTION is the only degree of freedom.

Probe family and metric are held identical to `experiments/h1-hba-shortcut/code/
audit.py` and `experiments/h2-imhi-shortcut/code/audit.py` so the numbers are
comparable across corpora: balanced accuracy, TF-IDF(sublinear, 1-2 grams) +
LogisticRegression(class_weight="balanced"), 5-fold StratifiedKFold, 3 seeds.

The stopword-only arm is the common ruler used throughout the project: delete
every content word, keep function words. Above chance means the classes differ in
register or production source rather than in the construct.

Written 2026-09-05 to close the gap flagged in `paper/FACTS_FOR_WRITING.md:175`
("ESConv numbers ... are not yet in a committed script"). The numbers this
produces supersede any previously quoted ESConv values.
"""
from __future__ import annotations

import argparse, collections, json, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = "/data/cl195/NeurIPS_workshop"
HERE = f"{ROOT}/experiments/h3-esconv-shortcut"
SRC = f"{HERE}/data/esconv.json"

# identical list to h2/audit.py
STOP = set(
    "the a an and or or but if of to in on at for with it is was were be been am "
    "are i you he she they we my your his her their this that these those so just "
    "like really very not no yes do does did have has had will would can could s t "
    "m re ve ll d as by from about out up down over then than there here when what "
    "who how all any some more most other into only own same too s".split())


def probe(X, y, seeds=3, stop_only=False):
    """Balanced accuracy of a linear bag-of-words probe. Mirrors h2/audit.py."""
    if stop_only:
        X = [" ".join(w for w in re.findall(r"[a-z']+", t.lower()) if w in STOP)
             for t in X]
    c = collections.Counter(y)
    if len(c) < 2 or min(c.values()) < 5:
        return float("nan")
    accs = []
    for sd in range(seeds):
        p = cross_val_predict(
            make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                          min_df=3, max_features=200_000),
                          LogisticRegression(max_iter=3000, C=1.0,
                                             class_weight="balanced")),
            X, y, cv=StratifiedKFold(5, shuffle=True, random_state=sd))
        accs.append(balanced_accuracy_score(y, p))
    return float(np.mean(accs))


def numeric_probe(V, y, seeds=3):
    """Same readout on a dense numeric feature matrix (strategy counts, length,
    survey ratings). Standardised, otherwise the same classifier."""
    V = np.asarray(V, float)
    if V.ndim == 1:
        V = V[:, None]
    c = collections.Counter(y)
    if len(c) < 2 or min(c.values()) < 5:
        return float("nan")
    accs = []
    for sd in range(seeds):
        p = cross_val_predict(
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=1.0,
                                             class_weight="balanced")),
            V, y, cv=StratifiedKFold(5, shuffle=True, random_state=sd))
        accs.append(balanced_accuracy_score(y, p))
    return float(np.mean(accs))


def load():
    raw = json.load(open(SRC))
    out = []
    for sp in ("train", "validation", "test"):
        for d in raw.get(sp, []):
            usr = [t["text"] for t in d["dialog"] if t["speaker"] == "usr"]
            sysm = [t for t in d["dialog"] if t["speaker"] == "sys"]
            strat = [t.get("strategy") for t in sysm if t.get("strategy")]
            ss = (d.get("survey_score") or {}).get("seeker") or {}
            out.append(dict(
                split=sp,
                problem_type=d.get("problem_type"),
                emotion_type=d.get("emotion_type"),
                experience_type=d.get("experience_type"),
                situation=d.get("situation") or "",
                seeker_text=" ".join(usr),
                supporter_text=" ".join(t["text"] for t in sysm),
                first_seeker_turn=usr[0] if usr else "",
                strategies=strat,
                n_turns=len(d["dialog"]),
                n_usr=len(usr),
                n_sys=len(sysm),
                seeker_words=sum(len(t.split()) for t in usr),
                survey=[float(ss.get(k, "nan") or "nan") for k in
                        ("initial_emotion_intensity", "empathy", "relevance",
                         "final_emotion_intensity")],
            ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = 1 if args.quick else 3
    os.makedirs(f"{HERE}/results", exist_ok=True)

    R = load()
    print(f"ESConv: {len(R)} dialogues  "
          f"({collections.Counter(r['split'] for r in R)})")

    VOCAB = sorted({s for r in R for s in r["strategies"]})
    print(f"{len(VOCAB)} supporter strategies: {VOCAB}\n")

    # ---- strategy features, NO TEXT -------------------------------------
    def counts_and_props(r):
        c = collections.Counter(r["strategies"])
        n = max(1, len(r["strategies"]))
        return [c[s] for s in VOCAB] + [c[s] / n for s in VOCAB]

    def seq_ngrams(r):
        """Strategy sequence as a token string -- unigrams+bigrams over the
        strategy IDs only. This is the closest analogue of A2a's asked/not-asked
        matrix that also carries ORDER."""
        toks = [s.replace(" ", "_") for s in r["strategies"]]
        return " ".join(toks + [f"{a}|{b}" for a, b in zip(toks, toks[1:])])

    # Long-tail classes are dropped at a natural frequency gap: problem_type
    # falls 156 -> 28 after five classes, emotion_type 40 -> 13 after seven. A
    # class with 2 members cannot be cross-validated 5-fold, and keeping it
    # would make balanced accuracy a function of which fold got the singleton.
    MIN_CLASS = 30
    TARGETS = [("problem_type", "problem_type"), ("emotion_type", "emotion_type"),
               ("experience_type", "experience_type")]
    out = {"n": len(R), "n_strategies": len(VOCAB), "strategies": VOCAB,
           "targets": {}}

    for tname, key in TARGETS:
        big = {k for k, v in collections.Counter(
            r[key] for r in R if r[key]).items() if v >= MIN_CLASS}
        dropped = collections.Counter(
            r[key] for r in R if r[key] and r[key] not in big)
        keep = [r for r in R if r[key] in big]
        y = [r[key] for r in keep]
        if dropped:
            print(f"  dropped {sum(dropped.values())} dialogues in "
                  f"{len(dropped)} classes with n < {MIN_CLASS}: "
                  f"{dict(dropped)}")
        cnt = collections.Counter(y)
        chance = 1.0 / len(cnt)
        major = max(cnt.values()) / len(y)
        print(f"{'='*92}\n{tname}: {len(cnt)} classes, n={len(keep)}, "
              f"chance {chance:.3f}, majority {major:.3f}\n{'='*92}")

        res = {"n": len(keep), "n_classes": len(cnt), "chance": chance,
               "majority": major}
        rows = [
            ("seeker text (the construct)",
             lambda: probe([r["seeker_text"] for r in keep], y, seeds)),
            ("supporter text",
             lambda: probe([r["supporter_text"] for r in keep], y, seeds)),
            ("situation field (pre-dialogue)",
             lambda: probe([r["situation"] for r in keep], y, seeds)),
            ("first seeker turn only",
             lambda: probe([r["first_seeker_turn"] for r in keep], y, seeds)),
            ("STRATEGY counts + proportions, no text",
             lambda: numeric_probe([counts_and_props(r) for r in keep], y, seeds)),
            ("STRATEGY sequence n-grams, no text",
             lambda: probe([seq_ngrams(r) for r in keep], y, seeds)),
            ("dialogue length (turns, words)",
             lambda: numeric_probe([[r["n_turns"], r["n_usr"], r["n_sys"],
                                     r["seeker_words"]] for r in keep], y, seeds)),
            ("seeker survey ratings",
             lambda: numeric_probe(
                 np.nan_to_num([r["survey"] for r in keep], nan=0.0), y, seeds)),
            ("stopwords only (seeker text)",
             lambda: probe([r["seeker_text"] for r in keep], y, seeds,
                           stop_only=True)),
        ]
        for label, fn in rows:
            v = fn()
            reg = ((v - chance) / (res["_text"] - chance)
                   if label.startswith("stopwords") and res.get("_text") else None)
            extra = f"   register {reg*100:5.1f}%" if reg is not None else ""
            print(f"  {label:42s} {v:.3f}{extra}")
            res[label] = v
            if label.startswith("seeker text"):
                res["_text"] = v
        res.pop("_text", None)
        out["targets"][tname] = res
        print()

    json.dump(out, open(f"{HERE}/results/esconv_audit.json", "w"), indent=1)
    print(f"wrote {HERE}/results/esconv_audit.json")


if __name__ == "__main__":
    main()
