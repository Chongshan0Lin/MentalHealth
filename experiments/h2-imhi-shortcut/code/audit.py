#!/usr/bin/env python3
"""H2 -- what shortcuts live in IMHI, the corpus MentaLLaMA was trained on?

Model-free, as H1 and A2a were. The reason to audit this corpus rather than
another is that it closes a loop the project currently leaves open: every
shortcut result we have about MentaLLaMA is a statement about the *model*
(CSS 0.205, SER 1.82, flip rate 0.559, the layer 8-21 probing band). IMHI is
that model's training data, so a defect found here is a candidate *cause*
rather than another correlate.

IMHI pools 9 source datasets with disjoint label spaces into one instruction
corpus. That pooling is itself the first hypothesis: if the source is
identifiable from the text, a model can narrow the label space before reading
any clinical content -- "this is a dreaddit item, so the answer is yes or no
about stress" -- which is exactly the corpus-identity pathway that L3
FALSIFIED for DAIC. It was falsifiable there because DAIC is a single corpus.
Here it is not.

Six probes:

  P1 source -> label     how much does knowing only the source dataset buy?
                         This is the ceiling of the corpus-identity pathway.
  P2 text -> source      is the source recoverable from the post? P1 is only
                         reachable by a model if P2 succeeds.
  P3 per-source text     the ordinary text probe, per source, against that
                         source's own majority baseline
                         CONTROL: stopwords-only, the same register probe used
                         on ptsd_in_the_wild in H1, so the two are comparable.
  P4 explanation leak    IMHI's gold explanations open with a template that
                         names the answer ("This post shows ... related to
                         suicide. Reasoning: ..."). Measured with word-boundary
                         matching, because the binary sources' labels are `yes`
                         and `no` and `no` is a substring of `not` and `know` --
                         a naive substring test reports 100 % on noise.
  P5 duplicates          the same post appearing under several sources, and
                         whether its labels then conflict
  P6 swmh subreddits     swmh's label set is suicide / depression / anxiety /
                         bipolar disorder / none, which are subreddit names. If
                         the label is really the scrape source, self-identifying
                         phrases should carry it and removing them should hurt.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, re, sys, warnings

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

ROOT = "/data/cl195/NeurIPS_workshop"
HERE = f"{ROOT}/experiments/h2-imhi-shortcut"
SRC = f"{ROOT}/data/processed/text_manifest.jsonl"
STOP = set(
    "the a an and or or but if of to in on at for with it is was were be been am "
    "are i you he she they we my your his her their this that these those so just "
    "like really very not no yes do does did have has had will would can could s t "
    "m re ve ll d as by from about out up down over then than there here when what "
    "who how all any some more most other into only own same too s".split())


def probe(X, y, seeds=3, stop_only=False):
    """Balanced accuracy of a linear bag-of-words probe, grouped CV."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 CV seed instead of 3")
    args = ap.parse_args()
    os.makedirs(f"{HERE}/results", exist_ok=True)
    seeds = 1 if args.quick else 3

    rows = [json.loads(l) for l in open(SRC)]
    imhi = [r for r in rows if r["corpus"] == "IMHI" and r.get("label")]
    print(f"IMHI: {len(imhi)} rows with a label, "
          f"{len({r['source_dataset'] for r in imhi})} source datasets\n")
    res = {}

    # ---- P1: how much does the SOURCE alone buy? -------------------------
    print("=" * 92)
    print("P1 -- label from SOURCE DATASET alone (no text read at all)")
    print("=" * 92)
    y = [r["label"] for r in imhi]
    src = [r["source_dataset"] for r in imhi]
    glob_major = collections.Counter(y).most_common(1)[0][1] / len(y)
    per_src = collections.defaultdict(collections.Counter)
    for s, l in zip(src, y):
        per_src[s][l] += 1
    src_major = sum(c.most_common(1)[0][1] for c in per_src.values()) / len(y)
    n_lab = len(set(y))
    print(f"  distinct labels across the pooled corpus : {n_lab}")
    print(f"  always the global majority label         : {glob_major:.3f}")
    print(f"  the majority label OF ITS OWN SOURCE     : {src_major:.3f}")
    print(f"  -> knowing only which dataset a post came from is worth "
          f"{src_major - glob_major:+.3f} accuracy,")
    print(f"     before a single clinical word is read.")
    res["P1"] = dict(n_labels=n_lab, global_majority=glob_major,
                     source_majority=src_major)

    # label-space narrowing: how many labels are even reachable per source?
    print(f"\n  labels reachable per source (the search space a model gets for free):")
    for s, c in sorted(per_src.items(), key=lambda x: -sum(x[1].values())):
        print(f"    {s:14s} {sum(c.values()):6d} rows  ->  {len(c)} of {n_lab} labels")

    # ---- P2: is the source recoverable from the text? --------------------
    print("\n" + "=" * 92)
    print("P2 -- SOURCE DATASET from the post text (P1 is only reachable if this works)")
    print("=" * 92)
    a = probe([r["text"] for r in imhi], src, seeds=seeds)
    a_stop = probe([r["text"] for r in imhi], src, seeds=seeds, stop_only=True)
    print(f"  balanced accuracy, 9-way          : {a:.3f}   (chance "
          f"{1/len(per_src):.3f})")
    print(f"  stopwords only, no content words  : {a_stop:.3f}")
    res["P2"] = dict(source_id=a, source_id_stopwords=a_stop,
                     chance=1 / len(per_src))

    # ---- P3: per-source text probe --------------------------------------
    print("\n" + "=" * 92)
    print("P3 -- per source: is the label recoverable from the post?")
    print("=" * 92)
    print(f"{'source':14s} {'n':>6s} {'#lab':>5s} {'chance':>7s} {'major':>7s} "
          f"{'TEXT':>6s} {'STOP':>6s}")
    print("-" * 92)
    res["P3"] = {}
    for s, c in sorted(per_src.items(), key=lambda x: -sum(x[1].values())):
        sub = [r for r in imhi if r["source_dataset"] == s]
        yy = [r["label"] for r in sub]
        t = probe([r["text"] for r in sub], yy, seeds=seeds)
        st = probe([r["text"] for r in sub], yy, seeds=seeds, stop_only=True)
        chance, major = 1 / len(c), c.most_common(1)[0][1] / len(sub)
        res["P3"][s] = dict(n=len(sub), n_labels=len(c), chance=chance,
                            majority=major, text=t, stopword=st)
        print(f"{s:14s} {len(sub):6d} {len(c):5d} {chance:7.3f} {major:7.3f} "
              f"{t:6.3f} {st:6.3f}")
    print("\n  STOP is the same register probe H1 ran on ptsd_in_the_wild "
          "(0.928 there),\n  so the two corpora are on one ruler.")

    # ---- P4: explanation template leakage --------------------------------
    print("\n" + "=" * 92)
    print("P4 -- does the gold explanation state the answer?")
    print("=" * 92)
    print(f"{'source':14s} {'#lab':>5s} {'label verbatim':>15s} {'template 1st sent':>19s}"
          f"  {'evidence':>9s}")
    print("-" * 92)
    res["P4"] = {}
    TEMPL = re.compile(r"^this post (shows|indicates|does not)", re.I)
    for s, c in sorted(per_src.items(), key=lambda x: -sum(x[1].values())):
        sub = [r for r in imhi if r["source_dataset"] == s
               and r.get("gold_explanation")]
        if not sub:
            continue
        hit = sum(1 for r in sub
                  if re.search(r"\b" + re.escape(r["label"].lower()) + r"\b",
                               r["gold_explanation"].lower()))
        tmpl = sum(1 for r in sub if TEMPL.match(r["gold_explanation"].strip()))
        # `no`/`yes` are substrings of ordinary words, so a binary label space
        # makes the verbatim test uninformative; say so rather than report it.
        ev = "weak" if len(c) == 2 else "strong"
        res["P4"][s] = dict(n=len(sub), verbatim=hit / len(sub),
                            template=tmpl / len(sub), n_labels=len(c))
        print(f"{s:14s} {len(c):5d} {hit:7d}/{len(sub):<6d} {hit/len(sub):5.1%} "
              f"{tmpl:7d}/{len(sub):<6d} {tmpl/len(sub):5.1%}  {ev:>9s}")
    print("\n  'strong' = multiclass label space, so a verbatim hit is not chance.")
    print("  MentaLLaMA was trained to GENERATE these explanations, i.e. trained")
    print("  on a format that states the answer first and reasons afterwards.")

    # ---- P5: duplicates across sources -----------------------------------
    print("\n" + "=" * 92)
    print("P5 -- the same post under more than one source")
    print("=" * 92)
    h = collections.defaultdict(list)
    for r in imhi:
        h[hashlib.md5(r["text"].strip().encode()).hexdigest()].append(r)
    multi = {k: v for k, v in h.items()
             if len({x["source_dataset"] for x in v}) > 1}
    dup_same = sum(1 for v in h.values() if len(v) > 1
                   and len({x["source_dataset"] for x in v}) == 1)
    print(f"  distinct posts {len(h)} for {len(imhi)} rows "
          f"({len(imhi)-len(h)} duplicate rows)")
    print(f"  posts under >1 source          : {len(multi)}")
    print(f"  posts duplicated WITHIN a source: {dup_same}")
    conf = {k: v for k, v in multi.items() if len({x["label"] for x in v}) > 1}
    print(f"  of the cross-source posts, with differing labels: {len(conf)}")
    for k, v in collections.Counter(
            frozenset(x["source_dataset"] for x in vv) for vv in multi.values()
    ).most_common(6):
        print(f"    {sorted(k)}: {v}")
    res["P5"] = dict(distinct=len(h), rows=len(imhi), cross_source=len(multi),
                     within_source_dups=dup_same, conflicting=len(conf))

    # ---- P6: swmh -- is the label the subreddit? -------------------------
    print("\n" + "=" * 92)
    print("P6 -- swmh (57 % of IMHI): the label set is a list of subreddits")
    print("=" * 92)
    sw = [r for r in imhi if r["source_dataset"] == "swmh"]
    yy = [r["label"] for r in sw]
    base = probe([r["text"] for r in sw], yy, seeds=seeds)
    # strip self-identifying disorder vocabulary: if the label is the scrape
    # source, the post often names its own community or condition
    NAMES = re.compile(
        r"\b(suicid\w*|kill myself|depress\w*|anxi\w*|bipolar|manic|mania|"
        r"panic attack\w*|self.harm\w*|r/\w+|subreddit)\b", re.I)
    scrub = probe([NAMES.sub(" ", r["text"]) for r in sw], yy, seeds=seeds)
    stop = probe([r["text"] for r in sw], yy, seeds=seeds, stop_only=True)
    print(f"  labels: {dict(collections.Counter(yy))}")
    print(f"  full text                                  : {base:.3f}")
    print(f"  after removing self-identifying vocabulary : {scrub:.3f}"
          f"   ({scrub-base:+.3f})")
    print(f"  stopwords only                             : {stop:.3f}")
    print(f"  chance {1/len(set(yy)):.3f}")
    res["P6"] = dict(full=base, scrubbed=scrub, stopword=stop,
                     chance=1 / len(set(yy)))

    json.dump(res, open(f"{HERE}/results/audit.json", "w"), indent=2)
    print(f"\nwrote {HERE}/results/audit.json")


if __name__ == "__main__":
    main()
