#!/usr/bin/env python3
"""Experiment C1 -- template / source normalization.

IMHI unifies 10 corpora into one schema, but each source keeps a recognizable
instruction wording. If that wording identifies the source, a model can skip the
psychology and apply the source's label prior instead. C1 measures whether it
does.

Four conditions, crossing "is the post present" with "is the template
source-specific":

                        native template        canonical template
    post present        native                 normalized
    post absent         native_nopost          normalized_nopost

  native      - the source's own question, verbatim from the manifest
  normalized  - one canonical frame, construct preserved. Only the surface
                wording changes: SDCNL gains the `Question:` prefix it uniquely
                lacks, and "does the poster suffer from X" / "does the post show
                risk of X" / "does the X dimension exist in the post" all become
                "does this post indicate X".
  *_nopost    - the question alone. This is the provenance payload: whatever
                accuracy survives here is what the template alone tells the
                model about the label distribution.

The two contrasts that matter:

  Δ_template = Acc(native) - Acc(normalized)
      template keying while the content is available.

  Δ_prior    = Acc(native_nopost) - Acc(normalized_nopost)
      the direct measurement. Both conditions are content-free, so any gap is
      the source-specific wording buying label-prior information -- exactly
      "recognize the dataset, then guess its prior".

Construct is preserved in every normalization: rewording the *task* rather than
the *template* would measure task destruction, not shortcut use. swmh and t-sid
are the sharpest pair -- near-identical tasks, different label spaces and
different priors -- and they stay distinguishable only by label space after
normalization.

Scored by teacher-forced label log-likelihood over each source's own label
space, never by parsing free generation.
"""
from __future__ import annotations

import argparse, json, os, re, sys, time, collections, random

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
MANIFEST = f"{ROOT}/data/processed/text_manifest.jsonl"

BINARY_CANON = "Question: Does this post indicate {c}?"
MULTI_CANON = "Question: What is the {c} shown in this post?"

# --- label verbalizers -------------------------------------------------------
# IMHI's multiclass gold answers are not bare labels. They are full sentences,
# and scoring a bare label instead lands far off-distribution: the first run
# produced a *constant* prediction for all four multiclass sources, with 4-nat
# gaps between labels against 0.2-0.6 nat of variation across posts. Label
# identity swamped the content entirely and every condition scored exactly 1/|L|.
#
# So each label is scored inside its source's own answer frame, taken verbatim
# from `repos/MentaLLaMA/test_data/test_instruction/*.csv`. The frame is held
# FIXED across native and normalized -- only the *question* wording varies --
# because varying the answer surface too would confound template identity with
# plain distribution shift, and the shared prefix contributes the same
# log-probability to every candidate.
#
# Binary sources need none of this: their gold answers do begin with a bare
# "Yes"/"No" (verified across all five), which is why they scored sensibly.
ANSWER_PREFIX = "This post shows "

VERBALIZER = {
    "swmh": {
        "depression":       "mental disorder symptoms related to depression",
        "suicide":          "mental disorder symptoms related to suicide",
        "anxiety":          "mental disorder symptoms related to anxiety",
        "bipolar disorder": "mental disorder symptoms related to bipolar disorder",
        "none":             "no mental disorder symptoms",
    },
    "SAD": {
        "work":                     "the stress cause related to work",
        "health issues":            "the stress cause related to health issues",
        "financial problem":        "the stress cause related to financial problem",
        "school":                   "the stress cause related to school",
        "family issues":            "the stress cause related to family issues",
        "emotion turmoil":          "the stress cause related to emotion turmoil",
        "social relationships":     "the stress cause related to social relationships",
        "everyday decision making": "the stress cause related to everyday decision making",
        "other":                    "other stress causes",
    },
    "CAMS": {
        "alienation":      "causes of depression related to alienation",
        "relationship":    "causes of depression related to relationship",
        "jobs and career": "causes of depression related to jobs and career",
        "medication":      "causes of depression related to medication",
        "bias or abuse":   "causes of depression related to bias or abuse",
        "none":            "no causes of depression",
    },
    "t-sid": {
        "suicide or self-harm tendency": "suicide or self-harm tendency",
        "depression":                    "depression",
        "ptsd":                          "PTSD",
        "none":                          "no mental disorders",
    },
}

# construct extraction from each source's native question wording
BINARY_PAT = [
    re.compile(r"does the poster suffer from (.+?)\?", re.I),
    re.compile(r"does the post show risk of (.+?)\?", re.I),
    re.compile(r"does the (.+?) dimension exist in the post\?", re.I),
    re.compile(r"does the post indicate (.+?)\?", re.I),
]
MULTI_CONSTRUCT = {
    "CAMS":  "cause of depression",
    "SAD":   "cause of stress",
    "swmh":  "mental disorder symptom",
    "t-sid": "mental disorder",
}


def construct_of(row):
    """The psychological construct the question asks about, source-independent."""
    q = row["question"]
    if row["task_type"] == "multiclass":
        return MULTI_CONSTRUCT[row["source_dataset"]]
    for p in BINARY_PAT:
        m = p.search(q)
        if m:
            return m.group(1).strip()
    raise ValueError(f"unparsed question: {q!r}")


def canonical(row):
    c = construct_of(row)
    return (MULTI_CANON if row["task_type"] == "multiclass"
            else BINARY_CANON).format(c=c)


def prompts(row):
    """-> {condition: prompt}. Multiclass prompts end in the source's answer
    frame so the scored span is the label inside its native sentence."""
    post = f'Consider this post: "{row["text"]}" '
    tail = " " + ANSWER_PREFIX if row["source_dataset"] in VERBALIZER else ""
    nat, can = row["question"] + tail, canonical(row) + tail
    return {
        "native":            post + nat,
        "normalized":        post + can,
        "native_nopost":     nat,
        "normalized_nopost": can,
    }


BINARY_SURFACE = {"yes": "Yes", "no": "No"}


def surfaces_for(src, space):
    """-> (list of scored strings, mapping back to canonical labels)"""
    v = VERBALIZER.get(src)
    if not v:
        # Binary gold answers begin with capitalised "Yes,"/"No,"; the manifest
        # stores them lowercased. Scoring the capitalised form keeps the binary
        # path in-distribution too. The change is symmetric across both labels,
        # so it cannot bias the contrast.
        return ([BINARY_SURFACE.get(l, l) for l in space],
                {BINARY_SURFACE.get(l, l): l for l in space})
    missing = [l for l in space if l not in v]
    if missing:
        raise ValueError(f"{src}: no verbalizer for {missing}")
    return [v[l] for l in space], {v[l]: l for l in space}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-source", type=int, default=300,
                    help="D3: matched subset, the same sample across conditions")
    ap.add_argument("--budget", type=int, default=1024,
                    help="post token budget, applied identically to every source")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = args.ckpt or f"{ROOT}/checkpoints/{args.model}"
    out_path = args.out or f"{ROOT}/runs/c1_{args.model}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = [json.loads(l) for l in open(MANIFEST)]
    rows = [r for r in rows if r.get("label") and r.get("text")]

    # label space per source, from the source's own observed labels
    spaces = collections.defaultdict(set)
    for r in rows:
        spaces[r["source_dataset"]].add(r["label"])
    spaces = {k: sorted(v) for k, v in spaces.items()}

    # matched subset, stratified by label, fixed seed
    rng = random.Random(args.seed)
    by_src = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_src[r["source_dataset"]][r["label"]].append(r)
    sample = []
    for src, by_lab in sorted(by_src.items()):
        pool = []
        per_lab = max(1, args.per_source // len(by_lab))
        for lab, rs in sorted(by_lab.items()):
            rng.shuffle(rs)
            pool += rs[:per_lab]
        rng.shuffle(pool)
        sample += pool[:args.per_source]
    print(f"sample: {len(sample)} rows over {len(by_src)} sources")
    for src in sorted(spaces):
        n = sum(1 for r in sample if r["source_dataset"] == src)
        print(f"  {src:12s} n={n:4d}  |L|={len(spaces[src])}  {spaces[src]}")

    t0 = time.time()
    sc = TextScorer(ckpt, device=args.device, adapter=args.adapter)
    print(f"\nloaded {args.model} in {time.time()-t0:.0f}s", flush=True)

    def clip(text):
        ids = sc.tok(text, add_special_tokens=False).input_ids
        return (text, False) if len(ids) <= args.budget else \
               (sc.tok.decode(ids[:args.budget]), True)

    # the *_nopost prompts do not depend on the post, so each distinct question
    # is scored once and reused across every row that shares it
    nopost_cache = {}

    n_done, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, r in enumerate(sample):
            src = r["source_dataset"]
            L = spaces[src]
            surf, back = surfaces_for(src, L)
            r = dict(r)
            r["text"], trunc = clip(r["text"])
            for cond, prompt in prompts(r).items():
                if cond.endswith("_nopost"):
                    key = (cond, prompt, src)
                    if key not in nopost_cache:
                        nopost_cache[key] = sc.score(prompt, surf)
                    scored = nopost_cache[key]
                else:
                    scored = sc.score(prompt, surf)
                pred = back[max(scored, key=lambda k: scored[k]["mean"])]
                f.write(json.dumps({
                    "uid": r["uid"], "source_dataset": src,
                    "corpus": r.get("corpus"), "task_type": r["task_type"],
                    "model": args.model, "condition": cond,
                    "question_native": r["question"],
                    "construct": construct_of(r),
                    "gold": r["label"], "label_space": L,
                    "pred": pred, "correct": pred == r["label"],
                    "logprobs": {back[k]: v["mean"] for k, v in scored.items()},
                    "n_words": r.get("n_words"), "truncated": trunc,
                }) + "\n")
                n_done += 1
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(sample)} rows | {n_done} scored | "
                      f"{el:.0f}s ({el/(i+1):.2f}s/row)", flush=True)
            f.flush()

    print(f"\nwrote {out_path}  ({n_done} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
