#!/usr/bin/env python3
"""Build the E2a / E5 / E4a evaluation sets.

Step 3 of INTERVENTION_DESIGN.md §7. CPU only.

E2a  natural-conflict set -- participants whose REAL question-selection profile
     points away from their label, using leave-one-out out-of-fold predictions
     from the A2a probe so no participant is scored by a model that saw them.

E5   construct-preservation pairs -- the mirror of A2. A2 fixed the participant
     and swapped the interviewer; E5 fixes the interviewer and swaps the
     participant. Construct sensitivity is how much p(depression) moves when
     the *evidence* changes with context held byte-identical. An intervention
     that flattens this has broken the model rather than debiased it.

E4a  prompt clusters -- topical partition of the scripted prompt IDs, for
     leave-one-cluster-out protocol-shift evaluation. Emitted as a PROPOSAL for
     review; it is not frozen until a human signs off (design §8.1).

Interviewer text is spliced with the same `interleave` helper A2b used, so
insertion positions are computed identically across every set in the suite.
"""
from __future__ import annotations

import collections, importlib.util, json, os, re, sys
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/suite/results"
PAT = re.compile(r"^Ellie: ([A-Za-z][A-Za-z0-9_]*) \((.*)\)\s*$")
MIN_N, MAX_FRAC = 20, 0.84          # identical to a2a_prompt_label_association
N_PAIRS = 200
SPAN_TOL = 15                       # max |Δ participant turns| within an E5 pair

_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
interleave = _a2b.interleave


def prompt_ids(row):
    return {m.group(1) for m in
            (PAT.match(l) for l in (row["text_ellie_only"] or "").split("\n"))
            if m}


# ------------------------------------------------------------------ E2a

def build_e2a(rows):
    subs = [r for r in rows if prompt_ids(r)]
    ids = collections.Counter(i for r in subs for i in prompt_ids(r))
    tested = sorted(i for i, n in ids.items()
                    if MIN_N <= n <= MAX_FRAC * len(subs))
    X = np.array([[1 if p in prompt_ids(r) else 0 for p in tested] for r in subs])
    y = np.array([r["phq8_binary"] for r in subs])

    # leave-one-out: participant i is scored by a probe fitted without them
    oof = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
        clf.fit(X[m], y[m])
        oof[i] = clf.predict_proba(X[i:i + 1])[0, 1]

    pred = (oof > 0.5).astype(int)
    conflict = pred != y
    items = [{"uid": r["uid"], "participant_id": r["participant_id"],
              "split": r["split"], "phq8_binary": int(r["phq8_binary"]),
              "shortcut_pred": int(p), "shortcut_prob": float(o),
              "is_conflict": bool(c)}
             for r, p, o, c in zip(subs, pred, oof, conflict)]

    print(f"E2a  n with prompt IDs = {len(subs)} / {len(rows)}   "
          f"dep rate {y.mean():.3f} (vs {np.mean([r['phq8_binary'] for r in rows]):.3f} overall)")
    print(f"     features {len(tested)}   LOO agreement {(pred == y).mean():.3f}")
    print(f"     CONFLICT n = {conflict.sum()}  "
          f"(gold=1 & pred=0: {((y == 1) & (pred == 0)).sum()}, "
          f"gold=0 & pred=1: {((y == 0) & (pred == 1)).sum()})")
    bysplit = collections.Counter(r["split"] for r, c in zip(subs, conflict) if c)
    print(f"     by split: {dict(bysplit)}")
    print("     ⚠ F6: the ID-carrying subset is NOT label-representative "
          "(22.8 % dep with IDs vs 41.5 % without) — E2a inherits that bias.")
    return items, tested


# ------------------------------------------------------------------- E5

def build_e5(rows, seed=0):
    """Pairs (i, i') sharing interviewer context j, differing in label."""
    rng = np.random.default_rng(seed)
    pos = [r for r in rows if r["phq8_binary"] == 1]
    neg = [r for r in rows if r["phq8_binary"] == 0]
    spans = {r["uid"]: r["n_participant_spans"] for r in rows}

    pairs, tries, deltas = [], 0, []
    while len(pairs) < N_PAIRS and tries < N_PAIRS * 200:
        tries += 1
        j = rows[rng.integers(len(rows))]
        i = pos[rng.integers(len(pos))]
        # nearest-span negative, excluding anyone already used as context
        cand = [c for c in neg
                if abs(spans[c["uid"]] - spans[i["uid"]]) <= SPAN_TOL
                and c["uid"] != j["uid"]]
        if not cand or i["uid"] == j["uid"]:
            continue
        ip = cand[rng.integers(len(cand))]
        key = (i["uid"], ip["uid"], j["uid"])
        if any(p["key"] == key for p in pairs):
            continue
        deltas.append(abs(spans[i["uid"]] - spans[ip["uid"]]))
        ell = [l for l in (j["text_ellie_only"] or "").split("\n") if l.strip()]
        pairs.append({
            "key": key, "context_uid": j["uid"], "n_context_turns": len(ell),
            "pos": {"uid": i["uid"], "split": i["split"], "gold": 1,
                    "text": interleave(i["text_participant_only"], ell)},
            "neg": {"uid": ip["uid"], "split": ip["split"], "gold": 0,
                    "text": interleave(ip["text_participant_only"], ell)},
        })

    print(f"\nE5   pairs built = {len(pairs)} (target {N_PAIRS}, {tries} draws)")
    print(f"     |Δ participant turns| within pair: median {np.median(deltas):.0f} "
          f"max {max(deltas)} (tolerance {SPAN_TOL})")
    print(f"     distinct contexts j = {len({p['context_uid'] for p in pairs})}")
    print("     gold follows the PARTICIPANT; interviewer text is byte-identical "
          "within each pair.")
    return [{k: v for k, v in p.items() if k != "key"} for p in pairs]


# ------------------------------------------------------------------ E4a

CLUSTERS = {
    "clinical_history": ["diagnos", "therapy", "ptsd", "depress", "medicat",
                         "counsel", "treat"],
    "help_seeking": ["seek_help", "why_seek", "helped", "advice", "support"],
    "affect_symptom": ["feel", "sleep", "temper", "mood", "energy", "appetite",
                       "cry", "guilt", "shy", "argument"],
    "trauma_event": ["trigger", "trauma", "last_time", "happened", "memory",
                     "regret", "hard"],
    "relationships": ["family", "friend", "relationship", "kids", "parent",
                      "bf_", "best_friend", "close"],
    "biographical": ["from", "born", "hometown", "school", "study", "travel",
                     "l_a", "live", "job", "work", "military", "weekend",
                     "enjoy", "hobb", "dream", "perfect_day"],
}


def build_e4a(rows, tested):
    texts = {}
    for r in rows:
        for l in (r["text_ellie_only"] or "").split("\n"):
            m = PAT.match(l)
            if m:
                texts.setdefault(m.group(1), m.group(2))

    assign, unassigned = {}, []
    for pid in tested:
        blob = f"{pid} {texts.get(pid, '')}".lower()
        hit = [c for c, kws in CLUSTERS.items() if any(k in blob for k in kws)]
        if len(hit) == 1:
            assign[pid] = hit[0]
        elif len(hit) > 1:
            assign[pid] = hit[0]           # first match wins; flagged below
            unassigned.append((pid, "AMBIGUOUS", hit, texts.get(pid, "")))
        else:
            assign[pid] = "unclustered"
            unassigned.append((pid, "NO MATCH", [], texts.get(pid, "")))

    counts = collections.Counter(assign.values())
    print(f"\nE4a  proposed clusters over {len(tested)} tested prompt IDs:")
    for c, n in counts.most_common():
        print(f"     {c:20s} {n:3d}")
    print(f"     needs human review: {len(unassigned)} "
          f"({sum(1 for u in unassigned if u[1] == 'AMBIGUOUS')} ambiguous, "
          f"{sum(1 for u in unassigned if u[1] == 'NO MATCH')} unmatched)")
    for pid, why, hit, q in unassigned[:12]:
        print(f"       {why:10s} {pid[:24]:25s} {str(hit):34s} {q[:34]}")
    print("     ⚠ PROPOSAL ONLY — design §8.1 requires this be frozen by hand "
          "before E4a is run.")
    return {"assignment": assign, "needs_review":
            [{"id": p, "why": w, "candidates": h, "question": q}
             for p, w, h, q in unassigned]}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(COND)]
    print(f"loaded {len(rows)} participants\n" + "=" * 78)

    e2a, tested = build_e2a(rows)
    e5 = build_e5(rows)
    e4a = build_e4a(rows, tested)

    with open(f"{OUT}/e2a_natural_conflict.json", "w") as f:
        json.dump(e2a, f, indent=2)
    with open(f"{OUT}/e5_construct_pairs.jsonl", "w") as f:
        for p in e5:
            f.write(json.dumps(p) + "\n")
    with open(f"{OUT}/e4a_prompt_clusters_PROPOSAL.json", "w") as f:
        json.dump(e4a, f, indent=2)
    print(f"\nwrote 3 artefacts to {OUT}")


if __name__ == "__main__":
    main()
