#!/usr/bin/env python3
"""E4a -- topical clustering of the 47 question-like scripted prompts.

Supersedes the keyword-matched proposal in e4a_prompt_clusters_PROPOSAL.json,
which left 39/75 unassigned because 28 of those 75 were backchannels rather
than questions (see ../analysis.md).

The clustering below is HAND-AUTHORED from the prompt texts, which is what the
design mandates (§8.1) -- at 47 items a keyword heuristic buys nothing and
hides its own errors. Two properties are enforced programmatically rather than
trusted: every prompt is assigned exactly once, and no prompt appears twice.

Purpose. E4a evaluates generalisation under PROTOCOL SHIFT: hold out the
participants whose interview is dominated by cluster k, train on the rest, test
on them. That is the operational test of whether question selection is a
shortcut or legitimate clinical evidence -- ablation cannot separate those two
(findings.md open question 5).

Clusters are topical, NOT association-based. Grouping by odds ratio would build
the label into the partition and make the held-out fold trivially predictable.
Mean OR per cluster is reported afterwards purely as a diagnostic.
"""
from __future__ import annotations

import collections, json, re
import numpy as np

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/suite/results"
PAT = re.compile(r"^Ellie: ([A-Za-z][A-Za-z0-9_]*) \((.*)\)\s*$")

# ---------------------------------------------------------------------------
# Hand-authored partition. Rationale per cluster is the protocol branch it
# represents -- these are the threads the adaptive interview can go down.
# ---------------------------------------------------------------------------
CLUSTERS = {
    # The adaptive clinical thread: reachable only after a mental-health
    # disclosure. Contains every prompt that survived BH correction in A2a.
    "clinical_care": [
        "why_seek_help", "therapy_going", "when_diagnosed", "therapy_changes",
        "therapy_useful", "trigger", "bouts_symptoms", "symptoms_cope",
        "behavior_changes", "sleep_affects",
    ],
    # Parenting branch -- reachable only if the participant has children.
    "parenting": [
        "parent_best", "parent_hardest", "easy_parent", "parent_differences",
    ],
    # Geography/relocation branch. Large, scripted, near-universal.
    "la_relocation": [
        "adapted_LA", "when_LA", "why_LA", "compares_LA", "like_about_LA",
        "dont_like_LA", "often_backB",
    ],
    # Leisure and travel -- the rapport-building default script.
    "travel_leisure": [
        "ideal_weekendC", "travel_trips", "travel_changed",
        "Ellie17Dec2012_10",
    ],
    # Self-description and social reflection.
    "self_concept": [
        "Ellie17Dec2012_07", "self_change", "BF_describe", "influence_positive",
    ],
    # Autobiographical narrative with affective loading, either valence.
    "affect_and_memory": [
        "feelguilty", "regret", "memory_erase", "feelbadly", "memorableB",
        "describe_felt", "difficult",
    ],
    # Family.
    "family": ["family_relationship"],
    # Work, decisions and their consequences.
    "life_events_decisions": [
        "still_working_on_X", "hard_decision", "change_directions",
        "landed_trouble", "avoid", "happy_didthat", "great_situation",
    ],
    # Topic-neutral deepening probes. These attach to ANY thread, so they
    # index depth of probing rather than subject matter. Kept separate for
    # exactly that reason -- lumping them into a topic would be wrong.
    "generic_probe": ["why2", "tell_about_that", "give_example"],
}


def main():
    kind = json.load(open(f"{OUT}/e4a_prompt_kind.json"))
    question_like = set(kind["question_like"])
    assoc = {r["id"]: r for r in json.load(open(
        f"{ROOT}/experiments/a2-ellie-counterfactual/results/"
        f"a2a_prompt_label_association.json"))["records"]}

    # ---- integrity checks (fail loudly rather than silently mis-partition)
    flat = [p for ps in CLUSTERS.values() for p in ps]
    dupes = [p for p, n in collections.Counter(flat).items() if n > 1]
    missing = sorted(question_like - set(flat))
    extra = sorted(set(flat) - question_like)
    assert not dupes, f"prompt assigned to >1 cluster: {dupes}"
    assert not missing, f"question-like prompt unassigned: {missing}"
    assert not extra, f"clustered a non-question prompt: {extra}"
    print(f"partition OK: {len(flat)} prompts over {len(CLUSTERS)} clusters, "
          f"exhaustive and disjoint\n")

    of = {p: c for c, ps in CLUSTERS.items() for p in ps}

    # ---- per-cluster diagnostics
    print(f"{'cluster':24s} {'k':>3s} {'mean OR':>8s} {'min':>6s} {'max':>6s}")
    print("-" * 56)
    for c, ps in CLUSTERS.items():
        ors = [assoc[p]["odds_ratio"] for p in ps if p in assoc]
        print(f"{c:24s} {len(ps):3d} {np.mean(ors):8.2f} "
              f"{min(ors):6.2f} {max(ors):6.2f}")

    # ---- viability: how many participants would each held-out fold contain?
    rows = [json.loads(l) for l in open(COND)]
    dom, shares = {}, []
    for r in rows:
        ids = [m.group(1) for m in
               (PAT.match(l) for l in (r["text_ellie_only"] or "").split("\n"))
               if m]
        asked = [of[i] for i in ids if i in of]
        if not asked:
            continue
        cnt = collections.Counter(asked)
        top, n = cnt.most_common(1)[0]
        dom[r["uid"]] = {"cluster": top, "share": n / len(asked),
                         "split": r["split"], "y": r["phq8_binary"],
                         "n_clustered_prompts": len(asked)}
        shares.append(n / len(asked))

    print(f"\nparticipants with clusterable prompts: {len(dom)} / {len(rows)}")
    print(f"dominant-cluster share: median {np.median(shares):.2f} "
          f"max {max(shares):.2f}")
    print(f"\n{'held-out cluster':24s} {'n':>4s} {'dep':>4s} {'ctrl':>5s} "
          f"{'viable?':>8s}")
    print("-" * 52)
    fold = collections.defaultdict(list)
    for u, d in dom.items():
        fold[d["cluster"]].append(d)
    for c in CLUSTERS:
        f = fold.get(c, [])
        pos = sum(d["y"] for d in f)
        ok = "yes" if len(f) >= 20 and pos >= 5 and len(f) - pos >= 5 else "NO"
        print(f"{c:24s} {len(f):4d} {pos:4d} {len(f)-pos:5d} {ok:>8s}")
    print("\nviable = >=20 participants with >=5 in each class")

    json.dump({"clusters": CLUSTERS, "of_prompt": of,
               "dominant_cluster": dom},
              open(f"{OUT}/e4a_prompt_clusters.json", "w"), indent=2)
    print(f"\nwrote {OUT}/e4a_prompt_clusters.json")


if __name__ == "__main__":
    main()
