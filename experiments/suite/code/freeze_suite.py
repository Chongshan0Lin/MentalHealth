#!/usr/bin/env python3
"""Freeze the evaluation suite: checksums, shapes, and an honest version history.

Step 5 of INTERVENTION_DESIGN.md §7. The design placed this BEFORE any method
ran. It did not happen then, and the suite changed afterwards, so this manifest
does not and cannot certify pre-registration. It records three things instead:

  1. exactly what each artefact is now (sha256, row counts, parameters),
  2. how each one got that way, with dates and causes,
  3. which experiment consumed which version.

Point 3 is the one that matters operationally: E2a exists in two versions and
nothing in the analyses states which produced which number. Writing that down is
worth more than a manifest that implies a clean freeze.
"""
from __future__ import annotations

import hashlib, json, os, subprocess

ROOT = "/data/cl195/NeurIPS_workshop"
OUT = f"{ROOT}/experiments/suite/results/SUITE_FREEZE.json"

ARTEFACTS = {
    "source_transcripts": "data/processed/daic_conditions.jsonl",
    "e5_construct_pairs": "experiments/suite/results/e5_construct_pairs.jsonl",
    "e2a_conflict_v1_75feat": "experiments/suite/results/e2a_natural_conflict.json",
    "e2a_conflict_v2_47feat": "experiments/suite/results/e2a_natural_conflict_q47.json",
    "e4a_clusters": "experiments/suite/results/e4a_prompt_clusters.json",
    "e4a_prompt_kind": "experiments/suite/results/e4a_prompt_kind.json",
    "e4a_clusters_SUPERSEDED": "experiments/suite/results/e4a_prompt_clusters_PROPOSAL.json",
    "e3_pools": "experiments/i2-e3-matched-and-prompting/results/e3_pools.json",
    "a2b_pool_definitions": "experiments/a2-ellie-counterfactual/code/run_a2b.py",
    "a2a_associations": "experiments/a2-ellie-counterfactual/results/a2a_prompt_label_association.json",
}

# What each artefact is, how it came to be, and what consumed it.
HISTORY = {
    "e5_construct_pairs": {
        "created": "2026-08-23",
        "by": "experiments/suite/code/build_suite.py",
        "params": {"N_PAIRS": 200, "SPAN_TOL": 15, "seed": 0},
        "changed_since": "no",
        "consumed_by": ["I3 (E5 base + P1)", "I3 variants (D1/D1m)",
                        "I5 (E5 under CoT)"],
    },
    "e2a_conflict_v1_75feat": {
        "created": "2026-08-23",
        "by": "experiments/suite/code/build_suite.py",
        "params": {"features": 75, "probe": "A2a all prompt IDs",
                   "conflict_n": 29, "test_split_n": 5},
        "changed_since": "SUPERSEDED by v2 on 2026-08-24",
        "cause": "28 of the 75 features were found to be backchannels, not "
                 "questions; the question-only probe scores higher (0.773 vs "
                 "0.753)",
        "consumed_by": [],
    },
    "e2a_conflict_v2_47feat": {
        "created": "2026-08-24",
        "by": "inline refit, logs/a2a_refit_47.log",
        "params": {"features": 47, "probe": "question-like only",
                   "cv_auc": 0.773, "conflict_n": 28, "test_split_n": 6},
        "changed_since": "no",
        "consumed_by": [],
    },
    "e4a_clusters": {
        "created": "2026-08-23",
        "by": "experiments/suite/code/build_e4a_clusters.py",
        "params": {"prompts": 47, "clusters": 9,
                   "partition": "hand-authored, validated exhaustive+disjoint"},
        "changed_since": "no",
        "consumed_by": ["I4 (cluster ablation)",
                        "I2 (biographical candidate pool for E3 matching)"],
    },
    "e4a_prompt_kind": {
        "created": "2026-08-23",
        "by": "inline, logs/build_suite.log follow-up",
        "params": {"question_like": 47, "backchannel_like": 28},
        "changed_since": "no",
        "consumed_by": ["I2 (E3 pool selection)", "I6 (M1/M2 bias model)",
                        "e2a v2 refit"],
    },
    "e4a_clusters_SUPERSEDED": {
        "created": "2026-08-23",
        "by": "experiments/suite/code/build_suite.py",
        "changed_since": "SUPERSEDED same day",
        "cause": "keyword matching left 39/75 unassigned because 28 were "
                 "backchannels",
        "consumed_by": [],
    },
    "e3_pools": {
        "created": "2026-08-24",
        "by": "experiments/i2-e3-matched-and-prompting/code/run_i2.py",
        "params": {"target_words": 38,
                   "selection": "length within 1 word (constraint), "
                                "association (objective)"},
        "changed_since": "selection rule was corrected BEFORE the scoring run: "
                         "the first version matched on length alone and swapped "
                         "OR-10.4 prompts for OR-2.3 ones. No results were "
                         "produced under the first version.",
        "consumed_by": ["I2 (E3 arms)"],
    },
    "a2b_pool_definitions": {
        "created": "2026-08-20",
        "params": {"cf_depression": 38, "cf_neutral": 50, "cf_empirical": 31},
        "changed_since": "no",
        "consumed_by": ["A2b", "I1", "B0-mech", "I5", "I6",
                        "readout controls"],
    },
}

# What the suite does NOT cover, stated so absence is not read as coverage.
NOT_FROZEN = {
    "E1_IID": "derived at analysis time from a1_*/a2b_* jsonl; no separate file",
    "E2b_constructed_conflict": "derived from a2b arms crossed against gold; "
                                "no separate file",
    "E4a_protocol_shift": "NOT CONSTRUCTIBLE from DAIC — protocol shift and "
                          "label shift are the same variable in an adaptive "
                          "interview (see ../analysis.md). Replaced by I4 "
                          "cluster-resolved ablation, which returned a null.",
    "E4b_cross_corpus": "never built; depends on L3, which remains asserted "
                        "rather than measured",
}


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()


def shape(path):
    if path.endswith(".jsonl"):
        with open(path) as f:
            return {"rows": sum(1 for _ in f)}
    if path.endswith(".json"):
        d = json.load(open(path))
        if isinstance(d, list):
            return {"items": len(d)}
        return {"keys": sorted(d)[:8], "n_keys": len(d)}
    return {"bytes": os.path.getsize(path)}


def main():
    man = {"frozen_at": subprocess.run(
        ["date", "-Iseconds"], capture_output=True, text=True).stdout.strip(),
        "caveat": (
            "RETROACTIVE. INTERVENTION_DESIGN.md §7 placed the freeze before "
            "any method ran; it did not happen then. The suite changed during "
            "the campaign (see per-artefact `changed_since`). This manifest "
            "guarantees reproducibility and forward stability, NOT that the "
            "targets were fixed before results were seen."),
        "artefacts": {}, "not_frozen": NOT_FROZEN}

    for name, rel in ARTEFACTS.items():
        p = f"{ROOT}/{rel}"
        if not os.path.exists(p):
            man["artefacts"][name] = {"path": rel, "MISSING": True}
            continue
        e = {"path": rel, "sha256": sha256(p), **shape(p)}
        e.update(HISTORY.get(name, {}))
        man["artefacts"][name] = e

    json.dump(man, open(OUT, "w"), indent=2)

    print(f"{'artefact':28s} {'shape':>12s}  sha256[:16]   status")
    print("-" * 92)
    for n, e in man["artefacts"].items():
        if e.get("MISSING"):
            print(f"{n:28s} {'--':>12s}  {'MISSING':16s}")
            continue
        sh = e.get("rows") or e.get("items") or e.get("n_keys") or e.get("bytes")
        cs = e.get("changed_since", "n/a")
        st = "SUPERSEDED" if "SUPERSEDED" in str(cs) else (
            "unused" if e.get("consumed_by") == [] else "active")
        print(f"{n:28s} {sh:>12}  {e['sha256'][:16]}  {st}")
    print(f"\nwrote {OUT}")
    print("\nUNUSED artefacts (built but never scored against):")
    for n, e in man["artefacts"].items():
        if e.get("consumed_by") == [] and not e.get("MISSING"):
            print(f"  - {n}")


if __name__ == "__main__":
    main()
