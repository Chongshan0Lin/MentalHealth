#!/usr/bin/env python3
"""I2 -- E3 token-matched counterfactual arms + P1 anti-shortcut prompting.

Steps 4 and 6 of INTERVENTION_DESIGN.md §7, run together because both are
prompt-level variations over the same model load.

PART A -- E3 token-matched arms
  A2's headline (+0.206) is safe: the clinical pool is SHORTER than the
  biographical one, so length runs against the effect. What is NOT safe is the
  "model reads the wrong half" dissociation, which sets cf_empirical (31 words)
  against cf_depression (38). A2's own analysis calls it suggestive, not
  established. These two arms equalise word count so the comparison carries.

  Matched pools are SELECTED, not invented: candidates are real corpus prompts
  drawn from the relevant e4a cluster (biographical) or ranked by A2a odds
  ratio (empirical), and the 5-subset whose total word count lands closest to
  the cf_depression reference is chosen. The rule is mechanical so the pools
  cannot be tuned toward a result.

PART B -- P1 anti-shortcut prompting
  The instruction is fixed verbatim in INTERVENTION_DESIGN.md §4 and is not
  edited after seeing any result. Applied to the real transcript and to the two
  confirmatory counterfactual arms, so P1 is measured both on IID input and
  under adversarial interviewer context.

Scoring is teacher-forced label log-likelihood throughout (EXPERIMENT_DESIGN.md
§0) -- no free generation, so no parse-failure confound. P2 (CoT) is NOT in
this script: it requires a generation path the harness does not have.
"""
from __future__ import annotations

import argparse, importlib.util, itertools, json, os, sys, time

sys.path.insert(0, "/data/cl195/NeurIPS_workshop")
from shortcut_harness.scoring import TextScorer

ROOT = "/data/cl195/NeurIPS_workshop"
COND = f"{ROOT}/data/processed/daic_conditions.jsonl"
OUT = f"{ROOT}/experiments/i2-e3-matched-and-prompting/results"
LABELS = ["Yes", "No"]
LABEL_OF = {1: "Yes", 0: "No"}

TEMPLATE = ('Consider this post: "{text}" '
            'Question: Does the poster suffer from depression?')

# Locked in INTERVENTION_DESIGN.md §4 before any result was seen.
P1_INSTRUCTION = (
    "Base your prediction only on clinically relevant evidence in the "
    "participant's own words. Do not infer depression from the interviewer's "
    "choice of questions, from sentiment, or from writing style. ")

_spec = importlib.util.spec_from_file_location(
    "run_a2b", f"{ROOT}/experiments/a2-ellie-counterfactual/code/run_a2b.py")
_a2b = importlib.util.module_from_spec(_spec)
sys.modules["run_a2b"] = _a2b
_spec.loader.exec_module(_a2b)
POOLS, interleave = dict(_a2b.POOLS), _a2b.interleave


def wc(pool):
    return len(" ".join(pool).split())


def build_matched_pools(verbose=True):
    """Select 5-prompt pools whose word count matches cf_depression."""
    target = wc(POOLS["cf_depression"])
    assoc = {r["id"]: r for r in json.load(open(
        f"{ROOT}/experiments/a2-ellie-counterfactual/results/"
        f"a2a_prompt_label_association.json"))["records"]}
    clusters = json.load(open(
        f"{ROOT}/experiments/suite/results/e4a_prompt_clusters.json"))
    kind = json.load(open(f"{ROOT}/experiments/suite/results/e4a_prompt_kind.json"))
    qlike = set(kind["question_like"])

    texts = {}
    import re
    PAT = re.compile(r"^Ellie: ([A-Za-z][A-Za-z0-9_]*) \((.*)\)\s*$")
    for line in open(COND):
        r = json.loads(line)
        for l in (r["text_ellie_only"] or "").split("\n"):
            m = PAT.match(l)
            if m:
                texts.setdefault(m.group(1), m.group(2))

    import math

    def pick(cands, target, objective):
        """Length-matched subset that best serves `objective`.

        Matching on length ALONE is not enough for the empirical pool: the
        length-closest subset happily swaps OR 10.4 prompts for OR 2.3 ones,
        which trades a length confound for an association-strength confound and
        destroys the very contrast the arm exists to make. So length is a
        constraint (within 1 word) and the association is the objective.
        """
        cands = sorted(cands)
        feasible = [c for c in itertools.combinations(cands, 5)
                    if abs(wc([texts[x] for x in c]) - target) <= 1]
        if not feasible:                      # fall back to closest length
            return min(itertools.combinations(cands, 5),
                       key=lambda c: abs(wc([texts[x] for x in c]) - target))
        return max(feasible, key=objective)

    def mean_log_or(combo):
        return sum(math.log(max(assoc[p]["odds_ratio"], 1e-9))
                   for p in combo) / len(combo)

    # biographical candidates: the default-script topical clusters. Objective
    # is neutrality -- odds ratio closest to 1 in log space.
    bio = [p for c in ("la_relocation", "travel_leisure")
           for p in clusters["clusters"][c] if p in texts and p in assoc]
    # empirical candidates: the 20 most label-associated question-like prompts,
    # from which the length-feasible subset with the strongest association wins
    emp = [p for p in sorted(
        [q for q in qlike if q in assoc],
        key=lambda x: -assoc[x]["odds_ratio"])[:20] if p in texts]

    nb = pick(bio, target, lambda c: -abs(mean_log_or(c)))
    ne = pick(emp, target, mean_log_or)
    POOLS["cf_neutral_matched"] = [texts[p] for p in nb]
    POOLS["cf_empirical_matched"] = [texts[p] for p in ne]

    if verbose:
        print(f"E3 matched pools (target = cf_depression = {target} words)\n")
        for nm in ["cf_depression", "cf_neutral", "cf_empirical",
                   "cf_neutral_matched", "cf_empirical_matched"]:
            print(f"  {nm:22s} {wc(POOLS[nm]):3d} words")
        print(f"\n  cf_neutral_matched   <- {list(nb)}")
        print(f"  cf_empirical_matched <- {list(ne)}")
        import math
        for nm, sel in [("cf_neutral_matched", nb), ("cf_empirical_matched", ne)]:
            g = math.exp(sum(math.log(max(assoc[p]["odds_ratio"], 1e-9))
                             for p in sel) / len(sel))
            print(f"\n  {nm}  geometric-mean OR = {g:.2f}")
            for p in sel:
                print(f"      OR={assoc[p]['odds_ratio']:5.2f}  {texts[p][:52]}")
        import math as _m
        for nm in ["cf_empirical", "cf_neutral"]:
            print(f"  (reference {nm}: pools fixed in run_a2b.py)")
        print()
    return POOLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MentaLLaMA-chat-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=int, default=4032)
    ap.add_argument("--pools-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pools = build_matched_pools()
    json.dump({k: v for k, v in pools.items()},
              open(f"{OUT}/e3_pools.json", "w"), indent=2)
    if args.pools_only:
        return

    rows = [json.loads(l) for l in open(COND)]
    sc = TextScorer(f"{ROOT}/checkpoints/{args.model}", device=args.device)
    print(f"loaded {args.model}", flush=True)

    # identical budget policy to A2b: C truncated ONCE and reused across arms,
    # so no arm differs in how much participant text the model saw
    longest = max(len(" ".join(p)) for p in pools.values())
    budget = args.budget - (longest // 3) - 32 - len(P1_INSTRUCTION) // 3

    # PART A arms (no instruction) + PART B arms (instruction prepended)
    JOBS = [("E3", "cf_neutral_matched", False),
            ("E3", "cf_empirical_matched", False),
            ("P1", "cf_none", True),
            ("P1", "cf_neutral", True),
            ("P1", "cf_depression", True)]

    out_path = f"{OUT}/i2_{args.model}.jsonl"
    n, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            gold = LABEL_OF[r["phq8_binary"]]
            ids = sc.tok(r["text_participant_only"],
                         add_special_tokens=False).input_ids
            trunc = len(ids) > budget
            C = sc.tok.decode(ids[:budget]) if trunc else r["text_participant_only"]

            for part, arm, instructed in JOBS:
                text = interleave(C, pools.get(arm, []))
                prompt = TEMPLATE.format(text=text)
                if instructed:
                    prompt = P1_INSTRUCTION + prompt
                scored = sc.score(prompt, LABELS)
                lp = {k: v["mean"] for k, v in scored.items()}
                pred = max(lp, key=lp.get)
                f.write(json.dumps({
                    "uid": r["uid"], "participant_id": r["participant_id"],
                    "split": r["split"], "model": args.model,
                    "part": part, "arm": arm, "instructed": instructed,
                    "gold": gold, "phq8_binary": r["phq8_binary"],
                    "logprobs": lp, "pred": pred, "correct": pred == gold,
                    "c_truncated": trunc, "c_tokens": min(len(ids), budget),
                }) + "\n")
                n += 1
            f.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(rows)} | {n} scored | {el:.0f}s "
                      f"({el/(i+1):.1f}s/participant)", flush=True)

    print(f"\nwrote {out_path}  ({n} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
