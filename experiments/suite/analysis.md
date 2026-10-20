# Evaluation suite — E2a / E5 / E4a build

Built 2026-08-23. Step 3 of `INTERVENTION_DESIGN.md` §7.
Code: `code/build_suite.py` · artefacts: `results/` · log: `logs/build_suite.log`.

---

## E2a — natural conflict set: **built, but underpowered**

Leave-one-out out-of-fold predictions from the A2a question-selection probe;
conflict = the probe's prediction disagrees with gold.

| quantity | value |
|---|---|
| participants carrying scripted IDs | 123 / 188 |
| LOO agreement with gold | 0.764 |
| **conflict n** | **29** (12 gold=1/pred=0 · 17 gold=0/pred=1) |
| by split | train 19 · dev 5 · **test 5** |

**Five test-split items is not an evaluation set.** The design estimated ~45–60
conflict cases; the true number is 29 because the probe agrees with gold 76 % of
the time. E2a can only be used pooled (n=29) with the LOO threshold device, and
then only on the zero-shot arm where that device is legitimate (I1 §5). It
cannot serve the encoder arm at all.

**E2b remains the usable conflict set** — 188 items, constructed rather than
found, already on disk. E2a is a natural-data supplement, not a replacement.

⚠ E2a additionally inherits F6: the ID-carrying subset has a depression rate of
**0.228 vs 0.293 overall**, so it is not label-representative.

---

## E5 — construct-preservation pairs: **built, clean**

The mirror of A2. A2 fixed the participant and swapped the interviewer; E5
fixes the interviewer and swaps the participant.

| quantity | value |
|---|---|
| pairs | **200** (from 202 draws) |
| distinct interviewer contexts *j* | 128 |
| \|Δ participant turns\| within pair | median 7, max 15 (tolerance 15) |

Each pair shares one interviewer context byte-identically and differs only in
whose participant turns are spliced in; gold follows the participant. Interviewer
text is inserted with the same `interleave` helper A2b used, so positions are
computed identically across the whole suite.

`construct sensitivity := mean |p(Y | C_i, S_j) − p(Y | C_i', S_j)|`

This is the RQ3 instrument and the design's pre-committed failure criterion
(§6 F-A) depends on it.

---

## E4a — clustering exposed a problem with the A2a feature set

The keyword clustering left 39 of 75 prompt IDs unmatched. Inspecting them
showed why: **28 of the 75 "scripted prompt IDs" are not questions at all.**
They are Ellie's backchannels, and many are affectively valenced reactions to
what the participant just disclosed — the DAIC authors' own IDs say so:

| valence | IDs |
|---|---|
| negative | `im_sorry`, `sorry_hear`, `that_sucks`, `been_hard`, `downer`, `hmm_downer`, `yeah_downer`, `isee_downer` |
| positive | `good_hear`, `thats_great`, `awesome`, `nice`, `sounds_interesting`, `cool3` |
| neutral | `uh_huh`, `mm`, `yeah2`, `right2`, `wow`, `(laughter)` |

This raised a real threat to A2a's headline: if the AUC 0.753 "question
selection" probe is partly reading affective backchannel, then the claim is not
about question selection but about interviewer reaction — a different and more
direct leak.

### It is not. The claim survives and improves.

Same CV protocol as A2a (10 seeds × 5-fold, `class_weight=balanced`, C=0.5):

| feature set | k | CV AUC | sd |
|---|---|---|---|
| all 75 — **A2a as published** | 75 | 0.753 | 0.025 |
| **question-like only** | **47** | **0.773** | 0.026 |
| backchannel-like only | 28 | 0.575 | 0.038 |

**Question selection alone reaches 0.773 — above the published 0.753.**
Backchannels alone reach 0.575, barely off the floor. The backchannel features
were diluting the probe, not driving it.

Three consequences:

1. **A2a's claim is strengthened, not threatened.** The number goes up and the
   feature set becomes principled. It also pre-empts an obvious reviewer
   objection with a measured answer rather than an argument.
2. **The A2a probe should be refit on the 47 question-like IDs** wherever it is
   used downstream — as the M1 PoE bias model, as the M2 group assignment, and
   for the E2a conflict selection above.
3. **E4a should cluster only the 47.** Clustering backchannels for a
   protocol-shift split is meaningless — they are not part of the protocol.

Split recorded in `results/e4a_prompt_kind.json`. The cluster proposal in
`results/e4a_prompt_clusters_PROPOSAL.json` is **not frozen** and still requires
the hand-review the design mandates (§8.1), now over 47 IDs rather than 75.

---

## E4a re-clustered — and the design does not survive it

`code/build_e4a_clusters.py`, output `results/e4a_prompt_clusters.json`.
Hand-authored partition of the 47 question-like prompts into 9 topical
clusters; validated programmatically as exhaustive and disjoint. Clusters are
topical, never association-based — grouping by odds ratio would build the label
into the partition.

| cluster | k | mean OR |
|---|---|---|
| clinical_care | 10 | 5.20 |
| parenting | 4 | 2.59 |
| generic_probe | 3 | 1.59 |
| la_relocation | 7 | 1.29 |
| travel_leisure | 4 | 1.13 |
| life_events_decisions | 7 | 1.07 |
| self_concept | 4 | 0.87 |
| affect_and_memory | 7 | 0.71 |
| family | 1 | 0.25 |

The OR gradient is a clean sanity check: `clinical_care` at 5.20 against
`family` at 0.25, with the default-script clusters in between, reproduces the
A2a story at cluster resolution without having been fitted to it.

### Failure 1 — leave-one-cluster-out folds are too small

| held-out cluster | n | dep | ctrl | viable |
|---|---|---|---|---|
| la_relocation | 41 | 10 | 31 | ✅ |
| affect_and_memory | 41 | 3 | 38 | ❌ class balance |
| clinical_care | 23 | 10 | 13 | ✅ |
| generic_probe | 15 | 4 | 11 | ❌ |
| life_events_decisions | 2 | 1 | 1 | ❌ |
| parenting | 1 | 0 | 1 | ❌ |
| travel_leisure / self_concept / family | 0 | — | — | ❌ |

**Only 2 of 9 clusters give a usable fold.** The cause is visible in one
statistic: the **median dominant-cluster share is 0.24**. No DAIC interview is
dominated by a single topic — every interview walks through most of the
protocol. "Participants whose interview is dominated by cluster *k*" is not a
partition this corpus supports.

### Failure 2 — exposure-based splits are confounded with the label *by construction*

The obvious repair is to split on exposure to the clinical thread rather than
dominance. It fails worse:

| split rule | exposed n (dep) | unexposed n (dep) |
|---|---|---|
| asked ≥1 BH-significant clinical prompt | 45 (**21 = 47 %**) | 78 (**7 = 9 %**) |
| asked ≥2 clinical_care prompts | 98 (25) | 25 (3) |
| asked ≥3 clinical_care prompts | 54 (**24 = 44 %**) | 69 (**4 = 6 %**) |

The two folds differ in base rate by a factor of seven. That is not protocol
shift — it is **label shift**, and it is unavoidable: the DAIC interview is
adaptive, so exposure to the clinical thread *is* the clinical judgement. The
two cannot be separated within this corpus.

**Consequence, and it is a real one.** `findings.md` open question 5 asks
whether question selection is a shortcut or legitimate evidence, and states
that the operational test must be generalisation under interviewer/protocol
shift. **That test cannot be constructed from DAIC alone.** Answering it
requires a second corpus with a different interview protocol. This should be
stated as a limitation rather than approximated with a confounded split.

### Replacement — cluster-resolved ablation

What *is* viable is ablation at cluster resolution: remove cluster *k*'s prompts
from every transcript and measure the change. No fold, no split, all 123
participants retained.

| cluster | participants asked ≥1 | coverage |
|---|---|---|
| affect_and_memory | 123 | 100 % |
| la_relocation | 122 | 99 % |
| self_concept | 120 | 98 % |
| life_events_decisions | 119 | 97 % |
| travel_leisure | 118 | 96 % |
| clinical_care | 115 | 93 % |
| generic_probe | 114 | 93 % |
| family | 96 | 78 % |
| parenting | 25 | 20 % |

This is a cluster-resolved A1 — it localises *which topical thread* carries the
effect, and pairs naturally with the D1/D1m result that the damage is lexical.
It is a sensitivity analysis, not a generalisation test, so **it does not
answer RQ4**. With E4a dead, **E4b (IMHI → DAIC transfer) becomes the only OOD
evidence in the design.**

---

## Suite frozen 2026-08-24 — with the history visible

`code/freeze_suite.py` → `results/SUITE_FREEZE.json`. sha256 + shape + provenance
for every artefact.

**The freeze is retroactive and the manifest says so.** `INTERVENTION_DESIGN.md`
§7 placed it before any method ran; it did not happen then, and the suite changed
during the campaign. The manifest therefore guarantees **reproducibility and
forward stability**, not that targets were fixed before results were seen. What
actually changed:

| artefact | change | cause |
|---|---|---|
| E2a | v1 (75 feat) → v2 (47 feat) | 28 of 75 features were backchannels; question-only probe scores 0.773 vs 0.753 |
| E4a clusters | proposal → hand-authored | keyword matching left 39/75 unassigned |
| E3 pools | two arms added post-hoc | length-matched control demanded by A2's own limitations |
| E3 selection rule | corrected **before** scoring | v1 matched length alone and swapped OR-10.4 prompts for OR-2.3; **no results exist under v1** |

### The freeze exposed a fact the analyses had obscured

**E2a was never used. Neither version.**

| artefact | status |
|---|---|
| `e2a_natural_conflict.json` (v1) | built, superseded, **never scored against** |
| `e2a_natural_conflict_q47.json` (v2) | built, **never scored against** |
| `e4a_prompt_clusters_PROPOSAL.json` | superseded same day |

E2a was constructed twice — once in the original suite build, once refit on the
47 question-like features — and no experiment ever evaluated a method on it. I1
used **E2b** (the constructed conflict set, 188 items derived from the A2b arms),
not E2a. The earlier note that E2a was a "supplement" understated it: it is
**dead weight**, and the reason is the one recorded at build time — 5–6 test-split
items cannot support an evaluation.

This is exactly the kind of thing a manifest is for. Neither analysis file was
wrong, but between them it was impossible to see that a set had been built,
revised, and never consumed.

## L3 verification — falsified, and E4b dies with it

Run 2026-08-24. `logs/l3_verification.log` · `results/l3_verification.json`.

L3 (`INTERVENTION_DESIGN.md` §0) asserted that the DAIC shortcut's transferable
form is **clinical-vocabulary salience**, present in IMHI as self-diagnosis and
treatment terms. It was the bridge making "DAIC for evaluation, IMHI for
training volume" coherent, and it was never measured. Measured now:

| source | n | AUC, clinical markers [95 % CI] | AUC, symptom words | % posts w/ any clinical marker |
|---|---|---|---|---|
| DR | 248 | **0.583** [0.546, 0.621] | 0.698 | 14.1 % |
| Irf | 300 | 0.523 [0.486, 0.560] | 0.535 | 12.3 % |
| MultiWD | 300 | 0.521 [0.480, 0.564] | 0.529 | 15.3 % |
| dreaddit | 300 | 0.523 [0.488, 0.558] | 0.600 | 10.3 % |
| loneliness | 265 | 0.510 [0.467, 0.555] | 0.609 | 15.8 % |
| **DAIC (reference)** | 123 | **0.773** | — | — |

Clinical markers (diagnosis / therapy / medication / clinician terms) predict at
**0.510–0.583**. Only DR clears chance at all, and barely. Against DAIC's 0.773
this is not a weaker version of the same phenomenon — it is **absence**. Only
10–16 % of posts contain any clinical marker, so the feature is too sparse to
carry a shortcut even in principle.

Symptom words do better (0.529–0.698), exactly as they should: symptoms **are**
the construct, not a side channel.

### The deeper reason, which was visible in advance

DAIC's shortcut lives in a **side channel** — the interviewer's questions, which
are not the patient's evidence, which is why deleting them (D1) *raises*
construct sensitivity by 76 %. **IMHI posts have no side channel.** Clinical
vocabulary in a Reddit post is written by the subject and is legitimate evidence.
So L3 was not merely unverified, it was close to **ill-posed**: it proposed
transferring a side-channel phenomenon to a corpus with no side channel.

### Consequences

1. **E4b is dropped.** Training on IMHI against a clinical-vocabulary shortcut
   and testing transfer to DAIC is incoherent when no such shortcut exists in
   IMHI.
2. **The design has no OOD result at all.** E4a was not constructible, I4's
   cluster ablation returned a null, E4b is now dropped. **RQ4 is unanswered**
   and the paper must say so rather than substitute a within-corpus proxy.
3. **L2's rationale is weakened.** "IMHI supplies training volume" only made
   sense via L3. The encoder arm was in fact trained entirely on DAIC-107, so
   nothing already run depends on this — but the option is closed.

## Status

| set | state |
|---|---|
| E1 / E2b | ✅ derived at analysis time from `a1_*`/`a2b_*`; no separate file |
| E5 | ✅ 200 pairs · consumed by I3, I3-variants, I5 |
| E3 | ✅ 5 arms incl. 2 length-matched · consumed by I2 |
| E2a | ⚠️ built twice, **never used** — 5–6 test items |
| E4a | ❌ **not constructible from DAIC** — replaced by I4, which returned a null |
| E4b | ❌ **dropped** — L3 falsified, no clinical-vocabulary shortcut in IMHI |
| **RQ4 (OOD)** | ❌ **unanswered — no route remains** |

**Open, needs a decision:** whether to refit the A2a probe on the 47
question-like features everywhere downstream. Recommended — it is both more
accurate and more defensible — but it changes the published 0.753 to 0.773 and
that number appears in `findings.md`, `RESULTS.md` and the progress report.
