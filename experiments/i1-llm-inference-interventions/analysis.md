# I1 — LLM-arm inference-time interventions

Run 2026-08-23. Step 1 of `INTERVENTION_DESIGN.md` §7.
Code: `code/run_i1.py` · results: `results/i1_summary.json` · log:
`logs/i1_llm_interventions.log`.

No GPU and no model load. Every number re-scores log-probabilities already on
disk from A1 and A2b; nothing here is a new measurement of the model, only a
new **decision rule** applied to existing measurements. MentaLLaMA-chat-7B
throughout.

---

## Headline

**Deleting the interviewer's turns at inference is the whole fix, and it is
free.** Paired over 188 participants, D1 beats no-intervention by
**+0.126 AUC [+0.043, +0.210]**, and on the conflict set it converts a model
that is *actively worse than chance* (AUC 0.329) into one at **0.777**.

No weights change. No training. No labels at inference.

---

## 1. Input-level interventions

Leave-one-out cross-fit threshold over all 188 participants — the threshold
scoring participant *i* is fitted on the other 187, so nothing is oracle. AUC
is threshold-free and therefore unfitted entirely. `*` = bootstrap lower bound
clears the label-shuffle floor (0.576).

| method | AUC [95 % CI] | acc | P(pred=Yes) |
|---|---|---|---|
| **P0** no intervention (`full`) | 0.663 [0.569, 0.749] | 0.739 | 0.128 |
| **D1** delete interviewer turns | **0.787\*** [0.719, 0.854] | 0.745 | 0.176 |
| **D1m** mask interviewer words, keep turns | **0.763\*** [0.683, 0.835] | 0.771 | 0.096 |
| — shortcut only (`ellie_only`) | 0.546 [0.456, 0.638] | 0.707 | 0.000 |

**Paired D1 − P0 = +0.126 [+0.043, +0.210]** — CI excludes zero.

Two things follow.

**P0 does not clear the noise floor; D1 and D1m do.** With the interviewer
present the model's ranking of patients is not statistically distinguishable
from chance. Remove the interviewer and it is.

**D1m recovers 0.763 of D1's 0.787 while leaving every interviewer turn in
place.** Only the words are replaced. So the damage is carried by the
**lexical content** of the questions, not by their presence, position or
count — an independent corroboration of A2's "keys on surface clinical
vocabulary" result, from a different experiment on different inputs.

---

## 2. The conflict set is where the effect is largest

E2b assigns each patient the interviewer context that points *away* from their
label: negatives get the clinical pool, positives the biographical one.

| method | AUC [95 % CI] | acc |
|---|---|---|
| **P0** uncorrected | **0.329** [0.246, 0.415] | 0.702 |
| **P3-c** two-pass / S-free score | **0.777\*** [0.701, 0.842] | 0.734 |

**P0's CI lies entirely below 0.5.** Under adversarial interviewer context the
model is not merely uninformative — it is *anti-correlated with the truth*. The
ranking inverts, because inflating every negative and deflating every positive
is exactly what the counterfactual arms do.

Ignoring the interviewer channel restores **+0.448 AUC**. This is the single
largest intervention effect in the project.

---

## 3. Counterfactual sensitivity

Mean `|p(Yes | C, S_clinical) − p(Yes | C, S_biographical)|`, test split, C
byte-identical:

| method | CF-Sens | [95 % CI] |
|---|---|---|
| P0 uncorrected | 0.1932 | [0.1714, 0.2170] |
| P3-b dev-fit shift | 0.0702 | [0.0588, 0.0822] |
| **P3-c two-pass** | **0.0000** | [0.0000, 0.0000] |

P3-c is exactly zero **by construction**, not by measurement: it discards `S`,
so no change to `S` can move it. That is worth stating plainly rather than
presenting as an empirical win — the interesting quantity is what it costs
elsewhere, and §1 says the cost is negative (D1 *gains* 0.126 AUC).

---

## 4. A methodological finding: P3-b is redundant with threshold fitting

The dev-fit shifts are real and correctly signed — `cf_neutral` −0.312,
`cf_empirical` +0.194, `cf_depression` +0.551, tracking the A2 arm ordering.
But **within a single arm, P3-b changes nothing**: AUC is identical to P0
(0.728 / 0.719 / 0.711) because subtracting a constant is a monotone transform,
and accuracy is identical too because the threshold was *also* fitted on dev
and simply absorbs the constant.

A per-arm additive correction and a per-arm fitted threshold are the same
degree of freedom. P3-b only earns its keep where a **single** threshold must
serve items drawn from different arms — i.e. on E2b, where it lifts AUC from
0.339 to 0.719 on the test split.

**Consequence for the design:** `INTERVENTION_DESIGN.md` §4 presents P3-b and
P3-c as successive refinements. They are not on a ladder. P3-c dominates —
simpler, label-free, and strictly better on every metric here — and P3-b should
be reported as the negative methodological result above rather than as a method.

---

## 5. Power: why the test split alone cannot carry this

Reported on the 47-participant test split, **nothing is distinguishable**:

| method | AUC on test (n=47) [95 % CI] |
|---|---|
| P0 | 0.746 [0.561, 0.896] |
| D1 | 0.789 [0.634, 0.920] |
| D1m | 0.762 [0.579, 0.914] |

Every CI spans ~0.3 AUC, all three overlap almost completely, and **none clears
the shuffle floor**. The apparent P0/D1 gap is +0.043 here versus +0.126 over
188 — the split estimate is noise-dominated.

The LOO cross-fit in §1 is the honest resolution and not a loosening of
discipline: the model is **zero-shot** and never trained on any DAIC split, so
the only fitted quantity anywhere in this experiment is a scalar threshold,
cross-fitted leave-one-out. AUC needs no fit at all.

**Carry this forward.** The encoder arm *does* train, so it cannot use this
device — B0-mech and everything downstream are bound to the 107/34/47 splits
and will be correspondingly underpowered. That asymmetry has to be stated in
the paper rather than papered over by quoting n=188 for one arm and n=47 for
the other.

---

## 6. What is solid, what is not

**Solid:**
- D1 − P0 = +0.126 [+0.043, +0.210], paired, CI excludes zero.
- E2b: 0.329 [0.246, 0.415] → 0.777 [0.701, 0.842]. Non-overlapping, and the
  uncorrected CI sits wholly below chance.
- D1m ≈ D1 (0.763 vs 0.787) ⇒ the shortcut is lexical, not structural.
- P0 fails to clear the noise floor while D1/D1m clear it.

**Not solid:**
- Accuracy differences among P0 / D1 / D1m (0.739 / 0.745 / 0.771) are inside
  their CIs. Only the AUC and conflict-set claims are supported.
- CF-Sens is computed on the test split (n=47) while §1 uses all 188. Tight CIs,
  but the inconsistency should be resolved before publication.
- One model. Menta is saturated and excluded by design; MentalBERT is untested.

**Scope:**
- `cf_none` (A2b) and `participant_only` (A1) are conceptually the same
  condition but come from different runs with different truncation budgets
  (4032 minus pool reserve vs 4032). They are not byte-identical and their AUCs
  differ slightly (0.777 vs 0.787). Neither substitutes for the other in a
  table.
- E2b uses fixed pools, not each patient's real questions, so it measures the
  effect of *a* conflicting context rather than of naturally occurring conflict.
  E2a (natural conflict) is still unbuilt.

---

## 7. What this does to the design

1. **The LLM arm is done, and it is a positive result.** D1 is the method;
   P3-c is the same correction viewed from the output side; P3-b is a negative
   methodological note. Only P1/P2 (prompting) remain untested on this arm.
2. **P3-b demoted** — see §4.
3. **The next decision is B0-mech** (§3 of the design). Everything above says
   the zero-shot mechanism is a lexically-triggered threshold offset repairable
   for free at the input. Whether a *trained* encoder behaves the same way is
   what decides if the train-time arm exists at all.
