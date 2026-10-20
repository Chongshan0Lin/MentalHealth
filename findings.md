# Findings — do behavioral foundation models learn constructs, or shortcuts?

Last updated 2026-08-20, after A2. Companion to `RESULTS.md` (detailed numbers)
and the per-experiment `experiments/*/analysis.md`.

---

## Current understanding

The project's through-line changed at A2, and the change is the contribution.

**Through A1 and C1** the pattern looked like: *shortcuts exist in the data;
models don't use them.* Ellie's questions alone give AUC 0.810 while the models
score at the noise floor on them; source identity is recoverable at 0.437 while
Δ_prior is zero.

**A2 overturned the model half of that.** With the participant's own words held
byte-identical, swapping five interviewer questions from biographical to
clinically loaded moves MentaLLaMA's depression probability by **+0.206
[+0.195, +0.217]** and flips **58.5 %** of predictions — dropping accuracy from
0.750 to 0.495, *below* the 0.707 majority baseline.

A1 and A2 do not disagree. They ask different questions, and the pair is the
finding:

- **Ablation** asks *can the label be decoded from the side channel alone?* —
  No (AUC 0.546, at floor).
- **Counterfactual** asks *does the side channel move the prediction when
  content is fixed?* — Yes, enormously.

A model can be unable to read a signal and still be governed by it. Ablation is
structurally blind to that case, because removing the side channel removes any
entangled legitimate signal at the same time.

---

## Patterns and insights

### 1. The shortcut pathway is not "identify the corpus"

The framing we started with — *side signal → recognise the dataset → predict
from its label prior* — is **capped at the majority baseline**, and C1 measured
the cap exactly: knowing it is swmh buys 0.347, which *is* swmh's majority
rate, while the model with content reaches 0.730. The pathway explains none of
the gap and cannot account for shortcut learning.

The replacement (proposed by the PI, confirmed here) is:

> side signal → carries *other* label-correlated information → that drives the
> prediction

In DAIC the carrier is **question selection**. The interview is adaptive, so
which prompts the interviewer chose is itself a clinical judgement:
`why_seek_help` OR 11.26, `therapy_going` OR 10.42, `when_diagnosed` OR 8.24 —
all surviving BH correction. **A binary asked/not-asked matrix over 75 prompt
IDs predicts depression at AUC 0.753** with no text at all.

### 2. It is a bias shift, not a change in discrimination

The mechanism is precise and is what makes this a shortcut rather than ordinary
context sensitivity:

| quantity | cf_neutral → cf_depression |
|---|---|
| AUC (ranking) | 0.764 → 0.734 — **barely moves** |
| P(pred = Yes) (operating point) | 0.160 → **0.745** |
| Δ for depressed | +0.183 |
| Δ for **not** depressed | **+0.215** — moves *further* |

Interviewer context acts as an additive bias on the decision threshold. The
ordering of participants survives; the threshold is dragged across it, and
accuracy collapses.

### 3. The model keys on the wrong part of the side channel

The semantically loaded pool moves the model **more** (+0.206) than the pool
A2a proved to be genuinely label-predictive (+0.124). Questions that merely
*sound* clinical outweigh questions that actually carry diagnostic information.

The model uses the side channel and picks up the spurious component while
largely missing the informative one. This dissociation is the sharpest result
in the project.

### 4. Encoded ≠ used, but encoding is not a safe resting state

C1's hidden-state probes recover corpus identity above the lexical ceiling at
every depth (MentaLLaMA mean-pooled 0.477–0.493 vs TF-IDF 0.437) and the
information survives to the **final-layer last-token state** that drives
generation (0.460 / 0.469). The shape is a plateau, not a mid-layer peak, so
"computed then discarded" is not supported.

Combined with A2, the reading is that side-channel information is available at
the decision point and *is* acted on — just not through the corpus-identity
route we first tested.

---

## Lessons and constraints

Specific, actionable, learned the hard way:

- **Never audit a shortcut by ablation alone.** A1's null and A2's +0.206 come
  from the same model on the same corpus. Any benchmark cleared by ablation can
  still carry an effect this size.
- **Accuracy-based SS is unusable when the operating point is degenerate.**
  MentaLLaMA answers No to 87 % of full interviews, Menta Yes to 84 %; a
  constant classifier scores exactly 0.500 balanced accuracy. Use AUC, which is
  threshold-free and (verified) invariant to PMI normalisation — the raw and PMI
  AUC tables matching is a free correctness check on the scorer.
- **Stratified sampling destroys prior-based measurements.** Per-label
  stratification in C1 flattened every corpus prior to near-maximal entropy and
  hid the effect; rescoring the same content-free predictions against the
  natural distribution revealed exact majority-class hits in 4/10 sources.
- **Score labels inside their native answer frame.** IMHI multiclass gold is a
  full sentence (`This post shows the stress cause related to X`); scoring bare
  labels collapsed all four multiclass sources to a constant prediction with
  4-nat gaps between labels and 0.2–0.6 nat of variation across posts.
- **Menta is saturated and a poor probe for anything threshold-based**
  (p(dep) 0.96–0.98, P(pred=Yes) ≈ 1.00 in every A2 arm, zero prediction
  flips). Use it for ranking metrics only.
- **`row_media` indexed one level too deep** (`row["videos"][idx][0]`
  subscripts the bytes and yields an int). Fixed; every B/D run depends on it.
- **B2 was never actually blocked.** `load_video(strip=True)` re-muxes to
  video-only in memory — verified `['video','audio','data'] → ['video']`. The
  earlier "not run, no stripped files on disk" note looked for pre-generated
  files that the design never intended to exist.
- **GPUs 6–9 host other users' jobs.** Check `nvidia-smi` before placing runs;
  cuda:0–3 were free.
- **lbfgs on 4096-d standardised features does not converge in reasonable
  time.** PCA to 256 first — it is a linear map, so the probe still measures
  linear decodability, in a better-conditioned basis.

---

## Open questions

1. **Does the A2 effect survive length matching?** The clinical pool is longer
   than the biographical one. A padded-neutral control is the immediate follow-up.
2. **Per-participant counterfactual.** A2 used fixed pools; using each
   transcript's *real* prompts would measure the effect at deployment scale.
3. **Does post-training amplify or suppress this?** — D, running now.
4. **Can the dormant encoding be activated?** The C1 probe weights give a
   corpus-identity direction; injecting it into the residual stream at
   inference would show whether the encoded feature is causally wired to the
   output. Inference-only, so it sits inside the project's no-retraining scope.
5. **Is question-selection bias a shortcut or legitimate evidence?** A human
   clinician would also read the interviewer's choices as informative. The
   operational test must be generalisation under interviewer/protocol shift,
   not ablation — this needs stating explicitly in the paper.

---

## Status

| experiment | state |
|---|---|
| A1 — DAIC ablation | ✅ complete, both text models |
| A2 — DAIC counterfactual | ✅ complete; **overturned the A1 reading** |
| C1 — provenance (template, probes) | ✅ complete, incl. hidden-state probes |
| D — SFT→GRPO→HARPO, B1 ladder | ⏳ running, 3 stages × 300 items |
| B2 — conflict set | ⏸ unblocked (strip verified), not yet built |
| C2/C3 — rewrite counterfactuals | ⏸ needs external API decision |
