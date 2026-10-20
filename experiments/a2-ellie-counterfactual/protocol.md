# A2 — Ellie-prompt counterfactual

**Locked 2026-08-20, before any A2 result was computed.**
(Pre-registration by timestamped file rather than git commit: the project root
holds ~183 GB of checkpoints and licence-restricted corpora, so initialising a
repository here risks staging data that must not be committed. The purpose —
temporal proof that the plan predates the results — is preserved.)

## Why this experiment, now

A1 established that the DAIC interviewer shortcut **exists and is strong**
(TF-IDF on Ellie's turns alone: AUC 0.810, above the participant's own words at
0.756) and that neither model **exploits it under ablation** (`ellie_only` sits
at the label-shuffle floor).

The PI's critique of the original framing is the reason A2 is now the central
experiment rather than a follow-on. The pathway we had been testing —

> side signal → identify the corpus → predict from that corpus's label prior

— is **bounded by the majority-class baseline** and therefore cannot explain
shortcut learning. C1 confirmed the bound empirically: knowing the corpus buys
exactly the prior (swmh: 0.347, which *is* swmh's majority rate), while the
model with content reaches 0.730. The pathway explains none of the gap.

The proposed replacement is:

> side signal → carries other label-correlated information (e.g. *which
> questions the interviewer chose to ask*) → that information drives prediction

DAIC is the ideal test bed because the interview is **adaptive**: the
interviewer's question selection is itself a clinical judgement, so which
prompts were asked is plausibly diagnostic independent of what the participant
said. That is exactly the mechanism the reframing proposes.

**Ablation cannot test this.** Removing Ellie's turns removes the side signal
and any entangled legitimate signal together. Only a counterfactual — content
held byte-identical, side signal swapped — isolates causal use.

## Design

163 distinct scripted prompt IDs across 123 transcripts (F5) supply the prompt
pools directly; no keyword inference or hand-clustering is required, which
removes the design's stated dependency on a hand-verified inventory.

### A2a — data-level (no model)

Which scripted prompts were asked disproportionately to depressed participants?
Per prompt ID: depression rate among participants asked vs not asked, Fisher
exact test, BH-corrected across the ID set.

- **Prediction**: a non-trivial subset of IDs is significantly label-associated.
  This is the *mechanism* the reframed pathway requires. If none are, the
  pathway has no substrate in DAIC and A2b is uninterpretable.

### A2b — model-level counterfactual

For each participant:

| arm | interviewer context S | participant evidence C |
|---|---|---|
| `cf_depression` | 5 depression-associated scripted prompts | their own turns, byte-identical |
| `cf_neutral` | 5 neutral scripted prompts, matched count | **same bytes** |
| `cf_none` | no interviewer turns | **same bytes** |

Pools are fixed across participants so S varies only between arms, never
between subjects. C is truncated once and reused verbatim in all three arms, so
any difference is attributable to S alone.

**Pools** (chosen on semantic content, before seeing A2a results):

- S₁ depression-associated: `depression_diagnosed`, `ptsd_diagnosed`,
  `feel_lately`, `easy_sleep`, `control_temper`
- S₂ neutral: `where_originally`, `study`, `travel_shoes`, `like_about_LA`,
  `dont_like_LA`

### Metric

`CF-Sens = p(depression | C, S₁) − p(depression | C, S₂)`, paired within
participant, bootstrap CI over participants (n ≥ 2000).

Report `cf_none` as the anchor: it shows where the prediction sits with no
interviewer context at all, so the two swapped arms can be read as deviations
in either direction rather than only relative to each other.

## Predictions (committed before running)

1. **If the reframed pathway holds**: CF-Sens > 0 with CI excluding zero —
   identical participant evidence yields a higher depression probability when
   the interviewer asked clinically loaded questions.
2. **If A1's null generalises**: CF-Sens ≈ 0 — the model reads the participant
   and ignores the interviewer even when the interviewer's questions are
   maximally suggestive. This would be the stronger null, because the
   counterfactual is a far more sensitive probe than the ablation A1 used.
3. **Diagnostic either way**: A2a tells us whether the substrate exists in the
   data, independent of what the models do. A large A2a effect with CF-Sens ≈ 0
   reproduces the A1 pattern (available, unused) via a mechanism that survives
   the PI's critique.

## Models

MentaLLaMA-chat-7B and Menta, same scoring path as A1 (teacher-forced label
log-likelihood over `{Yes, No}`, IMHI template, 4032-token budget).

## Known limitations, stated up front

- The three arms differ in prompt length; `cf_none` is shortest. Coverage of C
  is identical by construction, but total token count is not.
- The pools are fixed rather than per-participant matched, so this measures the
  effect of *a* clinically loaded context, not of the specific questions that
  participant was actually asked. A per-participant variant using each
  subject's real prompts is the natural follow-up if CF-Sens is non-zero.
