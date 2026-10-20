# A2 — analysis

Run 2026-08-20. Protocol locked in `protocol.md` before any result was computed.

## Headline

**The interviewer shortcut is used after all — the ablation in A1 was simply
blind to it.** With the participant's own words held byte-identical, swapping
five scripted interviewer questions from biographical to clinically loaded
moves MentaLLaMA's depression probability by **+0.206 [+0.195, +0.217]** and
flips **58.5 %** of predictions. Accuracy falls from 0.750 to 0.495 — below the
0.707 majority baseline.

A1 and A2 are not in conflict; they measure different things, and the pair is
the finding:

| | question asked | MentaLLaMA answer |
|---|---|---|
| A1 ablation (`ellie_only`) | can the label be *decoded* from interviewer turns alone? | **No** — AUC 0.546, at the label-shuffle floor |
| A2 counterfactual | does interviewer context *move the prediction* when participant evidence is fixed? | **Yes, massively** — Δp = +0.206, 58.5 % flips |

The model cannot read the label out of the interviewer's questions, yet its
prediction is strongly driven by them. Ablation was structurally incapable of
detecting this: removing Ellie's turns removes the side signal and any
entangled legitimate signal at once, and the resulting accuracy drop (or lack
of one) confounds the two.

## A2a — the mechanism exists in the corpus

Which scripted prompts Ellie asked is strongly label-associated. Four IDs
survive BH correction across 75 tested:

| prompt id | question | OR | dep rate asked vs not |
|---|---|---|---|
| `why_seek_help` | what got you to seek help | **11.26** | 0.56 vs 0.10 |
| `therapy_going` | do you still go to therapy now | **10.42** | 0.54 vs 0.10 |
| `trigger` | are they triggered by something | 8.50 | 0.58 vs 0.14 |
| `when_diagnosed` | how long ago were you diagnosed | 8.24 | 0.53 vs 0.12 |

These are self-evidently *adaptive follow-ups* — "do you still go to therapy"
is only asked once therapy has been disclosed. The reverse direction is the
same mechanism seen from the other side: `family_relationship`, `BF_describe`,
`feelbadly` (OR 0.25–0.28) are the default script, reached when there is no
clinical thread to pursue.

**Question selection alone predicts the label at AUC 0.753** (sd 0.025 over 10
CV seeds, shuffled 0.500) from a binary asked/not-asked matrix over 75 prompt
IDs — no text, no counts, no phrasing. This is the substrate the reframed
pathway requires, quantified.

## A2b — the models

### MentaLLaMA-chat-7B (n=188, 35 truncated)

| arm | mean p(dep) | acc | AUC | P(pred=Yes) |
|---|---|---|---|---|
| cf_neutral | 0.373 | **0.750** | 0.764 | 0.160 |
| cf_none | 0.446 | 0.707 | 0.777 | 0.351 |
| cf_empirical | 0.497 | 0.654 | 0.729 | 0.447 |
| cf_depression | 0.579 | **0.495** | 0.734 | 0.745 |

Paired contrasts (all CIs exclude zero):

| contrast | Δp(dep) | pred flips |
|---|---|---|
| clinical − biographical *(confirmatory)* | **+0.2057 [+0.1945, +0.2168]** | 0.585 |
| A2a-derived − biographical *(exploratory)* | +0.1236 [+0.1152, +0.1321] | 0.287 |
| clinical − no interviewer | +0.1329 [+0.1214, +0.1440] | 0.394 |
| biographical − no interviewer | −0.0728 [−0.0793, −0.0663] | 0.191 |

### It is a bias shift, not a change in discrimination

This is the precise mechanistic claim, and it is what makes the result a
shortcut rather than ordinary context sensitivity:

- **AUC barely moves** across arms: 0.764 → 0.734. The model's *ranking* of
  participants is nearly unchanged.
- **The operating point moves enormously**: P(pred=Yes) 0.160 → 0.745.
- **Both classes move together, and the non-depressed move further**:
  Δ = +0.183 for depressed, **+0.215 for not depressed**.

Interviewer context therefore acts as an additive bias on the decision
threshold, not as evidence that sharpens or degrades the ranking. Accuracy
collapses because the threshold is dragged across a distribution whose order
was left intact.

### The model responds to the wrong part of the side channel

The semantically loaded pool moves the model **more** (+0.206) than the pool
A2a identified as actually label-predictive (+0.124). The questions carrying
genuine diagnostic information ("do you still go to therapy now") shift it less
than questions that merely *sound* clinical ("have you been diagnosed with
depression").

So the model does use the side channel — but keys on **surface clinical
vocabulary**, not on the adaptive-interview structure that carries the label.
It picks up the part that is spurious and largely misses the part that is
informative. This dissociation is the sharpest single result in the project so
far.

### Menta — saturated, uninformative

Menta sits at p(dep) 0.96–0.98 with P(pred=Yes) ≈ 1.00 in every arm. Contrasts
reach significance (clinical − biographical = −0.0147 [−0.0243, −0.0056]) but
in a compressed regime, with **zero prediction flips** and a sign opposite to
MentaLLaMA's. Its operating point makes it a poor counterfactual probe; report
it as a null with the saturation stated, not as a contradicting result.

## What this changes

1. **The PI's reframing is vindicated and the old pathway is dead.** "Identify
   the corpus, apply its prior" is capped at the majority baseline (C1 measured
   the cap). The replacement — side signal carries other label-correlated
   information — is confirmed at the data level (A2a, AUC 0.753) and shown to
   drive the model at the behavioural level (A2b).
2. **"Models do not exhibit shortcut learning" must be withdrawn.** It was an
   artifact of testing only the ablation-detectable form.
3. **Ablation-only shortcut audits are unsafe.** A1's null and A2's large
   effect come from the same model on the same corpus. Any benchmark audited by
   ablation alone can carry an effect this size undetected.
4. **The deployment risk is concrete.** A clinician who asks more probing
   questions makes the model more likely to output "depressed" — regardless of
   what the patient said. The bias-shift characterisation says exactly how:
   the threshold moves, the ranking does not.

## Limitations

- **Length: checked, and it clears the main result but not the dissociation.**
  Pool sizes are `cf_neutral` 63 tokens, `cf_depression` 52, `cf_empirical` 34.
  The clinical pool is *shorter* than the biographical one yet raises
  p(depression) by +0.206, so the headline effect cannot be an artifact of
  more interviewer text — the confound runs opposite to the effect.
  **The "wrong half" dissociation is a different matter**: `cf_empirical` is a
  third shorter than `cf_depression`, so its weaker effect (+0.124 vs +0.206)
  is partly confoundable with length. That comparison needs a token-matched
  empirical pool before it can carry weight; as it stands it is suggestive, not
  established.
- Pools are fixed across participants, so this measures the effect of *a*
  clinically loaded context, not of the questions each subject was actually
  asked. A per-participant variant using each transcript's real prompts is the
  natural next step now that the effect is known to be non-zero.
- 35/188 participant transcripts were truncated to fit the budget; truncation
  is identical across arms so it cannot produce the contrast, but it does mean
  C is partial for those subjects.
- Two models only, one of which (Menta) is saturated and effectively
  uninformative here.
