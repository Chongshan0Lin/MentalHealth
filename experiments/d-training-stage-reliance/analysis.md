# D / B1 — analysis

Run 2026-08-20. Three OmniSapiens training stages, identical architecture, base
model, prompt, video preset (336p), sampling seed and evaluation. Training stage
is the only variable. n = 300 `meld_emotion` items, 5 conditions, 0 skipped;
all 300 common to all three stages.

Gold distribution: neutral 145, joy 47, anger 37, surprise 32, sadness 27,
disgust 8, fear 4. **Majority 0.483, chance 0.143.**

## Headline: benchmark accuracy 2.6× while discrimination barely moves

| `full` condition | SFT | GRPO | HARPO | Δ(HARPO−SFT) |
|---|---|---|---|---|
| **raw accuracy** | 0.260 | 0.600 | **0.687** | **+0.427** |
| **balanced accuracy** | 0.388 | 0.379 | 0.422 | +0.033 |

Post-training raises the reported number by 43 points — from *below* the 0.483
majority baseline to well above it — while balanced accuracy moves by 3 points,
comfortably inside its confidence interval (SFT [0.301, 0.493] vs HARPO
[0.362, 0.497]).

**The gain is not discrimination. It is the majority class.**

## The direct evidence: what each stage says with no input at all

`question_only` withholds the video, the audio *and* the transcript. Nothing is
left but the question and the option list.

| stage | `question_only` raw acc | predicts |
|---|---|---|
| SFT | 0.027 | a rare class |
| GRPO | 0.027 | a rare class |
| **HARPO** | **0.483** | **`neutral` — exactly the corpus majority rate** |

HARPO's content-free accuracy equals the majority rate to three decimals
because it answers `neutral` to every item. SFT and GRPO do not.

**The "recognise the setting, answer its prior" behaviour is acquired during
post-training.** It is not a property of the base model or of the SFT
checkpoint; it appears by the HARPO stage.

Note the two readings of `Acc_∅` and keep them apart:

- **Balanced accuracy = 0.143 = chance, identically for all three stages.** A
  constant predictor over 7 classes scores exactly 1/7. There is no
  *discriminative* leakage — the question does not let any stage tell items
  apart.
- **Raw accuracy tells the other half.** There *is* prior leakage, and only
  HARPO exploits it.

Reporting only the balanced figure would have shown a flat, clean 0.143 across
all three stages and hidden the entire finding.

## Media reliance goes negative, and deepens with training

`Δ_media = bal_acc(full) − bal_acc(text_only)`:

| stage | Δ_video | Δ_audio | **Δ_media** |
|---|---|---|---|
| SFT | +0.038 | −0.052 | −0.005 |
| GRPO | −0.016 | +0.025 | −0.060 |
| HARPO | −0.040 | −0.032 | **−0.071** |

Negative means the model is *better without the media*. At HARPO, `text_only`
reaches 0.493 balanced accuracy against 0.422 with the full input — supplying
audio and video costs 7 points, and the cost grows monotonically across
training stages (−0.005 → −0.060 → −0.071).

So post-training does not improve multimodal grounding on this benchmark. The
media channels become progressively more harmful while the reported number
climbs.

## Reading against the design's pre-commitment

The design asked for `ΔSR = SR_post-RL − SR_SFT` on the **B2 conflict set**,
which has not been run. What B1 supplies is a different and weaker proxy, and
the distinction should be kept in the write-up: B1 measures whether the media
channels contribute, not which channel wins when they disagree.

On that proxy the answer is unambiguous, and it is the case the design flagged
as the one that matters:

> `ΔSR > 0`: RL raises benchmark accuracy **while amplifying** shortcut
> reliance… Report `ΔSR` jointly with benchmark accuracy per stage. The
> headline result is the *decoupling*.

Accuracy climbs 2.6×; the climb is traceable to the majority-class prior; media
contribution goes negative. The decoupling is present in exactly the predicted
direction.

This is also evidence against the HARPO line's own claim. HumanOmniV2 and HARPO
advertise anti-shortcut context grounding; on `meld_emotion` the HARPO-stage
checkpoint is the *only* one of the three that defaults to the corpus prior when
starved of input, and it is the one whose media channels hurt most.

## What is solid and what is not

**Solid:**
- The raw-accuracy climb (+0.427) is large and cannot be noise.
- The `question_only` contrast (0.027 / 0.027 / 0.483) is categorical — HARPO
  emits a constant majority-class answer, the others do not.
- Δ_media is negative at every stage and monotone across stages.

**Not solid:**
- The balanced-accuracy differences between stages are **within confidence
  intervals** and should not be reported as a capability gain or loss. The
  claim is precisely that balanced accuracy did *not* move while raw accuracy
  did.
- Δ_video and Δ_audio individually flip sign across stages with overlapping
  CIs; only the pooled Δ_media is stable.

**Scope:**
- One source (`meld_emotion`), n = 300, one video preset. The design's ladder
  also called for Daily-Omni and IntentBench; neither was run here.
- B2 (conflict) is unrun, so `SR` proper — which channel wins under
  disagreement — remains unmeasured. That is the natural next experiment and it
  is now unblocked (the audio-strip path is verified).
- 336p preset locked from the F4 calibration and applied identically
  everywhere, so it cannot produce a between-stage difference.
