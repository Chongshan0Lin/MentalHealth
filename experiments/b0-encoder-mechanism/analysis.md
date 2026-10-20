# B0-mech — does a DAIC-trained encoder share the LLM's mechanism?

Run 2026-08-23/24. Step 2 of `INTERVENTION_DESIGN.md` §7 — the measurement that
gates the entire train-time arm.
Code: `code/run_b0_mech.py` · results: `results/b0_mech_summary.json` · log:
`logs/b0_mech_MentalBERT.log`.

MentalBERT, chunked mean-pooling (512-token chunks, no head truncation, no
`pooler_output`), 3 seeds, official 107/34/47 splits, early-stopped on dev AUC.

---

## Headline: the answer is neither branch the design pre-committed to

§3 predicted two outcomes — *same mechanism* (re-thresholding erases the
encoder's counterfactual collapse) or *different mechanism* (it does not).

**Neither happened. The encoder has no counterfactual collapse to explain.**

| | LLM (zero-shot) | Encoder (DAIC-trained) |
|---|---|---|
| decode label from S alone (ablation) | **0.546** — at the floor | **0.676** |
| S moves the prediction (CF-Sens) | **0.1932** | **0.0409** |
| neutral − clinical accuracy gap | **+0.256** | **+0.000** |
| P(pred=Yes), biographical → clinical | 0.160 → **0.745** | 0.220 → **0.234** |

This is the **exact inverse of A1/A2**, and it is the most interesting thing in
the experiment.

- **A1/A2 (LLM):** cannot decode the shortcut, yet is governed by it.
- **B0-mech (encoder):** *can* decode the shortcut, yet is barely moved by it.

A double dissociation. Ablation and counterfactual are not two sensitivities on
one axis — each can be positive while the other is negative, in opposite
directions, on the same corpus and the same shortcut.

---

## 1. Measurement 1 — the encoder learns at n=107

| seed | dev AUC | test AUC |
|---|---|---|
| 0 | 0.755 | 0.762 |
| 1 | 0.822 | 0.764 |
| 2 | 0.692 | 0.639 |
| **mean** | | **0.722 (sd 0.059)** |

Above the LLM's zero-shot `full` (0.663), below D1's 0.787. The pre-committed
abandonment criterion (CI overlapping the shuffle floor) is **not** triggered —
the encoder arm is viable. Seed spread is large, as expected at n=107.

## 2. Measurement 2 — the encoder *does* absorb the shortcut

`ellie_only`, interviewer turns only, no participant words:

| seed | test AUC |
|---|---|
| 0 | 0.723 |
| 1 | 0.634 |
| 2 | 0.671 |
| **mean** | **0.676 (sd 0.036)** |

**0.676 from the interviewer's questions alone**, against the zero-shot LLM's
0.546 (at its floor) and TF-IDF's 0.810 pooled.

This confirms the premise the design rested on: *a model trained on DAIC
absorbs the interviewer cue into its weights.* The encoder reaches 0.676 on the
side channel while reaching only 0.722 on the full input — the shortcut carries
almost as much as everything else combined.

## 3. Measurement 3 — but it does not act on it

`full`-trained encoder over the four A2 counterfactual arms:

| arm | AUC | acc@dev-thr | acc@opt | P(pred=Yes) |
|---|---|---|---|---|
| cf_neutral | 0.636 | 0.652 | 0.730 | 0.220 |
| cf_none | 0.656 | 0.681 | 0.730 | 0.319 |
| cf_empirical | 0.648 | 0.667 | 0.752 | 0.220 |
| cf_depression | 0.632 | 0.652 | 0.730 | 0.234 |

Everything is flat. Accuracy at the dev threshold is identical for the
biographical and clinical arms (0.652 both, gap **+0.000**); `acc@opt` is 0.730
in three of four arms; the operating point moves 0.220 → 0.234 where the LLM
moved 0.160 → 0.745.

`CF-Sens = 0.0409` against the LLM's `0.1932` — **4.7× smaller**.

The "fraction of the collapse that is threshold shift" prints `nan` because the
denominator is exactly zero. That is the correct signal, not a bug: there is no
collapse.

---

## 4. The confound that has to be resolved before any of this is claimed

**Chunk mean-pooling may be washing out the perturbation.**

The LLM reads the whole transcript in one attention context, where a
clinically-loaded phrase can be attended to strongly regardless of its token
count. The encoder splits the transcript into ~6 chunks of 512, mean-pools
tokens within each chunk, then mean-pools the chunks. Under mean-pooling a
span's influence is proportional to its **token count**, so a 38-word insertion
in a ~2,856-token transcript is weighted at roughly its length share and cannot
be amplified.

The insertion ratio itself is comparable across the two models (~1 % of input in
both, since `interleave` spreads the five prompts evenly and the LLM's budget is
4,032 tokens), so this is **not** a naive dilution argument. The claim is
narrower and harder to dismiss: *mean-pooling is a weaker aggregator than
attention, and cannot upweight a semantically salient span.*

Until this is tested, the low CF-Sens is confounded between two explanations:

1. **Trained encoders genuinely do not exhibit the counterfactual effect** —
   the interesting result.
2. **Mean-pooling cannot express it** — an architecture artifact, and the
   LLM/encoder comparison would then be measuring the readout, not the
   mechanism.

### The control was run. Verdict: **the readout matters, the conclusion holds.**

Retrained under identical protocol with only the readout swapped
(`logs/b0_mech_{attn,cls}.log`, `results/b0_mech_summary_{attn,cls}.json`):

| readout | `full` AUC | `ellie_only` AUC | **CF-Sens** | neutral − clinical acc gap |
|---|---|---|---|---|
| `mean` (original) | 0.722 | 0.676 | **0.0409** | +0.000 |
| `cls` | 0.704 | 0.628 | **0.1008** | +0.021 |
| `attn` | 0.721 | **0.712** | **0.0638** | −0.007 |
| **LLM (I1)** | 0.663 | 0.546 | **0.1932** | **+0.256** |

**The confound was real but not fatal.** CF-Sens spans 0.041–0.101 across
readouts — a 2.5× range — so the original "4.7× smaller than the LLM" was
readout-dependent and overstated. The honest claim is **2–5× smaller**, and the
quantitative comparison must always name the readout.

**The qualitative conclusion survives intact, on the metric that matters.** The
defining feature of the LLM effect is the *operating-point collapse*: the
accuracy gap of **+0.256** and `P(pred=Yes)` running 0.160 → 0.745. Every
encoder readout gives a gap of **−0.007 to +0.021** — indistinguishable from
zero, with `P(pred=Yes)` moving at most 0.156 → 0.262. No readout produces
anything resembling the threshold displacement.

Note also that `attn` **raises** `ellie_only` to 0.712 (sd 0.008, the most
stable measurement in the set) against the LLM's floor-level 0.546. Under the
readout best able to express the effect, the encoder decodes the shortcut
*better* and is moved by it *less*. That strengthens the dissociation rather
than weakening it.

The printed "fraction of the collapse that is threshold shift" reads −200 % and
133 % for `attn` and `cls`. Both are meaningless: the denominator is a gap of
≈0. Reported here only so the numbers in the logs are not mistaken for signal.

---

## 5. What it does to the design if it survives

The design's §5.3 fork assumed the encoder would show a *learned representation
bias* that threshold correction fails to repair, motivating D3/M1/M2. What was
actually measured is stranger and better:

The encoder **encodes** the shortcut (0.676) without **using** it (CF-Sens
0.041). That is C1's "encoded ≠ used" finding reproduced at the behavioural
level in a different model class — and it means the matched intervention for the
encoder is *not* PoE or group DRO either, because there is no counterfactual
reliance for them to remove.

If it holds, the paper's claim sharpens to:

> Shortcut *availability*, shortcut *encoding*, and shortcut *use* are three
> different things, and every existing audit method measures only one of them.
> The LLM uses without encoding; the encoder encodes without using.

D3/M1/M2 would then be **unnecessary rather than merely unmotivated**, and the
workshop paper is the LLM intervention result (I1) plus this dissociation.

---

## 6. Solid / not solid

**Solid:**
- Encoder learns at n=107: 0.722, above the abandonment threshold.
- Encoder decodes the shortcut from interviewer turns alone: 0.676 vs the LLM's
  0.546.
- Encoder's counterfactual response is flat: gap +0.000, P(Yes) 0.220 → 0.234.

**Not solid:**
- **CF-Sens 0.0409 is confounded by the mean-pooling readout** (§4). This is the
  blocking issue.
- Seed spread is large (M1 sd 0.059, range 0.639–0.764) at n=107. Differences
  between the encoder and LLM AUCs are not separable; only the CF-Sens and
  operating-point contrasts are large enough to survive it.
- Three seeds, one encoder, one checkpoint. MentalRoBERTa replication not run.
- The encoder saw untruncated transcripts; the LLM run truncated C to a
  4,032-token budget. Not byte-identical inputs, recorded in the code.
