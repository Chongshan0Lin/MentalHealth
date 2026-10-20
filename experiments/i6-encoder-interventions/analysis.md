# I6 — D3 / M1 / M2: the train-time arm returns nothing, and one method breaks

Run 2026-08-24 at the PI's explicit direction. Code:
`code/run_encoder_interventions.py` · results:
`results/encoder_interventions.json` · log: `logs/i6_encoder_interventions.log`.
MentalBERT, 3 seeds, identical training budget and readout to B0-mech.

---

## Results

| method | test AUC | sd | CF-Sens | neutral−clinical gap |
|---|---|---|---|---|
| **B0 ERM** (baseline) | **0.722** | 0.059 | 0.0409 | +0.000 |
| **D3** counterfactual augmentation | 0.734 | **0.008** | 0.0337 | −0.014 |
| **M1** product of experts | **0.540** | 0.007 | 0.0039 | +0.000 |
| **M2** group DRO | 0.659 | 0.084 | 0.0443 | −0.007 |
| **M2reg** group DRO + heavy L2 | 0.646 | 0.096 | 0.0734 | −0.043 |

**Not one method improves on ERM. Two make it substantially worse. One destroys
the model.**

---

## Method by method

**D3 — counterfactual augmentation: no effect on the target, one real benefit.**
AUC 0.734 vs 0.722, inside the seed spread. CF-Sens 0.0337 vs 0.0409 — a
reduction of 0.007 from a base that was already near zero, which is not a
meaningful debiasing. The one genuine result is **variance**: seed sd collapses
from 0.059 to **0.008**, by far the most stable model in the arm. Quadrupling
the training set with label-preserving counterfactuals stabilised training. That
is a useful practical finding and has nothing to do with shortcuts.

**M1 — PoE: catastrophic, and diagnostically so.** AUC **0.540**, barely above
chance, from a 0.722 baseline. This is not noise (sd 0.007 across seeds — it
fails *consistently*).

The mechanism is clear. PoE downweights gradients wherever the bias model
already predicts correctly. Our bias model is the A2a question-selection probe
at **CV AUC 0.773** — nearly as accurate as the full model. So PoE suppressed
the learning signal on most of a 107-example training set and left almost
nothing to learn from. **PoE assumes the bias model is weak relative to the
main model. On DAIC it is not.** That is a genuine, publishable observation
about when PoE is applicable, and it required running it to find.

**M2 / M2reg — group DRO: worse than ERM, and unstable.** 0.659 and 0.646
against 0.722, with seed sd of 0.084 and 0.096 — the noisiest results in the
project. Heavy L2 did not rescue it.

The cause was visible before the run and is now confirmed. Group sizes in the
107-example training set:

```
group sizes (train): {0: 70, 2: 20, 3: 10, 1: 7}
```

**Seven examples in the smallest group.** Minimising worst-group loss over a
cell of 7 optimises against sampling noise. This is an **n-limit, not a verdict
on group DRO** — Sagawa et al. operate on datasets three orders of magnitude
larger — and it must be reported that way.

---

## Reading the arm as a whole

The prediction recorded in the script's docstring before running was that these
methods would return nulls, because B0-mech measured the encoder as *encoding*
the shortcut (ellie_only AUC 0.620–0.712) without being *moved* by it (CF-Sens
0.0096–0.101, gap ≈ 0 across three readouts and two checkpoints), and encoder-D1
changed AUC by 0.002.

That is what happened, and the mechanism is the explanation. **D3, M1 and M2 all
remove a counterfactual reliance on `S`. This arm does not have one.** They
therefore cannot help, and they spend capacity — M1 catastrophically — to remove
something that is not there.

**What this is NOT evidence for.** Nothing here says PoE, group DRO or
counterfactual augmentation are ineffective in general. They are well-supported
methods on problems with the failure mode they target. What the arm shows is
narrower and more useful:

> Matching the intervention to the *measured mechanism* matters more than the
> choice of method. Applied to a model that encodes a shortcut without acting on
> it, three standard debiasing methods produce no benefit and one produces
> catastrophic harm — while on the *same corpus and shortcut*, a one-line input
> edit (D1) on a model that *does* act on it buys +0.126 AUC and doubles
> construct sensitivity.

That contrast is the paper's argument, and the train-time arm is now the
evidence for its negative half rather than a gap in the matrix.

---

## Solid / not solid

**Solid:**
- M1's collapse to 0.540 with sd 0.007 — consistent across all three seeds.
- D3's variance reduction, 0.059 → 0.008.
- No method reduces CF-Sens materially; the base was already ~0.04.

**Not solid:**
- **M2 is underpowered, not refuted.** 7 examples in the smallest group. Its
  0.659 says nothing about group DRO as a method.
- Three seeds, one encoder, one readout (`mean`). The readout control showed
  CF-Sens varies 2.5× across readouts, so these CF-Sens values are
  readout-specific.
- D3's augmentation composes participant text with fixed pools, so augmented
  examples have a different `S` distribution from the originals (which carry
  real interviewer turns). Standard CDA practice, but not a pure `S` swap.
- M1 used α = 1.0 without a sweep. A smaller α would suppress less gradient and
  might not collapse — the failure is demonstrated at this setting only.
