# H6 — Irf register counterfactual: the finding that broke the mechanism taxonomy

Run 2026-09-02. Code: `code/run_h6.py`, `code/analyze_h6.py` · results:
`results/h6_irf_{MentaLLaMA-chat-7B,Menta}.jsonl` (1,600 rows each) · log:
`results/h6_analysis.log`.

400 posts sampled from the natural distribution (seed 0), 4 arms, 2 models.
9 posts held out to build the context pools.

## Why this dataset

`Irf` carries the purest register shortcut in either audit. H2, model-free:

| probe | balanced acc |
|---|---|
| text (TF-IDF) | 0.643 (chance 0.500) |
| **stopwords only** | **0.622 — 97% of the full-text signal** |

Deleting every content word costs 0.021. That ratio is the highest anywhere in
the project (`mosei_emotion` is next at 92%).

Two reasons it was the right test to run after H5:

- **The models were trained on this data.** MentaLLaMA and Menta are
  instruction-tuned on IMHI. Every earlier counterfactual used models evaluated
  on corpora they had never trained on.
- **The classes are same-source.** In `ptsd_in_the_wild` the two classes came
  from different production sources entirely (veteran testimony vs TED talks),
  so register and topic were confounded. Both `Irf` classes are Reddit
  mental-health posts, so the contrast is closer to register alone.

## The arm H5 did not have

Same prepend design as H5 — each post byte-identical across arms, pools
token-matched at 150 words — plus a **signal-free control**:

| arm | prefix |
|---|---|
| `native` | none |
| `ctx_neutral` | a block from **MultiWD** |
| `ctx_no` | a block from `Irf` `no` posts |
| `ctx_yes` | a block from `Irf` `yes` posts |

MultiWD is the one dataset in either audit whose text carries **no** label signal
(H2: 0.499 against chance 0.500). A MultiWD prefix therefore adds length and
mental-health topic without adding class evidence. It turned out to be the arm
that made the result interpretable.

## Result: the effect is real, and points the wrong way

| model | arm | mean p(yes) | acc | AUC | P(pred=yes) |
|---|---|---|---|---|---|
| MentaLLaMA | `ctx_no` | 0.7405 | 0.752 | 0.714 | 0.840 |
| MentaLLaMA | `native` | 0.6504 | 0.757 | **0.849** | 0.740 |
| MentaLLaMA | `ctx_neutral` | 0.6424 | 0.767 | 0.843 | 0.780 |
| MentaLLaMA | `ctx_yes` | 0.6583 | 0.765 | 0.843 | 0.818 |
| Menta | `ctx_no` | 0.9948 | 0.670 | 0.573 | 0.998 |
| Menta | `native` | 0.9446 | 0.685 | 0.673 | 0.948 |
| Menta | `ctx_neutral` | 0.8767 | 0.700 | 0.712 | 0.892 |
| Menta | `ctx_yes` | 0.9709 | 0.675 | 0.694 | 0.983 |

CSS = p(yes | yes-register) − p(yes | no-register):

| model | CSS | 95% CI | flips |
|---|---|---|---|
| MentaLLaMA | **−0.0822** | [−0.0953, −0.0696] | 0.077 |
| Menta | **−0.0239** | [−0.0327, −0.0164] | 0.015 |

**Both CIs exclude zero and both signs are negative.** Prepending the `no`-class
register raises p(yes) *more* than prepending the `yes`-class register. If the
model were reading register as evidence, the sign would be positive.

## What the control arm shows is actually happening

| contrast (MentaLLaMA) | Δ p(yes) | 95% CI |
|---|---|---|
| `ctx_no` − signal-free | **+0.0981** | [+0.0842, +0.1124] |
| `ctx_yes` − signal-free | +0.0159 | [+0.0109, +0.0211] |
| signal-free − `native` | −0.0081 | [−0.0175, +0.0019] **n.s.** |

**Both `Irf` prefixes push p(yes) up; the `no` prefix pushes harder. The MultiWD
prefix does essentially nothing.**

So the model is not reading *which class the prefix belongs to*. It is reading
*whether a block of clinical-sounding first-person distress text is present at
all*, and answering "yes" more often when it is. The `no`-class posts happen to
read more like clinical description — the canonical example is a detailed
first-person account of starting Lexapro and its side effects — so they trigger
that response harder than the `yes`-class posts do.

This is a coarse **topic-presence effect**, not use of the register shortcut. The
signal-free arm is what separates the two; without it, the negative CSS would
have looked like a mysterious inversion rather than a misattribution.

## The sharper finding: the two questions behave oppositely

`Irf` bundles two constructs under one dataset name. Splitting MentaLLaMA's CSS
by question:

| question | n | CSS | 95% CI |
|---|---|---|---|
| thwarted belongingness | 202 | **−0.1801** | [−0.1962, −0.1647] |
| perceived burdensomeness | 198 | **+0.0177** | [+0.0127, +0.0230] |

**Same model, same prefixes, same sampled posts — the two questions give
opposite signs, 0.20 apart.** Menta splits the same way (−0.0015 vs −0.0468).

The label balance also differs sharply between them (thwarted belongingness
784 yes / 273 no; perceived burdensomeness 582 yes / 474 no), so they are not
interchangeable in difficulty either.

**`Irf` should not be reported as a single task.** Any aggregate number over it
averages two constructs on which the same model behaves differently.

## This refutes the taxonomy H4 suggested

After DAIC, `mmpsy` and `ptsd_in_the_wild`, the pattern looked like *lexical and
register shortcuts get used; structural ones do not*. H6 breaks that:

| dataset | mechanism | data-level | CSS | used? |
|---|---|---|---|---|
| DAIC | lexical | 0.810 | +0.206 | yes |
| `ptsd_in_the_wild` | register, **cross-source** | 0.948 (stopwords 0.880) | +0.282 | yes |
| `mmpsy_depression` | structural | 0.772 | +0.004 | no |
| **`Irf`** | **register, same-source** | 0.643 (stopwords 0.622) | **−0.082** | **no** |

Two register shortcuts, opposite outcomes. The distinguishing property is not
the mechanism label but **how coarse the difference is**:

- `ptsd_in_the_wild`'s classes come from *different production sources* — veteran
  testimony against TED talks. The register gap is enormous and survives at the
  level of topic, vocabulary and framing all at once.
- `Irf`'s classes are *the same kind of text* from the same platform. The
  register gap is a fine-grained distributional difference in function-word usage
  — visible to a TF-IDF probe at 0.622, invisible to the language model.

**Revised statement:** a shortcut is exploitable to the extent that its
discriminative signal is coarse enough to cross the model's representational
grain. Stopword-level distributional differences and segment counts are both
below that grain, which is why `Irf` and `mmpsy` fail together despite having
nothing else in common. The mechanism label (lexical / register / structural)
does not predict use; the granularity does.

This makes the discriminative-probe result an **upper bound that can overstate
model risk by a wide margin** — 0.622 balanced accuracy for a probe against a
wrong-signed 0.08 for the model it is supposed to be warning about.

## Limitations

- **Class imbalance**: 66.7% yes. Accuracy is close to the majority rate in
  several arms and should not be read on its own.
- **Menta is heavily saturated here** (p(yes) 0.88–0.99, AUC as low as 0.573
  under `ctx_no`). Its numbers corroborate the sign but should not be read
  quantitatively.
- **Prepending is not swapping.** As in H5, the manipulation adds a register
  context rather than exchanging one post's register for another's.
- 400 of 2,104 eligible posts, one seed. The question-split contrast is the
  result most worth re-running at full scale, since it is the one that changes
  how the dataset should be reported.
- MentaLLaMA's `native` AUC (0.849) is its highest anywhere in the project, and
  every prefix arm *lowers* it (0.843 / 0.843 / 0.714). Prefixes hurt ranking
  even when they do not encode the label.
