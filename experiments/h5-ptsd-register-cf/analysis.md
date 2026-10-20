# H5 — ptsd_in_the_wild register counterfactual: the largest effect in the project

Run 2026-09-02. Code: `code/run_h5.py`, `code/analyze_h5.py` · results:
`results/h5_ptsd_{SFT,RL,HARPO}.jsonl` (180 rows each) · log:
`results/h5_analysis.log`.

60 items evaluated (4 held out to build the register pools), 3 arms, the three
OmniSapiens training stages. Text-only.

## Why this experiment

Third shortcut mechanism, after DAIC (lexical — which questions were asked) and
`mmpsy` (structural — how many answers were recorded). Here the shortcut is
**register**: which production source the recording came from.

H1's model-free audit:

| probe | balanced acc |
|---|---|
| full transcript | **0.948** (chance 0.500) |
| after scrubbing *all* PTSD/military vocabulary | **0.953** — it rises |
| **stopwords only, every content word deleted** | **0.880** |
| first 100 words | 0.844 |
| first 25 words | 0.719 |

The positives are veteran testimony and clinical scenes; the negatives are
TED-style talks and lifestyle content. The classes differ on everything, not on
a removable word list, which is why scrubbing does not help.

## The design constraint, stated rather than worked around

**Construct and register are perfectly confounded in this corpus.** There is no
PTSD-positive vlog and no PTSD-negative testimony, so register cannot be swapped
*within* an item — the item is its register. That is a property of the dataset,
not a limitation of the method, and it is itself part of the finding.

What is available is the A2 move: hold the item's own transcript byte-identical
and **prepend** a fixed block carrying the other class's register.

| arm | content |
|---|---|
| `native` | the transcript as collected |
| `ctx_vlog` | + a fixed talk/vlog-register passage |
| `ctx_testimony` | + a fixed testimony-register passage |

Three controls are built in:

- **The two pools are token-matched at 120 words**, so the primary contrast is
  length-controlled by construction and cannot be read as "more text".
- **Pool source items are excluded from evaluation** (4 of 64), so no item is
  ever scored against a context drawn from itself.
- **Each item's own transcript is truncated once and reused verbatim** in all
  three arms, so the arms cannot differ in how much of the item the model saw.

H1 showed the register signal is front-loaded (first 25 words alone give 0.719),
so a prepended block sits exactly where that signal lives.

Text-only: the video channel is confounded by register too — talk footage versus
clinical footage — so including it would muddy the thing being isolated.
Placeholders are stripped identically in every arm, so that choice cannot
produce the contrast.

## Result: used, and more strongly than anywhere else

| stage | arm | mean p(ptsd) | accuracy | AUC |
|---|---|---|---|---|
| SFT | `ctx_vlog` | 0.5796 | **1.000** | 1.000 |
| SFT | `native` | 0.5648 | 0.967 | 0.997 |
| SFT | `ctx_testimony` | 0.7716 | **0.567** | 0.964 |
| GRPO | `ctx_vlog` | 0.6817 | 0.667 | 1.000 |
| GRPO | `native` | 0.7102 | 0.700 | 1.000 |
| GRPO | `ctx_testimony` | 0.7778 | **0.500** | 0.993 |
| HARPO | `ctx_vlog` | 0.5426 | **1.000** | 1.000 |
| HARPO | `native` | 0.5527 | **1.000** | 1.000 |
| HARPO | `ctx_testimony` | 0.8250 | **0.517** | 1.000 |

CSS = p(ptsd | testimony register) − p(ptsd | vlog register), paired:

| stage | CSS | 95% CI | flips | on true positives | on true negatives |
|---|---|---|---|---|---|
| SFT | **+0.192** | [+0.143, +0.243] | 0.433 | +0.011 | **+0.373** |
| GRPO | +0.096 | [+0.077, +0.114] | 0.167 | +0.033 | +0.159 |
| HARPO | **+0.282** | [+0.217, +0.349] | 0.483 | +0.034 | **+0.531** |

Direction against `native` confirms the sign: `ctx_testimony` raises
p(ptsd) (+0.207 / +0.068 / +0.272) while `ctx_vlog` barely moves it
(+0.015 / −0.029 / −0.010).

### 120 prepended words destroy a perfect classifier

HARPO scores **1.000 accuracy natively** and **0.517 — coin-flipping — under a
testimony-register prefix**, without one word of the item's own transcript
changing. SFT falls 0.967 → 0.567.

### It is a false-positive generator

The effect is almost entirely on **true negatives**: HARPO +0.531 against +0.034
on positives. Positives are already classified as PTSD and cannot move much;
negatives are pushed across the threshold. Give a TED talk a clinical-register
opening and the model diagnoses PTSD.

### Same signature as DAIC: the threshold moves, the ranking does not

AUC is essentially unchanged across arms (HARPO 1.000 → 1.000; SFT 1.000 →
0.964). The register context acts as an additive bias on the decision threshold,
not as evidence that changes the ordering — exactly the mechanism A2 identified
on DAIC, reproduced on a different corpus with a different shortcut type.

## Across training stages: non-monotone, and it must be reported that way

SFT 0.192 → GRPO **0.096** → HARPO **0.282**.

GRPO halves register reliance; HARPO returns it above the starting point. This
is **not** the monotone amplification the i10 sweep found for DAIC CSS across the
same three checkpoints (0.457 → 0.471 → 0.505), and the two should not be pooled
into a single "post-training amplifies shortcuts" claim. The honest statement is
that the effect of post-training on shortcut reliance is **shortcut-specific**:
monotone amplification for the interviewer channel, suppress-then-rebound for
register.

## Limitations

- **n = 60 (30/30) is the smallest sample in the project.** The CIs are tight
  but the item pool is small and drawn from 64 total.
- **Prepending is not swapping.** Because construct and register are confounded
  in this corpus, this measures the effect of *adding* a register context, not of
  exchanging one item's register for another's. A corpus containing
  PTSD-negative testimony would allow the stronger design; this one cannot.
- 6 of 60 transcripts were truncated to the 2,400-word budget. Truncation is
  identical across arms, so it cannot produce the contrast.
- The pools are fixed across items, so this is the effect of *a* register
  context, not of the specific register each item would naturally carry.
- Text-only, so this does not measure what the deployed multimodal
  configuration would do — though the video channel carries the same confound.

## What this subset should be used for

H1's verdict was **"not a clinical task; do not report"** — the negatives are not
patients, no clinical evaluation took place, and the prompt asserts a premise
false for half the data. H5 does not soften that. Any accuracy figure published
on this subset measures corpus identification.

Its value is as a **stress test**: it is the cleanest available demonstration
that a register prefix alone can convert a perfect classifier into a coin flip,
and that the damage lands on the negatives.
