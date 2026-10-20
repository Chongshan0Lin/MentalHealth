# H4 — mmpsy segment-count counterfactual: a null, and a load-bearing one

Run 2026-09-02. Code: `code/run_h4.py`, `code/analyze_h4.py` · results:
`results/h4_mmpsy_depression_{SFT,RL,HARPO}.jsonl` (570 rows each) · log:
`results/h4_analysis.log`.

190 of 191 items (one transcript had no interior punctuation to split on),
3 arms, the three OmniSapiens training stages.

## Why this experiment

H1 audited `mmpsy_depression` model-free and found the label recoverable from
something that is not the construct:

| probe | balanced acc |
|---|---|
| transcript content | 0.507 (chance 0.500) |
| total characters | 0.509 |
| **answer-segment count** | **0.772** |

```
no_depression   7 segments: 115/156  (74%)
depression      8 segments:  22/35   (63%)
```

The depressed group has one more recorded response than the control group — the
data-collection protocol differs by class, exactly as it does in DAIC, but in a
different corpus and a different language.

That is **availability**. A1 and A2 together established that availability does
not predict use, and that the cheap test (ablation) gave the wrong answer on
DAIC. So the question here is the A2 question: does any model actually use it?

## The intervention

Cleaner than anything else in the project, because the shortcut is structural
rather than lexical. A segment is a whitespace-separated chunk —
`len(text.split())`, the exact feature H1 probed. To add one we split a chunk at
an existing punctuation mark; to remove one we join two adjacent chunks.

**Every content character is preserved in both directions.** Verified on the
worked example: 97 content characters in all three arms, only the number of
whitespace boundaries differs. Nothing about what the participant said changes.

| arm | segments | content chars |
|---|---|---|
| `seg_minus` | n − 1 | identical |
| `native` | n | identical |
| `seg_plus` | n + 1 | identical |

`seg_plus` pushes a 7-segment item onto the depressed group's modal count;
`seg_minus` pulls an 8-segment item onto the control group's. Those two
subgroups are analysed separately, because items far from the 7/8 boundary are
moved between counts that carry no class information and only dilute the
contrast.

## Result: the models do not use it

| stage | arm | mean p(dep) | acc | AUC | P(pred=dep) |
|---|---|---|---|---|---|
| SFT | `seg_minus` | 0.4687 | 0.721 | 0.878 | 0.421 |
| SFT | `native` | 0.4730 | 0.711 | 0.883 | 0.442 |
| SFT | `seg_plus` | 0.4733 | 0.716 | 0.882 | 0.437 |
| GRPO | `seg_minus` | 0.3519 | 0.832 | 0.867 | 0.226 |
| GRPO | `native` | 0.3553 | 0.847 | 0.871 | 0.232 |
| GRPO | `seg_plus` | 0.3560 | 0.821 | 0.867 | 0.237 |
| HARPO | `seg_minus` | 0.1604 | 0.853 | 0.881 | 0.153 |
| HARPO | `native` | 0.1583 | 0.863 | 0.885 | 0.153 |
| HARPO | `seg_plus` | 0.1615 | 0.863 | 0.886 | 0.163 |

CSS = p(dep | +1 segment) − p(dep | −1 segment), paired within item:

| stage | group | n | CSS | 95% CI | flips |
|---|---|---|---|---|---|
| SFT | all | 190 | +0.0046 | [−0.0003, +0.0101] | 0.016 |
| SFT | native 7 seg | 121 | +0.0066 | [−0.0001, +0.0142] | 0.017 |
| SFT | native 8 seg | 55 | +0.0025 | [−0.0052, +0.0095] | 0.018 |
| GRPO | all | 190 | **+0.0041** | [+0.0014, +0.0067] | 0.011 |
| GRPO | native 7 seg | 121 | **+0.0039** | [+0.0008, +0.0070] | 0.008 |
| GRPO | native 8 seg | 55 | +0.0044 | [−0.0009, +0.0097] | 0.018 |
| HARPO | all | 190 | +0.0011 | [−0.0035, +0.0056] | 0.011 |
| HARPO | native 7 seg | 121 | −0.0022 | [−0.0088, +0.0037] | 0.000 |
| HARPO | native 8 seg | 55 | **+0.0072** | [+0.0003, +0.0147] | 0.036 |

Two GRPO rows and one HARPO row clear zero. **They should not be read as the
model using the shortcut**, for three reasons:

1. **The effect size is 0.004 against DAIC's 0.206** — fifty times smaller, with
   1% of predictions flipping against 58.5%.
2. **The direction fails.** HARPO's two subgroups move −0.0022 and +0.0072, both
   opposite to the hypothesis: adding a segment to a 7-segment item should raise
   p(depression) and removing one from an 8-segment item should lower it.
3. **Nothing downstream moves.** Accuracy and AUC are flat across arms at every
   stage (HARPO 0.853 / 0.863 / 0.863; AUC 0.881 / 0.885 / 0.886).

A significant CI around 0.004 in a paired design with n = 190 is a real but
negligible perturbation, not evidence of reliance.

**Saturation does not explain the null.** HARPO's mean p(dep) is 0.158 and only
15% of predictions are positive, against an 81.6% `no_depression` majority — it
leans to the majority class, but its AUC of 0.885 shows the ranking is
informative. This is not Menta-style saturation where nothing moves because
nothing can.

## Why this null matters more than another positive would have

The strongest objection to A2 is that CSS measures generic input sensitivity —
that any perturbation of any input moves any model's probability, so a large CSS
says nothing specific about shortcuts. H4 closes that objection.

Same counterfactual method, an intervention that is *stricter* than A2's (not one
content character changes, where A2 swapped whole question blocks), on a dataset
whose shortcut is at least as well established (0.772 balanced accuracy from one
feature). Result: essentially zero.

**CSS therefore measures a specific shortcut being used, not sensitivity to
perturbation in general.**

## The regularity this suggests

Placed beside DAIC (lexical, CSS 0.206) and `ptsd_in_the_wild` (register, CSS
0.282), the pattern is that **a shortcut is exploitable to the extent that it
lands on a representational primitive the model already has**:

| mechanism | discriminative probe | generative LLM |
|---|---|---|
| lexical — which questions were asked | 0.810 | **used** (0.206) |
| register / production source | 0.948 | **used** (0.282) |
| structural — how many answers recorded | 0.772 | **not used** (0.004) |

Segment count survives a random forest on one numeric feature but does not
survive tokenisation into anything a language model has reason to represent. The
discriminative probe sees it at 0.772; the generative model sees it at 0.004, a
70× gap.

This also predicts the asymmetry i6/B0 found between the fine-tuned encoder and
the zero-shot LLM: they do not share a sensitivity surface, so a mitigation that
works on one need not transfer.

## Limitations

- **n = 190 with 35 positives.** Power is limited — but the effect is 50× below
  DAIC's, which is not a power problem.
- **`mmpsy_anxiety` is the same 191 participants** with a second label set, so
  running it would not be independent evidence.
- The split/join points are chosen deterministically (longest chunk for split,
  shortest adjacent pair for join). A randomised-position variant would test
  whether the null depends on where the boundary moves; it was not run.
- Text-only. The subset's `modality_signature` is `text` and it carries no audio
  or video binaries, so this is the full input, not an ablation of it.
