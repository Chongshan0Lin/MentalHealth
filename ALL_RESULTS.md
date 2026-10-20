# ALL RESULTS — every number this project has produced

Generated 2026-09-02 directly from the result files on disk. Each section names
its source path so any figure can be traced back and recomputed.

**Provenance convention.** Numbers marked *(computed)* were recomputed from raw
`.jsonl` by this script. Numbers marked *(stored)* were read from a summary
`.json` written by the experiment that produced them. Nothing here is recalled
from memory or copied from prose.

**A note on protocols.** Several experiments measure the same quantity under
different harnesses and do not agree exactly — most visibly MentaLLaMA's DAIC
full-input AUC, which is 0.663 in A1/I1 (188 participants, `full` condition) and
0.684 in the i10 sweep. Both are correct for their own protocol. **Never mix them
inside one table.**

---

## A1 — DAIC ablation ladder

`runs/a1_{MentaLLaMA-chat-7B,Menta}.jsonl` · 188 participants × 7 conditions × 2
labels = 1,316 rows per model. Teacher-forced label log-likelihood over
`{Yes, No}`, IMHI template, 4,032-token budget applied identically per condition.

Label distribution: n = 188, 55 positive, **majority-class accuracy 0.707**,
chance balanced accuracy 0.500.

### A1.1 Data-level shortcut ceiling — TF-IDF + LogReg, 5-fold × 10 repeats *(stored, `logs/a1_lexical_ceiling.log`)*

| condition | CV AUC | shuffled control |
|---|---|---|
| **ellie_only (interviewer turns alone)** | **0.810** | 0.493 |
| ellie_only, prompt-IDs stripped | 0.808 | 0.485 |
| full_stripped | 0.794 | 0.546 |
| full | 0.776 | 0.559 |
| ellie_masked | 0.757 | 0.565 |
| participant_only | 0.756 | 0.570 |

The interviewer's questions alone out-predict both the full interview and the
participant's own words. Stratified by prompt-ID presence: with IDs 0.844,
without IDs 0.846 — the shortcut is not the annotation artifact.


### A1.2 MentaLLaMA-chat-7B — model AUC per condition *(computed)*

| condition | AUC | accuracy | P(pred=Yes) |
|---|---|---|---|
| `full` | 0.663 | 0.739 | 0.128 |
| `full_stripped` | 0.715 | 0.734 | 0.101 |
| `participant_only` | 0.787 | 0.718 | 0.340 |
| `ellie_only` | 0.546 | 0.691 | 0.016 |
| `ellie_only_stripped` | 0.513 | 0.691 | 0.016 |
| `ellie_masked` | 0.763 | 0.729 | 0.330 |
| `empty` | 0.500 | 0.293 | 1.000 |

### A1.2 Menta — model AUC per condition *(computed)*

| condition | AUC | accuracy | P(pred=Yes) |
|---|---|---|---|
| `full` | 0.778 | 0.431 | 0.840 |
| `full_stripped` | 0.778 | 0.431 | 0.840 |
| `participant_only` | 0.727 | 0.293 | 0.989 |
| `ellie_only` | 0.576 | 0.293 | 1.000 |
| `ellie_only_stripped` | 0.569 | 0.293 | 1.000 |
| `ellie_masked` | 0.720 | 0.293 | 1.000 |
| `empty` | 0.500 | 0.293 | 1.000 |

Label-shuffle floors sit at 0.572–0.583. `ellie_only` does not clear its floor
for either model: **the models cannot decode the label from the interviewer
channel**, which is what A2 later shows to be the wrong question.

F5 paired bootstrap, restricted to the 123 ID-carrying transcripts:
MentaLLaMA `ellie_only` AUC 0.647 with IDs → 0.443 stripped,
**Δ = +0.204 [+0.047, +0.357]** — without the strip we would have reported a
spurious replication driven entirely by leaked annotation tokens.

---

## A2 — DAIC interviewer counterfactual

Protocol locked in `experiments/a2-ellie-counterfactual/protocol.md` before any
result was computed.

### A2a — the mechanism, model-free *(stored, `results/a2a_prompt_label_association.json`)*

123 transcripts carry scripted prompt IDs; 75 IDs tested (asked to 20–103
participants); 4 survive Benjamini-Hochberg.

| scripted prompt | OR | dep. rate asked | not asked | q |
|---|---|---|---|---|
| “what got you to seek help” | 11.26 | 0.56 | 0.10 | 2.5e-05 |
| “do you still go to therapy now” | 10.42 | 0.54 | 0.10 | 2.5e-05 |
| “are they triggered by something” | 8.50 | 0.58 | 0.14 | 4.1e-04 |
| “how long ago were you diagnosed” | 8.24 | 0.53 | 0.12 | 2.0e-04 |

Reverse direction — the default script, reached when there is no clinical thread:

| scripted prompt | OR | asked | not asked |
|---|---|---|---|
| “tell me about your relationship with your fami” | 0.25 | 0.17 | 0.44 |
| “tell me about a time when someone made you fee” | 0.27 | 0.11 | 0.32 |
| “how would your best friend describe you” | 0.28 | 0.18 | 0.45 |

**Question-selection-only predictor** — binary asked/not-asked matrix over
75 prompt IDs, no text at all, n = 123:
**CV AUC 0.753** (sd 0.025 over 10 CV seeds; shuffled 0.500).

### A2b — the model-level counterfactual *(computed)*

Each participant's own turns truncated once and reused byte-identical across all
arms; only the five interleaved interviewer questions differ.

Pool token counts: `cf_neutral` 63, `cf_depression` 52, `cf_empirical` 34 — the
clinical pool is **shorter** than the biographical one, so the headline effect
cannot be an artifact of more interviewer text.


**MentaLLaMA-chat-7B** (n = 188)

| arm | mean p(dep) | accuracy | AUC | P(pred=Yes) |
|---|---|---|---|---|
| `cf_neutral` | 0.3730 | 0.750 | 0.764 | 0.160 |
| `cf_none` | 0.4458 | 0.707 | 0.777 | 0.351 |
| `cf_empirical` | 0.4965 | 0.654 | 0.729 | 0.447 |
| `cf_depression` | 0.5787 | 0.495 | 0.734 | 0.745 |

CSS (clinical − biographical) = **+0.2057**, prediction flips 0.585
— on depressed +0.1826, on **not** depressed +0.2153

**Menta** (n = 188)

| arm | mean p(dep) | accuracy | AUC | P(pred=Yes) |
|---|---|---|---|---|
| `cf_neutral` | 0.9849 | 0.293 | 0.703 | 1.000 |
| `cf_none` | 0.9666 | 0.293 | 0.731 | 0.989 |
| `cf_empirical` | 0.9724 | 0.298 | 0.726 | 0.995 |
| `cf_depression` | 0.9703 | 0.293 | 0.759 | 1.000 |

CSS (clinical − biographical) = **-0.0147**, prediction flips 0.000
— on depressed +0.0078, on **not** depressed -0.0239

**osim8b** (n = 188)

| arm | mean p(dep) | accuracy | AUC | P(pred=Yes) |
|---|---|---|---|---|
| `cf_neutral` | 0.6573 | 0.617 | 0.799 | 0.612 |
| `cf_none` | 0.5894 | 0.649 | 0.804 | 0.548 |
| `cf_empirical` | 0.8307 | 0.388 | 0.778 | 0.894 |
| `cf_depression` | 0.8325 | 0.383 | 0.808 | 0.910 |

CSS (clinical − biographical) = **+0.1752**, prediction flips 0.298
— on depressed +0.0551, on **not** depressed +0.2249

**qwen3_base** (n = 188)

| arm | mean p(dep) | accuracy | AUC | P(pred=Yes) |
|---|---|---|---|---|
| `cf_neutral` | 0.4396 | 0.718 | 0.792 | 0.436 |
| `cf_none` | 0.4340 | 0.713 | 0.798 | 0.431 |
| `cf_empirical` | 0.5193 | 0.686 | 0.779 | 0.500 |
| `cf_depression` | 0.6934 | 0.537 | 0.793 | 0.691 |

CSS (clinical − biographical) = **+0.2537**, prediction flips 0.255
— on depressed +0.1442, on **not** depressed +0.2990

MentaLLaMA's CI is [+0.1945, +0.2168]. **AUC barely moves (0.764 → 0.734) while
P(pred=Yes) goes 0.160 → 0.745**: the interviewer context acts as an additive
bias on the decision threshold, not as evidence that changes the ranking. The
non-depressed move further than the depressed, which is why accuracy collapses
below the majority baseline.

Menta is saturated (p ≈ 0.97, zero prediction flips) and is a null here for that
reason rather than a contradiction.

---

## C1 — IMHI / SDCNL provenance

`runs/c1_{MentaLLaMA-chat-7B,Menta}.jsonl` · 2,864 items × 4 conditions,
11,456 rows per model. 10 sources, matched subset stratified by label.

### C1.1 Per-source accuracy, MentaLLaMA *(computed)*

| source | n | \|L\| | majority | native | normalized | native_nopost | Δ_template |
|---|---|---|---|---|---|---|---|
| `CAMS` | 278 | 6 | 0.180 | 0.442 | 0.457 | 0.180 | -0.014 |
| `DR` | 248 | 2 | 0.605 | 0.702 | 0.899 | 0.605 | -0.198 |
| `Irf` | 300 | 2 | 0.500 | 0.773 | 0.720 | 0.623 | +0.053 |
| `MultiWD` | 300 | 2 | 0.500 | 0.710 | 0.697 | 0.587 | +0.013 |
| `SAD` | 297 | 9 | 0.111 | 0.643 | 0.286 | 0.111 | +0.357 |
| `SDCNL` | 300 | 2 | 0.500 | 0.650 | 0.657 | 0.500 | -0.007 |
| `dreaddit` | 300 | 2 | 0.500 | 0.770 | 0.710 | 0.500 | +0.060 |
| `loneliness` | 265 | 2 | 0.566 | 0.732 | 0.808 | 0.566 | -0.075 |
| `swmh` | 300 | 5 | 0.200 | 0.730 | 0.647 | 0.200 | +0.083 |
| `t-sid` | 276 | 4 | 0.272 | 0.699 | 0.670 | 0.272 | +0.029 |

`Δ_prior` = Acc(native_nopost) − Acc(normalized_nopost) is **exactly 0.000 for all
10 sources and both models** — but this is an identity, not a measurement: both
templates select the same argmax, and a constant prediction has the same accuracy
under either wording.

The DR contrast is the real template finding: native phrasing makes MentaLLaMA
answer *yes* to 222/248 posts (acc 0.702); the canonical phrasing balances it to
161/248 (acc **0.899**), **Δ = −0.198 [−0.254, −0.141]**. The native IMHI wording
*harms* performance.

Multiclass Δ_template is **not interpretable** — the canonical wording pushes
MentaLLaMA into single-label collapse (SAD: predictions concentrate on
`emotion turmoil`, 49 → 201 of 297).

### C1.2 Provenance probes

**Data level** *(stored, `logs/c1_provenance_probe.log`)* — TF-IDF + LogReg on
post text: 10-way source identity **acc 0.437** (chance 0.100, shuffled 0.100).
Sharpest pair, swmh vs t-sid (near-identical questions, different label spaces):
**AUC 0.967**, acc 0.892.

**Hidden states** *(stored, `logs/c1_hiddenprobe_*.log`)* — post encoded alone, no
question; PCA-256 then logistic regression, 5-fold. Chance 0.100, shuffled 0.100.

| layer | MentaLLaMA last | MentaLLaMA mean | | layer | Menta last | Menta mean |
|---|---|---|---|---|---|---|
| 8 | 0.387 | **0.490** | | 9 | 0.383 | 0.438 |
| 16 | 0.437 | **0.493** | | 18 | 0.434 | 0.476 |
| 24 | 0.460 | 0.477 | | 27 | 0.474 | 0.474 |
| 32 | 0.460 | **0.488** | | 36 | 0.469 | 0.472 |

MentaLLaMA's mean-pooled probe exceeds the 0.437 lexical ceiling at every depth,
and corpus identity survives to the **final-layer last-token state** that drives
generation (0.460 / 0.469). The shape is a plateau, not a mid-layer peak, so
"computed then discarded" is not supported. **Caveat: the 0.490-vs-0.437
comparison has no error bar** — the two come from different CV folds — and lbfgs
did not converge in 2,000 iterations, which understates rather than inflates the
probe.

**Behaviour under the natural label distribution** — the content-free prediction
rescored against real corpus priors rather than the stratified sample:
MentaLLaMA lands on the **exact majority class in 4 of 10 sources** (DR 0.758,
loneliness 0.782, swmh 0.347, t-sid 0.522), mean 0.475 against chance 0.373.

---

## D — SFT → GRPO → HARPO, modality ablation ladder

`experiments/d-training-stage-reliance/results/d_b1_meld_emotion_{SFT,RL,HARPO}.jsonl`
· 300 `meld_emotion` items × 5 conditions × 3 stages, 1,500 rows per stage,
0 skipped, all 300 common to all stages. Architecture, base model, prompt, 336p
video preset, sampling seed and evaluation identical — **training stage is the
only variable**.

Gold distribution: neutral 145, joy 47, anger 37, surprise 32, sadness 27,
disgust 8, fear 4. **Majority 0.483, chance 0.143.** *(computed)*

| stage | condition | raw acc | balanced acc | 95% CI | SS |
|---|---|---|---|---|---|
| SFT | `full` | 0.260 | 0.388 | [0.301, 0.493] | 1.000 |
| SFT | `audio_only` | 0.260 | 0.351 | [0.286, 0.417] | 1.339 |
| SFT | `video_only` | 0.280 | 0.441 | [0.356, 0.536] | 0.530 |
| SFT | `text_only` | 0.267 | 0.394 | [0.304, 0.494] | 0.955 |
| SFT | `question_only` | 0.027 | 0.143 | [0.143, 0.143] | 3.203 |
| RL | `full` | 0.600 | 0.379 | [0.317, 0.443] | 1.000 |
| RL | `audio_only` | 0.623 | 0.396 | [0.330, 0.465] | 0.863 |
| RL | `video_only` | 0.537 | 0.355 | [0.304, 0.405] | 1.207 |
| RL | `text_only` | 0.497 | 0.440 | [0.371, 0.531] | 0.500 |
| RL | `question_only` | 0.027 | 0.143 | [0.143, 0.143] | 2.964 |
| HARPO | `full` | 0.687 | 0.422 | [0.362, 0.497] | 1.000 |
| HARPO | `audio_only` | 0.717 | 0.462 | [0.401, 0.538] | 0.484 |
| HARPO | `video_only` | 0.680 | 0.454 | [0.366, 0.556] | 0.593 |
| HARPO | `text_only` | 0.660 | 0.493 | [0.399, 0.607] | 0.095 |
| HARPO | `question_only` | 0.483 | 0.143 | [0.143, 0.143] | 4.573 |

### The decoupling

| `full` | SFT | GRPO | HARPO | Δ |
|---|---|---|---|---|
| **raw accuracy** | 0.260 | 0.600 | **0.687** | **+0.427** |
| balanced accuracy | 0.388 | 0.379 | 0.422 | +0.033 |
| `question_only` raw acc | 0.027 | 0.027 | **0.483** | +0.457 |
| Δ_media (full − text_only, bal) | -0.005 | -0.060 | **-0.071** | -0.066 |

Raw accuracy rises 2.6× — from *below* the 0.483 majority baseline to well above
it — while balanced accuracy moves inside its confidence interval. The gain is
the majority class, not discrimination.

**`question_only` is the direct evidence.** With video, audio *and* transcript
withheld, HARPO answers `neutral` to every item, so its raw accuracy equals the
majority rate to three decimals. SFT and GRPO do not. Balanced accuracy is
0.143 = 1/7 for all three (a constant predictor), so **reporting only the
balanced figure would have hidden the entire finding**.

**Δ_media is negative and deepens monotonically**: supplying audio and video
*costs* balanced accuracy, increasingly so across training stages. Post-training
does not improve multimodal grounding on this benchmark.

---

## H4 — mmpsy segment-count counterfactual: **a null**

`experiments/h4-mmpsy-counterfactual/results/h4_mmpsy_depression_{SFT,RL,HARPO}.jsonl`
· 190 items × 3 arms × 3 stages (1 item unsplittable). Chinese, HBA-test subset.

**The intervention touches whitespace only** — a chunk is split at existing
punctuation to add a segment, or two chunks joined to remove one. Verified: all
97 content characters identical across arms.

Shortcut availability (H1, model-free): transcript content 0.507 ≈ chance 0.500;
**answer-segment count 0.772**. Native counts: 7 seg × 121, 8 seg × 55.
*(computed)*

| stage | arm | mean p(dep) | accuracy | AUC | P(pred=dep) |
|---|---|---|---|---|---|
| SFT | `seg_minus` | 0.4687 | 0.721 | 0.878 | 0.421 |
| SFT | `native` | 0.4730 | 0.711 | 0.883 | 0.442 |
| SFT | `seg_plus` | 0.4733 | 0.716 | 0.882 | 0.437 |
| RL | `seg_minus` | 0.3519 | 0.832 | 0.867 | 0.226 |
| RL | `native` | 0.3553 | 0.847 | 0.871 | 0.232 |
| RL | `seg_plus` | 0.3560 | 0.821 | 0.867 | 0.237 |
| HARPO | `seg_minus` | 0.1604 | 0.853 | 0.881 | 0.153 |
| HARPO | `native` | 0.1583 | 0.863 | 0.885 | 0.153 |
| HARPO | `seg_plus` | 0.1615 | 0.863 | 0.886 | 0.163 |

| stage | CSS (+1 seg − −1 seg) | 95% CI | flips |
|---|---|---|---|
| SFT | +0.0046 | [−0.0003, +0.0101] | 0.016 |
| GRPO | +0.0041 | [+0.0014, +0.0067] | 0.011 |
| HARPO | +0.0011 | [−0.0035, +0.0056] | 0.011 |

**Fifty times smaller than DAIC's +0.206, with 1% flips against 58.5%.** The
direction also fails: on HARPO the 7-segment group moves −0.0022 and the
8-segment group +0.0072, both opposite to the hypothesis. Accuracy and AUC are
flat across arms.

**The shortcut is in the data (0.772) and the models do not use it.** This is the
control that makes CSS interpretable elsewhere: it shows CSS measures a specific
shortcut being used, not generic sensitivity to any perturbation.

---

## H5 — ptsd_in_the_wild register counterfactual: **the largest effect measured**

`experiments/h5-ptsd-register-cf/results/h5_ptsd_{SFT,RL,HARPO}.jsonl` · 60 items
evaluated (4 held out to build the register pools) × 3 arms × 3 stages, 180 rows
per stage. Text-only. Each item's own transcript byte-identical across arms; the
two context pools are token-matched at 120 words.

Shortcut availability (H1, model-free): text-only balanced acc **0.948**;
**stopwords only 0.880**; scrubbing all PTSD/military vocabulary *raises* it to
0.953. Construct and register are perfectly confounded in this corpus — the
positives are veteran testimony and clinical scenes, the negatives TED-style
talks — so register is manipulated by prepending, not swapping. *(computed)*

| stage | arm | mean p(ptsd) | accuracy | AUC |
|---|---|---|---|---|
| SFT | `ctx_vlog` | 0.5796 | 1.000 | 1.000 |
| SFT | `native` | 0.5648 | 0.967 | 0.997 |
| SFT | `ctx_testimony` | 0.7716 | 0.567 | 0.964 |
| RL | `ctx_vlog` | 0.6817 | 0.667 | 1.000 |
| RL | `native` | 0.7102 | 0.700 | 1.000 |
| RL | `ctx_testimony` | 0.7778 | 0.500 | 0.993 |
| HARPO | `ctx_vlog` | 0.5426 | 1.000 | 1.000 |
| HARPO | `native` | 0.5527 | 1.000 | 1.000 |
| HARPO | `ctx_testimony` | 0.8250 | 0.517 | 1.000 |

| stage | CSS (testimony − vlog) | 95% CI | flips | on positives | on negatives |
|---|---|---|---|---|---|
| SFT | **+0.192** | [+0.143, +0.243] | 0.433 | +0.011 | **+0.373** |
| GRPO | +0.096 | [+0.077, +0.114] | 0.167 | +0.033 | +0.159 |
| HARPO | **+0.282** | [+0.217, +0.349] | 0.483 | +0.034 | **+0.531** |

**Prepending 120 words destroys accuracy without touching the item's own words.**
HARPO scores a perfect 1.000 natively and 0.517 — coin-flipping — under a
testimony-register prefix. AUC is essentially unchanged (1.000 → 1.000), the same
threshold-drag signature as DAIC.

The effect is almost entirely on **true negatives** (HARPO +0.531 against +0.034
on positives): a false-positive generator. Give a TED talk a clinical-register
opening and the model diagnoses PTSD.

**Across stages the trend is non-monotone** — SFT 0.192 → GRPO 0.096 →
HARPO 0.282. GRPO halves register reliance and HARPO returns it above the
starting point. This is *not* the monotone amplification seen on DAIC in the i10
sweep and should not be reported as one. n = 60 (30/30) is the smallest sample in
the project.

---

## i10 — full model sweep, 8 LLMs

`experiments/i10-model-sweep/results/suite_*.json` *(stored)*. All on DAIC.

| model | full AUC | removal AUC | CSS | flips | conflict AUC | construct sens | SER |
|---|---|---|---|---|---|---|---|
| MentaLLaMA | 0.684 | 0.546 | 0.205 | 0.559 | 0.346 | 0.113 | 1.82 |
| Menta | 0.764 | 0.576 | 0.027 | 0.000 | 0.781 | 0.014 | 1.86 |
| qwen3_base | 0.810 | 0.710 | 0.256 | 0.261 | 0.676 | 0.410 | 0.62 |
| OSim-8B | 0.805 | 0.722 | 0.196 | 0.298 | 0.773 | 0.279 | 0.70 |
| OmniSapiens-SFT | 0.800 | 0.686 | 0.457 | 0.479 | 0.305 | 0.278 | 1.64 |
| OmniSapiens-RL | 0.786 | 0.661 | 0.471 | 0.516 | 0.341 | 0.278 | 1.70 |
| OmniSapiens2.0 | 0.781 | 0.699 | 0.505 | 0.511 | 0.292 | 0.365 | 1.39 |
| HumanOmniV2 | 0.791 | 0.625 | 0.372 | 0.511 | 0.445 | 0.356 | 1.04 |

**Removal < full input for 8/8 — no exception.** This does *not* reproduce
Burdisso et al. (2024), who found interviewer-only *outperforming* full input
with classical encoders (0.90 F1 by deliberately exploiting the shortcut). The
contrast is an architecture/training difference, not a partial replication.

**The counterfactual disagreement holds for 7/8.** The three OmniSapiens stages
carry more than double MentaLLaMA's CSS (0.457 / 0.471 / 0.505), rising
monotonically across post-training. Menta is the sole exception (CSS 0.027, zero
flips) and is saturated — its construct sensitivity is 0.014, i.e. it barely
responds to *any* input change.

**Four models fall below AUC 0.5 on the conflict set** — MentaLLaMA 0.346,
OmniSapiens-SFT 0.305, RL 0.341, 2.0 0.292 — under adversarial interviewer
context their ranking is *anti-correlated with the truth*.

**SER (CSS / construct sensitivity) exceeds 1.0 for five of eight models**: the
interviewer channel moves the prediction more than the patient's own words do.

### Per-model interventions *(stored)*


**MentaLLaMA**

| intervention | AUC | CSS | construct sens |
|---|---|---|---|
| `none` | 0.684 | 0.205 | 0.113 |
| `delete_interviewer` | 0.780 | 0.000 | 0.203 |
| `blank_words` | 0.759 | 0.000 | 0.187 |
| `marginalise_2` | 0.760 | 0.000 | — |
| `marginalise_4` | 0.765 | 0.000 | — |
| `targeted_removal` | 0.668 | — | — |
| `threshold_proxy` | 0.683 | — | — |
| `instruct_ignore` | 0.753 | 0.094 | — |

**Menta**

| intervention | AUC | CSS | construct sens |
|---|---|---|---|
| `none` | 0.764 | 0.027 | 0.014 |
| `delete_interviewer` | 0.729 | 0.000 | 0.040 |
| `blank_words` | 0.725 | 0.000 | 0.022 |
| `marginalise_2` | 0.744 | 0.000 | — |
| `marginalise_4` | 0.740 | 0.000 | — |
| `instruct_ignore` | 0.734 | 0.141 | — |

**qwen3_base**

| intervention | AUC | CSS | construct sens |
|---|---|---|---|
| `none` | 0.810 | 0.256 | 0.410 |
| `delete_interviewer` | 0.798 | 0.000 | 0.579 |
| `blank_words` | 0.795 | 0.000 | 0.616 |
| `marginalise_2` | 0.793 | 0.000 | — |
| `marginalise_4` | 0.796 | 0.000 | — |
| `instruct_ignore` | 0.809 | 0.058 | — |

**OSim-8B**

| intervention | AUC | CSS | construct sens |
|---|---|---|---|
| `none` | 0.805 | 0.196 | 0.279 |
| `delete_interviewer` | 0.803 | 0.000 | 0.453 |
| `blank_words` | 0.792 | 0.000 | 0.443 |
| `marginalise_2` | 0.806 | 0.000 | — |
| `marginalise_4` | 0.804 | 0.000 | — |
| `targeted_removal` | 0.805 | — | — |
| `threshold_proxy` | 0.807 | — | — |
| `instruct_ignore` | 0.794 | 0.165 | — |

**OmniSapiens-SFT**

| intervention | AUC | CSS | construct sens |
|---|---|---|---|
| `none` | 0.800 | 0.457 | 0.278 |
| `delete_interviewer` | 0.818 | 0.000 | 0.544 |
| `blank_words` | 0.802 | 0.000 | 0.558 |
| `marginalise_2` | 0.812 | 0.000 | — |
| `marginalise_4` | 0.813 | 0.000 | — |
| `targeted_removal` | 0.797 | — | — |
| `threshold_proxy` | 0.808 | — | — |
| `instruct_ignore` | 0.813 | 0.506 | — |

**OmniSapiens-RL**

| intervention | AUC | CSS | construct sens |
|---|---|---|---|
| `none` | 0.786 | 0.471 | 0.278 |
| `delete_interviewer` | 0.812 | 0.000 | 0.535 |
| `blank_words` | 0.801 | 0.000 | 0.542 |
| `marginalise_2` | 0.809 | 0.000 | — |
| `marginalise_4` | 0.802 | 0.000 | — |
| `targeted_removal` | 0.788 | — | — |
| `threshold_proxy` | 0.793 | — | — |
| `instruct_ignore` | 0.808 | 0.531 | — |

**OmniSapiens2.0**

| intervention | AUC | CSS | construct sens |
|---|---|---|---|
| `none` | 0.781 | 0.505 | 0.365 |
| `delete_interviewer` | 0.818 | 0.000 | 0.571 |
| `blank_words` | 0.807 | 0.000 | 0.585 |
| `marginalise_2` | 0.804 | 0.000 | — |
| `marginalise_4` | 0.807 | 0.000 | — |
| `targeted_removal` | 0.784 | — | — |
| `threshold_proxy` | 0.791 | — | — |
| `instruct_ignore` | 0.809 | 0.577 | — |

**HumanOmniV2**

| intervention | AUC | CSS | construct sens |
|---|---|---|---|
| `none` | 0.791 | 0.372 | 0.356 |
| `delete_interviewer` | 0.780 | 0.000 | 0.429 |
| `blank_words` | 0.773 | 0.000 | 0.392 |
| `marginalise_2` | 0.783 | 0.000 | — |
| `marginalise_4` | 0.780 | 0.000 | — |
| `targeted_removal` | 0.785 | — | — |
| `threshold_proxy` | 0.791 | — | — |
| `instruct_ignore` | 0.787 | 0.249 | — |

---

## i7 — causal probing: LEACE concept erasure *(stored)*

`experiments/i7-causal-probing/results/phase3_leace_*.json`

| condition | shortcut sens | construct sens | SER | AUC |
|---|---|---|---|---|
| no erasure | 0.2057 | 0.1765 | 1.1653 | 0.7769 |
| LEACE, layers 8–24 | 0.1580 | 0.1745 | 0.9059 | 0.7664 |
| random direction, same norm | 0.2068 | 0.1774 | 1.1659 | 0.7767 |

Multi-layer erasure removes **23.2%** of shortcut dependence
(0.2057 → 0.1580) while construct
sensitivity (0.1765 → 0.1745) and AUC
(0.7769 → 0.7664) are essentially untouched. A
random direction of the same norm does nothing — the control is the point.

Single-layer (13) erasure removes only **6.3%**, so the multi-layer result is
the one to report: the signal is not one clean linear direction but is
re-instantiated non-linearly along the residual stream.

---

## i9 — format / instruction interventions *(stored)*

`experiments/i9-format-instruction/results/format_ser.json`

| variant | CSS | construct | SER | iid AUC | conflict AUC | flips |
|---|---|---|---|---|---|---|
| `base` | 0.2050 | 0.1803 | 1.1374 | 0.7797 | 0.3458 | 0.5585 |
| `fmt_prefix` | 0.1686 | 0.1748 | 0.9650 | 0.7990 | 0.3859 | 0.4947 |
| `fmt_suffix` | 0.1331 | 0.1597 | 0.8334 | 0.6695 | 0.4026 | 0.3723 |

---

## i8 — GRPO post-training mitigation *(stored)*

`experiments/i8-grpo-mitigation/results/eval_*.json`. Reported by its own
analysis as **a failed experiment that is worth reporting**.

| checkpoint | conflict AUC | conflict acc | CSS | construct | SER | iid AUC | parse rate |
|---|---|---|---|---|---|---|---|
| `baseline` | 0.0686 | 0.1330 | 0.2864 | 0.1354 | 2.1149 | 0.7630 | 0.1500 |
| `M0` | 0.2664 | 0.7074 | 0.0017 | 0.0025 | 0.6865 | 0.7245 | 1.0000 |
| `M0_bal_batch` | 0.3427 | 0.7074 | 0.0000 | 0.0006 | 0.0700 | 0.6752 | 1.0000 |
| `M1` | 0.2768 | 0.7074 | 0.0014 | 0.0033 | 0.4337 | 0.7096 | 1.0000 |

---

## i6 — encoder-arm interventions, MentalBERT, 3 seeds *(stored)*

`experiments/i6-encoder-interventions/results/encoder_interventions.json`

| method | AUC (mean) | AUC sd | CSS (mean) | construct sens | SER |
|---|---|---|---|---|---|
| `B0` | 0.6457 | 0.0719 | 0.0096 | 0.1343 | 0.0717 |
| `D3` | 0.7626 | 0.0270 | 0.0213 | 0.4171 | 0.0511 |
| `M1` | 0.5779 | 0.0108 | 0.0013 | 0.0193 | 0.0658 |
| `M2` | 0.6407 | 0.0542 | 0.0106 | 0.1197 | 0.0884 |
| `M2reg` | 0.6299 | 0.0752 | 0.0115 | 0.1222 | 0.0940 |

The encoder's counterfactual sensitivity is an order of magnitude below the
zero-shot LLM's (CSS ≈ 0.01–0.03 against 0.206), even though its raw ability to
predict from interviewer-only input is high. **The encoder has encoded the
shortcut but does not rely on it the way the LLM does** — which is why the
train-time mitigation arm returned nothing.

---

## H1 — HBA-test dataset audit, all 16 subsets *(stored, model-free)*

`experiments/h1-hba-shortcut/results/audit.json`. No model is loaded anywhere.
`text` = balanced accuracy from transcript alone; `stopwords` = same probe with
every content word deleted (high ⇒ the classes differ in register, not in a
removable word list).

| subset | n | chance | majority | text | stopwords | chars | segments |
|---|---|---|---|---|---|---|---|
| `mosei_senti` | 4659 | 0.143 | 0.239 | 0.260 | 0.203 | 0.157 | 0.167 |
| `meld_emotion` | 2610 | 0.143 | 0.481 | 0.288 | 0.187 | 0.196 | 0.186 |
| `meld_senti` | 2609 | 0.333 | 0.481 | 0.505 | 0.391 | 0.369 | 0.356 |
| `mosei_emotion` | 1753 | 0.167 | 0.601 | 0.281 | 0.258 | 0.178 | 0.171 |
| `cremad` | 1117 | 0.167 | 0.189 | 0.170 | 0.176 | 0.171 | 0.194 |
| `chsimsv2` | 1034 | 0.200 | 0.273 | 0.324 | — | 0.224 | 0.199 |
| `tess` | 420 | 0.143 | 0.143 | 0.102 | 0.143 | 0.159 | 0.143 |
| `urfunny` | 224 | 0.500 | 0.554 | 0.504 | 0.476 | 0.538 | 0.512 |
| `mmpsy_depression` | 191 | 0.500 | 0.817 | 0.507 | — | 0.509 | 0.772 |
| `mmpsy_anxiety` | 191 | 0.500 | 0.937 | 0.500 | — | 0.429 | 0.771 |
| `mmsd` | 103 | 0.500 | 0.573 | 0.664 | 0.532 | 0.461 | 0.471 |
| `ptsd_in_the_wild` | 64 | 0.500 | 0.500 | 0.948 | 0.880 | 0.631 | 0.619 |
| `daicwoz` | 47 | 0.500 | 0.702 | 0.607 | 0.527 | 0.307 | 0.413 |

Not scored (open-ended free text, ~1 distinct answer per item): `intentqa`
(2,134 rows / 1,813 distinct answers), `siq2` (875/865), `mimeqa` (173/173).

**Calibration pair.** `cremad` (0.170 vs chance 0.167) and `tess` (0.102 vs
0.143) use a fixed emotionally-neutral carrier sentence, so text-only accuracy
must sit at chance by construction. It does — this validates every other probe
in the table.

**`ptsd_in_the_wild` detail:** full text 0.948 · vocabulary-scrubbed **0.953**
(rises) · stopwords only 0.880 · first 100 words 0.844 · first 25 words 0.719 ·
length alone AUC 0.770. 19/32 positives contain the literal string `ptsd`;
0/32 negatives do.

**`mmpsy_*` detail:** content ≈ chance, but answer-segment count reaches 0.772 /
0.771. `no_depression` is 74% exactly 7 segments; `depression` 63% exactly 8.
Both subsets are the **same 191 participants** with two label sets.

---

## H2 — IMHI dataset audit *(stored, model-free)*

`experiments/h2-imhi-shortcut/results/audit.json`

**P1 label structure** — 23 labels; global majority 0.206; source-conditional
majority **0.442**.

**P2 source identification from post text** — acc **0.522** (chance 0.111);
stopwords only 0.372.

**P3 per-source probes**

| source | n | chance | majority | text | stopwords |
|---|---|---|---|---|---|
| `swmh` | 10865 | 0.200 | 0.347 | 0.684 | 0.374 |
| `MultiWD` | 2441 | 0.500 | 0.637 | 0.499 | 0.500 |
| `Irf` | 2113 | 0.500 | 0.646 | 0.643 | 0.622 |
| `t-sid` | 959 | 0.250 | 0.522 | 0.618 | 0.375 |
| `SAD` | 684 | 0.111 | 0.161 | 0.416 | 0.193 |
| `CAMS` | 625 | 0.167 | 0.294 | 0.303 | 0.286 |
| `loneliness` | 528 | 0.500 | 0.782 | 0.635 | 0.597 |
| `dreaddit` | 414 | 0.500 | 0.510 | 0.692 | 0.591 |
| `DR` | 405 | 0.500 | 0.758 | 0.769 | 0.712 |

**P6 `swmh` scrub test** — full 0.684 → label-vocabulary scrubbed
**0.594** → stopwords only 0.374 (chance 0.200).
Scrubbing the indicative vocabulary removes less than half the signal.

**P5 duplicates** — 15,515 distinct texts in 19,034 rows;
372 cross-source duplicates, 1040 within-source,
and **288 conflicting** (identical text, different label).

**P4 template recoverability** — fraction of rows whose source is identifiable
from the instruction template alone: `swmh` 1.00, `t-sid` 1.00, `SAD` 1.00,
`CAMS` 1.00; `MultiWD`/`Irf`/`loneliness`/`dreaddit`/`DR` 0.00.

---

## Index of raw artefacts

| experiment | results | analysis |
|---|---|---|
| A1 | `runs/a1_*.jsonl` (1,316 × 2) | `runs/A1_RESULTS.md` |
| A2 | `experiments/a2-ellie-counterfactual/results/` (752 × 4 models) | `analysis.md` ✓ |
| C1 | `runs/c1_*.jsonl` (11,456 × 2), `runs/c1_hidden_*.npz` | — |
| D | `experiments/d-training-stage-reliance/results/` (1,500 × 3) | `analysis.md` ✓ |
| H1 | `experiments/h1-hba-shortcut/results/audit.json` | `analysis.md` ✓ |
| H2 | `experiments/h2-imhi-shortcut/results/audit.json` | — |
| H3 | `experiments/h3-esconv-shortcut/data/` only | **not run** |
| H4 | `experiments/h4-mmpsy-counterfactual/results/` (570 × 3) | — |
| H5 | `experiments/h5-ptsd-register-cf/results/` (180 × 3) | — |
| B0 | `experiments/b0-encoder-mechanism/results/` | `analysis.md` ✓ |
| B2 | `experiments/b2-modality-conflict/results/` | — |
| i1–i9 | `experiments/i*/results/` | `analysis.md` ✓ (i1–i6, i8) |
| i10 | `experiments/i10-model-sweep/results/suite_*.json` (8 models) | — |
| suite | `experiments/suite/results/` | `analysis.md` ✓ |

**Gaps:** H3 (ESConv) has data but no code or results. B2, H2, H4, H5, i7, i9,
i10 have results but no written analysis. C1 has no `analysis.md` — its numbers
live in `RESULTS.md` and this file.

---

## H6 — Irf register counterfactual *(computed)*

`experiments/h6-irf-register-cf/results/h6_irf_*.jsonl` · 400 posts × 4 arms ×
2 models, 1,600 rows each. 9 posts held out to build the pools; pools
token-matched at 150 words. IMHI — **the models were instruction-tuned on this
corpus**.

Shortcut availability (H2, model-free): text 0.643, **stopwords only 0.622 —
97% of the full-text signal**, the highest such ratio in the project.

The `ctx_neutral` arm is a prefix drawn from **MultiWD**, the one dataset whose
text carries no label signal (0.499 vs chance 0.500). It is the zero point that
separates "reads the register" from "reacts to any clinical-sounding prefix".

| model | arm | mean p(yes) | acc | AUC | P(pred=yes) |
|---|---|---|---|---|---|
| MentaLLaMA-chat-7B | `ctx_no` | 0.7405 | 0.752 | 0.714 | 0.840 |
| MentaLLaMA-chat-7B | `native` | 0.6504 | 0.757 | 0.849 | 0.740 |
| MentaLLaMA-chat-7B | `ctx_neutral` | 0.6424 | 0.767 | 0.843 | 0.780 |
| MentaLLaMA-chat-7B | `ctx_yes` | 0.6583 | 0.765 | 0.843 | 0.818 |

**MentaLLaMA-chat-7B** — CSS (yes-register − no-register) = **-0.0822**
- vs signal-free prefix: `ctx_no` +0.0981, `ctx_yes` +0.0159, signal-free vs native -0.0081
- by question: thwarted belongingness -0.1801 (n=202), perceived burdensomeness +0.0177 (n=198)

| Menta | `ctx_no` | 0.9948 | 0.670 | 0.573 | 0.998 |
| Menta | `native` | 0.9446 | 0.685 | 0.673 | 0.948 |
| Menta | `ctx_neutral` | 0.8767 | 0.700 | 0.712 | 0.892 |
| Menta | `ctx_yes` | 0.9709 | 0.675 | 0.694 | 0.983 |

**Menta** — CSS (yes-register − no-register) = **-0.0239**
- vs signal-free prefix: `ctx_no` +0.1182, `ctx_yes` +0.0943, signal-free vs native -0.0680
- by question: thwarted belongingness -0.0015 (n=202), perceived burdensomeness -0.0468 (n=198)

CIs (MentaLLaMA): CSS [−0.0953, −0.0696]; `ctx_no` − signal-free
[+0.0842, +0.1124]; signal-free − native [−0.0175, +0.0019] **n.s.**
By question: belongingness [−0.1962, −0.1647], burdensomeness [+0.0127, +0.0230].

**The sign is negative for both models.** Both `Irf` prefixes raise p(yes) and
the *`no`*-class prefix raises it more, while the signal-free prefix does
nothing. The model is reacting to the presence of clinical-sounding first-person
text, not reading which class the prefix belongs to — a topic-presence effect,
not use of the register shortcut.

**The two questions give opposite signs, 0.20 apart** (−0.180 vs +0.018 on
MentaLLaMA; Menta splits the same way). `Irf` bundles two constructs under one
name and the same model behaves differently on each; it should not be reported
as a single task.

---

## Cross-experiment synthesis: what predicts whether a shortcut gets used

Four counterfactuals, each on a dataset whose shortcut was first established
model-free.

| dataset | mechanism | data-level probe | CSS | used? |
|---|---|---|---|---|
| DAIC-WOZ | lexical — which questions were asked | AUC 0.810; question-selection alone 0.753 | **+0.206** | **yes** |
| `ptsd_in_the_wild` | register, **cross-source** | 0.948 (stopwords 0.880) | **+0.282** | **yes** |
| `mmpsy_depression` | structural — how many answers recorded | 0.772 | +0.004 | no |
| `Irf` | register, **same-source** | 0.643 (stopwords 0.622) | **−0.082** | no (wrong sign) |

### The mechanism label does not predict use — granularity does

Two register shortcuts land on opposite sides, so "register shortcuts get used"
is false. What separates them is how coarse the class difference is:

- `ptsd_in_the_wild`'s classes come from **different production sources**
  entirely — veteran testimony against TED talks. The gap is simultaneously
  topical, lexical and stylistic.
- `Irf`'s classes are **the same kind of text from the same platform**. The gap
  is a fine-grained difference in function-word distribution: 0.622 to a TF-IDF
  probe, invisible to the language model.

`mmpsy`'s segment count fails for the same reason despite sharing nothing else
with `Irf`: a count of whitespace boundaries survives a random forest on one
numeric feature and does not survive tokenisation into anything a language model
represents.

**A shortcut is exploitable to the extent that its discriminative signal is
coarse enough to cross the model's representational grain.**

### Consequences

1. **A discriminative probe is an upper bound on model risk, and it can
   overstate by a wide margin.** `Irf`: 0.622 for the probe, wrong-signed −0.082
   for the model the probe is supposed to be warning about. Dataset audits and
   model audits answer different questions and neither substitutes for the other.
2. **The counterfactual needs a signal-free control arm.** Without
   `ctx_neutral`, H6's negative CSS would have read as a mysterious inversion
   instead of a misattributed topic-presence effect. H5 lacks this arm and
   should be re-run with it before its +0.282 is treated as settled.
3. **`mmpsy`'s null is what makes the positives interpretable.** Same method, a
   stricter intervention (not one content character changed), a shortcut at least
   as well established — and essentially zero. CSS measures a specific shortcut
   being used, not sensitivity to perturbation in general.
4. **Where models do use a shortcut, the signature is consistent**: AUC barely
   moves while the operating point is dragged across the threshold
   (DAIC 0.764→0.734 with P(pred) 0.160→0.745; ptsd 1.000→1.000 with accuracy
   1.000→0.517). The damage is a bias shift, and it lands on the negatives.

