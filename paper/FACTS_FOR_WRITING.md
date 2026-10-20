# Verified facts for the paper

Every number below was read off disk on 2026-09-01, with the producing file named
so it can be re-checked. Nothing here is from memory. Items are numbered to match
the ten requests.

**Three things in the request list turned out to already exist and to be stronger
than assumed** — the Form-1 sweep (§3), the causal ablation (§7), and the
probing↔behaviour correlation (§7b). Those sections say what is on disk. **One
item is genuinely impossible** (§9) and one is a real gap that is not on the list
(§11, no confidence intervals on the headline metrics).

---

## 1. DAIC-WOZ exact scale and splits

Official AVEC-2017 partition, counted from `data/raw/daic_woz/labels/*.csv`:

| split | file | n | positive | rate |
|---|---|---|---|---|
| train | `train_split_Depression_AVEC2017.csv` | 107 | 30 | 0.280 |
| dev | `dev_split_Depression_AVEC2017.csv` | 35 | 12 | 0.343 |
| test | `full_test_split.csv` | 47 | 14 | 0.298 |
| | **official total** | **189** | 56 | 0.296 |

**We use 188, not 189.** Participant **440** (dev, PHQ8\_Binary = 1, PHQ8\_Score
19) ships with audio but **no transcript file** — `data/processed/daic_manifest.jsonl`
line 131 records `"transcript_path": null`. This is a gap in the distributed
corpus, not a filtering decision of ours. Our working set is therefore:

```
data/processed/daic_conditions.jsonl : 188 participants
  train 107 / dev 34 / test 47        133 negative / 55 positive (29.3 %)
  four text fields per participant:
    text_full, text_participant_only, text_ellie_only, text_ellie_masked
```

**Label**: PHQ-8, binarised at the corpus's own threshold (`PHQ8_Binary`, ≥10).
The continuous `PHQ8_Score` and the eight item scores are also present, which is
what makes DAIC richer than the other corpora we audited.

**Sentence for the paper**: "We use the 188 of 189 AVEC-2017 participants for
which a transcript is distributed (participant 440 has audio only); splits are
the official 107/34/47."

---

## 2. Model list

All checkpoints are local under `checkpoints/`. Architecture read from each
`config.json`.

### Zero-shot / instruction LLMs (8, all evaluated on the full suite)

| tag in results | checkpoint | arch | layers × hidden |
|---|---|---|---|
| `mentallama_suite` | `MentaLLaMA-chat-7B` | llama | 32 × 4096 |
| `menta` | `Qwen3-4B-base` **+ LoRA** `Menta-LoRA/Menta` | qwen3 | 36 × 2560 |
| `qwen3_base` | `Qwen3-4B-base` (no adapter) | qwen3 | 36 × 2560 |
| `osim8b` | `OSim-8B` | qwen3 | 36 × 4096 |
| `omnisapiens_sft` | `OmniSapiens-SFT` | qwen2\_5\_omni\_thinker | — |
| `omnisapiens_rl` | `OmniSapiens-RL` | qwen2\_5\_omni\_thinker | — |
| `omnisapiens_20` | `OmniSapiens2.0` | qwen2\_5\_omni\_thinker | — |
| `humanomniv2` | `HumanOmniV2` | qwen2\_5\_omni\_thinker | — |

`menta` and `qwen3_base` share a backbone exactly (verified against the LoRA's
`base_model_name_or_path`), so that pair isolates clinical fine-tuning as the
only variable. **This is the controlled contrast the paper should lead with.**

### Fine-tuned encoders (2)

`MentalBERT` (bert, 12 × 768) and `MentalRoBERTa` (roberta, 12 × 768), fine-tuned
on DAIC, 3 seeds each. MentalBERT was run with four readouts — `mean`, `cls`,
`attn`, and the D1 variant — in `experiments/b0-encoder-mechanism/results/`.

### The two baselines in the mitigation table

Both come from `experiments/i1-llm-inference-interventions/results/i1_summary.json`.
**They are leave-one-out over all 188 participants (`family: "loo"`), not the
47-participant test split** — this must be stated, because the same methods
evaluated on test-only give different numbers (entry 0: AUC 0.746).

| | entry | n | AUC | 95 % CI | acc | P(Yes) |
|---|---|---|---|---|---|---|
| **zero-shot baseline** | `[20] P0 no intervention` | 188 | **0.6629** | [0.569, 0.749] | 0.739 | 0.128 |
| D1 delete interviewer | `[21]` | 188 | **0.7875** | [0.719, 0.854] | 0.745 | 0.176 |
| D1m mask words | `[22]` | 188 | 0.7629 | | | |
| shortcut-only reference | `[23]` | 188 | 0.5459 | | | |
| P0 on E2b conflict | `[25]` | 188 | **0.3292** | | | |
| P3-c on E2b | `[26]` | 188 | 0.7769 | | | |

The **+0.126** gain quoted in the paper is 0.7875 − 0.6629. ✓

The encoder baseline **0.722** is MentalBERT: `b0_mech_summary_attn.json`
`measurements[0]/auc_mean = 0.7208`; the plain-mean readout gives 0.7215
(`b0_mech_summary.json`). Its dependence is `cf_sens_mean = 0.0409`. Note the
encoder AUCs are on the 47-participant test split across 3 seeds and are
therefore much noisier than the LLM LOO numbers — the two baselines are **not on
the same evaluation protocol** and the paper should say so rather than tabulate
them as if they were.

---

## 3. Form 1 — the systematic sweep already exists

The request assumed only `why_seek_help` was found by hand. **The full sweep with
multiple-comparison correction was run and is on disk**:
`experiments/a2-ellie-counterfactual/results/a2a_prompt_label_association.json`,
code `code/a2a_prompt_label_association.py`.

**Method.** Parse `Ellie: <prompt_id> (<text>)` from every transcript. 123 of the
188 transcripts carry scripted prompt IDs. Test only IDs asked to between 20 and
103 participants (84 % cap) — an ID asked to everyone or almost nobody has no
contrast. That leaves **75 prompt IDs tested**. Per ID: 2×2 table of depression
rate among asked vs not-asked, **Fisher exact**, then **Benjamini–Hochberg**
across the 75.

**Results.** Base rate 0.2276 over the 123 ID-carrying transcripts.

| threshold | count |
|---|---|
| q < 0.05 (BH) | **4 of 75** |
| q < 0.10 (BH) | 7 of 75 |
| raw p < 0.05 | 14 of 75 (3.8 expected by chance) |

The four surviving BH correction:

| prompt id | n asked | dep\|asked | dep\|not | OR | p | q |
|---|---|---|---|---|---|---|
| `why_seek_help` | 34 | 0.559 | 0.101 | 11.26 | 3.3e-07 | 2.5e-05 |
| `therapy_going` | 35 | 0.543 | 0.102 | 10.42 | 6.6e-07 | 2.5e-05 |
| `when_diagnosed` | 32 | 0.531 | 0.121 | 8.24 | 7.8e-06 | 2.0e-04 |
| `trigger` | 24 | 0.583 | 0.141 | 8.50 | 2.2e-05 | 4.1e-04 |

All four **presuppose their own answer**: "what got you to seek help" can only be
asked of someone who sought help; "how long ago were you diagnosed" presupposes a
diagnosis. That is the mechanism, and it is visible in the question text, not
just in the statistics.

Associations run in **both directions** — `family_relationship` (OR 0.25),
`BF_describe` (0.28), `dream_job` (0.38), `travel_changed` (0.36) are asked
*more* of non-depressed participants. So it is a branch structure, not just extra
clinical probing.

**The joint claim.** A logistic regression on the **binary asked/not-asked matrix
over the 75 IDs — no text, no word counts, nothing but which prompts were
selected** — reaches

```
CV AUC = 0.753  (sd 0.025 over 10 StratifiedKFold seeds; shuffled labels 0.500)
```
(`a2a_question_selection_auc.json`, n = 123, 75 features.)

**Controls run 2026-09-01, not yet in any results file — re-run before citing:**

| control | result |
|---|---|
| interview length (# distinct prompts) alone | AUC 0.594; dep 36.2 vs non-dep 33.6 prompts, t-test p = 0.049 |
| after regressing length out of every column | AUC **0.747** (vs 0.753) — length is not the source |
| within short-half of length | AUC 0.788 |
| within long-half of length | AUC 0.815 |
| shuffled labels | 0.448 |
| drop the single strongest prompt (`why_seek_help`) | 0.743 |
| drop the 5 strongest prompts | 0.653 |

Within-stratum AUC is *higher* than pooled, so length dilutes rather than
produces the effect; and dropping the top prompt barely moves it, so the leakage
is distributed across the branch rather than carried by one item.

---

## 4. Quantified negative/positive evidence on the other corpora

All model-free, same probe family as A2a. Code: `experiments/h1-hba-shortcut/code/audit.py`,
`experiments/h2-imhi-shortcut/code/audit.py`; ESConv numbers are from a
2026-09-01 session and **are not yet in a committed script** — they must be
written into `experiments/h3-esconv-shortcut/code/` before citing.

The common ruler is the **stopword-only probe**: delete every content word, keep
only function words, re-train. Above chance means the classes differ in register
or production source rather than in the construct. Reported as
`register % = (STOP − chance)/(TEXT − chance)`.

### The claim the paper can make

> The DAIC shortcut requires an interviewer whose **wording is fixed** so that
> **selection** is the only degree of freedom. We tested 32 tasks across three
> further corpora; none reproduces it, and the one dataset that shares the
> structural precondition (ESConv, two-party, annotated helper moves) is at
> chance.

### ESConv — the decisive negative control

`thu-coai/esconv`, 1,300 dialogues (910/195/195), seeker + supporter turns, 8
annotated supporter strategies. **This is the direct analogue of A2a**: 8
strategy labels play the role of Ellie's 75 prompt IDs.

| probe | problem\_type (5 cls, chance 0.200) | emotion\_type (7 cls, chance 0.143) |
|---|---|---|
| seeker text (the construct) | 0.780 | 0.345 |
| supporter text | 0.697 | 0.277 |
| **strategy counts + proportions, no text** | **0.220** | **0.150** |
| **strategy sequence n-grams, no text** | **0.203** | **0.149** |
| dialogue length | 0.212 | 0.134 |
| seeker survey ratings | 0.228 | 0.180 |
| stopwords only | 0.417 (37 %) | 0.227 (40 %) |

**Protocol leakage = chance.** Compare DAIC's 0.753 vs shuffled 0.500. The
supporter's *words* carry information (0.697) but that is restatement of the
seeker, not an independent channel, so no (C, S) counterfactual is constructible.

One genuine ESConv defect, worth one sentence: `problem_type` is better predicted
by the 17-word pre-dialogue `situation` field (**0.812**) than by the 203-word
seeker dialogue (**0.777**). Content-word overlap between the two is only 0.55
and the first seeker turn alone gives 0.351, so this is not copying — the label
belongs to the `situation` annotation, not to the conversation.

### HBA — 16 subsets, two with severe shortcuts

`ptsd_in_the_wild`, full corpus n = 628 (500/64/64), balanced 313/315:

| probe | balanced acc |
|---|---|
| full transcript | **0.979** |
| **stopwords only, no content words** | **0.928** |
| (at n = 64 test only) full / stopword | 0.948 / 0.880 |
| after scrubbing all PTSD/military vocabulary (n=64) | 0.953 — unchanged |

Label word in the input: `ptsd` appears in 205/313 positives and **2/315**
negatives; `trauma` 128 vs 6; `veteran` 143 vs 2. The negative class is lifestyle
vlog content; the prompt asserts *"a patient undergoing clinical evaluation"*,
which is false for half the data. **This subset measures corpus identification.**

`mmpsy_depression`, n = 1,275 (893/191/191), 258 positive (20.2 %):

| feature | AUC | 95 % CI |
|---|---|---|
| **number of recorded answer segments** | **0.753** | [0.724, 0.783] |
| total characters | 0.648 | [0.609, 0.688] |
| segment count, controlling for length | 0.726 | |
| length, controlling for segment count | 0.567 | |

Segment count replicates on all three splits (7 seg → 0.057–0.084 depression
rate; 8 seg → 0.296–0.415; 9 seg → 0.442–1.000). **This is a collection artifact,
not a content channel** — and content does carry real signal (within-stratum
char-n-gram AUC 0.695 at 7 segments, 0.741 at 8), so the subset is not degenerate,
just contaminated.

Calibration controls that validate the probe family: `cremad` 0.170 vs chance
0.167 and `tess` 0.102 vs 0.143 — acted-emotion corpora with a fixed
emotion-neutral carrier sentence, so text *must* be at chance, and is.

### IMHI — 9 sources, 15 tasks

| finding | number |
|---|---|
| labels pooled across sources | 23 |
| always the global majority label | 0.206 |
| the majority label **of its own source** | **0.442** (+0.236) |
| source identity from text, 9-way | 0.521 (chance 0.111) |
| **source identity from stopwords alone** | **0.373** |
| explanations opening with a label-stating template | swmh, t-sid, SAD, CAMS = **13,133 rows, 69 % of IMHI** |

Register share per task (worst first): `Irf` thwarted-belongingness 91 %, `CAMS`
89 %, `DR` 79 %, `MultiWD` social 73 %, `loneliness` 64 %. Two MultiWD subtasks
are at chance (`spiritual wellness` 0.500 exactly, n = 407).

**Caveat that must be stated**: the stopword signal decomposes, and only part of
it is a shortcut. Discourse markers are at chance everywhere (0.518–0.583);
contractions and punctuation are weak (0.480–0.602). What carries it is
(a) **length** — `loneliness` word count AUC 0.728, yes-median 123 words vs
no-median 53, and structural features (0.689) beat reading the full text (0.609);
and (b) **pronouns**, which for a depression task are a recognised clinical marker
and should not be called a shortcut. The meta-analytic pronoun effect is r = 0.13
(≈ AUC 0.57), well below the 0.709 we observe on `DR`, so the DR result is not
*only* the clinical marker — but we cannot cleanly attribute it either.

---

## 5. HBA — full name and collection

**Human Behavior Atlas (HBA), v2.** From `data/raw/HBA-test/README.md`:

- Paper: *Human Behavior Atlas: Benchmarking Unified Psychological and Social
  Behavior Understanding*, **ICLR 2026**, arXiv:2510.04899
- Repository: `https://github.com/MIT-MI/human_behavior_atlas` (MIT Multimodal
  Interaction group)
- Dataset: `https://huggingface.co/datasets/HumanBehaviorAtlas/human_behavior_atlas`
- Licence: **CC BY-NC 4.0** (constituent datasets carry their own licences)
- Splits: **74,449 train / 7,646 validation / 18,204 test**
- Construction: **16 source datasets integrated into one unified schema** with
  audio, video and pre-extracted features. Not newly collected — it is a
  re-packaging. Sources: CMU-MOSEI (sentiment + emotion), IntentQA, MELD
  (sentiment + emotion), CREMA-D, Social-IQ 2.0, CH-SIMS v2, TESS, UR-Funny,
  MMPsy-Depression, MMPsy-Anxiety, MIMEQA, MMSD, PTSD-in-the-Wild, DAIC-WOZ.
- It is the training corpus of the **OmniSapiens** model family, three of which we
  audit.

**Unification defect worth reporting.** The label-alias map
(`unified_label_map_v6_with_iemocap.json`, `meta/synonyms`) maps `joy → happy`,
`sadness → sad`. It was applied as a **blind substring replacement**, producing
168 corrupted tokens across 5 subsets in both the model-facing `problem` field and
the gold `answer` field: `enjoy → enhappy`, `enjoying → enhappying`,
`joyful → happyful`, `enjoyable → enhappyable`. For the open-ended QA subsets the
gold string itself is wrong, so a model writing correct English is scored
incorrect. Separately, `meld_emotion` carries gold `joy`/`sadness` while its
prompt offers `happy`/`sad` — **610 of 2,610 rows (23.4 %) whose correct answer is
not on the menu**.

---

## 6. IMHI citation

**Interpretable Mental Health Instruction (IMHI)**, introduced with MentaLLaMA:

> Kailai Yang, Tianlin Zhang, Ziyan Kuang, Qianqian Xie, Jimin Huang, Sophia
> Ananiadou. *MentaLLaMA: Interpretable Mental Health Analysis on Social Media
> with Large Language Models.* Proceedings of the ACM Web Conference (**WWW**)
> 2024. DOI 10.1145/3589334.3648137. arXiv:2309.13567.
> Code/data: `https://github.com/SteveKGYang/MentaLLaMA`

⚠ **The author list above is from a web search and the middle authors are not
individually verified.** Fetch the BibTeX from the ACM DOI before submission — do
not typeset it as-is.

Paper's own description: 105K samples, **10 sources, 8 tasks**. Our local slice is
the **test split only: 19,034 rows with a label, 9 source datasets** — the
discrepancy (9 vs 10) needs one sentence, and the train split is not on disk.

Per-source counts in our slice: swmh 10,865 · MultiWD 2,441 · Irf 2,113 · t-sid
959 · SAD 684 · CAMS 625 · loneliness 528 · dreaddit 414 · DR 405.

**Why this corpus matters for our argument**: it is MentaLLaMA's training data,
and MentaLLaMA is the model whose shortcut behaviour we measure in most detail.
That makes an IMHI defect a candidate *cause* rather than another correlate.

---

## 7. Probing — the causal ablation exists, and is stronger than the request assumes

`experiments/i7-causal-probing/results/phase3_leace_L8-24.json`, code
`code/phase3_leace.py`. LEACE closed-form concept erasure applied to **every layer
in the 8–24 band simultaneously**, then the model is re-run and its *predictions*
re-measured. This is exactly "erase the probe direction and look at the
prediction", i.e. the causal ablation the request lists as missing.

| condition | shortcut dep. (s\_sens) | construct (c\_sens) | SER | AUC |
|---|---|---|---|---|
| no erasure | 0.2057 | 0.1765 | 1.165 | 0.7769 |
| **erase the real direction** | **0.1580 (−23.2 %)** | 0.1745 (−1.2 %) | 0.906 | 0.7664 (−0.010) |
| erase a random direction (control) | 0.2068 (+0.5 %) | 0.1774 | 1.166 | 0.7767 |

The **−6.3 %** figure in the request is the single-layer version
(`phase3_leace_layer13.json`: 0.2057 → 0.1928). The band version is the result to
report. The reading is clean: erasing the direction cuts shortcut dependence by a
quarter while construct sensitivity and utility are essentially untouched, and a
random direction of the same norm does nothing.

### 7b. Probing strength ↔ counterfactual dependence

Also already run, per layer, in `layerwise.json` (MentaLLaMA) and
`layerwise_multi.json` (4 models): the partial correlation between the projection
of the arm-difference onto the probe direction and the per-participant behavioural
shift Δp, controlling for displacement norm, with a **random-direction control**
at every layer.

| model | peak \|r\| | layers with CI excluding 0 | layers where a random direction wins | mean \|r\| real | mean \|r\| random | verdict |
|---|---|---|---|---|---|---|
| mentallama | +0.264 | 15/32 | **5/32** | 0.150 | 0.067 | **coupled** |
| osim8b | +0.309 | 19/36 | 15/36 | 0.135 | 0.101 | artifact — random wins ~half the time |
| menta | −0.132 | 1/36 | 18/36 | 0.059 | 0.071 | no coupling |
| qwen3\_base | +0.157 | 1/36 | 21/36 | 0.061 | 0.082 | no coupling |

**Only MentaLLaMA clears its own random-direction control.** The significant-layer
count alone over-claims for osim8b; reporting it without the random-direction
column would be the error this paper is about. 1/36 is exactly the α = 0.05 false
positive rate.

Q1 (is the signal *there*) is separate and is positive for all four: cross-fitted
Cohen's d peaks at 2.93 / 3.33 / 4.01 / 3.25 for mentallama / menta / qwen3\_base /
osim8b, with the shuffled-label control task at −0.01 to 0.24, and d > 1 at 92–97 %
of layers. **The unadapted base carries it at least as strongly as the clinically
fine-tuned model** (4.01 vs 3.33), so fine-tuning does not create the
representation. Depth is reported as layer/(n\_layers−1) because MentaLLaMA has 33
extraction points and the Qwen3 models 37.

Hidden states exist for 4 models only (`hidden_{menta,qwen3_base,osim8b}.npz` +
`daic_hidden_participant_alllayers.npz`); the four Omni models are not probed.

---

## 8. Metric definitions — to be typeset in the paper

Let `C` be the participant's turns and `S` the interviewer's turns. All scoring is
**teacher-forced label log-likelihood** over the label strings — the free
generation is never parsed. Write `p(Y | C, S) = σ(log P("Yes") − log P("No"))`.

**Shortcut dependence (CSS / CF-Sens).** Participant evidence held byte-identical,
interviewer context swapped between the two pools:

```
CSS = (1/N) Σ_i | p(Y | C_i, S_depression) − p(Y | C_i, S_neutral) |
```
N = 188. Lower is better. Also reported: **flip rate**, the fraction of
participants whose argmax label changes between the two arms.

**Construct sensitivity.** The exact mirror — interviewer context held
byte-identical, participant swapped:

```
construct-sens = (1/M) Σ_j | p(Y | C_pos(j), S_j) − p(Y | C_neg(j), S_j) |
```
M = 200 matched pairs. **Higher** is better.

**Shortcut-to-Evidence Ratio.**

```
SER = CSS / construct-sens
```
Scale-free, so it is immune to a mitigation that flattens both terms
proportionally — which is the failure mode we observe four times. **Its own
limitation must be stated**: it does not catch *total* degradation. When both
terms approach zero the ratio is 0/0 and improves meaninglessly. The GRPO arm
reached CSS 0.0000 with no defined SER while answering "not depressed" to all 188
patients. **SER is necessary but not sufficient and must always be reported
beside an absolute utility number.**

**Reference values** (MentaLLaMA, A2b truncation budget, original template):
CSS 0.2057, construct-sens 0.1154, SER 1.78, IID AUC 0.777.

⚠ **SER has two scales in our own results and this must be reconciled before
submission.** The E5-truncated evaluation gives construct-sens 0.1154 → SER 1.78;
the untruncated evaluation gives 0.1765–0.1803 → SER 1.14–1.17. Both appear in
different result files. Pick one and re-derive every SER in the paper from it.

---

## 9. Protocol-invariance test — why it cannot be run, stated precisely

This is the one request item that is **not achievable with public data**, and the
Limitations section should say why in these terms rather than as a generic
caveat.

**Within DAIC it is not constructible.** A protocol shift and a label shift are
the same intervention here: the only "other protocol" available inside DAIC is a
different set of Ellie prompts, and those prompts are the very thing whose
label-association we are measuring. Holding the construct fixed while changing the
protocol is not defined on this corpus.

**E-DAIC does not solve it, despite appearances.** The extended corpus is a
natural experiment on exactly this axis — 275 sessions of which the new ones are
conducted by a **fully autonomous** agent rather than a human wizard, with the
same PHQ-8 instrument and population, and it is **directly downloadable, no
application** (`https://dcapswoz.ict.usc.edu/wwwedaic/`, `data/{ID}_P.tar.gz`).
We downloaded and checked it. **It is unusable for our method**: the AVEC-2019
distribution replaced the human transcripts with ASR of a single channel and
dropped the speaker column. Verified on participant 300 —

```
DAIC-WOZ  300_TRANSCRIPT.csv : 87 Ellie turns + 87 Participant turns, speaker column
E-DAIC    300_Transcript.csv : 77 rows, columns Start_Time/End_Time/Text/Confidence

Ellie turns (≥3 words) recoverable from the E-DAIC transcript :  0 / 63
Participant turns recoverable                                 : 13 / 47
```

Our entire (C, S) decomposition requires separating the two speakers. E-DAIC
deletes that. Public metadata covers 219 of the 275 sessions (163 train + 56 dev);
it does carry PCL-C PTSD labels in addition to PHQ-8, which is of no use to us
without the speaker split.

**The structure does exist in real clinical corpora, and they are closed.**
[Tu et al., SIGDIAL 2024](https://aclanthology.org/2024.sigdial-1.55/) —
411 clinician-administered interviews (LBI + THH + Criterion A + **CAPS-5** +
6 MINI modules), 515 hours, 71,412 turns, speaker-separated — state the mechanism
as a design property: *"Core questions are mandatory; optional ones depend on the
answers to the core questions, so are often skipped during the interview."* That
is our A2a mechanism, in a standardised instrument, written down by the
instrument's users. The authors also state: *"Due to strict Institutional Review
Board (IRB) regulations concerning the confidentiality of real patient
information, we are unable to release the dataset, even in an anonymized
format."*

**This is the paragraph to write**: the leakage is a structural property of
scripted, conditionally-branching clinical interviews; the one public corpus with
that structure is DAIC-WOZ; corpora with the same structure and real clinical
labels are not releasable. That is why the phenomenon has gone unstudied, and it
is an argument for the paper rather than an apology.

---

## 10. Exact audit scale

Do not write "case study". The verified counts:

**Models × DAIC-WOZ full suite: 8 LLMs.** All eight have every detection metric
(`full_auc`, `removal_ellie_only_auc`, `counterfactual_css`,
`counterfactual_flips`, `construct_sens`, `ser`, `conflict_auc`, `p_yes`) and six
interventions (`delete_interviewer`, `blank_words`, `marginalise_2`,
`marginalise_4`, `instruct_ignore`, plus `none`). Files:
`experiments/i10-model-sweep/results/suite_*.json`.

Two arms are missing for two models: `targeted_removal` and `threshold_proxy`
were added after `menta` and `qwen3_base` ran, so those two cells are empty (6 of
8 models have all eight arms).

**Plus 2 fine-tuned encoders** (MentalBERT ×4 readouts, MentalRoBERTa), 3 seeds
each → **10 models total** on DAIC.

**Per-participant counterfactual scores: 4 models** (`a2b_*.jsonl`, 752 rows each
= 188 participants × 4 arms). **Probing: the same 4 models.**

**Datasets audited model-free: 4 corpora / 32 closed-label tasks.**

| corpus | tasks probed | n |
|---|---|---|
| DAIC-WOZ | 1 | 188 |
| HBA-test | 13 of 16 subsets (3 are open-ended QA) | 18,204 rows |
| IMHI | 15 (source × question) | 19,034 rows |
| ESConv | 3 label fields | 1,300 dialogues |

**Evaluation set sizes** (§ numbers from `experiments/suite/`):

| set | unit | n |
|---|---|---|
| E1 IID | participants | 47 test / 34 dev |
| E2a natural conflict | participants | **29** |
| E2b constructed conflict | participants | 188 (133 + 55) |
| **E3 counterfactual** | **pairs** | **188** |
| **E5 construct** | **pairs** | **200** (143 distinct participants, 128 contexts) |

---

## 11. Two gaps that are not on the request list but should be

**(a) No confidence intervals on the headline metrics.** `run_full_suite.py` has
exactly one bootstrap (`boot_auc`, line 59) and uses it in exactly one place
(`removal_ci`, line 168). `construct_sens`, `css` and `ser` are bare means and a
bare ratio (line 161). **Every CSS/SER number in the eight-model table is a point
estimate with no uncertainty.** A reviewer asking whether SER 0.62 differs from
0.70 cannot be answered from what is on disk.

The precision is actually good and should be shown — computed 2026-09-01 from the
four `a2b_*.jsonl` files by participant bootstrap, 4000 resamples:

| model | n | CSS | 95 % CI | SE |
|---|---|---|---|---|
| MentaLLaMA | 188 | 0.2057 | [0.1945, 0.2174] | 0.0058 |
| Menta | 188 | 0.0266 | [0.0184, 0.0362] | 0.0046 |
| osim8b | 188 | 0.1949 | [0.1627, 0.2284] | 0.0168 |
| qwen3\_base | 188 | 0.2537 | [0.1977, 0.3133] | 0.0290 |

Paired model comparisons on the same participants resolve ΔCSS ≈ 0.05:
Menta vs MentaLLaMA −0.179 [−0.192, −0.166] ✓; osim8b vs qwen3\_base −0.059
[−0.113, −0.005] ✓; MentaLLaMA vs osim8b +0.011 [−0.020, +0.041] ✗. Within-
participant SD of |Δp| is 0.063, so n = 188 detects ΔCSS = 0.03 at 80 % power and
n ≈ 308 would be needed for 0.01. **The paired design, not the sample size, is
what makes n = 188 adequate — say so explicitly, because "188 participants" reads
as small.**

⚠ When adding CIs to **E5**, resample **by participant, not by pair**: the 200
pairs use only 143 distinct participants (400 slots), so a pair-level bootstrap
will be anti-conservative. E3's 188 are independent and can be resampled directly.

**(b) Prior work exists on the DAIC interviewer leakage and must be cited.**
[*DAIC-WOZ: On the Validity of Using the Therapist's Prompts in Automatic
Depression Detection from Clinical Interviews*](https://arxiv.org/abs/2404.14463)
(Idiap, arXiv:2404.14463, April 2024; code
`https://github.com/idiap/bias_in_daic-woz`) reports Ellie-only models beating
participant-only models: E-longBERT F1 0.84 and E-GCN F1 0.88 versus P-longBERT
0.72 and P-GCN 0.85. **We are not the first to find interviewer-side leakage in
DAIC and the contribution statement must not claim we are.**

What is ours, checked against their paper: they ablate (delete a source entirely)
and analyse keyword distributions qualitatively; they run **no counterfactual**,
no LLMs (longBERT and GCN only), no mechanism analysis, and **propose no
mitigation**. We add the counterfactual with C held byte-identical, the
model-free dataset-level measurement, 8 LLMs plus 2 encoders, the threshold-shift
mechanism with probing and LEACE, and a mitigation taxonomy with the SER metric.
The honest framing is that they establish the channel is *usable* and we establish
that models *use* it, in what form, and what fixes it.

---

## Provenance index

| § | file |
|---|---|
| 1 | `data/raw/daic_woz/labels/*.csv`, `data/processed/daic_{manifest,conditions}.jsonl` |
| 2 | `checkpoints/*/config.json`, `experiments/i1-llm-inference-interventions/results/i1_summary.json`, `experiments/b0-encoder-mechanism/results/*.json` |
| 3 | `experiments/a2-ellie-counterfactual/results/a2a_prompt_label_association.json`, `a2a_question_selection_auc.json` |
| 4 | `experiments/h1-hba-shortcut/results/{audit,genre_leak,text_only_probe}.json`, `experiments/h2-imhi-shortcut/results/{audit,per_question}.json`, `experiments/h3-esconv-shortcut/data/esconv.json` |
| 5 | `data/raw/HBA-test/README.md`, `unified_label_map_v6_with_iemocap.json` |
| 7 | `experiments/i7-causal-probing/results/{phase3_leace_L8-24,phase3_leace_layer13,layerwise,layerwise_multi}.json` |
| 8 | `experiments/i9-format-instruction/code/run_format_ser.py`, `experiments/i8-grpo-mitigation/analysis.md` |
| 10 | `experiments/i10-model-sweep/results/suite_*.json`, `experiments/suite/results/` |

## Still to do before submission

1. Re-derive every SER from one truncation convention (§8 warning).
2. Add participant-clustered bootstrap CIs to CSS / construct-sens / SER (§11a) —
   check first whether the suite stored per-participant scores, or this needs GPU
   re-runs.
3. Commit the ESConv and A2a-control scripts so §3 controls and §4 ESConv numbers
   are reproducible (they are currently session-only).
4. Fetch the IMHI BibTeX from the ACM DOI; verify the author list (§6).
5. Fill `targeted_removal` / `threshold_proxy` for `menta` and `qwen3_base` (§10).
6. Read the Idiap paper in full and write the related-work paragraph (§11b).
