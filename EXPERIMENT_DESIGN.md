# Experiment Design — Do behavioral foundation models learn psychological constructs, or shortcuts to them?

Formalization of experiments **A–D** as `input → intervention → metric → hypothesis`.
Design follows the framing set out by the PI; every substrate claim below was
verified against the deployed artefacts (see `DEPLOYMENT_NOTES.md`).

Supersedes `SHORTCUT_STUDY_PLAN.md`.

---

## 0. Unified formalism

For an input decomposable into intended evidence **C** and dataset-specific
shortcut **S**, a model `f` yields a distribution over a closed label space
`L`:

```
p(y | x)  =  softmax_{y ∈ L}  mean-token logprob( y | prompt(x) )
```

All models are scored by **label log-likelihood over `L`**, never by parsing
free generation. Rationale: parse-failure rate correlates with perturbation
strength, which manufactures fake accuracy drops; and calibrated probabilities
are required for CF-Sensitivity and conflict-following rates.

Report a **PMI-normalized** variant alongside the raw one:
`log p(y|x) − log p(y|∅)`. The gap between them *is* the label-prior effect.

### Evidence chain

```
shortcut-only  →  counterfactual  →  conflict  →  cross-dataset
(S is usable)     (S is used)        (S dominates C)   (consequence)
```

Ablation only shows a shortcut **exists and is usable**. Counterfactual and
conflict show the model **actually uses** it. Keep these claims separate.

### Metrics

| Symbol | Definition | Reads as |
|---|---|---|
| `Acc_∅` | accuracy with C and S both removed (question/options only) | leakage in the benchmark itself |
| **SS** | `(Acc(x_S) − Acc_chance) / (Acc(x_full) − Acc_chance)` | Shortcut Sufficiency; →1 means C adds nothing |
| **CF-Sens** | `\|p(y\|C,S₁) − p(y\|C,S₂)\|`, C held byte-identical | shortcut is causally used |
| **R_A, R_V** | in conflict set, `P(pred = label of audio)`, `P(pred = label of video)` | per-modality following rate |
| **D_mod** | `R_A − R_V ∈ [−1,1]` | modality dominance |
| **SR** | `R_S` = P(prediction follows the shortcut channel) in conflict set | shortcut reliance |
| **ΔSR** | `SR_stage_{k+1} − SR_stage_k` | does post-training amplify or suppress reliance |

**Mandatory controls for every table below**
1. majority-class and chance baselines,
2. a `label_shuffle` run to establish the noise floor of each metric,
3. bootstrap CIs over samples (n ≥ 1000 resamples); never report point estimates,
4. `R_A + R_V + R_other = 1` — always report `R_other` (predictions following
   neither channel indicate the conflict item is simply out of distribution).

---

## A. DAIC-WOZ positive control

**Purpose:** validate that the pipeline recovers a shortcut already documented
in the literature (Ellie's interviewer prompts as a depression cue). If A
fails, nothing downstream is trustworthy.

**Substrate (verified):** 189 participants, `phq8_binary`, 50.2 h audio,
188 transcripts with `start_time / stop_time / speaker ∈ {Ellie, Participant}`.
Speaker-resolved timestamps make the four conditions constructible from the raw
files with no extra annotation.

### A1 — Ablation

| Input | Intervention | Metric | Hypothesis |
|---|---|---|---|
| Full interview (text; audio for omni models) | none — reference | Acc, F1, AUC | reference |
| Participant turns only | drop all Ellie turns | Acc, **SS_C** | C alone should retain most signal |
| **Ellie turns only** | drop all Participant turns | Acc, **SS_S** | **`SS_S` ≫ 0 replicates the known shortcut** |
| Ellie-masked | keep Ellie turn *positions* and timing, replace text with `[Q]` | Acc | isolates lexical content of prompts from interview *structure* |

`SS_S ≫ 0` with `Ellie-masked ≈ Participant-only` ⇒ the shortcut is lexical.
`Ellie-masked` still ≫ chance ⇒ the shortcut is structural (which questions got
asked / how many), a stronger and more interesting claim.

### A2 — Counterfactual (the real result)

| Input | Intervention | Metric | Hypothesis |
|---|---|---|---|
| (C = participant turns, byte-identical) + Ellie prompt S₁ | S₁ = depression-associated prompt (e.g. diagnosis/therapy history) | `p(depression)` | — |
| same C + Ellie prompt S₂ | S₂ = neutral prompt (e.g. hometown, hobbies) | `p(depression)` | — |
| | | **CF-Sens** | large CF-Sens with C untouched ⇒ prediction is driven by interviewer context, not participant evidence |

Prompt inventory: mine Ellie's turns across the corpus, cluster into
depression-associated vs neutral, and hand-verify the two pools before use.
Each participant contributes matched pairs, so the comparison is within-subject.

**Models:** MentaLLaMA, Menta (transcript path — cheap, run first);
OmniSapiens2.0, HumanOmniV2 (audio path).
**Encoder control:** MentalBERT / MentalRoBERTa on the same text conditions.

### Feasibility constraints (verified, must be handled)

- **Audio is single-channel and mixed.** Participant-only / Ellie-only audio
  must be built by concatenating segments from the transcript timestamps.
  Cross-talk at boundaries is unavoidable — report it as a limitation, and pad
  segment edges consistently across conditions so the bias is shared.
- **Whisper front-end caps at `chunk_length=300 s`; median interview is
  ~15 min.** Full-interview audio cannot be fed in one pass. Fix the protocol
  *before* running: either (i) fixed-window sampling of N×60 s segments with
  per-segment scoring then aggregation, or (ii) participant-turn concatenation
  truncated to 300 s. **The same windowing must be applied to every condition**,
  or the ablation measures windowing rather than modality.
- Text path has no such limit — run it first to establish the effect.

---

## B. Multimodal modality shortcut and conflict

**Purpose:** does the model answer without consulting audio/video, and when
channels disagree, which one wins?

**Substrate (verified in `data/raw/HBA-test`):** `audios` (WAV) and `videos`
(MP4) are **separate binary columns**, and `problem` carries independent
`<video>` / `<audio>` placeholders. Ablation and channel-swapping are therefore
pure data edits — no demuxing, no re-encoding.

Sources with both channels (test split): `meld_emotion` 2610 · `meld_senti` 2609
· `mosei_senti` 4659 · `mosei_emotion` 1662 · `chsimsv2` 1034 · `mmsd` 103 ·
`ptsd_in_the_wild` 64.
Audio-only: `cremad` 1117, `tess` 420. Text-only: `mmpsy_*` 191 each.

**Primary substrate: `meld_emotion`** — 7 categorical emotions, clips 0.7–7 s
(well under the 300 s cap), balanced enough to build conflict pairs.

### B1 — Ablation

| Input | Intervention | Metric | Hypothesis |
|---|---|---|---|
| `<video> <audio> <text>` | none | Acc | reference |
| audio only | drop `<video>` + video binary | Acc, D_mod prior | — |
| video only | drop `<audio>` + audio binary | Acc | — |
| text only | drop both media | Acc | transcript already carries the emotion |
| **question only** | drop media **and** the transcript line | **`Acc_∅`** | **`Acc_∅ ≫ chance` ⇒ the benchmark leaks via question/options** |

Run the identical ladder on **Daily-Omni** and **IntentBench** (both are MCQ,
so chance is well-defined: Daily-Omni is 4-way, answer distribution
B 318 / C 306 / D 295 / A 278 — near-uniform, so option-position bias is
measurable). Daily-Omni's official code already exposes `all / visual / audio`
modes, so its ablation is close to free.

### B2 — Conflict (the core result)

Construct pairs from `meld_emotion`: take sample *i* with label `y_i` and
sample *j* with `y_j ≠ y_i`; emit `(A_j, V_i, text = ?)`.

| Input | Intervention | Metric | Hypothesis |
|---|---|---|---|
| `(A_i, V_i)` | none | Acc | reference |
| `(A_j, V_i)` | swap audio binary from a different-label sample | **R_A, R_V, R_other** | — |
| `(A_i, V_j)` | swap video binary | **R_A, R_V, R_other** | symmetry check |
| both directions pooled | | **D_mod = R_A − R_V** | `\|D_mod\|` near 1 ⇒ single-modality dominance; near 0 with high `R_other` ⇒ conflict simply breaks the model |

**Text is a third channel.** The transcript line inside `problem` carries
lexical emotion cues. Decide explicitly and hold it fixed: recommended default
is to **blank the transcript in all conflict conditions**, so the contest is
strictly A vs V. A separate three-way arm (text vs A vs V) is a natural
extension but must not be conflated with the two-way result.

### ⚠ Blocking implementation detail (verified)

**The HBA MP4s carry their own AAC audio track** (`streams=[video h264, audio
aac, data]`). If the video's native audio reaches the model, a swapped-audio
conflict item contains *two competing audio sources* and `R_A`/`R_V` become
uninterpretable. Before any conflict run:

1. re-mux every video binary to video-only (`-an`), **and**
2. confirm `use_audio_in_video=False` in the Qwen2.5-Omni processor path.

Verify with a sanity item: video-only input whose stripped mp4 previously
contained clearly audible speech must score at chance on an audio-only-decidable
question.

**Models:** OmniSapiens2.0, HumanOmniV2.

---

## C. Text mental-health shortcut

**Purpose:** separate *psychologically meaningful* cues from *source-specific*
cues. Deleting "kill myself" and observing a drop is **not** evidence of
shortcut learning — that phrase is legitimate symptom evidence.

**Substrate:** `text_manifest.jsonl` — IMHI 9 test sets (19,051 rows) + SDCNL
(1,895). Deployed and indexed.

### C1 — Template / source normalization (cheapest, no rewriting needed)

| Input | Intervention | Metric | Hypothesis |
|---|---|---|---|
| native IMHI item | none | Acc per source | reference |
| template-normalized | strip source-specific instruction wording; recast all sources into one canonical prompt | ΔAcc per source | a drop concentrated in specific sources ⇒ the model was keying on template identity |
| source-identity probe | ask the model which corpus the item came from / linear probe on hidden states | probe accuracy, AUC | high AUC ⇒ source is linearly recoverable, a precondition for provenance shortcut |

This is mechanical — no text generation involved — so it runs first.

### C2 — Explicit self-diagnosis counterfactual

| Input | Intervention | Metric | Hypothesis |
|---|---|---|---|
| Version **A**: symptom description **+ explicit diagnostic label** ("I was diagnosed with depression…") | — | `p(depression)` | — |
| Version **B**: same symptoms, diagnostic label removed, construct preserved | minimal edit; C (anhedonia, low motivation) held equivalent | `p(depression)` | — |
| | | **Δ_diagnosis-word = p(A) − p(B)** | large Δ ⇒ self-diagnostic shortcut rather than symptom reasoning |

### C3 — Source-style counterfactual

| Input | Intervention | Metric | Hypothesis |
|---|---|---|---|
| C fixed, S = Reddit register | — | `p(y)` | — |
| C fixed, S = clinical-note register | style transfer, construct preserved | `p(y)` | — |
| C fixed, S = Twitter register | style transfer, construct preserved | `p(y)` | — |
| | | max-pairwise **CF-Sens** | construct-invariant input, register-driven prediction ⇒ style shortcut |

**SDCNL is the sharpest target here.** Its two classes are drawn from
`r/SuicideWatch` vs `r/depression`, so community register is confounded with the
label by construction — exactly `community language → Y` rather than
`construct → Y`. The manifest retains `url`/`author`, so subreddit provenance
can be audited even though the model only sees text.

### Dependency to resolve before C2/C3

C2 and C3 require **paired rewrites**. Options, in order of preference:

1. **Human-authored templates** for a small, controlled probe set (n ≈ 200–400).
   Highest validity, no circularity, and enough for a within-subject test.
2. **Neutral third-party rewriter** — `Qwen3-4B-base` is already deployed and is
   not a mental-health model. Must be validated: sample-check that C is
   preserved and only S changed.
3. **Never** use MentaLLaMA/Menta to rewrite items they will then be evaluated
   on — that is circular.

Whichever is chosen, publish the rewrite prompts and a manual adequacy audit;
this is the step most likely to be attacked in review.

**Models:** MentaLLaMA, Menta. **Control:** MentalBERT / MentalRoBERTa as
non-generative encoders (use `last_hidden_state[:,0]` — the pooler is randomly
initialised, see gotcha 18). They stay a baseline, not part of the main story.

---

## D. SFT → GRPO → HARPO: does reasoning RL mitigate or amplify shortcuts?

**The highest-value experiment**, because architecture and base model are held
fixed and *training stage is the only variable*.

### Verified feasibility ✅

| Stage | Checkpoint | Arch | Status |
|---|---|---|---|
| SFT | `HumanBehaviorAtlas/omnisapiens_sft` | `Qwen2_5OmniThinkerForConditionalGeneration` | 17.9 GB — **not yet downloaded** |
| GRPO RL | `ddvd233/OmniSapiens-7B-RL` | same | 17.9 GB — **not yet downloaded** |
| HARPO | `HumanBehaviorAtlas/OmniSapiens2.0` | same | ✅ deployed, smoke-tested |

All three share one architecture and load through the existing `sc-shortcut`
env with the same scoring path — no per-stage code. Cost to unblock: **35.8 GB**
(~3 min at observed throughput). Note the RL checkpoint lives under `ddvd233`,
not the `HumanBehaviorAtlas` org.

| Input | Intervention | Metric | Hypothesis |
|---|---|---|---|
| B1 ablation ladder, all three stages | identical inputs | `Acc_∅`, SS per stage | — |
| B2 conflict set, all three stages | identical conflict items | **SR_SFT, SR_GRPO, SR_HARPO** | — |
| | | **ΔSR = SR_post-RL − SR_SFT** | `ΔSR > 0`: RL raises benchmark accuracy **while amplifying** shortcut reliance. `ΔSR < 0`: reasoning RL genuinely improves multimodal grounding |

Report `ΔSR` **jointly with benchmark accuracy per stage**. The headline result
is the *decoupling*: accuracy climbing while `SR` also climbs is the finding
that matters, and it connects directly to the RLVR-shortcut literature.

Secondary, cheap and worth taking: HumanOmniV2 emits `<context>…</context>
<think>…</think>` traces. On conflict items, check whether the stated context
mentions the channel the prediction actually followed. Divergence = the
reasoning trace is post-hoc rationalization, not the causal path.

---

## Scope discipline

**Not in A–D:** OSim-8B. Reward-model / judge-targeting shortcuts (evaluator-
directed phrasing, response-length collapse) are a different failure mode with
a different metric family, and mixing them into a mental-health classification
story weakens both. Keep as a separate track.

**Not in the main story:** MentalBERT / MentalRoBERTa — domain-pretrained MLMs,
not instruction/reasoning systems. Encoder control only.

**Explicitly out of scope for v1:** shortcut-strength curves
(`P(S|Y) = 0.5, 0.6, 0.7 …`). These require training multiple controlled models;
every deployed checkpoint here is fixed. Do not claim a dose-response
relationship from observational comparisons.

---

## Execution order and cost

| Step | Blocks on | Cost |
|---|---|---|
| 0. Scoring harness + label-space scorer + label_shuffle floor | — | shared by all |
| 1. **A1 text** (DAIC ablation, MentaLLaMA/Menta) | nothing | hours |
| 2. **C1** (template normalization + source probe) | nothing | hours |
| 3. **B1** (ablation ladder, HBA + Daily-Omni + IntentBench) | video extraction from archives | ~1 day |
| 4. **A2** (Ellie counterfactual) | Ellie prompt inventory, hand-verified | ~1 day |
| 5. **B2** (conflict) | **mp4 audio-track stripping** | ~1 day |
| 6. **D** (three-stage SR) | 35.8 GB download; reuses B1+B2 exactly | ~1 day |
| 7. **C2/C3** (rewrite counterfactuals) | rewrite protocol + adequacy audit | longest lead time |

Steps 1–2 need nothing that is not already on disk. Step 7 has the longest lead
time because of the rewrite validation, so start drafting its protocol in
parallel with step 1.

---

---

# Locked decisions (PI, 2026-08-10)

These four were open; they are now fixed. Apply them uniformly — a condition that
deviates measures the deviation, not the shortcut.

### D1 — Experiment A audio: participant-turn concatenation, truncated to 300 s
Concatenate the target speaker's segments (from the transcript timestamps) and
take the first 300 s. Highest signal density per condition.

**Known cost, must be reported:** conditions cover different amounts of the
interview — `ellie_only` is a median 632 words against `participant_only`'s
1,446, so 300 s of Ellie spans a much larger slice of the session than 300 s of
participant speech. The comparison baseline is therefore not equal-coverage.
Mitigation: log actual covered wall-clock duration and word count per condition
per participant, report them alongside every accuracy number, and include
covered-duration as a covariate. Do **not** silently equalise — that would be a
different protocol.

### D2 — Experiment B2 conflict: transcript blanked, strict A vs V
Remove the transcript line in every conflict condition so `R_A + R_V + R_other`
is a clean two-way contest and `D_mod` is unambiguous. A three-way
`text vs A vs V` arm may be added later but is reported separately and never
pooled with the two-way result.

### D3 — Scale: matched probe subsets, n ≈ 300–500 per condition
Power-analyse to fix n, then use **the same sample across every condition**
(within-subject) with bootstrap CIs. CF-Sensitivity and SR are paired
comparisons, where the paired design buys far more power than raw sample size.
Keeps the full A–D ladder tractable across four checkpoints.

### D4 — C2/C3 rewriting: Anthropic API, `claude-opus-5`

Credentials live at `/data/cl195/.secrets/anthropic.env` (chmod 600, outside the
project tree so they are never published or committed). Both keys verified
working. ⚠️ They were pasted in plaintext chat — rotate when convenient.

**Model: `claude-opus-5`.** Rationale: C2/C3 is the reviewer-facing weak point of
the whole design — the claim rests on C being genuinely preserved while only S
changes, which is a judgment task, not a formatting task. It is also not
circular: Opus 5 is not a mental-health model and is not under evaluation here.

**Batch API for the bulk pass.** Corpus prep is offline, so use
`client.messages.batches.create` — 50 % cheaper, and latency is irrelevant.
Order the results by `custom_id`, never by position.

**Structured outputs**, so a rewrite either parses or fails loudly rather than
silently corrupting an item:

```python
client.messages.create(
    model="claude-opus-5", max_tokens=2048,
    output_config={"format": {"type": "json_schema", "schema": REWRITE_SCHEMA}},
    messages=[...],
)
```

**Cost is negligible and should not drive any design choice here.** ~400 items ×
4 variants ≈ 1,600 short generations ≈ **$8–17** total (Opus 5 at $5/$25 per
MTok, halved by Batch).

**Non-negotiable guards, since this is the attackable step:**
1. Publish every rewrite prompt verbatim in the paper appendix.
2. Hand-audit a random sample (≥ 50 pairs/variant) for *construct preserved,
   only S changed*; report the adequacy rate.
3. Blind the auditor to which side is A and which is B.
4. Never let MentaLLaMA or Menta author items they will then be scored on.

---

# Findings from build-out (2026-08-10) — these change the protocol

Recorded while implementing step 0. Each one silently corrupts a result if
ignored.

### F1 — HBA `meld_emotion` gold labels are not in its own option list
The prompt offers `anger, disgust, fear, happy, surprise, neutral, sad`, but
gold `class_label` uses **`joy` (402)** and **`sadness` (208)** — **610 of 2610
rows, 23 %**, whose correct answer cannot be selected. Naive scoring marks all
610 wrong and skews every ablation contrast. Fixed by an alias map in
`data/processed/hba_label_spaces.json`; `class_label` is the authority, prompt
wording is a surface form.

### F2 — Three HBA sources are open-ended, not classification
`intentqa` (2134, 1813 distinct answers), `siq2` (875 / 865), `mimeqa`
(173 / 173) store free-text gold in `class_label`. They cannot enter the
closed-label scoring path at all. **13 of 16 sources are closed-label.**

### F3 — Severe class imbalance in exactly the mental-health sources
Accuracy is near-meaningless on these; use balanced accuracy / macro-F1 and
always print the majority baseline.

| dataset | chance | majority | majority class |
|---|---|---|---|
| mmpsy_anxiety | .500 | **.937** | no_anxiety |
| mmpsy_depression | .500 | **.817** | no_depression |
| daicwoz | .500 | **.702** | false |
| mosei_emotion | .167 | .601 | happy |
| meld_emotion | .143 | .481 | neutral |
| tess | .143 | .143 | disgust |

A constant predictor scores 93.7 % on `mmpsy_anxiety`.

### F4 — Video downsampling is a confound, not a speed knob
Token cost for one 2.25 s 720p MELD clip: native **20,063** → 224p/2nd frame
**1,973** (10×). But shrinking the video channel handicaps it, and in B2 that
manufactures apparent **audio dominance** — the very quantity being measured.

**Required:** fix one preset, apply it to every condition and model, and
calibrate first — sweep presets on a held-out subset and choose the smallest
one at which *video-only* accuracy has not yet degraded relative to native.
Report the calibration curve. Presets live in `shortcut_harness/media.py`.

### F5 — DAIC transcripts leak internal prompt IDs
Ellie's turns in **123 of 188** transcripts carry the scripted prompt
identifier, e.g. `depression_diagnosed (have you been diagnosed with
depression)`, `ptsd_diagnosed (...)`, `happy_lasttime (...)`.

Two consequences, opposite in sign:

- **Good:** the A2 prompt pools do not need keyword inference — the corpus
  names each scripted prompt. `depression_diagnosed` was asked to 115
  participants, `ptsd_diagnosed` to 111. Ready-made matched pairs.
- **Bad:** the `ellie_only` condition would then contain tokens far more
  explicit than natural speech, inflating `SS_S` through an annotation artifact
  rather than the interviewer-prompt shortcut the literature reports.
  **A1 must strip the prompt IDs, and report stripped vs unstripped.**

### F6 — Transcript format is confounded with the label
Prompt-ID presence is significantly associated with diagnosis:
depression rate **22.8 % with IDs vs 41.5 % without**
(χ²=6.36, p=0.012; Fisher OR=0.41, p=0.011).

A format-only classifier reaches 0.649, which is **below** the 0.707 majority
baseline — so format is *not* a sufficient shortcut on its own, but it is a
real confound that interacts with `ellie_only`. Stratify or covary it; do not
ignore it.

---

# A1 executed (2026-08-19) — see `runs/A1_RESULTS.md`

Both text models run over all 188 participants × 7 conditions. Outcome in one
line: **the interviewer-prompt shortcut is present in the data at ceiling
strength (TF-IDF+LogReg on Ellie's turns alone: AUC 0.810, above `full` at
0.776), and neither zero-shot model uses it (AUC 0.546 / 0.576, at their noise
floors).**

Three amendments this forces on the design:

1. **The A-as-positive-control premise is wrong as written.** The documented
   DAIC shortcut belongs to models *trained on DAIC*; MentaLLaMA and Menta never
   saw it. `SS_S ≈ 0` here does not indict the pipeline — every
   participant-bearing condition clears the label-shuffle floor at AUC 0.72–0.79.
   A1 does **not** block B–D. A real pipeline control needs a model trained on
   the corpus under test.
2. **Accuracy-based SS is unusable on DAIC.** Both models emit a near-constant
   prediction (MentaLLaMA `No` 87 %, Menta `Yes` 84 %), so balanced accuracy
   pins to 0.500 whether or not information is present. Headline metrics moved
   to AUC-based SS, which is threshold-free and provably invariant to PMI
   normalization (a useful free check: the raw and PMI AUC tables must match).
3. **F5 confirmed, and it would have produced a false positive.** MentaLLaMA's
   `ellie_only` AUC drops 0.647 → 0.443 when the leaked prompt IDs are stripped
   (paired Δ = +0.204 [+0.047, +0.357]). The data-level shortcut, by contrast,
   survives stripping (0.810 → 0.808), so the strip must stay on for the model
   path and does not weaken the corpus claim.

---

## Falsifiability

The design must be able to return a null. Pre-commit to these readings:

- `SS_S ≈ 0` in **A** ⇒ pipeline fails to recover a documented shortcut ⇒ fix
  the pipeline before believing any of B–D.
- `Acc_∅ ≈ chance` and `R_other` high in **B** ⇒ no usable shortcut; report it.
- `ΔSR ≈ 0` within CI in **D** ⇒ reasoning RL is neutral on shortcut reliance —
  a publishable negative given how strongly HumanOmniV2 and HARPO both claim
  anti-shortcut grounding.
