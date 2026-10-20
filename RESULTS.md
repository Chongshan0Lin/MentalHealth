# Results — LLM shortcut learning on psych / mental-health data

Consolidated record of everything measured to date. Last updated **2026-08-19**.

Scope note: of experiments **A–D** in `EXPERIMENT_DESIGN.md`, only **A1** has been
run. Sections 2–4 below are measurements taken while building the harness — they
are properties of the *data and the substrate*, not of model behaviour, and
several of them silently corrupt results if ignored.

| Experiment | Status |
|---|---|
| **A1** — DAIC-WOZ ablation | ✅ complete, both text models (§1) |
| A2 — Ellie counterfactual | ❌ not run (prompt inventory built, needs hand-verification) |
| B1 / B2 — multimodal ablation + conflict | ❌ not run (B2 blocked on mp4 audio-track stripping) |
| C1 / C2 / C3 — text provenance + rewrites | ❌ not run |
| D — SFT → GRPO → HARPO | ❌ not run (all three checkpoints on disk) |

---

# 1. Experiment A1 — DAIC-WOZ ablation

Detail and caveats: `runs/A1_RESULTS.md`.
Code: `scripts/run_a1.py`, `scripts/analyze_a1.py`, `scripts/a1_lexical_ceiling.py`.
Data: `runs/a1_{MentaLLaMA-chat-7B,Menta}.jsonl` — 188 participants × 7 conditions
× 2 labels = 1,316 rows each. Logs: `logs/run_a1_*.log`, `logs/analyze_a1_*.log`,
`logs/a1_lexical_ceiling.log`.

**Headline: the interviewer-prompt shortcut is present in the data at ceiling
strength, and neither foundation model uses it.**

## 1.1 Setup

Both models scored by teacher-forced label log-likelihood over `{Yes, No}` —
never by parsing free generation — using MentaLLaMA's native IMHI template held
byte-identical across conditions. Transcripts truncated head-first to a
**4,032-token budget**, set by MentaLLaMA's 4,096 context and applied to Menta
too so truncation is not a cross-model confound. Coverage logged per row (D1:
reported, never silently equalised).

`Menta` = `Qwen3-4B-Instruct-2507` + released LoRA (r=8).

Label distribution: n=188, 55 positive, **majority-class accuracy 0.707**,
chance balanced accuracy 0.500.

## 1.2 Main table — AUC

AUC rather than accuracy, because both models collapse to a near-constant
prediction (§1.5). `*` = bootstrap lower bound clears the label-shuffle
95th-percentile floor (floors sit at 0.572–0.583 for the model columns).
Probe column is 5-fold × 10-repeat stratified CV; its shuffled control sits at
0.485–0.570.

| condition | TF-IDF+LogReg *(fitted on DAIC)* | MentaLLaMA *(zero-shot)* | Menta *(zero-shot)* |
|---|---|---|---|
| full | 0.776 | 0.663 | 0.778\* |
| full_stripped | 0.794 | 0.715\* | 0.778\* |
| participant_only | 0.756 | 0.787\* | 0.727\* |
| **ellie_only** | **0.810** | 0.546 | 0.576 |
| ellie_only_stripped | 0.808 | 0.513 | 0.569 |
| ellie_masked | 0.757 | 0.763\* | 0.720\* |
| empty (`Acc_∅`) | — | 0.500 | 0.500 |

Model AUCs with CIs:

| condition | MentaLLaMA AUC [95% CI] | floor | Menta AUC [95% CI] | floor |
|---|---|---|---|---|
| full | 0.663 [0.572, 0.749] | 0.581 | 0.778\* [0.702, 0.847] | 0.583 |
| full_stripped | 0.715\* [0.631, 0.795] | 0.576 | 0.778\* [0.708, 0.849] | 0.577 |
| participant_only | 0.787\* [0.715, 0.853] | 0.578 | 0.727\* [0.650, 0.806] | 0.573 |
| **ellie_only** | 0.546 [0.459, 0.635] | 0.572 | 0.576 [0.486, 0.665] | 0.576 |
| ellie_only_stripped | 0.513 [0.415, 0.607] | 0.575 | 0.569 [0.479, 0.660] | 0.573 |
| ellie_masked | 0.763\* [0.684, 0.836] | 0.574 | 0.720\* [0.633, 0.798] | 0.575 |

## 1.3 Findings

**(1) The documented shortcut is real, and it is the strongest signal in the
corpus.** A bag-of-words model given *only the interviewer's questions* — not one
word from the participant — reaches **AUC 0.810**, above the same model given the
participant's own words (0.756) or the entire interview (0.776). Which questions
Ellie asked is more diagnostic of depression than what the participant answered.

**(2) It is not the annotation artifact.** Stripping the scripted prompt IDs
leaves the probe at 0.808. The shortcut is equally strong in the 65 transcripts
that never carried IDs (0.846) as in the 123 that did (0.844 → 0.824 stripped).

**(3) Neither foundation model exploits it.** `ellie_only` sits at or below each
model's own noise floor. `SS_S ≈ 0`. Meanwhile every participant-bearing
condition clears the floor at 0.72–0.79 — this is not a pipeline that cannot
detect signal.

**(4) F5 confirmed, and it would have produced a false positive.** MentaLLaMA's
`ellie_only` AUC on the 123 ID-carrying transcripts is 0.647 unstripped and
0.443 stripped:

| model | condition | with IDs | stripped | paired Δ [95% CI] | verdict |
|---|---|---|---|---|---|
| MentaLLaMA | ellie_only | 0.647 | 0.443 | **+0.204 [+0.047, +0.357]** | artifact-driven |
| MentaLLaMA | full | 0.699 | 0.750 | −0.051 [−0.179, +0.070] | n.s. |
| Menta | ellie_only | 0.692 | 0.605 | +0.087 [−0.048, +0.228] | n.s. |
| Menta | full | 0.802 | 0.791 | +0.011 [−0.025, +0.050] | n.s. |

Without the strip we would have reported a spurious replication of the
documented shortcut, driven entirely by leaked annotation tokens.

**(5) No benchmark leakage.** With no transcript at all both models emit a
constant score, and their content-free prior is `Yes` — the *minority* class.
`Acc_∅` = 0.293 against a 0.707 majority baseline.

## 1.4 Threshold metrics (for the record)

Raw label log-likelihood, MentaLLaMA / Menta:

| condition | MentaLLaMA bal_acc [CI] | P(pred=Yes) | Menta bal_acc [CI] | P(pred=Yes) |
|---|---|---|---|---|
| full | 0.603 [0.539, 0.667] | 0.128 | 0.587 [0.543, 0.629] | 0.840 |
| participant_only | 0.683\* [0.606, 0.755] | 0.340 | 0.495 [0.473, 0.508] | 0.989 |
| ellie_only | 0.489 [0.474, 0.500] | 0.016 | 0.500 [0.500, 0.500] | 1.000 |
| ellie_masked | 0.691\* [0.615, 0.764] | 0.330 | 0.500 [0.500, 0.500] | 1.000 |
| empty | 0.500 | 1.000 | 0.500 | 1.000 |

## 1.5 Two methodological consequences

**Accuracy-based SS is unusable on DAIC.** MentaLLaMA answers `No` to 87 % of
full interviews; Menta answers `Yes` to 84 %, and to 100 % under every ablation.
A constant classifier scores exactly 0.500 balanced accuracy — indistinguishable
from genuine chance. Headline metrics were moved to **AUC-based SS**.

**A free correctness check on the scorer.** PMI subtracts a per-label constant,
i.e. a monotone shift of the score, so raw and PMI AUC tables *must* be
identical. They are. Balanced accuracy, being threshold-dependent, is not:

| model | `full` bal_acc raw → PMI | Δ |
|---|---|---|
| MentaLLaMA | 0.603 → 0.514 | **−0.088** |
| Menta | 0.587 → 0.713 | **+0.126** |

A large label-prior effect in both models, in opposite signs.

## 1.6 What this does to the falsifiability clause

The design pre-committed: *`SS_S ≈ 0` in A ⇒ pipeline fails to recover a
documented shortcut ⇒ fix the pipeline before believing any of B–D.*

**That clause conflates two distinct failures, and the linear probe separates
them.** The pipeline recovers signal wherever signal exists; the probe proves the
shortcut is there to be found at 0.810. What A1 measured is that two zero-shot
models do not use an available shortcut.

This is expected once stated plainly: **the documented DAIC shortcut is a
property of models trained on DAIC.** MentaLLaMA and Menta never saw DAIC
training data. A1 as specified is therefore *not a valid positive control for the
pipeline*.

**A1 does not block B–D.** Two amendments follow:

- A genuine pipeline control needs a model trained on the corpus under test.
  `OmniSapiens2.0` is the candidate (HBA's `daicwoz` is in its training lineage,
  §4 gotcha 17) — but the 47-row overlap is the DAIC *test* split, so train-side
  membership must be confirmed first.
- "Shortcut is available" and "shortcut is used" must stay separated in the
  write-up. A1 is a clean instance of the first without the second.

---

# 2. Video preset calibration (F4)

`OmniSapiens2.0`, `meld_emotion`, `video_only`, n=40 native-feasible clips.
Log: `logs/calib_video.log` · `data/processed/video_preset_calibration.json`.

Downsampling video is **not a speed knob** — shrinking the video channel
handicaps it, and in B2 that manufactures apparent *audio dominance*, the very
quantity being measured.

| preset | acc | bal_acc | Δbal_acc vs native | agree w/ native | tokens | s/item | speedup |
|---|---|---|---|---|---|---|---|
| native | 0.700 | 0.638 | — | 1.000 | 20,063 | 53.78 | 1.0× |
| half | 0.675 | 0.629 | −0.009 | 0.900 | 10,443 | 26.58 | 2.0× |
| **336p** | 0.725 | **0.648** | **+0.009** | 0.900 | 3,611 | 8.58 | **6.3×** |
| 224p | 0.700 | 0.548 | −0.091 | 0.875 | 1,973 | 5.46 | 9.9× |
| 224p_sparse | 0.675 | 0.613 | −0.025 | 0.800 | 1,028 | 3.18 | 16.9× |

**336p is the indicated choice** — video-only balanced accuracy has not degraded
(+0.009) at 6.3× speedup, while 224p costs −0.091 and would bias B2 toward a
false audio-dominance result. ⚠️ **This decision is not yet recorded anywhere in
the codebase**; `shortcut_harness/media.py` still exposes all five presets with
no locked default.

---

# 3. Dataset defects found during build-out (F1–F6)

Each of these silently corrupts a result if ignored.

**F1 — HBA `meld_emotion` gold labels are not in its own option list.** The prompt
offers `anger, disgust, fear, happy, surprise, neutral, sad`; gold `class_label`
uses **`joy` (402)** and **`sadness` (208)** — **610 of 2,610 rows, 23 %**, whose
correct answer cannot be selected. Naive scoring marks all 610 wrong and skews
every ablation contrast. Fixed by an alias map in
`data/processed/hba_label_spaces.json`.

**F2 — three HBA sources are open-ended, not classification.** `intentqa`
(2,134 rows / 1,813 distinct answers), `siq2` (875 / 865), `mimeqa` (173 / 173)
store free-text gold and cannot enter closed-label scoring. **13 of 16 sources
are closed-label.**

**F3 — severe class imbalance in exactly the mental-health sources.** A constant
predictor scores **93.7 %** on `mmpsy_anxiety`.

| dataset | chance | majority | majority class |
|---|---|---|---|
| mmpsy_anxiety | .500 | **.937** | no_anxiety |
| mmpsy_depression | .500 | **.817** | no_depression |
| daicwoz | .500 | **.702** | false |
| mosei_emotion | .167 | .601 | happy |
| meld_emotion | .143 | .481 | neutral |
| tess | .143 | .143 | disgust |

Confirmed to bite in practice — see §1.5.

**F4 — video downsampling is a confound.** See §2.

**F5 — DAIC transcripts leak internal prompt IDs.** Ellie's turns in **123 of 188**
transcripts carry the scripted identifier, e.g. `depression_diagnosed (have you
been diagnosed with depression)`. Double-edged: the A2 prompt pools need no
keyword inference (`depression_diagnosed` asked to 115 participants,
`ptsd_diagnosed` to 111), but `ellie_only` would contain tokens far more explicit
than natural speech. **A1 confirmed the bad horn empirically** — see §1.3(4).

**F6 — transcript format is confounded with the label.** Prompt-ID presence is
significantly associated with diagnosis: depression rate **22.8 % with IDs vs
41.5 % without** (χ²=6.36, p=0.012; Fisher OR=0.41, p=0.011). A format-only
classifier reaches 0.649, **below** the 0.707 majority baseline — so format is not
a sufficient shortcut on its own, but it is a real confound. A1 stratified
results by it; the two strata behave consistently.

---

# 4. Substrate

**Deployed:** 11 checkpoints (~110 GB) + 128 GB benchmark data = **183 GB**.
All three D-stage checkpoints (`omnisapiens_sft`, `OmniSapiens-7B-RL`,
`OmniSapiens2.0`) are on disk and share one architecture.

**HBA test split** — 18,204 rows, all 16 sources, media embedded as binary.

**Gotcha 16 — HBA test is overwhelmingly emotion/sentiment, not mental health.**
The four clinically-oriented sources (mmpsy_depression 191, mmpsy_anxiety 191,
ptsd_in_the_wild 64, daicwoz 47) total 493 rows, **2.7 % of the split**.

**Gotcha 17 — HBA's `daicwoz` portion is exactly 47 rows, the same count as the
DAIC test split held raw.** Relevant to §1.6.

**Gotcha 18 — do not use `pooler_output` from MentalBERT/MentalRoBERTa.** Both are
MLM checkpoints with no pooler in the weights; `pooler.dense.*` is randomly
initialised at load. Use `last_hidden_state[:, 0]` or mean pooling.

**~~Blocking for B2~~ — corrected 2026-08-20: B2 was never blocked.** HBA MP4s do
carry their own AAC track, but `load_video(strip=True)` — the default — re-muxes
to video-only **in memory** at load time; the design never intended
pre-generated stripped files, which is what the earlier "not run, nothing on
disk" note went looking for. Verified on a real MELD clip:
`['video','audio','data'] → ['video']`.

**Separate bug found and fixed the same day:** `row_media` indexed one level too
deep (`row["videos"][idx][0]`), which subscripts the bytes object and yields an
int; `av` then fails with `InvalidDataError`. The HBA columns are
`list<binary>`, so `row["videos"][idx]` is already the MP4. Every B/D run
depends on this function.

**Naming defect:** `checkpoints/Qwen3-4B-base` is actually
**Qwen3-4B-Instruct-2507** (262,144 ctx; the README names it). This is what
Menta's LoRA requires, so Menta runs — but if it is used as the "neutral
third-party rewriter" for C2/C3, it is an *instruct* model, not a base model.

---

# 5. Not yet measured

- **A2** — Ellie counterfactual. Prompt inventory built
  (`data/processed/daic_ellie_prompt_inventory.json`), still keyword-seeded and
  requiring hand-verification before use.
- **Encoder controls** — MentalBERT / MentalRoBERTa. Need a trained probe head, a
  different pipeline from label log-likelihood.
- **Coverage covariate regression** for A1. Data is in the JSONL (75/188 `full`
  transcripts hit the truncation budget for MentaLLaMA, 39/188 for Menta;
  `ellie_only` never truncates) but the regression is not run.
- **B, C, D** in full.
