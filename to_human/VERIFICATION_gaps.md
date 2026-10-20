# What is missing, unrecoverable, or needs a re-run

Companion to `to_human/VERIFICATION_confirmed.md`. Audit date 2026-09-05.

---

## C1. Never run — claimed or implied but no result file exists

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | **ESConv negative control** | **numbers exist in prose only; no script, no result file** | `experiments/h3-esconv-shortcut/code/` **empty**, `results/` **empty**; only `data/esconv.json` (5.8 MB) is present. `paper/FACTS_FOR_WRITING.md:175-177` says so itself: *"ESConv numbers are from a 2026-09-01 session and **are not yet in a committed script** — they must be written into `experiments/h3-esconv-shortcut/code/` before citing."* |
| 2 | `mosei_emotion` counterfactual | not run | predicted null (92 % stopword ratio); would test the granularity account as a prediction rather than post-hoc |
| 3 | C2 / C3 (IMHI external-API arms) | **deliberately blocked** | would require sending Reddit mental-health posts to an external API; held pending your explicit approval, never sent |
| 4 | B2 modality-conflict | data prepared, never run | `experiments/b2-modality-conflict/` has no results; the audio-strip path is verified so it is unblocked |

**ESConv is the urgent one.** The paper's §4 claim — *"the one dataset that
shares the structural precondition is at chance"* — rests on a table of eight
numbers (`strategy counts 0.220`, `strategy n-grams 0.203`, etc.) that cannot be
regenerated from anything in the repository. It is the decisive negative control
and it is currently unreproducible. Either commit the script or drop the table.

---

## C2. Metrics that cannot be recovered without new inference

### CSS / CS for `targeted_removal` and `threshold_proxy`

`run_full_suite.py:201-202` writes `css=None, cs=None` for both. The question is
whether that is a recording gap or a real one. It differs per method:

| Method | CSS meaningful? | CSS already available? | CS meaningful? | CS available? | Need re-run? |
|---|---|---|---|---|---|
| `threshold_proxy` | yes | **no, but fully derivable offline** | yes | **no, but derivable — and provably unchanged** | **no** |
| `targeted_removal` | yes, but weakly informative | no | yes | no | **yes, 2 arms × 8 models** |

#### `threshold_proxy` — derivable with no model calls

The intervention is a pure post-hoc correction in log-odds space
(`run_full_suite.py:189`):

```
S["threshold_proxy"] = S["full"] − (co[0] + co[1] * feat)
```

where `feat` is a regex clinical-word density over the interviewer turns
(`run_full_suite.py:179-183`, pattern
`\b(diagnos\w*|therap\w*|depress\w*|p_t_s_d|ptsd|sleep|medicat\w*|counsel\w*|symptom\w*|feel\w*)\b`,
normalised by word count), and `co` is stored in each `suite_*.json` as
`threshold_fit.coef`.

Everything needed is on disk:

- `scores.cf_depression` and `scores.cf_neutral` — per-participant log-odds, 188 each
- `threshold_fit.coef` — e.g. MentaLLaMA `[-0.45618639, 7.01515313]`
- the cf pools are **fixed strings in the run script**, so `feat` for each cf arm
  is one constant computable with the regex above, no model, no data

Procedure: compute `feat_dep` and `feat_neu` from the two pools, subtract
`co[0] + co[1]*feat_arm` from each arm's log-odds, sigmoid, then take the paired
mean. **Zero inference.**

**CS under `threshold_proxy` is provably unchanged.** The E5 construct arms hold
the interviewer turns fixed and vary only the participant's content, so `feat` is
identical across the E+/E− pair, so the same constant is subtracted from both
log-odds. The correction cannot change a paired within-participant contrast in
log-odds space — and after the sigmoid it changes it only by the curvature, which
is computable. This is worth one sentence in the paper: **threshold correction
cannot damage construct sensitivity, by construction**. That is a genuine
argument in its favour and it is currently unstated.

#### `targeted_removal` — needs a re-run

`run_full_suite.py:123-132` builds it by deleting the `clinical_care` prompt
cluster from the **real** transcript (`strip_clinical(r["text_full"])`), then
scoring once. There is no counterfactual pair, and the arm's text differs from
the cf pools, so nothing on disk can be recombined into a CSS.

Two ways to fix, and they answer different questions:

- **(a) 2 extra arms** — apply `strip_clinical` to the cf_depression and
  cf_neutral pools and re-score. Cost: 2 × 188 forward passes × 8 models.
  Caveat: `strip_clinical` would delete most of the cf_depression pool (it *is*
  clinical questions), so the two arms become near-identical and CSS falls toward
  0 for close to the same reason the delete/marginalise arms do. **Expect a
  near-tautological result.**
- **(b) the informative version** — leave the pools intact and strip clinical
  questions only from the *real* transcript, then measure CSS against a
  same-treatment neutral arm. This tests whether partial removal leaves a
  residual channel. This is the experiment the paper's §559 claim ("surgical
  removal does not work") actually needs.

**Recommendation: run (b), skip (a).** Currently the paper's targeted-removal
claim rests on an AUC delta alone, and (per `VERIFICATION_confirmed.md` §A2)
that delta is quoted against the wrong baseline.

---

## C3. Numbers that exist in two incompatible versions

### `threshold_proxy` AUC: 0.683 or 0.758?

| source | value | evaluated on | fitted on |
|---|---:|---|---|
| `suite_mentallama_suite.json` → `interventions.threshold_proxy.auc` | **0.6827** | **all 188** | train+dev (141) |
| `logs/mitigation_extra.log`, DIRECTION B | **0.758** [0.575, 0.910] | **test only, n = 47** | train+dev (141) |

`run_full_suite.py:189` applies the fitted correction to **every** row and then
`auc(y, S["threshold_proxy"])` scores all 188 — so 141 of the 188 were used to
fit the correction being evaluated. The i10 number is **in-sample contaminated**;
the log number is clean but has an interval spanning 0.34 of AUC.

**Neither is quotable as-is.** The clean-but-wide number is the honest one; say
n = 47 and give the CI. Do not use the i10 value for a utility claim.

### Encoder results: MentalBERT vs MentalRoBERTa

Covered in `VERIFICATION_confirmed.md` §A3. Summary of what is *lost*: the
**MentalBERT** intervention run survives only as text in
`logs/i6_encoder_interventions.log`; its JSON was overwritten on 2026-08-30 by
the MentalRoBERTa run. The paper's numbers are the MentalBERT ones.

**Cheapest fix:** re-run MentalBERT to a distinct filename —

```bash
cd /data/cl195/NeurIPS_workshop && source env.sh
CUDA_VISIBLE_DEVICES=4 $SC_PY experiments/i6-encoder-interventions/code/run_encoder_interventions.py \
  --ckpt checkpoints/MentalBERT --methods B0,D3,M1,M2,M2reg
# then rename the output before it clobbers the roberta file
```

Note the script hard-codes its output path, so **rename the existing
`encoder_interventions.json` first** or the RoBERTa results are lost in turn.
This is the same failure mode that caused the problem.

**Metric gap:** the MentalBERT run recorded no `C-Sens` column
(`AUC / sd / CF-Sens / gap` only). C-Sens for MentalBERT does not exist anywhere
and would come free with the re-run above.

---

## C4. Mitigation coverage matrix

`run` = result file on disk · `log-only` = value exists in a log, JSON overwritten ·
`n/a` = arm not applicable to that model class · `not run` = no artefact anywhere

### The 8-LLM panel, DAIC-WOZ, n = 188, 1 run each (no seeds)

| Method | MentaLLaMA | Menta | Qwen3-base | OSim-8B | Omni-SFT | Omni-RL | Omni-2.0 | HumanOmni | Metrics recorded |
|---|---|---|---|---|---|---|---|---|---|
| delete interviewer | run | run | run | run | run | run | run | run | auc, css†, cs |
| blank interviewer words | run | run | run | run | run | run | run | run | auc, css†, cs |
| marginalise 2 | run | run | run | run | run | run | run | run | auc, css† |
| marginalise 4 | run | run | run | run | run | run | run | run | auc, css† |
| targeted removal | run | **not run** | **not run** | run | run | run | run | run | auc only |
| threshold correction | run | **not run** | **not run** | run | run | run | run | run | auc only (in-sample) |
| instruct ignore | run | run | run | run | run | run | run | run | auc, **css (measured)**, — |
| **chain-of-thought** | **run** | not run | not run | not run | not run | not run | not run | not run | css, cs, auc ×3 arms |

† hard-coded `0.0` at `run_full_suite.py:197-200`, not measured.

`targeted_removal` and `threshold_proxy` are missing on Menta and Qwen3-base
because both arms were added after those two models ran
(`paper/FACTS_FOR_WRITING.md:497-499`). Cost to close: 2 arms × 2 models × 188 =
752 forward passes. Cheap; worth doing so the panel is complete at 8/8.

### Training-level methods — encoder only, never on any LLM

| Method | Arm | MentalBERT | MentalRoBERTa | Seeds | Test n | Metrics |
|---|---|---|---|---|---|---|
| counterfactual augmentation | `D3` | **log-only** (0.734 / 0.0337) | run (0.763 / 0.0213) | 3 | 47 | auc, cf_sens, gap (+c_sens on RoBERTa) |
| product-of-experts | `M1` | **log-only** (0.540 / 0.0039) | run (0.578 / 0.0013) | 3 | 47 | same |
| group DRO | `M2` | **log-only** (0.659 / 0.0443) | run (0.641 / 0.0106) | 3 | 47 | same |
| group DRO + heavy L2 | `M2reg` | **log-only** (0.646 / 0.0734) | run (0.630 / 0.0115) | 3 | 47 | same |
| ERM baseline | `B0` | **log-only** (0.722 / 0.0409) | run (0.646 / 0.0096) | 3 | 47 | same |

**Correction to the checklist's premise:** counterfactual augmentation and Group
DRO **were run**, 3 seeds each, on both encoders. They are missing from the
paper's tables, not from the repository.

### GRPO — LLM, 3 training runs, 1 eval each

| Run | Status | p_yes | CSS | CS | iid AUC |
|---|---|---:|---:|---:|---:|
| baseline | run (**parse_rate 0.15**) | 0.394 | 0.2864 | 0.1354 | 0.7630 |
| M0 | run, **collapsed** | 0.0000 | 0.00169 | 0.00246 | 0.7245 |
| M0_bal_batch | run, **collapsed** | 0.0000 | 0.0000444 | 0.000635 | 0.6752 |
| M1 | run, **collapsed** | 0.0000 | 0.00143 | 0.00329 | 0.7096 |

All three collapse to "No" on all 188 participants. No seed variation within a
config — three *configurations*, one run each. **"Collapse occurred for all
training seeds" cannot be said; "for all three configurations we tried" can.**

---

## C5. Missing write-ups

Six experiments have result files but no `analysis.md`, so their numbers exist
only as raw JSON plus whatever prose has been written elsewhere:

| Experiment | Has results | Has analysis.md | What is uncovered |
|---|---|---|---|
| `i10-model-sweep` | yes (8 files) | **no** | the entire 8-model panel — detection *and* mitigation |
| `i7-causal-probing` | yes (5 JSON + 6 npz) | **no** | all probing and LEACE numbers |
| `i9-format-instruction` | yes | **no** | format/instruction SER variants |
| `h7-full-matrix` | yes (16 jsonl + matrix.md) | **no** | cross-dataset counterfactual matrix |
| `h2-imhi-shortcut` | yes | **no** | all IMHI probe numbers, incl. the `Irf` values in §A5 |
| `b2-modality-conflict` | **no** | no | nothing run |

`i10` and `i7` are the two that carry paper-critical numbers. Everything the
paper says about them currently traces to raw JSON or to log files, with no
intervening document stating protocol, n, or caveats. That is how the
CSS-definition collision and the encoder overwrite survived this long.

---

## C6. Ranked list of what to run

| Priority | Task | Cost | Unblocks |
|---|---|---|---|
| **1** | Commit the ESConv script and regenerate its 8 numbers | small (CPU, model-free probes) | §4's decisive negative control, currently unreproducible |
| **2** | Re-run MentalBERT interventions to a **new filename** | ~5 × 3 seeds encoder fine-tunes | resolves A3; also yields the missing MentalBERT C-Sens |
| **3** | Derive `threshold_proxy` CSS/CS offline (no inference) | minutes | fills 2 of the 4 missing metric cells; supports a new claim (threshold correction cannot damage CS) |
| **4** | `targeted_removal` + `threshold_proxy` on Menta and Qwen3-base | 752 forward passes | completes the panel at 8/8 |
| **5** | Clean test-split-only AUC for `threshold_proxy` on all 8 | re-score, no new inference | removes the in-sample contamination in C3 |
| **6** | `targeted_removal` CSS, variant (b) | 188 × 8 forward passes | the §559 "surgical removal does not work" claim currently has no CSS behind it |
| 7 | Write `analysis.md` for `i10` and `i7` | writing only | the two files most needed for review |
| 8 | `mosei_emotion` counterfactual | 1 dataset × 8 models | turns the granularity account into a prediction |
| 9 | Probing ablations: rank 2–4 eraser, full-layer band, post-erasure downstream d | 3 LEACE variants | would let the paper say something about non-linearity instead of declaring it a limitation |

Items 3 and 5 need no GPU at all and close two of the checklist's questions
outright.
