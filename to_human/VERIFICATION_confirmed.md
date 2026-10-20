# Verification report — what the code and result files actually say

Audit date 2026-09-05. Every number below was read from a raw result file or a
run log, not from prose notes. Where two files disagree, both are given with
timestamps and the pipeline that produced each.

**Companion file:** `to_human/VERIFICATION_gaps.md` — what was never run and
what cannot be recovered without new inference.

---

## A. Confirmed inconsistencies

### A1. CSS is computed by two different formulas in two different pipelines — CRITICAL

| pipeline | file:line | implemented formula |
|---|---|---|
| DAIC mitigation (i10) | `experiments/i10-model-sweep/code/run_full_suite.py:160` | `css = lambda a,b: float(np.abs(sig(a)-sig(b)).mean())` → **E[\|Δ\|], absolute** |
| DAIC GRPO (i8) | `experiments/i8-grpo-mitigation/code/eval_grpo.py:117` | `np.abs(sig(...)-sig(...)).mean()` → **absolute** |
| DAIC format (i9) | `experiments/i9-format-instruction/code/run_format_ser.py:100` | `np.abs(...).mean()` → **absolute** |
| Cross-dataset (h7) | `experiments/h7-full-matrix/code/build_matrix.py:83` | `css, lo, hi = boot(P[sp["hi"]] - P[sp["lo"]])` → **E[Δ], signed** |
| mmpsy (h4) | `experiments/h4-mmpsy-counterfactual/code/analyze_h4.py:138` | `(P[seg_plus]-P[seg_minus]).mean()` → **signed** |
| ptsd (h5) | `experiments/h5-ptsd-register-cf/code/analyze_h5.py:101` | `(P[ctx_testimony]-P[ctx_vlog]).mean()` → **signed** |
| DAIC A2b | `experiments/a2-ellie-counterfactual/code/analyze_a2b.py:83-88` | reports **both**: signed `Δp(dep)` and `\|Δ\| mean` |

`build_matrix.py:95` has `def verdict(css, flips)` with an explicit
`if css < 0: return "**wrong sign**"` branch — the cross-dataset pipeline was
knowingly built on the signed definition.

**The paper's formal definition (absolute) matches i10/i8/i9 and contradicts
h4/h5/h7.** Negative CSS is impossible under the paper's formula and routine
under h7's.

#### The two definitions disagree on real DAIC models, not just cross-dataset

Recomputed from `experiments/a2-ellie-counterfactual/results/a2b_*.jsonl`
(n = 188 each, `p_yes` from the Yes/No logprob pair):

| model | signed E[Δ] | absolute E[\|Δ\|] | n moving positive | i10 reports |
|---|---|---|---|---|
| MentaLLaMA-chat-7B | **+0.2057** | 0.2057 | 188/188 | 0.2050 |
| Qwen3-4B-base | +0.2537 | 0.2537 | 147/188 (41 negatives sum to −0.0000) | 0.2561 |
| OSim-8B | **+0.1752** | **0.1949** | 107/188 | 0.1960 |
| Menta | **−0.0147** | **0.0266** | 52/188 | 0.0267 |

- **Menta's effect is negative.** 130 of 188 participants move *down* when the
  interviewer turns to clinical questions. The paper reports `0.027` as a small
  positive shortcut sensitivity; the signed quantity is `−0.0147`. **The sign is
  wrong in the paper.**
- **OSim-8B's absolute value overstates by 11 %** (0.195 vs 0.175) and conceals
  that 81/188 participants move the other way.
- MentaLLaMA is the one model where the two agree exactly (0 negatives), which
  is why the collision went unnoticed — the headline model is the special case.

**Recommendation.** Split the metric and report both:

- `CSS` (signed) `= E[p_dep − p_neu]` — directional, the quantity every
  cross-dataset table and every "wrong sign" verdict already uses.
- `|CSS|` (magnitude) `= E[|p_dep − p_neu|]` — what i10/i8/i9 record, and the
  right quantity for "how much does this model move at all".

Report `|CSS| − |signed CSS|` or the fraction moving positive alongside, so a
population that splits (OSim-8B, Menta) is visible rather than averaged away.

---

### A2. `tab:mitigate` mixes three different evaluation sets into one AUC column — CRITICAL

Source of the paper's table is `logs/ser_metric.log` plus
`logs/mitigation_extra.log`. Reading those logs:

| paper row | paper AUC | actual n | the baseline that row must be read against |
|---|---|---|---|
| No mitigation | 0.663 | 188 | — |
| Delete all interviewer turns | 0.787 | 188 | 0.663 ✓ |
| Blank out interviewer words | 0.763 | 188 | 0.663 ✓ |
| Average over 2 / 4 contexts | 0.758 / 0.764 | 188 | 0.663 ✓ |
| Instruct to ignore | 0.752 | 188 | 0.663 ✓ |
| Chain-of-thought | 0.579 | 188 | 0.663 ✓ |
| **Delete only the clinical questions** | **0.665** | **123** | **0.699**, not 0.663 |
| **Threshold correction, observable proxy** | **0.758** | **47 (test split)** | **0.746**, not 0.663 |

`logs/mitigation_extra.log`, DIRECTION C: *"baseline (nothing removed), n=123:
AUC 0.699 / remove only clinical questions 0.665 −0.034"*.
DIRECTION B: *"fitted on train+dev (n=141), applied to test (n=47) / no
correction (full transcript) 0.746 / feature-based correction 0.758"*.

Consequences as the table currently reads:

- **Targeted removal looks like +0.002; it is actually −0.034.** Sign is wrong.
- **Threshold correction looks like +0.095; it is actually +0.012**, and its CI
  on n=47 is [0.575, 0.910] — the effect is not distinguishable from zero.

The paper's own prose ("Surgical removal does not work", §559) is right; the
table contradicts it.

---

### A3. The encoder result file was overwritten by a different model — CRITICAL

Two runs wrote to the **same path**
`experiments/i6-encoder-interventions/results/encoder_interventions.json`:

| run | log | date | checkpoint |
|---|---|---|---|
| first | `logs/i6_encoder_interventions.log` | 2026-08-24 11:45 | **MentalBERT** (`BertModel LOAD REPORT from .../checkpoints/MentalBERT`) |
| second | `logs/i6_roberta.log` | 2026-08-30 13:22 | **MentalRoBERTa** (`RobertaModel LOAD REPORT from .../checkpoints/MentalRoBERTa`) |

The JSON on disk carries the **2026-08-30 MentalRoBERTa** numbers. Confirmed by
exact array match: i6 `B0.auc = [0.5757575757575758, 0.7445887445887446,
0.6168831168831169]` is byte-identical to `M1 train=full` in
`experiments/b0-encoder-mechanism/results/b0_mech_summary_roberta_mean.json`
(ckpt `MentalRoBERTa`), and differs from the MentalBERT run
(`b0_mech_summary.json`, `[0.7619, 0.7641, 0.6385]`).

| arm | MentalBERT (log, 2026-08-24) | MentalRoBERTa (JSON on disk, 2026-08-30) |
|---|---|---|
| B0 ERM | AUC 0.722, CF-Sens 0.0409 | AUC 0.646, CF-Sens 0.0096 |
| D3 counterfactual augmentation | 0.734, 0.0337 | 0.763, 0.0213 |
| **M1 product-of-experts** | **0.540, 0.0039** | 0.578, 0.0013 |
| M2 group DRO | 0.659, 0.0443 | 0.641, 0.0106 |
| M2reg group DRO + heavy L2 | 0.646, 0.0734 | 0.630, 0.0115 |

**The paper's PoE row (`CSS 0.004, AUC 0.540`) and its encoder baseline
(`AUC 0.722, dependence 0.041`) are the MentalBERT numbers.** They are correct
and internally consistent — but they exist **only in the log file**. The JSON a
reader or reviewer would open reports different numbers for the same arm names.

Also: the MentalBERT run recorded **no C-Sens column** (its table is
`AUC / sd / CF-Sens / gap`). C-Sens for the encoder exists only for
MentalRoBERTa.

---

### A4. Probing peak Cohen's d — the paper has MentaLLaMA and Qwen3-base swapped

Source: `experiments/i7-causal-probing/results/layerwise_multi.json`, computed by
`experiments/i7-causal-probing/code/layerwise_multi.py:xfit_d` (cross-fitted:
direction from the train fold, effect size on the held-out fold, 5-fold
`GroupKFold` on `participant_id`).

| model | peak d | peak layer | depth | AUC at peak | layers with d > 1 | % | source |
|---|---:|---:|---:|---:|---:|---:|---|
| MentaLLaMA-chat-7B | **2.93** | **10** | 0.31 | 0.9879 | 31 / 33 | **93.9 %** | `layerwise_multi.json` |
| Menta | 3.33 | 36 | 1.00 | 0.9545 | 35 / 37 | 94.6 % | same |
| Qwen3-4B-base | **4.01** | **18** | 0.50 | 0.9983 | 35 / 37 | 94.6 % | same |
| OSim-8B | 3.25 | 21 | 0.56 | 0.9387 | 33 / 37 | 89.2 % | same |

- The sentence **"probe separation peaks at layer 10 (d = 2.93)"** for
  MentaLLaMA is **correct**. Peak layer 10, d = 2.93.
- The **table cell** giving MentaLLaMA `d = 4.01` and Qwen3-base `d = 2.93` is
  **wrong — the two are swapped**. 4.01 is Qwen3-base at layer 18; 2.93 is
  MentaLLaMA at layer 10.
- The claimed range **"92–97 %"** is **wrong at the bottom end**. Actual range is
  **89.2–94.6 %**. OSim-8B fails d > 1 at four layers (0, 1, 2, 3), not two.
  Correct wording: **"d > 1 at 89–95 % of layers"**.
- Every model has d = 0.00 at layer 0 (the embedding layer) — worth stating,
  because a non-zero d there would mean the separation was already in the token
  embeddings and therefore trivial.

---

### A5. Model-free shortcut numbers — all five values are real, all measure different things

Every number the checklist flagged exists and is correct. The problem is that
the paper presents them in one column as if they were one metric.

| Dataset / subset | Experiment | Input | Target | Classifier | Metric | Split | Value | Source file |
|---|---|---|---|---|---|---|---:|---|
| DAIC-WOZ | A1.1 lexical ceiling | interviewer turns, **full text** (`ellie_only`) | PHQ-8 binary | TF-IDF + LogReg | **AUC** | 5-fold × 10 repeats | **0.810** | `logs/a1_lexical_ceiling.log`, tabulated `ALL_RESULTS.md:32` |
| DAIC-WOZ | A2a question selection | binary asked/not-asked over **75 prompt IDs, no text** | PHQ-8 binary | LogReg | **AUC** (sd 0.025) | 10 CV seeds, n=123 | **0.753** | `experiments/a2-ellie-counterfactual/results/a2a_question_selection_auc.json` (`auc_mean`) |
| HBA `mmpsy_depression` | H1 numeric probe | **number of segments** (`length_segments`) | depression | LogReg on 1 scalar | **balanced acc** | 5-fold StratifiedKFold × 5 seeds | **0.772** | `experiments/h1-hba-shortcut/results/audit.json` |
| HBA `mmpsy_depression` | H1 text probe | full text | depression | TF-IDF + LogReg | balanced acc | 5-fold × 3 seeds | 0.507 (= chance) | same |
| HBA `ptsd_in_the_wild` | H1 text probe | full text | PTSD | TF-IDF + LogReg | **balanced acc** | 5-fold × 3 seeds | **0.948** | same |
| HBA `ptsd_in_the_wild` | H1 stopword probe | **stopwords only** | PTSD | TF-IDF + LogReg | balanced acc | same | **0.880** | same |
| IMHI `Irf` | H2 P3 text probe | full post | Irf label (2-way) | TF-IDF + LogReg | **balanced acc** | 5-fold × 3 seeds | **0.643** | `experiments/h2-imhi-shortcut/results/audit.json` → `P3.Irf.text` |
| IMHI `Irf` | H2 P3 stopword probe | **stopwords only** | Irf label | TF-IDF + LogReg | balanced acc | same | **0.622** | `P3.Irf.stopword` |
| IMHI (**all 9 sources**) | H2 P2 source-ID probe | **stopwords only** | **which of 9 corpora** | TF-IDF + LogReg | **balanced acc**, chance 0.111 | same | **0.372** | `P2.source_id_stopwords` |
| IMHI (all 9 sources) | H2 P2 source-ID probe | full text | which of 9 corpora | TF-IDF + LogReg | balanced acc | same | 0.522 | `P2.source_id` |
| HBA `daicwoz` | H1 text probe | full text | depression | TF-IDF + LogReg | balanced acc | 5-fold × 3 seeds, **n = 47** | 0.607 | `h1 audit.json` |

Metric implementation for H1/H2: `balanced_accuracy_score` with
`cross_val_predict(StratifiedKFold(5, shuffle=True))` —
`experiments/h1-hba-shortcut/code/audit.py:107-135`,
`experiments/h2-imhi-shortcut/code/*.py:65-81`.

**Specific corrections:**

- **DAIC 0.810 vs 0.753 are both right and are different constructs.** 0.810 is
  the interviewer's *words*; 0.753 is *which questions were asked*, with no text
  at all. Do not present them as competing estimates of one quantity. Use 0.753
  wherever the claim is about adaptive question selection (it is the stronger,
  more surprising result: no lexical content required); use 0.810 for the
  lexical ceiling.
- **mmpsy 0.772 is balanced accuracy from a segment-count probe, not AUC, and
  not 0.753.** The "0.753 AUC" attributed to mmpsy in the cross-corpus
  discussion is DAIC's A2a number, misfiled.
- **Irf "0.373 source-identity AUC" is wrong on three counts**: the value is
  0.372, it is **balanced accuracy** not AUC, and it is a **9-way corpus-identity
  probe over all of IMHI**, not a property of `Irf`. It is also *below* the
  0.643/0.622 label probes, so it cannot be quoted as evidence about `Irf`.
- **ptsd 0.948 (full text) and 0.880 (stopwords) are both correct.** The
  stopword figure is the interesting one — register alone gets 0.880 against a
  0.500 majority.

---

### A6. `%` of layers with d > 1

Claimed 92–97 %. Actual, counted from `layerwise_multi.json`:
93.9 / 94.6 / 94.6 / 89.2 %. **Replace with "89–95 %".**

---

## B. Verified correct — no change needed

### B1. Behavioural coupling counts (checklist item 4) — the paper is right

**Operational definition**, from `layerwise_multi.py:139-150`:

- For each layer, `D = B − A` (per-participant representational displacement
  between `cf_depression` and `cf_neutral`), `u = mean(D)` normalised.
- `partial_r` = partial correlation between `D @ u` and the behavioural shift
  `dp = sig(logit_dep) − sig(logit_neu)`, **controlling for `‖D‖`** — so a layer
  cannot win by simply moving more.
- `ci` = bootstrap 95 % CI on that partial correlation.
- **"significant"** = `ci` excludes 0.
- **`random_r`** = the same partial correlation computed against a random
  direction drawn at the same layer; **"random wins"** = `|random_r| > |partial_r|`.

Counts, recomputed:

| model | significant layers | random-direction wins | mean \|r\| real | mean \|r\| random |
|---|---|---|---|---|
| MentaLLaMA | **15 / 32** | **5 / 32** | 0.150 | 0.067 |
| OSim-8B | 19 / 36 | 15 / 36 | 0.135 | 0.101 |
| Menta | 1 / 36 | 18 / 36 | 0.059 | 0.071 |
| Qwen3-4B-base | 1 / 36 | 21 / 36 | 0.061 | 0.082 |

Denominators exclude layer 0, where `partial_r` is `NaN` (zero displacement).
The paper's `/32` and `/36` are therefore **correct**, as are all eight counts.

**"MentaLLaMA is the only model where the direction beats matched random
directions" is fully supported.** MentaLLaMA loses to random on 5/32 layers;
every other model loses on 15–21 of 36. Menta and Qwen3-base sit at 1/36
significant, exactly the α = 0.05 false-positive rate.

MentaLLaMA's coupling band is contiguous, **layers 8–21**, peaking at
**layer 13 (r = +0.264)** — *not* at layer 10 where d peaks. This dissociation is
why `phase3_leace.py:4-7` erases at layers 8–24 and not at layer 10:
*"signal STRENGTH peaks at layer 10, but behavioural relevance peaks at layer
13 … We erase where the signal relates to BEHAVIOUR, not where it is most
decodable."*

### B2. The 8-model mitigation summary reproduces exactly

Regenerated from `experiments/i10-model-sweep/results/suite_*.json`
(command in section E). All seven means and all seven improve-counts match:

| method | mean ΔAUC | improves | paper | match |
|---|---:|---|---|---|
| `delete_interviewer` | +0.0146 | 4/8 | +0.0146, 4/8 | ✓ |
| `blank_words` | +0.0042 | 4/8 | +0.0042, 4/8 | ✓ |
| `marginalise_2` | +0.0114 | 5/8 | +0.0114, 5/8 | ✓ |
| `marginalise_4` | +0.0108 | 4/8 | +0.0108, 4/8 | ✓ |
| `targeted_removal` | −0.0034 | 2/6 | −0.0034, 2/6 | ✓ |
| `threshold_proxy` | +0.0041 | 4/6 | +0.0041, 4/6 | ✓ |
| `instruct_ignore` | +0.0107 | 4/8 | +0.0107, 4/8 | ✓ |

`instruct_ignore` CSS: improves 4/8, worsens 4/8 (rises on Menta,
OmniSapiens-SFT, -RL, -2.0) ✓.
`delete_interviewer` / `blank_words` CS: improves **8/8** each ✓.

Totals: **52 run cells** (2 arms × 2 models missing), 27 improve, 25 degrade, 0 ties.

**One caveat that should be stated in the caption:** the CSS = 0.0000 entries for
`delete_interviewer`, `blank_words`, `marginalise_2` and `marginalise_4` are
**hard-coded**, not measured — `run_full_suite.py:197-200` literally writes
`css=0.0`. This is correct (with the channel removed there is nothing to swap)
but it is zero **by construction**. Only `instruct_ignore` has a measured CSS
(`css=css(S["p1_cf_depression"], S["p1_cf_neutral"])`, line 204).

### B3. CoT numbers are correct

Source `logs/i5_p2_analysis.log` and `logs/i5_p2_fa.log`, MentaLLaMA-chat-7B,
n = 188, single run:

| quantity | value | CI |
|---|---:|---|
| CF-Sens (CSS) | **0.0302** | [0.0109, 0.0542] |
| construct sensitivity | **0.0653** | [0.0395, 0.0957] |
| AUC, `cf_none` arm | **0.579** | [0.488, 0.669] |
| AUC, `cf_neutral` arm | 0.660 | [0.565, 0.745] |
| AUC, `cf_depression` arm | 0.628 | [0.540, 0.710] |
| mean p, P(Yes) | 0.945 / **0.952** | — |

Paper's `CSS 0.030, AUC 0.579, CS 0.065` ✓ — all three correct, AUC being the
`cf_none` arm. **The collapse claim is understated, not overstated**: P(Yes)
reaches **0.979** on both counterfactual arms. Raw values, not deltas.

### B4. GRPO collapse is real, and it is all three runs — but "0" is rounded

`experiments/i8-grpo-mitigation/results/eval_*.json`:

| run | p_yes | CSS | construct_sens | iid_auc | conflict_auc | pair_acc | parse_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.3936 | 0.2864 | 0.1354 | 0.7630 | 0.0686 | 0.665 | **0.15** |
| M0 | **0.0000** | 0.00169 | 0.00246 | 0.7245 | 0.2664 | 0.625 | 1.00 |
| M0_bal_batch | **0.0000** | 0.0000444 | 0.000635 | 0.6752 | 0.3427 | 0.555 | 1.00 |
| M1 | **0.0000** | 0.00143 | 0.00329 | 0.7096 | 0.2768 | 0.595 | 1.00 |

- **Which class:** every GRPO run predicts **"No"** for **all 188** participants
  (`p_yes = 0.0` exactly).
- **Collapse occurred in all three training runs**, not one — this is stronger
  than the paper claims and should be said.
- **CSS is not exactly 0.** It is 0.0014–0.0017 for M0/M1 (rounds to 0.001–0.002)
  and 4.4e-5 for M0_bal_batch. **CS is not 0 either** — 0.0006–0.0033. Write
  "≤ 0.002" or give the values; "0.000" is only true for M0_bal_batch's CSS.
- **AUC does not collapse** (0.675–0.725, vs baseline 0.763). Ranking survives
  because the log-odds still vary even though the argmax never does. This is
  worth stating: it is another instance of the paper's own thesis that a single
  metric hides the failure.
- **The baseline row has `parse_rate = 0.15`.** Only 15 % of baseline
  generations parsed. Any baseline-vs-GRPO comparison inherits that; it must be
  disclosed.

### B5. Counterfactual augmentation and Group DRO **were run** — the checklist's premise is wrong

Both exist, 3 seeds each, in `logs/i6_encoder_interventions.log` (MentalBERT) and
`experiments/i6-encoder-interventions/results/encoder_interventions.json`
(MentalRoBERTa). Arm names from
`experiments/i6-encoder-interventions/code/run_encoder_interventions.py:4-6`:

- `D3` = counterfactual augmentation ("add (C, S', Y) copies to the training set")
- `M1` = product of experts ("train against a bias-only model of S")
- `M2` = group DRO ("minimise worst-group loss over g = (Y, s-hat)"), `M2reg` = same with heavy L2

They are **absent from the paper's tables, not from the repository.** See D5.

---

## C. Headline claim: "input removal missed the effect"

### The experiment

`removal_ellie_only_auc` in `suite_*.json` — the model scores the **interviewer's
turns alone**, participant text deleted entirely. `removal_ci` is a bootstrap
95 % CI. Compared against `counterfactual_css`, which holds the participant's
words byte-identical and swaps only the interviewer's questions.

### Evidence table, all 8 models, n = 188

| Model | Removal AUC (`ellie_only`) | 95 % CI | Clears chance? | Full AUC | CSS | flips | Interpretation |
|---|---:|---|---|---:|---:|---:|---|
| **MentaLLaMA-chat-7B** | **0.5459** | **[0.461, 0.632]** | **no** | 0.6836 | **0.2050** | **0.559** | **removal null, counterfactual large — the "miss"** |
| Menta | 0.5757 | [0.489, 0.665] | no | 0.7642 | 0.0267 | 0.000 | both null — a true negative, not a miss |
| Qwen3-4B-base | 0.7103 | [0.630, 0.784] | yes | 0.8096 | 0.2561 | 0.261 | removal detects it |
| OSim-8B | 0.7217 | [0.634, 0.798] | yes | 0.8053 | 0.1960 | 0.298 | removal detects it |
| OmniSapiens-SFT | 0.6863 | [0.602, 0.774] | yes | 0.7998 | 0.4573 | 0.479 | removal detects it |
| OmniSapiens-RL | 0.6614 | [0.572, 0.754] | yes | 0.7860 | 0.4710 | 0.516 | removal detects it |
| OmniSapiens-2.0 | 0.6991 | [0.607, 0.786] | yes | 0.7811 | 0.5051 | 0.511 | removal detects it |
| HumanOmniV2 | 0.6253 | [0.531, 0.716] | yes | 0.7913 | 0.3715 | 0.511 | removal detects it |

### Verdict

**`Partially supported / needs narrower wording`.**

- Removal returns a null (CI covering 0.5) on **2 of 8 models**.
- Of those two, only **MentaLLaMA** has a large counterfactual effect
  (CSS 0.205, 55.9 % flips). Menta is null on both tests — a correct agreement,
  not a miss.
- **So the "miss" is one model out of eight**, not a general property of removal.
  On the other six, removal correctly flags the shortcut.

**The defensible general claim is weaker but still real**: removal does not track
severity. **Spearman ρ between removal AUC and CSS across the 8 models is
+0.190** — the two tests rank the models almost independently. OmniSapiens-2.0
has the largest counterfactual effect in the panel (CSS 0.505) and a *middling*
removal AUC (0.699); Qwen3-base has the highest removal AUC in the panel (0.710)
and half the CSS (0.256).

Suggested replacement wording is in D8.

---

## D. Safe paper edits

**D1 (CSS).** Replace the single definition with two, and mark every table with
which one it uses. Keep the absolute form for the DAIC mitigation table (that is
what the code computed); use the signed form for every cross-dataset table and
every claim about direction. Correct Menta's DAIC CSS to **−0.0147 (signed)** /
0.0266 (magnitude), and add the "fraction moving positive" column.

**D2 (`tab:mitigate`).** Move the two mismatched rows into their own block with
their own baselines, or add an explicit `n` column. Targeted removal is
**−0.034 against 0.699 on n = 123**; threshold correction is **+0.012 against
0.746 on n = 47**, CI [0.575, 0.910].

**D3 (encoder).** Cite the MentalBERT log run for the numbers already in the
paper and say so, or re-run MentalBERT to a **distinct filename**. As it stands
the JSON contradicts the paper. Also add: the MentalBERT arm has no C-Sens.

**D4 (probing).** Swap the two table cells: MentaLLaMA **2.93** at layer 10,
Qwen3-base **4.01** at layer 18. Change "92–97 %" to **"89–95 %"**.

**D5 (mitigation coverage).** Counterfactual augmentation (D3) and Group DRO
(M2, M2reg) were run and can be added to the training-level block, at 3 seeds on
the encoder. Say explicitly that the training-level rows are encoder-only and
n = 47 test, not the 8-LLM panel.

**D6 (CoT / GRPO).** CoT values stand. For GRPO, write "CSS ≤ 0.002 and CS
≤ 0.003 across all three runs, with P(Yes) = 0 for all 188 participants in every
run" rather than "0.000"; and disclose the baseline's 15 % parse rate.

**D7 (CSS = 0 entries).** Add to the `tab:mitigate` caption that the four
removal/marginalisation CSS zeros are **hard-coded in
`run_full_suite.py:197-200`**, zero by construction. The paper already says this
in prose (`$^\dagger$ zero by construction`) — make sure it also covers
`marginalise_2/4`, not just delete/mask.

**D8 (headline).** Narrow to what the data shows. Suggested:

> On MentaLLaMA-chat-7B, removal returns 0.546 (95 % CI [0.461, 0.632], covering
> chance) while counterfactual substitution on the same model and the same 188
> transcripts moves the prediction by 0.205 and flips 56 % of decisions. Across
> all eight models the two tests rank severity almost independently
> (Spearman ρ = 0.19): removal's highest score belongs to a model with half the
> counterfactual sensitivity of the panel's worst case. Removal is therefore not
> a substitute for counterfactual testing — it can return a null on a model with
> a large effect, and it does not order models by how large that effect is.

**D9 (model-free numbers).** Give each its own row with input/target/metric
columns as in A5. Delete the "0.373 source-identity AUC for Irf" claim entirely —
it is a 9-way corpus probe at 0.372 balanced accuracy, not an Irf result.

---

## E. Authoritative files and regeneration commands

| # | What | Authoritative source | Regenerate |
|---|---|---|---|
| 1 | CSS, DAIC absolute | `experiments/i10-model-sweep/results/suite_*.json` → `detection.counterfactual_css` | `run_full_suite.py:160` |
| 1 | CSS, DAIC signed | `experiments/a2-ellie-counterfactual/results/a2b_*.jsonl` | see script below |
| 1 | CSS, cross-dataset signed | `experiments/h7-full-matrix/results/*.jsonl` | `python experiments/h7-full-matrix/code/build_matrix.py` |
| 2 | model-free DAIC lexical | `logs/a1_lexical_ceiling.log` | `python scripts/a1_lexical_ceiling.py` |
| 2 | model-free DAIC question selection | `experiments/a2-ellie-counterfactual/results/a2a_question_selection_auc.json` | `python experiments/a2-ellie-counterfactual/code/a2a_prompt_label_association.py` |
| 2 | HBA probes | `experiments/h1-hba-shortcut/results/audit.json` | `python experiments/h1-hba-shortcut/code/audit.py` |
| 2 | IMHI probes | `experiments/h2-imhi-shortcut/results/audit.json` | `experiments/h2-imhi-shortcut/code/` |
| 3,4 | probing | `experiments/i7-causal-probing/results/layerwise_multi.json` | `python experiments/i7-causal-probing/code/layerwise_multi.py` |
| — | LEACE | `experiments/i7-causal-probing/results/phase3_leace_{L8-24,layer13}.json` | `python experiments/i7-causal-probing/code/phase3_leace.py --layers 8..24` |
| 5,10 | 8-model sweep | `experiments/i10-model-sweep/results/suite_*.json` | `python experiments/i10-model-sweep/code/run_full_suite.py --tag ... --ckpt ...` |
| 5,9 | encoder, **MentalBERT** | `logs/i6_encoder_interventions.log` (JSON overwritten) | `python experiments/i6-encoder-interventions/code/run_encoder_interventions.py --ckpt checkpoints/MentalBERT` |
| 5,9 | encoder, MentalRoBERTa | `experiments/i6-encoder-interventions/results/encoder_interventions.json` | same, `--ckpt checkpoints/MentalRoBERTa` |
| 9 | CoT | `logs/i5_p2_analysis.log`, `logs/i5_p2_fa.log`, `experiments/i5-p2-cot/results/p2_MentaLLaMA-chat-7B.jsonl` | `experiments/i5-p2-cot/code/run_p2.py` |
| 9 | GRPO | `experiments/i8-grpo-mitigation/results/eval_*.json` | `experiments/i8-grpo-mitigation/code/eval_grpo.py` |
| 2 | tab:mitigate source | `logs/ser_metric.log`, `logs/mitigation_extra.log` | — (no script found; see gaps file) |

### Regenerating the mitigation Δ tables

```bash
cd /data/cl195/NeurIPS_workshop && source env.sh
$SC_PY - <<'PY'
import json, statistics as st
T=[('mentallama_suite','MentaLLaMA'),('menta','Menta'),('qwen3_base','Qwen3-4B-base'),
   ('osim8b','OSim-8B'),('omnisapiens_sft','OmniSap-SFT'),('omnisapiens_rl','OmniSap-RL'),
   ('omnisapiens_20','OmniSap-2.0'),('humanomniv2','HumanOmniV2')]
M=['delete_interviewer','blank_words','marginalise_2','marginalise_4',
   'targeted_removal','threshold_proxy','instruct_ignore']
D={n:json.load(open(f'experiments/i10-model-sweep/results/suite_{t}.json'))['interventions']
   for t,n in T}
for m in M:
    xs=[D[n][m]['auc']-D[n]['none']['auc'] for n in D if D[n].get(m,{}).get('auc') is not None]
    print(f"{m:20s} mean={st.mean(xs):+.4f} improves={sum(1 for x in xs if x>0)}/{len(xs)}")
PY
```

### Recomputing signed vs absolute CSS on DAIC

```bash
cd /data/cl195/NeurIPS_workshop && source env.sh
$SC_PY - <<'PY'
import json, glob, os, numpy as np
def pyes(lp):
    v=np.array([lp["Yes"],lp["No"]],float); e=np.exp(v-v.max()); return e[0]/e.sum()
for p in sorted(glob.glob('experiments/a2-ellie-counterfactual/results/a2b_*.jsonl')):
    by={}
    for l in open(p):
        r=json.loads(l); by.setdefault(r["arm"],{})[r.get("uid",r.get("participant_id"))]=r
    ids=sorted(set(by["cf_depression"])&set(by["cf_neutral"]))
    d=np.array([pyes(by["cf_depression"][i]["logprobs"])-pyes(by["cf_neutral"][i]["logprobs"])
                for i in ids])
    print(f"{os.path.basename(p):32s} signed={d.mean():+.4f} abs={np.abs(d).mean():.4f} "
          f"pos={np.mean(d>0):.3f} n={len(ids)}")
PY
```
