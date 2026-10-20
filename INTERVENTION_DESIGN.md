# Intervention Design — where should we intervene to recover construct-robust generalization?

Concrete implementation of `draft_research_plan.md`. Supersedes it.
Companion to `EXPERIMENT_DESIGN.md` (diagnosis phase, experiments A–D), whose
formalism, metric definitions and mandatory controls carry over unchanged.

**Core question.** Mental-health models exploit the interviewer shortcut
(established: A2, Δp = +0.206, 58.5 % prediction flips). Where should we
intervene — data, objective, representation, or inference — to remove that
reliance *without destroying the construct signal*?

**Central thesis (revised 2026-08-23, after §0.5).** The answer is not a
ranking of methods. It is that **the mechanism dictates the level of the fix**.
On the zero-shot LLM the shortcut is a *lexically-triggered threshold offset*
with no learned correlation behind it, and two free inference-time
interventions remove most of it. Train-time debiasing machinery (PoE, group
DRO, adversarial disentanglement) attacks a learned representation bias that
this model does not have. The encoder arm exists to demonstrate the contrast:
same corpus, same shortcut, a *different* mechanism, and therefore a different
correct intervention.

**The question that decides the paper is RQ3.** `shortcut reliance ↓` does not
imply `construct learning ↑`. A model that has stopped responding to the
interviewer may simply have stopped responding to anything. Every intervention
below is therefore scored on a **paired** criterion — shortcut sensitivity down
*and* construct sensitivity preserved — and the design pre-commits to reading a
drop in both as a failure (§6).

---

## 0. Locked decisions

| # | Decision | Rationale |
|---|---|---|
| **L1** | **Encoders are trainable; LLMs stay inference-only.** MentalBERT / MentalRoBERTa may be fine-tuned freely. MentaLLaMA-chat-7B and Menta are scored by teacher-forced label log-likelihood, never trained. | Partially lifts the `EXPERIMENT_DESIGN.md` no-retraining scope. 110M-param encoders make 3 seeds × 5 methods trivial; the 7B arm stays cheap and reproducible. The split is not a compromise — it *is* an axis (§5.3). |
| **L2** | **Canonical shortcut S₁ = interviewer question selection** (DAIC). Evaluation is DAIC-based throughout. IMHI supplies training volume where a method needs it. | Strongest evidence in the project. IMHI's provenance shortcut is capped at the majority baseline (C1) and has less to fix. |
| ~~**L3**~~ | ~~**Transferable form of S₁ = clinical-vocabulary salience**, instantiated as interviewer questions in DAIC and as self-diagnosis / clinical terms in IMHI.~~ **FALSIFIED 2026-08-24.** | Asserted from A2, never measured — then measured. A clinical-marker lexicon predicts the label in IMHI at **AUC 0.510–0.583** across five binary sources, against the DAIC question-selection probe's **0.773**, and only 10–16 % of posts contain any clinical marker at all. **There is no clinical-vocabulary shortcut in IMHI to transfer from.** Consequence: **E4b is dropped** and the design has **no OOD result**. See `experiments/suite/analysis.md`. |
| **L4** | **AUC and probability-space metrics only. No headline F1 or accuracy.** | House lesson (A1 §1.5, findings.md): both LLMs collapse to near-constant predictions; a constant classifier scores exactly 0.500 balanced accuracy. Threshold metrics are uninterpretable here. The draft's `ΔF1_IID` is replaced by `ΔAUC_IID` throughout. |
| **L5** | **3 seeds for every trained model; LLM arm needs none.** | Teacher-forced scoring is deterministic. Report seed spread as sd, not CI, for the encoder arm; bootstrap CIs over samples for both arms. |
| **L6** | **Workshop scope: 5 encoder methods + 4 inference methods, one shortcut, two model classes.** CAT, PRIFT, and the LR/batch sweep are second tier (§7). | NeurIPS workshop, near-term. The draft's full 10 × models × seeds matrix does not fit and its last four rows are the least motivated by our own findings. |
| **L7** | **A method may only be run on an arm whose mechanism it matches** (§0.5). PoE / group DRO / adversarial are encoder-arm only. Threshold and input-level corrections are the LLM arm. | Added 2026-08-23. The earlier draft of §3–§5 placed train-time methods on the zero-shot arm, which is incoherent: there is no training correlation there to unlearn. |

---

## 0.5 Established before any new run

Computed 2026-08-23 from logprobs already on disk (`runs/a1_*.jsonl`,
`experiments/a2-ellie-counterfactual/results/a2b_*.jsonl`). No GPU. These three
results reframe the design and are why §3–§5 were rewritten.

### (i) The A2 collapse is *entirely* threshold displacement

MentaLLaMA-chat-7B, paired over the same 188 participants, 2000 bootstrap
resamples:

| neutral − clinical accuracy gap | value |
|---|---|
| at the default threshold (`argmax`) | **+0.256** [+0.154, +0.356] |
| after re-thresholding | **+0.008** [−0.032, +0.048] |

Per-arm, re-thresholded accuracy is flat at **0.755–0.761 across every arm**
while the default-threshold accuracy runs 0.750 → 0.495. **97.9 % of the
collapse is threshold shift.** The optimal-threshold CI contains zero: once the
operating point is corrected, the clinically-loaded arm is statistically
indistinguishable from the biographical one.

The A2 bias-shift reading is therefore not merely a characterization — it is
*exhaustive*. Nothing in the weights needs to change to recover the lost
accuracy.

⚠️ **This is an oracle ceiling, not a method.** The optimal threshold is chosen
with reference to test labels and pools all 188. It bounds what any threshold
correction can achieve; it is not itself deployable. Converting it into a
label-free method is experiment **P3** (§4).

### (ii) A1's ablation ladder is already an intervention table

Same numbers, re-read as interventions rather than as diagnostics:

| condition | AUC [95 % CI] | acc@opt | reads as |
|---|---|---|---|
| `full` | 0.663 [0.571, 0.753] | 0.739 | no intervention |
| **`participant_only`** | **0.787** [0.717, 0.852] | **0.777** | **D1 — delete interviewer turns** |
| **`ellie_masked`** | **0.763** [0.686, 0.833] | 0.771 | keep turn structure, delete the words |
| `ellie_only` | 0.546 [0.453, 0.634] | 0.707 | shortcut alone — at the floor |

**Deleting the interviewer's turns at inference buys +0.124 AUC**, label-free
and at zero cost. On this arm D1 is not a naive baseline to be beaten; it
already works.

### (iii) The damage is lexical, not structural

`ellie_masked` keeps every interviewer turn in place and replaces only the
words, and it recovers 0.763 of the 0.787 that full deletion achieves. So the
harm is carried by the **lexical content** of the questions, not by their
presence or position.

This corroborates A2's "keys on surface clinical vocabulary" result from an
independent experiment, and it is what makes the mechanism *lexical trigger →
threshold offset* rather than *evidence integration*.

### Consequence for method selection

| mechanism component | matched intervention level | mismatched |
|---|---|---|
| lexical trigger in S | input-level: delete or mask S | representation-level |
| additive threshold offset | output-level: recalibration | representation-level |
| **no learned correlation** (zero-shot) | — | **PoE, group DRO, adversarial** |

---

## 1. Phase 0 — Freeze the evaluation suite

**This is the highest-priority deliverable and it comes before any method.**
Every intervention is scored on byte-identical test sets. Nothing below may be
regenerated after a method has been run against it.

Notation from `EXPERIMENT_DESIGN.md` §0: input decomposes as `x = (C, S)`,
`C` = participant evidence, `S` = interviewer context, `Y` = `phq8_binary`.

DAIC official splits: **train 107** (77 neg / 30 pos) · **dev 34** (23/11) ·
**test 47** (33/14).

### E1 — IID test

| field | value |
|---|---|
| data | DAIC test split, `cf_none` arm (real transcripts, unmodified) |
| n | 47 participants |
| metric | AUC, bootstrap CI ≥ 1000 resamples |
| purpose | normal performance did not break |
| exists? | ✅ derivable from `a2b_*.jsonl` for the LLM arm; new for encoders |

n = 47 is small and its CI will be wide. **Report dev (34) and test (47)
separately and never tune on test.** For the encoder arm additionally report
5-fold CV over the pooled 188 as a variance-reduced secondary number, clearly
labelled as such.

### E2 — Shortcut-conflict test

Two constructions, both required. They answer different questions and the pair
is more informative than either.

**E2a — natural conflict.** Participants where the A2a question-selection
probe (binary asked/not-asked over 75 prompt IDs, AUC 0.753) disagrees with
gold. Uses each participant's *real* interviewer context.

- Build: refit the A2a probe with leave-one-participant-out CV, take the
  out-of-fold prediction, select `sign(ŝ) ≠ Y`.
- n: to be computed at build time (expect ~45–60 of 188 at AUC 0.753).
- Reports: AUC and P(pred=Yes) on the conflict subset vs the concordant subset.

**E2b — constructed conflict.** Cross the counterfactual arms against gold so
the interviewer context points away from the label.

| cell | interviewer context | n |
|---|---|---|
| gold = No + `cf_depression` | clinically loaded | **133** |
| gold = Yes + `cf_neutral` | biographical | **55** |
| | **total** | **188** |

- Exists ✅ for the LLM arm — already on disk in
  `experiments/a2-ellie-counterfactual/results/a2b_*.jsonl`.
- Metric: `AUC_conflict`, and `P(pred = Yes)` per cell. A model immune to S₁
  scores E2b ≈ E1.

### E3 — Counterfactual pairs (C fixed, S varied)

The A2 design, with the length control its limitations section demands.

| arm | interviewer pool | status |
|---|---|---|
| `cf_neutral` | biographical, 63 tok | ✅ on disk |
| `cf_depression` | clinically loaded, 52 tok | ✅ on disk |
| `cf_empirical` | A2a-derived (high-OR), 34 tok | ✅ on disk |
| **`cf_empirical_matched`** | **A2a-derived, padded to 52 tok** | ⬜ **new — required** |
| **`cf_neutral_matched`** | **biographical, truncated to 52 tok** | ⬜ **new — required** |

**Why the two new arms are not optional.** A2's headline (+0.206) is safe —
the clinical pool is *shorter* than the biographical one, so the length
confound runs opposite to the effect. But the "model keys on the wrong half"
dissociation compares `cf_empirical` (34 tok) against `cf_depression` (52 tok)
and **is** length-confoundable. A2's own analysis calls it *suggestive, not
established*. Token-matching is what promotes it to a result, and it is the
sharpest claim in the project.

Metric: **CF-Sens** (`EXPERIMENT_DESIGN.md` §0), mean over participants of
`|p(Y|C,S₁) − p(Y|C,S₂)|`. Lower is better. Primary contrast is
`cf_depression − cf_neutral`.

### E4 — Generalization under protocol shift

This is the operational test that `findings.md` open question 5 demands. It is
the only way to settle whether question selection is a *shortcut* or
*legitimate clinical evidence*, because ablation cannot distinguish them.

**E4a — leave-one-prompt-cluster-out (within DAIC).** Partition the 75 prompt
IDs into clusters by topic. Train with participants whose interviews are
dominated by cluster *k* held out; test on them. A model reading question
selection as evidence degrades sharply under the shift; a model reading
participant content does not.

**E4b — cross-corpus vocabulary transfer.** Encoder trained on IMHI (L2/L3),
evaluated zero-shot on DAIC E1–E3. Tests whether an intervention against the
clinical-vocabulary shortcut transfers to the interviewer instantiation.
Available IMHI training volume: MultiWD 15.7k, Irf 12.9k, SAD 5.5k,
dreaddit 2.8k, DR 1.0k. Held-out cross-dataset targets: SDCNL (1,844 test),
MultiWD (658 test).

### E5 — Construct-preservation set (S fixed, C varied) ★

**The distinctive contribution, and the direct operationalization of RQ3.**
E3 fixes participant evidence and varies interviewer context. E5 is its exact
mirror: **fix the interviewer context and swap the participant.**

Construction — participant-swap pairs:

```
item(i,j)  =  interviewer turns of participant j  +  participant turns of participant i
gold(i,j)  =  Y_i                    # the label follows the evidence, not the context
```

Sample matched pairs `(i, j)` and `(i', j)` sharing interviewer context `j` but
differing in participant label (`Y_i = 1`, `Y_i' = 0`).

- **Construct sensitivity** `∂ŷ/∂C` := `mean |p(Y | C_i, S_j) − p(Y | C_i', S_j)|`
- Target n: 200 pairs, balanced, drawn within split.
- Interviewer turns are held byte-identical across the pair, so any movement is
  attributable to participant evidence alone.

An ideal intervention drives **CF-Sens (E3) down while holding construct
sensitivity (E5) flat**. A method that lowers both has flattened the model, not
debiased it. §6 makes that a pre-committed failure.

### Suite summary

| set | what varies | n | on disk? |
|---|---|---|---|
| E1 IID | nothing | 47 (dev 34) | LLM ✅ |
| E2a natural conflict | — (subset selection) | ~45–60 | ⬜ probe refit |
| E2b constructed conflict | S vs Y | 188 | LLM ✅ |
| E3 counterfactual | **S**, C fixed | 188 × 5 arms | 3/5 arms ✅ |
| E4a protocol shift | prompt distribution | by cluster | ⬜ |
| E4b cross-corpus | corpus | 1,844 + 658 | ⬜ |
| **E5 construct** | **C**, S fixed | 200 pairs | ⬜ |

---

## 2. Metrics and the success criterion

Five reported quantities. All with bootstrap CIs (≥1000 resamples) and a
`label_shuffle` noise floor, per `EXPERIMENT_DESIGN.md` §0 mandatory controls.

| # | metric | set | direction | definition |
|---|---|---|---|---|
| 1 | `ΔAUC_IID` | E1 | ≈ 0 | AUC(method) − AUC(ERM) |
| 2 | `AUC_conflict` | E2a/E2b | ↑ | AUC on the conflict set |
| 3 | **CF-Sens** | E3 | **↓** | `mean\|p(Y\|C,S₁) − p(Y\|C,S₂)\|` |
| 4 | `worst-group AUC` | E1 ∪ E2 | ↑ | `min_g AUC_g`, `g = (Y, ŝ)` |
| 5 | **construct sens.** | E5 | **flat or ↑** | `mean\|p(Y\|C_i,S_j) − p(Y\|C_i',S_j)\|` |

**Also report the operating point.** A2 established that the damage here is an
additive threshold shift, not a ranking change: AUC moved 0.764 → 0.734 while
P(pred=Yes) moved 0.160 → 0.745. An intervention could fix the threshold and
leave AUC untouched, or vice versa. **Report `P(pred = Yes)` per arm alongside
every AUC** — a method that stabilizes the operating point across E3 arms is
succeeding even if AUC is flat, and reporting AUC alone would hide it.

---

## 3. Arm 1 — Encoder interventions (trainable)

**Purpose, restated 2026-08-23.** This arm is no longer "the same comparison on
a second model". Its job is to establish that **the shortcut has a different
mechanism when the model is trained on the corpus**, and therefore requires a
different class of fix. A MentalBERT fine-tuned on DAIC-107 should absorb the
interviewer cue into its weights as a genuine learned correlation — the thing
PoE and group DRO are built for, and the thing threshold correction will *not*
repair. If that is what we observe, the paper's claim becomes a mechanism→level
mapping rather than a method leaderboard.

Model: **MentalBERT** primary, MentalRoBERTa as replication.
⚠️ Gotcha 18 (`RESULTS.md` §4): both are MLM checkpoints with **no pooler in
the weights** — `pooler.dense.*` is randomly initialised at load. Use
`last_hidden_state[:,0]` or mean pooling; never `pooler_output`.

Transcripts run ~3,800 BERT tokens against a 512 limit. **Do not head-truncate**
— the interviewer shortcut lives mid-interview, where the clinical follow-ups
are asked (Burdisso et al. 2024), so head truncation would delete the very
effect under test. Encode non-overlapping 512-token chunks, mean-pool within
chunk, mean-pool across chunks. Record chunk counts per condition.

Training budget held identical across all five methods: same optimizer, LR
schedule, epochs, early-stopping criterion on dev. 3 seeds each.

### B0-mech — the prerequisite that gates this whole arm

Before any intervention is run, three measurements on the ERM baseline:

| # | measurement | what it decides |
|---|---|---|
| 1 | train on `full`, test on `full` → AUC | does the encoder learn anything at n=107? |
| 2 | train on `ellie_only`, test on `ellie_only` → AUC | is the shortcut learnable by this model class? (TF-IDF reached **0.810**, so the corpus supports it) |
| 3 | `full`-trained model run over the A2 counterfactual arms → CF-Sens, ΔP(pred=Yes), **and acc@opt per arm** | **the decisive one** |

Measurement 3 is the mechanism test. Compute the §0.5(i) statistic on the
encoder: if re-thresholding also erases the encoder's counterfactual collapse,
the mechanism is the *same* as the LLM's and the train-time arm is
unnecessary — a null that would collapse the paper to inference-time methods
and should be reported as such. If re-thresholding does **not** repair it, the
mechanisms differ, and D3/M1/M2 are motivated.

**Pre-commit:** if measurement 1 gives AUC whose CI includes the label-shuffle
floor, n=107 is too small to train anything and the encoder arm is abandoned,
not patched with more epochs.

| ID | intervention | level | what it does here | cost |
|---|---|---|---|---|
| **B0** | **ERM** | baseline | plain fine-tune on DAIC train (107). The denominator for everything. | ★ |
| **D1** | **shortcut removal** | data | drop Ellie turns entirely (= A1 `participant_only`). The naive baseline. Does deleting S suffice? | ★ |
| **D3** | **counterfactual augmentation** ★ | data | augment train with `(C, S′, Y)` pairs built by the E3 machinery — same participant, swapped interviewer pool, label unchanged. Teaches invariance to S directly. | ★★ |
| **M1** | **PoE** ★ | objective | bias model `f_b(S) → Y` **already exists**: the A2a question-selection probe, AUC 0.753, `a2a_prompt_label_association.py`. Train main model on `softmax(z_m + α·z_b)`; gradient vanishes where the bias model already wins. | ★★ |
| **M2** | **Group DRO + regularization** ★ | objective | groups `g = (Y, ŝ)` from the A2a probe, 4 cells. Minimize worst-group loss. | ★★ |

**M2 carries a statistical-power caveat that must be stated, not hidden.**
DAIC train is 107 participants; four groups leaves ~10–35 examples in the
smallest cell. Sagawa et al. (2019) additionally show naive group DRO *fails*
on overparameterised nets without strong regularisation. Therefore M2 runs in
three variants — DRO, DRO + heavy L2, DRO + early stopping — and if the seed
spread swamps the effect, **that is the finding and it gets reported as an
n-limit**, not buried. This is the main argument for routing M2 through E4b
(IMHI training, 15k+ rows) as well.

D1 is expected to underperform D3. That contrast is itself a paper point —
`literature/survey.md` notes both Zhang & Poellabauer (2025) and DepFlow
mitigate by *making the spurious factor explicit and controlling it*, not by
deleting data.

---

## 4. Arm 2 — LLM interventions (inference-only)

Models: **MentaLLaMA-chat-7B** primary. **Menta is reporting-only** — it sits
at p(dep) 0.96–0.98 with zero prediction flips in every A2 arm; it is a
saturated probe and enters as a stated null, never as a contradicting result
(findings.md).

This arm carries the paper's positive result. Per L7 it contains **no
train-time methods** — there is no learned correlation here to unlearn.

| ID | intervention | level | status |
|---|---|---|---|
| **P0** | zero-shot baseline | — | ✅ on disk (`a1_*.jsonl`, `a2b_*.jsonl`) |
| **D1** | **delete interviewer turns** | input | ✅ **measured: AUC 0.663 → 0.787** (§0.5 ii) — needs split discipline only |
| **D1m** | **mask interviewer words, keep turns** | input | ✅ **measured: 0.763** (§0.5 ii) — isolates lexical vs structural |
| **P1** | anti-shortcut prompting | inference | ⬜ cheap — the one untested lever |
| **P2** | CoT | inference | ⬜ cheap |
| **P3-c** | **label-free two-pass recalibration** ★ | output | ✅ **run 2026-08-23 → I1** |
| ~~P3-b~~ | ~~dev-fit per-arm shift~~ | output | ❌ **demoted — redundant with per-arm threshold fitting (I1 §4)** |

D1 and D1m are already-run A1 conditions re-read as interventions. What they
still need is **split discipline** — the §0.5 numbers pool all 188 and use an
oracle threshold. Re-reporting them dev-fit / test-applied is step 1.

**P1** instruction, fixed verbatim before running:
> Base your prediction only on clinically relevant evidence in the
> participant's own words. Do not infer depression from the interviewer's
> choice of questions, from sentiment, or from writing style.

**P2** caveat: CoT requires free generation, which collides with `L4`/the
teacher-forced scoring rule. Resolution — generate the reasoning, then score
the label by teacher-forced log-likelihood **conditioned on the generated
reasoning**. Parse-failure rate is reported per arm; if it correlates with
perturbation strength, the arm is void (`EXPERIMENT_DESIGN.md` §0 rationale).

### P3 — from oracle ceiling to a label-free method

§0.5(i) established the ceiling: re-thresholding erases 97.9 % of the A2
collapse. But that threshold was chosen against test labels. P3's whole job is
to reach that ceiling **without labels at test time**. Three variants, in
increasing order of deployability:

**P3-a — oracle (upper bound, not a method).** Optimal threshold per arm, fit
on test. Already computed: acc@opt flat at 0.755–0.761 across all four arms.
Reported only as the ceiling.

**P3-b — dev-fit shift.** `δ̂(S)` estimated on the 34 dev participants, applied
to the 47 test participants:

```
δ̂(S)  =  mean_dev [ ℓ(C,S) − ℓ(C,S_ref) ],   ℓ = log p(Yes) − log p(No)
score  =  ℓ(C,S) − δ̂(S)
```

Requires knowing which interviewer context applies — observable in the A2 arm
design, only partially so in deployment.

**P3-c — two-pass, fully label-free ★ (the deployable one).** At inference,
score each transcript **twice**: once complete, once with the interviewer turns
stripped. The shift is then estimated *per patient* with no labels and no dev
set at all:

```
δ̂ᵢ  =  ℓ(Cᵢ, Sᵢ) − ℓ(Cᵢ, ∅)          # both passes available at inference
score =  ℓ(Cᵢ, ∅)                      # or  ℓ(Cᵢ,Sᵢ) − δ̂ᵢ, which is identical
```

Note what this degenerates to: **the corrected score just *is* the
participant-only score**, which §0.5(ii) already measured at AUC 0.787 against
`full`'s 0.663. So P3-c and D1 converge, and the two-pass framing explains
*why* D1 works — it is threshold correction by construction.

The open question P3-c makes precise, and which step 1 answers: is the residual
gap between D1 (0.787) and the oracle ceiling worth any further machinery, or
is deleting the interviewer turns simply the whole fix?

**Every input is on disk** — `a1_*.jsonl` and `a2b_*.jsonl` carry
per-participant logprobs with `split`, `gold`, `phq8_binary`. No GPU.

---

## 5. What the design is actually testing

### 5.1 Mapping to the draft's RQs

| RQ | answered by |
|---|---|
| RQ1 do interventions reduce shortcut dependence? | CF-Sens (E3) + `AUC_conflict` (E2) |
| RQ2 data-side vs model-side? | D1/D3 vs M1/M2 |
| **RQ3 is construct signal retained?** | **construct sensitivity (E5)** — the paired criterion |
| RQ4 does it generalize OOD? | E4a protocol shift, E4b cross-corpus |
| RQ5 stable across shortcut/model? | Arm 1 vs Arm 2 (§5.3) |

### 5.2 The screening table

Both arms scored on the identical E1–E5 suite. Methods are grouped by arm
because per L7 they are not interchangeable across arms.

| arm | method | ΔAUC_IID | AUC_conflict | CF-Sens ↓ | worst-group | construct sens. | E4 OOD |
|---|---|---|---|---|---|---|---|
| LLM | P0 zero-shot | — | | | | | |
| LLM | **D1 interviewer removal** | | | | | | |
| LLM | D1m interviewer masking | | | | | | |
| LLM | P1 anti-shortcut prompt | | | | | | |
| LLM | P2 CoT | | | | | | |
| LLM | **P3-c two-pass recalibration** | | | | | | |
| LLM | *P3-a oracle ceiling* | | | | | | |
| Enc | B0 ERM | — | | | | | |
| Enc | D1 removal | | | | | | |
| Enc | D3 CDA | | | | | | |
| Enc | M1 PoE | | | | | | |
| Enc | M2 Group DRO | | | | | | |

### 5.3 The two arms have different mechanisms — that is the result

L1 was chosen for cost, but §0.5 turned it into the paper's spine.

The zero-shot LLM's shortcut is a **lexically-triggered threshold offset** with
no training history: input-level deletion and output-level recalibration fix
it, and they converge on the same correction (§4, P3-c). A DAIC-trained encoder
should instead absorb the cue as a **learned representation bias**, where
threshold correction is expected to fail and PoE / group DRO become the matched
tools.

So the question is not *which method wins* but **whether the level of the
correct intervention is predictable from the mechanism**. B0-mech (§3) is what
tests it, and it can come out either way:

- **Mechanisms differ** → the mapping holds; the paper is a mechanism→level
  result with both arms as evidence.
- **Mechanisms are the same** (re-thresholding also repairs the encoder) → the
  train-time arm is unnecessary, and the paper is the narrower but still clean
  claim that this shortcut is fully correctable at inference for free.

Both outcomes are publishable at workshop length. Neither requires the method
leaderboard the draft assumed.

---

## 6. Falsifiability — pre-committed, before any method runs

Following the `EXPERIMENT_DESIGN.md` convention of pre-committing to how
results get read.

**F-A. The flattening failure (the one that matters).** If a method drives
CF-Sens down **and** construct sensitivity (E5) down by a comparable relative
amount, it has not debiased the model — it has reduced the model's sensitivity
to its input in general. Such a method is reported as a **failure**, however
good its conflict-set number looks. Threshold: a method may claim success only
if `Δ(CF-Sens) / CF-Sens₀` exceeds `Δ(construct sens.) / construct sens.₀` by a
margin whose CI excludes zero.

**F-B. Promotion rule (draft Phase 7).** A method advances to the second round
only if **all four** hold: `AUC_conflict` ↑ · CF-Sens ↓ · construct sensitivity
not significantly down · `ΔAUC_IID` not significantly down. Top 2–3 only.

**F-C. If no method beats ERM on the paired criterion**, that is a publishable
negative result for a workshop, and the write-up says so plainly rather than
promoting the least-bad row. Given that the survey found **almost every
existing method assumes retraining** while our effect is an operating-point
shift, a negative result on the train-time arm alongside a positive P3 is a
live and interesting outcome.

**F-D. Replication boundary.** A1/A2a replicate Burdisso et al. (2024), who
reached 0.90 F1 by intentionally exploiting this shortcut. The paper must cite
them as motivating prior work and claim as new only: (i) the ablation-vs-
counterfactual dissociation on zero-shot models, (ii) the bias-shift
characterization, (iii) whichever interventions survive F-B.

---

## 7. Execution order

Ordered by information gained per GPU-hour. Steps 1–2 need no GPU at all.

| # | step | needs | blocks |
|---|---|---|---|
| **0** | ~~ceiling computation~~ | — | ✅ **done 2026-08-23 → §0.5** |
| **1** | **LLM arm with split discipline** — D1, D1m, P3-b, P3-c dev-fit / test-applied, plus the E2b conflict cells | **nothing — data on disk** | the paper's positive result |
| **2** | **B0-mech** — encoder ERM + the three §3 measurements | 1 GPU, minutes | **gates the entire encoder arm** |
| **3** | Build E2a (probe refit) + E5 pair list + E4a clusters | CPU | the whole suite |
| **4** | Run E3's two token-matched arms (`cf_*_matched`) | 1 GPU, ~1 h | promotes A2's dissociation to a result |
| **5** | Freeze suite E1–E5. **No regeneration after this point.** | — | all methods |
| **6** | P1, P2 on MentaLLaMA | 1 GPU | LLM arm complete |
| **7** | D3, M1, M2 × 3 seeds — **only if step 2 says mechanisms differ** | 1–2 GPUs | encoder screening table |
| **8** | E4a/E4b for surviving methods only | 1–2 GPUs | RQ4 |
| **9** | Apply F-B, promote top 2–3, write up | — | — |

**Step 2 moved ahead of suite construction deliberately.** It is minutes of GPU
time and it decides whether steps 7–8 happen at all. Building the full E1–E5
suite before knowing whether the encoder arm exists would be work done on
spec.

GPUs 0–5 are free; 6–9 are running other users' jobs (`nvidia-smi` before
placing anything — findings.md).

**Second tier, explicitly deferred** (draft M4/M5/M7): CAT needs reliable
span-level causal annotation we do not have; PRIFT is the most promising of the
three and is the first thing to add if time allows, since a latent
shortcut direction is exactly what C1's probe weights already give us; the
LR/batch sweep is cheap but least motivated by our findings. Draft Phase 8
(combining D3 + M2) is out of workshop scope.

---

## 8. Open dependencies

1. **E4a prompt clustering needs a decision** — topical clusters over 75 prompt
   IDs, hand-grouped or by co-occurrence. Hand-grouping is defensible at this
   scale and should be frozen before use, like the A2 pools were.
2. **E5's participant-swap items are synthetic transcripts.** They read as
   coherent interviews only if turn counts roughly match. Sample pairs with
   similar `n_participant_spans` (present in `daic_conditions.jsonl`) and
   report the mismatch distribution.
3. **IMHI clinical-vocabulary shortcut (L3) is asserted, not measured.** Before
   E4b runs, verify it exists — the C1 data on disk (10 sources × 4 conditions,
   `runs/c1_*.jsonl`) should support a quick check.
4. **`RESULTS.md` is stale** (dated 2026-08-19, claims only A1 has run) and
   `findings.md` still says D is running. Reconcile before writing, since both
   are cited as the evidence base above.
