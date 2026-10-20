# Handoff — camera-ready revision, NeurIPS 2026 Workshop GenAI4Health (Submission 241)

Written 2026-10-19 for a conversation with no prior context. Assume the reader
has been given the submitted paper; this document covers everything that is
**not** in the paper — repository layout, what has already been verified, what
the reviews actually demand, and the traps this project has already fallen into.

**Companion artifact (the prioritised plan, same content in browsable form):**
https://claude.ai/code/artifact/265ab71b-2528-44bc-b86b-f74264a96a76

> A URL of the form `claude.ai/artifact/5jhSYV4M86R58Dc5uzgXxZ` was mentioned in
> the handoff request. It does not appear in this account's artifact list and is
> not the page produced here. Use the link above.

---

## 1. Where things stand

**The paper was accepted as a poster** (reviewers HFWs = 7, HUFa = 6, avg 6.5;
AC 1nCE recommended accept). Decision posted 2026-10-03. **No camera-ready
deadline had been posted on OpenReview as of 2026-10-19 — check before planning.**

The AC's summary of required work: *"The concerns about intervention
comparability, training-data overlap, and model/measurement choices are
addressable through clarification and tempered claims."* **No reviewer asks for a
new experiment to rescue a conclusion.**

Current state: **analysis complete, writing not started.** Two reviewer concerns
that could not be answered by rewording were verified against code and
checkpoints (section 3). Everything else is drafting.

Repository root: `/data/cl195/NeurIPS_workshop`. Paper source: `paper/main.tex`,
`paper/refs.bib`. Figures: `figures/` (see section 6).

---

## 2. The five work items

Full drafted replacement text for each lives in the artifact linked above.

| # | Item | Raised by | Type of work |
|---|---|---|---|
| 1 | Calibrate intervention strength; temper the cross-dataset claim | HFWs, AC | writing + small CPU计算 |
| 2 | Disclose the OmniSapiens / HBA training overlap | HFWs, AC | writing + one figure edit |
| 3 | Justify teacher-forced Yes/No scoring; surface the CoT result | HUFa, AC | writing only |
| 4 | Defend `Qwen3-4B-base` rather than drop it | HUFa, AC | writing only |
| 5 | Larger / closed-source models | HUFa, AC | **author decision** |

**Recommended order.** Items 2, 3 and 4 all land in `\section{Setup}` and are
mutually independent — do them in one sitting, no code required. Item 1 touches
the paper's main argument and should come after. Item 5 last.

### Item 5 is the only open decision

Either (a) run one larger open model on DAIC-WOZ only — the counterfactual
protocol is ~750 forward passes, well under an hour on one free GPU — or
(b) state the scope limit. The limitation is worth writing either way, because
it is real: layer-wise probing and concept erasure need hidden states, and the
counterfactual protocol needs per-item label log-likelihoods, which most
commercial APIs do not expose.

---

## 3. Two verifications already done — do not redo these

### 3.1 OmniSapiens models were trained on Human Behavior Atlas. CONFIRMED.

`checkpoints/OmniSapiens-SFT/label_scheme.json` carries seven supervised domain
heads: `mental_health_depression`, `mental_health_ptsd`,
`mental_health_anxiety`, `emotion`, `sentiment_intensity`, `humour`, `sarcasm`.
These map one-for-one onto HBA's task families, and HBA-test contains both
`daicwoz` and `ptsd_in_the_wild`. The model card lists
`datasets: keentomato/human_behavior_atlas`; its prose says only "Evaluated on",
so the card alone is ambiguous — **the classification heads are the decisive
evidence**, since evaluation does not produce them.

- **Costs us:** the PTSD column is confounded. Every model that moves on PTSD is
  HBA-trained (OmniSapiens ×3, HumanOmniV2); none of the others move.
- **Does not cost us:** DAIC-WOZ. MentaLLaMA (+0.205) and Qwen3-4B-base (+0.256)
  both show substantial reliance and neither is HBA-trained.
- **Do not overclaim:** the card supports exposure at the *task-family* level,
  not that DAIC-WOZ specifically was in the training mix.
- Note our scoring used the LM head throughout, never the classification heads.

### 3.2 Interventions differ in kind and magnitude — but the direction is the opposite of what the reviewer assumed. CONFIRMED.

HFWs suspected DAIC's effect is large because its manipulation is the most
salient. By surface magnitude the ordering reverses:

| Dataset | Manipulation | Construction | Magnitude | Effect |
|---|---|---|---|---|
| DAIC-WOZ | five scripted questions swapped | interleave | **38 words = 2.6 % of a ~1490-word prompt** | moves all 8 models |
| mmpsy | one chunk split/joined at punctuation | re-segment | 2 of ~7 boundaries = 28.6 % | moves nothing |
| PTSD | register passage swapped | **prepend** (`pools[arm] + " " + own`) | not yet measured | moves only HBA-trained models |
| Irf | register block swapped | **prepend** | not yet measured | weak, often reversed |

**DAIC's intervention is the smallest of the four and produces the largest
effect**, which supports the availability-vs-reliance claim rather than
undermining it. But the reviewer is right that this was never measured, and that
PTSD/Irf *prepend* while DAIC *interleaves* — a construction difference the paper
never flagged.

**Still to compute** (CPU, seconds): the word count and prompt share of the
PTSD and Irf prepended passages, to fill the two empty cells above. Arm builders
are in `experiments/h5-ptsd-register-cf/code/run_h5.py` and
`experiments/h6-irf-register-cf/code/run_h6.py`.

---

## 4. Four self-audit fixes, unrelated to the reviews, for the same revision

These came from our own verification pass, not from reviewers. All are
corrections to numbers already in the paper.

1. **CSS is computed two different ways.** `E[|Δ|]` in the DAIC sweep
   (`run_full_suite.py:160`), `E[Δ]` cross-dataset
   (`h7-full-matrix/code/build_matrix.py:83`). They disagree materially:
   **Menta's DAIC effect is negative** — signed −0.0147, absolute 0.0267, with
   130 of 188 participants moving the opposite way. Pick one definition for the
   whole paper (signed is recommended) and add a "fraction moving positive"
   column so split populations stay visible.
2. **Encoder training-level arms were rerun and the numbers changed.** The bias
   expert had been fitted on all 123 ID-bearing transcripts including test (it
   reached AUC 0.963 on test — memorisation). Refitted on train only and raised
   from 3 to 10 seeds: **Group DRO becomes indistinguishable from ERM**
   (ΔAUC +0.014, t = 0.43), and the old "DRO costs 4.6 points, L2 recovers half"
   narrative was seed noise and must be withdrawn. **Counterfactual augmentation
   is the only method that works** (ΔAUC +0.075, t = 2.89; C-Sens 0.254 → 0.474).
   New numbers: `experiments/i6-encoder-interventions/results/fix_*.json`;
   full write-up `to_human/ENCODER_RERUN_RESULTS.md`.
3. **GRPO baseline footnote.** Its CSS is 0.2864 where every other MentaLLaMA
   baseline is ~0.205. Cause verified by code reading: the GRPO evaluation
   inherits the RL input budget — **1800 words, truncating 61/188 transcripts**
   — against the shared sweep's 3868 tokens, truncating 17/188. Prompt template,
   pools, n, checkpoint and scoring rule are all identical. One sentence of
   explanation is the correct fix; the data is sound.
4. **Mitigation table quotes two rows against the wrong baseline.** Targeted
   removal is **−0.034 against its own n=123 baseline**, not the +0.002 implied
   by comparing to the n=188 baseline — the sign flips. Threshold correction
   should use the n=47 test-split figure. Detail in
   `to_human/VERIFICATION_confirmed.md §A2`.

---

## 5. Supporting documents already written

| File | Contents |
|---|---|
| `to_human/VERIFICATION_confirmed.md` | Full audit: confirmed inconsistencies between paper and code, with file:line evidence |
| `to_human/VERIFICATION_gaps.md` | What was never run, what cannot be recovered without new inference |
| `to_human/VERIFICATION_new_results.md` | Results of the nine supplementary experiments run to close those gaps |
| `to_human/MITIGATION_IMPLEMENTATION_AUDIT.md` | Code-level audit of all five mitigation families; flags 14 statements that would mislead if generalised |
| `to_human/ENCODER_RERUN_RESULTS.md` | The 10-seed leak-free encoder rerun, all 60 runs |
| `experiments/i10-model-sweep/analysis.md`, `experiments/i7-causal-probing/analysis.md` | Written analyses for the two experiments that carry the most paper-critical numbers |
| `figures/README.md` | How the figures are generated, deviations from spec, suggested captions |
| `figures/captions.tex` | Three `figure` environments, paste-ready |

---

## 6. Figures — finished, one small edit pending

`figures/figure{2,3,4}_*.{pdf,svg,png}`, generated by
`figures/code/make_figures.py`. **Every value is recomputed from raw experiment
output at plot time**, never transcribed, so figures cannot drift from the text.

Authored at **5.5 in** to match `neurips.sty`'s `textwidth=5.5in`
(`\documentclass{article}`, single column). `\includegraphics[width=\textwidth]`
places them 1:1. **Do not re-author at 7.2 in** — an earlier draft did, and
`width=\textwidth` silently shrank every label by 0.764×.

Pending: add a `‡` mark to the four HBA-trained models in Figure 2, per item 2.
`make_figures.py --no-titles` drops Figure 2's in-figure title if space is tight.

**Build note:** `paper/Makefile` sets `FIGURES_FOLDER := figures`, resolving to
`paper/figures/`, but the figures live one level up. Symlink
(`ln -s ../figures paper/figures`) or point `\graphicspath` at the real location.

---

## 7. Traps this project has already hit — read before running anything

### Operational

- **`pgrep -f` / `pkill -f` match the issuing shell's own command line.** This
  bit the project three times: it killed the running shell twice (exit 144), and
  it stalled three dispatch scripts whose wait-loops matched their own argv, so a
  GPU idled ~15 minutes before anyone noticed. **Wait on PIDs with `kill -0`;
  kill by PID.** If you must match a pattern, include the interpreter path
  (`sc-shortcut/bin/python experiments/...`), which a wrapper shell does not have.
- **GPUs 0, 2 and 7 on this machine fail `torch.zeros(8, device='cuda')`.** Test
  each GPU before dispatching. Other users share GPUs 1 and 3; check free memory
  rather than assuming.
- **Hardcoded output paths cause silent data loss.** A MentalRoBERTa run
  overwrote `encoder_interventions.json` with the MentalBERT results still cited
  in the paper; those survived only in a log file. Scripts now take `--out`.
  Back up before re-running anything that writes a fixed filename.
- **`/`, `$HOME` and `/tmp` have no free space.** All scratch work goes under
  `/data/cl195`. A heredoc written to `/tmp` once produced an empty job list and
  a launcher silently started nothing.
- **A heredoc on a command consumes stdin**, so `curl ... | python - <<'PY'`
  silently feeds the script no data. Write to a file first.

### Analytical

- **Do not infer process from outcome asymmetry.** "Right on positives, wrong on
  negatives" is exactly what one *consistent* shortcut rule produces. In DAIC all
  188 participants move in the same direction with near-identical magnitude
  (+0.183 positives, +0.215 negatives), yet all 79 correct→wrong flips are
  negative-class. One rule, asymmetric outcomes.
- **CSS = 0 for delete / mask / marginalise is hard-coded**
  (`run_full_suite.py:197-200`), not measured. Those methods remove the channel
  the metric swaps, so zero is definitional. Only `instruct_ignore` has a measured
  CSS — and it *rises* on 4 of 8 models.
- **Three seeds cannot separate these encoder arms.** Seed SD reaches 0.078 on a
  47-item test set; two arms each had one collapsed seed out of three. Any rerun
  needs ≥10 seeds or it produces numbers no one can interpret.
- **LEACE is a linear method.** That it removes 23.2 % of the behavioural effect
  is evidence the encoding is substantially *linear*, not non-linear. An earlier
  draft wrote "re-instantiated non-linearly"; that claim is not supported and was
  withdrawn. `paper/main.tex` already states the correct limitation — that a
  non-linearly encoded shortcut would be invisible to this analysis.
- **The encoder baseline has almost nothing to mitigate.** Its CF-Sens is 0.036
  against MentaLLaMA's 0.205. Null results from the training-level arms are
  results from a model that barely uses the shortcut, not evidence a method fails
  in general. The original script pre-registered this concern in its docstring.

### Process

- **"Check this" means read the code.** A request to verify the GRPO baseline
  discrepancy was answered correctly by reading two scripts; a GPU job was then
  launched unnecessarily to confirm it and had to be killed. Trace the code
  first and only escalate to compute if the code is genuinely ambiguous.
- **Figure annotations carry semantics from position.** On a diverging bar chart
  horizontal position encodes sign. Labels once got placed into whichever
  quadrant had whitespace, putting "4/8 improve" on the positive (worsening)
  side — the bars were right, the figure read as mislabelled.

---

## 8. Immediate next action

Draft the three `\section{Setup}` paragraphs (items 2, 3, 4). They are
independent of each other, require no code, and close two and a half of the four
reviewer concerns. Replacement text is drafted in the artifact.

In parallel and non-blocking: add the `‡` marks to Figure 2, and compute the
PTSD/Irf prepend magnitudes to finish the item-1 calibration table.
