# Supplementary experiments — round 2

Run 2026-09-05. Closes items 1–6 and part of 9 of `VERIFICATION_gaps.md §C6`.
Nine jobs across GPUs 4/5/6/8/9 plus three CPU-only derivations.
Total wall time ≈ 2 h.

| # | experiment | needs GPU | status | artefact |
|---|---|---|---|---|
| 1 | ESConv negative control, committed script | no | **done** | `experiments/h3-esconv-shortcut/{code/audit.py,results/esconv_audit.json}` |
| 2 | MentalBERT interventions, re-run to a safe filename | yes | **done** | `logs/round2/j3_mentalbert.log` |
| 3 | threshold-correction CSS derived offline | no | **done** | `experiments/i11-threshold-offline/results/threshold_offline.json` |
| 3b | threshold proxy refitted inside its support | no | **done** | `.../threshold_refit.json` |
| 4 | `targeted_removal`+`threshold_proxy` on Menta, Qwen3-base | yes | **done** | `experiments/i10-model-sweep/results/suite_{menta,qwen3_base}.json` |
| 5 | test-split-only AUC | no | **done** | `.../threshold_offline.json` |
| 6 | `targeted_removal` CSS, 8 models | yes | **done** | `experiments/i12-targeted-cf/results/*.json` |
| 9a | LEACE over all 32 layers | yes | **done** | `experiments/i7-causal-probing/results/phase3_leace_L1-32.json` |
| 9b | post-erasure downstream recovery | yes | **done** | `experiments/i13-erasure-recovery/results/recovery_L13.json` |
| 7 | `analysis.md` for i10 and i7 | no | **done** | those directories |
| 8 | `mosei_emotion` counterfactual | yes | **not run** | see §7 |

---

## 1. ESConv — the negative control now reproduces

`experiments/h3-esconv-shortcut/code/audit.py`, written from scratch to the same
probe family and metric as the H1/H2 audits (balanced accuracy, TF-IDF 1–2 grams
+ LogisticRegression(class_weight=balanced), 5-fold StratifiedKFold, 3 seeds).

Data notes that the earlier session's numbers did not record: the split key is
`validation`, not `valid` (1300 dialogues = 910/195/195); `problem_type` has 13
raw classes and `emotion_type` 11, with long tails that cannot be 5-fold
cross-validated. Classes are dropped at the natural frequency gap
(problem_type 156 → 28, emotion_type 40 → 13), i.e. n ≥ 30, which yields exactly
the 5 and 7 classes the earlier table reports.

| probe | problem_type (5 cls, chance 0.200) | prior value | emotion_type (7 cls, chance 0.143) | prior value |
|---|---:|---:|---:|---:|
| seeker text (the construct) | **0.780** | 0.780 | **0.344** | 0.345 |
| supporter text | 0.698 | 0.697 | 0.282 | 0.277 |
| **strategy counts + proportions, no text** | **0.216** | 0.220 | **0.144** | 0.150 |
| **strategy sequence n-grams, no text** | 0.221 | 0.203 | 0.171 | 0.149 |
| dialogue length | 0.210 | 0.212 | 0.159 | 0.134 |
| seeker survey ratings | 0.228 | 0.228 | 0.180 | 0.180 |
| stopwords only | 0.418 (37.5 %) | 0.417 (37 %) | 0.230 (43.4 %) | 0.227 (40 %) |
| situation field (pre-dialogue) | 0.806 | 0.812 | 0.352 | — |
| first seeker turn only | 0.354 | 0.351 | 0.248 | — |

**The claim holds and is now reproducible.** Protocol leakage sits at chance:
0.216/0.221 against 0.200, and 0.144/0.171 against 0.143 — versus DAIC's 0.753
against a 0.500 shuffled control. Seven of nine rows land within 0.006 of the
previously quoted values; the two n-gram rows and emotion-type length come out
0.018–0.025 higher on my implementation, still at chance, so no conclusion moves.

The `situation` defect replicates too: the 17-word pre-dialogue field predicts
`problem_type` better (0.806) than the whole seeker dialogue (0.780).

---

## 2. MentalBERT — the paper's numbers are confirmed, the JSON was the problem

Re-run with `--ckpt checkpoints/MentalBERT`, 5 arms × 3 seeds. The MentalRoBERTa
JSON was copied to `encoder_interventions_MentalRoBERTa.json` **before** the run,
so this time nothing was destroyed.

| arm | this run | 2026-08-24 log | paper | match |
|---|---|---|---|---|
| B0 ERM | 0.705 / 0.0413 | 0.722 / 0.0409 | 0.722 / 0.041 | CF-Sens ✓, AUC within seed spread |
| D3 counterfactual augmentation | 0.734 / 0.0332 | 0.734 / 0.0337 | — | ✓ |
| **M1 product-of-experts** | **0.540 / 0.0039** | **0.540 / 0.0039** | **0.540 / 0.004** | **exact** |
| M2 group DRO | 0.659 / 0.0444 | 0.659 / 0.0443 | — | ✓ |
| M2reg group DRO + heavy L2 | 0.692 / 0.0524 | 0.646 / 0.0734 | — | differs, seed spread sd 0.032–0.096 |

**The paper's PoE row is correct.** The overwrite diagnosis in
`VERIFICATION_confirmed.md §A3` is confirmed: the JSON on disk was MentalRoBERTa,
the paper quoted MentalBERT, and MentalBERT reproduces to three decimals.

### New: the missing C-Sens column, and it is the interesting one

The 2026-08-24 MentalBERT run recorded no construct sensitivity. It does now:

| arm | AUC | CF-Sens ↓ | **C-Sens ↑** | SER ↓ |
|---|---:|---:|---:|---:|
| B0 ERM | 0.705 | 0.0413 | 0.2256 | 0.18 |
| D3 counterfactual augmentation | 0.734 | 0.0332 | **0.4342** | 0.08 |
| **M1 product-of-experts** | **0.540** | **0.0039** | **0.0406** | 0.10 |
| M2 group DRO | 0.659 | 0.0444 | 0.2104 | 0.21 |
| M2reg | 0.692 | 0.0524 | 0.3844 | 0.14 |

**PoE cuts shortcut dependence by 91 % and construct sensitivity by 82 % at the
same time.** It is not removing the shortcut; it is removing the model's
responsiveness to everything, and the C-Sens column is the direct evidence — it
was missing when the paper's PoE row was written.

**Counterfactual augmentation is the opposite and is the best arm here**: the only
one that lowers CF-Sens (0.0413 → 0.0332) *while nearly doubling* C-Sens
(0.2256 → 0.4342) and raising AUC (0.705 → 0.734). SER 0.18 → 0.08. On this
encoder it dominates both DRO variants and PoE on all three axes. The paper
currently does not report it.

---

## 3. Threshold correction — CSS derived offline, and the fix

Full detail in the previous version of this file; the results are unchanged.
`|CSS|` improves on 4 of 6 models and worsens on 2, and the **signed** CSS flips
negative on all six: the correction does not remove the shortcut, it overshoots.

Cause: the proxy is fitted where clinical-word density never exceeds **0.0387**
and applied to a pool at **0.1395**, a **3.6× extrapolation**, with fitted slopes
up to 90.09 drawn from correlations of 0.04–0.24.

### The fix, and it needs no GPU

| model | \|CSS\| base | as-implemented | **clipped to support** | refit within support |
|---|---:|---:|---:|---:|
| MentaLLaMA | 0.2050 | 0.0664 | **0.1454** | 0.0639 |
| OSim-8B | 0.1960 | 0.1346 | **0.1526** | 0.1602 |
| OmniSap-SFT | 0.4573 | **0.5969** | **0.3229** | 0.3403 |
| OmniSap-RL | 0.4710 | 0.3731 | **0.3991** | 0.3179 |
| OmniSap-2.0 | 0.5051 | 0.3588 | **0.4469** | 0.3392 |
| HumanOmniV2 | 0.3715 | **0.6562** | **0.1958** | 0.2398 |
| **improves** | — | **4/6** | **6/6** | **6/6** |

**Clamping the feature to its fitted range turns a method that made two of six
models worse into one that improves all six.** Clipping uses only training-set
information, so unlike the refit column it is deployable. The refit column fits
on the very arms it is evaluated on and is an oracle upper bound, not a method.

---

## 4. The 8 × 7 intervention panel is now complete

Menta and Qwen3-4B-base were re-run in full, which both filled the four empty
cells and gave those two models the per-participant `scores` arrays they lacked.
Every previously-present cell reproduced exactly.

| model | delete_int | blank_words | marg_2 | marg_4 | targeted | threshold | instruct |
|---|---|---|---|---|---|---|---|
| MentaLLaMA | +0.0964 | +0.0759 | +0.0769 | +0.0810 | -0.0154 | -0.0009 | +0.0692 |
| Menta | -0.0347 | -0.0392 | -0.0202 | -0.0238 | +0.0107 | +0.0031 | -0.0297 |
| Qwen3-4B-base | -0.0120 | -0.0150 | -0.0161 | -0.0135 | +0.0018 | +0.0021 | -0.0007 |
| OSim-8B | -0.0025 | -0.0134 | +0.0012 | -0.0010 | -0.0008 | +0.0012 | -0.0114 |
| OmniSap-SFT | +0.0183 | +0.0021 | +0.0123 | +0.0132 | -0.0024 | +0.0079 | +0.0127 |
| OmniSap-RL | +0.0260 | +0.0154 | +0.0230 | +0.0165 | +0.0018 | +0.0069 | +0.0218 |
| OmniSap-2.0 | +0.0372 | +0.0260 | +0.0230 | +0.0259 | +0.0029 | +0.0102 | +0.0280 |
| HumanOmniV2 | -0.0115 | -0.0185 | -0.0086 | -0.0116 | -0.0064 | -0.0006 | -0.0038 |
| **mean** | +0.0146 | +0.0042 | +0.0114 | +0.0108 | -0.0010 | +0.0037 | +0.0107 |
| **improves** | 4/8 | 4/8 | 5/8 | 4/8 | 4/8 | 6/8 | 4/8 |

**56 cells, 31 improve, 25 degrade. Still no method improves all eight.**

Two counts change from the audit: `targeted_removal` **2/6 → 4/8** and
`threshold_proxy` **4/6 → 6/8**, because both new models happen to be helped.
`threshold_proxy` at 6/8 is now the widest-coverage method in the panel, ahead
of `marginalise_2` (5/8) — but its per-model effects are an order of magnitude
smaller (+0.001 to +0.010 against delete's +0.096), and §3 shows its CSS
behaviour is unsound as implemented.

---

## 5. Targeted removal — CSS at last, and it does not work

`experiments/i12-targeted-cf/`, all 8 models. Both arms are built from the same
clinical-stripped transcript with the two fixed pools interleaved, so the
participant's words are byte-identical across arms.

**The internal control is what matters.** Only **123 of 188** transcripts carry
the `Ellie: <id> (text)` prompt IDs the strip regex matches; on the other **65**
the intervention is a no-op. Those 65 are a within-experiment control that needs
no separate baseline.

| model | baseline \|CSS\| | stripped (n=123) | no-op control (n=65) | difference | reduced? |
|---|---:|---:|---:|---:|---|
| OmniSap-2.0 | 0.5051 | 0.0289 | 0.0916 | -0.0627 | yes |
| OmniSap-RL | 0.4710 | 0.0479 | 0.0925 | -0.0446 | yes |
| OmniSap-SFT | 0.4573 | 0.0825 | 0.1098 | -0.0274 | yes |
| HumanOmniV2 | 0.3715 | 0.0476 | 0.0911 | -0.0436 | yes |
| Qwen3-4B-base | 0.2561 | 0.0276 | 0.0405 | -0.0129 | yes |
| MentaLLaMA | 0.2050 | 0.1336 | 0.1217 | +0.0119 | **no** |
| OSim-8B | 0.1960 | 0.1013 | 0.0821 | +0.0192 | **no** |
| Menta | 0.0267 | 0.0789 | 0.0431 | +0.0358 | **no** |

**Three of eight models show CSS going the wrong way, five show a reduction.**
Not universal, and it fails on exactly the model the paper's headline rests on:
MentaLLaMA's stripped subset is **higher** than its no-op control
(0.1336 vs 0.1217). This is the CSS-level evidence for §559's "surgical removal
does not work", which previously rested on an AUC delta alone — and that delta
was quoted against the wrong baseline (`VERIFICATION_confirmed.md §A2`).

**Do not compare these to the 0.196–0.505 baseline column by subtraction.** The
baseline protocol replaces the interviewer entirely with the pool; this one keeps
the real interviewer minus its clinical questions and adds the pool on top, which
dilutes the pool's share of the context. The 123-vs-65 contrast is the valid
comparison; the baseline column is there only to identify each model.

---

## 6. Where the shortcut lives: erasing everywhere buys nothing, because it is rebuilt

### 6a. LEACE over all 32 layers

| erased band | shortcut sens | vs none | C-sens | AUC |
|---|---:|---:|---:|---:|
| none | 0.2057 | 1.00× | 0.1765 | 0.777 |
| layer 13 only | 0.1928 | 0.94× | — | — |
| layers 8–24 (17 layers) | 0.1580 | **0.77×** | 0.1745 | 0.766 |
| **layers 1–32 (all 32)** | **0.1589** | **0.77×** | 0.1742 | 0.760 |
| control, all 32, shuffled labels | 0.2154 | 1.05× | 0.1878 | 0.788 |

**Doubling the erased band from 17 layers to all 32 changes nothing**
(−23.2 % → −22.8 %, within noise). This **rules out redundancy outside the band**
as the explanation for the residual 77 %: there is nothing in layers 1–7 or
25–32 that a wider erasure picks up.

### 6b. The direction is recomputed within about four blocks

`experiments/i13-erasure-recovery/`. Erase at one layer, then probe Cohen's d at
every layer downstream, cross-fitted exactly as in `layerwise_multi.py`.

**An indexing fact that has to be stated first.** In this transformers version
`hidden_states[L]` is a *pre-hook* snapshot: hooking `model.model.layers[k]`
leaves `hidden_states[k+1]` byte-identical and first changes
`hidden_states[k+2]`. Verified directly by zeroing block 12's output — `hs[13]`
was unchanged and `hs[14]` onward went to zero norm. So with `--erase-layer 13`
(which hooks `layers[12]`) the first observable erased state is **hs[14]**, and
the apparent "d unchanged at the erased layer" is a snapshot artefact, not a
failed erasure.

Reading from hs[14]:

| observed layer | d, no erasure | d, erased at block 12 | fraction surviving |
|---|---:|---:|---:|
| hs[13] (pre-hook snapshot) | 2.305 | 2.305 | — not yet erased |
| **hs[14]** — first post-erasure | 1.933 | **0.905** | **0.47** |
| hs[15] | 2.092 | 1.222 | 0.58 |
| hs[16] | 1.968 | 1.762 | 0.90 |
| hs[17] | 2.427 | 2.131 | 0.88 |
| hs[18] | 2.315 | 2.384 | **1.03** |
| hs[19]–hs[31] | ≈2.4 | ≈2.4 | 0.97–1.05 |
| hs[32] final | 2.240 | 2.111 | 0.94 |

**The model rebuilds the direction within about four blocks and carries it,
undiminished, to the output.** By hs[18] the erasure is completely undone.

Together with 6a this settles the question the audit left open. The residual
77 % is not redundancy outside the erased band (6a rules that out) and it is not
that the erasure fails at its own layer (it removes 53 % there under this
readout). **It is recomputation**: the shortcut is regenerated from the
interviewer tokens at every block, so a hook that fires once per block is
outrun by reconstruction happening inside the next one.

**What this licenses the paper to say — and not say.** It supports
"re-instantiated along the residual stream layer after layer"; it does **not**
support the word *non-linear*. A linear eraser still removes 53 % of a linear
probe's signal at the layer it acts on, and the linear probe reads the concept
at AUC 0.94–0.99 everywhere. What is non-linear here, if anything, is the
*computation* that rebuilds the direction, not the *representation* of it.

Two caveats: erasure at hs[14] removes 53 % of d, not 100 %, because the eraser
is fitted on per-token states while the probe reads mean-pooled ones, so LEACE's
guarantee does not transfer exactly to this readout. And the layer labels in
`phase3_leace_*.json` are offset by one relative to the probe layers in
`layerwise_multi.json` for the same reason — this affects only which layers are
*named*, not the behavioural erasure numbers, which come from re-scoring the
model end to end.

---

## 7. Not run, and why

**`mosei_emotion` counterfactual.** Skipped deliberately. Unlike the other eight
items it closes no gap in the paper: it tests a *prediction* of the granularity
account (a predicted null on a 92 %-stopword corpus) rather than correcting a
claim. It also needs a counterfactual arm design that does not exist —
mmpsy/ptsd/Irf each got a bespoke manipulation, and inventing a fourth would
have consumed the GPU budget the six correcting experiments needed. It remains
the natural next experiment.

**Numeric C-Sens under threshold correction.** The invariance proof stands
(the E5 pair shares its interviewer text, so the same constant is subtracted from
both log-odds and the paired contrast is exactly invariant). The probability-space
number still needs 2 × 188 re-scores per model because `run_full_suite.py` keeps
the E5 arrays in a local variable and stores only the scalar.

---

## 8. Two operational notes

A stray `pkill -f chain_recovery.sh` matched this session's own shell and killed
the command that was about to launch the recovery job. Nothing was lost — the
job had not started and the three running jobs were untouched — and it was
relaunched. The same self-match class of bug had already stalled the first
dispatch attempt, where three lane scripts waited on `pgrep -f` patterns that
appeared in their own argv and would never have exited; GPU 9 idled about
fifteen minutes before this was caught. Both are fixed (wait on PIDs with
`kill -0`), but the pattern is worth remembering.

GPUs 0, 2 and 7 all fail `torch.zeros(8, device='cuda')` on this machine, not
just GPU 2 as previously recorded. Usable set is 4, 5, 6, 8, 9 with 1 and 3 held
by other users.
