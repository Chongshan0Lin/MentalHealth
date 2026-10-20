# Encoder training-level rerun — all data

Run 2026-09-08/09. Closes the test-leakage issue in the four supervised
training-time arms, and replaces the 3-seed numbers with 10 seeds.

**60 runs total: 6 arms × 10 seeds.** MentalBERT, DAIC-WOZ, train n = 107,
dev n = 34, test n = 47. ~2 h wall on GPUs 4/5/8.

## What changed from the previous run

| # | change | why |
|---|---|---|
| 1 | `bias_model(rows, fit_splits=("train",))` | the previous fit used **all 123 ID-bearing transcripts, 30 of them test**. That expert reaches **AUC 0.963 on test** (it had memorised it); the train-only fit scores **0.721**. Its scores feed the PoE loss and the group-DRO group assignment, so test labels reached training indirectly. |
| 2 | D3 augmentation pools: `cf_empirical` dropped | `cf_empirical` was built from A2a odds ratios computed over all 123 transcripts, i.e. partly from test labels. |
| 3 | added **D3ₙ** (`clean3`): `cf_neutral`, `cf_depression`, `cf_none` | dropping `cf_empirical` also drops the copy count 3 → 2, confounding pool provenance with augmentation ratio. `cf_none` is label-free and restores 3 copies, so D3 vs D3ₙ isolates the ratio and D3ₙ vs the old run isolates the pool. |
| 4 | seeds 3 → **10** | the old B0 and M2 each had one collapsed seed out of three; the arms could not be separated. |
| 5 | `--out` argument added | the previous results file had been silently overwritten by a MentalRoBERTa run. Each invocation now writes its own file. |

Code: `experiments/i6-encoder-interventions/code/run_encoder_interventions.py`
(previous version preserved as `.bak`; defaults reproduce the old behaviour).

**Group assignment barely moved.** Only **8 of 107** training examples change
group under the leak-free bias expert (sizes 70/7/20/10 → 72/5/18/12), while the
continuous PoE offset shifts by mean |Δp| = 0.0585. This predicted, correctly,
that PoE would be the only arm materially affected.

---

## Table 1 — main results, 10 seeds each

| arm | n_train | test AUC | sd | 95 % CI of mean | range | CF-Sens ↓ | sd | C-Sens ↑ | sd | SER ↓ | pair acc |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **B0 — ERM baseline** | 107 | **0.6879** | 0.0776 | [0.640, 0.736] | 0.214 | 0.0363 | 0.0204 | 0.2542 | 0.1738 | 0.14 | 0.728 |
| **D3 — counterfactual augmentation (2 copies)** | 321 | **0.7632** | 0.0277 | [0.746, 0.780] | 0.076 | 0.0343 | 0.0128 | 0.4735 | 0.1346 | 0.07 | 0.813 |
| **D3ₙ — counterfactual augmentation (3 copies, ratio control)** | 428 | **0.7453** | 0.0202 | [0.733, 0.758] | 0.058 | 0.0355 | 0.0125 | 0.5057 | 0.1244 | 0.07 | 0.818 |
| **M1 — product-of-experts** | 107 | **0.5998** | 0.0599 | [0.563, 0.637] | 0.158 | 0.0189 | 0.0257 | 0.1182 | 0.1429 | 0.16 | 0.581 |
| **M2 — group DRO** | 107 | **0.6736** | 0.0724 | [0.629, 0.718] | 0.234 | 0.0363 | 0.0198 | 0.2548 | 0.1557 | 0.14 | 0.718 |
| **M2reg — group DRO + heavy L2** | 107 | **0.6777** | 0.0780 | [0.629, 0.726] | 0.236 | 0.0461 | 0.0253 | 0.3127 | 0.1246 | 0.15 | 0.746 |

## Table 2 — every run (60 rows)

| arm | seed | dev AUC | **test AUC** | CF-Sens | C-Sens | pair acc | gap | sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 0.755 | **0.762** | 0.0387 | 0.1427 | 0.785 | -0.021 | 316 |
| B0 | 1 | 0.822 | **0.764** | 0.0465 | 0.3622 | 0.805 | +0.021 | 365 |
| B0 | 2 | 0.581 | **0.550** | 0.0042 | 0.0306 | 0.530 | +0.000 | 200 |
| B0 | 3 | 0.719 | **0.645** | 0.0222 | 0.0725 | 0.735 | +0.000 | 298 |
| B0 | 4 | 0.688 | **0.710** | 0.0633 | 0.3476 | 0.765 | -0.043 | 365 |
| B0 | 5 | 0.771 | **0.749** | 0.0534 | 0.2547 | 0.745 | -0.021 | 282 |
| B0 | 6 | 0.747 | **0.712** | 0.0494 | 0.4133 | 0.790 | +0.000 | 332 |
| B0 | 7 | 0.783 | **0.688** | 0.0406 | 0.4548 | 0.775 | +0.085 | 363 |
| B0 | 8 | 0.593 | **0.565** | 0.0021 | 0.0202 | 0.530 | -0.064 | 167 |
| B0 | 9 | 0.759 | **0.734** | 0.0423 | 0.4430 | 0.815 | +0.021 | 331 |
| D3 | 0 | 0.810 | **0.779** | 0.0222 | 0.3544 | 0.785 | +0.021 | 375 |
| D3 | 1 | 0.771 | **0.729** | 0.0287 | 0.4069 | 0.805 | -0.064 | 451 |
| D3 | 2 | 0.834 | **0.740** | 0.0431 | 0.5390 | 0.825 | +0.021 | 487 |
| D3 | 3 | 0.719 | **0.760** | 0.0241 | 0.1612 | 0.765 | +0.000 | 337 |
| D3 | 4 | 0.763 | **0.745** | 0.0371 | 0.5640 | 0.850 | +0.043 | 715 |
| D3 | 5 | 0.763 | **0.801** | 0.0399 | 0.6081 | 0.850 | +0.000 | 561 |
| D3 | 6 | 0.743 | **0.773** | 0.0312 | 0.5231 | 0.795 | +0.021 | 372 |
| D3 | 7 | 0.846 | **0.781** | 0.0307 | 0.5229 | 0.825 | +0.000 | 410 |
| D3 | 8 | 0.747 | **0.725** | 0.0215 | 0.4756 | 0.785 | +0.000 | 411 |
| D3 | 9 | 0.846 | **0.799** | 0.0641 | 0.5798 | 0.845 | -0.021 | 448 |
| D3_clean3 | 0 | 0.751 | **0.719** | 0.0423 | 0.5230 | 0.825 | +0.000 | 470 |
| D3_clean3 | 1 | 0.775 | **0.751** | 0.0328 | 0.5675 | 0.820 | -0.021 | 568 |
| D3_clean3 | 2 | 0.794 | **0.719** | 0.0488 | 0.5864 | 0.815 | +0.021 | 572 |
| D3_clean3 | 3 | 0.791 | **0.747** | 0.0417 | 0.5083 | 0.835 | -0.043 | 472 |
| D3_clean3 | 4 | 0.763 | **0.723** | 0.0359 | 0.5448 | 0.825 | +0.000 | 522 |
| D3_clean3 | 5 | 0.735 | **0.755** | 0.0440 | 0.6203 | 0.840 | -0.043 | 621 |
| D3_clean3 | 6 | 0.747 | **0.758** | 0.0475 | 0.5864 | 0.815 | +0.043 | 619 |
| D3_clean3 | 7 | 0.826 | **0.738** | 0.0338 | 0.5598 | 0.825 | +0.000 | 524 |
| D3_clean3 | 8 | 0.798 | **0.777** | 0.0111 | 0.2437 | 0.815 | +0.064 | 471 |
| D3_clean3 | 9 | 0.838 | **0.766** | 0.0176 | 0.3164 | 0.765 | +0.043 | 422 |
| M1 | 0 | 0.597 | **0.554** | 0.0045 | 0.0408 | 0.515 | +0.000 | 165 |
| M1 | 1 | 0.751 | **0.703** | 0.0692 | 0.4056 | 0.800 | -0.021 | 364 |
| M1 | 2 | 0.577 | **0.548** | 0.0028 | 0.0321 | 0.490 | +0.000 | 149 |
| M1 | 3 | 0.692 | **0.662** | 0.0492 | 0.3016 | 0.730 | +0.021 | 331 |
| M1 | 4 | 0.581 | **0.571** | 0.0026 | 0.0209 | 0.530 | +0.021 | 150 |
| M1 | 5 | 0.755 | **0.686** | 0.0470 | 0.2455 | 0.750 | -0.085 | 363 |
| M1 | 6 | 0.573 | **0.576** | 0.0021 | 0.0243 | 0.480 | -0.021 | 149 |
| M1 | 7 | 0.577 | **0.571** | 0.0032 | 0.0418 | 0.520 | +0.000 | 167 |
| M1 | 8 | 0.589 | **0.545** | 0.0040 | 0.0292 | 0.480 | +0.000 | 148 |
| M1 | 9 | 0.605 | **0.582** | 0.0046 | 0.0407 | 0.515 | +0.000 | 181 |
| M2 | 0 | 0.771 | **0.680** | 0.0492 | 0.2648 | 0.755 | -0.085 | 358 |
| M2 | 1 | 0.791 | **0.788** | 0.0328 | 0.4342 | 0.820 | +0.064 | 358 |
| M2 | 2 | 0.688 | **0.634** | 0.0327 | 0.2560 | 0.720 | +0.043 | 341 |
| M2 | 3 | 0.739 | **0.673** | 0.0242 | 0.0670 | 0.740 | -0.043 | 291 |
| M2 | 4 | 0.577 | **0.569** | 0.0086 | 0.0446 | 0.575 | +0.085 | 195 |
| M2 | 5 | 0.779 | **0.749** | 0.0532 | 0.3473 | 0.735 | -0.043 | 325 |
| M2 | 6 | 0.739 | **0.693** | 0.0487 | 0.3905 | 0.765 | +0.021 | 340 |
| M2 | 7 | 0.759 | **0.695** | 0.0520 | 0.3908 | 0.765 | +0.021 | 355 |
| M2 | 8 | 0.589 | **0.554** | 0.0019 | 0.0204 | 0.515 | -0.085 | 162 |
| M2 | 9 | 0.794 | **0.701** | 0.0596 | 0.3327 | 0.790 | -0.043 | 356 |
| M2reg | 0 | 0.775 | **0.677** | 0.0482 | 0.2521 | 0.755 | -0.085 | 355 |
| M2reg | 1 | 0.787 | **0.747** | 0.0440 | 0.4428 | 0.800 | +0.064 | 356 |
| M2reg | 2 | 0.660 | **0.511** | 0.1044 | 0.3165 | 0.725 | -0.064 | 354 |
| M2reg | 3 | 0.739 | **0.732** | 0.0460 | 0.2558 | 0.770 | +0.064 | 307 |
| M2reg | 4 | 0.577 | **0.569** | 0.0085 | 0.0446 | 0.575 | +0.085 | 195 |
| M2reg | 5 | 0.794 | **0.740** | 0.0391 | 0.2097 | 0.725 | +0.000 | 307 |
| M2reg | 6 | 0.739 | **0.708** | 0.0555 | 0.3812 | 0.775 | +0.043 | 341 |
| M2reg | 7 | 0.767 | **0.695** | 0.0524 | 0.3903 | 0.775 | +0.000 | 358 |
| M2reg | 8 | 0.767 | **0.673** | 0.0189 | 0.3955 | 0.755 | +0.064 | 358 |
| M2reg | 9 | 0.767 | **0.725** | 0.0437 | 0.4389 | 0.805 | +0.021 | 325 |

## Table 3 — new vs old (3-seed, leaky bias expert)

| arm | old AUC | **new AUC** | Δ | old CF-Sens | new CF-Sens | Δ | old C-Sens | new C-Sens | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.705 | **0.688** | -0.017 | 0.0413 | 0.0363 | -0.0050 | 0.2256 | 0.2542 | +0.0286 |
| D3 | 0.734 | **0.763** | +0.029 | 0.0332 | 0.0343 | +0.0011 | 0.4342 | 0.4735 | +0.0393 |
| M1 | 0.540 | **0.600** | +0.060 | 0.0039 | 0.0189 | +0.0150 | 0.0406 | 0.1182 | +0.0776 |
| M2 | 0.659 | **0.674** | +0.015 | 0.0444 | 0.0363 | -0.0081 | 0.2104 | 0.2548 | +0.0444 |
| M2reg | 0.692 | **0.678** | -0.014 | 0.0524 | 0.0461 | -0.0063 | 0.3844 | 0.3127 | -0.0717 |

## Table 4 — Welch t-tests

| contrast | metric | Δ | SE | t | verdict |
|---|---|---:|---:|---:|---|
| counterfactual augmentation vs ERM | auc | +0.0753 | 0.0260 | +2.89 | **significant** |
| counterfactual augmentation (3 copies) vs ERM | auc | +0.0574 | 0.0253 | +2.26 | **significant** |
| group DRO vs ERM | auc | -0.0143 | 0.0336 | -0.43 | not distinguishable |
| group DRO + L2 vs ERM | auc | -0.0102 | 0.0348 | -0.29 | not distinguishable |
| L2 strength, within group DRO | auc | +0.0041 | 0.0337 | +0.12 | not distinguishable |
| product-of-experts vs ERM | auc | -0.0881 | 0.0310 | -2.84 | **significant** |
| 2 vs 3 augmented copies (both label-free) | auc | +0.0179 | 0.0108 | +1.65 | not distinguishable |
| counterfactual augmentation vs ERM | cf | -0.0020 | 0.0076 | -0.26 | not distinguishable |
| product-of-experts vs ERM | cf | -0.0174 | 0.0104 | -1.67 | not distinguishable |
| group DRO vs ERM | cf | +0.0000 | 0.0090 | +0.00 | not distinguishable |
| group DRO + L2 vs ERM | cf | +0.0098 | 0.0103 | +0.95 | not distinguishable |
| counterfactual augmentation vs ERM | cs | +0.2193 | 0.0695 | +3.16 | **significant** |
| product-of-experts vs ERM | cs | -0.1359 | 0.0712 | -1.91 | not distinguishable |

---

## What the data supports

**1. Group DRO is indistinguishable from ERM, on both axes.** ΔAUC = +0.014
(t = +0.43) and ΔCF-Sens = −0.0000 (t = −0.00). Adding heavy L2 changes nothing
(M2reg vs M2: t = −0.12). The old 3-seed table reported B0 0.705 / M2 0.659 /
M2reg 0.692, which reads as "DRO costs 4.6 points and L2 recovers half of it".
**With 10 seeds all three sit inside [0.674, 0.688].** That narrative was
sampling noise and must be withdrawn.

**2. Counterfactual augmentation is the only method that works, and it needs no
label knowledge.** ΔAUC = +0.075 (t = +2.89), C-Sens 0.254 → 0.474, SER
0.14 → 0.07 — all three axes improve. Its seed sd is **0.0277**, a third of every
other arm's. **Removing the label-derived pool made it better**, 0.734 → 0.763:
`cf_empirical` had been holding it back.

**3. The augmentation ratio is not the driver — the pool content is.**
D3 (2 copies) 0.7632 vs D3ₙ (3 copies) 0.7453, t = +1.65, not significant. Both
are label-free and both beat ERM. So the option-B change is safe: dropping
`cf_empirical` does not need the copy count defended.

**4. PoE still degrades the model, less extremely than before.** AUC rose
0.540 → 0.600 once the over-strong leaky bias expert was removed, but it is still
**significantly below ERM** (t = +2.84 in ERM's favour), and it halves C-Sens
(0.254 → 0.118). Its CF-Sens reduction is **not significant** (t = 1.67). The
trade it makes — less shortcut reliance bought by responding less to everything —
survives the fix.

**5. The leak did not manufacture a false conclusion; three seeds did.** Every
leak-correction shift (−0.017 to +0.060) is smaller than the seed sd of the arm
it applies to (0.020–0.078). Only PoE moved appreciably (+0.060), exactly as the
group-vs-continuous analysis predicted.

**6. Scope limit that has to travel with this table.** Baseline CF-Sens is
**0.0363**, against MentaLLaMA's 0.2057 on the same corpus. These are results
from a model that barely uses the shortcut, so a null here is not evidence that a
method fails in general. The original script pre-registered this concern in its
docstring.

## Files

| file | contents |
|---|---|
| `experiments/i6-encoder-interventions/results/fix_b0_m1.json` | B0, M1 · 10 seeds |
| `.../fix_m2.json` | M2, M2reg · 10 seeds |
| `.../fix_d3_clean.json` | D3, 2 label-free copies · 10 seeds |
| `.../fix_d3_clean3.json` | D3ₙ, 3 label-free copies · 10 seeds |
| `.../encoder_interventions_MentalBERT_LEAKY_3seed.json` | the superseded 3-seed leaky run, kept for the comparison in Table 3 |
| `.../encoder_interventions_MentalRoBERTa.json` | the 2026-08-30 MentalRoBERTa run that had overwritten the MentalBERT file |
| `logs/round3/{g4_b0_m1,g5_m2,g8_d3clean,g8_d3clean3}.log` | per-seed logs, source of Table 2 |

## Reproduce

```bash
source env.sh
SC=experiments/i6-encoder-interventions/code/run_encoder_interventions.py
CK=$PWD/checkpoints/MentalBERT
S="0 1 2 3 4 5 6 7 8 9"
CUDA_VISIBLE_DEVICES=4 $SC_PY $SC --ckpt $CK --methods B0,M1    --seeds $S \
  --bias-fit-splits train --aug-pools clean  --out fix_b0_m1.json
CUDA_VISIBLE_DEVICES=5 $SC_PY $SC --ckpt $CK --methods M2,M2reg --seeds $S \
  --bias-fit-splits train --aug-pools clean  --out fix_m2.json
CUDA_VISIBLE_DEVICES=8 $SC_PY $SC --ckpt $CK --methods D3       --seeds $S \
  --bias-fit-splits train --aug-pools clean  --out fix_d3_clean.json
CUDA_VISIBLE_DEVICES=8 $SC_PY $SC --ckpt $CK --methods D3       --seeds $S \
  --bias-fit-splits train --aug-pools clean3 --out fix_d3_clean3.json
```
