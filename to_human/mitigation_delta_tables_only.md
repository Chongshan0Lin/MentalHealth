# Mitigation Δ tables — DAIC-WOZ, 7 methods × 8 models

`experiments/i10-model-sweep/results/suite_*.json` · n = 188 · text-only ·
every cell = Δ vs that model's own `none` baseline · `—` = arm not run.

---

## Table 1 — Δ AUC (higher is better)

| model | baseline | delete_int | blank_words | marg_2 | marg_4 | targeted | threshold | instruct |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 0.6836 | +0.0964 | +0.0759 | +0.0769 | +0.0810 | -0.0154 | -0.0009 | +0.0692 |
| Menta | 0.7642 | -0.0347 | -0.0392 | -0.0202 | -0.0238 | — | — | -0.0297 |
| Qwen3-4B-base | 0.8096 | -0.0120 | -0.0150 | -0.0161 | -0.0135 | — | — | -0.0007 |
| OSim-8B | 0.8053 | -0.0025 | -0.0134 | +0.0012 | -0.0010 | -0.0008 | +0.0012 | -0.0114 |
| OmniSap-SFT | 0.7998 | +0.0183 | +0.0021 | +0.0123 | +0.0132 | -0.0024 | +0.0079 | +0.0127 |
| OmniSap-RL | 0.7860 | +0.0260 | +0.0154 | +0.0230 | +0.0165 | +0.0018 | +0.0069 | +0.0218 |
| OmniSap-2.0 | 0.7811 | +0.0372 | +0.0260 | +0.0230 | +0.0259 | +0.0029 | +0.0102 | +0.0280 |
| HumanOmniV2 | 0.7913 | -0.0115 | -0.0185 | -0.0086 | -0.0116 | -0.0064 | -0.0006 | -0.0038 |
| **mean Δ** | | +0.0146 | +0.0042 | +0.0114 | +0.0108 | -0.0034 | +0.0041 | +0.0107 |
| **improves** | | 4/8 | 4/8 | 5/8 | 4/8 | 2/6 | 4/6 | 4/8 |

---

## Table 2 — Δ CSS (lower is better)

| model | baseline | delete_int | blank_words | marg_2 | marg_4 | targeted | threshold | instruct |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 0.2050 | -0.2050 † | -0.2050 † | -0.2050 † | -0.2050 † | — | — | -0.1110 |
| Menta | 0.0267 | -0.0267 † | -0.0267 † | -0.0267 † | -0.0267 † | — | — | +0.1147 |
| Qwen3-4B-base | 0.2561 | -0.2561 † | -0.2561 † | -0.2561 † | -0.2561 † | — | — | -0.1985 |
| OSim-8B | 0.1960 | -0.1960 † | -0.1960 † | -0.1960 † | -0.1960 † | — | — | -0.0310 |
| OmniSap-SFT | 0.4573 | -0.4573 † | -0.4573 † | -0.4573 † | -0.4573 † | — | — | +0.0483 |
| OmniSap-RL | 0.4710 | -0.4710 † | -0.4710 † | -0.4710 † | -0.4710 † | — | — | +0.0603 |
| OmniSap-2.0 | 0.5051 | -0.5051 † | -0.5051 † | -0.5051 † | -0.5051 † | — | — | +0.0716 |
| HumanOmniV2 | 0.3715 | -0.3715 † | -0.3715 † | -0.3715 † | -0.3715 † | — | — | -0.1224 |
| **improves** | | n/a † | n/a † | n/a † | n/a † | — | — | 4/8 |

† **zero by construction, not by measurement.** These four methods delete or
average away the interviewer turns, and CSS is measured by swapping those turns —
with nothing left to swap, CSS ≡ 0. Not evidence of mitigation. `instruct_ignore`
is the only column where CSS is genuinely measured, and it *rises* on 4 of 8.

---

## Table 3 — Δ CS (higher is better)

| model | baseline | delete_int | blank_words |
|---|---|---|---|
| MentaLLaMA | 0.1126 | +0.0907 | +0.0744 |
| Menta | 0.0144 | +0.0253 | +0.0075 |
| Qwen3-4B-base | 0.4101 | +0.1692 | +0.2055 |
| OSim-8B | 0.2787 | +0.1747 | +0.1641 |
| OmniSap-SFT | 0.2783 | +0.2658 | +0.2793 |
| OmniSap-RL | 0.2778 | +0.2577 | +0.2639 |
| OmniSap-2.0 | 0.3646 | +0.2059 | +0.2199 |
| HumanOmniV2 | 0.3558 | +0.0733 | +0.0358 |
| **improves** | | 8/8 | 8/8 |

Recorded only for these two arms. The only 8/8 column in the sweep.

---

**Metrics.** AUC = ranking of participants against gold. CSS = counterfactual
shortcut sensitivity, mean paired Δp when only the interviewer's questions are
swapped biographical→clinical with the participant's words byte-identical. CS =
construct sensitivity, mean paired Δp when the participant's own symptom content
changes.
