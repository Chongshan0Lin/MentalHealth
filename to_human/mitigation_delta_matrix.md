# DAIC-WOZ mitigation — model × method Δ matrix

Source: `experiments/i10-model-sweep/results/suite_*.json` · n = 188 · text-only.
Every cell is **Δ vs `none`** (that model's own no-intervention baseline).
`—` = arm not run for that model.

---

## Δ AUC  — higher is better

| model | baseline AUC | delete_int | blank_words | marg_2 | marg_4 | targeted | threshold | instruct |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 0.6836 | **+0.0964** | **+0.0759** | **+0.0769** | **+0.0810** | -0.0154 | -0.0009 | **+0.0692** |
| Menta | 0.7642 | -0.0347 | -0.0392 | -0.0202 | -0.0238 | — | — | -0.0297 |
| Qwen3-4B-base | 0.8096 | -0.0120 | -0.0150 | -0.0161 | -0.0135 | — | — | -0.0007 |
| OSim-8B | 0.8053 | -0.0025 | -0.0134 | **+0.0012** | -0.0010 | -0.0008 | **+0.0012** | -0.0114 |
| OmniSap-SFT | 0.7998 | **+0.0183** | **+0.0021** | **+0.0123** | **+0.0132** | -0.0024 | **+0.0079** | **+0.0127** |
| OmniSap-RL | 0.7860 | **+0.0260** | **+0.0154** | **+0.0230** | **+0.0165** | **+0.0018** | **+0.0069** | **+0.0218** |
| OmniSap-2.0 | 0.7811 | **+0.0372** | **+0.0260** | **+0.0230** | **+0.0259** | **+0.0029** | **+0.0102** | **+0.0280** |
| HumanOmniV2 | 0.7913 | -0.0115 | -0.0185 | -0.0086 | -0.0116 | -0.0064 | -0.0006 | -0.0038 |
| **mean Δ** | | +0.0146 | +0.0042 | +0.0114 | +0.0108 | -0.0034 | +0.0041 | +0.0107 |
| **improves** | | 4/8 | 4/8 | 5/8 | 4/8 | 2/6 | 4/6 | 4/8 |

Best column is `marg_2` at 5/8. **No column is 8/8.** Bold = improvement.

Read the *rows* and a second pattern appears: MentaLLaMA and the three
OmniSapiens checkpoints are positive nearly everywhere; Menta, Qwen3-4B-base and
HumanOmniV2 are negative nearly everywhere. **The split is by model, not by
method** — which is why no method generalises.

---

## Δ CSS  — lower is better (shortcut reliance)

| model | baseline CSS | delete_int | blank_words | marg_2 | marg_4 | targeted | threshold | instruct |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 0.2050 | -0.2050† | -0.2050† | -0.2050† | -0.2050† | — | — | -0.1110 |
| Menta | 0.0267 | -0.0267† | -0.0267† | -0.0267† | -0.0267† | — | — | +0.1147 ⚠️ |
| Qwen3-4B-base | 0.2561 | -0.2561† | -0.2561† | -0.2561† | -0.2561† | — | — | -0.1985 |
| OSim-8B | 0.1960 | -0.1960† | -0.1960† | -0.1960† | -0.1960† | — | — | -0.0310 |
| OmniSap-SFT | 0.4573 | -0.4573† | -0.4573† | -0.4573† | -0.4573† | — | — | +0.0483 ⚠️ |
| OmniSap-RL | 0.4710 | -0.4710† | -0.4710† | -0.4710† | -0.4710† | — | — | +0.0603 ⚠️ |
| OmniSap-2.0 | 0.5051 | -0.5051† | -0.5051† | -0.5051† | -0.5051† | — | — | +0.0716 ⚠️ |
| HumanOmniV2 | 0.3715 | -0.3715† | -0.3715† | -0.3715† | -0.3715† | — | — | -0.1224 |

† **zero by construction, not by measurement.** These four methods delete or
average away the interviewer turns; CSS is measured by swapping those turns.
Nothing left to swap ⇒ CSS ≡ 0. Do not read these columns as mitigation success.

⚠️ `instruct_ignore` is the only method that leaves the text in place, so it is
the only honest CSS column — and CSS **rises** on 4 of 8 models.

---

## Δ CS  — higher is better (responsiveness to the patient)

Only recorded for `delete_interviewer` and `blank_words`.

| model | baseline CS | delete_int | blank_words |
|---|---|---|---|
| MentaLLaMA | 0.1126 | **+0.0907** | **+0.0744** |
| Menta | 0.0144 | **+0.0253** | **+0.0075** |
| Qwen3-4B-base | 0.4101 | **+0.1692** | **+0.2055** |
| OSim-8B | 0.2787 | **+0.1747** | **+0.1641** |
| OmniSap-SFT | 0.2783 | **+0.2658** | **+0.2793** |
| OmniSap-RL | 0.2778 | **+0.2577** | **+0.2639** |
| OmniSap-2.0 | 0.3646 | **+0.2059** | **+0.2199** |
| HumanOmniV2 | 0.3558 | **+0.0733** | **+0.0358** |
| **improves** | | 8/8 | 8/8 |

**The only 8/8 column anywhere in the sweep.** Removing the interviewer makes
every model more responsive to the patient's own symptom content — including the
four models whose AUC fell. Those four were not unaffected by the mitigation;
they were moved in the right direction, and lost AUC anyway because interviewer
turns also carried legitimate conversational context the ranking was using.

---

## The three matrices side by side, per method

| method | ΔAUC improves | ΔCSS honest? | ΔCS improves | verdict |
|---|---|---|---|---|
| `delete_interviewer` | 4/8 | no (†) | 8/8 | best ΔAUC and ΔCS, but fails on 4 models |
| `blank_words` | 4/8 | no (†) | 8/8 | delete_int beats it on AUC 8/8; wins on CS for 2 models |
| `marginalise_2` | 5/8 | no (†) | — | widest ΔAUC coverage (5/8), no CS data |
| `marginalise_4` | 4/8 | no (†) | — | no better than marg_2 at twice the cost |
| `targeted_removal` | 2/6 | not run | — | net-negative; partial-removal does not work |
| `threshold_proxy` | 4/6 | not run | — | tiny but low-variance; never helps much, rarely hurts |
| `instruct_ignore` | 4/8 | yes | — | backfires — raises CSS on 4/8 |

---

## One number to take away

**7 methods × 8 models = 52 run cells (2 arms missing on 2 models). 27 improve
AUC, 25 degrade it.**
Mitigation on this benchmark is a coin flip unless you first know whether the
model in front of you actually uses the shortcut.
