# Detection tables — data for §\ref{sec:detect}

Source: `experiments/i10-model-sweep/results/suite_*.json`, field paths
`detection.full_auc`, `detection.removal_ellie_only_auc`,
`detection.counterfactual_css`. All 8 audited LLMs present; no gaps.

---

## ⚠ Read this before pasting

**The draft's MentaLLaMA numbers come from a different protocol than the sweep.**

| quantity | draft text | `suite_mentallama_suite.json` |
|---|---|---|
| full-input AUC | 0.663 | **0.684** |
| CSS | 0.206 | **0.205** |

The 0.663 is the A1/I1 `full` condition over all 188 participants; the suite
runs its own harness. The CSS difference is negligible (0.2050 vs 0.2057,
same measurement rounded differently), but **0.663 vs 0.684 is a real
protocol difference, not rounding.**

Do not mix them in one table. Either quote the suite value (0.684) throughout
both tables so every row is like-for-like, or keep 0.663 in the prose and add a
footnote saying the table reports the sweep harness. The tables below use the
**suite** values throughout.

---

## Table 1 — Removal test (`tab:removal`)

| Model | Full-input AUC | Interviewer-only AUC | Removal < full? |
|---|---|---|---|
| MentaLLaMA | 0.684 | 0.546 | yes |
| Menta | 0.764 | 0.576 | yes |
| qwen3\_base | 0.810 | 0.710 | yes |
| OSim-8B | 0.805 | 0.722 | yes |
| OmniSapiens-SFT | 0.800 | 0.686 | yes |
| OmniSapiens-RL | 0.786 | 0.661 | yes |
| OmniSapiens2.0 | 0.781 | 0.699 | yes |
| HumanOmniV2 | 0.791 | 0.625 | yes |

**The pattern holds across all 8 models without exception**: interviewer-only
AUC is below full-input AUC everywhere. Margins range from 0.083 (OmniSapiens2.0)
to 0.188 (Menta).

**This does *not* reproduce \citet{burdisso2024daicwoz}.** They found the reverse
with classical encoders — interviewer-only *outperforming* full input, reaching
0.90 F1 by deliberately exploiting the shortcut. Every zero-shot LLM here fails
that test. That contrast is the motivation for the second detection method, and
it is stronger stated as a genuine architecture/training difference than as a
partial replication.

Note also that removal AUC is **far above chance** for several models
(qwen3\_base 0.710, OSim-8B 0.722, OmniSapiens2.0 0.699). "Below full input" is
not the same as "uninformative" — the interviewer channel alone carries
substantial signal for most models; it simply carries less than the full input.

---

## Table 2 — Removal vs counterfactual substitution (`tab:cf`)

| Model | Full-input AUC | Removal AUC | CSS |
|---|---|---|---|
| MentaLLaMA | 0.684 | 0.546 | 0.205 |
| Menta | 0.764 | 0.576 | 0.027 |
| qwen3\_base | 0.810 | 0.710 | 0.256 |
| OSim-8B | 0.805 | 0.722 | 0.196 |
| OmniSapiens-SFT | 0.800 | 0.686 | 0.457 |
| OmniSapiens-RL | 0.786 | 0.661 | 0.471 |
| OmniSapiens2.0 | 0.781 | 0.699 | 0.505 |
| HumanOmniV2 | 0.791 | 0.625 | 0.372 |

### Does the removal/counterfactual disagreement generalise?

**Yes for 7 of 8 — and the OmniSapiens family is far more affected than
MentaLLaMA.**

| Model | CSS | pred. flips | reading |
|---|---|---|---|
| MentaLLaMA | 0.205 | 0.559 | comparable to MentaLLaMA |
| Menta | 0.027 | 0.000 | **no effect** — the one exception |
| qwen3\_base | 0.256 | 0.261 | comparable to MentaLLaMA |
| OSim-8B | 0.196 | 0.298 | comparable to MentaLLaMA |
| OmniSapiens-SFT | 0.457 | 0.479 | **strongest** — larger than MentaLLaMA |
| OmniSapiens-RL | 0.471 | 0.516 | **strongest** — larger than MentaLLaMA |
| OmniSapiens2.0 | 0.505 | 0.511 | **strongest** — larger than MentaLLaMA |
| HumanOmniV2 | 0.372 | 0.511 | **strongest** — larger than MentaLLaMA |

**The three post-training stages carry the largest effect in the study**:
SFT 0.457, RL 0.471, HARPO 0.505 — all more than double MentaLLaMA's 0.205, with
roughly half of all predictions flipping. CSS *rises monotonically* across the
three stages (+0.048 SFT→HARPO), which is a direct amplification result on
matched architecture.

**Menta is the sole exception (CSS 0.027, 0.000 flips)** and should be reported
as saturated rather than as evidence against the effect — it also shows the
lowest construct sensitivity in the table (0.014), i.e. it barely responds to
*any* input change, so its counterfactual null is uninformative.

---

## Supporting columns (not in the draft tables, available if useful)

| Model | conflict AUC | construct sens. | SER |
|---|---|---|---|
| MentaLLaMA | 0.346 | 0.113 | 1.82 |
| Menta | 0.781 | 0.014 | 1.86 |
| qwen3\_base | 0.676 | 0.410 | 0.62 |
| OSim-8B | 0.773 | 0.279 | 0.70 |
| OmniSapiens-SFT | 0.305 | 0.278 | 1.64 |
| OmniSapiens-RL | 0.341 | 0.278 | 1.70 |
| OmniSapiens2.0 | 0.292 | 0.365 | 1.39 |
| HumanOmniV2 | 0.445 | 0.356 | 1.04 |

`conflict_auc` is the adversarial-context set. Four models fall **below 0.5** —
MentaLLaMA 0.346, OmniSapiens-SFT 0.305, OmniSapiens-RL 0.341, OmniSapiens2.0
0.292 — i.e. under adversarial interviewer context their ranking is
*anti-correlated with the truth*, not merely uninformative. HumanOmniV2 (0.445)
is also below chance.

`SER` (shortcut-to-evidence ratio, CSS / construct sensitivity) exceeds 1.0 for
five models — the interviewer channel moves the prediction *more* than the
patient's own words do. MentaLLaMA 1.82, Menta 1.86, OmniSapiens-RL 1.70,
SFT 1.64, HumanOmniV2 1.04. Only qwen3\_base (0.62) and OSim-8B (0.70) weight
the evidence above the shortcut.

This is arguably a stronger headline than CSS alone and is fully computed.

