# DAIC-WOZ mitigation — full data, 7 methods × 8 models

Raw data: `experiments/i10-model-sweep/results/suite_*.json`, one file per model,
field path `interventions.<method>.{auc, css, cs}`. Baselines from `detection.*`.
Not run by me — this is the Aug 21–31 sweep; `i10-model-sweep` has no `analysis.md`,
so this file is the first written analysis of it.

**n = 188 DAIC-WOZ participant transcripts**, text-only, teacher-forced label
log-likelihood over the closed label space. AUC is the metric; higher is better.

---

## 0. What the three metrics mean

| metric | definition | good direction |
|---|---|---|
| **AUC** | ranking of participants by p(depressed) against gold | up |
| **CSS** | counterfactual shortcut sensitivity — mean paired Δp when only the interviewer's questions are swapped biographical→clinical, participant words byte-identical | **down** |
| **CS** | construct sensitivity — mean paired Δp when the *participant's own* symptom content is changed | **up** |

CSS and CS are the two halves that must move in opposite directions for a
mitigation to be real. A method that only lowers CSS may have made the model
deaf to everything, not just to the shortcut.

---

## Table 1 — baseline, no intervention

| model | AUC | CSS | flips | CS | P(yes) | SER | conflict AUC | ablation AUC |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA-chat-7B | 0.6836 | 0.2050 | 0.559 | 0.1126 | 0.106 | 1.820 | 0.3458 | 0.5459 |
| Menta (Qwen3-4B+LoRA) | 0.7642 | 0.0267 | 0.000 | 0.0144 | 0.856 | 1.863 | 0.7805 | 0.5757 |
| Qwen3-4B-base | 0.8096 | 0.2561 | 0.261 | 0.4101 | 0.415 | 0.625 | 0.6761 | 0.7103 |
| OSim-8B | 0.8053 | 0.1960 | 0.298 | 0.2787 | 0.452 | 0.703 | 0.7728 | 0.7217 |
| OmniSapiens-SFT | 0.7998 | 0.4573 | 0.479 | 0.2783 | 0.468 | 1.643 | 0.3055 | 0.6863 |
| OmniSapiens-RL | 0.7860 | 0.4710 | 0.516 | 0.2778 | 0.436 | 1.695 | 0.3407 | 0.6614 |
| OmniSapiens-2.0 | 0.7811 | 0.5051 | 0.511 | 0.3646 | 0.410 | 1.385 | 0.2924 | 0.6991 |
| HumanOmniV2 | 0.7913 | 0.3715 | 0.511 | 0.3558 | 0.367 | 1.044 | 0.4452 | 0.6253 |

Note the spread in baseline CSS — 0.027 (Menta) to 0.505 (OmniSapiens-2.0).
**The models are not all suffering from the same disease**, which is the whole
reason no single method covers them.

---

## Table 2 — absolute AUC, every cell

| model | none | delete_int | blank_words | marg_2 | marg_4 | targeted | threshold | instruct |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA-chat-7B | 0.6836 | 0.7800 | 0.7595 | 0.7605 | 0.7646 | 0.6682 | 0.6827 | 0.7528 |
| Menta (Qwen3-4B+LoRA) | 0.7642 | 0.7295 | 0.7249 | 0.7440 | 0.7404 | — | — | 0.7344 |
| Qwen3-4B-base | 0.8096 | 0.7976 | 0.7946 | 0.7934 | 0.7961 | — | — | 0.8089 |
| OSim-8B | 0.8053 | 0.8028 | 0.7919 | 0.8065 | 0.8044 | 0.8045 | 0.8066 | 0.7939 |
| OmniSapiens-SFT | 0.7998 | 0.8180 | 0.8019 | 0.8121 | 0.8130 | 0.7974 | 0.8077 | 0.8125 |
| OmniSapiens-RL | 0.7860 | 0.8120 | 0.8014 | 0.8090 | 0.8025 | 0.7878 | 0.7929 | 0.8078 |
| OmniSapiens-2.0 | 0.7811 | 0.8183 | 0.8071 | 0.8041 | 0.8070 | 0.7839 | 0.7913 | 0.8090 |
| HumanOmniV2 | 0.7913 | 0.7798 | 0.7729 | 0.7827 | 0.7797 | 0.7850 | 0.7907 | 0.7875 |

## Table 3 — ΔAUC vs no intervention

| model | delete_int | blank_words | marg_2 | marg_4 | targeted | threshold | instruct | best method |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA-chat-7B | **+0.0964** | +0.0759 | +0.0769 | +0.0810 | -0.0154 | -0.0009 | +0.0692 | delete_int (+0.0964) |
| Menta (Qwen3-4B+LoRA) | -0.0347 | -0.0392 | **-0.0202** | -0.0238 | — | — | -0.0297 | marg_2 (-0.0202) |
| Qwen3-4B-base | -0.0120 | -0.0150 | -0.0161 | -0.0135 | — | — | **-0.0007** | instruct (-0.0007) |
| OSim-8B | -0.0025 | -0.0134 | +0.0012 | -0.0010 | -0.0008 | **+0.0012** | -0.0114 | threshold (+0.0012) |
| OmniSapiens-SFT | **+0.0183** | +0.0021 | +0.0123 | +0.0132 | -0.0024 | +0.0079 | +0.0127 | delete_int (+0.0183) |
| OmniSapiens-RL | **+0.0260** | +0.0154 | +0.0230 | +0.0165 | +0.0018 | +0.0069 | +0.0218 | delete_int (+0.0260) |
| OmniSapiens-2.0 | **+0.0372** | +0.0260 | +0.0230 | +0.0259 | +0.0029 | +0.0102 | +0.0280 | delete_int (+0.0372) |
| HumanOmniV2 | -0.0115 | -0.0185 | -0.0086 | -0.0116 | -0.0064 | **-0.0006** | -0.0038 | threshold (-0.0006) |

---

## Table 4 — per-method verdict on AUC  *(the "nothing works everywhere" table)*

| method | models run | improves | degrades | mean Δ | worst Δ | best Δ | degrades on |
|---|---|---|---|---|---|---|---|
| `delete_interviewer` | 8/8 | **4** | 4 | +0.0146 | -0.0347 | +0.0964 | Menta (Qwen3-4B+LoRA), Qwen3-4B-base, OSim-8B, HumanOmniV2 |
| `blank_words` | 8/8 | **4** | 4 | +0.0042 | -0.0392 | +0.0759 | Menta (Qwen3-4B+LoRA), Qwen3-4B-base, OSim-8B, HumanOmniV2 |
| `marginalise_2` | 8/8 | **5** | 3 | +0.0114 | -0.0202 | +0.0769 | Menta (Qwen3-4B+LoRA), Qwen3-4B-base, HumanOmniV2 |
| `marginalise_4` | 8/8 | **4** | 4 | +0.0108 | -0.0238 | +0.0810 | Menta (Qwen3-4B+LoRA), Qwen3-4B-base, OSim-8B, HumanOmniV2 |
| `targeted_removal` | 6/8 | **2** | 4 | -0.0034 | -0.0154 | +0.0029 | MentaLLaMA-chat-7B, OSim-8B, OmniSapiens-SFT, HumanOmniV2 |
| `threshold_proxy` | 6/8 | **4** | 2 | +0.0041 | -0.0009 | +0.0102 | MentaLLaMA-chat-7B, HumanOmniV2 |
| `instruct_ignore` | 8/8 | **4** | 4 | +0.0107 | -0.0297 | +0.0692 | Menta (Qwen3-4B+LoRA), Qwen3-4B-base, OSim-8B, HumanOmniV2 |

**Best coverage is 5/8** (`marginalise_2`). No method reaches 8/8, and no method
even reaches 6/8. `targeted_removal` is net-negative on average.

---

## Table 5 — CSS after intervention  *(and why four columns are not evidence)*

| model | none | delete_int | blank_words | marg_2 | marg_4 | targeted | threshold | instruct |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA-chat-7B | 0.2050 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | 0.0940 |
| Menta (Qwen3-4B+LoRA) | 0.0267 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | 0.1414 |
| Qwen3-4B-base | 0.2561 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | 0.0576 |
| OSim-8B | 0.1960 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | 0.1649 |
| OmniSapiens-SFT | 0.4573 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | 0.5056 |
| OmniSapiens-RL | 0.4710 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | 0.5313 |
| OmniSapiens-2.0 | 0.5051 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | 0.5766 |
| HumanOmniV2 | 0.3715 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | — | 0.2492 |

**The 0.0000 column block is a definitional artifact, not a result.**
`delete_interviewer`, `blank_words`, `marginalise_2` and `marginalise_4` all
remove or average away the interviewer turns. CSS is measured by swapping those
turns. Once they are gone there is nothing left to swap, so CSS is *constructed*
to be exactly zero — it was not *reduced* to zero. Reporting these four as
"CSS → 0, shortcut eliminated" would be circular.

`instruct_ignore` is the only method that leaves the interviewer text in place,
so it is the only column where CSS is a genuine measurement — and it backfires:

| model | CSS before | CSS after `instruct_ignore` | Δ |
|---|---|---|---|
| MentaLLaMA-chat-7B | 0.2050 | 0.0940 | -0.1110 |
| Menta (Qwen3-4B+LoRA) | 0.0267 | 0.1414 | +0.1147 ⚠️ |
| Qwen3-4B-base | 0.2561 | 0.0576 | -0.1985 |
| OSim-8B | 0.1960 | 0.1649 | -0.0310 |
| OmniSapiens-SFT | 0.4573 | 0.5056 | +0.0483 ⚠️ |
| OmniSapiens-RL | 0.4710 | 0.5313 | +0.0603 ⚠️ |
| OmniSapiens-2.0 | 0.5051 | 0.5766 | +0.0716 ⚠️ |
| HumanOmniV2 | 0.3715 | 0.2492 | -0.1224 |

**Telling the model "ignore the interviewer" makes it *more* interviewer-sensitive
on 4 of 8 models** (Menta (Qwen3-4B+LoRA), OmniSapiens-SFT, OmniSapiens-RL, OmniSapiens-2.0). Prompt-level instruction is not a
mitigation here; on the OmniSapiens line it is an amplifier.

---

## Table 6 — CS (construct sensitivity), the one axis where something is universal

CS is only recorded for `none`, `delete_interviewer` and `blank_words`.

| model | none | delete_int | Δ | blank_words | Δ |
|---|---|---|---|---|---|
| MentaLLaMA-chat-7B | 0.1126 | 0.2033 | +0.0907 | 0.1871 | +0.0744 |
| Menta (Qwen3-4B+LoRA) | 0.0144 | 0.0396 | +0.0253 | 0.0218 | +0.0075 |
| Qwen3-4B-base | 0.4101 | 0.5793 | +0.1692 | 0.6156 | +0.2055 |
| OSim-8B | 0.2787 | 0.4534 | +0.1747 | 0.4428 | +0.1641 |
| OmniSapiens-SFT | 0.2783 | 0.5441 | +0.2658 | 0.5576 | +0.2793 |
| OmniSapiens-RL | 0.2778 | 0.5355 | +0.2577 | 0.5418 | +0.2639 |
| OmniSapiens-2.0 | 0.3646 | 0.5705 | +0.2059 | 0.5846 | +0.2199 |
| HumanOmniV2 | 0.3558 | 0.4291 | +0.0733 | 0.3916 | +0.0358 |

**Removing the interviewer raises construct sensitivity on 8/8 models for
`delete_interviewer` and 8/8 for `blank_words` — no exceptions.**

This is the only 8/8 effect anywhere in the sweep, and it points the opposite way
from Table 4. Every model listens harder to the participant's actual symptoms once
the interviewer is gone, including the four models whose AUC *fell*. So for those
four the removal did not fail to change the model — it changed the model in the
right direction and the AUC still dropped, because interviewer turns also carried
legitimate conversational context that the ranking was using.

---

## Table 7 — who benefits: baseline shortcut reliance vs mitigation gain

| model | baseline CSS | ΔAUC delete_int | helped? |
|---|---|---|---|
| OmniSapiens-2.0 | 0.5051 | +0.0372 | yes |
| OmniSapiens-RL | 0.4710 | +0.0260 | yes |
| OmniSapiens-SFT | 0.4573 | +0.0183 | yes |
| HumanOmniV2 | 0.3715 | -0.0115 | no |
| Qwen3-4B-base | 0.2561 | -0.0120 | no |
| MentaLLaMA-chat-7B | 0.2050 | +0.0964 | yes |
| OSim-8B | 0.1960 | -0.0025 | no |
| Menta (Qwen3-4B+LoRA) | 0.0267 | -0.0347 | no |

| method | Spearman ρ (baseline CSS, ΔAUC) |
|---|---|
| `delete_interviewer` | +0.524 |
| `blank_words` | +0.500 |
| `marginalise_2` | +0.524 |
| `marginalise_4` | +0.524 |
| `instruct_ignore` | +0.619 |

All positive, ρ ≈ +0.50 to +0.62 on n=8. The direction is consistent across every
method, though n=8 puts wide error bars on any single ρ. **Interventions help the
models that were actually using the shortcut and cost the models that were not.**

**MentaLLaMA is the visible exception** and it is worth stating rather than
smoothing over: it has only middling baseline CSS (0.205) yet takes by far the
largest gain (+0.096). What separates it is not CSS but its baseline AUC — 0.684,
the lowest in the panel by 8 points, and the only model whose conflict AUC (0.346)
sits below chance. It is the one model badly enough broken by the interviewer
channel that removing it recovers real headroom. Rank correlation with baseline
AUC is therefore the competing explanation, and n=8 cannot separate the two.
Removing interviewer turns from a model that never leaned on them is pure loss of
context — the same phenomenon as the H4/H6 nulls on `mmpsy` and `Irf`, seen from
the mitigation side.

---

## Table 8 — coverage gaps

| method | missing cells |
|---|---|
| `delete_interviewer` (auc) | none |
| `blank_words` (auc) | none |
| `marginalise_2` (auc) | none |
| `marginalise_4` (auc) | none |
| `targeted_removal` (auc) | Menta (Qwen3-4B+LoRA), Qwen3-4B-base |
| `threshold_proxy` (auc) | Menta (Qwen3-4B+LoRA), Qwen3-4B-base |
| `instruct_ignore` (auc) | none |
| `targeted_removal` (css) | MentaLLaMA-chat-7B, Menta (Qwen3-4B+LoRA), Qwen3-4B-base, OSim-8B, OmniSapiens-SFT, OmniSapiens-RL, OmniSapiens-2.0, HumanOmniV2 |
| `threshold_proxy` (css) | MentaLLaMA-chat-7B, Menta (Qwen3-4B+LoRA), Qwen3-4B-base, OSim-8B, OmniSapiens-SFT, OmniSapiens-RL, OmniSapiens-2.0, HumanOmniV2 |

`targeted_removal` and `threshold_proxy` were never run on Menta or Qwen3-4B-base;
both require a fitted threshold/feature model that only exists for 6 of the 8. CS
was only computed for `none`, `delete_interviewer`, `blank_words`. Any claim about
those methods is over 6 models, not 8.

---

## What the full matrix actually supports

1. **No method improves AUC on all 8 models.** Best is 5/8. This is the headline
   and it survives the extra detail.
2. **The four removal methods' CSS = 0 is circular** and should not be quoted as
   a mitigation result. Only `instruct_ignore` measures CSS honestly.
3. **`instruct_ignore` backfires**: prompting the model to ignore the interviewer
   raises CSS on 4/8, most sharply on the OmniSapiens line (0.457→0.506,
   0.471→0.531, 0.505→0.577).
4. **The only universal effect is on CS**: removal raises construct sensitivity
   8/8. Mitigation works on the mechanism even where it fails on the metric.
5. **Benefit tracks baseline shortcut reliance** (ρ ≈ +0.5). Mitigation is not a
   default-on safety measure; applied blind it degrades the models that were
   already clean.

The practical consequence: **you must run the counterfactual detection first and
only then decide whether to intervene.** That is an argument for the detection
half of the paper, made by the mitigation half.

---

## Related mitigation data elsewhere

| what | path |
|---|---|
| this sweep (7 methods × 8 models) | `experiments/i10-model-sweep/results/suite_*.json` |
| format/instruction variants | `experiments/i9-format-instruction/results/format_ser.json` |
| single-model deep dive + written analysis | `experiments/i1-llm-inference-interventions/{analysis.md, results/i1_summary.json}` |
| train-time GRPO attempt (failed) | `experiments/i8-grpo-mitigation/results/eval_{baseline,M0,M0_bal_batch,M1}.json` |
| MentalBERT encoder arm, 3 seeds | `experiments/i6-encoder-interventions/results/encoder_interventions.json` |
