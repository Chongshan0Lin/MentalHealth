# i10 — the 8-model DAIC sweep: analysis

Written 2026-09-05, after the verification audit. Every number here was read from
`results/suite_*.json`; the script is `code/run_full_suite.py`.

## Protocol

One model, one load, the whole battery. n = 188 DAIC-WOZ participants, text
only, teacher-forced label log-likelihood over `{Yes, No}` with a 4,032-token
budget applied identically to every arm. Scores are stored as
`Yes − No` log-odds, so every derived metric is recomputable offline.

Eight models, two scorer paths (`text` for Llama/Qwen3, `omni` for
Qwen2_5OmniThinker with `media=None`). The controlled pair is
`Qwen3-4B-base` vs `Menta` — the same base weights with and without a clinical
LoRA — which is what lets the sweep answer whether domain fine-tuning introduces
the shortcut or merely inherits it.

## Detection

| model | full AUC | removal (`ellie_only`) | 95 % CI | \|CSS\| | flips | CS | SER | P(yes) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| MentaLLaMA-chat-7B | 0.6836 | **0.5459** | [0.461, 0.632] | 0.2050 | 0.559 | 0.1126 | 1.820 | 0.106 |
| Menta | 0.7642 | **0.5757** | [0.489, 0.665] | 0.0267 | 0.000 | 0.0144 | 1.863 | 0.856 |
| Qwen3-4B-base | 0.8096 | 0.7103 | [0.630, 0.784] | 0.2561 | 0.261 | 0.4101 | 0.625 | 0.415 |
| OSim-8B | 0.8053 | 0.7217 | [0.634, 0.798] | 0.1960 | 0.298 | 0.2787 | 0.703 | 0.452 |
| OmniSapiens-SFT | 0.7998 | 0.6863 | [0.602, 0.774] | 0.4573 | 0.479 | 0.2783 | 1.643 | 0.468 |
| OmniSapiens-RL | 0.7860 | 0.6614 | [0.572, 0.754] | 0.4710 | 0.516 | 0.2778 | 1.695 | 0.436 |
| OmniSapiens-2.0 | 0.7811 | 0.6991 | [0.607, 0.786] | 0.5051 | 0.511 | 0.3646 | 1.385 | 0.410 |
| HumanOmniV2 | 0.7913 | 0.6253 | [0.531, 0.716] | 0.3715 | 0.511 | 0.3558 | 1.044 | 0.367 |

### The two detection methods rank the models almost independently

Spearman ρ between the removal AUC and \|CSS\| across the eight models is
**+0.190**. Removal's highest score (Qwen3-4B-base, 0.710) belongs to a model
with half the counterfactual sensitivity of the panel's worst case
(OmniSapiens-2.0, 0.505, removal only 0.699).

Removal returns a null — a CI covering 0.5 — on **two** models. Of those, only
MentaLLaMA has a large counterfactual effect; **Menta is null on both tests**,
which is agreement, not a miss. So the sweep does *not* support "removal misses
the shortcut" as a general claim; it supports the narrower and still serious one
that removal can return a null on a model with a large effect, and does not
order models by severity.

### Fine-tuning does not create the shortcut

`Qwen3-4B-base` (no adaptation) carries \|CSS\| **0.2561**; `Menta`, the same
base with a clinical LoRA, carries **0.0267**. The clinical fine-tune *reduces*
measured counterfactual sensitivity — but see the caveat below, because that
0.0267 is a magnitude hiding a sign.

### CSS on this panel is the ABSOLUTE form, and for two models that matters

`run_full_suite.py:160` computes `E[|Δp|]`. Recomputed as the signed `E[Δp]`
from the stored `a2b_*.jsonl` scores:

| model | signed | absolute | participants moving positive |
|---|---:|---:|---:|
| MentaLLaMA | +0.2057 | 0.2057 | 188/188 |
| Qwen3-4B-base | +0.2537 | 0.2537 | 147/188 (the 41 negatives are ~1e-8) |
| OSim-8B | +0.1752 | 0.1949 | 107/188 |
| **Menta** | **−0.0147** | 0.0266 | **52/188** |

**Menta's effect is negative**: 130 of 188 participants move *down* under
clinical questioning. Reporting 0.027 as a small positive sensitivity is wrong in
sign. OSim-8B's absolute figure overstates by 11 % and hides that 43 % of
participants move the other way. Any table built from this file must say which
form it uses.

## Interventions

Seven methods. `targeted_removal` and `threshold_proxy` were added after `menta`
and `qwen3_base` had run, so those four cells were empty until the 2026-09-05
re-run.

ΔAUC against each model's own `none`:

| model | delete_int | blank_words | marg_2 | marg_4 | targeted | threshold | instruct |
|---|---:|---:|---:|---:|---:|---:|---:|
| MentaLLaMA | **+0.0964** | +0.0759 | +0.0769 | +0.0810 | −0.0154 | −0.0009 | +0.0692 |
| Menta | −0.0347 | −0.0392 | −0.0202 | −0.0238 | — | — | −0.0297 |
| Qwen3-4B-base | −0.0120 | −0.0150 | −0.0161 | −0.0135 | — | — | −0.0007 |
| OSim-8B | −0.0025 | −0.0134 | +0.0012 | −0.0010 | −0.0008 | +0.0012 | −0.0114 |
| OmniSapiens-SFT | +0.0183 | +0.0021 | +0.0123 | +0.0132 | −0.0024 | +0.0079 | +0.0127 |
| OmniSapiens-RL | +0.0260 | +0.0154 | +0.0230 | +0.0165 | +0.0018 | +0.0069 | +0.0218 |
| OmniSapiens-2.0 | +0.0372 | +0.0260 | +0.0230 | +0.0259 | +0.0029 | +0.0102 | +0.0280 |
| HumanOmniV2 | −0.0115 | −0.0185 | −0.0086 | −0.0116 | −0.0064 | −0.0006 | −0.0038 |
| **mean** | +0.0146 | +0.0042 | +0.0114 | +0.0108 | −0.0034 | +0.0041 | +0.0107 |
| **improves** | 4/8 | 4/8 | **5/8** | 4/8 | 2/6 | 4/6 | 4/8 |

52 run cells: 27 improve, 25 degrade, 0 ties. **No method improves all eight.**
Best coverage is `marginalise_2` at 5/8.

**Read the rows, not the columns.** MentaLLaMA and the three OmniSapiens
checkpoints are positive almost everywhere; Menta, Qwen3-4B-base and HumanOmniV2
are negative almost everywhere. The split is by model, not by method — which is
why no method generalises. Benefit tracks baseline \|CSS\| (Spearman ρ ≈ +0.50
to +0.62 across methods): interventions help the models that were using the
shortcut and cost the models that were not. MentaLLaMA is the visible exception,
taking the largest gain from middling \|CSS\|; what distinguishes it is the
lowest baseline AUC in the panel (0.684) and the only below-chance conflict AUC
(0.346), so "most headroom" is a competing explanation that n = 8 cannot
separate.

### Three caveats that must travel with this table

1. **The CSS zeros are hard-coded.** `run_full_suite.py:197-200` writes
   `css=0.0` for `delete_interviewer`, `blank_words`, `marginalise_2` and
   `marginalise_4`. With the channel removed there is nothing to swap, so this
   is correct — but it is zero *by construction*, not a measurement. Only
   `instruct_ignore` has a measured CSS (line 204), and it **rises** on 4 of 8
   models (Menta, and all three OmniSapiens), most sharply 0.505 → 0.577.
2. **`threshold_proxy`'s AUC is in-sample contaminated.** Line 189 applies a
   correction fitted on train+dev (n = 141) to all 188 rows, and the AUC scores
   all 188. Test-only figures are in `experiments/i11-threshold-offline/`.
3. **`targeted_removal` is a no-op on 65 of 188 transcripts.** Only 123 carry
   the `Ellie: <id> (text)` prompt IDs the strip regex matches, so the arm
   dilutes its own effect by a third when scored over all 188.

### The one axis where something is universal

CS (responsiveness to the patient's own symptom content) rises under
`delete_interviewer` and `blank_words` on **8/8 models**, no exceptions
(+0.008 to +0.279). This is the only 8/8 effect in the sweep, and it points the
opposite way from the AUC table: the four models whose AUC *fell* were still
moved in the right direction mechanically. Their AUC dropped because interviewer
turns also carried legitimate conversational context the ranking was using.

## Files

`results/suite_{mentallama_suite,menta,qwen3_base,osim8b,omnisapiens_sft,omnisapiens_rl,omnisapiens_20,humanomniv2}.json`.
Each holds `detection`, `interventions`, `threshold_fit` and — for the six
models run after the arm was added — `scores`, 13 per-participant arrays of 188
log-odds, which is what makes offline re-derivation possible.
