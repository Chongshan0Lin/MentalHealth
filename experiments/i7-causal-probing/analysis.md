# i7 — probing and causal erasure: analysis

Written 2026-09-05. Numbers from `results/{layerwise_multi,phase3_leace_L8-24,
phase3_leace_layer13,phase1_probes}.json`; scripts in `code/`.

## What is probed

Not corpus identity, not the label. The probe classifies, at each layer, the
model's own hidden state for **the same participant under two interviewer
contexts** — `cf_neutral` (biographical questions) versus `cf_depression`
(clinically loaded questions), with the participant's words byte-identical
(`layerwise_multi.py:100`). What it decodes is *which kind of question the
interviewer asked*: the shortcut channel itself.

Hidden states: `hidden_{menta,qwen3_base,osim8b}.npz` and
`daic_hidden_participant_alllayers.npz`, shape `(188 participants, 3 arms,
37 layers, hidden)`. Four models only; the four Omni checkpoints are not probed.

## Q1 — the signal is present at nearly every layer

Cohen's d is **cross-fitted** (`layerwise_multi.py:xfit_d`): the direction comes
from the training fold and the effect size is measured on the held-out fold,
with `GroupKFold(5)` on `participant_id` so a participant's two arms never
straddle the split. Fitting a direction to maximise separation and then
measuring separation on the same data is circular and always large; this avoids
that.

| model | peak d | peak layer | depth | AUC at peak | layers with d > 1 | % |
|---|---:|---:|---:|---:|---:|---:|
| MentaLLaMA-chat-7B | **2.93** | **10** | 0.31 | 0.9879 | 31/33 | 93.9 % |
| Menta | 3.33 | 36 | 1.00 | 0.9545 | 35/37 | 94.6 % |
| Qwen3-4B-base | **4.01** | **18** | 0.50 | 0.9983 | 35/37 | 94.6 % |
| OSim-8B | 3.25 | 21 | 0.56 | 0.9387 | 33/37 | 89.2 % |

Control task (arm labels flipped for a random half of participants, group
structure and class balance preserved): d = 0.00 to 0.24 everywhere.

**The correct range is d > 1 at 89–95 % of layers, not 92–97 %.** OSim-8B fails
at four layers (0–3), the others at two (0–1).

Two things are worth stating explicitly:

- **d = 0.00 at layer 0 for all four models.** Layer 0 is the embedding layer.
  If the separation were already there it would be the trivial observation that
  two different strings are different. It is not; the direction is computed.
- **The unadapted base carries it at least as strongly as the clinical
  fine-tune** (Qwen3-4B-base 4.01 vs Menta 3.33). Fine-tuning does not create
  this representation.

Per-layer shape differs by model and is not a shared property: MentaLLaMA peaks
early (L10) then plateaus at ~2.0; Menta rises monotonically to its final layer;
Qwen3-base and OSim-8B peak mid-network and decay to 1.84 and 1.23 by the output.

**A caveat we have not closed.** The two arms are different strings, so some
separability is expected on those grounds alone. What guards against the trivial
reading is the layer-0 zero and the control task, not the magnitude of d itself.
The dataset contains a third arm, `cf_neutral_matched` (token-length-matched
neutral context), which is **not used** in the d computation; probing
`cf_neutral` vs `cf_neutral_matched` would separate "clinical context" from "any
textual difference" and would settle it. Not run.

## Q2 — presence is not use, and only one model uses it

For each layer, `D = B − A` is the per-participant displacement between arms,
`u` its normalised mean. `partial_r` is the partial correlation between `D·u`
and the behavioural shift `Δp`, **controlling for ‖D‖** so a layer cannot win by
moving more; `ci` is its bootstrap 95 % CI; `random_r` is the same quantity
against a random direction drawn at that layer.

| model | significant layers (CI excludes 0) | random direction wins | mean \|r\| real | mean \|r\| random | verdict |
|---|---:|---:|---:|---:|---|
| **MentaLLaMA** | **15/32** | **5/32** | 0.150 | 0.067 | **coupled** |
| OSim-8B | 19/36 | 15/36 | 0.135 | 0.101 | artifact |
| Menta | 1/36 | 18/36 | 0.059 | 0.071 | no coupling |
| Qwen3-4B-base | 1/36 | 21/36 | 0.061 | 0.082 | no coupling |

Denominators exclude layer 0, where `partial_r` is `NaN`.

**Only MentaLLaMA clears its own random-direction control.** OSim-8B has *more*
significant layers than MentaLLaMA and a higher peak (+0.309 vs +0.264), and it
is noise: a random direction beats the real one on 15 of 36 layers. Reporting
the significant-layer count without the random column would rank OSim-8B first —
the exact error this project is about, reappearing at the representational
level. Menta and Qwen3-base sit at 1/36, precisely the α = 0.05 false-positive
rate.

### Encoding strength and behavioural relevance peak at different layers

For MentaLLaMA, d peaks at **layer 10** but coupling peaks at **layer 13**
(r = +0.264), with a contiguous significant band over **layers 8–21**. Above
layer 21 the signal is still strongly encoded (d ≈ 2.0–2.4) but no longer
predicts behaviour.

This dissociation determines where to intervene, and "where it is most
decodable" is the wrong answer — hence `phase3_leace.py:4-7`: *"signal STRENGTH
peaks at layer 10, but behavioural relevance peaks at layer 13 … We erase where
the signal relates to BEHAVIOUR, not where it is most decodable."*

One soft point to disclose: MentaLLaMA's **layer 1** is significant
(r = +0.264) with d = 0.34, i.e. a layer that barely encodes the concept
correlating strongly with behaviour, isolated between non-significant L2–L7. It
is most likely a token-level artifact and should be labelled as such rather than
counted.

## Q3 — causal erasure (LEACE)

Closed-form concept erasure in the amnesic-probing framework. For a binary
concept the eraser is **rank-one** — that is forced by the method, not an
implementation shortcut (`phase3_leace.py:13`). The shuffled-label direction,
fitted with the same machinery and the same covariance, is the control: removing
*any* rank-one direction from the residual stream costs something.

MentaLLaMA, erasing at layers 8–24 simultaneously:

| condition | shortcut sens | construct sens | SER | AUC |
|---|---:|---:|---:|---:|
| none | 0.2057 | 0.1765 | 1.165 | 0.7769 |
| **real direction** | **0.1580** | 0.1745 | 0.906 | 0.7664 |
| random, matched norm | 0.2068 | 0.1774 | 1.166 | 0.7767 |

**−23.2 % of the shortcut effect, with construct sensitivity and AUC
essentially untouched, and the random control doing nothing.** The single-layer
version (`phase3_leace_layer13.json`) removes only **6.3 %** (0.2057 → 0.1928).

### What the single-vs-multi gap does and does not show

It shows the shortcut is **not carried by one direction that could be found and
deleted once**. It does **not** show non-linearity, and the paper must not say
so: LEACE is a linear method and it recovered 23.2 %; a non-linearly encoded
concept would be invisible to it. The linear probe is already at AUC 0.94–0.99,
leaving no headroom for a non-linear probe to beat.

Three explanations fit the gap equally well and the erasure numbers alone cannot
separate them: (a) redundant copies outside the erased band — supported by
d > 1 at 89–95 % of layers; (b) a concept subspace of rank > 1; (c) the model
**recomputing** the direction downstream at every block. Only (c) is about
non-linearity, and even then it is non-linearity of the *computation*, not of
the *representation*.

`experiments/i13-erasure-recovery/` tests (c) directly: erase at one layer, then
probe every downstream layer. If d recovers, the direction is rebuilt; if it
stays at zero, it was merely transported.

The correct limitation sentence is the one already in `paper/main.tex:367` —
*"A shortcut encoded non-linearly would also be invisible to this analysis"* —
which is the opposite claim from "we found non-linearity" and is the one the
tools support.

## A note on `phase1_probes.json`

Every cell reports AUC 1.000 with `beats_lexical: false`, because the lexical
baseline is also 1.000. This is not a null; it is a ceiling on both sides, and
the comparison has no resolving power. Do not cite `beats_lexical` from this
file.

## Files

| file | what |
|---|---|
| `layerwise_multi.json` | per-layer d, control d, AUC, `rel_disp`, `partial_r`, CI, `random_r` — 4 models |
| `layerwise.json` | the same for MentaLLaMA at higher layer resolution |
| `phase3_leace_L8-24.json` | the reported erasure |
| `phase3_leace_layer13.json` | the single-layer version (6.3 %) |
| `phase1_probes*.json` | ceiling-saturated, see above |
| `hidden_*.npz`, `daic_hidden*.npz` | raw states, `(188, arms, layers, hidden)` |
