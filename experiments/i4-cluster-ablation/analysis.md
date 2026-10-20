# I4 — cluster-resolved ablation: a null

Run 2026-08-24. Replacement for the dead E4a. Code:
`code/run_cluster_ablation.py` · results:
`results/cluster_ablation_MentaLLaMA-chat-7B.jsonl` (1,353 rows) · logs:
`logs/i4_cluster_ablation.log`, `logs/i4_analysis.log`.

Strip each topical cluster's prompts from every transcript, leaving participant
lines and all other interviewer lines byte-identical, and measure the change.
n = 123 ID-carrying transcripts; baseline (nothing ablated) AUC = 0.699.

---

## Result: nothing localises

| ablated cluster | k | AUC | ΔAUC [95 % CI] |
|---|---|---|---|
| la_relocation | 7 | 0.747 | +0.048 [−0.046, +0.145] |
| **all_clustered** | **47** | 0.727 | **+0.027** [−0.089, +0.140] |
| travel_leisure | 4 | 0.711 | +0.012 [−0.059, +0.082] |
| affect_and_memory | 7 | 0.705 | +0.006 [−0.066, +0.083] |
| parenting | 4 | 0.692 | −0.008 [−0.056, +0.034] |
| generic_probe | 3 | 0.685 | −0.014 [−0.132, +0.103] |
| family | 1 | 0.682 | −0.017 [−0.093, +0.063] |
| life_events_decisions | 7 | 0.681 | −0.018 [−0.115, +0.077] |
| self_concept | 4 | 0.673 | −0.026 [−0.105, +0.049] |
| **clinical_care** | 10 | 0.665 | **−0.034** [−0.092, +0.020] |

**Not one confidence interval excludes zero.** This is a null, and it is
reported as one.

## Two things worth reading anyway

**1. Removing the clinical thread does not help — it trends the wrong way.**
`clinical_care` (the 10 prompts containing every BH-significant association,
geometric-mean OR 5.20) is the *worst* ablation at ΔAUC −0.034. If the shortcut
were carried by the clinically-loaded questions as a topical thread, deleting
them should have recovered a chunk of D1's +0.126. It recovers nothing.

**2. Removing all 47 scripted questions recovers +0.027; removing all
interviewer turns recovers +0.126.** The scripted questions are therefore not
where the damage lives. What D1 removes and this does not: backchannels,
unscripted interviewer lines, and the turn structure itself.

Taken with I1's D1m result — masking interviewer *words* while keeping the turns
recovers 0.763 of D1's 0.787 — the picture is that the harm is **distributed
across the interviewer's language generally**, not concentrated in any topical
subset of the scripted protocol. That is consistent with A2/I2's finding that
the model keys on surface clinical *vocabulary* rather than on the adaptive
structure: vocabulary is spread through everything Ellie says, including her
reactions, and does not respect the cluster boundaries.

## Why the null is weak evidence, not strong

- **n = 123** with per-cluster CIs spanning ±0.10 AUC. This experiment could not
  have detected an effect smaller than about 0.10 even if present, and D1's
  whole effect is 0.126. It is underpowered by construction.
- Only ID-carrying transcripts can be cluster-ablated, and that subset is not
  label-representative (F6: 22.8 % vs 29.3 % depression rate).
- Clusters vary from k=1 (`family`) to k=10 (`clinical_care`), so ablations are
  not comparable in magnitude of text removed.

## Status

E4a was replaced by this because protocol shift is not constructible from DAIC
(see `../suite/analysis.md`). The replacement returns a null. **The design now
has no positive localisation result and no OOD result** — E4b (IMHI → DAIC
transfer) is the only remaining route to RQ4, and it rests on L3, which is still
asserted rather than measured.
