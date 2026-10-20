# I2 — E3 token-matched arms + P1 anti-shortcut prompting

Run 2026-08-24. Steps 4 and 6 of `INTERVENTION_DESIGN.md` §7.
Code: `code/run_i2.py` · results: `results/i2_MentaLLaMA-chat-7B.jsonl`,
`results/e3_pools.json` · logs: `logs/i2_e3_p1.log`, `logs/i2_analysis.log`.
MentaLLaMA-chat-7B, 940 rows, teacher-forced label log-likelihood.

---

## Part A — the "wrong half" dissociation is now **established**

A2's sharpest claim was that the model responds to questions that merely
*sound* clinical more than to questions that genuinely predict the label. A2's
own analysis called it *suggestive, not established*, because the comparison
set a 31-word pool against a 38-word one.

### The matched pools

Selected mechanically, never hand-picked: length is a hard constraint (±1 word
of the `cf_depression` reference) and association is the objective, so the
selector cannot trade one confound for the other.

| pool | words | geometric-mean OR |
|---|---|---|
| `cf_depression` (reference) | 38 | clinically-sounding, not selected on OR |
| `cf_neutral_matched` | 38 | **1.00** |
| `cf_empirical_matched` | 37 | **7.85** |

`cf_empirical_matched` retains four of the original top-five prompts
(`why_seek_help` 11.26, `therapy_going` 10.42, `trigger` 8.50,
`when_diagnosed` 8.24). An earlier version of the selection rule matched on
length alone and silently swapped those for OR-2.3 prompts — trading a length
confound for an association confound. Fixed before running.

### Result

All arms within one word of each other, participant text byte-identical:

| contrast | Δp(depression) [95 % CI] | prediction flips |
|---|---|---|
| clinical-sounding − neutral | **+0.1932** [+0.1805, +0.2056] | 0.543 |
| actually-predictive − neutral | **+0.0849** [+0.0768, +0.0929] | 0.197 |
| **clinical − actually-predictive** | **+0.1083** [+0.0981, +0.1187] | 0.378 |

**Questions that sound clinical move the model 2.3× more than questions that
actually carry the diagnosis**, with length equalised and association strength
controlled. The key contrast's CI excludes zero by a wide margin.

Note the direction of the residual length effect: `cf_empirical_matched` is
*longer* than `cf_empirical` (37 vs 31 words) yet produces a *smaller* shift
(+0.085 vs +0.124 against its respective baseline). Length is not driving any
of this — if anything it runs against the effect, exactly as it did for the A2
headline.

**This promotes the project's sharpest result from suggestive to established.**
The model exploits the interviewer channel and keys on surface clinical
vocabulary while largely missing the adaptive-interview structure that actually
predicts the label.

---

## Part B — P1 anti-shortcut prompting: halves the shortcut, at a cost

Instruction fixed verbatim in the design before any result was seen.

### It works on the target metric

| | CF-Sens (clinical vs biographical) [95 % CI] |
|---|---|
| P0 no instruction | **0.2057** [0.1947, 0.2171] |
| **P1 instructed** | **0.0972** [0.0903, 0.1035] |

**A 53 % reduction in counterfactual sensitivity**, with non-overlapping
confidence intervals. Simply telling the model to ignore the interviewer's
choice of questions halves how much those questions move it.

The operating point improves too, where it was worst: under clinical context
`P(pred=Yes)` falls 0.745 → 0.543.

### But discrimination does not improve, and IID drifts

| condition | AUC P0 | AUC P1 | P(Yes) P0 → P1 |
|---|---|---|---|
| `cf_none` (real transcript) | 0.777 | **0.752** | 0.351 → 0.441 |
| `cf_neutral` | 0.764 | **0.746** | 0.160 → 0.282 |
| `cf_depression` | 0.734 | 0.733 | 0.745 → 0.543 |

AUC is flat-to-slightly-worse everywhere (−0.025, −0.018, −0.001), all well
inside their CIs. The instruction compresses the model's response toward the
middle rather than sharpening it: every arm's operating point moves toward
~0.4–0.5.

### The verdict is **pending E5**, and this is exactly why E5 exists

Halving CF-Sens while leaving AUC flat is the signature that
`INTERVENTION_DESIGN.md` §6 F-A pre-commits to treating with suspicion. Two
readings fit the numbers equally well so far:

1. **Genuine debiasing** — the model stops attending to interviewer context and
   keeps using participant evidence.
2. **Flattening** — the model becomes less responsive to its input in general,
   and CF-Sens falls only because *everything* falls.

**Nothing measured in this experiment can distinguish them.** The construct-
preservation set (E5) can: it varies the participant with the interviewer held
fixed, so it measures whether the model still tracks the *evidence*. E5 was
launched with `--with-p1` for precisely this contrast.

Per F-A, P1 may claim success only if its relative CF-Sens drop (−53 %) exceeds
its relative construct-sensitivity drop by a margin whose CI excludes zero. If
construct sensitivity also falls by ~50 %, P1 is a **failure** however good the
CF-Sens number looks.

---

## Status

| item | state |
|---|---|
| E3 token-matched arms | ✅ dissociation **established**, +0.1083 [+0.0981, +0.1187] |
| P1 anti-shortcut prompting | ⚠️ CF-Sens −53 %, AUC flat — **verdict pending E5** |
| P2 CoT | ⬜ not run — needs a generation path the harness lacks |

P1 is the **5th intervention tested** and the first that is neither a data edit
nor a threshold correction.
