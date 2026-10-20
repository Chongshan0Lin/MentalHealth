# I5 — P2 chain-of-thought: the lowest CF-Sens in the project, and a failure

Run 2026-08-24. Code: `code/run_p2.py` · results:
`results/p2_MentaLLaMA-chat-7B.jsonl` (564 rows) · logs: `logs/i5_p2_cot.log`,
`logs/i5_p2_analysis.log`. MentaLLaMA-chat-7B, greedy decoding (deterministic,
no seeds), label scored teacher-forced **conditioned on** the generated
reasoning — never parsed out of it.

---

## The methodological safeguard held

CoT is the one intervention that needs free generation, so the pre-registered
risk was that degenerate generation would correlate with perturbation strength
and void the comparison. It did not:

| arm | n | degenerate rate | mean generated tokens |
|---|---|---|---|
| cf_none | 188 | **0.000** | 117.6 ± 15.8 |
| cf_neutral | 188 | **0.000** | 110.8 ± 19.8 |
| cf_depression | 188 | **0.000** | 122.6 ± 12.5 |

Zero degeneracy everywhere, comparable lengths. The arms are clean and the
result below is interpretable.

---

> **Bottom line.** P2 fails, but not for the reason first recorded. It clears
> the paired construct-preservation criterion (F-A) and is rejected by the IID
> utility clause (F-B). The interesting output of this experiment is the
> demonstration that F-A alone is insufficient — see the F-A section below.

## P2 achieves the best CF-Sens of any intervention — by breaking the model

| intervention | CF-Sens [95 % CI] |
|---|---|
| P0 none | 0.2057 [0.1947, 0.2171] |
| P1 instruction | 0.0972 [0.0903, 0.1035] |
| **P2 CoT** | **0.0302** [0.0109, 0.0542] |

On the headline shortcut metric P2 wins outright — an **85 % reduction**, three
times better than P1. Judged on CF-Sens alone it would be the paper's lead
result.

It is not. Look at anything else:

| condition | AUC P0 → P2 | mean p(dep) | P(pred = Yes) |
|---|---|---|---|
| `cf_none` | 0.777 → **0.579** | 0.945 | **0.952** |
| `cf_neutral` | 0.764 → **0.660** | 0.973 | **0.979** |
| `cf_depression` | 0.734 → **0.628** | 0.973 | **0.979** |

**CoT makes the model answer "depressed" to 95–98 % of participants.** Mean
probability sits at 0.95–0.97 in every arm. Discrimination collapses by up to
0.198 AUC — on `cf_none` the model falls from 0.777 to barely above chance.

CF-Sens is low because the score is **pinned near the ceiling and nothing can
move it**, not because the model stopped using the interviewer.

This looks like the flattening failure `INTERVENTION_DESIGN.md` §6 F-A was
written to catch — and the next section shows that **F-A does not catch it**.
What rejects P2 is F-B's IID-utility clause (`ΔAUC_IID` not significantly down;
it is down 0.198).

---

## P2 **passes** F-A — and that is a finding about the criterion, not about P2

E5 was subsequently scored under CoT (`code/run_e5_cot.py`,
`results/e5_cot_MentaLLaMA-chat-7B.jsonl`, 400 generations, degeneracy 0.007).

The prediction recorded in that script *before* it ran was that construct
sensitivity would collapse alongside CF-Sens, confirming flattening on the
criterion built to detect it. **That prediction was wrong.**

| variant | construct sens. [95 % CI] | pair acc | mean p | CF-Sens |
|---|---|---|---|---|
| base | 0.1154 [0.1037, 0.1275] | 0.700 | 0.423 | 0.2057 |
| P1 | 0.1062 [0.0953, 0.1176] | 0.680 | 0.422 | 0.0972 |
| D1 | 0.2033 [0.1864, 0.2212] | 0.765 | 0.480 | 0 |
| D1m | 0.1866 [0.1703, 0.2037] | 0.750 | 0.476 | 0 |
| **P2 CoT** | **0.0653** [0.0395, 0.0957] | **0.610** | **0.965** | 0.0302 |

| intervention | shortcut drop | construct drop | margin [95 % CI] | F-A |
|---|---|---|---|---|
| P1 | −52.7 % | −8.0 % | +44.7 % [+38.7 %, +51.1 %] | PASS |
| D1 | −100 % | +76.1 % | +176.1 % [+160.6 %, +193.9 %] | PASS |
| D1m | −100 % | +61.7 % | +161.7 % [+144.6 %, +180.6 %] | PASS |
| **P2** | **−85.3 %** | **−43.4 %** | **+41.9 %** [+19.0 %, +66.9 %] | **PASS** |

Construct sensitivity did fall hard — 0.1154 → 0.0653, a 43 % drop — but CF-Sens
fell harder, 85 %. F-A is a **ratio** test, so P2 clears it, with a margin
statistically indistinguishable from P1's.

### The hole this exposes

**F-A catches proportional flattening. It does not catch asymmetric
degradation.** A model can fall apart in absolute terms and still pass, provided
its shortcut sensitivity falls somewhat faster than its construct sensitivity.
P2 is exactly that model:

- mean p(depression) = **0.965** — it answers "depressed" to almost everyone
- pairwise accuracy **0.610** vs base 0.700 — it ranks patients *worse*
- IID AUC **0.579** vs 0.777 — near chance

None of those enter F-A, because F-A never looks at an absolute level.

**F-B is what rejects P2**, via its `ΔAUC_IID` clause. So the two criteria
disagree, and the disagreement is the useful part: **the paired criterion is
necessary but not sufficient, and the design was right to require both.** Had
F-A been the only gate — as the draft plan's metric list effectively proposed —
P2 would have been promoted alongside P1 on a margin of +41.9 %.

**Verdict: P2 FAILS, on F-B.** The earlier rejection stands; what changed is
that it is now known to be F-B and not F-A that does the work, and that is a
result about the evaluation design worth reporting in the paper.

---

## The qualitative finding is the most legible evidence in the project

The CoT traces are readable, and they show the mechanism directly. Counting
whether the generated reasoning mentions a clinical topic
(diagnosis / PTSD / therapy / sleep / temper):

| arm | CoT mentions clinical topic | mentions biographical topic |
|---|---|---|
| cf_none | 0.612 | 0.027 |
| cf_neutral | 0.628 | 0.021 |
| **cf_depression** | **0.862** | 0.021 |

**The participant's words are byte-identical across all three arms.** The
+23-point jump comes entirely from what the *interviewer* asked.

And the model states the confusion outright. From a `cf_depression` trace:

> "**The poster mentions being diagnosed with depression and having a
> therapist.** This suggests that they have been seeking professional help for
> their mental health."

The poster mentions no such thing. **Ellie** asked *"have you been diagnosed
with depression"*. The model has attributed the interviewer's question to the
participant and then reasoned from it as though it were disclosed evidence.

This is the shortcut, in the model's own words. Everything else in the project
infers the mechanism from probability shifts; this exhibits it. It is the
strongest illustrative material available for the paper, and it comes from the
intervention that failed.

---

## Solid / not solid

**Solid:**
- Zero degeneracy in all arms; lengths comparable. The arm is not confounded.
- P2 CF-Sens 0.0302, the lowest measured, CI excluding P1's.
- Saturation is unambiguous: P(pred=Yes) ≥ 0.95 in every arm.
- AUC drop of 0.198 on `cf_none`, CIs non-overlapping ([0.706, 0.844] vs
  [0.488, 0.669]).
- Clinical-topic attribution 0.628 → 0.862 with participant text held
  byte-identical.

**Not solid:**
- **P2 passes F-A and is rejected only by F-B.** The rejection is sound but
  rests on a single clause (IID utility). A reviewer who weights the paired
  criterion above absolute utility could argue for P2, and the paper should
  address that head-on rather than presenting the rejection as unanimous.
- Construct sensitivity under CoT has a wide CI ([0.0395, 0.0957]) — the
  saturated regime is noisy, so the −43.4 % drop is less precisely estimated
  than the other rows.
- The clinical/biographical keyword counters are regex heuristics over
  free text, not annotation. They are indicative, and the quoted trace is the
  actual evidence.
- One CoT trigger phrase, one decoding strategy, one model. A different prompt
  might not saturate.
