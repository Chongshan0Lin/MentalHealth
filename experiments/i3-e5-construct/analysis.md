# I3 / E5 — construct preservation, and every intervention's verdict

Run 2026-08-24. Code: `code/run_e5.py`, `code/run_e5_variants.py` · results:
`results/e5_MentaLLaMA-chat-7B.jsonl`, `results/e5_variants_MentaLLaMA-chat-7B.jsonl`
(800 rows each) · logs: `logs/i3_e5.log`, `logs/i3_e5_variants.log`,
`logs/i3_e5_analysis.log`, `logs/i3_fa_all.log`. MentaLLaMA-chat-7B, 200 pairs.

The mirror of A2: interviewer context held **byte-identical**, the participant
swapped. `construct sensitivity = mean |p(Y|C_i,S_j) − p(Y|C_i',S_j)|` —
**higher is better**, because the model *should* move when the evidence changes.

---

## Headline: D1 does not merely preserve the construct signal, it doubles it

| variant | construct sens. [95 % CI] | pairwise acc | CF-Sens |
|---|---|---|---|
| base | 0.1154 [0.1037, 0.1275] | 0.700 | 0.2057 |
| P1 | 0.1062 [0.0953, 0.1176] | 0.680 | 0.0972 |
| **D1** delete interviewer turns | **0.2033** [0.1864, 0.2212] | **0.765** | 0 † |
| **D1m** mask interviewer words | **0.1866** [0.1703, 0.2037] | 0.750 | 0 † |

† zero **by construction** — the E5 arms differ only in interviewer text, which
D1 deletes and D1m neutralises.

`INTERVENTION_DESIGN.md` §6 **F-A**, fixed before any of this was measured:

| intervention | shortcut drop | construct drop | margin [95 % CI] | verdict |
|---|---|---|---|---|
| P1 | −52.7 % | −8.0 % | +44.7 % [+38.4 %, +51.2 %] | **PASS** |
| **D1** | −100 % † | **+76.1 %** | +176.1 % [+160.3 %, +193.9 %] | **PASS** |
| **D1m** | −100 % † | **+61.7 %** | +161.7 % [+145.5 %, +180.1 %] | **PASS** |

### How much credit the margin deserves

**Half of D1's margin is unearned and must be said so.** CF-Sens = 0 is
trivially true: nothing can be sensitive to a channel that has been deleted.
Reporting "−100 % shortcut reliance" as an achievement would be circular.

**The construct-sensitivity increase is not trivial, and it is the real
result.** Nothing forced it upward. The opposite was the live risk — deleting
the interviewer removes context, and a model that needed that context to
interpret the participant should have got *worse*. Instead sensitivity to the
participant rose 76 %, and pairwise accuracy with it (0.700 → 0.765).

> Base state: the model responds to the **interviewer** (0.206) **1.8× more
> strongly** than to the **patient** (0.115). Delete the interviewer and its
> response to the patient rises to 0.203 — approximately what it had been
> spending on the interviewer.

**The side channel was crowding out the evidence channel.** That is a
mechanistic claim about *why* the shortcut is harmful, not merely that it is,
and it is the strongest single finding in the project.

The D1 > D1m ordering (0.2033 vs 0.1866) reproduces their AUC ordering from I1
(0.787 vs 0.763) on a completely independent measurement.

### Consequence for the paper

**D1 beats P1 on both halves of the criterion.** P1 halves shortcut reliance at
a cost of 8 % of construct sensitivity; D1 eliminates it while *gaining* 76 %,
and is simpler and label-free. D1 is the lead result; P1 is the interesting
secondary one, because it shows a plain instruction achieves a large part of
the same effect without touching the input.

---

## The finding underneath the verdict

Read the two baselines against each other:

| | base value |
|---|---|
| CF-Sens — how much the **interviewer** moves the model | **0.2057** |
| construct sens. — how much the **patient** moves the model | **0.1154** |

**The model is 1.8× more sensitive to who is asking than to who is answering.**

That is a cleaner statement of the pathology than anything in A2, and it needed
E5 to say it. A2 established that interviewer context moves the prediction a
lot; only the mirror experiment establishes that it moves it *more than the
clinical evidence does*. Both numbers come from the same model, same corpus,
same scoring, differing only in which half of the input is held fixed.

Pairwise accuracy of 0.700 confirms the model is not simply ignoring the
patient — it ranks the depressed participant above the non-depressed one in 70 %
of pairs. It attends to both channels; it just weights the wrong one more.

---

## Solid / not solid

**Solid:**
- Construct sensitivity barely moves under P1 (−8.0 %), CIs heavily overlapping,
  while CF-Sens halves. The margin CI excludes zero by a wide margin.
- Base CF-Sens (0.206) > base construct sensitivity (0.115), paired within the
  same model and scoring path.

- D1 and D1m raise construct sensitivity well outside the base CI
  (0.1864–0.2212 and 0.1703–0.2037 against 0.1037–0.1275), and raise pairwise
  accuracy with it.

**Not solid:**
- Pairwise accuracy 0.700 → 0.680 under P1 is within noise and should not be
  read as a cost.
- **D1/D1m change the stimulus, not only the decision rule.** E5 varies the
  participant with the interviewer fixed; D1 removes the interviewer, so its
  construct sensitivity is measured on a *different input distribution* from
  the base condition. The comparison is the right one to make — it is what the
  deployed system would see — but it is not a like-for-like decision-rule swap
  the way P1 is.
- E5 items are **synthetic transcripts** — participant *i*'s turns spliced under
  participant *j*'s interviewer context. They read as coherent interviews only
  to the extent that turn counts match (median |Δ| = 7 turns, max 15). A model
  sensitive to that seam would inflate construct sensitivity. Note this cuts
  *against* D1, whose items have no seam at all.
- One model. The encoder arm has no E5 measurement, so the construct-preservation
  criterion has been applied only to the LLM.
- **P3-c was not scored separately.** I1 §4 established it reduces algebraically
  to the participant-only score, so its E5 numbers are D1's by construction.
  Reporting it as an independent pass would be double-counting.
