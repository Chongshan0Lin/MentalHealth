# I7 — Amnesic probing: does the interviewer signal causally drive the shift?

**Status:** proposed, not run. Written before any data is collected.
**Method basis:** amnesic probing \[Elazar et al. 2021, arXiv:2006.00995\];
erasure via LEACE \[Belrose et al. 2023, arXiv:2306.03819\] rather than INLP,
following \[Dobrzeniecka et al. 2025, arXiv:2506.11673\], which compares INLP,
mean projection and LEACE for exactly this intervention. All three verified via
arXiv API and added to `paper/refs.bib`.

This experiment exists because `paper/main.tex` §4.4 reports a probing result
that does not support the claim it sits beside. Either this runs, or §4.4 is
deleted.

---

## 0. The correction that defines the experiment

**Do not reuse the existing corpus-identity probe.** It was trained on hidden
states from IMHI and SDCNL — verified: `grep -c daic runs/c1_*.jsonl` returns
**0** — and asks a *between-corpus* question. The counterfactual effect occurs
*within* DAIC, where the corpus never changes and only the interviewer's
questions do.

| | existing C1 probe | what we need |
|---|---|---|
| corpus | IMHI / SDCNL | **DAIC** |
| question | which dataset is this? | **biographical or clinical interviewer context?** |
| granularity | between-corpus | **within-corpus** |
| supports | Form 3 (corpus prior) | **Forms 1–2 (question selection, vocabulary)** |

The probe target is redefined as: *given a hidden state, was this transcript
presented with the biographical or the clinical interviewer pool?* That is the
variable the counterfactual actually manipulates.

---

## 1. Phase 1 — Train the right probe

**Data.** The 188 DAIC participants under `cf_neutral` and `cf_depression`,
matching the arms whose behavioural effect is already measured in
`a2b_MentaLLaMA-chat-7B.jsonl`. 376 forward passes with hidden-state capture.

**Layers.** 8, 16, 24, 32 (final) of the 32-layer model, plus mean-pooled and
last-token variants — the same grid C1 used, so the two probes are comparable.

**Probe.** Logistic regression, label = arm.

**Two mandatory controls.**

1. **Control task (Elazar et al.).** Train an identical probe on the same
   representations with **randomly shuffled labels**. This yields a direction
   matched in dimensionality, sample size and training procedure — the correct
   baseline for "what does removing this many degrees of freedom cost by
   itself." This is stronger than ablating a random unit vector, which matches
   only the norm.
2. **Lexical ceiling.** TF-IDF + logistic regression on the raw text. Decoding
   the arm is close to trivial from surface tokens alone — a probe could succeed
   by detecting the word *diagnosed* — so the hidden-state probe must exceed
   this or there is no representational claim to make.

**Grouped cross-validation.** Every participant contributes one item to *each*
class, so ordinary CV would place the same participant in train and test. Use
leave-one-participant-out.

**Gate.** If no layer exceeds the lexical ceiling with a CI excluding it, this
context signal is not linearly separable beyond surface vocabulary. That is a
publishable negative result on its own. Stop; delete §4.4.

---

## 2. Phase 2 — Does the represented signal track behaviour?

**Procedure.** For each participant *i* and layer *ℓ*:

```
Δh_i(ℓ) = ŵ_ℓ · ( h_clinical_i(ℓ) − h_biographical_i(ℓ) )
Δp_i    = p(dep | clinical_i) − p(dep | biographical_i)      # already on disk
```

Report Pearson and Spearman correlation of `Δh` with `Δp`, per layer, with
bootstrap CIs and a scatter plot at the strongest layer. Compare across depth:
does the correlation grow, peak, or vanish?

**The control that makes this interpretable.** `Δh` and `Δp` come from the same
forward pass, so they correlate for trivial reasons — a participant whose
representation moved a lot will show a prediction that moved a lot regardless of
direction. **The headline is the partial correlation controlling for total
displacement** `‖h_clinical_i − h_biographical_i‖`. A raw correlation may be
shown alongside, never alone.

**Gate.** Significant partial correlation at one or more layers (r > 0.3,
CI excluding zero) → proceed. All layers null — as happened for the
corpus-prior route, where the analogous proxy correlated at r = 0.07 — → report
the null honestly and stop. Do not force the causal stage.

---

## 3. Phase 3 — Causal erasure (only if Phase 2 is positive)

**Method.** LEACE, closed-form, applied once at the selected layer. INLP is
avoided deliberately: its iterative nullspace projection needs many rounds and
damages unrelated information, and Dobrzeniecka et al. find closed-form erasure
cleaner for this intervention. Mean projection only zeroes the first moment
along the direction; LEACE guarantees the concept is no longer linearly
recoverable.

**Procedure.**
1. At the layer selected in Phase 2, apply LEACE to erase the probe direction.
2. **Continue the remaining forward pass from that layer** — no retraining.
3. Measure `Δp_amnesic`: how much of the biographical-vs-clinical prediction gap
   survives erasure.
4. **Apply the identical erasure to the control-task direction** from Phase 1,
   giving `Δp_control` — the collateral damage from removing the same number of
   degrees of freedom carrying no real information.

**Decision rule.**

| Observation | Conclusion |
|---|---|
| `Δp_amnesic ≪ Δp_original` **and** `Δp_amnesic ≪ Δp_control` | **Causal.** This direction mediates the behavioural effect. |
| `Δp_amnesic ≈ Δp_control` | Not causal. The drop is collateral damage from removing any information. |

Without the control-task comparison the causal conclusion does not stand. This
is the single most important methodological element of the design.

**A third control we add beyond Elazar's protocol.** Report **SER** — shortcut
sensitivity divided by construct sensitivity — on the frozen 200-pair set,
before and after erasure. We already have two cases in this project
(chain-of-thought, product-of-experts) where a method drove shortcut sensitivity
toward zero by destroying the model. Erasure is vulnerable to the same failure.
Success requires `S-sens` to fall **while `C-sens` holds**, and AUC on `cf_none`
to be preserved.

---

## 4. Layer scanning and budget control

Do not run Phase 3 at every layer. Use Phase 2 to select the **one or two**
layers with the strongest partial correlation, and add **one randomly chosen
middle layer** as a spatial control. Whether the causal effect concentrates at
particular depths is itself informative: if it appears only late, the feature is
integrated near the decision rather than early in the stack.

---

## 5. Cost

| Phase | Compute | Wall-clock |
|---|---|---|
| P1 extraction | 376 passes with hidden-state capture | ~30 min |
| P1 probes | CPU, LOO over 4 layers × 2 pools + control + TF-IDF | ~25 min |
| P2 | CPU, reuses P1 outputs and existing log-probs | ~5 min |
| P3 | LEACE closed-form + 188 × 2 arms × (real + control + random layer) | ~1 h |

Hidden states: 376 × 4 layers × 2 pools × 4096 × 4 bytes ≈ **49 MB**.

**Total: well under one day on a single GPU.** LEACE being closed-form is what
keeps Phase 3 cheap; INLP would have multiplied it.

---

## 6. What each outcome licenses

| Outcome | Claim | Paper |
|---|---|---|
| P1 fails | not linearly separable beyond vocabulary | delete §4.4 |
| P2 fails | represented, not tracking behaviour | §4.4 → two-line null |
| P3 ≈ control | correlational only | §4.4 reports correlation, explicitly not causation |
| P3 ≪ control, SER improves | **linearly represented direction causally mediates the effect** | §4.4 becomes a positive mechanistic result |

**The bonus if P3 succeeds.** Our mitigation taxonomy has four levels — input,
inference, output, training — and **no representation-level method**. A LEACE
erasure that improves SER while preserving AUC is exactly that, and fills the
one structural gap in the taxonomy.

---

## 7. Pre-committed reporting

Fixed before data exists:

- The **lexical ceiling** is reported at every layer whether or not the probe
  beats it. C1's final-layer margin was zero and the first draft omitted it;
  that will not recur.
- The **partial** correlation is the headline for Phase 2.
- The **control-task erasure** appears in the same table as the real erasure,
  never in an appendix.
- If any gate fails, **§4.4 is cut, not softened.** A section reporting
  decodability while implying use is the exact error this paper criticises.

---

## 8. Honest boundaries

**This is linear evidence only.** If the shortcut is encoded non-linearly,
linear probes and LEACE cannot detect it, and a null in Phase 1 or 2 does not
prove the information is absent — only that it is not linearly accessible.
Non-linear probes or circuit-level analysis would be needed, and are out of
scope here.

**Success is local, not global.** A positive Phase 3 shows that *this direction,
at this layer, mediates this behaviour*. It does not establish that the model's
shortcut mechanism as a whole works this way, and the paper must not extrapolate
that far.

**It produces no out-of-distribution evidence.** Everything remains one corpus,
one interviewer, one protocol. This strengthens the mechanistic claim and does
nothing for the generalisation limitation, which stays the paper's central
weakness.
