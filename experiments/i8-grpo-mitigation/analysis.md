# I8 — GRPO post-training: a failed experiment that is worth reporting

Run 2026-08-27/30. Code: `code/{build_pairs,train_grpo,eval_grpo,length_sweep}.py`
· results: `results/` · logs: `logs/i8_*.log`.
Design: `PLAN.md`, adapted from the PI's RL draft after
\[Zhong et al. 2026, arXiv:2606.09092\].

**The experiment did not answer the question it was built for.** Every reward
configuration degenerated before the invariance reward could be measured. What
it did produce is the cleanest demonstration in this project that shortcut
metrics reward a broken model.

---

## 1. What was run

| arm | reward | intended role |
|---|---|---|
| **M0** | `R_accuracy` | diagnostic — does vanilla RLVR amplify the shortcut? |
| **M1** | `R_accuracy + λ·R_inv` | mitigation — does an invariance reward reduce it? |
| **M0_bal_batch** | class-balanced `R_accuracy`, `scale_rewards="batch"` | added after M0 collapsed, to restore a non-degenerate baseline |

Identical data, backbone (`MentaLLaMA-chat-7B` + LoRA r=16), seed and
hyper-parameters (rollout 8, β=0, lr 2e-5, 2 epochs) across arms. 107 DAIC
training participants, paired with a counterfactual arm each epoch → 214 pairs.

**`R_format` was dropped** (§4 below). **Input truncated to 1200 words**, chosen
by a length sweep: the counterfactual effect is *stronger* there (Δp +0.290)
than at full length (+0.205), and 16 concurrent rollouts fit in one 48 GB card.

---

## 2. Result: all three arms collapse to a constant answer

Training trajectory, mean `P(pred = Yes)` over rollouts (data base rate 0.280):

| arm | steps 0–20 | 20–60 | 60–120 | 120–180 | 180+ |
|---|---|---|---|---|---|
| M0 | 0.486 | 0.232 | 0.010 | 0.002 | **0.000** |
| M1 | 0.498 | 0.258 | 0.010 | 0.002 | **0.000** |
| M0_bal_batch | 0.476 | 0.269 | 0.008 | 0.001 | **0.000** |

Final `p_yes` was identical to sixteen decimal places across all three arms
(0.001488095 = 1/672: exactly one rollout out of 672 said "Yes").

Held-out evaluation, teacher-forced scoring, all 188 participants:

| | baseline | M0 | M1 | M0_bal_batch |
|---|---|---|---|---|
| **P(pred=Yes)** | 0.394 | **0.000** | **0.000** | **0.000** |
| CSS (shortcut) | 0.2864 | 0.0017 | 0.0014 | **0.0000** |
| construct sensitivity | 0.1354 | 0.0025 | 0.0033 | **0.0006** |
| SER | 2.115 | 0.687 | 0.434 | — |
| IID AUC | 0.763 | 0.725 | 0.710 | **0.675** |
| conflict AUC | 0.069 | 0.266 | 0.277 | — |

Training was genuinely effective — the three LoRA adapters have distinct weight
hashes and all three differ behaviourally from the baseline. They simply
converged on the same degenerate policy.

---

## 3. The finding: a useless model tops every shortcut metric

Read only the columns the mitigation literature reports:

- **CSS falls 99.4 %** (0.2864 → 0.0017), far beyond anything else in this
  project — chain-of-thought, the previous best, managed −85 %.
- **SER improves** (2.115 → 0.434).
- **Conflict AUC improves 4×** (0.069 → 0.277).

On those three numbers this is the strongest mitigation we have produced. The
model answers **"not depressed" to all 188 patients**.

CSS is near zero not because the model stopped attending to the interviewer but
because it stopped attending to anything. Construct sensitivity fell 98 % at the
same time. Conflict AUC "improved" for the same reason: the baseline's ranking
is *inverted* by the shortcut (0.069), while a constant predictor has no ranking
to invert, so its AUC drifts back toward 0.5. **That is not recovery, it is
absence.**

This is the fourth instance of the same failure in this work — after
chain-of-thought (AUC 0.579), product-of-experts (0.540) and the suffix format
instruction (0.670) — and the first we induced deliberately rather than
discovered.

### It also breaks SER, the metric we proposed

SER = CSS / construct sensitivity was introduced to catch flattening. It catches
**asymmetric** degradation, which is what chain-of-thought does. It does **not**
catch **total** degradation: when both terms approach zero the ratio is 0/0 and
improves meaninglessly. M0_bal_batch has CSS exactly 0.0000 and no defined SER
at all.

**SER is necessary but not sufficient, and must always be reported with an
absolute utility number.** That is a limitation of our own metric and belongs in
the paper.

---

## 4. Three design decisions, recorded because they were not obvious

**`R_format` was dropped.** Evaluation teacher-forces the label and never
parses, so a format reward buys nothing. Meanwhile I9 measured that *any*
mention of output format in the prompt costs 18 % of the counterfactual effect
at full length and 72 % under the truncation RL requires. Keeping it would have
trained both arms in a regime where the phenomenon was already suppressed.

**Class weighting was a no-op, and finding out why mattered.** GRPO normalises
advantages within each group, `A_i = (r_i − mean)/std`. All rollouts of one
prompt share a label, so a class weight is constant within the group and cancels
exactly. Worse, once the policy is constant, groups it gets uniformly wrong have
zero reward variance and therefore **zero gradient** — the minority class goes
silent and the collapse self-reinforces. `scale_rewards="batch"` was supposed to
fix this by giving those groups a negative advantage. It did not.

**An evaluation bug was caught by a reproduction check.** The first eval still
appended a `<think>…</think><answer>` scaffold left over from before `R_format`
was dropped, which suppressed baseline CSS from 0.286 to 0.066 and made all four
models look alike. After the fix the baseline reproduced the length sweep's
1800-word value exactly (0.2864 vs 0.2864). The rule that caught it — always
re-derive a known number before trusting new ones — is worth keeping.

---

## 5. What this does and does not establish

**Does not establish:** whether `R_inv` mitigates shortcut learning. Every arm
degenerated before the comparison could be made. The experiment's stated
question is unanswered.

**Does not distinguish** two explanations of the collapse:
1. reward design is irrelevant here;
2. 107 participants with 30 positives contain no learnable signal.

The second is the more likely and the more mundane, and nothing we ran separates
them.

**Does establish**, within one backbone and one seed, that a degenerate policy
scores as the best mitigation in this study on every shortcut metric we report.

---

## 6. How to write it

Not as an RL mitigation result. As a short cautionary subsection:

> We attempted to train interviewer-invariance with GRPO. Under three reward
> configurations the policy collapsed to a constant majority-class answer, so the
> effect of the invariance reward could not be measured. The collapsed model
> nonetheless posts the best shortcut-reduction numbers in this paper — CSS −99 %,
> SER improved, conflict AUC quadrupled — while answering "not depressed" to
> every patient.

Reporting the first three numbers without the fourth would be exactly the error
this paper is about.

**To actually answer the RL question** would need a different setup: IMHI-scale
training data, or a smaller backbone where 107 examples are not hopeless. Both
are new experiments, not hyper-parameter changes.
