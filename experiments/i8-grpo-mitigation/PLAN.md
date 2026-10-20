# I8 — GRPO post-training to mitigate the interviewer shortcut

Adaptation of the PI's RL draft to this project's data, metrics and prior
findings. Method basis: RLVR/GRPO for shortcut-robust post-training
\[Zhong et al. 2026, arXiv:2606.09092\]. All four references verified via arXiv
API and added to `paper/refs.bib`.

**Status:** designed, not run. Four adaptations are flagged below; one of them
is a hard problem with the reward as drafted.

---

## 0. What carries over unchanged

The draft is adopted as written for: two arms only (M0 vs M1), identical data /
backbone / hyperparameters between them so any difference is attributable to
`R_inv`, GRPO with group-relative advantage, `β = 0`, rollout size 8, lr 2e-5,
2 epochs, `λ = 1` with no sweep.

Counterfactual pairs come free: every participant already has four interviewer
arms on disk (`cf_none`, `cf_neutral`, `cf_depression`, `cf_empirical`) plus two
length-matched ones, built by `run_a2b.py` and `run_i2.py` and frozen in
`SUITE_FREEZE.json`. `x_i'` is the same participant under a different arm, with
symptom evidence byte-identical and the label unchanged — exactly the
construction the draft asks for.

---

## 1. Adaptation A — `R_inv` as drafted is gameable, and this is the paper's own
recurring failure mode

**The exploit.** `R_inv = 1` when the model predicts the same label for `x_i`
and `x_i'`. A model that answers "not depressed" to everything is perfectly
consistent, so it collects `R_inv = 1` on every pair. With DAIC's base rate it
also collects `R_accuracy = 0.72`. Total reward:

```
R_format 1.00  +  R_accuracy 0.72  +  λ·R_inv 1.00  =  2.72 / 3.00
```

A constant predictor reaches 91 % of the maximum reward. That is a strong
attractor, and it is precisely the behaviour the draft's own fourth reference
describes \[arXiv:2604.15149\].

**Why we are confident this is the risk and not a hypothetical.** This project
has already measured the identical failure three times under different
mechanisms:

| method | shortcut metric | what it actually did |
|---|---|---|
| chain-of-thought | best CF-Sens of any method ($-85\%$) | answered "depressed" to 98 % of patients; AUC 0.777 → 0.579 |
| product-of-experts | lowest CF-Sens measured (0.041 → 0.004) | AUC 0.722 → 0.540 |
| (LEACE erasure) | pending | third instance under test |

Every one of them minimised the shortcut objective by ceasing to discriminate.
`R_inv` rewards exactly that quantity, directly, in the training signal.

**Decision: run M1 as drafted, but instrument it and pre-commit a void
criterion.** We do not change the PI's reward. We add:

1. **Monitoring.** Log `P(pred = Yes)` over rollouts at every step for both
   arms. This is the diagnostic, and it costs nothing.
2. **Pre-registered void criterion.** If M1's rollout predictions exceed **90 %
   one class** at convergence, `R_inv` has been hacked and M1's headline numbers
   are void regardless of how good CSS looks.
3. **A fallback already specified**, run only if the criterion trips:
   **M1′** with `R_inv = 1` iff the pair is *consistent **and** correct*. This
   removes the constant-prediction attractor with a one-line change while
   keeping the invariance intent.

---

## 2. Adaptation B — accuracy → AUC on the conflict set

The draft asks for *shortcut-conflict accuracy*. On our frozen conflict set
(E2b, 188 items) a constant "not depressed" predictor scores **0.702**, and the
majority baseline on the test split is 0.707. Accuracy cannot distinguish a
debiased model from a degenerate one — the reason every headline number in this
project is AUC.

**Report AUC on E2b**, with accuracy alongside for comparability with the ToM
paper's convention.

This requires a continuous score from a model trained to emit
`<answer>...</answer>`. Resolution in Adaptation D.

---

## 3. Adaptation C — CSS alone would rank the broken model first

CSS as drafted, $|p(Y|x) - p(Y|x')|$, is exactly this project's **CF-Sens**. We
have already established that it is not sufficient on its own: chain-of-thought
posts the best CSS in the entire study while being the worst model in it.

**Report the full triple for both arms**, using instruments already frozen:

| quantity | instrument | direction |
|---|---|---|
| CSS (= CF-Sens) | `cf_depression` vs `cf_neutral`, C byte-identical | ↓ |
| **construct sensitivity** | the frozen 200 E5 pairs — interviewer fixed, patient swapped | **↑ / flat** |
| **SER** = CSS / construct sensitivity | both of the above | ↓ |
| AUC on `cf_none` | IID utility guard | flat |

M1 may only be called a success if CSS falls **while construct sensitivity
holds**. Baselines for the untrained model: CSS 0.206, construct sensitivity
0.115, SER 1.78.

---

## 4. Adaptation D — the format reward collides with our scoring rule

Every prediction in this project is read as teacher-forced label
log-likelihood, never parsed from generation, because parse-failure rate
correlates with perturbation strength and would manufacture accuracy drops in
exactly the arms under test.

`R_format` requires the model to emit `<think>…</think><answer>…</answer>`, so
after training the natural evaluation is to parse. Doing so would break
comparability with every number in the paper.

**Resolution.** Train with `R_format` as drafted. Evaluate by **teacher-forcing
the label token inside the answer tag**: score
`…<answer>Yes</answer>` against `…<answer>No</answer>` and take the log-odds.
This yields a continuous, parse-free score on the same scale as the rest of the
project, and Adaptation B's AUC follows.

Report parse-failure rate per arm as a diagnostic. If it correlates with arm,
say so.

---

## 5. Adaptation E — n = 107, and what we do about it

DAIC's training split is **107 participants** (77 negative, 30 positive). At
rollout 8 × 2 epochs that is 1,712 rollouts per arm — very small for RL, and the
same constraint that made group DRO underpowered in I6 (smallest group: 7
examples).

**Mitigation, not a fix.** Each participant supplies four interviewer arms, so
pairing `x_i` with a *randomly drawn different arm* each epoch gives roughly
4× the effective distinct inputs without touching the participant count. The
label and symptom evidence never change, so this is legitimate augmentation
rather than more data.

**This is reported as an n-limit, not hidden.** If M0 and M1 differ by less than
the seed spread, that is the finding.

---

## 6. Compute

Full fine-tuning of a 7B model with AdamW needs ~98 GB (14 weights + 28 grads +
56 optimizer). The GPUs here are 48 GB, so **LoRA is required** — `peft 0.20.0`
is installed, `trl 1.12.0` provides `GRPOTrainer`, both verified working.

| item | setting |
|---|---|
| backbone | `MentaLLaMA-chat-7B` (the model the whole paper analyses) |
| adapter | LoRA r=16, α=32, attention + MLP projections |
| precision | bf16 |
| rollouts | 8 per prompt, paired sampling of `x_i` and `x_i'` in one batch |
| est. memory | ~24 GB |
| est. time | ~3–4 h per arm |

Note the backbone choice makes M0 informative in its own right: it tests
whether **vanilla RLVR on DAIC amplifies the shortcut we measured zero-shot**,
which is the RL analogue of our Form-3 finding that post-training introduced a
corpus-prior default.

---

## 7. What gets reported

Two arms, four numbers each, plus the monitor:

| | M0 (`R_f + R_a`) | M1 (`+ λR_inv`) | untrained baseline |
|---|---|---|---|
| AUC on E2b conflict set | | | 0.329 |
| CSS | | | 0.206 |
| construct sensitivity | | | 0.115 |
| SER | | | 1.78 |
| AUC on `cf_none` (utility guard) | | | 0.777 |
| **P(pred=Yes) at convergence** | | | — |

**Success for M1:** AUC on the conflict set up, CSS down, construct sensitivity
held, `P(pred=Yes)` not degenerate. Any three without the fourth is the
flattening result again.

**M0 is a result too**, not just a control: if vanilla RLVR raises conflict-set
performance while CSS also rises, that reproduces at the RL stage what we
measured across SFT → GRPO → HARPO in Form 3.

---

## 8. Honest boundaries

- One backbone, one corpus, one seed budget. The draft explicitly scopes out
  multi-backbone and SFT controls; we keep that scope.
- λ = 1 without a sweep, per the draft. If M1 fails we will not know whether a
  different λ would have worked, and the write-up must say so rather than
  implying λ = 1 is optimal.
- No out-of-distribution evaluation exists anywhere in this project
  (§Limitations); RL does not change that.
