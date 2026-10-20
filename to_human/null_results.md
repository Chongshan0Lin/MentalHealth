# Null results — shortcuts that are present in the data but not used by the model

Generated 2026-09-04 from the result files on disk. Every row pairs the
**shortcut rate** (how well the shortcut predicts the label with no model
involved) against the **model's use of it** (how far the prediction moves when
that shortcut is manipulated and everything else is held fixed).

A large shortcut rate with a near-zero or wrong-signed model effect is the
pattern this document collects.

**Metric.** CSS (counterfactual shortcut sensitivity) is the paired change in
the model's predicted probability when only the shortcut-bearing content is
swapped, with the item's own content byte-identical across arms. Positive means
the model moves in the direction the shortcut predicts.

**Reference point.** On DAIC-WOZ the same measurement gives **CSS = +0.206**
[+0.195, +0.217] with 58.5% of predictions flipping, and on
`ptsd_in_the_wild` **+0.282** [+0.217, +0.349]. Those are what "used" looks
like. Everything below is what "not used" looks like.

---

## 1. `mmpsy_depression` — structural shortcut, not used

HBA-test, Chinese, 190 items (35 positive, 81.6% majority). The shortcut is the
**number of recorded answer segments**: the depressed group has one more
response than the control group (`no_depression` 74% exactly 7 segments,
`depression` 63% exactly 8).

Intervention touches **whitespace only** — a chunk split at existing punctuation
to add a segment, or two joined to remove one. All 97 content characters
identical across arms.

| shortcut rate (model-free) | balanced acc |
|---|---|
| transcript content | 0.507 (chance 0.500) |
| total characters | 0.509 |
| **answer-segment count** | **0.772** |

| model | CSS | 95% CI | flips | acc: −1 seg → native → +1 seg | AUC range |
|---|---|---|---|---|---|
| OmniSapiens-SFT | **+0.0046** | [-0.0003, +0.0102] | 0.016 | 0.721 → 0.711 → 0.716 | 0.878–0.883 |
| OmniSapiens-RL (GRPO) | **+0.0041** | [+0.0016, +0.0065] | 0.011 | 0.832 → 0.847 → 0.821 | 0.867–0.871 |
| OmniSapiens2.0 (HARPO) | **+0.0011** | [-0.0037, +0.0054] | 0.011 | 0.853 → 0.863 → 0.863 | 0.881–0.886 |

**Fifty times smaller than DAIC, with 1% of predictions flipping against 58.5%.**
Accuracy and AUC are flat across arms at every stage.

Two of these CIs exclude zero. They should still not be read as use — the
direction fails by subgroup:

| model | items starting at 7 seg (→ 8 = depressed mode) | items starting at 8 seg (→ 7 = control mode) |
|---|---|---|
| OmniSapiens-SFT | +0.0066 [−0.0001, +0.0142] | +0.0025 [−0.0052, +0.0095] |
| OmniSapiens-RL | +0.0039 [+0.0008, +0.0070] | +0.0044 [−0.0009, +0.0097] |
| OmniSapiens2.0 | **−0.0022** [−0.0088, +0.0037] | +0.0072 [+0.0003, +0.0147] |

If the model read segment count, adding a segment to a 7-segment item should
raise p(depression) and removing one from an 8-segment item should lower it.
HARPO moves the wrong way on both.

**Saturation does not explain it.** HARPO's AUC is 0.881–0.886 — the ranking is
informative; only the shortcut manipulation fails to move it.

---

## 2. `Irf` — register shortcut, wrong sign

IMHI, 400 posts sampled (66.7% yes). **The models were instruction-tuned on this
corpus.** The shortcut is register: deleting every content word costs 0.021.

| shortcut rate (model-free) | balanced acc |
|---|---|
| text (TF-IDF) | 0.643 (chance 0.500) |
| **stopwords only** | **0.622 — 97% of the full-text signal** |

CSS = p(yes | yes-register prefix) − p(yes | no-register prefix). Each post's own
text byte-identical across arms; the three prefix pools token-matched at 150
words.

| model | CSS | 95% CI | flips | native AUC | AUC under prefixes |
|---|---|---|---|---|---|
| MentaLLaMA-chat-7B | **-0.0822** | [-0.0952, -0.0693] | 0.077 | 0.849 | 0.714–0.843 |
| Menta | **-0.0239** | [-0.0327, -0.0165] | 0.015 | 0.673 | 0.573–0.712 |

**Both signs are negative**: prepending the `no`-class register raises p(yes)
*more* than prepending the `yes`-class register. If the model read register as
evidence, the sign would be positive.

### What the signal-free control arm shows

`ctx_neutral` is a prefix drawn from **MultiWD**, the one dataset in either audit
whose text carries no label signal (0.499 against chance 0.500).

| model | `ctx_no` − signal-free | `ctx_yes` − signal-free | signal-free − native |
|---|---|---|---|
| MentaLLaMA-chat-7B | +0.0981 [+0.0843, +0.1118] | +0.0159 [+0.0109, +0.0209] | -0.0081 [-0.0176, +0.0013] n.s. |
| Menta | +0.1182 [+0.0947, +0.1422] | +0.0943 [+0.0746, +0.1152] | -0.0680 [-0.0859, -0.0519]  |

**Both class prefixes push p(yes) up; the signal-free prefix does nothing.** The
model is reacting to the presence of clinical-sounding first-person text, not to
which class the prefix belongs to. That is a topic-presence effect, not use of
the register shortcut — and without this control arm the negative CSS would have
looked like an inversion rather than a misattribution.

### The two questions in `Irf` behave oppositely

| model | thwarted belongingness | perceived burdensomeness |
|---|---|---|
| MentaLLaMA-chat-7B | -0.1801 [-0.1967, -0.1646] (n=202) | +0.0177 [+0.0127, +0.0231] (n=198) |
| Menta | -0.0015 [-0.0023, -0.0008] (n=202) | -0.0468 [-0.0629, -0.0318] (n=198) |

Same model, same prefixes, same posts — **opposite signs, 0.20 apart on
MentaLLaMA**. `Irf` bundles two constructs under one dataset name and should not
be reported as a single task.

---

## 3. IMHI template/prior shortcut — no effect, all 10 sources, both models

C1 tested whether rewording the source's own instruction template changes the
label prior the model falls back on. `Δ_prior` = accuracy(native template, no
post) − accuracy(canonical template, no post).

| model | sources tested | max \|Δ_prior\| | sources with Δ_prior ≠ 0 |
|---|---|---|---|
| MentaLLaMA-chat-7B | 10 | 0.030 (MultiWD only) | 1 / 10 |
| Menta | 10 | **0.000** | 0 / 10 |

Per-source Δ_prior is exactly 0.000 for CAMS, DR, Irf, SAD, SDCNL, dreaddit,
loneliness, swmh and t-sid on both models.

**Caveat — this is close to an identity, not a measurement.** Both templates
select the same argmax, and a constant prediction scores the same under either
wording, so Δ_prior ≈ 0 follows almost by construction. It is reported here for
completeness, not as independent evidence.

The informative version of the same test is the **content-free accuracy** itself,
scored against the natural label distribution: MentaLLaMA's no-post prediction
lands on the exact corpus majority class in **4 of 10 sources** (DR 0.758,
loneliness 0.782, swmh 0.347, t-sid 0.522), mean 0.475 against chance 0.373 —
that part is *not* null.

---

## 4. Per-model nulls on DAIC itself

From the 8-model sweep (`experiments/i10-model-sweep/results/suite_*.json`), on
the dataset where the effect is otherwise strongest:

| model | shortcut rate (removal AUC) | CSS | flips | construct sens | verdict |
|---|---|---|---|---|---|
| Menta | 0.576 | 0.027 | 0.000 | 0.014 | **null — saturated** |
| OSim-8B | 0.722 | 0.196 | 0.298 | 0.279 | moderate |
| qwen3_base | 0.710 | 0.256 | 0.261 | 0.410 | strong |
| MentaLLaMA | 0.546 | 0.205 | 0.559 | 0.113 | moderate |
| HumanOmniV2 | 0.625 | 0.372 | 0.511 | 0.356 | strong |
| OmniSapiens-SFT | 0.686 | 0.457 | 0.479 | 0.278 | strong |
| OmniSapiens-RL | 0.661 | 0.471 | 0.516 | 0.278 | strong |
| OmniSapiens2.0 | 0.699 | 0.505 | 0.511 | 0.365 | strong |

**Menta is the only model that returns a null on DAIC** (CSS 0.027, **zero**
prediction flips). It is saturated: its construct sensitivity is 0.014, i.e. it
barely responds to *any* input change, so this is an uninformative null rather
than evidence against the effect. It behaves the same way in `Irf` above
(p(yes) 0.88–0.99 across all arms).

---

## 5. The encoder arm — encoded but not used

MentalBERT fine-tuned on DAIC, 3 seeds
(`experiments/i6-encoder-interventions/results/encoder_interventions.json`).

| method | AUC (mean ± sd) | CSS (mean) | construct sens | SER |
|---|---|---|---|---|
| `B0` | 0.6457 ± 0.0719 | 0.0096 | 0.1343 | 0.0717 |
| `D3` | 0.7626 ± 0.0270 | 0.0213 | 0.4171 | 0.0511 |
| `M1` | 0.5779 ± 0.0108 | 0.0013 | 0.0193 | 0.0658 |
| `M2` | 0.6407 ± 0.0542 | 0.0106 | 0.1197 | 0.0884 |
| `M2reg` | 0.6299 ± 0.0752 | 0.0115 | 0.1222 | 0.0940 |

The encoder's CSS is **an order of magnitude below the zero-shot LLM's**
(≈0.01–0.03 against 0.206) even though its ability to predict from
interviewer-only input is high. It has *encoded* the shortcut without *relying*
on it the way the LLM does — which is why the train-time mitigation arm returned
nothing, and why a mitigation validated on one architecture need not transfer to
the other.

---

## 6. No discriminative leakage from the question alone

`question_only` withholds video, audio **and** transcript, leaving only the
question and the option list (`meld_emotion`, 300 items, 7 classes).

| model | balanced acc | chance | reading |
|---|---|---|---|
| OmniSapiens-SFT | 0.143 | 0.143 | at chance — constant prediction |
| OmniSapiens-RL | 0.143 | 0.143 | at chance — constant prediction |
| OmniSapiens2.0 | 0.143 | 0.143 | at chance — constant prediction |

Balanced accuracy is exactly 1/7 for all three stages: the question does not let
any model tell items apart. **This null is real but partial** — raw accuracy on
the same arm is 0.027 / 0.027 / **0.483**, and 0.483 is the corpus majority rate,
because OmniSapiens2.0 answers `neutral` to every item. There is no
*discriminative* leakage; there is *prior* leakage, and only the HARPO-stage
checkpoint exploits it.

---

## What the nulls establish

1. **CSS measures a specific shortcut being used, not sensitivity to
   perturbation.** `mmpsy` applies a stricter intervention than DAIC's — not one
   content character changes — on a shortcut worth 0.772 to a model-free probe,
   and returns 0.004. Without this, a large CSS elsewhere could be dismissed as
   generic input sensitivity.

2. **A discriminative probe is an upper bound on model risk and can overstate it
   badly.** `Irf`: 0.622 for the probe, wrong-signed −0.082 for the model the
   probe is warning about. Dataset audits and model audits answer different
   questions.

3. **The counterfactual needs a signal-free control arm.** Without
   `ctx_neutral`, `Irf`'s negative CSS reads as an inversion rather than a
   misattributed topic-presence effect.

4. **What separates the nulls from the positives is granularity, not mechanism.**
   Both register shortcuts were tested; `ptsd_in_the_wild` (classes from
   different production sources — veteran testimony vs TED talks) is used at
   +0.282, `Irf` (same platform, same genre, difference only in function-word
   distribution) is not. `mmpsy`'s segment count fails for the same reason while
   sharing nothing else with `Irf`. **A shortcut is exploitable to the extent
   that its discriminative signal is coarse enough to cross the model's
   representational grain.**

5. **A testable prediction follows.** `mosei_emotion` has a stopword/text ratio
   of 92%, close to `Irf`'s 97%, so this account predicts a null there. It has
   not been run.

