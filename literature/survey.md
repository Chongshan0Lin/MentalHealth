# Literature survey — shortcut diagnosis → mitigation

Searched 2026-08-21 (Semantic Scholar + arXiv, 169 unique papers retrieved to
`literature/raw*.jsonl`). Everything below carries metadata from the search;
nothing here is recalled from memory.

---

## 1. Our A1 finding is already published

**Burdisso, Reyes-Ramírez, Villatoro-Tello, Sánchez-Vega et al. (2024).**
*DAIC-WOZ: On the Validity of Using the Therapist's Prompts in Automatic
Depression Detection from Clinical Interviews.* Clinical NLP Workshop.
arXiv:2404.14463 · **37 citations**

> "…models using interviewer's prompts learn to focus on a specific region of
> the interviews, **where questions about past experiences with mental health
> issues are asked**, and use them as discriminative shortcuts… we achieve a
> **0.90 F1 score by intentionally exploiting it**, the highest result reported
> to date on this dataset using only textual information."

This is A1 and A2a. Their "specific region where questions about past mental
health experiences are asked" is exactly our `why_seek_help` (OR 11.26),
`therapy_going` (10.42), `when_diagnosed` (8.24). **Treat A1/A2a as
replication, not discovery** — and cite them as the motivating prior work.

What remains ours: they study models *trained on DAIC* and detect the shortcut
by **ablation**. We study *zero-shot foundation models*, where ablation returns
a null (AUC 0.546, at floor) and only the counterfactual reveals the effect
(Δp = +0.206). The methodological claim — ablation is not sufficient — is not
in their paper.

---

## 2. Mitigation on this exact problem

**Zhang & Poellabauer (2025).** *Mitigating Interviewer Bias in Multimodal
Depression Detection: An Approach with Adversarial Learning and Contextual
Positional Encoding.* EMNLP Findings. doi:10.18653/v1/2025.findings-emnlp.650

Adversarial learning to strip interviewer information from the representation,
plus positional encoding that makes the model's use of interview position
explicit rather than latent. Directly downstream of Burdisso. **Read this
first** — it is the closest existing answer to "what do we do about it".

**Li, Zhang, Li, Guo et al. (2026).** *DepFlow: Disentangled Speech Generation
to Mitigate Semantic Bias in Depression Detection.* arXiv:2601.00303

Same disease, different organ: DAIC couples linguistic sentiment to the label,
so models learn semantic shortcuts and fail on "Camouflaged Depression"
(socially positive language, depressed state). Fix is a depression-conditioned
TTS pipeline with adversarially disentangled, speaker- and content-invariant
embeddings (ROC-AUC 0.693), used to *generate* counterfactual training audio.

Note the shared move: **both mitigate by making the spurious factor explicit
and then removing or controlling it**, rather than by deleting the data.

---

## 3. The canonical debiasing families (train-time)

**Clark, Yatskar & Zettlemoyer (2019).** *Don't Take the Easy Way Out:
Ensemble Based Methods for Avoiding Known Dataset Biases.* arXiv:1909.03683
— Train a naive model that predicts **only** from the known bias, then train
the real model in a product-of-experts ensemble with it, so gradients flow only
where the bias model is wrong. Requires you to name the bias in advance.

**Sanh, Wolf, Belinkov & Rush (2020).** *Learning from Others' Mistakes:
Avoiding Dataset Biases Without Modeling Them.* arXiv:2012.01300
— Removes that requirement. A **limited-capacity** model preferentially learns
the shortcuts, so its errors identify them without anyone specifying what they
are. Same PoE machinery.

**Sagawa, Koh, Hashimoto & Liang (2019).** *Distributionally Robust Neural
Networks for Group Shifts.* arXiv:1911.08731 · **1749 citations**
— Optimise worst-group rather than average loss. Key finding: naive group DRO
fails on overparameterised nets; it needs **strong regularisation** (heavy L2
or early stopping) to give 10–40 point worst-group gains.

---

## 4. Inference-time correction — the family that fits our constraints

**Wu, Huang & Wu (2025).** *Beyond Spurious Signals: Debiasing Multimodal Large
Language Models via Counterfactual Inference and Adaptive Expert Routing.*
arXiv:2509.15361
— Causal-mediation framing: separate core semantics from spurious textual and
visual context using counterfactual examples, then route through
modality-specific debiasing experts (MoE). Evaluated on multimodal sarcasm and
sentiment.

This is the only retrieved family that does not require retraining the base
model, which matters because our project scope is explicitly inference-only on
released checkpoints.

---

## 5. Post-training and spuriousness — bears on our D

**Shuieh, Singhal, Shanker, Heyer et al. (2025).** *Assessing Robustness to
Spurious Correlations in Post-Training Language Models.* arXiv:2505.05704
— Compares SFT / DPO / KTO across synthetic tasks at 10 % vs 90 % spuriousness,
with two artifact types (Feature Ambiguity, Distributional Narrowness). Finds
models "often but not always" degrade under higher spuriousness, and that
preference-based methods can be *relatively* more robust than SFT on maths
reasoning.

Our D is the complementary real-data case, and points the other way on this
benchmark: raw accuracy 0.260 → 0.687 across SFT → GRPO → HARPO while balanced
accuracy is flat, with the HARPO stage the only one defaulting to the corpus
majority under a content-free prompt.

---

## How these papers get from diagnosis to prescription

Five recurring moves, in rough order of cost:

| family | mechanism | needs retraining? | needs the bias named? |
|---|---|---|---|
| Data surgery | drop or mask the shortcut-bearing region | no | yes |
| PoE / ensemble | fit a bias-only model, subtract it | **yes** | Clark: yes · Sanh: no |
| Worst-group objective | group DRO + strong regularisation | **yes** | yes (group labels) |
| Adversarial disentanglement | remove the factor from the representation | **yes** | yes |
| Counterfactual inference | subtract the shortcut's direct effect at inference | **no** | yes |

**Almost every method assumes retraining.** Under our no-retraining scope only
the first and last are available.

---

## The gap our result opens

None of the retrieved mitigations exploit the specific structure we measured:
interviewer context moves the model's **operating point**, not its **ranking**
(AUC 0.764 → 0.734, but P(pred = Yes) 0.160 → 0.745).

If the damage is a threshold shift, the correction is a **context-conditional
recalibration** — estimate the shift induced by the interviewer context and
subtract it from the decision threshold, leaving weights untouched. That is
strictly cheaper than every method in the table above, and it is directly
testable from the A2b log-probabilities we already have on disk.

Whether it works is an empirical question we can answer today; that it is the
*right shape* of fix follows from the bias-shift finding, which is ours.
