# Shortcut usage by model — three datasets, eight models each

Generated 2026-09-04. Companion to the 8-model DAIC table; same shape, same
metric, so rows are directly comparable across all four datasets.

## What the columns mean

- **Shortcut rate** — how well the shortcut predicts the label with **no model
  involved** (a TF-IDF or single-feature probe). This is the shortcut's
  *availability*.
- **CSS** — counterfactual shortcut sensitivity: the paired change in the
  model's predicted probability when **only the shortcut-bearing content is
  swapped** and the item's own content is byte-identical across arms. This is
  the shortcut's *use*. Positive = the model moves in the direction the shortcut
  predicts.
- **flips** — fraction of items whose predicted label changes between the two
  swapped arms.
- **worst-arm acc** — accuracy in the arm where the counterfactual hurts most,
  against native accuracy. The gap is the damage.

**Reference point.** On **DAIC-WOZ** the same measurement gives CSS from 0.027
(Menta, saturated) to **0.505** (OmniSapiens2.0), with up to 56% of predictions
flipping. That is what "used" looks like.

**Verdict thresholds**: |CSS| < 0.02 → null · < 0.10 → weak · < 0.25 →
moderate · ≥ 0.25 → strong · negative → wrong sign.

---

## 1. `mmpsy_depression` — structural shortcut

HBA-test · Chinese · 190 items · 81.6% majority

The shortcut is **how many answer segments were recorded** — the depressed group has one more response than the control group (`no_depression` 74% exactly 7 segments, `depression` 63% exactly 8). The counterfactual splits one chunk at existing punctuation or joins two adjacent chunks: **every content character is preserved**, only the number of whitespace boundaries changes.

Shortcut rate (model-free): **segment count 0.772** (transcript content 0.507, chance 0.500)

| model | n | CSS | 95% CI | flips | native AUC | native acc | worst-arm acc | verdict |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 190 | **-0.0006** | [-0.0023, +0.0011] | 0.037 | 0.463 | 0.274 | 0.274 | **null** |
| Menta | 190 | **+0.0017** | [-0.0084, +0.0107] | 0.037 | 0.790 | 0.353 | 0.332 | **null** |
| qwen3_base | 190 | **-0.0000** | [-0.0000, +0.0000] | 0.000 | 0.642 | 0.816 | 0.816 | **null** |
| OSim-8B | 190 | **-0.0006** | [-0.0009, -0.0003] | 0.000 | 0.768 | 0.816 | 0.816 | **null** |
| OmniSapiens-SFT | 190 | **+0.0046** | [-0.0002, +0.0101] | 0.016 | 0.883 | 0.711 | 0.711 | **null** |
| OmniSapiens-RL | 190 | **+0.0041** | [+0.0015, +0.0066] | 0.011 | 0.871 | 0.847 | 0.821 | **null** |
| OmniSapiens2.0 | 190 | **+0.0011** | [-0.0036, +0.0056] | 0.011 | 0.885 | 0.863 | 0.853 | **null** |
| HumanOmniV2 | 190 | **+0.0028** | [+0.0007, +0.0048] | 0.005 | 0.783 | 0.821 | 0.816 | **null** |

**8/8 models.**

Every model returns a null — three architectures, four training families, no exception. The largest absolute value is 0.0046, about 1/45 of DAIC's. **A structural shortcut that a single-feature forest probe reads at 0.772 is invisible to every generative LLM tested.**

---

## 2. `ptsd_in_the_wild` — register shortcut, cross-source

HBA-test · English · 60 items · 30/30 balanced

The shortcut is **register / production source** — positives are veteran testimony and clinical scenes, negatives are TED-style talks. Construct and register are perfectly confounded in this corpus, so the counterfactual prepends a token-matched 120-word block of the other class's register while the item's own transcript stays byte-identical.

Shortcut rate (model-free): **text 0.948, stopwords only 0.880** (chance 0.500)

| model | n | CSS | 95% CI | flips | native AUC | native acc | worst-arm acc | verdict |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 60 | **-0.0021** | [-0.0051, +0.0007] | 0.000 | 0.473 | 0.500 | 0.500 | **null** |
| Menta | — | — | — | — | — | — | — | *not run* |
| qwen3_base | 60 | **+0.0197** | [+0.0023, +0.0466] | 0.017 | 0.993 | 0.967 | 0.967 | **null** |
| OSim-8B | 60 | **+0.0164** | [+0.0105, +0.0221] | 0.000 | 0.980 | 0.500 | 0.500 | **null** |
| OmniSapiens-SFT | 60 | **+0.1920** | [+0.1433, +0.2421] | 0.433 | 0.997 | 0.967 | 0.567 | moderate |
| OmniSapiens-RL | 60 | **+0.0960** | [+0.0778, +0.1146] | 0.167 | 1.000 | 0.700 | 0.500 | weak |
| OmniSapiens2.0 | 60 | **+0.2824** | [+0.2144, +0.3484] | 0.483 | 1.000 | 1.000 | 0.517 | **strong** |
| HumanOmniV2 | 60 | **+0.0227** | [+0.0160, +0.0299] | 0.000 | 0.973 | 0.500 | 0.500 | weak |

**7/8 models.**

**Only the OmniSapiens family is affected.** The other models sit at CSS ≤ 0.023 with ≤ 1.7% flips. HumanOmniV2 shares the Qwen2.5-Omni architecture with all three OmniSapiens checkpoints and shows +0.023 — so this is **not architectural, it is training**: HBA is OmniSapiens' own training and evaluation corpus.

Where it does bite, it is severe: OmniSapiens2.0 goes from **1.000 accuracy to 0.517** under a 120-word prefix that changes nothing about the item itself.

---

## 3. `Irf` — register shortcut, same-source

IMHI · English · 400 posts · 66.7% yes

The shortcut is **register** — deleting every content word costs only 0.021. Same prepend design, plus a **signal-free control arm** drawn from MultiWD (the one dataset whose text carries no label signal, 0.499 vs chance 0.500).

Shortcut rate (model-free): **text 0.643, stopwords only 0.622 — 97% of full-text signal** (chance 0.500)

| model | n | CSS | 95% CI | flips | native AUC | native acc | worst-arm acc | verdict |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 400 | **-0.0822** | [-0.0954, -0.0695] | 0.077 | 0.849 | 0.757 | 0.752 | **wrong sign** |
| Menta | 400 | **-0.0239** | [-0.0322, -0.0164] | 0.015 | 0.673 | 0.685 | 0.670 | **wrong sign** |
| qwen3_base | 400 | **-0.0073** | [-0.0209, +0.0048] | 0.022 | 0.716 | 0.713 | 0.675 | **null** |
| OSim-8B | 400 | **+0.0238** | [+0.0173, +0.0308] | 0.015 | 0.792 | 0.745 | 0.672 | weak |
| OmniSapiens-SFT | 400 | **-0.0010** | [-0.0014, -0.0006] | 0.000 | 0.761 | 0.682 | 0.667 | **null** |
| OmniSapiens-RL | 400 | **-0.0119** | [-0.0132, -0.0106] | 0.000 | 0.755 | 0.677 | 0.667 | **null** |
| OmniSapiens2.0 | 400 | **-0.0078** | [-0.0089, -0.0067] | 0.000 | 0.745 | 0.685 | 0.667 | **null** |
| HumanOmniV2 | 400 | **-0.0462** | [-0.0533, -0.0392] | 0.025 | 0.753 | 0.718 | 0.660 | **wrong sign** |

**8/8 models.**

**No model uses this shortcut correctly.** Seven of eight are negative or null; only OSim-8B is weakly positive (+0.024).

The three wrong-signed models reflect a **topic-presence effect**, not register use: both class prefixes raise p(yes) and the `no`-class prefix raises it more, while the signal-free MultiWD prefix does essentially nothing. Without that control arm the negative CSS would read as a mysterious inversion.

---

## Reading the three tables together

| shortcut type | example | availability | who uses it |
|---|---|---|---|
| **semantic / lexical** | DAIC — which questions the interviewer asked | 0.810 | **all 8 models** (CSS 0.027–0.505) |
| **coarse register** | `ptsd_in_the_wild` — testimony vs talk | 0.948 | **only models trained on that corpus** (OmniSapiens 0.096–0.282; others ≤ 0.023) |
| **fine register** | `Irf` — function-word distribution | 0.643 | **none** (7/8 null or wrong sign) |
| **structural** | `mmpsy` — number of answer segments | 0.772 | **none** (8/8 null) |

Two conditions have to hold before a shortcut is used, and availability is
neither of them:

1. **The signal must be coarse enough to cross the model's representational
   grain.** Segment counts and function-word distributions survive a
   discriminative probe and do not survive tokenisation into anything a language
   model represents.
2. **For register, the model must have seen that corpus.** The `ptsd` register
   gap is large enough in principle, but only the models trained on HBA exploit
   it. Semantic shortcuts need no such exposure, which is why DAIC affects
   everything.

**Consequence for auditing.** A model-free probe is an *upper bound on risk that
can overstate it badly*: `Irf` reads 0.622 to a probe and −0.082 to the models
the probe is meant to warn about. Dataset audits and model audits answer
different questions and neither substitutes for the other.

