# Shortcut mechanisms and their use by models

Three datasets, eight models each, plus what the shortcut in each one actually
is. Generated 2026-09-04 from result files on disk.

---

## 1. Summary — four mechanisms, four outcomes

| dataset | what correlates with the label | probe strength | used by LLMs? |
|---|---|---|---|
| **DAIC-WOZ** | **which questions the interviewer asked** — the wording is in the transcript | 0.810 AUC | **yes, all 8 models** (CSS 0.027–0.505) |
| **`ptsd_in_the_wild`** | **register / production source** | 0.948 (stopwords alone 0.880) | **only models trained on that corpus** |
| **`Irf`** | **function-word distribution** | 0.643 (stopwords alone 0.622) | **no** — 7/8 null or wrong-signed |
| **`mmpsy_depression`** | **how many answer segments were recorded** | 0.772 | **no** — 8/8 null |

Only the first is a *semantic* signal, and it is the only one every model uses.

### How to read the tables below

- **Shortcut rate** — how well the shortcut predicts the label with **no model
  involved**. This is *availability*.
- **CSS** (counterfactual shortcut sensitivity) — the paired change in the
  model's predicted probability when **only the shortcut-bearing content is
  swapped**, with the item's own content byte-identical across arms. This is
  *use*. Positive = the model moves the way the shortcut predicts.
- **flips** — fraction of items whose predicted label changes between the two
  swapped arms.
- **worst-arm acc** — accuracy in the arm where the counterfactual hurts most.
  The gap from native accuracy is the damage.

Thresholds: |CSS| < 0.02 → null · < 0.10 → weak · < 0.25 → moderate ·
≥ 0.25 → strong · negative → wrong sign.

---

## 2. What the shortcuts actually are

### Register — the "accent" of a text

Register is the style a text is written in: word choice, sentence shape, rhythm,
formality, person. In `ptsd_in_the_wild` the two classes were **collected from
entirely different places**, so their registers cannot be confused:

| | positive (PTSD) | negative (no PTSD) |
|---|---|---|
| source | news segments, veteran testimony | YouTube lifestyle videos |
| example | *"Leon 12 a Chandler mother of two is sharing her struggles with ptsd after witnessing the Boston Marathon bombing back in 2013..."* | *"What's up, GQ? This is Jonathan Majors, and these are my 10 essentials. First, we're gonna start with this guy. I know what you're thinking. It's just a beanie."* |
| register | third-person broadcast, formal, clinical vocabulary | second-person conversational, brisk, addressed to camera |

**A model does not need to decide whether this person has PTSD. It only needs to
decide whether this is a news report or a vlog.**

Decisive evidence: scrubbing *every* PTSD and military word **raises** accuracy
from 0.948 to 0.953. The difference was never in those words, which is why no
vocabulary blocklist can fix this dataset.

### Function-word distribution — statistics over the grammar

Function words carry grammar rather than content: `the`, `a`, `of`, `I`, `you`,
`and`, `but`, `is`, `to`. The stopword probe deletes **every content word**:

```
original       : I'm afraid the suicide scale is true and it gets worse
function words : I'm      the              is true and it gets
```

On `Irf` that still reaches **0.622** (chance 0.500, full text 0.643) —
**deleting every content word costs 0.021**, so 97% of the signal survives. Both
classes are Reddit mental-health posts from the same platform, so nothing
coarser separates them. The difference is in first-person frequency, negation
density, connective habits, sentence length.

### Why one is used and the other is not

| | `ptsd` (cross-source register) | `Irf` (function-word distribution) |
|---|---|---|
| size of the difference | enormous — topic, vocabulary and style differ at once | subtle — a frequency shift in grammatical words |
| visible to a human reader? | immediately | no; only measurable statistically |
| TF-IDF probe | 0.948 | 0.643 |
| **used by LLMs** | **partly** — the OmniSapiens family | **never** |

**A discriminative probe accumulates thousands of tiny frequency differences
into one separable score. A language model does not do that accumulation — it
reads meaning.** A shortcut living in function-word statistics is therefore
visible to the probe and invisible to the model.

The same explanation covers `mmpsy`, which shares nothing else with `Irf`.

### `mmpsy` — a shortcut that was demoted, not removed

`mmpsy_depression` **is** interview data, but the interviewer's turns were
stripped during data preparation. **0 of 191 transcripts contain any speaker
label.** A full transcript:

```
1. 进行放松，不必想太多。          "Relax, don't overthink."
2. 我身边没有处于这种状态的朋友。    "Nobody around me is in that state."
3. 开心。                        "Happy."
4. 跟同学交流。                   "Talking with classmates."
5. 最近学习还可以，心情不错。        "Studies are okay lately, I'm in a good mood."
```

These are plainly *answers* — "Happy." answers something. **The questions were
asked but not recorded.** The same adaptive-interview protocol that creates
DAIC's shortcut operates here, but its only trace is a number: how many answers
were recorded. 74% of controls have exactly 7 segments; 63% of depressed cases
have exactly 8, because the clinician asked one more follow-up.

| | DAIC-WOZ | `mmpsy` |
|---|---|---|
| interviewer turns | **in the transcript** | **stripped out** |
| what carries the label | the *wording* of the questions | a *count* of the questions |
| signal type | lexical / semantic | structural |
| probe reads it at | 0.810 | 0.772 |
| LLM reads it at | **0.206** | **0.004** |

`mmpsy`'s data preparation accidentally applied the standard mitigation —
removing interviewer turns is what the literature recommends, and is our own
best intervention (+0.126 AUC on DAIC). But **the shortcut was not removed, it
was demoted**: from a lexical form the model reads to a structural form only a
statistical probe reads. Risk for today's models is zero; the correlation is
still in the data, and any discriminative model trained on this corpus will
still learn it. This is a concrete **dormant shortcut**.

---

## 3. Per-dataset results, eight models each


### 3.1 `mmpsy_depression` — structural shortcut

HBA-test · Chinese · 190 items · 81.6% majority. The counterfactual splits one chunk at existing punctuation or joins two adjacent chunks: **every content character is preserved**, only the number of whitespace boundaries changes.

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

Every model returns a null — three architectures, four training families, no exception. The largest absolute value is 0.0046, about 1/45 of DAIC's. **A structural shortcut a single-feature forest probe reads at 0.772 is invisible to every generative LLM tested.**

### 3.2 `ptsd_in_the_wild` — register shortcut, cross-source

HBA-test · English · 60 items · 30/30 balanced. Construct and register are perfectly confounded in this corpus, so the counterfactual prepends a token-matched 120-word block of the other class's register while the item's own transcript stays byte-identical.

Shortcut rate (model-free): **text 0.948, stopwords only 0.880** (chance 0.500)

| model | n | CSS | 95% CI | flips | native AUC | native acc | worst-arm acc | verdict |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 60 | **-0.0021** | [-0.0051, +0.0007] | 0.000 | 0.473 | 0.500 | 0.500 | **null** |
| Menta | 60 | **+0.0754** | [+0.0476, +0.1080] | 0.067 | 0.943 | 0.650 | 0.517 | weak |
| qwen3_base | 60 | **+0.0197** | [+0.0022, +0.0465] | 0.017 | 0.993 | 0.967 | 0.967 | **null** |
| OSim-8B | 60 | **+0.0164** | [+0.0104, +0.0222] | 0.000 | 0.980 | 0.500 | 0.500 | **null** |
| OmniSapiens-SFT | 60 | **+0.1920** | [+0.1437, +0.2435] | 0.433 | 0.997 | 0.967 | 0.567 | moderate |
| OmniSapiens-RL | 60 | **+0.0960** | [+0.0780, +0.1148] | 0.167 | 1.000 | 0.700 | 0.500 | weak |
| OmniSapiens2.0 | 60 | **+0.2824** | [+0.2176, +0.3497] | 0.483 | 1.000 | 1.000 | 0.517 | **strong** |
| HumanOmniV2 | 60 | **+0.0227** | [+0.0155, +0.0292] | 0.000 | 0.973 | 0.500 | 0.500 | weak |

**8/8 models.**

**Only the OmniSapiens family is affected.** The other models sit at |CSS| ≤ 0.075. HumanOmniV2 shares the Qwen2.5-Omni architecture with all three OmniSapiens checkpoints and shows +0.023 — so this is **not architectural, it is training**: HBA is OmniSapiens' own training and evaluation corpus.

Where it does bite it is severe: OmniSapiens2.0 goes from **1.000 accuracy to 0.517** under a 120-word prefix that changes nothing about the item itself.

### 3.3 `Irf` — register shortcut, same-source

IMHI · English · 400 posts · 66.7% yes. Same prepend design, plus a **signal-free control arm** drawn from MultiWD — the one dataset whose text carries no label signal (0.499 vs chance 0.500).

Shortcut rate (model-free): **text 0.643, stopwords only 0.622** (chance 0.500)

| model | n | CSS | 95% CI | flips | native AUC | native acc | worst-arm acc | verdict |
|---|---|---|---|---|---|---|---|---|
| MentaLLaMA | 400 | **-0.0822** | [-0.0956, -0.0695] | 0.077 | 0.849 | 0.757 | 0.752 | **wrong sign** |
| Menta | 400 | **-0.0239** | [-0.0322, -0.0164] | 0.015 | 0.673 | 0.685 | 0.670 | **wrong sign** |
| qwen3_base | 400 | **-0.0073** | [-0.0208, +0.0047] | 0.022 | 0.716 | 0.713 | 0.675 | **null** |
| OSim-8B | 400 | **+0.0238** | [+0.0173, +0.0309] | 0.015 | 0.792 | 0.745 | 0.672 | weak |
| OmniSapiens-SFT | 400 | **-0.0010** | [-0.0014, -0.0006] | 0.000 | 0.761 | 0.682 | 0.667 | **null** |
| OmniSapiens-RL | 400 | **-0.0119** | [-0.0132, -0.0106] | 0.000 | 0.755 | 0.677 | 0.667 | **null** |
| OmniSapiens2.0 | 400 | **-0.0078** | [-0.0089, -0.0067] | 0.000 | 0.745 | 0.685 | 0.667 | **null** |
| HumanOmniV2 | 400 | **-0.0462** | [-0.0533, -0.0392] | 0.025 | 0.753 | 0.718 | 0.660 | **wrong sign** |

**8/8 models.**

**No model uses this shortcut correctly.** Seven of eight are negative or null; only OSim-8B is weakly positive (+0.024).

The wrong-signed models reflect a **topic-presence effect**: both class prefixes raise p(yes) and the `no`-class prefix raises it more, while the signal-free MultiWD prefix does essentially nothing. Without that control arm the negative CSS would read as a mysterious inversion.

---

## 4. Do the datasets agree about which models are susceptible?

| model | DAIC | ptsd | Irf | mmpsy |
|---|---|---|---|---|
| MentaLLaMA | +0.2050 | -0.0021 | -0.0822 | -0.0006 |
| Menta | +0.0267 | +0.0754 | -0.0239 | +0.0017 |
| qwen3_base | +0.2561 | +0.0197 | -0.0073 | -0.0000 |
| OSim-8B | +0.1960 | +0.0164 | +0.0238 | -0.0006 |
| OmniSapiens-SFT | +0.4573 | +0.1920 | -0.0010 | +0.0046 |
| OmniSapiens-RL | +0.4710 | +0.0960 | -0.0119 | +0.0041 |
| OmniSapiens2.0 | +0.5051 | +0.2824 | -0.0078 | +0.0011 |
| HumanOmniV2 | +0.3715 | +0.0227 | -0.0462 | +0.0028 |

Pearson r across the 8 models:

| | DAIC | ptsd | irf | mmpsy |
|---|---|---|---|---|
| **DAIC** | 1.00 | 0.64 | 0.18 | 0.56 |
| **ptsd** | 0.64 | 1.00 | 0.34 | 0.43 |
| **irf** | 0.18 | 0.34 | 1.00 | 0.13 |
| **mmpsy** | 0.56 | 0.43 | 0.13 | 1.00 |

**DAIC ↔ ptsd = 0.64.** The models most moved by the shortcut on one are most
moved on the other — these are the only two datasets with an LLM-readable
shortcut, so this reflects a genuine model-level susceptibility. OmniSapiens
ranks highest on both; Menta lowest.

**`Irf` correlates with nothing (0.13–0.34)** because its column is almost all
noise: |CSS| ≤ 0.08, mostly around 0.01, so there is no real variance to
correlate.

**`mmpsy` ↔ DAIC = 0.56 is spurious and should not be used.** That entire column
ranges from −0.0006 to +0.0046 — it is ordering measurement noise. With n = 8
the coefficient means nothing.

A caveat on the 0.64: it is driven mainly by the three OmniSapiens checkpoints
ranking highest on both, and those are exactly the models trained on HBA, of
which `ptsd` is a subset. How much of the correlation is general susceptibility
versus corpus exposure **cannot be separated with the current data** — it would
need a dataset with a coarse shortcut that is *not* in HBA.

---

## 5. What this means for auditing

**A model-free probe is an upper bound on risk, and it can overstate that risk
badly.** On `Irf` the probe reads 0.622 while the models it is meant to warn
about read −0.082 — the wrong sign entirely.

Two conditions must hold before a shortcut is used, and availability is neither:

1. **The signal must be coarse enough to cross the model's representational
   grain.** Segment counts and function-word distributions survive a
   discriminative probe and do not survive tokenisation into anything a language
   model represents.
2. **For register shortcuts, the model must have seen that corpus.** The `ptsd`
   register gap is large enough in principle, but only the models trained on HBA
   exploit it.

Semantic shortcuts require neither condition, which is why DAIC's interviewer
wording moves every model tested.

**Dataset audits and model audits answer different questions, and neither
substitutes for the other.**

