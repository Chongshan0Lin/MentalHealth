# What the shortcuts actually are

A plain-language explanation of the three shortcut mechanisms found in the
non-DAIC datasets, why two of them are invisible to language models, and what
that implies for auditing.

---

## The four mechanisms, side by side

| dataset | what correlates with the label | probe strength | used by LLMs? |
|---|---|---|---|
| **DAIC-WOZ** | **which questions the interviewer asked** — the wording is in the transcript | 0.810 AUC | **yes, all 8 models** (CSS 0.027–0.505) |
| **`ptsd_in_the_wild`** | **register / production source** | 0.948 (stopwords alone 0.880) | **only models trained on that corpus** (OmniSapiens 0.096–0.282; others ≤ 0.075) |
| **`Irf`** | **function-word distribution** | 0.643 (stopwords alone 0.622) | **no** (7/8 null or wrong-signed) |
| **`mmpsy_depression`** | **how many answer segments were recorded** | 0.772 | **no** (8/8 null) |

Only the first is a *semantic* signal. That is the one every model uses.

---

## Register — the "accent" of a text

Register is the style a text is written in: word choice, sentence shape, rhythm,
formality, person. Same content, different register, depending on where and why
it was produced.

In `ptsd_in_the_wild` the two classes were **collected from entirely different
places**, so their registers cannot be confused:

| | positive (PTSD) | negative (no PTSD) |
|---|---|---|
| source | news segments, veteran testimony | YouTube lifestyle videos |
| example | *"Leon 12 a Chandler mother of two is sharing her struggles with ptsd after witnessing the Boston Marathon bombing back in 2013..."* | *"What's up, GQ? This is Jonathan Majors, and these are my 10 essentials. First, we're gonna start with this guy. I know what you're thinking. It's just a beanie."* |
| register | third-person broadcast, formal, clinical vocabulary | second-person conversational, brisk, addressed to camera |

**A model does not need to decide whether this person has PTSD. It only needs to
decide whether this is a news report or a vlog.**

The decisive evidence: scrubbing *every* PTSD and military word from the
transcripts **raises** accuracy from 0.948 to 0.953. The difference was never in
those words — it is in the register as a whole, which is why no vocabulary
blocklist can fix this dataset.

---

## Function-word distribution — statistics over the grammar

Function words (stopwords) are the words carrying grammar rather than content:
`the`, `a`, `of`, `I`, `you`, `and`, `but`, `is`, `to`.

The stopword probe deletes **every content word** and asks whether the label is
still recoverable:

```
original       : I'm afraid the suicide scale is true and it gets worse
function words : I'm      the              is true and it gets
```

On `Irf` that still reaches **0.622** (chance 0.500, full text 0.643). **Deleting
every content word costs 0.021** — 97% of the signal survives.

So the difference is not in *what* was said but in *how* it was said, at the
level of grammatical statistics: first-person frequency, negation density,
connective habits, sentence length. Both classes are Reddit mental-health posts
from the same platform, so nothing coarser separates them.

---

## Why one is used and the other is not

| | `ptsd` (cross-source register) | `Irf` (function-word distribution) |
|---|---|---|
| size of the difference | enormous — topic, vocabulary and style all differ at once | subtle — a frequency shift in grammatical words |
| visible to a human reader? | immediately | no; only measurable statistically |
| TF-IDF probe | 0.948 | 0.643 |
| **used by LLMs** | **partly** — the OmniSapiens family | **never** — 8/8 null or wrong sign |

**A discriminative probe accumulates thousands of tiny frequency differences
into one separable score. A language model does not do that accumulation — it
reads meaning.** So a shortcut that lives in function-word statistics is visible
to the probe and invisible to the model.

The same explanation covers `mmpsy`, which shares nothing else with `Irf`: a
count of whitespace-separated answer segments is a single number a random forest
reads at 0.772, and is not a quantity a language model has any reason to
represent. Both fail for the same underlying reason.

---

## The special case of `mmpsy`: a shortcut that was demoted

`mmpsy_depression` **is** interview data — but the interviewer's turns were
stripped during data preparation. A full transcript looks like this (0 of 191
transcripts contain any speaker label):

```
1. 进行放松，不必想太多。          "Relax, don't overthink."
2. 我身边没有处于这种状态的朋友。    "Nobody around me is in that state."
3. 开心。                        "Happy."
4. 跟同学交流。                   "Talking with classmates."
5. 最近学习还可以，心情不错。        "Studies are okay lately, I'm in a good mood."
...
```

These are plainly *answers* — "Happy." answers something, "Nobody around me is
in that state." answers something else. **The questions were asked but not
recorded.**

So the same adaptive-interview protocol that creates DAIC's shortcut operates
here too, but the only trace it leaves is a number: **how many answers were
recorded.** The depressed group has one more than the control group (74% of
controls have exactly 7 segments; 63% of depressed have exactly 8), because the
clinician asked one more follow-up.

| | DAIC-WOZ | `mmpsy` |
|---|---|---|
| interviewer turns | **in the transcript** | **stripped out** |
| what carries the label | the *wording* of the questions asked | a *count* of the questions asked |
| signal type | lexical / semantic | structural |
| probe reads it at | 0.810 | 0.772 |
| LLM reads it at | **0.206** | **0.004** |

**`mmpsy`'s data preparation accidentally applied the standard mitigation.**
Removing interviewer turns is exactly what the literature recommends and what
our own best intervention does (deleting the interviewer's turns is worth
+0.126 AUC on DAIC).

But note what actually happened: **the shortcut was not removed, it was
demoted** — from a lexical form the model can read to a structural form only a
statistical probe can read. Risk for today's models is zero; the correlation is
still in the data, and any discriminative model trained on this corpus will
still learn it.

That is a concrete instance of a **dormant shortcut**: the signal is present,
the current architecture simply cannot see it.

---

## What this means for auditing

**A model-free probe is an upper bound on risk, and it can overstate that risk
by a wide margin.** On `Irf` the probe reads 0.622 while the models it is meant
to warn about read −0.082 — the wrong sign entirely.

Two conditions must hold before a shortcut is actually used, and availability is
neither of them:

1. **The signal must be coarse enough to cross the model's representational
   grain.** Segment counts and function-word distributions survive a
   discriminative probe and do not survive tokenisation into anything a language
   model represents.
2. **For register shortcuts, the model must have seen that corpus.** The `ptsd`
   register gap is large enough in principle, but only the OmniSapiens family —
   whose training and evaluation corpus is HBA — exploits it. HumanOmniV2 shares
   the same Qwen2.5-Omni architecture and shows only +0.023, so this is a
   training effect, not an architectural one.

Semantic shortcuts require neither condition, which is why DAIC's interviewer
wording moves every model tested.

**Dataset audits and model audits answer different questions, and neither
substitutes for the other.**
