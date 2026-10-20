# Dataset shortcut audit — which benchmarks leak, and which don't

Sources: `experiments/h1-hba-shortcut/results/audit.json` (HBA-test, 16 subsets)
and `experiments/h2-imhi-shortcut/results/audit.json` (IMHI, 9 subsets).

**Every probe below is model-free.** No model is loaded anywhere in these
audits. They establish that a shortcut *exists and is usable*; they say nothing
about whether any model *uses* it. That second question is open for every
dataset here except DAIC — see §5.

**How to read the columns.** `text` is balanced accuracy from the transcript
alone. `stopwords` is the same probe with every content word deleted — if it
stays high, the classes differ in *register* (how people talk) rather than in
the construct, which is the signature of two production sources rather than a
removable word list. `lift` is text − chance.

---

## Table 1 — Datasets with a shortcut: severe

The label is recoverable from something that is not the construct, at a level
that makes the reported task uninterpretable.

| dataset | n | chance | majority | text | stopwords | verdict |
|---|---|---|---|---|---|---|
| `ptsd_in_the_wild` | 64 | 0.500 | 0.500 | **0.948** | 0.880 | **not a clinical task** — two different corpora |
| `mmpsy_depression` | 191 | 0.500 | 0.817 | **0.507** | — | protocol artifact (see Table 2) |
| `mmpsy_anxiety` | 191 | 0.500 | 0.937 | **0.500** | — | protocol artifact + 93.7% skew |

### `ptsd_in_the_wild` — the label is the corpus

| probe | balanced acc |
|---|---|
| full transcript, TF-IDF | **0.948** |
| after scrubbing *all* PTSD/military vocabulary | **0.953** (rises) |
| **stopwords only — every content word deleted** | **0.880** |
| first 100 words only | 0.844 |
| first 25 words only | 0.719 |
| transcript length alone | AUC 0.770 |

Positive class is veteran testimony; negative class is lifestyle vlog content
("what's in my bag", "my essentials"). 19/32 positives contain the literal
string `ptsd`; 0/32 negatives do.

Three properties make this unfixable rather than merely leaky: the label word is
in the input, scrubbing the vocabulary *raises* accuracy, and function words
alone reach 0.880. The prompt also asserts a premise false for half the data —
"a patient undergoing clinical evaluation" — when the negatives are not patients
and no evaluation took place.

### `mmpsy_*` — DAIC's shortcut in another language

Content carries nothing; the *number of recorded answers* carries a great deal.

| subset | chance | text | total characters | **answer segments** |
|---|---|---|---|---|
| `mmpsy_depression` | 0.500 | 0.507 | 0.509 | **0.772** |
| `mmpsy_anxiety` | 0.500 | 0.500 | 0.429 | **0.771** |

```
depression      segments: {7: 7,   8: 22, 9: 5, 14: 1}   63% have exactly 8
no_depression   segments: {7: 115, 8: 33, 6: 4, 10: 2}   74% have exactly 7
```

The depressed group has one more recorded response than the control group. The
data-collection protocol differs by class, so a structural property of the
record predicts the label with no behavioural content read. 0.772 comes from a
forest on that single feature — the pattern is an interval ("exactly 8"), not a
cut point, so the best single threshold reaches only 0.691.

Both subsets are the **same 191 participants** with two label sets, so they are
not independent evidence. Both prompts are malformed (two instructions
concatenated, the first truncated mid-sentence) and misdescribe their own label
space (PHQ-9/GAD-7 are five-level scales; the options offered are binary).

---

## Table 2 — Datasets with a shortcut: register leakage (milder)

Function words alone recover most of what the full text recovers. The construct
contributes little beyond how the speakers talk.

| dataset | chance | text | stopwords | stopwords / text | source |
|---|---|---|---|---|---|
| `mosei_emotion` | 0.167 | 0.281 | **0.258** | 92% | HBA |
| `mosei_senti` | 0.143 | 0.260 | **0.203** | 78% | HBA |
| `meld_senti` | 0.333 | 0.505 | **0.391** | 77% | HBA |
| `Irf` | 0.500 | 0.643 | **0.622** | 97% | IMHI |
| `swmh` | 0.200 | 0.684 | **0.374** | 55% | IMHI |
| `t-sid` | 0.250 | 0.618 | **0.375** | 61% | IMHI |

`mosei_emotion` (92%) and `Irf` (97%) are the worst: deleting every content word
costs almost nothing. Report these with the stopword control alongside, or not
at all.

**IMHI corpus-level leakage.** Source identity is recoverable from post text at
**0.522** (chance 0.111); stopwords alone give **0.372**. For `swmh`,
scrubbing the label-indicative vocabulary still leaves **0.594** (chance 0.200).
The corpus also contains **288 conflicting duplicates** (identical text, different
labels) and 372 cross-source duplicates out of 19,034 rows.

---

## Table 3 — Datasets with no detectable shortcut

| dataset | n | chance | text | lift over chance | why |
|---|---|---|---|---|---|
| `cremad` | 1117 | 0.167 | 0.170 | 0.003 | **calibration control** — fixed neutral carrier sentence |
| `tess` | 420 | 0.143 | 0.102 | -0.041 | **calibration control** — fixed neutral carrier sentence |
| `MultiWD` | 2441 | 0.500 | 0.499 | -0.001 | genuinely audited, nothing found |

`cremad` and `tess` are the **calibration pair**: every speaker reads the same
emotionally neutral carrier sentence, so text-only accuracy *must* sit at chance
by construction. It does (0.170 vs 0.167; 0.102 vs 0.143). Nothing elsewhere in
this audit is read as a shortcut unless it clears this floor — these two are what
make the rest of the audit trustworthy.

`MultiWD` is the only dataset that was audited without a prior expectation and
came back clean: text 0.499 against chance 0.500, stopwords 0.500. The transcript
carries no label information at all.

---

## Table 4 — Usable with caveats

Modest text signal, no register leak. Fine to report once the label-space
defects are fixed.

| dataset | n | chance | majority | text | note |
|---|---|---|---|---|---|
| `meld_emotion` | 2610 | 0.143 | 0.481 | 0.288 | **610/2610 (23.4%) have an unselectable gold answer** — `joy`/`sadness` against `happy`/`sad` options |
| `chsimsv2` | 1034 | 0.200 | 0.273 | 0.324 | — |
| `urfunny` | 224 | 0.500 | 0.554 | 0.504 | `class_label` is `true`/`false` against `humour`/`not humour` options |
| `mmsd` | 103 | 0.500 | 0.573 | 0.664 | n=103, text 0.664 is high relative to chance — treat with care |
| `daicwoz` | 47 | 0.500 | 0.702 | 0.607 | only 47 rows; `true`/`false` vs `depression`/`no depression` |

**Not scored at all** (open-ended free text, 1 distinct answer per item, so no
closed-label probe applies): `intentqa` (2134 rows, 1813 distinct answers),
`siq2` (875/865), `mimeqa` (173/173).

**A string-damage defect crosses several subsets.** The `joy` → `happy` alias
was applied as a blind substring replacement, corrupting 168 tokens in both the
model-facing `problem` field and the gold `answer` field:
`enjoy → enhappy`, `joyful → happyful`, `enjoyable → enhappyable`. Affected:
`siq2`, `ptsd_in_the_wild`, `mosei_emotion`, `meld_senti`, `intentqa`. For the
open-ended subsets this *inverts* the grading — a model writing correct English
is scored wrong.

Eight subsets have a `class_label` vocabulary that differs from the `answer`
vocabulary. Any evaluation that string-matches the wrong one silently scores zero.

---

## 5. The open question these tables do not answer

Every number above is model-free. They establish **availability**, not **use**.

DAIC is the only dataset where we have measured both, and there the two answers
disagreed sharply: ablation said the models ignore the interviewer shortcut
(AUC 0.546, at the noise floor), while the counterfactual said the same models
are strongly governed by it (CSS 0.206, 58.5% of predictions flip). Availability
did not predict use, and the cheaper test gave the wrong answer.

So for every dataset in Tables 1 and 2, the question *does any model actually use
this?* is currently unanswered. The two worth testing first are the two in
Table 1, because their shortcut mechanisms differ from DAIC's and from each
other:

- **`mmpsy_*`** — the shortcut is *structural* (how many answer segments were
  recorded), not lexical. The counterfactual is unusually clean: hold the
  transcript byte-identical and add or remove one empty answer segment.
- **`ptsd_in_the_wild`** — the shortcut is *register* / source identity. The
  counterfactual is a register transfer with the construct held fixed.

A non-zero CSS on either would generalise the DAIC result from one dataset to a
family spanning two languages and three distinct shortcut mechanisms.

