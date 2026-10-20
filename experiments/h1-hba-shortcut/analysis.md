# H1 — What shortcuts live in the HBA-test datasets?

Run 2026-08-31. Code: `code/audit.py` · results: `results/audit.json`.
Model-free throughout: no model is loaded anywhere in this experiment.

This is the A2a move applied to a second benchmark. A2a established that DAIC's
label is recoverable from *which questions the interviewer chose to ask*, with no
model involved. The same question is asked here of all 16 HBA-test subsets: **can
the label be recovered from something that is not the construct?** If it can,
there is a shortcut for a model to find, and no model run is needed to say so.

**Text only.** Audio and video are withheld deliberately. A benchmark that
advertises itself as measuring human behaviour, but whose labels are recoverable
from the transcript alone, is not measuring behaviour — and withholding the other
channels is the only way to see that.

---

## 0. The probe is calibrated on two datasets whose answer we know

`cremad` and `tess` are acted-emotion corpora in which every speaker reads the
*same fixed carrier sentence*, chosen to be emotionally neutral. Text-only
accuracy on them must be at chance by construction. It is:

| subset | chance | text-only balanced acc |
|---|---|---|
| cremad | 0.167 | **0.170** |
| tess | 0.143 | **0.102** |

Nothing below is read as a shortcut unless it clears this floor.

---

## 1. `ptsd_in_the_wild` is corpus identification, not PTSD detection

n = 64, perfectly balanced 32/32, so chance is 0.500.

| probe | balanced acc |
|---|---|
| full transcript, TF-IDF | **0.948** |
| after scrubbing *all* PTSD/military vocabulary | **0.953** |
| first 100 words only | 0.844 |
| **stopwords only — no content words at all** | **0.880** |
| first 25 words only | 0.719 |
| transcript length alone | AUC 0.770 |

The top features name the two source corpora rather than the construct:

```
ptsd     ptsd, va, war, vietnam, military, veterans, marine, iraq, corps,
         served, traumatic, therapy, treatment, symptoms
no_ptsd  essentials, wear, favorite, watch, business, hair, travel, bag,
         fan, wearing, cool, nice, gotta, gonna
```

The negative class is lifestyle vlog content ("what's in my bag", "my
essentials"); the positive class is veteran testimony. Three findings follow, in
increasing order of severity:

1. **The label word is in the input.** 19 of 32 positives contain the literal
   string `ptsd`; 0 of 32 negatives do. 13 of 32 positives contain `trauma`;
   0 negatives do.
2. **Scrubbing that vocabulary changes nothing** (0.948 → 0.953). The corpora
   differ on everything, not on a removable word list, so no sanitisation fixes
   this dataset.
3. **Function words alone reach 0.880.** With every content word deleted, the
   register still separates the classes. That is the signature of two production
   sources, and it is what makes the task unfixable rather than merely leaky.

The prompt asserts a premise that is false for half the data: *"Above is a video
along with the transcript of a patient undergoing clinical evaluation."* The
negative examples are not patients and there was no clinical evaluation. One
negative ends: *"Go out. Talk to people. Listen to people. And most importantly,
be prepared to be amazed. Thanks."*

**Any number reported on this subset — unimodal or multimodal — measures corpus
identification.** It cannot be interpreted as PTSD detection.

---

## 2. `mmpsy_*`: the label is in the *number of recorded answers*

The Chinese subsets need the two length measures separated, because they
disagree and only one carries the signal:

| subset | chance | text | total characters | **answer segments** |
|---|---|---|---|---|
| mmpsy_depression | 0.500 | 0.507 | 0.509 | **0.772** |
| mmpsy_anxiety | 0.500 | 0.500 | 0.429 | **0.771** |

Transcript *content* carries nothing, and total verbosity carries nothing, but
**how many separate answers were recorded** carries a great deal:

```
depression      segments: {7: 7,  8: 22, 9: 5, 14: 1}    63% have exactly 8
no_depression   segments: {7: 115, 8: 33, 6: 4, 10: 2}   74% have exactly 7
```

The depression group has one more recorded response than the control group.
This is DAIC's shortcut in a different corpus and a different language: the
*data collection protocol* differs by class, so a structural property of the
record predicts the label without any behavioural content being read.

The 0.772 comes from a forest on that single feature, not a threshold, because
the informative pattern is an interval ("exactly 8") rather than a cut point —
the best single threshold reaches only 0.691.

Two further problems in the same subsets:

- **Severe skew.** `mmpsy_anxiety` is 93.7 % majority (12 positives in 191) and
  `mmpsy_depression` is 81.7 %. Always answering the majority beats every
  content-based result reported above.
- **The prompt is malformed.** Two instructions are concatenated and the first is
  truncated mid-sentence:

  > "…what is the speaker's likely\n\nBased on the speech content, tone, and
  > prosody, classify the speaker's depression severity using the PHQ-9
  > categories: depression, no depression.\n\n?"

  It also misdescribes its own label space: PHQ-9 and GAD-7 categories are
  five-level severity scales, but the options offered are binary.

- The two subsets are the **same 191 participants** with two label sets, so they
  are not independent evidence.

---

## 3. `joy` → `happy` was applied as a blind substring replacement

168 corrupted tokens across 5 subsets, in both the model-facing `problem` field
and the gold `answer` field:

```
enjoy → enhappy       enjoying → enhappying     enjoyed → enhappyed
joyful → happyful     joyous → happyous         enjoyable → enhappyable
```

Example from `siq2`, where the corrupted string *is* the gold answer:

```
answer      : "The officers are enhappying themselves as they are laughing
               over a cup of coffee together."
class_label : "the_officers_are_enjoying_themselves_as_they_are_laughing_…"
```

For the open-ended QA subsets this inverts the grading: a model that writes
correct English is scored wrong. Affected: `siq2`, `ptsd_in_the_wild`,
`mosei_emotion`, `meld_senti`, `intentqa`.

---

## 4. `meld_emotion`: 23.4 % of items have an unselectable answer

610 of 2610 rows carry gold labels `joy` / `sadness` while the prompt's option
list offers `happy` / `sad`. The correct answer is not on the menu. This is the
same aliasing failure as §3, seen from the label side rather than the text side.

Eight subsets have a `class_label` vocabulary that differs from the `answer`
vocabulary (`true`/`false` vs `humour`/`not humour`, `angry` vs `anger`,
`strongly_negative` vs `highly negative`, …). Any evaluation that string-matches
against the wrong one of the two silently scores zero.

---

## 5. Register leakage in the MOSEI subsets

| subset | chance | text | **stopwords only** |
|---|---|---|---|
| mosei_senti | 0.143 | 0.260 | **0.203** |
| mosei_emotion | 0.167 | 0.281 | **0.258** |

For `mosei_emotion` the function words alone recover almost all of what the full
text recovers (0.258 of 0.281). The construct contributes very little beyond
register. Milder than §1 but the same kind of defect.

---

## 6. What is usable

| subset | verdict |
|---|---|
| `cremad`, `tess` | clean — text at chance by design |
| `meld_emotion`, `meld_senti`, `chsimsv2`, `urfunny`, `mmsd` | modest text signal, no register leak; usable with the label-space fix |
| `mosei_senti`, `mosei_emotion` | register leakage; report with the stopword control |
| `mmpsy_depression`, `mmpsy_anxiety` | protocol artifact + 82–94 % skew + broken prompt |
| `ptsd_in_the_wild` | **not a clinical task**; do not report |
| `daicwoz` (HBA slice) | only 47 rows, and `class_label` uses `true`/`false` against `depression`/`no depression` options |

---

## 7. How this relates to the DAIC result

The paper's DAIC finding is that an *adaptive data-collection protocol* writes
the label into the input, so a model can reach high accuracy without reading the
patient. §2 is the same finding in a different corpus and language: the number of
recorded responses differs by class. §1 is the more extreme version — there the
two classes were *collected from different sources entirely*, so protocol and
label are perfectly confounded.

This generalises the paper's claim from one dataset to a family: **in mental
health corpora the label tends to be recoverable from how the data was gathered,
not only from what the participant said.** Every probe in this file is
model-free, so this is a statement about the datasets and holds for any model
that is ever evaluated on them.
