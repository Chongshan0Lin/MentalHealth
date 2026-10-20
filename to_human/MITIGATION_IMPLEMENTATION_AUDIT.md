# Mitigation section — implementation audit

Written 2026-09-08 by reading code and result files only. Where the code and the
paper prose disagree, the code is reported. Every claim carries a file path and
a line or symbol reference.

**Legend used throughout:**

- **(a)** existing published method, implemented faithfully
- **(b)** our task-specific adaptation
- **(c)** exploratory / oracle variant — not deployable as stated
- **(d)** would be misleading if described generally in the paper

---

## 1. Targeted question removal

### 1.1 How the questions are selected

**They are not selected by a statistical test at run time.** The removal set is
a **hand-authored topical cluster**, read from a frozen file.

- Removal set: `experiments/suite/results/e4a_prompt_clusters.json`,
  key `clusters.clinical_care` — **10 prompt IDs, fixed**:
  `behavior_changes, bouts_symptoms, sleep_affects, symptoms_cope,
  therapy_changes, therapy_going, therapy_useful, trigger, when_diagnosed,
  why_seek_help`
- Authoring script: `experiments/suite/code/build_e4a_clusters.py`, constant
  `CLUSTERS` (lines 38-84). Its docstring is explicit:

  > "The clustering below is **HAND-AUTHORED** from the prompt texts … Clusters
  > are topical, **NOT association-based**. Grouping by odds ratio would build
  > the label into the partition."

**So: not Fisher, not BH, not top-k, no threshold.** There is no selection
criterion to report — the set is a curated topical partition of the 47
question-like scripted prompts into 9 clusters, of which `clinical_care` is one.

### 1.2 The label-knowledge question — this is the part that matters

The partition was **not fitted** on labels, but it was **authored knowing the
label associations**. `build_e4a_clusters.py:41-43` says so:

> "The adaptive clinical thread … **Contains every prompt that survived BH
> correction in A2a.**"

And A2a itself (`experiments/a2-ellie-counterfactual/code/a2a_prompt_label_association.py`):

| property | value | source |
|---|---|---|
| test | **Fisher exact** per prompt ID | `line 24` `from scipy.stats import fisher_exact`; `line 73` `orr, p = fisher_exact(tab)` |
| correction | **Benjamini-Hochberg**, own implementation | `bh()` at lines ~35-44; applied `line 82` |
| significance threshold | **q < 0.05** | `line 85` `sig = [r for r in recs if r["q"] < 0.05]` |
| which IDs are tested | asked to **20 ≤ n ≤ 0.84·N** participants | `line 29` `MIN_N, MAX_FRAC = 20, 0.84` |
| **data used** | **all 123 ID-bearing transcripts, no split** | `line 46` `rows = [json.loads(l) for l in open(COND)]`; no `split` filter anywhere |
| labels used | `s["y"] = r["phq8_binary"]` for every transcript | `line 55` |

**A2a uses the labels of every transcript, including the test split.** Four IDs
survived BH: `why_seek_help` (OR 11.26), `therapy_going` (10.42), `trigger`
(8.50), `when_diagnosed` (8.24) — all four are inside `clinical_care`.

### 1.3 Verdict on how to describe it

**It should be described as shortcut-aware, and the label-knowledge disclosed.**
Not "oracle" in the strict sense — the removal set is 10 IDs chosen topically,
not the 4 significant ones, and a topical clinical cluster is something a
domain expert could write without seeing any labels. But the author *had* seen
them, and the cluster is a superset of the significant set by construction.

Honest wording:

> Removal targets a hand-authored topical cluster of the ten clinical-thread
> prompts. The cluster was written with knowledge of which prompts were
> label-associated (A2a, Fisher exact + BH, q < 0.05, computed over all 123
> ID-bearing transcripts without a train/test split), so it is **shortcut-aware
> rather than blind**; it is not fitted to labels and requires no label access
> at inference.

**(b)** the cluster · **(d)** calling it "targeted removal" without saying the
target was chosen with label knowledge.

### 1.4 Removal at inference time

`experiments/i10-model-sweep/code/run_full_suite.py:124-132`:

```python
PAT = re.compile(r"^Ellie: ([A-Za-z][A-Za-z0-9_]*) \((.*)\)\s*$")
CL  = set(json.load(open(".../e4a_prompt_clusters.json"))["clusters"]["clinical_care"])
def strip_clinical(t):
    return "\n".join(l for l in (t or "").split("\n")
                      if not (PAT.match(l) and PAT.match(l).group(1) in CL))
S["targeted_removal"] = np.array(
    [s1(BASE.format(text=trunc(strip_clinical(r["text_full"])))) for r in rows])
```

Whole lines are dropped, matched on the **parenthesised prompt ID only** — not
on the question text, not on keywords.

### 1.5 Transcripts without prompt IDs — a reporting hazard

`PAT` requires the literal `Ellie: <id> (<text>)` form. **Only 123 of the 188
transcripts carry it** (verified by re-running the regex over
`data/processed/daic_conditions.jsonl`). On the other **65, `strip_clinical` is
a no-op** and the arm is byte-identical to `full`.

`run_full_suite.py:201` nevertheless scores AUC over all 188. So the reported
`targeted_removal` AUC is **diluted by a third with unmodified transcripts**.

`logs/mitigation_extra.log`, DIRECTION C, reports the correct comparison:

```
baseline (nothing removed), n=123: AUC 0.699
remove only clinical questions     0.665   -0.034
remove all 47 scripted questions   0.727   +0.027
remove ALL interviewer turns (D1)  0.787   +0.126   (n=188)
```

**(d)** Quoting `targeted_removal` AUC against the n=188 baseline (0.663) turns
−0.034 into +0.002 — a sign flip. See `VERIFICATION_confirmed.md §A2`.

### 1.6 CSS for this arm

`run_full_suite.py:201` writes `css=None`. We ran it separately:
`experiments/i12-targeted-cf/`, 8 models, using the 123-vs-65 split as an
internal control. **Five models show a reduction, three show the wrong
direction**, including MentaLLaMA (0.1336 stripped vs 0.1217 no-op control).

---

## 2. Counterfactual marginalisation

### 2.1 The contexts

`S^(k)` are **three fixed, hand-written five-question pools plus an empty one**,
defined once in `experiments/a2-ellie-counterfactual/code/run_a2b.py:42-64` and
imported by the suite (`run_full_suite.py:44-47`):

| arm | pool | provenance |
|---|---|---|
| `cf_none` | `[]` — no interviewer turns | — |
| `cf_neutral` | 5 biographical questions (origin, study, travel, likes/dislikes about L.A.) | hand-written |
| `cf_depression` | 5 clinically loaded questions (diagnosis, PTSD, mood, sleep, temper) | hand-written |
| `cf_empirical` | 5 questions, **the top A2a odds-ratio prompts** — comments give OR 11.26 / 10.42 / 8.50 / 8.24 / 4.00 | **derived from label associations** |

- **Global and fixed.** Not sampled, not per-participant, not per-model. The
  same 15 strings are used for every participant and every one of the 8 models.
- Insertion is deterministic: `interleave()` (`run_a2b.py:68-86`) places prompts
  at even intervals computed from the participant's line count only. Docstring:
  *"Positions depend only on the number of participant turns, so they are the
  same in every arm; C's bytes are never touched."*
- **No randomness, no seed dependence anywhere in this arm.**

### 2.2 Aggregation — the exact formula

`run_full_suite.py:192-193`:

```python
m2 = (S["cf_neutral"] + S["cf_depression"]) / 2
m4 = sum(S[a] for a in ARMS) / 4        # ARMS = cf_none, cf_neutral, cf_depression, cf_empirical
```

`S[a]` is **log-odds**, not probability: `s1()` at `run_full_suite.py:108-110`
returns `logP("Yes") − logP("No")` under teacher forcing.

So the implemented estimator is an **unweighted arithmetic mean in log-odds
space**:

$$\hat{s}_K(C_i)=\frac{1}{K}\sum_{k=1}^{K}\Big[\log p(\texttt{Yes}\mid C_i,S^{(k)})-\log p(\texttt{No}\mid C_i,S^{(k)})\Big]$$

with $K=2$ over $\{S^{\text{neutral}},S^{\text{depression}}\}$ and $K=4$ over
$\{\varnothing,S^{\text{neutral}},S^{\text{depression}},S^{\text{empirical}}\}$.

**This is a geometric mean of odds, not an average of probabilities.** If the
paper says "average the predictions", that is wrong as written.

### 2.3 Two things to disclose

**(d) `marginalise_4` contains a label-derived context.** `cf_empirical` was
built from the A2a odds ratios, i.e. from labels over all 123 transcripts. A
4-context average that includes it is **not** a label-free inference-time
method. `marginalise_2` is clean — both its pools are hand-written.

**(d) CSS = 0 for both is hard-coded, not measured.**
`run_full_suite.py:199-200` literally writes `css=0.0`. It is correct in
substance (averaging over contexts makes the score independent of which single
context was shown) but it is zero by construction. Only `instruct_ignore` has a
measured CSS (`line 204`).

**(a)** context marginalisation as a family · **(b)** the specific pools and
log-odds averaging · **(c)** `marginalise_4`, because of `cf_empirical`.

---

## 3. Proxy-based threshold correction

### 3.1 The proxy

**Interviewer-only clinical-word density.** `run_full_suite.py:179-183`:

```python
CLIN = re.compile(r"\b(diagnos\w*|therap\w*|depress\w*|p_t_s_d|ptsd|sleep|"
                  r"medicat\w*|counsel\w*|symptom\w*|feel\w*)\b", re.I)
feat = np.array([len(CLIN.findall(r["text_ellie_only"] or "")) /
                 max(1, len((r["text_ellie_only"] or "").split())) for r in rows])
```

A 10-stem regex over **`text_ellie_only`** — the interviewer's turns only, never
the participant's — normalised by interviewer word count. It is an observable
property of the input: no labels, no model.

### 3.2 The fitted model and the correction

`run_full_suite.py:184-191`:

```python
shift = S["full"] - S["participant_only"]      # log-odds attributable to the channel
sp    = np.array([r["split"] for r in rows])
tr    = sp != "test"                            # train + dev, n = 141
A     = np.vstack([np.ones(tr.sum()), feat[tr]]).T
co, *_ = np.linalg.lstsq(A, shift[tr], rcond=None)
S["threshold_proxy"] = S["full"] - (co[0] + co[1] * feat)
```

**Fit** — ordinary least squares of the channel-induced log-odds shift on the proxy:

$$\widehat{\Delta}(x)=c_0+c_1\cdot\mathrm{dens}(S_x),\qquad \Delta_i=s^{\text{full}}_i-s^{\text{participant-only}}_i$$

**Correction — applied in log-odds space:**

$$s^{\text{corrected}}_i=s^{\text{full}}_i-\big(c_0+c_1\cdot\mathrm{dens}(S_i)\big)$$

Not probability, not logit-of-something-else: the same `Yes − No` log-odds
scale as every other arm.

**Split:** coefficients fitted on `split != "test"` (**n = 141**), i.e. train+dev.

### 3.3 Two problems in how it is scored and applied

**(d) The stored AUC is in-sample contaminated.** Line 189 applies the
correction to **all 188 rows** and `line 202` scores AUC over all 188 — but 141
of those were used to fit `co`. Clean test-only numbers are in
`experiments/i11-threshold-offline/results/threshold_offline.json`:

| model | AUC full (188) | thr (188), contaminated | full (47) | **thr (47), clean** | Δ₄₇ 95 % CI |
|---|---:|---:|---:|---:|---|
| MentaLLaMA | 0.6836 | 0.6827 | 0.7468 | 0.7619 | [−0.0022, +0.0396] |
| OSim-8B | 0.8053 | 0.8066 | 0.7922 | 0.7987 | [+0.0000, +0.0226] |
| OmniSap-SFT | 0.7998 | 0.8077 | 0.7641 | 0.7857 | [−0.0229, +0.0673] |
| OmniSap-RL | 0.7860 | 0.7929 | 0.7597 | 0.7749 | [−0.0262, +0.0530] |
| OmniSap-2.0 | 0.7811 | 0.7913 | 0.7543 | 0.7792 | [−0.0175, +0.0738] |
| HumanOmniV2 | 0.7913 | 0.7907 | 0.7522 | 0.7879 | [−0.0208, +0.0973] |

Positive on all six on the clean split; **no interval clearly excludes zero.**

**(d) The correction extrapolates 3.6× outside its fitted support.** Measured
proxy range on real transcripts: **min 0.0000, max 0.0387, mean 0.0228**. The
counterfactual pools sit at `cf_depression` **0.1395** and `cf_neutral`
**0.0000**. Fitted slopes reach `c₁ = 90.09` (OmniSap-SFT) from a
`corr(feat, shift)` of only **0.043–0.244**, producing shifts up to **+10.67
log-odds**.

Consequence, from `threshold_offline.json`: signed CSS flips negative on **all
six** models — the correction overshoots — and on two of six the magnitude gets
*worse* (OmniSap-SFT 0.457 → 0.597; HumanOmniV2 0.372 → 0.656).

### 3.4 The clipped and refit variants

Both are in `experiments/i11-threshold-offline/results/threshold_refit.json`.

- **Clipped**: `shift = c0 + c1 * min(dens(S), max(feat_train))`, i.e. the proxy
  is clamped to the fitted support before the correction is applied. Uses
  training-set information only. **Improves \|CSS\| on 6/6.**
- **Refit**: coefficients re-solved on the two counterfactual arms themselves.
  **Improves 6/6 and drives MentaLLaMA's signed CSS to −0.0001**, but it is
  fitted on the very arms it is evaluated on.

**Answering the two questions directly:**

- **The clipped variant is a follow-up robustness fix**, introduced on
  2026-09-05 after the extrapolation was diagnosed. **It was not pre-specified**
  and must be reported as post-hoc. **(c)** → **(b)** once re-validated.
- **The refit variant is an oracle upper bound only.** It answers "is the method
  intrinsically capable" and must never be reported as a deployable result.
  **(c)**

### 3.5 CS under this correction — proved, not measured

The E5 construct pair holds the interviewer's turns fixed and varies only the
participant's content, so `feat` is identical for both members and the same
constant is subtracted from both log-odds. **The paired contrast is exactly
invariant in log-odds space.** The probability-space residual is not computed:
`run_full_suite.py` keeps the E5 arrays in a local `E` and stores only the
scalar `cs`, so per-item E5 scores are not on disk.

**(a)** counterfactual/threshold correction as a family · **(b)** the regex
proxy and the log-odds shift model.

---

## 4. Supervised training-time debiasing

All four arms: `experiments/i6-encoder-interventions/code/run_encoder_interventions.py`.

### 4.0 Shared setup

| item | value | source |
|---|---|---|
| split | `r["split"]` from `data/processed/daic_conditions.jsonl` → train / dev / test | `main()` line ~176 |
| **n_train = 107**, n_test = 47 | logs confirm `n_train=107` (428 for D3) | `logs/round2/j3_mentalbert.log` |
| seeds | **[0, 1, 2]** — 3 seeds per arm | `--seeds` default |
| encoder | `ChunkedEncoder`, AdamW, lr 2e-5, ≤20 epochs, early stop patience 5, grad-accum 8 | `train_one()` signature |
| model selection | best **dev AUC**, state restored | `train_one()` lines ~148-155 |

**Which checkpoints:** the script defaults to `--ckpt checkpoints/MentalBERT`.
Both were run, **to the same output path**, and the second overwrote the first:

| run | log | date | checkpoint |
|---|---|---|---|
| 1 | `logs/i6_encoder_interventions.log` | 2026-08-24 | **MentalBERT** |
| 2 | `logs/i6_roberta.log` | 2026-08-30 | **MentalRoBERTa** — overwrote the JSON |
| 3 | `logs/round2/j3_mentalbert.log` | 2026-09-05 | **MentalBERT**, re-run after backup |

**(d)** `results/encoder_interventions.json` currently holds MentalRoBERTa. The
paper's PoE numbers are MentalBERT's. Confirmed reproduced 2026-09-05 (M1
0.540 / 0.0039, exact match). Backup at
`results/encoder_interventions_MentalRoBERTa.json`.

### 4.1 The bias model — shared by M1 and M2, and it leaks

`bias_model(rows)`, lines ~55-80:

- **Input:** binary asked/not-asked vector over the **47 question-like scripted
  prompt IDs**. No text, no counts, no participant words. So it is a
  **shortcut-only expert**, not a weak full-input model.
- **Classifier:** `LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)`
- **Frozen:** fitted once before training, never updated. Yes, frozen.
- Transcripts without IDs fall back to the corpus base rate.

**(d) It is fitted on `rows` — every transcript, including test —
using `r["phq8_binary"]`.** There is no split filter in `bias_model`. The
resulting `pbias` feeds M1's training loss and M2's group assignment, so both
methods are trained with a component that has seen test labels. The effect is
indirect (only train rows enter the loss) but it must be disclosed.

### 4.2 Counterfactual augmentation (D3)

```python
def augment(train):
    aug = list(train)
    for r in train:
        for arm in ("cf_neutral", "cf_depression", "cf_empirical"):
            aug.append({**r, "text_full": interleave(r["text_participant_only"], POOLS[arm]),
                        "_aug": arm})
    return aug
```

- **Changed:** the interviewer's turns, replaced wholesale by one of three fixed
  pools. **Held fixed:** the participant's own words, byte-identical.
- **Labels preserved**, unchanged (`{**r, ...}` keeps `phq8_binary`).
- **3 augmented copies per original**, so 4× the training set: 107 → 428.
- **(d)** one of the three pools is `cf_empirical`, the label-derived one. D3 is
  therefore **shortcut-aware** in the same way `marginalise_4` is.

### 4.3 Group DRO (M2)

**Group definition** (`main()`, line ~186):

```python
groups = {r["uid"]: 2 * r["phq8_binary"] + int(pbias[r["uid"]] > 0.5) for r in rows}
```

→ **4 groups = label × predicted-shortcut-status**, where shortcut status is the
bias model's thresholded prediction. Not interviewer-condition status.

**Objective** (`train_one()`, `method == "M2"`):

```python
li = ce_none(z, t)
gw_ = gw.clone()
gw_[g] = gw_[g] * torch.exp(0.01 * li.detach())
gw = (gw_ / gw_.sum()).detach()
loss = gw[g] * li.squeeze(0) * ng
```

Online exponentiated-gradient update of the group weights with **step size
η = 0.01**, renormalised each example, then the per-example loss is reweighted
by `gw[g] * n_groups`. This is Sagawa et al. (2019)'s online algorithm **(a)**,
applied per-example rather than per-minibatch **(b)** (n_train = 107 with
grad-accum 8).

### 4.4 Group DRO + heavy L2 (M2reg)

**Only `weight_decay` differs**: `wd = 0.1 if method == "M2reg" else 0.01`
(`main()`, line ~194). Everything else — groups, η, lr, epochs, patience — is
identical. Motivated by Sagawa et al.'s finding that group DRO needs strong
regularisation on overparameterised nets; the docstring cites it. **(a)**

### 4.5 Product-of-Experts (M1)

```python
pb = pbias[r["uid"]]
zb = torch.tensor([[np.log(max(1 - pb, 1e-6)), np.log(max(pb, 1e-6))]], ...)
loss = ce(z + alpha * zb, t)          # alpha = 1.0
```

- **Bias expert:** the frozen question-selection logistic model of §4.1.
- **Input it sees:** the 47-dim asked/not-asked vector — **shortcut-only**.
- **Combination:** the bias expert's **log-probabilities are added to the main
  model's logits** before the softmax, i.e. **logit space**, α = 1.0. This is a
  product in probability space.
- **Loss:** ordinary cross-entropy on the *combined* logits, back-propagated
  into the main model only (`zb` is a constant tensor). Examples the bias model
  already gets right generate little gradient. This is Clark et al. (2019). **(a)**

### 4.6 Results — MentalBERT, 3 seeds, test n = 47

| arm | AUC | CF-Sens ↓ | C-Sens ↑ | SER ↓ |
|---|---:|---:|---:|---:|
| B0 ERM | 0.705 | 0.0413 | 0.2256 | 0.18 |
| **D3 counterfactual augmentation** | **0.734** | **0.0332** | **0.4342** | **0.08** |
| M1 PoE | 0.540 | 0.0039 | **0.0406** | 0.10 |
| M2 group DRO | 0.659 | 0.0444 | 0.2104 | 0.21 |
| M2reg | 0.692 | 0.0524 | 0.3844 | 0.14 |

**(d) PoE's 91 % CF-Sens reduction comes with an 82 % C-Sens reduction.** The
C-Sens column did not exist in the 2026-08-24 run that the paper's PoE row was
written from. D3 is the only arm that improves all three axes and it is
currently unreported.

---

## 5. GRPO with an invariance penalty

`experiments/i8-grpo-mitigation/code/{build_pairs,train_grpo,eval_grpo}.py`.

### 5.1 Base checkpoint and adapter

`MentaLLaMA-chat-7B` (`--model` default), trained with **LoRA**, not full
fine-tuning: `r=16, lora_alpha=32, lora_dropout=0.05`, targets
`q,k,v,o,gate,up,down_proj`.

### 5.2 The reward — decomposed

$$R \;=\; R_{\text{accuracy}} \;+\; \lambda\,R_{\text{inv}}$$

**Task term.** Two forms:

- `r_accuracy` — `1.0 if pred_of(c) == label else 0.0`
- `make_r_accuracy_balanced(p_pos)` — inverse-frequency weights
  `w[Yes] = 0.5/p_pos`, `w[No] = 0.5/(1−p_pos)`. Rationale in the docstring:
  28 % of DAIC labels are positive, so constant "No" scores 0.72 under plain
  accuracy; weighting caps the constant policy at 0.50.

**Shortcut term** — `make_r_inv(mode, lam, log_path)`, **pair-level**:

- For each counterfactual pair, take the **majority prediction across that
  arm's rollouts**, for each of the two arms.
- `mode="consistent"`: award `lam` **if the two majorities agree**, else 0.
- `mode="correct"` (the M1′ fallback): award `lam` only if they agree **and**
  the agreed answer is right.

**λ = 1.0** (`--lam` default). **Yes, the reward uses counterfactual interviewer
contexts** — pairs are built in `build_pairs.py` from the frozen suite arms,
holding symptom evidence byte-identical and changing only the interviewer's
questions, emitted **adjacently** with `shuffle_dataset=False` so both members
land in one generation batch.

**(b) This is a task-specific reward we designed. It is not standard GRPO.**
`R_inv` is a bespoke shortcut-invariance term over counterfactual pairs. The
paper must call it *"GRPO with a counterfactual-invariance reward"*; describing
it as a GRPO baseline would be wrong. **(d)**

**A known exploit, documented in the code (`train_grpo.py:17-26`).** `R_inv`
rewards agreement, and **a constant policy agrees perfectly** — so R_inv is
hackable by collapse. The `monitor_only` reward function logs `p_yes` and
`parse_rate` every step precisely to catch this, and `mode="correct"` exists as
the fallback. **The collapse happened anyway** (§5.5).

### 5.3 Hyperparameters (`GRPOConfig`, lines 186-210)

| setting | value |
|---|---|
| `num_generations` (group size) | **8** |
| `generation_batch_size` | 16 = 8 × 2, so both pair members generate together |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 8 |
| `scale_rewards` | `"group"` (TRL default) or `"batch"` |
| `beta` (**KL coefficient**) | **0.0 — no KL penalty**, "per the design" |
| `learning_rate` | **2e-5** |
| `num_train_epochs` | 1 (epochs baked into the pair file) |
| `max_completion_length` | 64 |
| `shuffle_dataset` | **False** (pair adjacency) |
| dtype | bfloat16, gradient checkpointing on |
| seed | 0 |
| **temperature** | **not set — TRL default** |

**(d)** Temperature is never specified in the code; do not state a value in the
paper without checking the installed TRL default.

### 5.4 The three evaluated runs

Tag is built at `train_grpo.py:156-158`:
`tag = arm + ("prime" if reward_inv=="correct") + ("_bal" if balanced) + ("" if scale_rewards=="group" else "_"+scale_rewards)`

| tag | arm | task reward | R_inv | reward scaling |
|---|---|---|---|---|
| **M0** | M0 | plain accuracy | **off** (monitor only, λ=0) | group |
| **M0_bal_batch** | M0 | **balanced** accuracy | **off** (monitor only) | **batch** |
| **M1** | M1 | plain accuracy | **on, `consistent`, λ=1.0** | group |

A fourth checkpoint `ckpt_M0_bal` exists with **no eval file** — trained, never
evaluated.

**So M0 and M0_bal_batch are ablations of the task reward and the advantage
scaling; only M1 carries the shortcut penalty.**

### 5.5 Results and the collapse

`results/eval_*.json`:

| run | p_yes | CSS | construct_sens | iid AUC | conflict AUC | parse_rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.3936 | 0.2864 | 0.1354 | 0.7630 | 0.0686 | **0.15** |
| M0 | **0.0000** | 0.00169 | 0.00246 | 0.7245 | 0.2664 | 1.00 |
| M0_bal_batch | **0.0000** | 0.0000444 | 0.000635 | 0.6752 | 0.3427 | 1.00 |
| M1 | **0.0000** | 0.00143 | 0.00329 | 0.7096 | 0.2768 | 1.00 |

**All three collapse to constant "No" on all 188 participants.** The training
monitors agree: `degeneracy_{M0,M0_bal_batch,M1}.json` all report
`p_yes_final = 0.00149`.

**(d) "CSS 0.000" and "CS 0" are rounded.** CSS is 0.0014–0.0017 for M0/M1 and
4.4e-5 for M0_bal_batch; CS is 0.0006–0.0033. Write "≤ 0.002".

**(d) AUC does not collapse** (0.675–0.725 vs baseline 0.763). Ranking survives
because the teacher-forced log-odds still vary even though the argmax never
does — another instance of the paper's own thesis.

**(d) Three configurations, one run each.** There is no seed variation within a
configuration. "Collapse occurred for all training seeds" cannot be said;
"for all three configurations we tried" can.

### 5.6 Parsing, and why baseline parse rate is 15 %

**Training** parses with `pred_of(c)`: `ANS = re.compile(r"\b(yes|no)\b", re.I)`,
first match anywhere in the completion, capitalised.

**Evaluation never parses.** `eval_grpo.py:4-9`: every score is a
**teacher-forced label log-likelihood**, `logP("Yes") − logP("No")`, because
"parse-failure rate correlates with perturbation strength". `parse_rate` is a
**separate diagnostic**, measured on only the **first 20 rows** with
`max_new_tokens=200`, greedy (`do_sample=False`) — `eval_grpo.py:74-82, 133`.

**Why 0.15 vs 1.00:** the baseline is untrained MentaLLaMA-chat-7B, which emits
IMHI-style free-text explanations that frequently contain no standalone `yes`/`no`
token within 200 tokens. GRPO explicitly optimises `r_accuracy`, whose only
route to reward is emitting a parseable bare `Yes`/`No` — so the trained
policies hit 1.00 trivially, and did so **by collapsing to the single token
"No"**. The parse-rate jump is a symptom of the collapse, not evidence of
improved instruction-following.

**(d)** Reporting the baseline's 0.15 parse rate without this explanation
invites the reading that GRPO "fixed" output formatting.

---

## Summary of what must not be said generally

| # | Claim to avoid | Why |
|---|---|---|
| 1 | "targeted removal" without qualification | the removal set was authored with knowledge of A2a's label associations, computed over all 123 transcripts with no split |
| 1 | quoting its AUC against the n=188 baseline | it is a no-op on 65/188; correct comparison is n=123, and the sign flips |
| 2 | "we average the model's predictions" | the code averages **log-odds**, a geometric mean of odds |
| 2 | `marginalise_4` as label-free | one of its four contexts (`cf_empirical`) is derived from label associations |
| 2 | "CSS → 0" for delete/mask/marginalise | hard-coded `css=0.0` at `run_full_suite.py:197-200` |
| 3 | the stored threshold AUC | fitted on 141 of the 188 rows it is scored on |
| 3 | the clipped variant as designed-in | it is a post-hoc fix found on 2026-09-05 |
| 3 | the refit variant at all, as a method | fitted on its own evaluation arms |
| 4 | M1/M2 as clean train-time baselines | their bias model is fitted on all rows including test |
| 4 | D3 as label-free augmentation | uses the label-derived `cf_empirical` pool |
| 4 | PoE's CSS reduction alone | it cuts construct sensitivity by 82 % at the same time |
| 4 | encoder results from the JSON | it holds MentalRoBERTa; the paper's numbers are MentalBERT |
| 5 | "GRPO baseline" | it is GRPO with a bespoke counterfactual-invariance reward |
| 5 | "CSS 0.000 / CS 0" | rounded; true values ≤ 0.002 / ≤ 0.0033 |
| 5 | "all seeds collapsed" | three configurations, one run each, no seed replication |
| 5 | the parse-rate jump as a benefit | it is a symptom of collapsing to a single token |
