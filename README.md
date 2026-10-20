# Auditing and Mitigating Shortcut Learning in Mental Health Language Models

Code and aggregate results for *"Does Your Model Really Understand Mental Health
Data? Auditing and Mitigating Shortcut Learning in Mental Health Language
Models"* — NeurIPS 2026 Workshop GenAI4Health (poster).

The paper separates two things that are usually conflated: whether a
label-correlated shortcut is **available** in a dataset, and whether a model
actually **relies** on it. We audit four mental health corpora with model-free
probes, then measure reliance across eight language models with counterfactual
interventions that hold the participant's own words byte-identical and change
only the shortcut-bearing input.

## What is in this repository

| Path | Contents |
|---|---|
| `experiments/` | One directory per experiment: `code/`, `results/`, and an `analysis.md` where written |
| `scripts/` | Standalone runners for the earliest experiments (A1, C1) |
| `shortcut_harness/` | Shared scoring and media-loading helpers |
| `figures/` | Figures 2–4, the script that generates them, and paste-ready LaTeX captions |
| `paper/` | LaTeX source and bibliography |
| `to_human/` | Verification reports and result summaries written during the audit |
| `literature/` | Related-work survey |

Result files store **log-probabilities, gold labels and predictions only** — no
source text. That is a deliberate design choice and it is what makes this
repository publishable.

## What is not here, and why

**No data and no model weights.** Both are excluded by `.gitignore`:

- **DAIC-WOZ** is distributed under a use agreement that forbids
  redistribution, and the transcripts are clinical interviews with identifiable
  participants. Request access from the original distributor.
- **IMHI** is Reddit-derived; its terms forbid republishing post content.
- **Human Behavior Atlas**, **ESConv**, **mmpsy** and **ptsd_in_the_wild** are
  available from their own sources.
- **Checkpoints** (118 GB) are public models; see the paper's Setup section for
  the exact identifiers.
- **Training logs** are excluded because the GRPO runs were configured with
  `log_completions=True` and therefore contain verbatim transcripts.

To reproduce, obtain the corpora yourself and place them under `data/`; the
scripts expect `data/processed/daic_conditions.jsonl` and the per-corpus files
named in each experiment's code.

## Reproducing a figure

```bash
source env.sh
$SC_PY figures/code/make_figures.py        # writes PDF, SVG and PNG
```

Every value in every figure is recomputed from the raw result files at plot
time rather than transcribed, so a figure cannot drift from the numbers in the
text. See `figures/README.md`.

## Status

The paper is accepted; the camera-ready revision is in progress.
`HANDOFF_camera_ready.md` records what remains, what has already been verified,
and the pitfalls this project has already hit.
