# DEPLOYMENT_NOTES — day-1 deployment

**Deployed:** 2026-08-09 · **Host:** 10× RTX A6000 (48 GB) · **Status:** all 3 models + DAIC-WOZ verified working

---

## Final state

| Component | Path | Size |
|---|---|---|
| MentaLLaMA-chat-7B | `checkpoints/MentaLLaMA-chat-7B` | 26 GB |
| OmniSapiens2.0 | `checkpoints/OmniSapiens2.0` | 17 GB |
| HumanOmniV2 | `checkpoints/HumanOmniV2` | 17 GB |
| DAIC-WOZ (audio + transcript + labels) | `data/raw/daic_woz` | 5.4 GB |
| Manifest | `data/processed/daic_manifest.jsonl` | 189 rows |
| Conda env | `/data/cl195/conda/envs/sc-shortcut` | — |

**Total: 64 GB** (vs ~250 GB for the naive full-download route).

Environment: `torch 2.6.0+cu124 · transformers 5.14.1 · huggingface_hub 1.27.0 · librosa 0.11.0 · pandas 3.0.5 · python 3.11`

One env serves all three models — MentaLLaMA is `LlamaForCausalLM` and both
Omni models are `Qwen2_5OmniThinkerForConditionalGeneration`, all supported
natively by stock transformers. **No repo clone, no FlashAttention build, no
decord.** That is what made a same-day deployment possible.

---

## Usage

```bash
source /data/cl195/NeurIPS_workshop/env.sh     # MUST be sourced first (see gotcha 1)
$SC_PY scripts/smoke_text.py                   # MentaLLaMA
CUDA_VISIBLE_DEVICES=1 $SC_PY scripts/smoke_omni.py OmniSapiens2.0 --seconds 60
CUDA_VISIBLE_DEVICES=1 $SC_PY scripts/smoke_omni.py HumanOmniV2 --seconds 60
```

VRAM: MentaLLaMA 13.8 GB, Omni models 18.5 GB each (60 s audio). Load time ~25 s.

---

## Data inventory

`data/processed/daic_manifest.jsonl` — one row per participant with
`uid, participant_id, split, phq8_binary, phq8_score, gender, audio_path,
transcript_path, audio_seconds, n_turns, n_participant_turns, participant_words`.

| split | n | positive | audio | transcript | median len | median participant words |
|---|---|---|---|---|---|---|
| train | 107 | 30 (28%) | 107 | 107 | 14.7 min | 1198 |
| dev | 35 | 12 (34%) | 35 | **34** | 16.6 min | 1397 |
| test | 47 | 14 (30%) | 47 | 47 | 16.1 min | 1414 |

Total 189 participants, 50.2 hours of audio.

**DAIC-WOZ contains no video.** The distributed archives carry only derived
features (CLNF landmarks/HOG/gaze/AUs/pose, COVAREP, FORMANT), plus raw audio
and transcripts. Any video-modality work needs a different corpus.

Audio: 16 kHz mono WAV. Transcripts: tab-separated, `start_time / stop_time /
speaker / value`, speakers `Ellie` (virtual interviewer) and `Participant`.

---

## Gotchas hit during deployment (all fixed, all load-bearing)

**1. `$HOME` is 100 % full (4.5 GB free, 29.3 GB quota).**
Every cache must be redirected to `/data` or downloads die mid-flight.
`env.sh` sets `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, `PIP_CACHE_DIR`,
`XDG_CACHE_HOME`, `TMPDIR`, `CONDA_PKGS_DIRS`. Source it before anything.

**2. The DAIC-WOZ host WAF 403s most User-Agents.**
Both the default `python-requests/...` UA and a spoofed `Mozilla/5.0` are
rejected on the directory index. `curl/7.81.0` is accepted. Set in
`scripts/fetch_daic.py`.

**3. Selective zip extraction over HTTP Range — 96 % bandwidth saved.**
The server supports Range requests, so `scripts/fetch_daic.py` reads each
zip's central directory and pulls only `_AUDIO.wav` + `_TRANSCRIPT.csv`,
skipping `_CLNF_hog.bin` (≈85 % of each archive) and the landmark dumps.
Result: **3.04 GB transferred instead of 90 GB, in 51 s** with 12 workers.
Add members via `--members audio,transcript,covarep,...` if needed later.

**4. `440_P.zip` is truncated server-side.**
Valid PK header, no end-of-central-directory record; the archive stops inside
`440_CLNF_hog.bin`. Members stored after that point — COVAREP, FORMANT,
CLNF_pose and **TRANSCRIPT** — are unrecoverable.
`440_AUDIO.wav` sits at offset 0 and was recovered by inflating the raw
deflate stream directly (1414 s, 16 kHz, verified). **440 is in the dev split**,
so dev has 35 audio but only 34 transcripts. Not a fetch bug — the file is bad
at the source.

**5. `487_P.zip` contains a macOS AppleDouble sidecar.**
Both `._487_TRANSCRIPT.csv` (4 KB resource fork) and the real
`487_TRANSCRIPT.csv` (20 KB) are present. A naive `endswith()` match grabs the
sidecar. Fixed with exact-basename matching plus a `__MACOSX` guard.

**6. Some transcripts are cp1252, not UTF-8.**
`pd.read_csv(..., encoding='utf-8')` raises on byte `0xb0`.
`build_daic_manifest.py` falls back utf-8 → cp1252 → latin-1 and records the
encoding used per file.

**7. Do NOT cast `input_features` to bfloat16 for the Omni models.**
Passing the Whisper features through `.to(torch.bfloat16)` silently corrupts
them and the model degenerates into repeated `Human` tokens with no error.
Keep them float32; the audio tower casts internally.

**8. `OmniSapiens2.0` ships an empty `generation_config.json`.**
It contains only `_from_model_config` — no `eos_token_id`. `generate()`
therefore never stops at `<|im_end|>`, and the model emits a correct short
answer then rambles on in fabricated `Human:` turns. Pin
`eos_token_id=151645, pad_token_id=151643` explicitly. HumanOmniV2 sets these
correctly and needs no fix.
*This one is easy to misread as "the model is broken" — it is not; the answer
before the rambling is correct.*

**9. Resource-sheet corrections.** `mHealthAI/Menta` has **no merged
transformers weights** — only a 16 GB GGUF plus a 24 MB LoRA adapter
(`r=8`, q/k/v/o) against `Qwen/Qwen3-4B-Instruct-2507`, with optimizer state
left in. `ooochen/MMEVerse` is 0.16 GB of annotations only; the media must come
from upstream MER. Neither is deployed.

---

## Verified smoke-test outputs

- **MentaLLaMA** — on `daic/train/303` (label 0) generated *"No, the poster does
  not suffer from depression"* with reasoning; label log-probs `No −0.68` vs
  `Yes −2.05`. Label-scoring path works, so calibrated probabilities are
  available rather than only free text.
- **OmniSapiens2.0** — 60 s of participant audio → *"The person is calm."*
  Also transcribes accurately when asked ("hi I'm Ellie thanks for coming").
- **HumanOmniV2** — emits its documented `<context>…</context><think>…` format
  unprompted.

---

---

# Wave 2 — small models + text datasets (2026-08-09)

| Added | Size | Status |
|---|---|---|
| OSim-8B | 16 GB | ✅ smoke-tested, 16.4 GB VRAM |
| Menta (Qwen3-4B base + LoRA) | 7.6 + 0.04 GB | ✅ smoke-tested, 8.1 GB VRAM |
| Emotion-LLaMA-v2 | 1.6 GB | ⚠️ weights only, **not runnable** (see below) |
| IMHI test sets (9 datasets) | 48 MB | ✅ indexed |
| IMHI train/val (5 released) | 71 MB | ✅ on disk |
| SDCNL | 10 MB | ✅ indexed |
| MultiWD raw | 3 MB | ✅ on disk |

**Project total: 89 GB.**

## Text data inventory — `data/processed/text_manifest.jsonl` (20,946 rows)

| dataset | rows | task | labels | distribution |
|---|---|---|---|---|
| swmh | 10,882 | multi | 5 | depression 3774 / none 2009 / suicide 1947 / anxiety 1901 / bipolar 1234 |
| MultiWD | 2,441 | binary | 2 | no 1554 / yes 887 |
| Irf | 2,113 | binary | 2 | yes 1366 / no 747 |
| SDCNL | 1,895 | binary | 2 | yes 980 / no 915 |
| t-sid | 959 | multi | 4 | none 501 / suicide-or-self-harm 253 / depression 154 / ptsd 51 |
| SAD | 684 | multi | 9 | work 110 / health 85 / financial 81 / school 76 / other 76 |
| CAMS | 625 | multi | 6 | alienation 184 / relationship 174 / none 107 / jobs 68 / medication 64 |
| loneliness | 528 | binary | 2 | yes 413 / no 115 |
| dreaddit | 414 | binary | 2 | yes 211 / no 203 |
| DR | 405 | binary | 2 | yes 307 / no 98 |

Schema: `uid, source_dataset, corpus, split, task_type, text, question, label,
gold_explanation, n_words`.

## Wave-2 gotchas

**10. The IMHI *test* sets for CAMS, loneliness, swmh and t-sid ARE public.**
The MentaLLaMA README's "Released: Not yet" column refers to *training* data
only. All 9 test sets ship in `repos/MentaLLaMA/test_data/`, with row counts
matching the README table exactly. **Only CLP is genuinely absent.** This
shrinks the outstanding access list from five datasets to one.

**11. IMHI labels are not a column.** They are the opening clause of the
`gpt-3.5-turbo` gold explanation and must be parsed out. Phrasings differ per
dataset: binary sets open with `Yes,`/`No,`; `swmh` uses "symptoms **related
to** X"; `SAD` uses "the stress cause related to X" plus a catch-all "other
stress causes"; `CAMS`/`t-sid` have their own forms. `build_text_manifest.py`
handles all of them and stores the raw clause so a mis-parse is recoverable.

**12. 17 swmh gold responses (0.16 %) are off-schema.** They skip the fixed
prefix and name conditions outside swmh's 5-label space (psychosis,
schizophrenia, agoraphobia, BPD, serotonin syndrome). Left as `label: null`
rather than force-fitted — this is noise in the source data, not a parser bug.

**13. OSim-8B has the same missing-EOS behaviour as OmniSapiens2.0** — it
answers, then continues into fabricated `User:` turns. Pin the stop token when
generating.

**14. Emotion-LLaMA-v2 is weights-only.** The repo contains just `stage1.pth`
and `stage2.pth` (858 MB each) — no config, no tokenizer, no base model. It
needs the project repo's architecture, a separate base LLM, and MMEVerse media
that is annotation-only on HF. Downloaded but **not deployed or verified**.

---

# Wave 3 — evaluation benchmarks, test splits only (2026-08-09)

The benchmarks each deployed model was actually evaluated on.

| Benchmark | Serves | Size | Items |
|---|---|---|---|
| HBA test split | OmniSapiens2.0 | 70.5 GB | 18,204 rows / 60 shards |
| WorldSense | HumanOmniV2 | 18.1 GB | 1,662 videos |
| IntentBench | HumanOmniV2 | 8.1 GB | 2,689 QA |
| Daily-Omni | HumanOmniV2 | 3.9 GB | 1,197 QA |
| osim-post-training `test/` | OSim-8B | 0.02 GB | 27 val parquets |

**Project total: 183 GB.**

**15. Three of the five are evaluation-only releases.** IntentBench,
Daily-Omni and WorldSense have no train/test split — the entire repo *is* the
test set, so "fetch only the test split" saves nothing there. Real filtering
applied only to `osim-post-training` (`test/` only: 0.02 GB instead of 0.19 GB)
and HBA (`test-*.parquet` only: 70.5 GB instead of 448 GB).

## HBA test split composition — `data/raw/HBA-test`

Schema: `problem, answer, dataset, modality_signature, task, class_label,
audios, videos, images, ext_video_feats, ext_audio_feats`. Media is embedded
binary inside the parquet, so columns cannot be fetched selectively.

All 16 sources present (18,204 rows):

| source | n | source | n |
|---|---|---|---|
| mosei_senti | 4,659 | siq2 | 875 |
| meld_emotion | 2,610 | tess | 420 |
| meld_senti | 2,609 | urfunny | 224 |
| intentqa | 2,134 | mmpsy_depression | 191 |
| mosei_emotion | 1,753 | mmpsy_anxiety | 191 |
| cremad | 1,117 | mimeqa | 173 |
| chsimsv2 | 1,034 | mmsd | 103 |
| | | ptsd_in_the_wild | 64 |
| | | daicwoz | 47 |

Modality: `text_video_audio` 16,238 · `text_audio` 1,584 · `text` 382.

**16. HBA test is overwhelmingly emotion/sentiment, not mental health.**
The four clinically-oriented sources — mmpsy_depression (191), mmpsy_anxiety
(191), ptsd_in_the_wild (64), daicwoz (47) — total 493 rows, **2.7 % of the
split**. The other 97 % is MOSEI/MELD/CREMA-D-style affect data. Worth knowing
before treating HBA test as a mental-health benchmark.

**17. HBA's `daicwoz` portion is exactly 47 rows — the same count as the
DAIC-WOZ test split we hold raw.** So the HBA-unified DAIC data is the DAIC
test split, and it overlaps with `data/raw/daic_woz`. The two copies differ in
preprocessing (HBA embeds media and a unified prompt schema; ours is raw
audio + transcript).

## Benchmark QA structure

- **IntentBench** — 2,689 items. `problem_type`: multiple choice 2,356 / judge
  200 / emer_ov_mc 133. `source`: Socialiq 2,356 / MDPE 200 / EMER 133.
  `Type`: why 698 / how 632 / other 507 / what 480 / deception 200 /
  emotion 133 / who-which 25 / when 14.
- **Daily-Omni** — 1,197 4-way MCQ. Answer distribution B 318 / C 306 / D 295 /
  A 278. `Type`: Event Sequence 306 / AV Event Alignment 238 / Context 193 /
  Reasoning 175 / Inference 154 / Comparative 131. Durations 30 s (647), 60 s (550).
- **WorldSense** — 1,662 video entries keyed by `video_id`, each with
  `duration, domain, sub_category, audio_class, video_caption`. Subtitles ship
  separately in `worldsense_subtitles.zip`.

Video archives are still packed (`videos.zip`, `Videos.tar`,
`worldsense_videos_*.zip`) — extract when needed; unpacking all of them roughly
doubles the footprint.

---

## Not deployed (deliberately)

HBA full corpus (448 GB), MER2025 (442 GB), IntentTrain / AVQA-R1 / OmniInstruct
(training data), HumanOmni-7B (needs repo-specific env + FlashAttention),
Menta, OSim-8B, AffectGPT, Emotion-LLaMA v1/v2, SCD-MLLM (no checkpoint).

---

# Wave 4 — gated encoders (2026-08-09)

Terms accepted on the HF website; both downloaded and smoke-tested.

| Model | Arch | Params | Size |
|---|---|---|---|
| MentalBERT (`mental/mental-bert-base-uncased`) | bert | 110M | 0.44 GB |
| MentalRoBERTa (`mental/mental-roberta-base`) | roberta | 125M | 0.50 GB |

Domain adaptation is clearly present — masked-fill on *"I have been feeling
very \<mask\> lately and cannot sleep"* returns `tired, depressed, sick,
lonely, down, anxious, exhausted, stressed` (BERT) and `depressed, down,
tired, sick, anxious, ill, suicidal, lonely` (RoBERTa).

**18. Do not use `pooler_output` from either encoder.** Both are MLM
checkpoints with no pooler in the weights, so `pooler.dense.{weight,bias}` is
randomly initialised at load time and the load report flags it MISSING. Use
`last_hidden_state[:, 0]` or mean pooling instead. These are encoders, not
chat models — `AutoModelForMaskedLM` or `AutoModel`, never `generate()`.

---

## Access still outstanding

- **Obtained ✅:** DAIC-WOZ, MentalBERT, MentalRoBERTa.
- **HF gated, not requested:** `MERChallenge/MER2025` (442 GB, only needed for
  AffectGPT).
- **Needs a written request:** **CLP** (CLPsych 2015, formal DUA) — the only
  IMHI test set genuinely missing, see gotcha 10.
- **Not yet sourced:** Menta's depression-severity corpus and its CSSRS
  suicide-risk corpus. Only community mirrors were found on HF
  (`av9ash/CSSR-S_labelled_suicidewatch_posts_reddit`); provenance unverified,
  so nothing was downloaded.
