# Foundation Models for Mental Health / Psychological / Social-Behavior Understanding
**Deployment-oriented resource sheet**  
**Last verified:** 2026-08-09

> Purpose: hand this file to Claude / another coding agent for local reproduction or deployment.
>
> Important:
> - **Training data** and **evaluation-only benchmarks** are distinguished where the project makes that distinction.
> - “Not found” means I did not locate an **official public** repo/checkpoint at the time of verification.
> - Dataset/model licenses may differ from code licenses. Check each upstream license before redistribution or commercial use.
> - Mental-health models below are research models, not clinical diagnostic systems.

## Summary sheet

| Model / project | Main scope & modalities | Official GitHub | Training dataset usage | Dataset access | Trained model / checkpoint | Model access | Local deployment notes | Access / license caveat |
|---|---|---|---|---|---|---|---|---|
| **OmniSapiens / Human Behavior Atlas (HBA) / OmniSapiens 2.0** | Unified human-behavior understanding; **text + audio + video**; emotion, sentiment, humor/sarcasm, social reasoning, depression, anxiety, PTSD, etc. | https://github.com/MIT-MI/human_behavior_atlas | **16 HBA sources:** CMU-MOSEI Sentiment; IntentQA; MELD Sentiment; MELD Emotion; CMU-MOSEI Emotion; CREMA-D; Social-IQ 2.0; CH-SIMS v2; TESS; UR-Funny; MMPsy-Depression; MMPsy-Anxiety; MIMEQA; MMSD; PTSD-in-the-Wild; DAIC-WOZ. HBA release has ~100,299 unified samples. | Unified HBA: https://huggingface.co/datasets/HumanBehaviorAtlas/human_behavior_atlas | `OmniSapiens-7B-RL`, `omnisapiens_sft`, BAM adapters, **OmniSapiens2.0** | Recommended: https://huggingface.co/HumanBehaviorAtlas/OmniSapiens2.0  ; older releases are linked from GitHub | Best starting point for reproducing a **unified human-behavior model** because code + unified data + SFT/RL pipeline + weights are public. Use the repo’s inference code/config rather than assuming generic text-only `AutoModel`. | HBA dataset is **CC BY-NC 4.0**. Check licenses of original constituent datasets too. |
| **HumanOmni** | Human-centric omni-modal understanding; **video + audio + text**; emotion, facial behavior, speaker-aware ASR, human-centric reasoning | https://github.com/HumanMLLM/HumanOmni | Authors report **2.4M human-centric video clips + 14M instructions**, plus **50K clips / 100K+ manually annotated instructions** for emotion recognition, facial description and speaker-specific ASR. GitHub does **not** enumerate every raw source dataset. `DFEW` appears in example training configs; DFEW/MAFW are also used for evaluation. | No single complete raw-training-corpus release located. Follow dataset/config references in repo. | `HumanOmni-7B-Video`, `HumanOmni-7B-Audio`, final **HumanOmni-7B** | Final checkpoint: https://huggingface.co/StarJiaxing/HumanOmni-7B | Custom multimodal repo. README environment is roughly Python >=3.10, CUDA >=12.1, PyTorch >=2.2, Transformers >=4.45, FlashAttention. Use repository `inference.py`. | Training corpus provenance is less transparent than HBA; do not assume DFEW/MAFW constitute the full training set. |
| **HumanOmniV2** | Omni-modal **human intention/emotion/social reasoning**; video + audio + text; explicitly addresses modality/context shortcut | https://github.com/HumanMLLM/HumanOmniV2 | Cold-start / RL data includes **IntentTrain**, **Video-R1**, **AVQA-R1-6K**, and **OmniInstruct**. Training is cold-start SFT -> GRPO Stage 1 -> GRPO Stage 2. | IntentTrain: https://huggingface.co/datasets/PhilipC/IntentTrain ; Video-R1: https://github.com/tulerfeng/Video-R1 ; AVQA-R1-6K: https://huggingface.co/datasets/harryhsing/AVQA-R1-6K ; OmniInstruct: https://huggingface.co/datasets/m-a-p/OmniInstruct | **HumanOmniV2** (~9B multimodal model) | https://huggingface.co/PhilipC/HumanOmniV2 | Use official repo because inference requires the omni-modal preprocessing/model stack. Good model to inspect for `<context> -> <think> -> <answer>` and context/logical-reward training. | Model checkpoint is Apache-2.0. `IntentTrain` is **CC BY-NC-ND 4.0**; upstream datasets have their own terms. |
| **MentaLLaMA** | Unified **text-only mental-health instruction model**; depression, stress, suicidality, wellness, interpersonal risk, mental disorders | https://github.com/SteveKGYang/MentaLLaMA | IMHI integrates **10 source datasets:** DR, CLP, Dreaddit, SWMH, T-SID, SAD, CAMS, Loneliness, MultiWD, IRF. Project reports ~105K instruction samples overall. GitHub directly releases only subsets where redistribution is permitted. | Start from repo data instructions: https://github.com/SteveKGYang/MentaLLaMA | Multiple releases: 7B chat, 13B, 33B-LoRA, MentalBART, MentalT5. Easiest local choice: **MentaLLaMA-chat-7B-hf** | https://huggingface.co/klyang/MentaLLaMA-chat-7B-hf | Standard HF causal LM. Good candidate for `transformers` / `vLLM`. The HF model can be served locally much more easily than most multimodal models here. | Research-oriented mental-health model. Some IMHI source datasets cannot be redistributed freely. Check per-dataset terms. |
| **Menta** | Small local **text-only mental-health model**; multi-task Reddit mental-health prediction | https://github.com/hong-labs/Menta | Four Reddit-derived corpora organized into six tasks: **Dreaddit** (stress), a **depression corpus** (binary + severity), **SDCNL** (suicidal ideation), and a **C-SSRS/CSSRS-based suicide-risk corpus** (binary + severity). Repo/project wording differs slightly across releases. | Project: https://hong-labs.github.io/menta-project/ ; training code/data instructions: https://github.com/hong-labs/Menta | **Menta** based on the Qwen3-4B family; HF Transformers weights + GGUF | https://huggingface.co/mHealthAI/Menta | **Easiest offline/local deployment in this list.** GGUF supports llama.cpp / llama-cpp-python; HF weights support Transformers. Suitable for quick local proof-of-concept. | Model repo is Apache-2.0; underlying Reddit datasets have separate usage/privacy constraints. |
| **MentalBERT** | Mental-health domain-adapted **BERT masked LM**, text only | No official paper repo located | Domain-adaptive pretraining on **mental-health-related Reddit text**. The public model card does not provide a complete downloadable raw-pretraining-corpus manifest. | No consolidated official raw corpus located from the model release. | `mental-bert-base-uncased` | https://huggingface.co/mental/mental-bert-base-uncased | Easy to load with Transformers, but this is **not a generative/chat model**. Useful as encoder/baseline rather than a direct conversational deployment. | HF model is **gated**; login/acceptance required. License listed as **CC BY-NC 4.0**. |
| **MentalRoBERTa** | Mental-health domain-adapted **RoBERTa masked LM**, text only | No official paper repo located | Same family: domain-adaptive pretraining on **mental-health-related Reddit text**; complete raw corpus manifest is not provided with the public model card. | No consolidated official raw corpus located from the model release. | `mental-roberta-base` | https://huggingface.co/mental/mental-roberta-base | Standard Transformers encoder; not a generative chat model. | HF model is **gated**; license listed as **CC BY-NC 4.0**. |
| **SCD-MLLM** | Cross-dataset multimodal **depression** modeling; audio/video/text-style multimodal learning | **No official GitHub found** | Paper uses **CMDC, AVEC2014, DAIC-WOZ, DVlog, EATD** across heterogeneous depression datasets. | No official unified training-data release located. Access must be handled through the original dataset owners. | **No official trained checkpoint found** | Paper: https://arxiv.org/abs/2512.06447 | Cannot directly deploy/reproduce from released weights. Would require reimplementation from paper + access to the five source datasets. | Treat dataset access as potentially restricted; DAIC-WOZ in particular requires its own access process. |
| **Emotion-LLaMA** | Multimodal emotion understanding/reasoning; **video + audio + text** | https://github.com/ZebangCheng/Emotion-LLaMA | Core training resources are **MER2023** video data plus **MERR** annotations. MERR reports ~28,618 coarse-grained and ~4,487 fine-grained annotations. | MER dataset/application: http://merchallenge.cn/datasets ; annotations/resources are linked in GitHub | `Emotion_LLaMA.pth` / project checkpoint | Official Google Drive checkpoint: https://drive.google.com/file/d/1pNngqXdc3cKr9uLNW-Hu3SKvOpjzfzGY/view?usp=sharing | More manual than a standard HF model. Requires project-specific encoders/components (e.g. Llama-2, HuBERT/vision modules) and repo inference pipeline. | MER raw videos are not simply freely redistributable; follow MER challenge EULA/application. Code is BSD-3-Clause. |
| **Emotion-LLaMA v2 / MMEVerse** | Unified tri-modal emotion recognition + reasoning; **audio + video + text** | https://github.com/ooochen-30/Emotion-LLaMA-v2 | Uses **MMEVerse**, a unified multi-dataset emotion corpus. The paper/repo consolidate multiple public emotion datasets into a common training/evaluation framework; use the HF dataset configs for the exact training split rather than inferring all benchmark names as training data. | https://huggingface.co/datasets/ooochen/MMEVerse | Stage-1 / Stage-2 **Emotion-LLaMA-v2** checkpoints | https://huggingface.co/ooochen/Emotion-LLaMA-v2 | Use official repo `inference.py` / app path. Requires separate multimodal encoders and model components; not just a single text `AutoModel` call. | Code BSD-3-Clause. Some underlying emotion datasets (including MER-family data) impose research/EULA restrictions. |
| **AffectGPT** | Unified multimodal affect/emotion recognition + description/reasoning | https://github.com/zeroQiaoba/AffectGPT | Main released AffectGPT checkpoint is trained on **MER-Caption+**, with ~132,171 training videos reported by the repo. Do **not** confuse MER-UniBench’s 9 evaluation datasets with the training set. | MER2025 / MER-Caption resources: https://huggingface.co/datasets/MERChallenge/MER2025 | **AffectGPT** pretrained checkpoint | https://huggingface.co/MERChallenge/AffectGPT | Use repo inference pipeline (e.g. `inference_sample.py`). Requires audio/video/subtitle/face preprocessing plus the project’s multimodal encoders. | Repo describes the model as a **research preview / non-commercial use**; MER2025 data is gated and subject to academic/EULA restrictions. |
| **OdysSim / OSim** | **Behavioral foundation model** for human/user behavior simulation and reasoning; primarily text, with broader collection variants | https://github.com/sunnweiwei/OdysSim | Mid-training corpus: **62 behavioral datasets**, ~21.2M interactions/rows and ~10.6B tokens, organized over SOUL dimensions (conversation, social situations, cognition, roles, evaluation). Post-training corpus adds task-specific RL/evaluation data. | Mid-training: https://huggingface.co/datasets/cmu-lti/osim-mid-training ; Post-training: https://huggingface.co/datasets/cmu-lti/osim-post-training | Recommended main checkpoint: **OSim-8B**; collection also contains mid-trained/other-size variants | OSim-8B: https://huggingface.co/cmu-lti/osim-8b ; collection: https://huggingface.co/collections/cmu-lti/odyssim | Standard HF-style text model is relatively straightforward to deploy with Transformers. Useful as a **behavioral foundation-model** reference, but its goal is behavior simulation rather than clinical classification. | Model is MIT; mid-training data aggregates many upstream datasets, so upstream licenses vary and must be checked. |

---

## Training vs evaluation datasets — do not mix these up

### HumanOmniV2
**Training / RL resources**
- IntentTrain: https://huggingface.co/datasets/PhilipC/IntentTrain
- Video-R1: https://github.com/tulerfeng/Video-R1
- AVQA-R1-6K: https://huggingface.co/datasets/harryhsing/AVQA-R1-6K
- OmniInstruct: https://huggingface.co/datasets/m-a-p/OmniInstruct

**Evaluation benchmarks**
- IntentBench
- Daily-Omni
- WorldSense

### AffectGPT
**Main checkpoint training**
- MER-Caption+

**MER-UniBench evaluation family**
- MER2023
- MER2024
- SIMS
- SIMS v2
- CMU-MOSI
- CMU-MOSEI
- IEMOCAP
- MELD
- OV-MERD+

Do **not** automatically treat all MER-UniBench datasets as AffectGPT training data.

### HumanOmni
The repo reports a large internally curated / processed corpus (2.4M clips, 14M instructions) but does not provide a clean one-to-one manifest of all raw source datasets.  
Do **not** infer that DFEW/MAFW are the complete training corpus merely because they appear in configs/evaluation.

---

## Recommended deployment order

### Tier 1 — easiest to get running locally

1. **Menta**
   - Model: https://huggingface.co/mHealthAI/Menta
   - Why: GGUF + llama.cpp support; ~4B-scale.
   - Best first smoke test on a laptop/workstation.

2. **MentaLLaMA-7B**
   - Model: https://huggingface.co/klyang/MentaLLaMA-chat-7B-hf
   - Why: standard causal-LM workflow; good for `transformers` or `vLLM`.

3. **OSim-8B**
   - Model: https://huggingface.co/cmu-lti/osim-8b
   - Why: standard Hugging Face-style model; useful behavioral foundation-model baseline.

### Tier 2 — most relevant unified multimodal models, but require project code

4. **OmniSapiens 2.0**
   - Repo: https://github.com/MIT-MI/human_behavior_atlas
   - Model: https://huggingface.co/HumanBehaviorAtlas/OmniSapiens2.0
   - Dataset: https://huggingface.co/datasets/HumanBehaviorAtlas/human_behavior_atlas
   - Strongest first choice if the goal is to inspect/reproduce a unified psychological/social-behavior training pipeline.

5. **HumanOmniV2**
   - Repo: https://github.com/HumanMLLM/HumanOmniV2
   - Model: https://huggingface.co/PhilipC/HumanOmniV2
   - Strong choice for studying context-grounded multimodal reasoning / shortcut mitigation.

6. **HumanOmni**
   - Repo: https://github.com/HumanMLLM/HumanOmni
   - Model: https://huggingface.co/StarJiaxing/HumanOmni-7B

### Tier 3 — multimodal affect models with more custom dependencies

7. **Emotion-LLaMA v2**
8. **AffectGPT**
9. **Emotion-LLaMA**

### Not directly deployable from an official trained release

10. **SCD-MLLM**
   - No official repo/checkpoint located as of 2026-08-09.

### Encoder baselines rather than generative deployment targets

11. **MentalBERT**
12. **MentalRoBERTa**

---

## Suggested instructions to give Claude

You can paste the following after this file:

```text
Read this resource sheet first.

Goal:
1. Verify that the repository and model checkpoint for the target model are still accessible.
2. Clone ONLY the official repository listed here.
3. Read the repository README, requirements/environment files, and inference scripts before installing anything.
4. Do not retrain the model. First deploy the released trained checkpoint for inference.
5. Create a fresh conda environment for each model unless two repos explicitly use compatible dependency versions.
6. Before downloading large checkpoints/datasets, report:
   - expected model download size,
   - required GPU VRAM / CPU RAM if documented or inferable,
   - CUDA / PyTorch / Transformers versions,
   - whether FlashAttention, ffmpeg, decord, or other system packages are needed,
   - whether Hugging Face authentication or gated-data approval is required.
7. Start with one minimal inference example from the official repo.
8. Keep a DEPLOYMENT_NOTES.md containing every command, dependency pin, error, fix, and final working command.
9. Do not treat evaluation-only datasets as training data.
10. Do not attempt to bypass gated or licensed dataset access.
```

---

## Useful generic download commands

### Hugging Face model snapshot

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download <ORG/MODEL> \
  --local-dir ./checkpoints/<MODEL_NAME>
```

For gated models:

```bash
huggingface-cli login
```

### Clone a GitHub repository

```bash
git clone <OFFICIAL_GITHUB_URL>
cd <REPO_NAME>
```

### Clone a Hugging Face dataset with Git LFS when the repo requires it

```bash
git lfs install
git clone https://huggingface.co/datasets/<ORG/DATASET>
```

Always follow the model repository's own download/preprocessing instructions when they conflict with these generic commands.

---

## If the research goal is a new unified mental-health foundation model

The most useful code/data references to inspect first are:

1. **HBA / OmniSapiens**
   - heterogeneous datasets -> unified schema -> joint multimodal training
2. **MentaLLaMA**
   - heterogeneous mental-health text datasets -> unified instruction schema
3. **HumanOmniV2**
   - multimodal grounding + anti-shortcut/context-reward training
4. **Emotion-LLaMA v2 / MMEVerse**
   - heterogeneous multimodal affect datasets -> unified dataset/instruction pipeline
5. **AffectGPT**
   - convert affect labels toward richer natural-language emotion descriptions/reasoning
6. **OdysSim**
   - very large-scale aggregation of heterogeneous behavioral datasets

This ordering is for **code/data design inspection**, not a claim that all six solve the same task.
