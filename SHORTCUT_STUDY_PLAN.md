> **SUPERSEDED (2026-08-10).** Early self-directed draft; the authoritative
> design is `EXPERIMENT_DESIGN.md`, which formalizes the PI-specified A-D
> experiments. Kept only for history.

# 本地部署計畫：研究 mental-health / behavior foundation models 的 shortcut learning

**目標**：不是「把 12 個模型都跑起來」，而是建立一套**跨模型可比較的 shortcut 診斷 harness**，並只部署能回答特定 shortcut 問題的模型。

**上游資料來源**：`foundation_models_mental_health_behavior_deployment.md`（該表註記 last verified 2026-08-09；所有 repo / checkpoint 連結在 clone 當下需重新驗證，不預設仍然可用）。

---

## Part 0. 先確認的前提（不阻塞，先給預設假設）

| 項目 | 預設假設 | 影響 |
|---|---|---|
| GPU | 至少 1× 24GB；若有 1× 80GB 則走 A 路徑 | 決定 7B/9B 模型是否需要量化 |
| 磁碟 | `/data/cl195` 有 ≥ 500GB 可用 | 多模態資料集是主要消耗 |
| HF token | 已有，但未申請 gated 資料 | MentalBERT/RoBERTa、MER2025 需接受條款 |
| DAIC-WOZ / MER 授權 | **尚未取得** | 決定 Phase 3 能否做 |

**路徑分支**
- **A 路徑（80GB VRAM）**：全部模型 bf16 原生載入，不量化。
- **B 路徑（24GB VRAM）**：文字模型走 4-bit（bitsandbytes NF4）或 vLLM AWQ；多模態模型只跑 batch=1 + CPU offload，**接受吞吐量低**，因為 shortcut 實驗是小樣本高重複，不需要高吞吐。

---

## Part 1. 先把研究問題釘死：要測哪幾類 shortcut

這一步決定後面所有部署。針對這個領域，有 6 類 shortcut 值得測，優先序如下：

### S1. Partial-input / 詞彙捷徑（文字）★最高 CP 值
Reddit 衍生資料集（Dreaddit、SDCNL、CSSRS、DR、CLP、SWMH）的標籤常常由**子版特有詞彙**決定，而非心理狀態本身。
- 探針：只給 input 的一部分、遮蔽 class-indicative tokens、換詞、否定。
- **關鍵對照組**：TF-IDF + logistic regression。若線性模型逼近 FM，該 benchmark 本身就是 shortcut-solvable，FM 的分數沒有意義。

### S2. Modality shortcut（多模態）★最有論文價值
多模態模型是否只用字幕/文字，audio/video 形同裝飾。
- 探針：單模態消融、跨樣本模態錯配（把 A 的 audio 配 B 的 video）、時序打亂、模態靜音。
- HumanOmniV2 明確宣稱處理 modality shortcut → **天然的正/負對照**：拿它 vs HumanOmni（前代，無此機制）比較，是最乾淨的實驗設計。

### S3. Label prior / majority-class 捷徑
在不平衡資料（DAIC-WOZ 憂鬱、CSSRS 風險等級）上，模型是否退化成常數預測。
- 探針：per-class recall、confusion matrix、以及在 label 分布反轉的重採樣子集上的表現。

### S4. Dataset-provenance 捷徑（統一語料特有）★這是 HBA / IMHI / MMEVerse 獨有的問題
HBA 把 16 個來源統一成一個 schema，IMHI 統一 10 個，MMEVerse 統一多個。若每個來源的 instruction 模板/格式可辨識，且各來源 label 分布不同，模型可以「先認出資料集，再猜該資料集的先驗」。
- 探針：(a) 讓模型直接預測樣本來自哪個資料集；(b) 在 hidden states 上訓 source-dataset linear probe；(c) 把模板正規化成統一格式後，看準確率掉多少。

### S5. Train/eval 汙染
HBA 訓練集含 CMU-MOSEI、MELD，而這兩者同時是常用評測基準；AffectGPT 的 MER-UniBench 也有此風險（原文件已明確警告不要混用）。
- 探針：記憶化測試（給前半段，看能否逐字續出後半段標註）、train-member vs non-member 的信心差。

### S6. Prompt-format / option-position 捷徑
MCQ 選項順序、答案分布、`<think>` 模板是否被當成線索。
- 探針：選項洗牌下的預測一致性、空白 context 下的答案分布。

---

## Part 2. 模型選型（只部署 5 個，不是 12 個）

| 階段 | 模型 | 回答哪個問題 | 為什麼選它 |
|---|---|---|---|
| P1 | **Menta** (Qwen3-4B) | S1, S3, S6 | 最輕；GGUF/HF 雙格式；6 個 Reddit 任務正好是詞彙捷徑重災區 |
| P1 | **MentaLLaMA-chat-7B** | S1, S3, S4, S6 | 標準 causal LM；IMHI 統一 10 個來源 → provenance 捷徑的文字版測試床 |
| P1 | **MentalRoBERTa**（+ MentalBERT） | S1 對照 | encoder baseline；配合 TF-IDF 形成「捷徑天花板」三層對照 |
| P2 | **HumanOmniV2** | S2, S6 | 宣稱有 anti-shortcut context reward → 正對照 |
| P2 | **HumanOmni-7B** | S2 | 同家族前代、無 anti-shortcut 機制 → 負對照 |
| P3（可選） | **OmniSapiens2.0** | S4, S5 | 16 來源統一 + 資料公開，是 provenance/汙染研究的唯一完整標的 |

**明確不做**：SCD-MLLM（無 checkpoint）、Emotion-LLaMA v1/v2、AffectGPT（依賴重、資料 gated、且 S2 已由 HumanOmni 家族回答）。OSim-8B 是行為模擬而非分類，與 shortcut 分類實驗不同框架，除非要做 S4 的大規模語料版本，否則延後。

---

## Part 3. 目錄與環境規劃

```
/data/cl195/NeurIPS_workshop/
├── foundation_models_mental_health_behavior_deployment.md
├── SHORTCUT_STUDY_PLAN.md          # 本檔
├── DEPLOYMENT_NOTES.md             # 每個指令/錯誤/修法/最終可用指令，逐條記錄
├── envs/                           # conda env 匯出的 yml（每模型一份，可重現）
├── repos/                          # 官方 repo clone，不改動上游程式碼
├── checkpoints/                    # HF snapshot
├── data/
│   ├── raw/                        # 原始下載
│   ├── unified/                    # 轉成 harness 統一 schema 後
│   └── perturbed/                  # 擾動後的變體（可重現：存 seed + 生成腳本）
└── shortcut_harness/               # 自寫程式碼，唯一需要維護的部分
    ├── adapters/                   # 每模型一個，實作統一介面
    ├── perturb/                    # 擾動運算子
    ├── scoring/                    # label log-likelihood 打分
    ├── metrics/                    # shortcut 指標
    └── runs/                       # 結果 JSONL + 分析 notebook
```

**環境原則**：每個模型一個獨立 conda env（`sc-menta`、`sc-mentallama`、`sc-humanomni`、`sc-humanomniv2`）。harness 本身另開一個純 CPU env（`sc-harness`），只依賴 pandas/numpy/sklearn/scipy，**不依賴任何模型框架**——這樣分析程式碼永遠不會被 torch 版本衝突弄壞。模型端與 harness 端透過**磁碟上的 JSONL** 溝通，不共用 Python 進程。

---

## Part 4. 統一介面設計（先寫這個，再下載模型）

### 4.1 統一樣本 schema
```json
{
  "uid": "dreaddit/test/00123",
  "source_dataset": "dreaddit",
  "task": "stress_binary",
  "modalities": {"text": "...", "audio": "path/or/null", "video": "path/or/null"},
  "label": "stress",
  "label_space": ["stress", "no_stress"],
  "meta": {"split": "test", "subgroup": {...}}
}
```

### 4.2 Adapter 介面（每個模型只需實作這一個）
```python
class ModelAdapter:
    def score(self, sample, label_space) -> dict[str, float]:
        """回傳每個候選標籤的 log-probability。不是自由生成。"""
```

**這一點是整個計畫的技術核心**：shortcut 研究需要的是**校準過的機率**，不是生成文字。所以一律用 label log-likelihood 打分（teacher-forcing 每個候選標籤字串，取平均 log-prob），而不是解析模型的自由輸出。理由：
- 自由生成的解析失敗率會與擾動強度相關，造成假的「效能下降」；
- 有機率才能算 ECE、信心差、以及擾動下的 KL 散度；
- 對 MCQ 模型可直接同時測 S6（選項洗牌後機率是否穩定）。

另外務必加上 **PMI 正規化**（`log P(label|prompt) − log P(label|空 prompt)`）。未正規化與正規化的差值本身就是 S3 label-prior 捷徑的直接量測。

### 4.3 輸出 JSONL（每筆一列）
```json
{"uid":"...","model":"menta","condition":"full|text_only|masked_cues|shuffled_video|...",
 "logprobs":{"stress":-0.31,"no_stress":-1.42},"pred":"stress","correct":true,"seed":0}
```
所有分析都只讀這個檔。

---

## Part 5. 分階段執行

### Phase 0：零 GPU 的前置實驗（1–2 天，先做，價值最高）
1. 取得可公開下載的文字資料集：Dreaddit、SDCNL、MentaLLaMA repo 允許重分發的 IMHI 子集。**不碰 gated 資料。**
2. 轉成 §4.1 統一 schema。
3. 跑三個 baseline：majority-class、TF-IDF+LogReg、bag-of-words 只留停用詞（極端 partial-input）。
4. 用 PMI 算出每個 class 的 top-k indicative tokens → 這份清單就是後面 `masked_cues` 擾動的輸入。
5. **決策點**：若 LogReg 在某資料集上已接近文獻報告的 FM 分數，該資料集直接標記為「shortcut-saturated」，後續作為主要證據而非陪襯。

### Phase 1：文字模型部署（Menta → MentaLLaMA → MentalRoBERTa）
每個模型的固定流程（寫進 DEPLOYMENT_NOTES.md）：
1. 先驗證連結可達（`huggingface-cli download --dry-run` 或 HTTP HEAD），**下載前先報告**：模型大小、VRAM 需求、CUDA/torch/transformers 版本、是否需 gated 授權。
2. 建 conda env → 讀官方 README/requirements → 安裝 → 匯出 yml 到 `envs/`。
3. 官方最小 inference 範例跑通（smoke test）。
4. **再**接上 adapter，驗證 `score()` 在 10 個樣本上與官方 pipeline 的預測一致。
5. 跑全套擾動條件。

預估下載量：Menta ~8GB（GGUF Q4 ~2.5GB）、MentaLLaMA-7B ~13.5GB、MentalRoBERTa ~0.5GB（gated，需先接受條款）。

### Phase 2：多模態 modality shortcut（HumanOmniV2 vs HumanOmni）
1. 兩者分開 env（README 要求約 Python≥3.10 / CUDA≥12.1 / torch≥2.2 / transformers≥4.45 / FlashAttention；另需 ffmpeg、decord）。**FlashAttention 編譯是這一階段最常見的卡點**，B 路徑可先關掉 FA 用 sdpa 跑通再說。
2. 評測資料選**非訓練集**的公開多模態集（避免 S5 污染混淆 S2 結論）。
3. 擾動運算子（見 §6）在**檔案層級**產生：預先生成錯配/打亂/靜音的媒體檔，存進 `data/perturbed/`，這樣兩個模型吃到完全相同的輸入。
4. 核心對比：`Δ = Acc(full) − Acc(modality_dropped)`，逐模態、逐模型畫出。若 HumanOmniV2 的 Δ_audio、Δ_video 顯著大於 HumanOmni，即為其 anti-shortcut 訓練有效的正面證據；若相近，則是很強的負面結果。

### Phase 3（可選）：OmniSapiens2.0 + HBA 的 provenance / 汙染研究
只有在 Phase 1–2 完成且磁碟充足時才啟動。HBA 統一集約 100K 樣本、含影音，**磁碟可能到數百 GB**，先只抓 metadata/文字欄位做 S4 的模板正規化實驗，確認有訊號再決定是否下載媒體。HBA 為 CC BY-NC 4.0，且其構成資料集各有授權，不做重分發。

---

## Part 6. 擾動運算子清單（`perturb/`）

| 運算子 | 對應 shortcut | 說明 |
|---|---|---|
| `text_only` / `audio_only` / `video_only` | S2 | 單模態保留 |
| `drop_modality(m)` | S2 | 移除單一模態 |
| `mismatch_modality(m)` | S2 | 換成另一樣本的同模態內容（保留「有輸入」但語意無關） |
| `shuffle_frames` | S2 | 時序打亂，測是否只用單幀外觀 |
| `mask_top_k_cues(k)` | S1 | 遮蔽 PMI top-k class-indicative tokens |
| `truncate(first_n / last_n)` | S1 | partial-input |
| `negate` / `antonym_swap` | S1 | 反轉語意，預測應翻轉 |
| `paraphrase` | S1 | 語意保留，預測應不變 |
| `identity_swap` | S1 + 公平性 | 換人稱/族群詞，預測應不變 |
| `normalize_template` | S4 | 洗掉來源特有的 instruction 格式 |
| `shuffle_options` | S6 | MCQ 選項位置 |
| `empty_context` | S3/S6 | 只給問題不給內容，測 label prior |
| `label_shuffle_control` | 全體 | 隨機標籤對照，用來校準所有指標的 noise floor |

每個運算子固定 seed，且輸出寫入 `data/perturbed/` 並附生成腳本雜湊，確保可重現。

---

## Part 7. 指標

1. **Shortcut Sufficiency**：`Acc(partial) − Acc(majority)`，正規化到 `Acc(full) − Acc(majority)`。接近 1 = 完整輸入沒帶來額外資訊。
2. **Modality Reliance** `Δ_m`：完整 vs 移除 m 的準確率差；另報 `mismatch` 版本（更嚴格）。
3. **Prediction Flip Rate**：擾動後預測改變的比例，分「應變（negate）」與「不應變（paraphrase/identity_swap）」兩類，前者低=麻木，後者高=脆弱。
4. **Prior Gap**：PMI 正規化前後準確率差 → S3 強度。
5. **Provenance Probe AUC**：hidden state 上的 source-dataset 線性探針 → S4 強度。
6. **Calibration**：ECE，以及擾動下的信心保持度（模型丟失資訊卻不降信心 = 危險）。
7. **Subgroup gap**：有 metadata 時的分組表現差距。

所有指標都要對 `label_shuffle_control` 報告 noise floor，並用 bootstrap CI，不報單點數字。

---

## Part 8. 已知風險與坑

- **授權**：DAIC-WOZ、MER 系列、MentalBERT/RoBERTa 均為 gated。不繞過任何授權流程；未取得則該實驗標記為 blocked 並在論文中說明。Reddit 衍生資料另有隱私條款，不重分發、不公開原始貼文內容。
- **汙染混淆**：測 S2 時若用到模型訓練過的資料集，Δ 會被記憶化污染。Phase 2 的評測集必須先對照原文件「training vs evaluation」章節逐一排除。
- **FlashAttention / decord 編譯**：多模態階段最常見的環境失敗點，先用 sdpa + torchvision decoder 跑通再優化。
- **不要改上游程式碼**：所有客製邏輯留在 `shortcut_harness/`，`repos/` 保持 pristine，便於報告可重現性。
- **不重訓**：全程只做 released checkpoint 的推論診斷。

---

## 建議的起手順序

```
Phase 0（無 GPU，先做）→ 建 harness 骨架 + adapter 介面
  → Phase 1 Menta（最輕，驗證 harness 正確）
  → Phase 1 MentaLLaMA + MentalRoBERTa
  → Phase 2 HumanOmni / HumanOmniV2 對照實驗（論文主結果）
  → Phase 3 OmniSapiens2.0（若時間與磁碟允許）
```
