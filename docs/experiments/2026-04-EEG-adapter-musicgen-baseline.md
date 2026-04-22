# 实验沉淀：EEG v0.1 → Adapter → MusicGen 基线（2026-04）

本文整理截至 2026-04 已完成事项、结论与建议的下一步，便于交接与开题/汇报引用。

---

## 1. 目标（本阶段）

- 与协作者对齐 **EEG 条件向量** 的机读契约（不依赖 EMOPIA clip 名）。
- 在同学侧 SNN 未就绪时，用 **公开 DEAP** 跑通 **特征 → Adapter → 生成**。
- 将 **MusicGen 基座、T5 监督目标、生成脚本** 对齐，避免「训练老师与推理解码器不一致」的歧义。

---

## 2. 已完成事项

### 2.1 数据契约与脚本

| 内容 | 位置 |
|------|------|
| 规范 v0.1 | `docs/contracts/eeg_feature_export_v0.1.md`（+ JSON Schema） |
| ADR | `docs/decisions/0002-eeg-feature-export-contract-v0.1.md` |
| Pydantic 校验 / 合成 / 校验 CLI | `scripts/eeg_export/`（`record_v0_1.py`、`make_synthetic_export.py`、`validate_pt.py`） |
| DEAP → v0.1 基线导出 | `scripts/eeg_public/deap_to_v01.py`、`README_DEAP.md`（通道时间均值+方差 → 80 维；V/A 四象限 → `emotion_id`） |

### 2.2 Adapter 训练与推理（`adapter_strategy_v1/`）

| 内容 | 说明 |
|------|------|
| `eeg_v01_loader.py` | 加载并校验 v0.1 `.pt` |
| `train_adapter.py` | `--eeg-v01`；`input_dim` 随特征维；checkpoint 存 `input_dim` |
| `prepare_text_targets.py` | `--model-dir`：从**同一套** MusicGen 取 T5 hidden 作监督 |
| `generate_adapter.py` | `--from-eeg-v01`、`--model-dir`、`--test-text`、采样/贪心、`--guidance-scale`、按 config 写 WAV 采样率 |

### 2.3 已跑通的数据与训练

- DEAP 预处理 Python 已置于数据盘；导出 **1280 条**（32×40）、**80 维** `feature`。
- 已用 `prepare_text_targets`（含 `res/musicgen_finetuned`）与 DEAP 特征训练 Adapter（多 epoch，MSE 可显著下降）。

---

## 3. 结论与踩坑（重要）

### 3.1 本地 `musicgen_finetuned` vs `facebook/musicgen-small`

- 在 **纯文本** `generate`（`--test-text`）下，曾出现「极短/噪」等异常听感。
- **官方 `facebook/musicgen-small`** 在修正 **写盘与解码参数** 后，**纯文本生成正常**（见下）。
- **原因归纳**（工程侧）：  
  - 文本条件生成宜使用文档常见 **`guidance_scale=3.0`**，不宜长期误用 `1.0` 作为「唯一默认」去判断基座好坏。  
  - WAV **采样率**须用 `model.config.audio_encoder.sampling_rate`，不可硬编码 32000。  
  - 导出波形与 `diagnose.py` 一致，使用 **`audio[0, 0]`**（左声道），并对 int16 做 **clip**。
- **与 `src/train_e2e.py` 的关系**：微调时 **T5/text_encoder 冻结**，主要动解码相关部分；**通用文本生成**仍可能因微调而变差，需用 **`--test-text` + small** 单独验基线。

### 3.2 DEAP + Adapter 听感

- 训练 MSE 低 **不等价** 于听感好；**贪心解码**易使多条样本 **逐 token 相同**（不是脚本单条 bug）。
- 条件路径上 **`guidance_scale`** 默认 **1.0**（与旧「直接条件注入」习惯一致）；与 **纯文本 3.0** 分工不同，需在文档和命令行中区分。

---

## 4. 建议的下一步（按优先级）

1. **统一「老师 + 解码器」**（择一主线）  
   - **方案 A（推荐用于可复现听感）**：`prepare_text_targets.py --model-dir facebook/musicgen-small`，`generate_adapter.py --model-dir facebook/musicgen-small`，**重训** Adapter（DEAP 或合成），再听 DEAP 条件生成。  
   - **方案 B**：坚持本地 `musicgen_finetuned`，则 `prepare_text_targets` / `train` / `generate` 均应对齐该目录；并接受其**纯文本基线**可能弱于 small。

2. **同学侧真实 v0.1 `.pt`**：小样本 `validate_pt.py` → `train_adapter` → `generate`；标签与 `emotion_id` 规则书面约定。

3. **实验规范**：划 **train/val**（如按被试），避免 Adapter 仅记训练集；记录 `guidance_scale`、采样率、`max_new_tokens`。

4. **可选增强**：DEAP 特征从统计量换 **频带/深度学习编码**；契约维 **D** 不变即可。

---

## 5. 相关命令速查

```bash
# DEAP → v0.1
python scripts/eeg_public/deap_to_v01.py \
  --deap-dir /path/to/data_preprocessed_python \
  --out res/continuous_features/deap_v01_export.pt

# T5 监督（与最终 generate 用同一 --model-dir）
python adapter_strategy_v1/prepare_text_targets.py --model-dir facebook/musicgen-small

# 训练
cd adapter_strategy_v1
python train_adapter.py --eeg-v01 ../res/continuous_features/deap_v01_export.pt --epochs 50

# 验官方 small 纯文本（应正常）
python generate_adapter.py --model-dir facebook/musicgen-small \
  --test-text "happy energetic upbeat music" --max-new-tokens 1000

# DEAP 条件生成
python generate_adapter.py --model-dir facebook/musicgen-small \
  --eeg-v01 ../res/continuous_features/deap_v01_export.pt --eeg-indices 0,1,2,3
```

---

## 6. 引用文件

- 协作契约：`docs/contracts/README.md`
- 周志可补一条指向本页：`docs/logs/`
- 旧 finetuned 相关产物可拷至：`archive/adapter_runs/2026-04-finetuned/`（见该目录 `README.md`）

## 7. 基座体量：small 够了还是要 medium / large？

- **方法验证、接 EEG、尽快迭代**：**small 通常够用**（省显存、快，论文里说明「在 MusicGen-small 上验证管线」完全成立）。
- **更在意听感、演示/被试听**：在 **同一条数据与 Adapter 参数量** 下，**medium** 往往是性价比最高的升级；**large** 音质上限更高，但 **推理与显存** 明显更重，且你们监督仍是 T5 级条件，**瓶颈常在条件与数据量**，不一定与模型体量同比例提升。
- **实践建议**：主线先 **锁 small** 到论文/组会一版可复现结果；若评审或合作方只卡音质，**再开对照实验**（同 DEAP 特征、同 Adapter 结构、只换 `facebook/musicgen-medium`）汇报增益即可，不必一开始上大模型。
