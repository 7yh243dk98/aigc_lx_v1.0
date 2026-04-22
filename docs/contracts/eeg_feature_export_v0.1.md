# EEG 条件特征导出契约 v0.1

**目标**：与音乐侧对齐，使每个 EEG trial 或时间窗对应 **一条** conditioning 向量；**不依赖 EMOPIA clip 名称**。

---

## 1. 单条记录 schema（canonical）

每条记录为一个 `dict`，**key 均为 ASCII**。

### Required

| Key | 类型 | 说明 |
|-----|------|------|
| `dataset` | `str` | 语料名，如 `DEAP`、`SEED`、校内库代号 |
| `subject_id` | `str` 或 `int` | 被试标识 |
| `segment_id` | `str` | 在 **dataset 内唯一**（trial 或 window） |
| `feature` | `float32` 1D | 优先 `shape == (128,)`；否则 `shape == (D,)` 并填 `feature_dim` |

### Optional

| Key | 类型 | 说明 |
|-----|------|------|
| `trial_id` | `str` | 原始 trial 编号 |
| `label` | `str` | 如 `depression` / `control` |
| `window_start_ms` | `int` | 窗口起点（毫秒） |
| `window_end_ms` | `int` | 窗口终点（毫秒） |
| `snn_version` | `str` | 模型或导出脚本版本，便于复现 |
| `feature_dim` | `int` | **`D != 128` 时必填**；与 `feature.shape[0]` 一致 |
| `feature_dtype` | `str` | 固定记 `"float32"` |
| `feature_space` | `str` | 如 `penultimate` |
| `export_version` | `str` | 如 `eeg_feature_export_v0.1` |
| `emotion_id` | `int` | **训练对接用（可选）**：与 EMOPIA 四类 T5 目标对齐时填 `0..3`；无音乐侧对齐需求可不导出 |

---

## 2. 规则摘要

1. **一条记录 = 一个 trial 或一个 window**。仅 trial 级时可用 `segment_id = trial_id`。
2. **禁止**用音乐 clip 名作为 EEG 侧主键；与 EMOPIA 的配对在**自采闭环**或后续映射表中完成。
3. **`feature` 必须 `float32`**；维度优先 128，否则导出原始 `D` 并写明 `feature_dim`，**导出阶段不做 Linear(D→128)**。
4. **`segment_id` 在 dataset 内唯一**。

---

## 3. 推荐文件格式

1. **`torch.save(list_of_dicts)`**：元数据异构时最省事，`feature` 保持 `Tensor`。
2. **`npz`**：全体 `feature` 同维度 `D` 时，可堆成 `(N, D)`，其余字段平行数组。

---

## 4. 机读校验

- JSON Schema：`eeg_feature_export_v0.1.schema.json`（适用于把 `feature` 转成 `list` 的 JSON 校验）。
- Python：`scripts/eeg_export/record_v0_1.py`（Pydantic + `torch.Tensor`）。

---

## 5. 合成样例

```bash
conda activate aigc-m-py311
python scripts/eeg_export/make_synthetic_export.py
python scripts/eeg_export/validate_pt.py synthetic_eeg_export.pt
```

生成文件默认在**当前工作目录**；可改为写入 `res/continuous_features/`（需自行创建目录）。
