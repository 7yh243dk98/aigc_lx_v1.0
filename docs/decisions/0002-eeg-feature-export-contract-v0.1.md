# 0002 - EEG 条件特征导出契约 v0.1

状态：Accepted  
日期：2026-04-19

## 背景

音乐侧 Adapter 需要**稳定、可机读**的 EEG 条件输入；协作方使用**非 EMOPIA** 的公开/校内数据集，**不要求**与音乐 clip 名逐条对齐。

## 决策

- 采用 **`eeg_feature_export_v0.1`**：每条 trial/window 一条记录，`feature` 为 `float32` 一维向量；优先长度 128，否则导出原始维度 `D` 并填写 `feature_dim`。
- 主键语义：`dataset` + `subject_id` + `segment_id`（`segment_id` 在 dataset 内唯一）。
- 规范文档：`docs/contracts/eeg_feature_export_v0.1.md`  
- JSON Schema：`docs/contracts/eeg_feature_export_v0.1.schema.json`（适用于 JSON 数组形式的 feature）  
- Python 校验：`scripts/eeg_export/record_v0_1.py`（Pydantic + Tensor）

## 后果

- 双方可先靠 **合成样例**（`scripts/eeg_export/make_synthetic_export.py`）对齐，再替换真实导出。
- 与 EMOPIA 的配对推迟到**映射表**或**自采闭环**，不在此契约中强制。

## 相关

- `scripts/eeg_export/README_EXPORT.md`
