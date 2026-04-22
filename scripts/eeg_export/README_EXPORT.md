# EEG Feature Export（v0.1）

规范全文见：`docs/contracts/eeg_feature_export_v0.1.md`。

## 公开数据（DEAP → v0.1）

同学模型未就绪时，可用 **DEAP 预处理 Python** 导出契约格式，见 **`scripts/eeg_public/README_DEAP.md`** 与 `deap_to_v01.py`。

## 快速生成合成数据（N=16）

```bash
# 在项目根目录
python scripts/eeg_export/make_synthetic_export.py
```

生成 `synthetic_eeg_export.pt` 与 `synthetic_eeg_export.npz`（默认写在**当前工作目录**）。

## 校验 .pt

```bash
python scripts/eeg_export/validate_pt.py synthetic_eeg_export.pt
```

## 与 `train_adapter` 对接

合成数据已带可选字段 `emotion_id`（0..3），可直接试训：

```bash
python scripts/eeg_export/make_synthetic_export.py
cd adapter_strategy_v1
python train_adapter.py --eeg-v01 ../synthetic_eeg_export.pt --epochs 1 --batch-size 8
```

若真实导出无 `emotion_id`，可临时用 `--eeg-v01-default-emotion-id 0` 做冒烟（不推荐用于正式实验）。

## 推理：用 v0.1 特征生成音频

需已训练好 `adapter_strategy_v1/checkpoints/adapter_last.pt`（`input_dim` 与导出 `feature` 维一致）：

```bash
cd adapter_strategy_v1
python generate_adapter.py --from-eeg-v01 ../synthetic_eeg_export.pt --eeg-indices 0,1,2
```

记录无 `emotion_id` 时加 `--mask-emotion-id 0`（选用哪一类的 `attention_mask`）。

## Pydantic 模型

`record_v0_1.py` 中 `EEGExportRecordV01` 可用于加载后强类型校验。

依赖：`pip install pydantic`（已写入根目录 `environment.yml` 的 pip 列表）。
