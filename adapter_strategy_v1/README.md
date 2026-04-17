# Adapter 策略 v1

本目录是一个隔离的最小可运行方案，用于：

1. 冻结 MusicGen 主体；
2. 仅训练轻量 Adapter；
3. 避免全量/部分解冻 decoder 导致的音质崩溃。

## 为什么单独建目录

原始 `src/` 中有较多探索脚本。  
为了避免新旧路线混在一起，本目录只放“新策略”相关文件。

## 文件说明

- `models.py`：轻量 `EmotionToTextAdapter` 模型
- `prepare_text_targets.py`：一次性生成文本编码目标嵌入
- `train_adapter.py`：用 masked MSE 训练 Adapter
- `generate_adapter.py`：冻结 MusicGen + 已训练 Adapter 生成音频

## 快速运行

```powershell
python adapter_strategy_v1/prepare_text_targets.py
python adapter_strategy_v1/train_adapter.py
python adapter_strategy_v1/generate_adapter.py
```

## 说明

- 当前版本在样本不存在 `eeg_feature` 时，会回退到情绪 one-hot 输入（便于流程验证）。
- 若数据中提供真实 EEG 特征（`item["eeg_feature"]`，128 维），`train_adapter.py` 会自动优先使用。
- 生成时固定 `guidance_scale=1.0`，用于规避直接条件注入时 CFG 的 shape mismatch 问题。

