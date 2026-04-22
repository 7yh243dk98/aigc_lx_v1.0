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
- `prepare_text_targets.py`：一次性生成文本编码目标嵌入（`--model-dir`：Hub id 或本地目录，**需与最终 `generate_adapter` 使用的 MusicGen 一致**）
- `train_adapter.py`：用 masked MSE 训练 Adapter
- `generate_adapter.py`：冻结 MusicGen + Adapter；`--model-dir`、`--test-text`（纯文本验基座）、`--from-eeg-v01`；WAV 采样率取自 `model.config.audio_encoder.sampling_rate`

## 快速运行

```bash
# 建议显式指定与推理一致的 MusicGen，例如官方 small：
python adapter_strategy_v1/prepare_text_targets.py --model-dir facebook/musicgen-small
python adapter_strategy_v1/train_adapter.py --eeg-v01 res/continuous_features/deap_v01_export.pt
python adapter_strategy_v1/generate_adapter.py --model-dir facebook/musicgen-small
```

## 说明

- 当前版本在样本不存在 `eeg_feature` 时，会回退到情绪 one-hot 输入（便于流程验证）。
- 若数据中提供真实 EEG 特征（`item["eeg_feature"]`，128 维），`train_adapter.py` 会自动优先使用。
- **公开 DEAP → v0.1 导出**：`scripts/eeg_public/deap_to_v01.py` + `README_DEAP.md`；训练用 `train_adapter.py --eeg-v01 <导出.pt>`。
- **生成参数**：`--test-text` 纯文本试听时默认 **`guidance_scale=3.0`**（与 HF 文档常见设置一致）；用 **Adapter / `encoder_outputs`** 时默认 **`guidance_scale=1.0`**，可用 `--guidance-scale` 覆盖。
- **实验结论与下一步**：见 `docs/experiments/2026-04-EEG-adapter-musicgen-baseline.md`。

