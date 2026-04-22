# AIGC-M

基于脑电情绪条件的音乐生成（MusicGen 微调 + 轻量 Adapter）。研究背景与导师信息见各文档中的说明。

## 仓库入口

| 资源 | 说明 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | 主线架构、数据流、踩坑与稳定结论（**主说明书**） |
| [docs/README.md](docs/README.md) | 文档索引：周志、ADR、实验台账、知识库 |
| [adapter_strategy_v1/README.md](adapter_strategy_v1/README.md) | 冻结 MusicGen + Adapter 方案与命令 |
| [environment.yml](environment.yml) | Conda 环境（`aigc-m-py311`） |

大块数据与权重通过根目录软链指向外置盘：`res/`、`output/`、`mat/`、`adapter_strategy_v1/checkpoints/`（见 `CLAUDE.md` 与 ADR）。

## 快速开始（概要）

```bash
conda env create -f environment.yml
conda activate aigc-m-py311
# 端到端微调主线
python src/preprocess_emopia.py
python src/train_e2e.py
python src/generate_e2e.py
# Adapter 方案（与上表一致时重算 target）
python adapter_strategy_v1/prepare_text_targets.py --model-dir facebook/musicgen-small
python adapter_strategy_v1/train_adapter.py --eeg-v01 res/continuous_features/deap_v01_export.pt
python adapter_strategy_v1/generate_adapter.py --model-dir facebook/musicgen-small
```

研究笔记与 Q&A 草稿在 **`docs/knowledge/qa/`**（原 `github_archive/Q@A` 已并入此处，**勿再**维护双份）。

## License

Research use only.
