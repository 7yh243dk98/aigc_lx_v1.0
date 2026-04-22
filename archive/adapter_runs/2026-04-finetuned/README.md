# 2026-04：finetuned MusicGen 时代的 Adapter 产物（归档）

迁移到 **`facebook/musicgen-small` 全链路** 前，与本机 `res/musicgen_finetuned` + 旧 `target_text_embeddings` / `adapter_last` 相关的实验，可把**副本**放在此目录留档，避免与「官方 small 主线」混淆。

## 建议迁入的文件（按需复制，勿必全有）

| 原路径（示例） | 说明 |
|----------------|------|
| `adapter_strategy_v1/target_text_embeddings.pt`（旧版） | 若曾用 `prepare_text_targets --model-dir res/musicgen_finetuned` |
| `adapter_strategy_v1/target_text_embeddings.json` | 对应元数据 |
| `adapter_strategy_v1/checkpoints/adapter_last.pt`（DEAP 等旧训） | 旧 Adapter 权重 |
| 其它命名备份如 `*.bak_*` | 已手动备份的可再拷一份 |

复制后，在仓库根保留的 **active** 文件应仅对应 **当前主线**（例如全用 `facebook/musicgen-small` 重算、重训后的版本）。

## 注意

- `res/musicgen_finetuned` **整目录**体积大，一般**不**移进 git；此处只存**小文件**与说明。模型目录可在数据盘保留，本 README 仅作索引。
- 大文件若不上传远端，仅本地留档即可。
