# archive_legacy 使用说明

该目录用于保存**历史代码与探索材料**，不属于当前主流程，但对论文撰写（方法演进、失败路径、消融过程）有参考价值。

## 目录约定

- `src_legacy/`：历史版本主脚本（从 `src/` 迁出）
- `src_legacy_drop_snapshot/`：最早 `drop/` 快照，原样保留
- `experiments/`：调试与验证脚本、临时实验
- `experiments/practice/`：练习与草稿文件
- `notebooks/`：历史 notebook 与 checkpoint
- `misc/`：旧依赖记录、零散说明文件

## 当前主线（不在本目录）

当前建议只维护并运行以下主线代码：

- `src/preprocess_emopia.py`
- `src/train_e2e.py`
- `src/generate_e2e.py`
- `src/diagnose.py`
- `adapter_strategy_v1/*`

## 使用建议

- 新实验先在主线目录开发，确认弃用后再迁入此目录。
- 论文写作时可按时间从此目录回溯“方案演进”和“踩坑记录”。
