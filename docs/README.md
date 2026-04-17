# 文档索引

本目录用于承载 `CLAUDE.md` 之外的过程性与结构化文档，避免主说明书膨胀。

## 子目录

- `logs/`：按周记录研发进展、问题与下周计划
- `decisions/`：关键技术决策（ADR 风格）
- `experiments/`：实验记录模板与实验台账
- `knowledge/`：可复用知识库（Q&A、阅读摘要、术语笔记）
- `knowledge/papers/`：论文结构化笔记（PDF 原文在 `mat/`）

## 使用约定

- `CLAUDE.md` 只保留：当前主线、近期目标、关键踩坑与稳定结论。
- 过程细节（每日/每周动作）放入 `logs/`。
- 重要“为什么这样做”放入 `decisions/`。
- 可复现实验过程和结果放入 `experiments/`。
- 可复用的小资源与问答笔记放入 `knowledge/`。
- 论文阅读建议“双轨”：原文放 `mat/`，总结放 `knowledge/papers/`。
