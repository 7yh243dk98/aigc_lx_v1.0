# 0002 - 合并 `github_archive/` 到主线文档

状态：Accepted  
日期：2026-04-22

## 背景（Context）

`github_archive/` 含与根目录 `src/` 重复的脚本、与 `docs/knowledge/qa/` 完全一致的笔记，以及另一份 `README` / `CLAUDE.md` / `config` 副本。双轨维护易漂移、易误跑旧路径。

## 决策（Decision）

- 以**仓库根**为唯一真源：代码用 `src/`、`adapter_strategy_v1/`；研究笔记用 **`docs/knowledge/qa/`**。
- 删除目录 **`github_archive/`**（不保留子模块）。
- 在仓库根增加 **`README.md`**，对外简介与入口链接集中于此；HuggingFace 配套 `config.json` 以 **`res/.../musicgen_*/` 内文件** 为准，不单独在根下保留副本。
- 原 `github_archive/mat/PIIS2211124724008039.txt`（论文英文摘录）迁入 **`docs/knowledge/papers/`**，避免随目录删除丢失。

## 影响与后果（Consequences）

- 克隆本仓库后不再出现「两套 `src`」的歧义；旧链接若指向 `github_archive/`，需改为根下路径或 `docs/knowledge/qa/`。
