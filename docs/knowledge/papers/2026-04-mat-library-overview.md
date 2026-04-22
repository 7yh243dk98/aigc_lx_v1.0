# mat 文献库通读总览（2026-04）

原文 PDF 位于 `mat/`（外置盘：`/mnt/data/aigc_data/refs/mat`）。本页为通读摘要，公式与图表以 PDF 为准。

---

## 1. Dash & Agres, 2024 — AI-Based Affective Music Generation: Survey (ACM Comput. Surv.)

**要点**：可控「情感音乐生成」综述——任务定义、可控信号、算法分类、音乐特征操纵方式、挑战与开放问题。

**与 AIGC-M**：写 related work、选评估维度（效价/唤醒、可控性）；论证「显式情感控制 + 评估」的必要性。128 维生理/状态条件可归入 physiology-conditioned AMG。

---

## 2. Luo 等, 2022 — Music generation based on emotional EEG (ICIAI)

**要点**：DEAP 上 SVM 效价二分类；EMOPIA 上 seq2seq LSTM 情感音乐生成；EEG 特征映射到音乐参数再生成。

**与 AIGC-M**：早期 EEG→情绪→音乐流水线，可作**历史演进**引用；与当前 MusicGen + Adapter 不同代，不宜作 SOTA 主对标。

---

## 3. Lv 等, 2024 — Auditory entrainment… cortical-BNST-NAc triple time locking (Cell Reports)

**要点**：难治性抑郁；颅内 BNST–NAc + 颞区 EEG；抗抑郁效果与**主观喜爱/享受**关联强；theta、gamma、三重时间锁相等机制。

**与 AIGC-M**：闭环叙事与终点设计——除效价外，**享受度、theta** 有高质量机制文献支撑；适合 Discussion 与预注册指标。

---

## 4. Postolache 等, 2025 — Naturalistic Music Decoding from EEG via Latent Diffusion (arXiv:2405.09062)

**要点**：少手工预处理下，用 **ControlNet** 条件化预训练扩散模型（AudioLDM2 路线）做自然音乐解码；NMED-T；指标含 FAD、CLAP 相关、MSE 等。

**与 AIGC-M**：**冻结大生成器 + 轻量条件分支**范式；阶段 2 客观指标（FAD、嵌入相似度）可直接借鉴。当前工程为 MusicGen + Adapter，与此文同哲学、不同骨干；可作 Phase B 或 related work。

---

## 5. Shen 等, 2024 — A First Look at AIGC-based Music Therapy (IEEE TCE)

**要点**：传统音乐治疗局限；AIGC 机遇；**闭环音乐治疗**与消费者电子场景。

**与 AIGC-M**：引言与系统级立意；与 `CLAUDE.md` 总目标一致。

---

## 6. Shen 等, 2025 — AIGC-based Closed-Loop Music Intervention (IEEE TAFFC, author version)

**要点**：**3 导联 EEG** 情绪模型 + **扩散**音乐生成，闭环演示；强调实时评估与干预循环。

**与 AIGC-M**：闭环产品形态参考；条件为离散情绪 + 扩散，本项目为连续 128 维 + MusicGen——**同属闭环干预，条件形式不同**。

---

## 7. Shen 等 — Echoes of the Brain (IEEE TMM)

**要点**：阶段一：**对比学习**对齐 EEG 与音乐（CLAP 类）表征；阶段二：**DiT + EEG ControlNet** 重建。强调 EEG–音乐表征对齐的重要性。

**与 AIGC-M**：「先对齐、再生成」；当前 Adapter 可视为**任务特化的浅层对齐**。Related work 核心引用；若引入对比学习/共享嵌入空间，此文为主要参照。

---

## 8. Zhao 等, 2026 — FMEFF / FastDTW 音乐–EEG 融合与享受度 (IEEE TAFFC)

**要点**：MFCC + **FastDTW** 对齐音乐与 EEG；预测**音乐享受度**；享受与情绪维度区分；分脑区可解释性。

**与 AIGC-M**：闭环可增「享受评分 + 同步化类特征」；与 Lv 2024 在机制—测量上可呼应。

---

## 主题—文献速查

| 主题 | 优先文献 |
|------|----------|
| 冻结大模型 + 轻量条件 | Postolache 2025；Echoes（ControlNet 思想） |
| 闭环与 TAFFC 叙事 | Shen 2024 / 2025 |
| EEG–音乐对齐 | Echoes |
| 客观评估（FAD、CLAP 类） | Postolache 2025 |
| 机制与终点（theta、享受） | Lv 2024；Zhao 2026 |
| 可控 AMG 综述 | Dash 2024 |

---

## 维护

- 新增 PDF 时：在 `mat/` 放入文件后，可本地 `pdftotext` 抽文本辅助笔记，并在此文件或单篇 `YYYYMMDD-title.md` 中增补条目。
- 单篇精读请复制 `TEMPLATE.md` 撰写。
