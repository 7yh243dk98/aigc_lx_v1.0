# 连续特征条件训练方案（v1）

## 目标

在保证 MusicGen 稳定音质的前提下，将任务从 `emotion_id -> 音乐` 升级为 `连续特征 -> 音乐`：

- 冻结 MusicGen 主干
- 仅训练轻量条件编码器
- 2-4 周内得到可复现结果

---

## 分阶段推进

### 阶段 A（快速基线，1-2 周）

- 输入：段级连续特征（优先 128 维；备选 valence/arousal 两维扩展到 128 维）
- 模型：`ConditionEncoder + Frozen MusicGen`
- 输出：稳定、可控、无噪声崩溃的生成结果

### 阶段 B（能力增强，2-3 周）

- 输入：若可用，则改为时序特征 `[B, T, D]`
- 模型：时序编码器（Transformer/GRU）+ 适配头输出 `[B, seq_len, 768]`
- 增加更强的控制损失（如情绪一致性头）

---

## 数据规范（最小可用）

每条样本至少包含：

- `sample_id`：字符串（必须能映射到音频/token 样本）
- `feature`：浮点向量（推荐 `D=128`）
- `emotion_id`：`[0,1,2,3]`（可选，但建议保留用于监控）

若提供时序特征：

- `feature_seq`：`[T, D]`
- `feature_mask`：`[T]`（可选）

推荐存储：

- `res/continuous_features/features.pt`
- 字典形式：
  - `features_by_id[sample_id] = torch.Tensor([D])` 或 `[T, D]`

---

## 训练结构（v1）

1. `ConditionEncoder`：
   - 段级输入用 MLP
   - 时序输入用 Transformer/GRU
2. `AdapterHead`：
   - 将隐藏状态映射到 `[seq_len, 768]`
3. 冻结 `MusicgenForConditionalGeneration`：
   - 使用 `encoder_outputs=(cond_hidden,)`
   - 使用 `attention_mask=cond_mask`
   - 固定 `guidance_scale=1.0`

---

## 损失设计

### 必选

- `L_token`：MusicGen token 重建损失（`model(..., labels=...)` 的交叉熵）

### 可选（建议）

- `L_embed`：对齐文本锚点嵌入的 MSE（稳定训练）
- `L_emotion`：情绪分类/回归一致性损失

总损失：

`L = L_token + lambda1 * L_embed + lambda2 * L_emotion`

初始系数建议：

- `lambda1 = 0.1`
- `lambda2 = 0.2`

---

## 超参数（安全起步）

- Batch size：`4-8`
- 学习率（编码器/适配头）：`1e-4`
- 优化器：`AdamW`
- Weight decay：`1e-2`
- Epoch：`30`
- 梯度裁剪：`1.0`
- Early stopping patience：`5`

---

## 评估清单

### 客观指标

- 训练/验证 token loss
- CLAP score
- FAD

### 主观听感

- 四情绪试听（建议 >=5 人）
- 评分维度：音质 / 情绪匹配 / 多样性（1-5 分）

### 控制可靠性

- 特征插值测试：
  - 对特征 A->B 做插值，检查输出情绪是否平滑变化

---

## 风险与防护

- 风险：没有真实连续特征  
  - 防护：如果检测到 only one-hot 伪特征，直接 fail fast

- 风险：小数据过拟合  
  - 防护：冻结 MusicGen，仅训条件编码器 + early stopping

- 风险：loss 降低但听感变差  
  - 防护：每 N 个 epoch 固定导出试听检查点

---

## 本周可执行 TODO

1. 与同学确定特征文件格式
2. 增加严格数据加载器（不再 fallback 到 one-hot）
3. 实现 `train_continuous_adapter.py`（包含 `L_token`）
4. 跑首轮消融：
   - A1：one-hot 基线
   - A2：段级连续特征
5. 导出 8 段 demo 音频 + CLAP/FAD 表格

