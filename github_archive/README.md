# AIGC-M: EEG-Emotion Conditioned Music Generation for Depression Treatment

基于脑电情绪特征的治疗性音乐生成系统

## 项目简介

本项目微调 [MusicGen](https://huggingface.co/facebook/musicgen-small)，通过情绪条件（来自 EEG 信号提取的情绪特征）生成可用于治疗抑郁的音乐。核心思路：将离散情绪标签映射为 T5 编码器分布兼容的条件向量，输入 MusicGen 的 cross-attention 实现情绪条件生成。

**研究方向**：兰州大学 · 导师田福泽/胡斌 · 目标期刊 IEEE TAFFC

## 技术架构

```
情绪标签 (Q1-Q4)
  → EmotionEmbedding (T5 文本编码初始化，可微调)
  → [B, seq_len, 768]
  → MusicGen.enc_to_dec_proj (768→1024)
  → Decoder cross-attention + 自回归生成
  → EnCodec 解码 → WAV
```

### 关键设计

- **T5 初始化情绪嵌入**：用 T5 编码情绪描述文本（如 "happy energetic upbeat cheerful music"）初始化可学习嵌入，确保与预训练 cross-attention 的输入分布一致
- **部分冻结微调**：冻结大部分参数，只训练 cross-attention、enc_to_dec_proj 和 decoder 后 4 层 self-attention
- **EMOPIA 数据集**：1071 段带情绪标签的钢琴 MIDI，经 FluidSynth + SoundFont 合成高质量 WAV，再用 EnCodec 编码为 token

## 文件结构

```
├── CLAUDE.md              # 详细项目说明（架构、踩坑记录、决策日志）
├── README.md              # 本文件
├── requirements.txt       # Python 依赖
├── src/
│   ├── preprocess_emopia.py  # MIDI → WAV → EnCodec tokens
│   ├── train_e2e.py          # MusicGen 端到端微调
│   ├── generate_e2e.py       # 微调后生成情绪音乐
│   └── diagnose.py           # 诊断工具（分布比较、pipeline 验证）
├── config/                   # 模型配置（不含权重）
├── Q@A/                      # 研究思路与讨论记录
├── mat/                      # 参考论文摘要
└── output_samples/           # 生成样例（T5 初始化验证）
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 另需下载（不包含在仓库中）：
# 1. EMOPIA 数据集 → res/emopia_data/EMOPIA_2.2/
# 2. FluidSynth v2.5.2 → res/fluidsynth-v2.5.2-win10-x64-cpp11/
# 3. GeneralUser GS SoundFont → res/GeneralUser-GS.sf2

# 预处理
python src/preprocess_emopia.py

# 训练
python src/train_e2e.py

# 生成
python src/generate_e2e.py
```

## 主要踩坑

1. **情绪嵌入分布不匹配**（致命）：随机初始化的嵌入 std≈0.05 vs T5 输出 std≈0.27，cross-attention 几乎忽略条件信号 → 用 T5 文本编码初始化解决
2. **labels 形状**：必须是 `[B, T, num_codebooks]` 而非 `[B, num_codebooks, T]`
3. **guidance_scale**：直接传 encoder_outputs 时必须设为 1.0，否则 CFG 机制报 tensor 维度错误
4. **decoder_start_token_id**：需手动从 generation_config 赋值到 config.decoder

详见 `CLAUDE.md` 第 8 节完整踩坑记录。

## 情绪标签

| ID | Valence | Arousal | 描述 |
|----|---------|---------|------|
| Q1 | 高 | 高 | 开心/兴奋 |
| Q2 | 低 | 高 | 愤怒/紧张 |
| Q3 | 低 | 低 | 悲伤/忧郁 |
| Q4 | 高 | 低 | 平静/放松 |

## 当前状态

- [x] 数据预处理 pipeline（FluidSynth + EnCodec）
- [x] 训练代码（MusicGen 微调 + checkpoint 续训）
- [x] 诊断并修复情绪嵌入分布不匹配
- [ ] 用 v2 架构完整训练 + 生成质量评估
- [ ] EEG 情绪特征对接
- [ ] 闭环实时系统

## License

Research use only.
