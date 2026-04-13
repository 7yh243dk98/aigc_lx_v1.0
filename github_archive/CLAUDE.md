# AIGC-M 项目说明书：脑电-情绪-音乐闭环干预系统

## 1. 项目概述

### 1.1 项目背景
本项目由兰州大学本科生（大二下）与导师田福泽/胡斌共同开展，目标发表于 **IEEE TAFFC（一区）**。

### 1.2 研究方向演进
| 阶段 | 方向 | 说明 |
|------|------|------|
| v1 | EEG情绪 → 匹配音乐 | 悲伤→悲伤音乐（简单映射） |
| v2 | EEG情绪 → 反向调节音乐 | 悲伤→开心音乐（情绪调节） |
| v3（当前） | 神经靶向治疗性音乐生成 | 激活特定皮层区域，治疗抑郁等精神障碍 |

### 1.3 核心创新点
- **反向情绪调节**：不是情绪匹配，而是生成能改善负面情绪的音乐
- **神经靶向干预**：利用 theta 节律（4-8Hz）调制等手段，针对性激活脑区
- **闭环系统**：实时 EEG 反馈 → 动态调整音乐生成策略

---

## 2. 系统架构

### 2.1 功能树
```
AI音乐治疗干预系统
├── 脑电信号处理
│   ├── EEG特征提取（MNE-Python）
│   │   ├── theta功率（4-8Hz）
│   │   └── wPLI 功能连接
│   └── 潜在空间对齐
│       ├── EEG编码器
│       └── 音乐解码器
├── 生成式音乐合成（当前核心）
│   ├── EMOPIA情绪音乐数据集
│   ├── MusicGen端到端微调
│   └── EnCodec音频编解码
├── 情绪调节干预
│   ├── 情绪状态识别（CNN / LSTM）
│   └── 干预策略适配（反向调节映射）
├── 闭环治疗框架（未来）
│   ├── 实时反馈控制（MNE-Python + LSL）
│   └── 治疗效果评估
└── 智能体协同（未来）
    ├── 治疗Agent
    └── 技能库管理
```

### 2.2 数据流向
```
EEG信号（他人提供）
  ↓
情绪特征解析 (src/emotion_parser.py - 未来)
  ↓  输出：离散标签(Q1-Q4) + 连续维度(valence/arousal)
情绪条件输入
  ↓
音乐生成 (src/train_e2e.py / src/generate_e2e.py)
  ↓  MusicGen端到端微调
音频输出 (output/e2e_generated/*.wav)
  ↓
反馈评估（闭环，未来）
```

---

## 3. 目录结构

```
aigc-m/
├── CLAUDE.md                 # 本文件：项目全局说明
├── src/                      # 源代码
│   ├── preprocess_emopia.py  # ★ 核心：MIDI → WAV → EnCodec tokens 预处理
│   ├── train_e2e.py          # ★ 核心：MusicGen 端到端微调训练
│   ├── generate_e2e.py       # ★ 核心：微调后生成情绪音乐
│   ├── diagnose.py           # 诊断脚本（分布比较、pipeline 验证）
│   ├── aigc.py               # 旧版：文本条件 demo（已搁置）
│   ├── preprocess.py         # 旧版：v2 文本嵌入预处理（已搁置）
│   ├── train.py              # 旧版：v2 Transformer EmotionEncoder（已搁置）
│   └── emotion_to_music.py   # 旧版：情绪到音乐映射（已搁置）
├── res/                      # 资源与数据
│   ├── emopia_data/EMOPIA_2.2/  # EMOPIA 数据集
│   │   ├── midis/               # 1071 个 MIDI 文件
│   │   └── label.csv            # 情绪标签 (Q1-Q4)
│   ├── emopia_tokens/           # 预处理输出
│   │   └── emopia_tokens.pt     # {name, emotion_id, audio_codes} 列表
│   ├── musicgen_finetuned/      # 微调后的模型权重
│   │   └── emotion_embed.pt     # EmotionEmbedding 权重
│   ├── GeneralUser-GS.sf2       # SoundFont 文件（高质量 MIDI 合成）
│   └── fluidsynth-v2.5.2-win10-x64-cpp11/  # FluidSynth 可执行文件
├── output/
│   ├── e2e_generated/           # 微调后生成的 WAV 文件
│   └── verify_fix/              # 修复验证生成的 WAV 文件
├── Q@A/                         # 研究思路记录
├── mat/                         # 参考论文
└── venv/                        # Python 虚拟环境
```

---

## 4. 核心模块详解

### 4.1 预处理：`src/preprocess_emopia.py`

**功能**：EMOPIA MIDI → WAV → EnCodec tokens

**数据流**：
```
MIDI 文件 → FluidSynth/pretty_midi 合成 → WAV (32kHz, 16bit, mono)
  → EnCodec 编码 → audio_codes [4 codebooks, T tokens]
  → 保存为 emopia_tokens.pt
```

**关键配置**：
| 参数 | 值 | 说明 |
|------|-----|------|
| SAMPLE_RATE | 32000 | EnCodec/MusicGen 原生采样率 |
| SF2_PATH | `res/GeneralUser-GS.sf2` | SoundFont 文件 |
| 音频截断 | 30秒（FluidSynth）/ 10秒（降级） | 避免超长序列导致显存溢出 |

### 4.2 训练：`src/train_e2e.py`

**架构（v2，T5 初始化）**：
```
EmotionEmbedding(emotion_id)
  → conditioning[emotion_id]  ← 由 T5 编码情绪文本初始化
  → [B, seq_len, 768] + attention_mask [B, seq_len]
  ↓
model(encoder_outputs=(...,), attention_mask=..., labels=...)
  ↓ 内部自动：enc_to_dec_proj (768→1024)
  ↓ 内部自动：shift_tokens_right（自回归右移）
  ↓ 内部自动：decoder forward + CrossEntropyLoss
```

**EmotionEmbedding 结构（v2，当前版本）**：
```python
# T5 编码 4 段情绪描述文本作为初始值
conditioning: nn.Parameter  # [4, max_seq_len, 768] ← 可学习，从 T5 编码初始化
attention_masks: buffer      # [4, max_seq_len]      ← 不可学习

# forward(emotion_ids) → (hidden_states, masks)
```

**初始化文本**：
| 情绪 ID | 文本描述 |
|---------|---------|
| 0 (Q1) | "happy energetic upbeat cheerful music with bright melody" |
| 1 (Q2) | "angry intense aggressive powerful music with strong beat" |
| 2 (Q3) | "sad melancholic gentle slow music with minor key" |
| 3 (Q4) | "calm peaceful relaxing ambient music with soft tone" |

**冻结策略**：
| 组件 | 状态 | 说明 |
|------|------|------|
| text_encoder (T5) | 完全冻结 | 仅初始化时用一次，运行时通过 encoder_outputs 绕过 |
| audio_encoder (EnCodec) | 完全冻结 | 仅预处理时使用 |
| enc_to_dec_proj | **解冻** | 768→1024 投影层 |
| decoder.encoder_attn (全部24层) | **解冻** | 交叉注意力，接收情绪条件 |
| decoder.self_attn (后4层: 20-23) | **解冻** | 让模型学习音乐模式 |
| decoder 其余层 | 冻结 | 保留预训练知识 |

**训练超参数**：
| 参数 | 值 |
|------|-----|
| batch_size | 4 |
| learning_rate | 1e-4 |
| weight_decay | 0.01 |
| epochs | 50 |
| max_tokens | 512 |
| gradient clipping | max_norm=1.0 |

**关键设计决策**：
- **使用 `model()` forward 而非直接调 `model.decoder()`**：通过 `encoder_outputs=(emotion_hidden_states,)` 传入条件，model.forward() 内部自动处理 `enc_to_dec_proj`、`shift_tokens_right`、decoder forward 和 loss 计算
- **labels 形状必须是 `[B, T, num_codebooks]`**：decoder loss 计算按 `labels[..., codebook]` 索引最后一维
- **labels 用 -100 padding**：CrossEntropyLoss 的 `ignore_index=-100`
- **需设置 `decoder_start_token_id`**：`shift_tokens_right` 需要此值填充右移后的首位，从 `generation_config` 读取 (2048)
- **emotion embedding 用 T5 文本编码初始化**：确保 cross-attention 从第一步就能接收到正确分布的条件信号

### 4.3 生成：`src/generate_e2e.py`

**推理路径**：
```
EmotionEmbedding(emotion_id)  → [1, seq_len, 768] + mask
  ↓
model.generate(encoder_outputs=(...,), attention_mask=..., guidance_scale=1.0)
  ↓ 内部自动调用 enc_to_dec_proj (768→1024)
  ↓ 内部自动调用 decoder 自回归生成
audio_values → 归一化 → int16 → WAV 文件
```

**生成参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| max_new_tokens | 256 | 约8秒音频 |
| guidance_scale | 1.0 | **必须设为 1.0**，禁用 CFG（见踩坑 8.6） |
| temperature | 0.7 | 控制随机性 |
| top_p | 0.9 | 核采样 |
| do_sample | True | 随机采样 |

### 4.4 情绪标签体系

EMOPIA 数据集使用 Russell 四象限模型：

| ID | 标签 | Valence | Arousal | 描述 |
|----|------|---------|---------|------|
| 0 | Q1 | 高 | 高 | 开心/兴奋 |
| 1 | Q2 | 低 | 高 | 愤怒/紧张 |
| 2 | Q3 | 低 | 低 | 悲伤/忧郁 |
| 3 | Q4 | 高 | 低 | 平静/放松 |

**数据分布**（1071样本）：Q1=246, Q2=263, Q3=253, Q4=309（基本均匀）

---

## 5. 技术栈

### 5.1 核心依赖
| 库 | 用途 |
|----|------|
| transformers | MusicgenForConditionalGeneration, T5, EnCodec |
| torch | 训练框架 |
| pretty_midi | MIDI 解析 |
| scipy | WAV 读写 |
| FluidSynth (命令行) | 高质量 MIDI→WAV 合成 |
| GeneralUser GS (.sf2) | SoundFont 音色库 |

### 5.2 模型规格
| 组件 | 模型 | 维度 |
|------|------|------|
| Text Encoder | T5-small | 768 |
| Projection | enc_to_dec_proj | 768→1024 |
| Decoder | MusicGen-small (24层) | 1024 |
| Audio Codec | EnCodec | 4 codebooks |

---

## 6. 环境配置

### 6.1 基础环境
- **OS**: Windows 11 (25H2)
- **Python**: 3.11, 虚拟环境在 `venv/`
- **GPU**: CUDA (如有)
- **Shell**: PowerShell（**注意**：不支持 `&&`，用 `;` 分隔命令）

### 6.2 HuggingFace 镜像
国内访问需在代码顶部设置：
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 6.3 FluidSynth 配置
- **可执行文件**：`res/fluidsynth-v2.5.2-win10-x64-cpp11/bin/fluidsynth.exe`
- **SoundFont**：`res/GeneralUser-GS.sf2`
- **DLL 依赖**：需将 FluidSynth 的 bin 目录加入 PATH

---

## 7. 运行指南

### 7.1 完整流程
```bash
# 1. 激活虚拟环境
D:\pyprojects\aigc-m\venv\Scripts\Activate.ps1

# 2. 预处理：MIDI → tokens
python src/preprocess_emopia.py

# 3. 训练：端到端微调 MusicGen
python src/train_e2e.py

# 4. 生成：4种情绪各生成一首
python src/generate_e2e.py
```

### 7.2 预期输出
- 预处理：`res/emopia_tokens/emopia_tokens.pt`（约1071个样本）
- 训练：`res/musicgen_finetuned/`（模型权重 + emotion_embed.pt）
- 生成：`output/e2e_generated/Q1_happy.wav` 等4个文件

---

## 8. 已知问题与踩坑记录

### 8.1 EnCodec API
| 问题 | 原因 | 解决 |
|------|------|------|
| `'EncodecModel' has no attribute 'sampling_rate'` | API 变更 | `audio_encoder.encode(audio, None)` |
| audio_codes shape 不对 | nb_chunks 维度 | `permute(1,0,2).reshape(4, -1)` |

### 8.2 MusicGen 训练
| 问题 | 原因 | 解决 |
|------|------|------|
| text_encoder 需要 input_ids | `model()` forward 强制调 T5 | 传 `encoder_outputs` 参数跳过 text_encoder |
| loss 居高不下(3.8) | `cross_attention` 从未解冻 | MusicGen 用 `encoder_attn` 而非 `cross_attention` |
| 模型学习 padding 噪音 | labels 用 0 填充 | labels 用 -100 填充（CrossEntropy ignore） |
| labels 形状错误（致命） | labels 为 `[B,4,T]` 但 loss 按 `labels[...,codebook]` 索引 | labels 必须是 `[B,T,4]` |
| input_ids 泄露答案（致命） | 同时传 input_ids 和 labels 跳过了 shift_tokens_right | 只传 labels，让 model.forward() 自动右移 |
| decoder_start_token_id 为 None | config.decoder 中未设置 | 从 generation_config 读取并赋值 (2048) |

### 8.3 情绪嵌入分布不匹配（致命，v1 架构的根因）
| 问题 | 原因 | 解决 |
|------|------|------|
| 生成全是噪声，peak≈0.002 | nn.Embedding 随机初始化产生 std≈0.05 的向量，而 T5 编码器输出 std≈0.27，差 5 倍 | **v2 架构：用 T5 编码情绪文本初始化嵌入，使分布匹配** |
| 四个情绪听起来一样 | 随机嵌入中 (Q1,Q2) cos=0.68, (Q3,Q4) cos=0.78，模型分不清 | T5 初始化后余弦相似度更均匀（0.64-0.74），且模型已预训练能区分 |
| loss 几乎不下降 (5.54→5.39) | cross-attention 收到的条件信号太弱，等同无条件训练 | T5 初始化后条件信号正确，loss 应能正常下降 |

**诊断方法**：
```python
# 对比 T5 编码器输出 vs 随机情绪嵌入的分布
T5 输出: mean≈0, std≈0.27, range [-2.4, 2.0]    ← cross-attention 期望的分布
旧嵌入: mean≈0, std≈0.05, range [-0.4, 0.5]     ← 信号太弱
新嵌入: mean≈0, std≈0.27（与 T5 一致）            ← 修复后
```

**验证结果**：修复后未经微调的预训练模型 + T5 初始化嵌入，生成 peak 从 0.002 → 0.124（提升 60 倍），且 happy/sad 幅值有明显差异。

### 8.4 FluidSynth
| 问题 | 原因 | 解决 |
|------|------|------|
| `module 'fluidsynth' has no attribute 'Synth'` | pyfluidsynth 找不到 C 库 | 改用 subprocess 命令行调用 |
| exit code 3221225781 (0xC0000135) | 缺少 VC++ 运行时 DLL | 下载 glib 版本（包含完整依赖） |

### 8.5 推理陷阱
| 问题 | 原因 | 解决 |
|------|------|------|
| 训练-推理路径必须一致 | 训练用 `model(encoder_outputs=...)` 推理用 `model.generate(encoder_outputs=...)` | 两者都经过同一条 enc_to_dec_proj 路径 |
| `model.decoder.generate()` 报错 | decoder 没有 generate 方法 | 用 `model.generate(encoder_outputs=...)` |

### 8.6 Classifier-Free Guidance 陷阱
| 问题 | 原因 | 解决 |
|------|------|------|
| generate() 报 tensor shape mismatch (16 vs 8) | MusicGen 默认 guidance_scale=3.0，需要 conditional+unconditional 两组 encoder_outputs | **必须**传 `guidance_scale=1.0` 禁用 CFG |
| 不传 guidance_scale 导致 SDPA 维度错误 | CFG 期望 batch 翻倍，但直接传 encoder_outputs 没做 unconditional 拼接 | 同上 |

### 8.7 外部建议验证
| 来源 | 建议 | 实际验证结果 |
|------|------|-------------|
| DeepSeek | "参数名应为 `encoder_hidden_states` 而非 `encoder_outputs`" | **错误**。源码明确 `MusicgenForConditionalGeneration.forward()` 接受 `encoder_outputs`，`encoder_hidden_states` 只是内部局部变量。若传 `encoder_hidden_states` 会进入 `**kwargs` 被静默忽略 |

### 8.8 环境问题
| 问题 | 原因 | 解决 |
|------|------|------|
| PowerShell 中 `&&` 报错 | PS 不支持 `&&` | 用 `;` 分隔 |
| HuggingFace 连接超时 | 国内网络 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| 蓝屏崩溃 | GPU 显存溢出触发驱动崩溃 | 减少同时加载的模型数量，及时释放显存 |

---

## 9. 架构演进记录

### v1：随机初始化嵌入（已废弃）
```
nn.Embedding(4, 768) → Linear → LayerNorm → GELU → Linear → expand to [B, 32, 768]
```
**问题**：随机初始化产生的向量与 T5 编码器输出分布不匹配（std 差 5 倍），cross-attention 几乎忽略条件信号。训练 30 epoch 后 loss 从 5.54 仅降到 5.39，生成全是噪声。

### v2：T5 文本编码初始化（当前版本）
```
T5.encode("happy energetic upbeat cheerful music...") → [1, seq_len, 768]
→ 存为 nn.Parameter, 可微调
→ 附带 attention_mask buffer 处理不等长 padding
```
**优势**：
1. 初始分布与 T5 输出一致（std≈0.27），cross-attention 立即生效
2. 不同情绪文本自然产生不同嵌入，无需学习基础区分
3. 可微调参数，训练中进一步适应 EMOPIA 数据特征
4. 兼顾了"用现有预训练知识"和"可学习定制化"

---

## 10. 未来方向与路线图

### 10.1 近期目标
- [x] FluidSynth 高质量预处理（v2.5.2 已部署，1071 样本已处理）
- [x] 修复 labels 形状、shift_tokens_right、decoder_start_token_id
- [x] 诊断并修复情绪嵌入分布不匹配问题（v2 T5 初始化）
- [ ] 用 v2 架构从头训练并验证生成质量
- [ ] 实现反向情绪调节映射（输入悲伤→生成开心音乐）
- [ ] 添加 theta 节律（4-8Hz）振幅调制（后处理方式）

### 10.2 中期目标（论文核心）
- [ ] 端到端 theta 调制：将 theta 参数作为条件输入微调 MusicGen
- [ ] EEG 情绪特征解析模块（接收上游特征）
- [ ] 被试实验验证（健康被试，需伦理审批）
- [ ] IEEE TAFFC 论文撰写

### 10.3 远期目标（闭环系统）
- [ ] 实时 EEG 采集与在线处理（MNE-Python + LSL）
- [ ] 闭环反馈控制：EEG → 情绪识别 → 音乐调整
- [ ] 治疗 Agent 系统

---

## 11. 关键决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| - | 选择 MusicGen 而非从零训练 | 科研初期效率优先，MusicGen 有成熟的条件生成架构 |
| - | 开环系统优先于闭环 | 降低复杂度，先验证音乐生成质量 |
| - | 通过 encoder_outputs 绕过 T5 | 用情绪 ID 而非文本，传 encoder_outputs 跳过 text_encoder 且保留完整 forward 逻辑 |
| 2026-04-12 | labels [B,T,4] + model() forward | decoder loss 按 `labels[...,codebook]` 索引；让 forward 自动处理 shift |
| 2026-04-13 | **情绪嵌入改用 T5 文本编码初始化** | 随机初始化 std≈0.05 vs T5 输出 std≈0.27，导致 cross-attention 条件信号无效。T5 初始化后 peak 从 0.002→0.124 |
| 2026-04-13 | generate 必须传 guidance_scale=1.0 | MusicGen 默认 guidance_scale=3.0，直接传 encoder_outputs 时缺少 unconditional 分支会导致 SDPA 维度错误 |

---

## 12. 代码规范

- 遵循 **PEP8**
- 代码注释使用中文
- 文件顶部必须设置 `HF_ENDPOINT` 镜像
- 情绪标签统一使用 0-3 (Q1-Q4) 整数编码
- 音频采样率统一 32000Hz
- 科研方向变更时，旧代码**搁置而非删除**（保留在 src/ 下）
- 优先使用成熟开源模型（MusicGen, MNE-Python），避免从零实现

---

## 13. 参考论文

- `mat/Shen 等 - 2024` - AI音乐治疗精神障碍综述
- `mat/Shen 等 - 2025` - AIGC闭环音乐干预增强情绪调节
- `mat/Shen 等 - Echoes of the Brain` - EEG音乐重建（潜在空间对齐+引导扩散）
