# AIGC-M 项目说明书：脑电-情绪-音乐闭环干预系统

## 0. 文档导航（新增）

- `CLAUDE.md`：当前主线、近期目标、关键结论与高价值踩坑
- `docs/README.md`：文档总索引
- `docs/logs/`：按周研发日志（过程记录）
- `docs/decisions/`：关键技术决策（ADR）
- `docs/experiments/`：实验台账与模板
- `docs/knowledge/`：知识库（Q&A 与可复用笔记）
- `docs/knowledge/papers/`：论文结构化笔记（原文见 `mat/`）

> 维护原则：`CLAUDE.md` 保持“短而新”，详细过程信息下沉到 `docs/`。

---

## 1. 项目概述

### 1.1 项目背景
本项目由兰州大学本科生（大二下）与导师田福泽/胡斌共同开展，目标发表于 **IEEE TAFFC（一区）**。

### 1.2 研究方向演进
| 阶段 | 方向 | 说明 |
|------|------|------|
| v1 | EEG情绪 → 匹配音乐 | 悲伤→悲伤音乐（简单映射） |
| v2 | EEG情绪 → 反向调节音乐 | 悲伤→开心音乐（情绪调节） |
| v3 | 神经靶向治疗性音乐生成 | 激活特定皮层区域，治疗抑郁等精神障碍 |
| **v4（当前）** | **连续特征条件生成 + 小规模 EEG 验证** | 冻结 MusicGen + 轻量适配器 + theta/wPLI 闭环验证 |

### 1.3 核心创新点
- **反向情绪调节**：不是情绪匹配，而是生成能改善负面情绪的音乐
- **神经靶向干预**：利用 theta 节律（4-8Hz）调制等手段，针对性激活脑区
- **闭环系统**：实时 EEG 反馈 → 动态调整音乐生成策略
- **连续特征控制**：从离散情绪标签（0-3）升级到连续 EEG 特征向量（128d+），实现细粒度音乐条件生成

### 1.4 当前阶段定位
> **Phase A（进行中）**：冻结 MusicGen 主干 + 训练轻量 Adapter，用连续 EEG 特征替换 one-hot 标签，输出可发表的对比实验结果（目标 IEEE TAFFC）。
> **Phase B（规划中）**：引入时序特征 + Transformer Adapter + Diffusion 模型探索（储备顶刊素材）。

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
│   ├── MusicGen（冻结主干 + 轻量条件模块）
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
EEG信号（同学提供）
  ↓
情绪特征提取（128d 连续向量 / theta功率 / wPLI）
  ↓
┌────────────────────────────────────────────────┐
│ 路线 A（当前）: Adapter Strategy v1            │
│   EmotionToTextAdapter(128d → [seq, 768])     │
│   → 冻结 MusicGen.generate()                  │
│   → WAV 输出                                  │
├────────────────────────────────────────────────┤
│ 路线 B（历史，已暂停）: 端到端微调            │
│   EmotionEmbedding(emotion_id → T5 init)      │
│   → 微调 MusicGen decoder                     │
│   → 音质退化，已确认不可行                    │
└────────────────────────────────────────────────┘
  ↓
音频输出 (output/adapter_generated/*.wav)
  ↓
EEG 验证实验（theta/wPLI 指标）→ 闭环反馈（未来）
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
├── adapter_strategy_v1/         # ★ 当前主路线：冻结 MusicGen + 轻量 Adapter
│   ├── README.md                # 方案概述（中文）
│   ├── models.py                # EmotionToTextAdapter 定义
│   ├── prepare_text_targets.py  # T5 编码目标嵌入（离线，一次性）
│   ├── train_adapter.py         # Adapter 训练（当前用 one-hot，待接入真实 EEG）
│   ├── generate_adapter.py      # 冻结 MusicGen + Adapter 生成
│   ├── checkpoints/             # Adapter 训练权重
│   ├── continuous_condition_training_plan.md  # 连续特征训练路线文档
│   └── eeg_validation_protocol_v1.md         # EEG 验证实验草案
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

**冻结策略（更新：优先稳定音质）**：
| 组件 | 状态 | 说明 |
|------|------|------|
| text_encoder (T5) | 完全冻结 | 仅初始化时用一次，运行时通过 encoder_outputs 绕过 |
| audio_encoder (EnCodec) | 完全冻结 | 仅预处理时使用 |
| enc_to_dec_proj | **可解冻（小学习率）** | 768→1024 投影层 |
| decoder.encoder_attn (全部24层) | **默认冻结（谨慎解冻）** | 小数据场景下全解冻易破坏预训练先验 |
| decoder.self_attn (后4层: 20-23) | **默认冻结** | 近期实验显示易导致采样崩溃为噪声 |
| decoder 其余层 | 冻结 | 保留预训练知识 |

> 备注：当前阶段不再优先“端到端大范围微调”，先走“冻结主干 + 轻量条件适配”的稳定路线。

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

### 4.3 Adapter 路线（当前主力）：`adapter_strategy_v1/`

**架构**：
```
EEG特征 (128d) → EmotionToTextAdapter (3层MLP)
  → [B, seq_len, 768]（对齐 T5 文本嵌入空间）
  → 冻结 MusicGen.generate(encoder_hidden_states=..., guidance_scale=1.0)
  → WAV 输出
```

**EmotionToTextAdapter 结构**：
```python
Linear(128 → 768) → LayerNorm → ReLU
→ Linear(768 → 1536) → ReLU
→ Linear(1536 → 768 * seq_len) → reshape [B, seq_len, 768]
```

**训练目标**：Masked MSE Loss（适配器输出 vs. T5 编码的情绪文本嵌入）

**当前已知问题**：
| 问题 | 原因 | 下一步 |
|------|------|--------|
| 生成只有"一个音节" | 训练输入为 128d one-hot（只有 4 个有效维度），信息量严重不足 | **接入真实连续 EEG 特征** |
| 输入特征与条件空间映射能力弱 | one-hot → T5 嵌入空间的映射过于简单，MLP 没有足够的信息可学 | 段级 EEG 特征（128d 连续值）+ 数据增强 |
| 尚未验证 guidance_scale>1.0 兼容性 | Adapter 输出直接作为 encoder_hidden_states，CFG 需额外处理 | 保持 guidance_scale=1.0 |

**参考文档**：
- 训练路线：`adapter_strategy_v1/continuous_condition_training_plan.md`
- EEG 验证：`adapter_strategy_v1/eeg_validation_protocol_v1.md`

### 4.4 生成（旧路线）：`src/generate_e2e.py`

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

### 4.5 情绪标签体系

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

### 7.1 当前主流程（Adapter 路线）
```bash
# 1. 激活虚拟环境
D:\pyprojects\aigc-m\venv\Scripts\Activate.ps1

# 2. 预处理：MIDI → tokens（已完成，跳过）
python src/preprocess_emopia.py

# 3. 准备 T5 目标嵌入（一次性）
python adapter_strategy_v1/prepare_text_targets.py

# 4. 训练 Adapter
python adapter_strategy_v1/train_adapter.py

# 5. 生成：4种情绪各生成一首
python adapter_strategy_v1/generate_adapter.py
```

### 7.2 旧流程（端到端微调，已暂停）
```bash
python src/train_e2e.py      # 微调 MusicGen（已确认导致音质退化）
python src/generate_e2e.py   # 从微调模型生成
```

### 7.3 预期输出
- 预处理：`res/emopia_tokens/emopia_tokens.pt`（约1071个样本）
- T5 目标嵌入：`adapter_strategy_v1/target_text_embeddings.pt`
- Adapter 权重：`adapter_strategy_v1/checkpoints/adapter_last.pt`
- 生成：`output/adapter_generated/Q1_happy_adapter.wav` 等4个文件
- （旧）微调权重：`res/musicgen_finetuned/`（已不推荐使用）

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

### 8.9 音质退化与“全噪声”问题（当前关键）
| 现象 | 根因判断 | 当前策略 |
|------|----------|----------|
| finetuned 输出几乎全噪声 | 小数据（EMOPIA）下大范围解冻导致灾难性遗忘；teacher forcing loss 下降但采样质量崩溃 | **暂停重微调**，改为冻结 MusicGen 主干，仅训练轻量条件模块 |
| loss 下降但听感不升反降 | 训练目标与主观听感不完全一致，且自回归采样误差累积 | 增加听感抽检 + CLAP/FAD 等指标，不再只看 loss |
| 预训练模型仍能生成可听音频 | 说明模型能力在，问题在微调路径而非模型本身 | 先建立 text-conditioned 高质量基线，再逐步接入 EEG 条件 |

### 8.10 输入特征信息量不足（当前核心瓶颈）
| 现象 | 根因判断 | 下一步 |
|------|----------|--------|
| Adapter 生成音频只有"一个音节"，单调无变化 | `train_adapter.py` 使用 128d one-hot 作为输入（仅 4 个有效 bit），Adapter 无法从中学到有意义的映射 | **接入真实连续 EEG 特征**（128d 浮点向量） |
| 不同情绪生成结果差异微弱 | one-hot 编码下 4 类情绪在 128d 空间中过于稀疏，MLP 难以学习细粒度差异 | 段级 EEG 特征包含丰富的频谱/时域信息，预期可显著改善 |
| 历史端到端训练也存在类似问题 | `train_e2e.py` 使用离散 `emotion_id`（0-3）索引预计算嵌入，本质也是离散输入 | 两条路线的输入问题根源相同：缺乏真实连续特征 |

**关键结论**：模型架构（MusicGen + Adapter）不是瓶颈，**输入数据**才是。当前最高优先级是与同学对接真实 EEG 特征格式并集成到训练流程中。

---

## 9. 架构演进记录

### v1：随机初始化嵌入（已废弃）
```
nn.Embedding(4, 768) → Linear → LayerNorm → GELU → Linear → expand to [B, 32, 768]
```
**问题**：随机初始化产生的向量与 T5 编码器输出分布不匹配（std 差 5 倍），cross-attention 几乎忽略条件信号。训练 30 epoch 后 loss 从 5.54 仅降到 5.39，生成全是噪声。

### v2：T5 文本编码初始化（端到端微调，已暂停）
```
T5.encode("happy energetic upbeat cheerful music...") → [1, seq_len, 768]
→ 存为 nn.Parameter, 可微调
→ 附带 attention_mask buffer 处理不等长 padding
```
**优势**：初始分布匹配、情绪自然区分、可微调。
**问题**：微调 decoder 导致灾难性遗忘，生成退化为噪声。

### v3：冻结主干 + 轻量 Adapter（当前版本）
```
EEG特征 (128d) → EmotionToTextAdapter (MLP)
→ [B, seq_len, 768]（对齐 T5 文本嵌入空间）
→ 冻结 MusicGen.generate()
```
**优势**：
1. MusicGen 权重完全不动，音质不退化
2. 只训练 <5M 参数的 MLP，不易过拟合
3. 条件嵌入对齐 T5 输出空间，MusicGen 直接"认识"
4. 输入端可灵活替换（one-hot → 连续特征 → 时序特征）

**当前局限**：输入仍为 one-hot，待接入真实 EEG 特征

---

## 10. 未来方向与路线图

### 10.1 近期目标（Phase A：连续特征 + Adapter）

**已完成**：
- [x] FluidSynth 高质量预处理（v2.5.2 已部署，1071 样本已处理）
- [x] 修复 labels 形状、shift_tokens_right、decoder_start_token_id
- [x] 诊断并修复情绪嵌入分布不匹配问题（v2 T5 初始化）
- [x] 完成 finetuned vs pretrained 听感对比，确认“重微调导致音质退化”
- [x] 新建 `adapter_strategy_v1/` 独立目录，隔离新方案与历史脚本
- [x] 跑通 Adapter pipeline：目标嵌入准备 → 适配器训练 → 生成（one-hot 版）
- [x] 输出连续特征训练路线文档：`continuous_condition_training_plan.md`
- [x] 输出 EEG 验证实验草案：`eeg_validation_protocol_v1.md`
- [x] 阅读参考论文（5 篇）并提炼可借鉴要点
- [x] 确认核心瓶颈：**输入特征信息量不足**（one-hot 4bit vs 连续 128d）

**当前阻塞项**：
- **→ [ ] 与同学对接 EEG 特征格式（pt/npy, key 名称, 维度规范）** ← 最高优先级
- [ ] 改写 `train_adapter.py`：移除 one-hot fallback，强制加载真实连续特征

**待执行**：
- [ ] 实现 `train_continuous_adapter.py`（含 L_token 重建损失）
- [ ] 首轮消融实验：one-hot baseline vs 连续段级特征
- [ ] 导出 8 条 demo 音频 + CLAP/FAD 评估表
- [ ] 实现反向情绪调节映射（输入悲伤→生成开心音乐）
- [ ] 添加 theta 节律（4-8Hz）振幅调制（后处理方式）

### 10.2 中期目标（Phase B：论文核心实验）
- [ ] 引入时序 EEG 特征：Adapter 输入从 `[B,128]` 升级到 `[B,T,128]`
- [ ] 升级 Adapter 架构：MLP → Transformer Adapter（处理时序依赖）
- [ ] 可选：探索 Diffusion 模型（AudioLDM2 + ControlNet，参考 `2405.09062v6.pdf`）
- [ ] 小规模 EEG 被试验证实验（健康被试，需伦理审批，参考 `eeg_validation_protocol_v1.md`）
- [ ] IEEE TAFFC 论文撰写

### 10.3 远期目标（闭环系统）
- [ ] 实时 EEG 采集与在线处理（MNE-Python + LSL）
- [ ] 闭环反馈控制：EEG → 情绪识别 → 音乐调整 → 效果评估
- [ ] Diffusion 模型替代 MusicGen（更高质量 + 更灵活条件控制）
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
| 2026-04-14 | **暂停 MusicGen 重微调，转向冻结主干 + 轻量条件适配** | 实测 finetuned 听感劣化为噪声，pretrained 仍具可听性；说明问题在微调策略而非模型本身 |
| 2026-04-14 | 新建 `adapter_strategy_v1` 并隔离新方案 | 当前项目脚本较混乱，采用“新目录增量开发”降低回归风险 |
| 2026-04-14 | **确认输入特征为核心瓶颈** | one-hot（128d 中仅 4 bit 有效信息）训练 Adapter 只能产生单调音频；必须接入真实连续 EEG 特征 |
| 2026-04-14 | 采纳两阶段战略（MusicGen 先 → Diffusion 后） | 当前 MusicGen + Adapter 足以产出 TAFFC 论文；Diffusion 储备为顶刊升级路线，避免分散精力 |
| 2026-04-14 | 输出三份关键文档（训练路线 / EEG 验证 / README） | 以文档驱动后续开发，确保方向明确、可复现、适合团队协作 |

---

## 12. 代码规范

- 遵循 **PEP8**
- 代码注释使用中文
- 文件顶部必须设置 `HF_ENDPOINT` 镜像
- 情绪标签统一使用 0-3 (Q1-Q4) 整数编码
- 音频采样率统一 32000Hz
- 科研方向变更时，旧代码**搁置而非删除**（迁移到 `archive_legacy/`）
- 优先使用成熟开源模型（MusicGen, MNE-Python），避免从零实现

---

## 13. 参考论文

| 论文 | 核心价值 |
|------|---------|
| `mat/Shen 等 - 2024` - AI音乐治疗精神障碍综述 | AIGC 闭环系统框架；MusicGen + EEG 情绪识别思路 |
| `mat/Shen 等 - 2025` - AIGC闭环音乐干预增强情绪调节 | 3通道 EEG 轻量情绪识别 + 扩散模型条件音乐生成 |
| `mat/Shen 等 - Echoes of the Brain` | **关键参考**：EEG-音乐对比学习对齐 + DiT ControlNet，小数据冻结主干策略 |
| `mat/2405.09062v6` - Naturalistic Music Decoding from EEG | AudioLDM2 + ControlNet 从原始 EEG 重建音乐，无需手工预处理 |
| `mat/PIIS2211124724008039` - Auditory Entrainment | theta-gamma 嵌套 + BNST-NAc 三重时间锁相机制；音乐抗抑郁与**主观享受**相关而非特定情绪 |

### 13.1 论文借鉴要点汇总

1. **冻结主干 + 轻量条件分支**：Echoes of the Brain、2405.09062v6 均采用此策略，与当前 Adapter 路线一致
2. **对比学习跨模态对齐**：EEG → CLAP 共享空间，可在 Phase B 引入
3. **theta 节律是核心指标**：4-8Hz 功率、wPLI 功能连接、theta-gamma nesting 三个层次
4. **主观享受 > 特定情绪**：PIIS2211124724008039 论证了音乐抗抑郁的关键是享受度，支持"反向调节"设计
5. **3通道 EEG 可行**：Shen 2025 证明 Fp1/Fp2/Fpz 即可完成情绪识别，降低实验门槛

---

## 14. Linux 重整记录（2026-04-17）

### 14.1 目录重整原则

- `src/` 仅保留当前主流程脚本（可直接复现实验）
- 历史代码与临时实验统一迁入 `archive_legacy/`
- 数据、输出、权重迁移到可跨系统共享盘 `/mnt/data/aigc_data`

### 14.2 当前主流程脚本

- `src/preprocess_emopia.py`
- `src/train_e2e.py`
- `src/generate_e2e.py`
- `src/diagnose.py`
- `adapter_strategy_v1/*`

### 14.3 历史归档目录

- `archive_legacy/src_legacy/`：从 `src/` 迁出的旧版主脚本
- `archive_legacy/src_legacy_drop_snapshot/`：早期 `drop/` 快照
- `archive_legacy/experiments/`：调试与验证脚本
- `archive_legacy/notebooks/`：历史 notebook/checkpoint
- `archive_legacy/misc/`：旧依赖记录与零散文件

### 14.4 Linux 路径映射

- `res -> /mnt/data/aigc_data/datasets/res`
- `output -> /mnt/data/aigc_data/outputs/output`
- `adapter_strategy_v1/checkpoints -> /mnt/data/aigc_data/checkpoints/adapter_strategy_v1/checkpoints`
- `mat -> /mnt/data/aigc_data/refs/mat`

### 14.5 环境管理

- 已从仓库内 `venv/` 迁移到 Miniconda
- 环境文件：`environment.yml`
- 推荐环境名：`aigc-m-py311`

### 14.6 执行日志（补充）

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-04-17 | 创建共享盘标准目录 `/mnt/data/aigc_data/{datasets,checkpoints,outputs,cache}` | 完成，作为 Linux/Windows 共用数据根目录 |
| 2026-04-17 | 将 `res/` 迁移到 `/mnt/data/aigc_data/datasets/res` 并在仓库根目录建立软链接 | 完成，原有脚本路径保持兼容 |
| 2026-04-17 | 将 `output/` 迁移到 `/mnt/data/aigc_data/outputs/output` 并建立软链接 | 完成，输出目录外置 |
| 2026-04-17 | 将 `adapter_strategy_v1/checkpoints/` 迁移到共享盘并建立软链接 | 完成，训练权重外置 |
| 2026-04-17 | 新增一键重整脚本 `scripts/reorganize_linux.sh` | 完成，可重复执行，支持新机器快速恢复目录结构 |
| 2026-04-17 | 删除仓库内旧 `venv/`（约 6.6G） | 完成，项目体积显著下降 |
| 2026-04-17 | 创建 Conda 环境 `aigc-m-py311` 并验证关键依赖导入 | 完成，`transformers/torch/pretty_midi/scipy` 导入正常 |
| 2026-04-17 | 将历史代码与实验脚本迁移至 `archive_legacy/` | 完成，主线目录与历史目录职责分离 |

### 14.7 结构化归档说明（用于论文“方法演进”）

- 保留“当前可复现主流程”在 `src/` 与 `adapter_strategy_v1/`，用于结果复验和论文复现实验。
- 保留“历史尝试与失败路径”在 `archive_legacy/`，用于撰写方法演进、负结果分析与决策依据。
- 保留 `src_legacy_drop_snapshot/` 原始快照，避免“回忆偏差”导致的历史版本失真。
- 目录级别完成“主线/归档”分离后，后续新增实验可先在主线开发，确认弃用后再迁入归档。
