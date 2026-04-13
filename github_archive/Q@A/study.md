# 🚀 寒假5周深度学习冲刺计划

> **目标**：快速掌握深度学习核心知识，能够理解和开始实践AIGC音乐治疗研究
> 
> **时间**：5周寒假
> 
> **学习原则**：理论+实践并重，以应用为导向，边学边做

---

## 📅 整体时间规划

| 周次 | 主题 | 核心目标 | 验收标准 |
|------|------|----------|----------|
| Week 1 | 深度学习基础 | 理解神经网络原理，掌握PyTorch | 能训练一个图像分类模型 |
| Week 2 | 生成模型入门 | 掌握VAE和GAN基础 | 运行并理解VAE生成图像 |
| Week 3 | Diffusion扩散模型 | 理解扩散模型原理（核心） | 运行Stable Diffusion并微调 |
| Week 4 | 音频处理+多模态 | 音频特征提取、跨模态学习 | 完成音频分类小项目 |
| Week 5 | 论文阅读+实践 | 阅读导师论文，开始实验 | 复现论文中的某个实验 |

---

## 📚 Week 1: 深度学习基础（打地基）

### 🎯 本周目标
- 理解神经网络的前向传播和反向传播
- 掌握PyTorch基本操作
- 能独立训练一个简单模型

### 📖 学习内容

#### Day 1-2: 神经网络基础理论
**学习资源：**
- [ ] 3Blue1Brown - 《神经网络》系列视频（4集，中文字幕）
  - 链接：https://www.bilibili.com/video/BV1bx411M7Zx
  - 时长：约1小时
  - **任务**：看完后手写一个感知器代码

- [ ] 吴恩达《深度学习专项课程》- 第1课 Week1-2
  - 平台：Coursera（可免费旁听）
  - **任务**：完成编程作业

**关键概念：**
- 激活函数（ReLU、Sigmoid、Softmax）
- 损失函数（交叉熵、MSE）
- 梯度下降和反向传播
- 过拟合与正则化（Dropout、L2）

#### Day 3-4: PyTorch实战入门
**学习资源：**
- [ ] PyTorch官方教程 - 《60分钟闪电战》
  - 链接：https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
  - **任务**：跑通所有示例代码

- [ ] 动手实践项目：MNIST手写数字识别
  ```python
  # 要实现的内容：
  - 数据加载（torchvision.datasets）
  - 定义CNN模型
  - 训练循环（train loop）
  - 评估和可视化
  ```

**PyTorch核心概念：**
- Tensor操作和自动微分
- Dataset和DataLoader
- nn.Module模型定义
- 优化器（torch.optim）

#### Day 5-6: 进阶实践 + CNN卷积网络
**学习资源：**
- [ ] 李沐《动手学深度学习》- 卷积神经网络章节
  - 链接：https://zh.d2l.ai/chapter_convolutional-neural-networks/
  - **任务**：理解卷积、池化操作

**实践项目：图像分类**
- [ ] 使用CIFAR-10数据集训练ResNet
- [ ] 学习使用预训练模型（torchvision.models）
- [ ] 实验不同的数据增强技术

#### Day 7: 周总结 + 小项目
**综合项目：猫狗分类器**
- [ ] 从Kaggle下载猫狗数据集
- [ ] 使用迁移学习（预训练ResNet/EfficientNet）
- [ ] 达到90%+准确率
- [ ] 保存模型并测试

**本周检查清单：**
- [ ] 能解释神经网络的工作原理
- [ ] 熟练使用PyTorch定义和训练模型
- [ ] 理解卷积神经网络的原理
- [ ] 完成至少2个实践项目

---

## 🎨 Week 2: 生成模型入门（VAE + GAN）

### 🎯 本周目标
- 理解生成模型与判别模型的区别
- 掌握VAE（变分自编码器）原理和实现
- 了解GAN基本概念

### 📖 学习内容

#### Day 1-2: 自编码器（AutoEncoder）
**学习资源：**
- [ ] 李沐视频 - 《自编码器》
  - **任务**：实现基础AutoEncoder

**实践项目：**
```python
# 实现内容：
- 基础AutoEncoder（重建MNIST）
- 理解编码器-解码器结构
- 可视化潜在空间（latent space）
```

**关键概念：**
- 编码器和解码器
- 重建损失
- 潜在空间表示（后续很重要！）

#### Day 3-4: VAE（变分自编码器）⭐
**学习资源：**
- [ ] 论文精读：《Auto-Encoding Variational Bayes》
  - 可以看Lil'Log的博客解读
  - 链接：https://lilianweng.github.io/posts/2018-08-12-vae/

- [ ] 代码实现：PyTorch VAE教程
  - GitHub搜索：pytorch-vae

**实践项目：**
- [ ] 实现VAE生成手写数字
- [ ] 理解重参数化技巧（reparameterization trick）
- [ ] 在潜在空间中插值生成新图像

**核心理解：**
- KL散度损失的作用
- 潜在空间的连续性（为什么重要）
- VAE在音乐生成中的应用（MusicVAE）

#### Day 5-6: GAN（生成对抗网络）
**学习资源：**
- [ ] Ian Goodfellow的GAN论文（快速浏览）
- [ ] 李沐视频 - 《生成对抗网络》
- [ ] PyTorch GAN教程

**实践项目：**
- [ ] 实现DCGAN生成人脸（CelebA数据集）
- [ ] 理解生成器和判别器的对抗训练
- [ ] 观察训练过程中的模式崩溃问题

**关键概念：**
- 对抗损失（Adversarial Loss）
- 生成器和判别器的平衡
- GAN的训练技巧和常见问题

#### Day 7: 生成模型对比 + 音乐应用
**学习任务：**
- [ ] 阅读综述：生成模型对比（VAE vs GAN vs Diffusion）
- [ ] 了解音乐生成模型：
  - MusicVAE（Google Magenta）
  - MuseGAN
  - Jukebox（OpenAI）

**实践：**
- [ ] 运行MusicVAE生成MIDI
  - GitHub: magenta/magenta
  - Colab Demo体验

**本周检查清单：**
- [ ] 理解VAE的原理和潜在空间概念
- [ ] 能实现并训练VAE/GAN
- [ ] 了解生成模型在音乐中的应用
- [ ] 完成至少1个生成模型项目

---

## 🌊 Week 3: Diffusion扩散模型（重中之重）⭐⭐⭐

### 🎯 本周目标
- 深入理解扩散模型原理（你论文的核心技术）
- 运行Stable Diffusion并理解其架构
- 掌握引导扩散（Guided Diffusion）概念

### 📖 学习内容

#### Day 1-2: 扩散模型理论基础
**学习资源：**
- [ ] Lil'Log博客 - 《What are Diffusion Models?》
  - 链接：https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
  - 中文翻译版也有

- [ ] 论文阅读：DDPM（Denoising Diffusion Probabilistic Models）
  - 重点看：前向扩散过程、反向去噪过程
  - 公式不用全懂，理解核心思想

**核心概念：**
- 前向扩散过程（逐步加噪）
- 反向去噪过程（逐步去噪生成）
- 马尔可夫链和条件概率
- 噪声调度（noise schedule）

**可视化理解：**
- [ ] 观看视频：扩散模型动画演示
- [ ] 自己画图理解前向/反向过程

#### Day 3-4: DDPM实现与实践
**学习资源：**
- [ ] Hugging Face Diffusers教程
  - 链接：https://huggingface.co/docs/diffusers/
  - **任务**：跑通所有基础教程

**实践项目：**
```python
# 实现内容：
- 使用Diffusers库训练简单扩散模型
- 数据集：CIFAR-10或自定义小数据集
- 理解UNet架构在扩散模型中的作用
- 实现采样过程（DDPM/DDIM）
```

**关键代码：**
- DDPMScheduler使用
- UNet2DModel定义
- 训练循环实现

#### Day 5-6: Stable Diffusion与引导机制⭐
**学习资源：**
- [ ] Stable Diffusion论文和博客
- [ ] 理解Latent Diffusion（潜在扩散）
- [ ] Classifier-Free Guidance原理

**实践项目：**
- [ ] 本地运行Stable Diffusion
  - 使用Diffusers库
  - 尝试文本生成图像
  
- [ ] **重点**：理解引导扩散
  - Classifier Guidance
  - Classifier-Free Guidance
  - 这是你第三篇论文的核心！

**代码实践：**
```python
# 要实现的：
- 加载预训练Stable Diffusion模型
- 修改guidance_scale参数观察效果
- 理解CLIP如何引导生成过程
- 尝试图像编辑（img2img）
```

#### Day 7: 扩散模型在音频/音乐中的应用
**学习任务：**
- [ ] 阅读音频扩散模型论文：
  - DiffWave（音频生成）
  - Riffusion（音乐生成）
  - AudioLDM（音频潜在扩散）

- [ ] 运行Riffusion Demo
  - 网站：https://www.riffusion.com/
  - GitHub代码尝试

**思考问题：**
- 图像扩散如何迁移到音频？
- 频谱图（Spectrogram）作为中间表示
- 如何将扩散模型用于你的研究？

**本周检查清单：**
- [ ] 深入理解扩散模型的前向和反向过程
- [ ] 能使用Diffusers库训练和推理
- [ ] 理解引导扩散机制（论文关键）
- [ ] 了解扩散模型在音频领域的应用
- [ ] 运行至少1个扩散模型项目

---

## 🎵 Week 4: 音频处理 + 多模态学习

### 🎯 本周目标
- 掌握音频信号处理基础
- 理解音乐特征提取和表示
- 了解多模态学习（EEG-音乐对齐）

### 📖 学习内容

#### Day 1-2: 音频信号处理基础
**学习资源：**
- [ ] librosa官方教程
  - 链接：https://librosa.org/doc/latest/tutorial.html
  - **任务**：跑通所有示例

**核心概念：**
- 采样率、比特率
- 时域和频域表示
- 傅里叶变换（FFT）
- 短时傅里叶变换（STFT）
- 梅尔频谱（Mel Spectrogram）
- MFCC特征

**实践项目：**
```python
# 实现内容：
- 加载音频文件（librosa.load）
- 可视化波形和频谱图
- 提取音频特征（MFCC、Chroma、Tempo）
- 音频数据增强（时间拉伸、音调变换）
```

#### Day 3-4: 音乐信息检索与表示
**学习资源：**
- [ ] MIDI处理：pretty_midi库
- [ ] 音乐理论基础（快速了解）
  - 音高、节奏、和声、音色

**实践项目：音频分类**
- [ ] 使用GTZAN数据集（音乐流派分类）
- [ ] 提取梅尔频谱特征
- [ ] 训练CNN分类器
- [ ] 达到80%+准确率

**MIDI处理实践：**
```python
# 实现内容：
- 读取MIDI文件（pretty_midi）
- 提取音符序列
- 可视化钢琴卷帘（Piano Roll）
- 将MIDI转换为音频
```

#### Day 5-6: 多模态学习基础⭐
**学习资源：**
- [ ] CLIP论文和代码
  - 理解图像-文本对齐
  - 对比学习（Contrastive Learning）

- [ ] 多模态表征学习综述
  - 跨模态检索
  - 模态对齐和融合

**核心概念：**
- 对比学习损失（InfoNCE Loss）
- 共享嵌入空间（Shared Embedding）
- 跨模态注意力机制
- **EEG-音乐对齐**（你论文的核心）

**实践项目：**
- [ ] 运行CLIP示例（图像-文本检索）
- [ ] 理解如何将CLIP范式应用到EEG-音乐

#### Day 7: EEG信号处理入门
**学习资源：**
- [ ] MNE-Python库教程
  - 链接：https://mne.tools/stable/auto_tutorials/index.html
  - **任务**：了解基本操作

**EEG基础概念：**
- 脑电波频段（Delta、Theta、Alpha、Beta、Gamma）
- EEG信号预处理（滤波、去伪迹）
- 时频分析
- 事件相关电位（ERP）

**实践：**
- [ ] 加载示例EEG数据
- [ ] 可视化EEG信号
- [ ] 提取频域特征

**思考问题：**
- 如何从EEG信号提取情绪特征？
- EEG特征如何映射到音乐特征？
- 潜在空间对齐的具体实现？

**本周检查清单：**
- [ ] 掌握音频加载、特征提取、可视化
- [ ] 完成音频分类项目
- [ ] 理解MIDI表示和处理
- [ ] 理解多模态学习和对齐机制
- [ ] 了解EEG信号基础

---

## 📄 Week 5: 论文阅读 + 研究实践

### 🎯 本周目标
- 深入阅读导师的三篇论文
- 尝试复现论文中的部分实验
- 规划后续研究方向

### 📖 学习内容

#### Day 1-2: 精读第一篇论文（2024综述）
**论文：** *A First Look at Generative Artificial Intelligence Based Music Therapy for Mental Disorders*

**阅读策略：**
1. **第一遍（2小时）**：
   - [ ] 阅读摘要、引言、结论
   - [ ] 看所有图表和架构图
   - [ ] 列出不懂的概念和技术

2. **第二遍（3小时）**：
   - [ ] 详读方法部分
   - [ ] 理解系统架构
   - [ ] 记录关键技术点

3. **第三遍（2小时）**：
   - [ ] 对比相关工作
   - [ ] 分析实验设计
   - [ ] 思考改进方向

**输出成果：**
- [ ] 论文笔记（1-2页）
- [ ] 技术路线图
- [ ] 关键词列表

#### Day 3-4: 精读第二篇论文（2025闭环系统）
**论文：** *Enhancing Emotion Regulation: An AIGC-Based Closed-Loop Music Intervention*

**重点关注：**
- [ ] 闭环系统设计
- [ ] 情绪识别模块
- [ ] 音乐生成模块
- [ ] 反馈机制设计
- [ ] 个性化策略

**实践任务：**
- [ ] 画出系统架构图
- [ ] 列出需要的数据集
- [ ] 识别可以改进的模块

#### Day 5-6: 精读第三篇论文（EEG音乐重建）⭐
**论文：** *Echoes of the Brain: Reconstructing Music from EEG via Latent Space Alignment*

**重点关注：**
- [ ] EEG特征提取方法
- [ ] 潜在空间对齐机制（核心）
- [ ] 引导扩散的具体实现
- [ ] 评估指标和实验设计

**实践任务：**
- [ ] 寻找论文相关的开源代码
- [ ] 下载相关数据集（如果有）
- [ ] 尝试复现特征提取部分

**关键问题思考：**
- [ ] 如何将EEG信号映射到音乐潜在空间？
- [ ] 引导扩散如何整合EEG信息？
- [ ] 如何评估重建音乐的质量？

#### Day 7: 综合实践 + 规划
**任务1：复现实验**
选择以下之一尝试复现：
- [ ] 音频特征提取和可视化
- [ ] 简单的音乐生成模型
- [ ] EEG信号处理pipeline

**任务2：技术调研**
- [ ] 搜索相关GitHub项目
- [ ] 整理可用的数据集：
  - 情绪音乐数据集
  - EEG-音乐配对数据
  - 精神健康相关数据

**任务3：研究规划**
- [ ] 列出可以改进的方向
- [ ] 制定下学期研究计划
- [ ] 准备与导师讨论的问题清单

**本周检查清单：**
- [ ] 深入理解三篇论文的核心思想
- [ ] 完成论文笔记和技术总结
- [ ] 尝试复现部分实验
- [ ] 明确后续研究方向

---

## 🛠️ 开发环境配置

### 必装软件和库

#### Python环境（推荐3.9-3.11）
```bash
# 创建虚拟环境
conda create -n aigc-music python=3.10
conda activate aigc-music

# 深度学习框架
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 音频处理
pip install librosa soundfile pretty_midi music21
pip install pydub ffmpeg-python

# 扩散模型
pip install diffusers transformers accelerate

# EEG处理
pip install mne

# 其他工具
pip install numpy pandas matplotlib seaborn
pip install jupyter notebook
pip install tensorboard wandb  # 实验管理
```

#### 硬件需求
- **GPU**：强烈推荐（NVIDIA，至少6GB显存）
  - 没有GPU可以用Google Colab（免费）
- **内存**：16GB+推荐
- **硬盘**：至少50GB空闲空间（数据集+模型）

---

## 📊 每周学习时间分配建议

| 时间段 | 活动 | 时长 |
|--------|------|------|
| 上午 9:00-12:00 | 理论学习（视频/论文） | 3小时 |
| 下午 14:00-17:00 | 代码实践 | 3小时 |
| 晚上 19:00-21:00 | 项目实战/复习 | 2小时 |
| **每周总计** | | **35-40小时** |

### 学习建议
- ⏰ **番茄工作法**：25分钟专注 + 5分钟休息
- 📝 **做笔记**：用Notion/Obsidian记录关键知识点
- 🤝 **找学习伙伴**：互相督促和讨论
- 🎯 **每日总结**：记录今天学到了什么、遇到什么问题

---

## 📚 核心学习资源汇总

### 在线课程
1. **吴恩达Deep Learning专项** ⭐⭐⭐
   - 平台：Coursera
   - 免费旁听，付费证书

2. **李沐《动手学深度学习》** ⭐⭐⭐
   - 网站：https://zh.d2l.ai
   - 配套视频：B站搜索"李沐"

3. **Fast.ai Practical Deep Learning** ⭐⭐
   - 网站：https://course.fast.ai
   - 适合快速上手

### 博客和文章
- **Lil'Log** - 高质量技术博客（生成模型必看）
- **Distill.pub** - 可视化深度学习
- **Hugging Face博客** - 最新技术动态

### GitHub资源
```
# 必看项目
- pytorch/examples - PyTorch官方例子
- huggingface/diffusers - 扩散模型库
- magenta/magenta - 音乐生成（Google）
- microsoft/muzic - 音乐AI工具集
```

### 论文阅读
- **Papers with Code** - 找代码实现
- **Connected Papers** - 可视化论文关系
- **Arxiv Sanity** - 论文推荐

---

## ✅ 里程碑检查点

### Week 1结束
- [ ] 能独立训练图像分类模型
- [ ] PyTorch代码流畅度80%+

### Week 2结束
- [ ] 实现VAE生成图像
- [ ] 理解潜在空间概念

### Week 3结束 ⭐
- [ ] 运行Stable Diffusion
- [ ] 理解扩散模型原理
- [ ] 了解引导扩散机制

### Week 4结束
- [ ] 完成音频分类项目
- [ ] 掌握音频特征提取
- [ ] 理解多模态学习

### Week 5结束 🎯
- [ ] 读完三篇论文
- [ ] 复现部分实验
- [ ] 明确研究方向

---

## 🎯 5周后你将具备的能力

### 技术能力
✅ 熟练使用PyTorch开发深度学习模型
✅ 理解并实现生成模型（VAE、GAN、Diffusion）
✅ 掌握音频信号处理和特征提取
✅ 了解多模态学习和跨模态对齐
✅ 能阅读和理解顶会论文

### 研究能力
✅ 理解导师论文的核心技术
✅ 能复现简单实验
✅ 具备改进和创新的基础
✅ 能独立调研相关工作

### 实践能力
✅ 能快速搭建实验环境
✅ 会使用开源工具和预训练模型
✅ 具备调试和优化模型的经验
✅ 能管理实验和记录结果

---

## 💡 学习Tips和常见问题

### 遇到困难怎么办？
1. **看不懂公式**：先理解直觉，公式可以慢慢啃
2. **代码报错**：Google + Stack Overflow + ChatGPT
3. **训练不收敛**：检查学习率、数据预处理、模型架构
4. **进度慢**：降低标准，先跑通再优化

### 保持动力的方法
- 🎮 把学习游戏化（完成任务打勾很爽）
- 🏆 设置小奖励（完成周目标奖励自己）
- 📢 分享进度（朋友圈/小红书记录）
- 🤝 加入学习社区（GitHub、Reddit、知乎）

### 时间管理
- 早上精力好 → 学理论
- 下午适合 → 写代码
- 晚上复盘 → 整理笔记
- **每周留1天休息**，避免burnout

---

## 🚀 开始行动

### 今天就开始（Day 0）
- [ ] 配置开发环境（Python + PyTorch）
- [ ] 注册必要账号（GitHub、Hugging Face、Colab）
- [ ] 下载第一周的学习资源
- [ ] 跑通一个"Hello World"级别的PyTorch代码

### 第一个小目标
**Week 1 Day 1**：今天看完3Blue1Brown的神经网络视频，手写一个感知器

---

## 📞 需要帮助？

在学习过程中遇到任何问题，可以：
1. **查文档**：PyTorch、Hugging Face官方文档
2. **搜索**：Google、Stack Overflow
3. **问我**：把代码和错误信息发给我
4. **社区**：PyTorch论坛、Reddit r/MachineLearning

---

## 🎉 寒假结束时的成果展示

准备以下内容向导师汇报：
1. **学习笔记和总结**（可以是博客/Notion）
2. **代码仓库**（GitHub，包含实践项目）
3. **论文阅读报告**（三篇论文的理解和思考）
4. **初步实验结果**（复现的部分）
5. **下学期研究计划**（具体可行的目标）

---

**💪 5周时间，你完全可以入门深度学习并开始研究工作！**

**记住：行动 > 完美，完成 > 完美，持续 > 爆发**

**开始你的深度学习之旅吧！🚀**
