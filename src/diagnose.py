"""
诊断脚本：系统性定位为什么生成的音乐全是噪声
"""
import os
from pathlib import Path
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
import torch.nn as nn
import numpy as np
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile as wavfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}\n")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "diagnose"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================================================================
# 测试 1: 用原始预训练模型 + 文本条件生成（验证 pipeline 基线）
# ====================================================================
print("=" * 60)
print("测试 1: 原始预训练模型 + 文本条件")
print("=" * 60)

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
pretrained_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
pretrained_model = pretrained_model.to(device)
pretrained_model.eval()

text_prompts = [
    "happy energetic upbeat pop music",
    "sad melancholic slow piano music",
]

inputs = processor(text=text_prompts, padding=True, return_tensors="pt").to(device)

with torch.no_grad():
    audio_out = pretrained_model.generate(**inputs, max_new_tokens=256, do_sample=True)

for i, prompt in enumerate(text_prompts):
    audio = audio_out[i, 0].cpu().numpy()
    peak = np.max(np.abs(audio))
    tag = "happy" if i == 0 else "sad"
    path = os.path.join(OUTPUT_DIR, f"test1_text_{tag}.wav")
    if peak > 0:
        audio_norm = (audio / peak * 0.95 * 32767).astype(np.int16)
    else:
        audio_norm = (audio * 32767).astype(np.int16)
    wavfile.write(path, 32000, audio_norm)
    print(f"  [{tag}] peak={peak:.6f} -> {path}")

# ====================================================================
# 测试 2: 查看 T5 编码器输出的分布（作为 baseline）
# ====================================================================
print("\n" + "=" * 60)
print("测试 2: T5 编码器输出 vs 情绪嵌入分布对比")
print("=" * 60)

with torch.no_grad():
    t5_outputs = pretrained_model.text_encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
    )
    t5_hidden = t5_outputs[0]  # [B, seq_len, 768]
    print(f"  T5 输出 shape: {t5_hidden.shape}")
    print(f"  T5 输出 mean: {t5_hidden.mean().item():.6f}")
    print(f"  T5 输出 std:  {t5_hidden.std().item():.6f}")
    print(f"  T5 输出 min:  {t5_hidden.min().item():.6f}")
    print(f"  T5 输出 max:  {t5_hidden.max().item():.6f}")

    t5_projected = pretrained_model.enc_to_dec_proj(t5_hidden)
    print(f"\n  T5 经过 enc_to_dec_proj 后:")
    print(f"  shape: {t5_projected.shape}")
    print(f"  mean: {t5_projected.mean().item():.6f}")
    print(f"  std:  {t5_projected.std().item():.6f}")

# 加载我们的情绪嵌入
COND_SEQ_LEN = 32

class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions=4, hidden_dim=768, seq_len=32):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(num_emotions, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, emotion_ids):
        x = self.embed(emotion_ids)
        x = self.proj(x)
        x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        return x

embed_path = PROJECT_ROOT / "res" / "musicgen_finetuned" / "emotion_embed.pt"
emotion_embed = EmotionEmbedding().to(device)
emotion_embed.load_state_dict(torch.load(embed_path, map_location=device))
emotion_embed.eval()

with torch.no_grad():
    for eid in range(4):
        emo_out = emotion_embed(torch.tensor([eid], device=device))  # [1, 32, 768]
        print(f"\n  情绪嵌入 ID={eid} shape: {emo_out.shape}")
        print(f"  mean: {emo_out.mean().item():.6f}")
        print(f"  std:  {emo_out.std().item():.6f}")
        print(f"  min:  {emo_out.min().item():.6f}")
        print(f"  max:  {emo_out.max().item():.6f}")

    # 四个情绪嵌入之间的余弦相似度
    print(f"\n  四个情绪嵌入之间的余弦相似度:")
    all_embeds = []
    for eid in range(4):
        e = emotion_embed(torch.tensor([eid], device=device))[0, 0]  # [768]
        all_embeds.append(e)
    for i in range(4):
        for j in range(i + 1, 4):
            cos = torch.nn.functional.cosine_similarity(
                all_embeds[i].unsqueeze(0), all_embeds[j].unsqueeze(0)
            ).item()
            print(f"    情绪 {i} vs {j}: cos_sim = {cos:.4f}")

# ====================================================================
# 测试 3: 用预训练模型 + 我们的情绪嵌入（不微调）生成
# ====================================================================
print("\n" + "=" * 60)
print("测试 3: 预训练模型 + 情绪嵌入（未微调模型）")
print("=" * 60)

EMOTION_NAMES = {0: "Q1_happy", 1: "Q2_angry", 2: "Q3_sad", 3: "Q4_calm"}

with torch.no_grad():
    for eid in range(4):
        emo = emotion_embed(torch.tensor([eid], device=device))
        mask = torch.ones(1, COND_SEQ_LEN, device=device)
        audio_out = pretrained_model.generate(
            inputs=None,
            encoder_outputs=(emo,),
            attention_mask=mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )
        audio = audio_out[0, 0].cpu().numpy()
        peak = np.max(np.abs(audio))
        name = EMOTION_NAMES[eid]
        path = os.path.join(OUTPUT_DIR, f"test3_pretrained_emo_{name}.wav")
        if peak > 0:
            audio_norm = (audio / peak * 0.95 * 32767).astype(np.int16)
        else:
            audio_norm = (audio * 32767).astype(np.int16)
        wavfile.write(path, 32000, audio_norm)
        print(f"  [{name}] peak={peak:.6f}")

# ====================================================================
# 测试 4: 用 T5 编码的文本嵌入替代情绪嵌入，直接用预训练模型生成
# ====================================================================
print("\n" + "=" * 60)
print("测试 4: 预训练模型 + T5 文本编码（作为理想 encoder_outputs）")
print("=" * 60)

emotion_texts = [
    "happy energetic upbeat cheerful music",
    "angry intense aggressive powerful music",
    "sad melancholic gentle slow music",
    "calm peaceful relaxing ambient music",
]

for eid, text in enumerate(emotion_texts):
    inp = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        t5_out = pretrained_model.text_encoder(
            input_ids=inp.input_ids,
            attention_mask=inp.attention_mask,
        )
        # 直接用 encoder_outputs 传给 generate
        audio_out = pretrained_model.generate(
            inputs=None,
            encoder_outputs=t5_out,
            attention_mask=inp.attention_mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )
    audio = audio_out[0, 0].cpu().numpy()
    peak = np.max(np.abs(audio))
    name = EMOTION_NAMES[eid]
    path = os.path.join(OUTPUT_DIR, f"test4_t5text_{name}.wav")
    if peak > 0:
        audio_norm = (audio / peak * 0.95 * 32767).astype(np.int16)
    else:
        audio_norm = (audio * 32767).astype(np.int16)
    wavfile.write(path, 32000, audio_norm)
    print(f"  [{name}] text='{text}' peak={peak:.6f}")

# ====================================================================
# 测试 5: 用微调模型 + 情绪嵌入（当前训练结果）
# ====================================================================
print("\n" + "=" * 60)
print("测试 5: 微调模型 + 情绪嵌入（当前训练结果）")
print("=" * 60)

finetuned_dir = PROJECT_ROOT / "res" / "musicgen_finetuned"
if os.path.exists(os.path.join(finetuned_dir, "model.safetensors")):
    finetuned_model = MusicgenForConditionalGeneration.from_pretrained(finetuned_dir)
    finetuned_model = finetuned_model.to(device)
    finetuned_model.eval()

    with torch.no_grad():
        for eid in range(4):
            emo = emotion_embed(torch.tensor([eid], device=device))
            mask = torch.ones(1, COND_SEQ_LEN, device=device)
            audio_out = finetuned_model.generate(
                inputs=None,
                encoder_outputs=(emo,),
                attention_mask=mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )
            audio = audio_out[0, 0].cpu().numpy()
            peak = np.max(np.abs(audio))
            name = EMOTION_NAMES[eid]
            path = os.path.join(OUTPUT_DIR, f"test5_finetuned_emo_{name}.wav")
            if peak > 0:
                audio_norm = (audio / peak * 0.95 * 32767).astype(np.int16)
            else:
                audio_norm = (audio * 32767).astype(np.int16)
            wavfile.write(path, 32000, audio_norm)
            print(f"  [{name}] peak={peak:.6f}")
    del finetuned_model
else:
    print("  微调模型不存在，跳过")

# ====================================================================
# 测试 6: 检查训练数据质量
# ====================================================================
print("\n" + "=" * 60)
print("测试 6: 检查训练数据")
print("=" * 60)

data = torch.load(PROJECT_ROOT / "res" / "emopia_tokens" / "emopia_tokens.pt", weights_only=False)
print(f"  样本数: {len(data)}")

emotion_counts = {}
token_lengths = []
for item in data:
    eid = item['emotion_id']
    codes = item['audio_codes']
    emotion_counts[eid] = emotion_counts.get(eid, 0) + 1
    token_lengths.append(codes.shape[1])

print(f"  各情绪样本数: {emotion_counts}")
print(f"  token 长度: min={min(token_lengths)}, max={max(token_lengths)}, mean={np.mean(token_lengths):.1f}")

sample = data[0]
codes = sample['audio_codes']
print(f"\n  第一个样本 audio_codes shape: {codes.shape}")
print(f"  数值范围: min={codes.min().item()}, max={codes.max().item()}")
print(f"  唯一值数量: {codes.unique().numel()}")
print(f"  前5个token (codebook 0): {codes[0, :5].tolist()}")

# 检查 pad_token_id (2048) 是否出现在数据中
pad_count = (codes == 2048).sum().item()
print(f"  值为 2048 的 token 数量: {pad_count}")

print("\n诊断完成！请检查 output/diagnose/ 目录中的 wav 文件")
print("对比：test1（文本条件）应该是正常音乐，test3/test5 是情绪嵌入的效果")
