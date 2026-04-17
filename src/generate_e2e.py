"""
用微调后的 MusicGen 生成情绪音乐
情绪嵌入使用 T5 文本编码初始化的可学习参数
"""
import os
from pathlib import Path
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np
import torch
import torch.nn as nn
from transformers import MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile

# ========== 情绪嵌入层（与训练代码一致） ==========
class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions=4, hidden_dim=768, max_seq_len=32):
        super().__init__()
        self.conditioning = nn.Parameter(torch.zeros(num_emotions, max_seq_len, hidden_dim))
        self.register_buffer('attention_masks', torch.zeros(num_emotions, max_seq_len, dtype=torch.long))

    def forward(self, emotion_ids):
        hidden_states = self.conditioning[emotion_ids]
        masks = self.attention_masks[emotion_ids]
        return hidden_states, masks

# ========== 配置 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "res" / "musicgen_finetuned"
OUTPUT_DIR = PROJECT_ROOT / "output" / "e2e_generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTION_NAMES = {0: "Q1_happy", 1: "Q2_angry", 2: "Q3_sad", 3: "Q4_calm"}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# ========== 加载模型 ==========
print("加载微调后的 MusicGen...")
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_DIR)
model = model.to(device)
model.eval()

# 加载情绪嵌入（包含 conditioning 参数 + attention_masks buffer）
emotion_embed = EmotionEmbedding(num_emotions=4, hidden_dim=768, max_seq_len=32).to(device)
state = torch.load(os.path.join(MODEL_DIR, "emotion_embed.pt"), map_location=device)
emotion_embed.load_state_dict(state)
emotion_embed.eval()

print("模型加载完成\n")

# ========== 生成 ==========
SAMPLE_RATE = 32000
DURATION_TOKENS = 256

for emotion_id in range(4):
    emotion_name = EMOTION_NAMES[emotion_id]
    print(f"生成 {emotion_name}...")

    with torch.no_grad():
        eid = torch.tensor([emotion_id], device=device)
        encoder_hidden_states, attention_mask = emotion_embed(eid)

        audio_values = model.generate(
            inputs=None,
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=attention_mask,
            max_new_tokens=DURATION_TOKENS,
            guidance_scale=1.0,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    audio = audio_values[0, 0].cpu().numpy()
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    audio_int16 = (audio * 32767).astype(np.int16)
    output_path = os.path.join(OUTPUT_DIR, f"{emotion_name}.wav")
    wavfile.write(output_path, SAMPLE_RATE, audio_int16)
    print(f"  -> {output_path} (raw_peak={peak:.6f})")

print("\n全部生成完成！")
