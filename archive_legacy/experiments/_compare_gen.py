"""对比：预训练模型 vs 微调模型，用同一个 emotion_embed 生成"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch, gc
import torch.nn as nn
import numpy as np
from transformers import MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions=4, hidden_dim=768, max_seq_len=32):
        super().__init__()
        self.conditioning = nn.Parameter(torch.zeros(num_emotions, max_seq_len, hidden_dim))
        self.register_buffer('attention_masks', torch.zeros(num_emotions, max_seq_len, dtype=torch.long))
    def forward(self, emotion_ids):
        return self.conditioning[emotion_ids], self.attention_masks[emotion_ids]

embed = EmotionEmbedding().to(device)
state = torch.load(r"D:\pyprojects\aigc-m\res\musicgen_finetuned\emotion_embed.pt", map_location=device)
embed.load_state_dict(state)
embed.eval()

OUT = r"D:\pyprojects\aigc-m\output\compare"
os.makedirs(OUT, exist_ok=True)

def gen_one(model, embed, eid, tag):
    with torch.no_grad():
        h, m = embed(torch.tensor([eid], device=device))
        audio = model.generate(
            inputs=None, encoder_outputs=(h,), attention_mask=m,
            max_new_tokens=256, guidance_scale=1.0, do_sample=True, temperature=0.7,
        )
    a = audio[0, 0].cpu().numpy()
    peak = np.max(np.abs(a))
    if peak > 0:
        a = (a / peak * 0.95 * 32767).astype(np.int16)
    else:
        a = (a * 32767).astype(np.int16)
    path = os.path.join(OUT, f"{tag}_Q{eid+1}.wav")
    wavfile.write(path, 32000, a)
    return peak

# Test 1: 预训练模型
print("=== 预训练模型 + 训练后的 emotion_embed ===")
model_pre = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device).eval()
for eid in range(4):
    p = gen_one(model_pre, embed, eid, "pretrained")
    print(f"  Q{eid+1} peak={p:.6f}")
del model_pre; gc.collect(); torch.cuda.empty_cache()

# Test 2: 微调模型
print("\n=== 微调模型 + 训练后的 emotion_embed ===")
model_ft = MusicgenForConditionalGeneration.from_pretrained(r"D:\pyprojects\aigc-m\res\musicgen_finetuned").to(device).eval()
for eid in range(4):
    p = gen_one(model_ft, embed, eid, "finetuned")
    print(f"  Q{eid+1} peak={p:.6f}")

print("\n完成! 检查 output/compare/")
