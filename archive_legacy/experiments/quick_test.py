"""最小测试：只加载模型 + T5 编码，不做生成"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

print("加载模型...")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = model.to(device)
model.eval()
print("模型加载完成")

EMOTION_TEXTS = [
    "happy energetic upbeat cheerful music with bright melody",
    "angry intense aggressive powerful music with strong beat",
    "sad melancholic gentle slow music with minor key",
    "calm peaceful relaxing ambient music with soft tone",
]
NAMES = ["Q1_happy", "Q2_angry", "Q3_sad", "Q4_calm"]

with torch.no_grad():
    inputs = processor(text=EMOTION_TEXTS, padding=True, return_tensors="pt").to(device)
    t5_out = model.text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    h = t5_out.last_hidden_state
    print(f"\nT5 shape: {h.shape}, mean={h.mean():.4f}, std={h.std():.4f}")
    print(f"attention_mask:\n{inputs.attention_mask}")

    # 余弦相似度
    import torch.nn.functional as F
    for i in range(4):
        for j in range(i+1, 4):
            mi = inputs.attention_mask[i].bool()
            mj = inputs.attention_mask[j].bool()
            vi = h[i][mi].mean(0)
            vj = h[j][mj].mean(0)
            cos = F.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0)).item()
            print(f"  {NAMES[i]} vs {NAMES[j]}: cos={cos:.4f}")

print("\n测试完成!")
