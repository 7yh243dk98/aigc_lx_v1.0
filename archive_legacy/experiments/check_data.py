import torch
from collections import Counter

d = torch.load(r'D:\pyprojects\aigc-m\res\emopia_tokens\emopia_tokens.pt', weights_only=False)
print(f"样本数: {len(d)}")
s = d[0]['audio_codes'].shape
print(f"第一个audio_codes shape: {s}")
emotions = [x['emotion_id'] for x in d]
print(f"情绪分布: {dict(Counter(emotions))}")

# 检查微调模型权重是否存在
import os
fdir = r'D:\pyprojects\aigc-m\res\musicgen_finetuned'
for f in os.listdir(fdir):
    sz = os.path.getsize(os.path.join(fdir, f))
    print(f"  {f}: {sz/1024:.1f} KB")
