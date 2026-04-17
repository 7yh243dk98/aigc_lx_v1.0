"""诊断：训练后的 emotion_embed 是否退化"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn

device = 'cpu'

class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions=4, hidden_dim=768, max_seq_len=32):
        super().__init__()
        self.conditioning = nn.Parameter(torch.zeros(num_emotions, max_seq_len, hidden_dim))
        self.register_buffer('attention_masks', torch.zeros(num_emotions, max_seq_len, dtype=torch.long))
    def forward(self, emotion_ids):
        return self.conditioning[emotion_ids], self.attention_masks[emotion_ids]

# 加载训练后的 emotion_embed
embed = EmotionEmbedding()
state = torch.load(r"D:\pyprojects\aigc-m\res\musicgen_finetuned\emotion_embed.pt", map_location=device)
embed.load_state_dict(state)

c = embed.conditioning.data
m = embed.attention_masks

print("=== 训练后的 emotion_embed 统计 ===")
print(f"conditioning shape: {c.shape}")
print(f"attention_masks:\n{m}")
print(f"\n全局: mean={c.mean():.4f}, std={c.std():.4f}")

for eid in range(4):
    valid = m[eid].bool()
    valid_data = c[eid][valid]
    zero_data = c[eid][~valid]
    print(f"\n情绪 {eid}:")
    print(f"  有效位置 ({valid.sum()}) mean={valid_data.mean():.4f}, std={valid_data.std():.4f}, "
          f"min={valid_data.min():.4f}, max={valid_data.max():.4f}")
    print(f"  padding位置 ({(~valid).sum()}) mean={zero_data.mean():.4f}, std={zero_data.std():.4f}")

# 各情绪之间的余弦相似度
print(f"\n=== 训练后余弦相似度 ===")
for i in range(4):
    for j in range(i+1, 4):
        vi = c[i][m[i].bool()].mean(0)
        vj = c[j][m[j].bool()].mean(0)
        cos = nn.functional.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0)).item()
        print(f"  情绪 {i} vs {j}: cos={cos:.4f}")

# 对比：初始 T5 编码统计（从 quick_test 结果已知）
print(f"\n=== 对比 T5 初始值 ===")
print(f"T5 初始: mean=-0.003, std=0.273 (有效位置)")
print(f"训练后: mean={c[0][m[0].bool()].mean():.4f}, std={c[0][m[0].bool()].std():.4f} (情绪0有效位置)")
