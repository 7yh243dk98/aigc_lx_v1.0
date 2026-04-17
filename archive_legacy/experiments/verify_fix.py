"""
轻量验证：T5 初始化的情绪嵌入能否生成合理音乐
只做最关键的检查，避免显存溢出
"""
import os, gc
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import numpy as np
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile as wavfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}\n")

OUTPUT_DIR = r"D:\pyprojects\aigc-m\output\verify_fix"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTION_TEXTS = [
    "happy energetic upbeat cheerful music with bright melody",
    "angry intense aggressive powerful music with strong beat",
    "sad melancholic gentle slow music with minor key",
    "calm peaceful relaxing ambient music with soft tone",
]
EMOTION_NAMES = {0: "Q1_happy", 1: "Q2_angry", 2: "Q3_sad", 3: "Q4_calm"}

# ========== 加载模型 ==========
print("加载模型...")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = model.to(device)
model.eval()

# ========== Step 1: T5 编码 + 分布检查 ==========
print("\n=== Step 1: T5 编码情绪文本 ===")
with torch.no_grad():
    inputs = processor(text=EMOTION_TEXTS, padding=True, return_tensors="pt").to(device)
    t5_out = model.text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    t5_hidden = t5_out.last_hidden_state
    print(f"T5 输出 shape: {t5_hidden.shape}")
    print(f"mean={t5_hidden.mean():.4f}, std={t5_hidden.std():.4f}")
    print(f"attention_mask: {inputs.attention_mask}")

    # 余弦相似度
    for i in range(4):
        for j in range(i + 1, 4):
            mask_i = inputs.attention_mask[i].bool()
            mask_j = inputs.attention_mask[j].bool()
            vec_i = t5_hidden[i][mask_i].mean(dim=0)
            vec_j = t5_hidden[j][mask_j].mean(dim=0)
            cos = nn.functional.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0)).item()
            print(f"  {EMOTION_NAMES[i]} vs {EMOTION_NAMES[j]}: cos={cos:.4f}")

# ========== Step 2: 逐个生成（节省显存） ==========
print("\n=== Step 2: 用 T5 encoder_outputs 逐个生成 ===")
with torch.no_grad():
    for eid in range(4):
        name = EMOTION_NAMES[eid]
        enc_out = t5_hidden[eid:eid+1]
        mask = inputs.attention_mask[eid:eid+1]

        audio_out = model.generate(
            inputs=None,
            encoder_outputs=(enc_out,),
            attention_mask=mask,
            max_new_tokens=256,
            guidance_scale=1.0,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        audio = audio_out[0, 0].cpu().numpy()
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio_norm = (audio / peak * 0.95 * 32767).astype(np.int16)
        else:
            audio_norm = (audio * 32767).astype(np.int16)
        path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        wavfile.write(path, 32000, audio_norm)
        print(f"  [{name}] raw_peak={peak:.6f} -> {path}")
        del audio_out
        torch.cuda.empty_cache()

# ========== Step 3: 快速训练 2 epoch 看 loss ==========
print("\n=== Step 3: 快速训练 2 epoch ===")

# EmotionEmbedding（与 train_e2e.py 一致）
class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions=4, hidden_dim=768, max_seq_len=32):
        super().__init__()
        self.conditioning = nn.Parameter(torch.zeros(num_emotions, max_seq_len, hidden_dim))
        self.register_buffer('attention_masks', torch.zeros(num_emotions, max_seq_len, dtype=torch.long))

    def init_from_t5(self, t5_hidden, attention_mask):
        seq_len = t5_hidden.shape[1]
        max_seq_len = self.conditioning.shape[1]
        if seq_len <= max_seq_len:
            self.conditioning.data[:, :seq_len, :] = t5_hidden.cpu()
            self.attention_masks[:, :seq_len] = attention_mask.cpu()
        else:
            self.conditioning.data = t5_hidden[:, :max_seq_len, :].cpu()
            self.attention_masks = attention_mask[:, :max_seq_len].cpu()
        print(f"  T5 init: mean={self.conditioning.mean():.4f}, std={self.conditioning.std():.4f}")

    def forward(self, emotion_ids):
        return self.conditioning[emotion_ids], self.attention_masks[emotion_ids]

emotion_embed = EmotionEmbedding().to(device)
emotion_embed.init_from_t5(t5_hidden, inputs.attention_mask)
emotion_embed = emotion_embed.to(device)

# 释放不用的变量
del t5_hidden, inputs, processor
gc.collect()
torch.cuda.empty_cache()

# 加载数据
data_path = r"D:\pyprojects\aigc-m\res\emopia_tokens\emopia_tokens.pt"
from torch.utils.data import Dataset, DataLoader

class EmopiaDataset(Dataset):
    def __init__(self, data_path, max_tokens=512):
        data = torch.load(data_path, weights_only=False)
        self.samples = []
        for item in data:
            codes = item['audio_codes']
            if codes.shape[1] > max_tokens:
                codes = codes[:, :max_tokens]
            self.samples.append({'emotion_id': item['emotion_id'], 'audio_codes': codes})
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    emotion_ids = torch.tensor([b['emotion_id'] for b in batch])
    max_len = max(b['audio_codes'].shape[1] for b in batch)
    labels = torch.full((len(batch), max_len, 4), fill_value=-100, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b['audio_codes'].shape[1]
        labels[i, :t, :] = b['audio_codes'].T
    return emotion_ids, labels

dataset = EmopiaDataset(data_path)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
print(f"  数据集: {len(dataset)} 样本")

if model.config.decoder.decoder_start_token_id is None:
    model.config.decoder.decoder_start_token_id = model.generation_config.decoder_start_token_id

for p in model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if any(k in name for k in ['encoder_attn', 'enc_to_dec_proj']):
        p.requires_grad = True
num_layers = model.decoder.config.num_hidden_layers
for name, p in model.named_parameters():
    for idx in range(num_layers - 4, num_layers):
        if f'decoder.layers.{idx}.self_attn' in name:
            p.requires_grad = True

emotion_embed.train()
model.train()

trainable = [p for p in model.parameters() if p.requires_grad] + list(emotion_embed.parameters())
optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)

for epoch in range(2):
    total_loss = 0
    count = 0
    for emotion_ids, labels in loader:
        emotion_ids = emotion_ids.to(device)
        labels = labels.to(device)
        enc_hidden, mask = emotion_embed(emotion_ids)
        outputs = model(encoder_outputs=(enc_hidden,), attention_mask=mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        count += 1
    print(f"  Epoch {epoch+1}: loss = {total_loss / count:.4f}")

print("\n验证完成! 请检查 output/verify_fix/ 下的 wav 文件")
