"""
预处理脚本 v2：保留 text_encoder 的完整序列输出（不做池化）
每种情绪生成多条文本描述 → text_encoder编码 → 保留完整序列 → 作为训练目标
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# ========== 1. 定义每种情绪的文本描述 ==========
emotion_prompts = {
    1: [
        "a happy upbeat piano melody with fast tempo and major key",
        "a joyful energetic pop piano piece, bright and cheerful",
        "an excited lively piano music with strong rhythm, uplifting mood",
        "a bright and playful piano tune, feeling of celebration and joy",
        "an enthusiastic fast piano piece with positive energy",
        "a triumphant cheerful piano melody with dancing rhythm",
        "a fun bouncy piano song with happy vibes and quick notes",
        "a radiant optimistic piano melody full of warmth and delight",
    ],
    2: [
        "a tense anxious piano piece with dissonant chords and fast tempo",
        "an angry aggressive piano music with heavy and intense sound",
        "a dramatic urgent piano piece with dark minor key and strong beats",
        "a restless agitated piano melody with chaotic rhythm",
        "a fierce intense piano music with powerful and stormy feeling",
        "a nervous suspenseful piano piece with rapid notes and tension",
        "a furious dark piano music with pounding heavy chords",
        "a threatening ominous piano piece with aggressive fast passages",
    ],
    3: [
        "a sad slow melancholic piano melody in minor key",
        "a sorrowful gentle piano piece with deep emotional pain",
        "a depressing lonely piano music, very slow and heartbreaking",
        "a mournful dark piano melody with heavy sadness",
        "a gloomy slow piano piece expressing grief and loneliness",
        "a tearful emotional piano ballad with soft and sad tone",
        "a somber quiet piano melody filled with regret and sorrow",
        "a weeping delicate piano piece with fragile and broken melody",
    ],
    4: [
        "a calm relaxing ambient piano music with gentle soft notes",
        "a peaceful serene piano melody, slow and soothing",
        "a tranquil meditative piano piece with warm and comforting tone",
        "a dreamy soft piano music for relaxation and sleep",
        "a tender quiet piano melody with peaceful atmosphere",
        "a gentle lullaby-like piano piece, calm and harmonious",
        "a serene flowing piano melody like a quiet stream",
        "a warm cozy piano piece with slow graceful notes and comfort",
    ],
}

# ========== 2. 加载模型 ==========
print("加载 MusicGen 模型...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model.eval()

# ========== 3. 提取完整序列嵌入 ==========
print("提取情绪文本嵌入（完整序列）...")

# 统一 seq_len：找所有prompt中最长的
all_prompts = [p for prompts in emotion_prompts.values() for p in prompts]
max_seq_len = 0
for prompt in all_prompts:
    tokens = processor(text=[prompt], return_tensors="pt")
    max_seq_len = max(max_seq_len, tokens['input_ids'].shape[1])
print(f"  最大序列长度: {max_seq_len}")

# 每条prompt单独编码，保留完整序列
all_data = []  # [(emotion_id, hidden_states)]

with torch.no_grad():
    for emo_id, prompts in emotion_prompts.items():
        for prompt in prompts:
            inputs = processor(
                text=[prompt],
                padding='max_length',
                max_length=max_seq_len,
                return_tensors="pt",
            )

            encoder_outputs = model.text_encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
            )

            # 保留完整序列: [1, seq_len, 768] → [seq_len, 768]
            hidden = encoder_outputs.last_hidden_state.squeeze(0)  # [seq_len, 768]
            mask = inputs['attention_mask'].squeeze(0)  # [seq_len]

            all_data.append({
                'emotion_id': emo_id - 1,  # 0~3
                'hidden_states': hidden,    # [seq_len, 768]
                'attention_mask': mask,     # [seq_len]
            })

        print(f"  情绪 {emo_id}: {len(prompts)} 条 → 序列嵌入 [{max_seq_len}, 768]")

# ========== 4. 保存 ==========
save_path = os.path.join(os.path.dirname(__file__), '..', 'res', 'emotion_seq_embeddings.pt')
torch.save({
    'data': all_data,
    'seq_len': max_seq_len,
    'hidden_dim': 768,
}, save_path)
print(f"\n已保存到 {os.path.abspath(save_path)}")
print(f"共 {len(all_data)} 条样本，序列长度 {max_seq_len}，隐藏维度 768")
