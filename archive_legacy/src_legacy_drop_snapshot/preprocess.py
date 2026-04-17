"""
预处理脚本：为每种情绪类别生成 MusicGen text_encoder 的目标嵌入
思路：用多条文本描述每种情绪 → text_encoder编码 → 取平均 → 作为训练目标
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# ========== 1. 定义每种情绪的文本描述 ==========
# EMOPIA 4类情绪：
#   1 = 高兴/激动 (高效价, 高唤醒)
#   2 = 愤怒/焦虑 (低效价, 高唤醒)
#   3 = 悲伤/忧郁 (低效价, 低唤醒)
#   4 = 平静/放松 (高效价, 低唤醒)

emotion_prompts = {
    1: [
        "a happy upbeat piano melody with fast tempo and major key",
        "a joyful energetic pop piano piece, bright and cheerful",
        "an excited lively piano music with strong rhythm, uplifting mood",
        "a bright and playful piano tune, feeling of celebration and joy",
        "an enthusiastic fast piano piece with positive energy",
        "a triumphant cheerful piano melody with dancing rhythm",
    ],
    2: [
        "a tense anxious piano piece with dissonant chords and fast tempo",
        "an angry aggressive piano music with heavy and intense sound",
        "a dramatic urgent piano piece with dark minor key and strong beats",
        "a restless agitated piano melody with chaotic rhythm",
        "a fierce intense piano music with powerful and stormy feeling",
        "a nervous suspenseful piano piece with rapid notes and tension",
    ],
    3: [
        "a sad slow melancholic piano melody in minor key",
        "a sorrowful gentle piano piece with deep emotional pain",
        "a depressing lonely piano music, very slow and heartbreaking",
        "a mournful dark piano melody with heavy sadness",
        "a gloomy slow piano piece expressing grief and loneliness",
        "a tearful emotional piano ballad with soft and sad tone",
    ],
    4: [
        "a calm relaxing ambient piano music with gentle soft notes",
        "a peaceful serene piano melody, slow and soothing",
        "a tranquil meditative piano piece with warm and comforting tone",
        "a dreamy soft piano music for relaxation and sleep",
        "a tender quiet piano melody with peaceful atmosphere",
        "a gentle lullaby-like piano piece, calm and harmonious",
    ],
}

# ========== 2. 加载模型 ==========
print("加载 MusicGen 模型...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model.eval()

# ========== 3. 提取每种情绪的文本嵌入 ==========
print("提取情绪文本嵌入...")
emotion_embeddings = {}

with torch.no_grad():
    for emo_id, prompts in emotion_prompts.items():
        embeddings_list = []
        for prompt in prompts:
            inputs = processor(text=[prompt], padding=True, return_tensors="pt")
            # 通过 text_encoder 得到隐状态
            encoder_outputs = model.text_encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
            )
            # last_hidden_state: [1, seq_len, 768]
            # 取平均池化得到一个 768 维向量
            hidden = encoder_outputs.last_hidden_state  # [1, seq_len, 768]
            mask = inputs['attention_mask'].unsqueeze(-1)  # [1, seq_len, 1]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [1, 768]
            embeddings_list.append(pooled)

        # 对同一情绪的多条描述取平均
        avg_embedding = torch.stack(embeddings_list).mean(dim=0)  # [1, 768]
        emotion_embeddings[emo_id] = avg_embedding.squeeze(0)  # [768]
        print(f"  情绪 {emo_id}: 已处理 {len(prompts)} 条描述 → 768维嵌入")

# ========== 4. 保存 ==========
save_path = os.path.join(os.path.dirname(__file__), '..', 'res', 'emotion_embeddings.pt')
torch.save(emotion_embeddings, save_path)
print(f"\n已保存到 {os.path.abspath(save_path)}")
print(f"内容: {{1: tensor[768], 2: tensor[768], 3: tensor[768], 4: tensor[768]}}")
