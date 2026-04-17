import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration


# 你的映射网络：128维情绪特征 → 768维（替代text_encoder）
class EmotionEncoder(nn.Module):
    def __init__(self, input_dim=128, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# 加载MusicGen
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# 创建你的情绪编码器
emotion_encoder = EmotionEncoder(input_dim=128, output_dim=768)

# 模拟一个EEG情绪特征（128维，以后替换成同学的真实数据）
fake_emotion = torch.randn(1, 128)

# # 映射成768维
# emotion_hidden = emotion_encoder(fake_emotion)  # [1, 768]

# # 扩展成序列格式（decoder需要 [batch, seq_len, 768]）
# encoder_hidden_states = emotion_hidden.unsqueeze(1).repeat(1, 6, 1)  # [1, 6, 768]

# # 用情绪向量替代文本编码，直接送给decoder生成音乐
# audio_values = model.generate(
#     encoder_outputs=(encoder_hidden_states,),
#     max_new_tokens=256,
# )


# 映射成768维
emotion_hidden = emotion_encoder(fake_emotion)  # [1, 768]

# 扩展成序列格式
seq_len = 8
encoder_hidden_states = emotion_hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [1, 8, 768]
attention_mask = torch.ones(1, seq_len, dtype=torch.long)

# # 用情绪向量替代文本编码，直接送给decoder生成音乐
# audio_values = model.generate(
#     encoder_outputs=(encoder_hidden_states,),
#     attention_mask=attention_mask,
#     guidance_scale=1.0,
#     max_new_tokens=256,
# )
torch.manual_seed(0)
happy_emotion = torch.randn(1, 128) + 1.0    # 偏正值
sad_emotion = torch.randn(1, 128) - 1.0      # 偏负值
calm_emotion = torch.randn(1, 128) * 0.3     # 幅度小，平静

emotions = {"happy": happy_emotion, "sad": sad_emotion, "calm": calm_emotion}

for name, emotion in emotions.items():
    emotion_hidden = emotion_encoder(emotion)
    encoder_hidden_states = emotion_hidden.unsqueeze(1).repeat(1, seq_len, 1)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    audio_values = model.generate(
        encoder_outputs=(encoder_hidden_states,),
        attention_mask=attention_mask,
        guidance_scale=1.0,
        max_new_tokens=256,
    )

    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(f"{name}_emotion.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
    print(f"{name}_emotion.wav 已生成")

print("全部完成！")

# 保存
sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("emotion_music.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

print("用情绪向量生成音乐完成！")