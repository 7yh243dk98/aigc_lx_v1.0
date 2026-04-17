"""
训练脚本：训练 EmotionEncoder 将情绪标签映射到 MusicGen 的条件空间
然后用训练好的 EmotionEncoder 生成不同情绪的音乐

训练思路：
  情绪标签(1~4) → EmotionEncoder → 768维嵌入 → MSE Loss ← text_encoder的目标嵌入
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration


# ========== 1. EmotionEncoder 模型 ==========
class EmotionEncoder(nn.Module):
    """将情绪特征映射到 MusicGen 条件空间(768维)"""
    def __init__(self, num_emotions=4, emotion_dim=128, hidden_dim=256, output_dim=768):
        super().__init__()
        # 情绪标签 → 128维嵌入（模拟EEG特征维度）
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)
        # 128维 → 768维 映射网络
        self.mapper = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, emotion_ids):
        """
        emotion_ids: [batch] 情绪标签 (0~3)
        return: [batch, 768] 条件嵌入
        """
        x = self.emotion_embedding(emotion_ids)  # [batch, 128]
        return self.mapper(x)  # [batch, 768]


# ========== 2. 数据集 ==========
class EmotionDataset(Dataset):
    """情绪-嵌入 数据集"""
    def __init__(self, emotion_embeddings, samples_per_class=100):
        """
        emotion_embeddings: {1: tensor[768], 2: tensor[768], 3: tensor[768], 4: tensor[768]}
        samples_per_class: 每类情绪重复采样次数（数据增强）
        """
        self.data = []
        for emo_id, embedding in emotion_embeddings.items():
            for _ in range(samples_per_class):
                # 标签从0开始 (0,1,2,3)
                self.data.append((emo_id - 1, embedding))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emo_id, target_embedding = self.data[idx]
        return torch.tensor(emo_id, dtype=torch.long), target_embedding


# ========== 3. 训练 ==========
def train():
    # 加载预处理好的目标嵌入
    embed_path = os.path.join(os.path.dirname(__file__), '..', 'res', 'emotion_embeddings.pt')
    emotion_embeddings = torch.load(embed_path, weights_only=True)
    print(f"加载目标嵌入: {len(emotion_embeddings)} 类情绪")

    # 创建数据集和数据加载器
    dataset = EmotionDataset(emotion_embeddings, samples_per_class=200)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"数据集大小: {len(dataset)}")

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = EmotionEncoder(num_emotions=4, emotion_dim=128, output_dim=768).to(device)
    print(f"设备: {device}")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 训练循环
    num_epochs = 100
    print(f"\n开始训练 ({num_epochs} epochs)...")
    for epoch in range(num_epochs):
        total_loss = 0
        for emotion_ids, target_embeddings in dataloader:
            emotion_ids = emotion_ids.to(device)
            target_embeddings = target_embeddings.to(device)

            # 前向传播
            predicted = encoder(emotion_ids)
            loss = criterion(predicted, target_embeddings)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    # 保存模型
    save_path = os.path.join(os.path.dirname(__file__), '..', 'res', 'emotion_encoder.pt')
    torch.save(encoder.state_dict(), save_path)
    print(f"\n模型已保存到 {os.path.abspath(save_path)}")
    return encoder


# ========== 4. 用训练好的模型生成音乐 ==========
def generate_music(encoder):
    print("\n加载 MusicGen 生成音乐...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model.eval()

    device = next(encoder.parameters()).device
    encoder.eval()

    emotion_names = {0: "happy", 1: "angry", 2: "sad", 3: "calm"}
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'trained')
    os.makedirs(output_dir, exist_ok=True)

    for emo_id, name in emotion_names.items():
        with torch.no_grad():
            # EmotionEncoder 生成条件嵌入
            emotion_input = torch.tensor([emo_id], dtype=torch.long).to(device)
            condition = encoder(emotion_input)  # [1, 768]

            # 扩展为序列格式
            seq_len = 8
            encoder_hidden_states = condition.unsqueeze(1).repeat(1, seq_len, 1).cpu()
            attention_mask = torch.ones(1, seq_len, dtype=torch.long)

            # MusicGen 生成
            audio_values = model.generate(
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=attention_mask,
                guidance_scale=1.0,
                max_new_tokens=512,
            )

            # 保存
            sampling_rate = model.config.audio_encoder.sampling_rate
            filepath = os.path.join(output_dir, f"{name}_trained.wav")
            scipy.io.wavfile.write(filepath, rate=sampling_rate, data=audio_values[0, 0].numpy())
            print(f"  已生成: {filepath}")

    print("\n全部生成完成！")


# ========== 主入口 ==========
if __name__ == "__main__":
    encoder = train()
    generate_music(encoder)
