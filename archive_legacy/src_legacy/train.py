"""
训练脚本 v2：用完整序列嵌入训练 EmotionEncoder
EmotionEncoder 输出完整序列 [seq_len, 768]，而不是单向量重复

训练思路：
  情绪标签 → Embedding(128维) → Transformer → 序列嵌入 [seq_len, 768]
  Loss = MSE(预测序列, text_encoder完整序列输出)
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration


# ========== 1. EmotionEncoder v2 ==========
class EmotionEncoder(nn.Module):
    """
    情绪标签 → 完整条件序列 [seq_len, 768]
    架构：Embedding → 线性投影 → Transformer → 输出序列
    """
    def __init__(self, num_emotions=4, emotion_dim=128, hidden_dim=768,
                 seq_len=16, nhead=8, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # 情绪 → 128维嵌入（和EEG特征维度对齐）
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)

        # 128维 → 768维 投影
        self.input_proj = nn.Linear(emotion_dim, hidden_dim)

        # 可学习的位置编码（让每个位置生成不同的内容）
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        # Transformer 编码器（核心：让序列中每个位置互相交互）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, emotion_ids):
        """
        emotion_ids: [batch] 情绪标签 (0~3)
        return: [batch, seq_len, 768] 条件序列
        """
        batch_size = emotion_ids.shape[0]

        # 情绪嵌入 → 投影到768维
        x = self.emotion_embedding(emotion_ids)     # [batch, 128]
        x = self.input_proj(x)                      # [batch, 768]

        # 扩展为序列 + 加位置编码（每个位置不同）
        x = x.unsqueeze(1).expand(-1, self.seq_len, -1)  # [batch, seq_len, 768]
        x = x + self.pos_embedding                        # 加上位置信息

        # Transformer 让序列内部交互
        x = self.transformer(x)    # [batch, seq_len, 768]
        x = self.norm(x)

        return x


# ========== 2. 数据集 ==========
class EmotionSeqDataset(Dataset):
    """情绪-序列嵌入 数据集"""
    def __init__(self, data_list, augment_copies=10):
        """
        data_list: [{'emotion_id': int, 'hidden_states': tensor, 'attention_mask': tensor}, ...]
        augment_copies: 每条数据复制几份（小数据集增强）
        """
        self.data = data_list * augment_copies

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['emotion_id'], dtype=torch.long),
            item['hidden_states'],    # [seq_len, 768]
            item['attention_mask'],   # [seq_len]
        )


# ========== 3. 训练 ==========
def train():
    # 加载预处理数据
    embed_path = os.path.join(os.path.dirname(__file__), '..', 'res', 'emotion_seq_embeddings.pt')
    saved = torch.load(embed_path, weights_only=False)
    data_list = saved['data']
    seq_len = saved['seq_len']
    print(f"加载数据: {len(data_list)} 条样本, 序列长度 {seq_len}")

    # 数据集（每条复制25份 → 32*25=800条）
    dataset = EmotionSeqDataset(data_list, augment_copies=25)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"数据集大小: {len(dataset)}")

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = EmotionEncoder(
        num_emotions=4,
        emotion_dim=128,
        hidden_dim=768,
        seq_len=seq_len,
        nhead=8,
        num_layers=2,
    ).to(device)

    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"设备: {device}, 模型参数量: {param_count:,}")

    # 优化器
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.MSELoss()

    # 训练
    num_epochs = 200
    print(f"\n开始训练 ({num_epochs} epochs)...")
    for epoch in range(num_epochs):
        total_loss = 0
        for emotion_ids, target_hidden, target_mask in dataloader:
            emotion_ids = emotion_ids.to(device)
            target_hidden = target_hidden.to(device)
            target_mask = target_mask.to(device).unsqueeze(-1)  # [batch, seq_len, 1]

            # 前向传播
            predicted = encoder(emotion_ids)  # [batch, seq_len, 768]

            # 只在有效token位置计算loss（忽略padding）
            loss = (((predicted - target_hidden) ** 2) * target_mask).sum() / target_mask.sum() / 768

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 20 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {lr:.6f}")

    # 保存
    save_path = os.path.join(os.path.dirname(__file__), '..', 'res', 'emotion_encoder_v2.pt')
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'seq_len': seq_len,
    }, save_path)
    print(f"\n模型已保存到 {os.path.abspath(save_path)}")
    return encoder, seq_len


# ========== 4. 生成音乐 ==========
def generate_music(encoder, seq_len):
    print("\n加载 MusicGen 生成音乐...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model.eval()

    device = next(encoder.parameters()).device
    encoder.eval()

    emotion_names = {0: "happy", 1: "angry", 2: "sad", 3: "calm"}
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'trained_v2')
    os.makedirs(output_dir, exist_ok=True)

    for emo_id, name in emotion_names.items():
        with torch.no_grad():
            emotion_input = torch.tensor([emo_id], dtype=torch.long).to(device)
            # 生成完整条件序列 [1, seq_len, 768]
            encoder_hidden_states = encoder(emotion_input).cpu()
            attention_mask = torch.ones(1, seq_len, dtype=torch.long)

            audio_values = model.generate(
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=attention_mask,
                guidance_scale=1.0,
                max_new_tokens=512,
            )

            sampling_rate = model.config.audio_encoder.sampling_rate
            filepath = os.path.join(output_dir, f"{name}_v2.wav")
            scipy.io.wavfile.write(filepath, rate=sampling_rate, data=audio_values[0, 0].numpy())
            print(f"  已生成: {filepath}")

    print("\n全部生成完成！去 output/trained_v2/ 听听效果")


if __name__ == "__main__":
    encoder, seq_len = train()
    generate_music(encoder, seq_len)
