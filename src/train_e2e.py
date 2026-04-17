"""
端到端微调 MusicGen：情绪标签 → 音乐生成
核心改进：用 T5 文本编码初始化情绪嵌入，确保与预训练 cross-attention 分布匹配
"""
import os
from pathlib import Path
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# ========== 1. 数据集 ==========
class EmopiaDataset(Dataset):
    def __init__(self, data_path, max_tokens=512):
        data = torch.load(data_path, weights_only=False)
        self.samples = []
        for item in data:
            codes = item['audio_codes']  # [4, T]
            if codes.shape[1] > max_tokens:
                codes = codes[:, :max_tokens]
            self.samples.append({
                'emotion_id': item['emotion_id'],
                'audio_codes': codes,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    emotion_ids = torch.tensor([b['emotion_id'] for b in batch])
    max_len = max(b['audio_codes'].shape[1] for b in batch)
    labels = torch.full((len(batch), max_len, 4), fill_value=-100, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b['audio_codes'].shape[1]
        labels[i, :t, :] = b['audio_codes'].T  # [4, t] -> [t, 4]
    return emotion_ids, labels

# ========== 2. 情绪嵌入层（T5 初始化） ==========
EMOTION_TEXTS = [
    "happy energetic upbeat cheerful music with bright melody",
    "angry intense aggressive powerful music with strong beat",
    "sad melancholic gentle slow music with minor key",
    "calm peaceful relaxing ambient music with soft tone",
]

class EmotionEmbedding(nn.Module):
    """
    情绪 ID → 条件向量序列
    用 T5 文本编码初始化，确保与 MusicGen cross-attention 的预训练分布一致。
    conditioning 是可学习参数，训练过程中会逐步微调。
    """
    def __init__(self, num_emotions=4, hidden_dim=768, max_seq_len=32):
        super().__init__()
        self.conditioning = nn.Parameter(torch.zeros(num_emotions, max_seq_len, hidden_dim))
        self.register_buffer('attention_masks', torch.zeros(num_emotions, max_seq_len, dtype=torch.long))

    def init_from_t5(self, model, processor, device):
        """用 T5 编码情绪文本描述来初始化（仅首次训练调用）"""
        with torch.no_grad():
            inputs = processor(text=EMOTION_TEXTS, padding=True, return_tensors="pt").to(device)
            t5_out = model.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            hidden = t5_out.last_hidden_state  # [4, seq_len, 768]
            seq_len = hidden.shape[1]
            max_seq_len = self.conditioning.shape[1]

            if seq_len <= max_seq_len:
                self.conditioning.data[:, :seq_len, :] = hidden.cpu()
                self.attention_masks[:, :seq_len] = inputs.attention_mask.cpu()
            else:
                self.conditioning.data = hidden[:, :max_seq_len, :].cpu()
                self.attention_masks = inputs.attention_mask[:, :max_seq_len].cpu()

        print(f"  T5 初始化完成: seq_len={seq_len}, "
              f"mean={self.conditioning.data.mean():.4f}, "
              f"std={self.conditioning.data.std():.4f}")

    def forward(self, emotion_ids):
        hidden_states = self.conditioning[emotion_ids]    # [B, max_seq_len, 768]
        masks = self.attention_masks[emotion_ids]          # [B, max_seq_len]
        return hidden_states, masks

# ========== 3. 冻结策略 ==========
def freeze_model(model):
    """冻结模型，解冻 cross-attention + enc_to_dec_proj + decoder 后4层 self-attention"""
    for name, param in model.named_parameters():
        param.requires_grad = False

    unfreeze_keywords = ['encoder_attn', 'enc_to_dec_proj']
    num_layers = model.decoder.config.num_hidden_layers
    for layer_idx in range(num_layers - 4, num_layers):
        unfreeze_keywords.append(f'decoder.layers.{layer_idx}.self_attn')

    for name, param in model.named_parameters():
        if any(k in name for k in unfreeze_keywords):
            param.requires_grad = True
            print(f"  解冻: {name}")

# ========== 4. 训练 ==========
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "res" / "emopia_tokens" / "emopia_tokens.pt"
    dataset = EmopiaDataset(data_path, max_tokens=512)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    print(f"数据集: {len(dataset)} 样本")

    save_dir = project_root / "res" / "musicgen_finetuned"
    ckpt_path = os.path.join(save_dir, "checkpoint.pt")
    has_checkpoint = os.path.exists(ckpt_path)

    if has_checkpoint:
        print(f"发现 checkpoint，从已微调权重恢复...")
        model = MusicgenForConditionalGeneration.from_pretrained(save_dir)
    else:
        print("加载 MusicGen 预训练模型...")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    if model.config.decoder.decoder_start_token_id is None:
        model.config.decoder.decoder_start_token_id = model.generation_config.decoder_start_token_id

    print("冻结参数...")
    freeze_model(model)

    # 情绪嵌入（T5 初始化或从 checkpoint 恢复）
    emotion_embed = EmotionEmbedding(num_emotions=4, hidden_dim=768, max_seq_len=32)

    if has_checkpoint:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        emotion_embed.load_state_dict(ckpt['emotion_embed'])
        print(f"  情绪嵌入从 checkpoint 恢复")
    else:
        print("用 T5 编码初始化情绪嵌入...")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model_tmp = model.to(device)
        emotion_embed.init_from_t5(model_tmp, processor, device)
        model = model_tmp.cpu()
        del processor

    emotion_embed = emotion_embed.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += list(emotion_embed.parameters())
    print(f"可训练参数量: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    start_epoch = 0
    if has_checkpoint:
        if ckpt.get('optimizer'):
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            print("  optimizer 状态为空，使用新初始化的 optimizer")
        start_epoch = ckpt['epoch'] + 1
        print(f"  从 Epoch {start_epoch + 1} 继续（上次 loss: {ckpt['loss']:.4f}）")

    model = model.to(device)
    model.train()
    num_epochs = 50

    print(f"\n开始训练 ({num_epochs} epochs, 从 Epoch {start_epoch + 1} 开始)...")
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for emotion_ids, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            emotion_ids = emotion_ids.to(device)
            labels = labels.to(device)

            encoder_hidden_states, attention_mask = emotion_embed(emotion_ids)

            outputs = model(
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        torch.save(emotion_embed.state_dict(), os.path.join(save_dir, "emotion_embed.pt"))
        torch.save({
            'epoch': epoch,
            'loss': avg_loss,
            'emotion_embed': emotion_embed.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        print(f"  Epoch {epoch+1} 已存档")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"训练完成! 模型保存到: {save_dir}")

if __name__ == "__main__":
    train()
