import os
from typing import Dict, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import EmotionToTextAdapter


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENS_PATH = PROJECT_ROOT / "res" / "emopia_tokens" / "emopia_tokens.pt"
TARGETS_PATH = PROJECT_ROOT / "adapter_strategy_v1" / "target_text_embeddings.pt"
SAVE_DIR = PROJECT_ROOT / "adapter_strategy_v1" / "checkpoints"


class AdapterDataset(Dataset):
    """
    Minimal dataset for adapter training.
    Priority:
    1) use item['eeg_feature'] if available and shape is [128]
    2) fallback to one-hot emotion feature [128] where first 4 dims are one-hot
    """

    def __init__(self, tokens_path: str):
        data = torch.load(tokens_path, weights_only=False)
        self.items = []
        for item in data:
            eid = int(item["emotion_id"])
            feat = None
            if "eeg_feature" in item:
                raw = torch.tensor(item["eeg_feature"], dtype=torch.float32).flatten()
                if raw.numel() == 128:
                    feat = raw
            if feat is None:
                feat = torch.zeros(128, dtype=torch.float32)
                feat[eid] = 1.0
            self.items.append((feat, eid))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.items[idx]


def load_targets(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    hidden = obj["hidden_states"].float()  # [4, seq_len, 768]
    attn = obj["attention_mask"].long()  # [4, seq_len]
    return {"hidden_states": hidden, "attention_mask": attn}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    targets = load_targets(TARGETS_PATH)
    target_hidden = targets["hidden_states"].to(device)
    target_mask = targets["attention_mask"].to(device).unsqueeze(-1).float()  # [4, seq_len, 1]
    seq_len = target_hidden.shape[1]

    dataset = AdapterDataset(TOKENS_PATH)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EmotionToTextAdapter(input_dim=128, hidden_dim=768, output_seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    num_epochs = 50
    print(f"device={device}, samples={len(dataset)}, seq_len={seq_len}")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total = 0.0
        for features, emotion_ids in tqdm(loader, desc=f"epoch {epoch}/{num_epochs}"):
            features = features.to(device)
            emotion_ids = emotion_ids.to(device)

            pred = model(features)  # [B, seq_len, 768]
            tgt = target_hidden[emotion_ids]  # [B, seq_len, 768]
            mask = target_mask[emotion_ids]  # [B, seq_len, 1]

            # masked MSE over valid tokens only
            loss = (((pred - tgt) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.item())

        avg = total / max(len(loader), 1)
        print(f"epoch={epoch} mse={avg:.6f}")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "avg_mse": avg,
                "seq_len": seq_len,
            },
            os.path.join(SAVE_DIR, "adapter_last.pt"),
        )

    print(f"done. checkpoint -> {os.path.join(SAVE_DIR, 'adapter_last.pt')}")


if __name__ == "__main__":
    main()

