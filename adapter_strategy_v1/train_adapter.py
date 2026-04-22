import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eeg_v01_loader import load_validated_eeg_v01
from models import EmotionToTextAdapter


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENS_PATH = PROJECT_ROOT / "res" / "emopia_tokens" / "emopia_tokens.pt"
TARGETS_PATH = PROJECT_ROOT / "adapter_strategy_v1" / "target_text_embeddings.pt"
SAVE_DIR = PROJECT_ROOT / "adapter_strategy_v1" / "checkpoints"


class AdapterDataset(Dataset):
    """
    Adapter 训练用样本：(conditioning 向量 [D], emotion_id 0..3)。

    模式 A — EMOPIA tokens（默认）：
      优先 item['eeg_feature'] 且长度 128，否则 one-hot(emotion_id)。

    模式 B — eeg_feature_export_v0.1 的 torch.save(list_of_dicts)：
      使用每条记录的 feature；emotion_id 来自记录内 emotion_id，
      或由 --eeg-v01-default-emotion-id 统一指定（用于无标签试跑）。
      特征维 D 可不为 128（与 EmotionToTextAdapter(input_dim=D) 一致）。
    """

    def __init__(
        self,
        tokens_path: Optional[str] = None,
        eeg_v01_path: Optional[str] = None,
        eeg_v01_default_emotion_id: Optional[int] = None,
    ):
        self.items: List[Tuple[torch.Tensor, int]] = []

        if eeg_v01_path:
            records = load_validated_eeg_v01(eeg_v01_path)
            for r in records:
                feat = r.feature.clone()
                if r.emotion_id is not None:
                    eid = int(r.emotion_id)
                elif eeg_v01_default_emotion_id is not None:
                    eid = int(eeg_v01_default_emotion_id)
                else:
                    raise ValueError(
                        "eeg v0.1 记录缺少 emotion_id，且未设置 --eeg-v01-default-emotion-id。"
                        "与 EMOPIA 四类 T5 目标对齐时请导出 emotion_id(0..3)，或训练时指定默认类别。"
                    )
                self.items.append((feat, eid))
        elif tokens_path:
            data = torch.load(tokens_path, weights_only=False)
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
        else:
            raise ValueError("必须指定 tokens_path 或 eeg_v01_path")

        if not self.items:
            raise ValueError("数据集为空")

        dims = {int(t[0].shape[0]) for t in self.items}
        if len(dims) != 1:
            raise ValueError(
                f"所有样本 feature 维度必须一致，当前为 {sorted(dims)}；"
                "请在导出或 tokens 侧统一 D。"
            )
        self.input_dim = next(iter(dims))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.items[idx]


def load_targets(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    hidden = obj["hidden_states"].float()  # [4, seq_len, 768]
    attn = obj["attention_mask"].long()  # [4, seq_len]
    return {"hidden_states": hidden, "attention_mask": attn}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EmotionToTextAdapter")
    p.add_argument(
        "--tokens-path",
        type=str,
        default=str(TOKENS_PATH),
        help="emopia_tokens.pt（与 --eeg-v01 互斥）",
    )
    p.add_argument(
        "--eeg-v01",
        type=str,
        default=None,
        help="eeg_feature_export_v0.1 的 .pt（list of dicts）；指定时不再读 tokens",
    )
    p.add_argument(
        "--eeg-v01-default-emotion-id",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="当 v0.1 记录无 emotion_id 时，全体样本使用该类别（仅建议调试）",
    )
    p.add_argument("--targets-path", type=str, default=str(TARGETS_PATH))
    p.add_argument("--save-dir", type=str, default=str(SAVE_DIR))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.eeg_v01 and not Path(args.eeg_v01).is_file():
        raise FileNotFoundError(f"--eeg-v01 文件不存在: {args.eeg_v01}")
    if not args.eeg_v01 and not Path(args.tokens_path).is_file():
        raise FileNotFoundError(f"--tokens-path 文件不存在: {args.tokens_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    targets = load_targets(args.targets_path)
    target_hidden = targets["hidden_states"].to(device)
    target_mask = targets["attention_mask"].to(device).unsqueeze(-1).float()  # [4, seq_len, 1]
    seq_len = target_hidden.shape[1]

    if args.eeg_v01:
        dataset = AdapterDataset(
            eeg_v01_path=args.eeg_v01,
            eeg_v01_default_emotion_id=args.eeg_v01_default_emotion_id,
        )
    else:
        dataset = AdapterDataset(tokens_path=args.tokens_path)

    input_dim = dataset.input_dim
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = EmotionToTextAdapter(
        input_dim=input_dim, hidden_dim=768, output_seq_len=seq_len
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    print(
        f"device={device}, samples={len(dataset)}, input_dim={input_dim}, "
        f"seq_len={seq_len}, eeg_v01={args.eeg_v01!r}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for features, emotion_ids in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"):
            features = features.to(device)
            emotion_ids = emotion_ids.to(device)

            pred = model(features)  # [B, seq_len, 768]
            tgt = target_hidden[emotion_ids]  # [B, seq_len, 768]
            mask = target_mask[emotion_ids]  # [B, seq_len, 1]

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
                "input_dim": input_dim,
                "eeg_v01": args.eeg_v01,
            },
            os.path.join(args.save_dir, "adapter_last.pt"),
        )

    print(f"done. checkpoint -> {os.path.join(args.save_dir, 'adapter_last.pt')}")


if __name__ == "__main__":
    main()
