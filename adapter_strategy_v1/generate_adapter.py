import os
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from transformers import MusicgenForConditionalGeneration

from models import EmotionToTextAdapter


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "res" / "musicgen_finetuned"
ADAPTER_CKPT = PROJECT_ROOT / "adapter_strategy_v1" / "checkpoints" / "adapter_last.pt"
TARGETS_PATH = PROJECT_ROOT / "adapter_strategy_v1" / "target_text_embeddings.pt"
OUTPUT_DIR = PROJECT_ROOT / "output" / "adapter_generated"

EMOTION_NAMES = {0: "Q1_happy", 1: "Q2_angry", 2: "Q3_sad", 3: "Q4_calm"}


def build_feature(emotion_id: int) -> torch.Tensor:
    # fallback feature for demo: one-hot in first 4 dims
    x = torch.zeros(128, dtype=torch.float32)
    x[emotion_id] = 1.0
    return x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"未找到本地模型目录: {MODEL_DIR}")

    targets = torch.load(TARGETS_PATH, map_location="cpu", weights_only=False)
    attn_mask_bank = targets["attention_mask"].long()  # [4, seq_len]
    seq_len = int(attn_mask_bank.shape[1])

    print(f"加载本地模型: {MODEL_DIR}")
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    adapter = EmotionToTextAdapter(input_dim=128, hidden_dim=768, output_seq_len=seq_len).to(device)
    ckpt = torch.load(ADAPTER_CKPT, map_location=device, weights_only=False)
    adapter.load_state_dict(ckpt["model_state"])
    adapter.eval()

    sample_rate = 32000
    max_new_tokens = 256

    for eid in range(4):
        with torch.no_grad():
            feature = build_feature(eid).unsqueeze(0).to(device)  # [1, 128]
            cond_hidden = adapter(feature)  # [1, seq_len, 768]
            cond_mask = attn_mask_bank[eid].unsqueeze(0).to(device)  # [1, seq_len]

            audio_values = model.generate(
                inputs=None,
                encoder_outputs=(cond_hidden,),
                attention_mask=cond_mask,
                max_new_tokens=max_new_tokens,
                guidance_scale=1.0,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        audio = audio_values[0, 0].detach().cpu().numpy()
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        pcm = (audio * 32767).astype(np.int16)
        out = os.path.join(OUTPUT_DIR, f"{EMOTION_NAMES[eid]}_adapter.wav")
        wavfile.write(out, sample_rate, pcm)
        print(f"saved {out} (raw_peak={peak:.6f})")


if __name__ == "__main__":
    main()

