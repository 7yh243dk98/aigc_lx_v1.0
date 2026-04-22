import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

EMOTION_TEXTS = {
    0: "happy energetic upbeat cheerful music with bright melody",
    1: "angry intense aggressive powerful music with strong beat",
    2: "sad melancholic gentle slow music with minor key",
    3: "calm peaceful relaxing ambient music with soft tone",
}

DEFAULT_HUB = "facebook/musicgen-small"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "adapter_strategy_v1" / "target_text_embeddings.pt"
META_PATH = PROJECT_ROOT / "adapter_strategy_v1" / "target_text_embeddings.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="导出四类情绪英文的 T5 last_hidden_state，供 train_adapter 作监督目标"
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="本地 MusicGen 目录（如 res/musicgen_finetuned）；默认用 Hub 上的 facebook/musicgen-small",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_dir:
        model_src = (PROJECT_ROOT / args.model_dir).resolve()
        if not model_src.is_dir():
            raise FileNotFoundError(f"--model-dir 不是目录: {model_src}")
        load_path = str(model_src)
        processor = AutoProcessor.from_pretrained(load_path)
        model = MusicgenForConditionalGeneration.from_pretrained(load_path).to(device)
        audit_name = load_path
    else:
        load_path = DEFAULT_HUB
        processor = AutoProcessor.from_pretrained(load_path)
        model = MusicgenForConditionalGeneration.from_pretrained(load_path).to(device)
        audit_name = DEFAULT_HUB

    model.eval()

    rows = [EMOTION_TEXTS[i] for i in range(4)]
    with torch.no_grad():
        batch = processor(text=rows, padding=True, return_tensors="pt").to(device)
        out = model.text_encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        hidden = out.last_hidden_state.detach().cpu()  # [4, seq_len, 768]
        mask = batch.attention_mask.detach().cpu()  # [4, seq_len]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save({"hidden_states": hidden, "attention_mask": mask, "texts": EMOTION_TEXTS}, OUTPUT_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_source": audit_name,
                "shape_hidden_states": list(hidden.shape),
                "shape_attention_mask": list(mask.shape),
                "output_path": str(OUTPUT_PATH),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"model_source={audit_name}")
    print(f"saved targets -> {OUTPUT_PATH}")
    print(f"saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
