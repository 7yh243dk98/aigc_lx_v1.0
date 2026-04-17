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

MODEL_NAME = "facebook/musicgen-small"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "adapter_strategy_v1" / "target_text_embeddings.pt"
META_PATH = PROJECT_ROOT / "adapter_strategy_v1" / "target_text_embeddings.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
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
                "model_name": MODEL_NAME,
                "shape_hidden_states": list(hidden.shape),
                "shape_attention_mask": list(mask.shape),
                "output_path": OUTPUT_PATH,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved targets -> {OUTPUT_PATH}")
    print(f"saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()

