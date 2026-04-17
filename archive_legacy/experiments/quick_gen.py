"""测试：用 T5 编码的 encoder_outputs 生成 1 段音乐"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch, gc
import numpy as np
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile as wavfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

OUTPUT_DIR = r"D:\pyprojects\aigc-m\output\verify_fix"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("加载模型...")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = model.to(device)
model.eval()

texts = [
    "happy energetic upbeat cheerful music with bright melody",
    "sad melancholic gentle slow music with minor key",
]
names = ["Q1_happy", "Q3_sad"]

with torch.no_grad():
    inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)
    t5_out = model.text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    t5_hidden = t5_out.last_hidden_state

    for i, name in enumerate(names):
        enc = t5_hidden[i:i+1]
        mask = inputs.attention_mask[i:i+1]
        print(f"\n生成 {name}... (enc shape={enc.shape}, mask={mask})")

        audio_out = model.generate(
            inputs=None,
            encoder_outputs=(enc,),
            attention_mask=mask,
            max_new_tokens=256,
            guidance_scale=1.0,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        audio = audio_out[0, 0].cpu().numpy()
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio_norm = (audio / peak * 0.95 * 32767).astype(np.int16)
        else:
            audio_norm = (audio * 32767).astype(np.int16)
        path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        wavfile.write(path, 32000, audio_norm)
        print(f"  peak={peak:.6f} -> {path}")
        del audio_out
        torch.cuda.empty_cache()

print("\n完成!")
