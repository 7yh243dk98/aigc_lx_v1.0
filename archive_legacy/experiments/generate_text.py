"""
用 MusicGen 标准文本条件 pipeline 生成情绪音乐
不做任何微调，直接用预训练模型 + 精心设计的文本 prompt
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile as wavfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

OUTPUT_DIR = r"D:\pyprojects\aigc-m\output\text_conditioned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("加载 MusicGen 预训练模型...")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = model.to(device)
model.eval()

PROMPTS = {
    "Q1_happy": "cheerful happy upbeat pop music with bright piano melody and energetic rhythm",
    "Q2_angry": "intense aggressive powerful rock music with heavy distorted guitar and fast drums",
    "Q3_sad":   "sad melancholic slow piano ballad with gentle strings and minor key melody",
    "Q4_calm":  "calm peaceful ambient music with soft pad synthesizer and gentle nature sounds",
}

SAMPLE_RATE = 32000

for name, prompt in PROMPTS.items():
    print(f"\n生成 {name}: '{prompt}'")

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )

    audio = audio_values[0, 0].cpu().numpy()
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    audio_int16 = (audio * 32767).astype(np.int16)

    path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    wavfile.write(path, SAMPLE_RATE, audio_int16)
    print(f"  -> {path} (raw_peak={peak:.4f}, duration={len(audio_int16)/SAMPLE_RATE:.1f}s)")

    del audio_values
    torch.cuda.empty_cache()

print("\n全部生成完成!")
