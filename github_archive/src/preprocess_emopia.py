"""
数据预处理：EMOPIA MIDI → 训练数据
流程：MIDI → 音频 → EnCodec token → 保存
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
import pretty_midi
from scipy.io import wavfile
from transformers import MusicgenForConditionalGeneration
from tqdm import tqdm

# ========== 配置 ==========
EMOPIA_DIR = r"D:\pyprojects\aigc-m\res\emopia_data\EMOPIA_2.2"
MIDI_DIR = os.path.join(EMOPIA_DIR, "midis")
LABEL_FILE = os.path.join(EMOPIA_DIR, "label.csv")
OUTPUT_DIR = r"D:\pyprojects\aigc-m\res\emopia_tokens"
SAMPLE_RATE = 32000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1. 读取标签 ==========
print("读取情绪标签...")
labels = {}
with open(LABEL_FILE, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            name = parts[0]
            emotion_id = int(name[1]) - 1  # Q1→0, Q2→1, Q3→2, Q4→3
            labels[name] = emotion_id

print(f"共 {len(labels)} 个文件标签")

# ========== 2. 加载EnCodec ==========
print("加载 EnCodec...")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
audio_encoder = model.audio_encoder
audio_encoder.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audio_encoder = audio_encoder.to(device)

# ========== 3. MIDI → 音频 → Token ==========
# 尝试用 FluidSynth 高质量合成，否则降级到 pretty_midi
SF2_PATH = r"D:\pyprojects\aigc-m\res\GeneralUser-GS.sf2"
FLUIDSYNTH_DIR = r"D:\pyprojects\aigc-m\res\fluidsynth-v2.5.2-win10-x64-cpp11\bin"
FLUIDSYNTH_EXE = os.path.join(FLUIDSYNTH_DIR, "fluidsynth.exe")
USE_FLUIDSYNTH = os.path.exists(SF2_PATH) and os.path.exists(FLUIDSYNTH_EXE)

# 添加 FluidSynth 到 PATH（解决 DLL 依赖问题）
if USE_FLUIDSYNTH:
    os.environ['PATH'] = FLUIDSYNTH_DIR + os.pathsep + os.environ.get('PATH', '')
    print("使用 FluidSynth 高质量合成")
else:
    print("降级为 pretty_midi 合成")

def midi_to_wav(midi_path, output_wav, sr=32000):
    try:
        if USE_FLUIDSYNTH:
            import subprocess
            cmd = [
                FLUIDSYNTH_EXE,
                "-ni",
                "-g", "1.0",
                "-r", str(sr),
                "-F", output_wav,
                "-O", "s16",
                SF2_PATH,
                midi_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"  FluidSynth 错误 (code {result.returncode}): {result.stderr}")
                print(f"  stdout: {result.stdout}")
                return False
            return True
        else:
            # 降级方案：pretty_midi 正弦波合成
            pm = pretty_midi.PrettyMIDI(midi_path)
            audio = pm.synthesize(fs=sr)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            max_samples = sr * 10
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            audio = (audio * 32767).astype(np.int16)
            wavfile.write(output_wav, sr, audio)
            return True
    except Exception as e:
        print(f"  错误: {e}")
        return False

MAX_AUDIO_SEC = 30

def wav_to_tokens(wav_path):
    sr, audio = wavfile.read(wav_path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    max_samples = sr * MAX_AUDIO_SEC
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    audio = torch.from_numpy(audio).float()
    audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    
    with torch.no_grad():
        audio = audio.to(device)
        encoder_outputs = audio_encoder.encode(audio, None)
        audio_codes = encoder_outputs.audio_codes  # [batch, codebooks, T] or [batch, chunks, codebooks, T]
        audio_codes = audio_codes.squeeze(0)  # remove batch
        if audio_codes.dim() == 3:
            nb_chunks, num_codebooks, chunk_len = audio_codes.shape
            audio_codes = audio_codes.permute(1, 0, 2).reshape(num_codebooks, -1)
    return audio_codes.cpu()

# ========== 4. 处理文件 ==========
print("处理MIDI文件...")
all_data = []
midi_files = [f for f in os.listdir(MIDI_DIR) if f.endswith('.mid')]
processed = 0

for midi_file in tqdm(midi_files):
    name = midi_file.replace('.mid', '')
    if name not in labels:
        continue
    
    midi_path = os.path.join(MIDI_DIR, midi_file)
    temp_wav = os.path.join(OUTPUT_DIR, "temp.wav")
    
    if not midi_to_wav(midi_path, temp_wav, SAMPLE_RATE):
        continue
    
    try:
        tokens = wav_to_tokens(temp_wav)
        all_data.append({
            'name': name,
            'emotion_id': labels[name],
            'audio_codes': tokens,
        })
        processed += 1
    except Exception as e:
        print(f"  编码错误 {name}: {e}")
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

print(f"\n成功处理 {processed} 个文件")

# ========== 5. 保存 ==========
save_path = os.path.join(OUTPUT_DIR, "emopia_tokens.pt")
torch.save(all_data, save_path)
print(f"保存到: {save_path}")

from collections import Counter
emotion_counts = Counter([d['emotion_id'] for d in all_data])
print("各类情绪数量:", dict(emotion_counts))
