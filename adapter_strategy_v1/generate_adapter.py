import argparse
import os
import re
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from eeg_v01_loader import load_validated_eeg_v01
from models import EmotionToTextAdapter


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "res" / "musicgen_finetuned"
ADAPTER_CKPT = PROJECT_ROOT / "adapter_strategy_v1" / "checkpoints" / "adapter_last.pt"
TARGETS_PATH = PROJECT_ROOT / "adapter_strategy_v1" / "target_text_embeddings.pt"
OUTPUT_DIR = PROJECT_ROOT / "output" / "adapter_generated"

EMOTION_NAMES = {0: "Q1_happy", 1: "Q2_angry", 2: "Q3_sad", 3: "Q4_calm"}


def resolve_model_dir(s: str | None) -> str:
    """
    本地路径（相对项目根或绝对）或 Hub id（如 facebook/musicgen-small）。
    默认：res/musicgen_finetuned
    """
    if not s or not str(s).strip():
        p = MODEL_DIR
        if not p.exists():
            raise FileNotFoundError(f"未找到默认模型目录: {p}")
        return str(p.resolve())
    s = str(s).strip()
    p = Path(s)
    if p.is_dir():
        return str(p.resolve())
    p2 = PROJECT_ROOT / s
    if p2.is_dir():
        return str(p2.resolve())
    if "/" in s and not s.startswith("."):
        return s
    raise FileNotFoundError(f"不是有效目录或 Hub id: {s}")


def _debug_wav_name(model_src: str) -> str:
    tag = re.sub(r"[^\w.\-]+", "_", model_src.replace("/", "__"))[:80]
    return f"debug_text_baseline__{tag}.wav"


def build_feature(emotion_id: int, dim: int) -> torch.Tensor:
    if dim < 4:
        raise ValueError(f"input_dim={dim} 过小，无法放置 0..3 的 demo one-hot")
    x = torch.zeros(dim, dtype=torch.float32)
    x[emotion_id] = 1.0
    return x


def _safe_stem(s: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.ASCII)
    return s[:120] if len(s) > 120 else s


def _audio_to_mono_numpy(audio: torch.Tensor) -> np.ndarray:
    """
    与 HF 文档 / diagnose 一致：batch 0 的左声道 audio[0,0]；若单声道则为 [0,0] 或 [0,:]。
    """
    x = audio.float().detach().cpu()
    if x.dim() == 3 and x.shape[0] >= 1 and x.shape[1] >= 1:
        return x[0, 0].numpy()
    if x.dim() == 2:
        return x[0].numpy()
    if x.dim() == 1:
        return x.numpy()
    raise ValueError(f"unexpected audio tensor shape {tuple(x.shape)}")


def _float_mono_to_wav16(path: str, mono: np.ndarray, sample_rate: int) -> None:
    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono = mono / peak * 0.95
    pcm = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(str(path), int(sample_rate), pcm)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MusicGen + adapter 推理")
    p.add_argument("--adapter-ckpt", type=str, default=str(ADAPTER_CKPT))
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--from-eeg-v01",
        type=str,
        default=None,
        help="eeg_feature_export_v0.1 的 .pt：按条用 feature 生成 wav",
    )
    p.add_argument(
        "--eeg-indices",
        type=str,
        default=None,
        help='要生成的记录下标，如 "0,1,2"；默认全部',
    )
    p.add_argument(
        "--mask-emotion-id",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="记录无 emotion_id 时，用该类的 attention_mask",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="关闭采样，贪心解码，往往更长、更稳，避免后段迅速崩成噪/无声",
    )
    p.add_argument("--temperature", type=float, default=0.7, help="do_sample 时有效")
    p.add_argument("--top-p", type=float, default=0.9, dest="top_p", help="do_sample 时有效")
    p.add_argument(
        "--ablate-use-target-hidden",
        action="store_true",
        help="调试：不用 Adapter，直接用语义类的 target_text_embeddings 的 hidden 生成，检查是否仍后段无内容",
    )
    p.add_argument(
        "--test-text",
        type=str,
        default=None,
        help="调试用：不走 Adapter，用 MusicGen+processor 纯文本生成一条 baseline.wav 检查本地权重/解码",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="每首生成前可设不同 seed，避免采样时几条听起来雷同（仅 do_sample 时）",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="MusicGen 权重：项目内相对/绝对目录，或 Hub id。默认 res/musicgen_finetuned；听原版对比: facebook/musicgen-small",
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="分类器无关引导。None: --test-text 时默认 3.0 (HF 文档常用)；用 Adapter/encoder 时默认 1.0 以免与旧脚本不兼容。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    model_src = resolve_model_dir(args.model_dir)

    if args.test_text is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MusicGen 加载: {model_src}")
        model = MusicgenForConditionalGeneration.from_pretrained(model_src).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_src)
        sample_rate = int(model.config.audio_encoder.sampling_rate)
        gscale = 3.0 if args.guidance_scale is None else float(args.guidance_scale)
        with torch.no_grad():
            inputs = processor(text=[args.test_text], return_tensors="pt").to(device)
            if args.seed is not None:
                torch.manual_seed(args.seed)
            audio = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                temperature=1.0 if args.greedy else args.temperature,
                top_p=1.0 if args.greedy else args.top_p,
                guidance_scale=gscale,
            )
        out = os.path.join(args.output_dir, _debug_wav_name(model_src))
        os.makedirs(args.output_dir, exist_ok=True)
        audio_np = _audio_to_mono_numpy(audio)
        peak_raw = float(np.max(np.abs(audio_np)))
        out_path = Path(out).resolve()
        _float_mono_to_wav16(out_path, audio_np, sample_rate)
        sz = out_path.stat().st_size
        print(
            f"debug text-only (无 Adapter) -> {out_path}\n"
            f"  model={model_src} sample_rate={sample_rate} guidance_scale={gscale}\n"
            f"  shape(raw)={tuple(audio.shape)} peak_raw={peak_raw:.6f} bytes={sz}\n"
            f"  若仍像噪声：改 --greedy 或显式设 --guidance-scale 1~3 对比；并确认从 Hub 下的是完整权重。"
        )
        return

    targets = torch.load(TARGETS_PATH, map_location="cpu", weights_only=False)
    attn_mask_bank = targets["attention_mask"].long()  # [4, seq_len]
    hidden_bank = targets["hidden_states"].float()  # [4, seq_len, 768]
    seq_len = int(attn_mask_bank.shape[1])

    print(f"MusicGen 加载: {model_src}")
    if args.model_dir and "facebook" in model_src and "finetuned" not in model_src:
        print(
            "  注意: 与 train_adapter 用的 target 若来自另一套 T5/长度，"
            "EEG+Adapter 条件可能分布不一致；本开关主要用于对比基座听感 (test-text 或自测)。"
        )
    model = MusicgenForConditionalGeneration.from_pretrained(model_src).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    sample_rate = int(model.config.audio_encoder.sampling_rate)
    gscale = 1.0 if args.guidance_scale is None else float(args.guidance_scale)

    ckpt_path = Path(args.adapter_ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"未找到 adapter 权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = int(ckpt.get("input_dim", 128))
    adapter = EmotionToTextAdapter(
        input_dim=input_dim, hidden_dim=768, output_seq_len=seq_len
    ).to(device)
    adapter.load_state_dict(ckpt["model_state"])
    adapter.eval()
    print(f"adapter input_dim={input_dim} <- {ckpt_path}")
    print(f"  audio sample_rate={sample_rate} guidance_scale={gscale}")

    max_new_tokens = args.max_new_tokens
    do_sample = not args.greedy
    temperature = 1.0 if not do_sample else args.temperature
    top_p = 1.0 if not do_sample else args.top_p

    def write_one(full_audio: torch.Tensor, out_path: str) -> None:
        audio = _audio_to_mono_numpy(full_audio)
        peak = float(np.max(np.abs(audio)))
        p = Path(out_path).resolve()
        _float_mono_to_wav16(p, audio, sample_rate)
        print(f"saved {p} (raw_peak={peak:.6f}, bytes={p.stat().st_size})")

    if args.from_eeg_v01:
        records = load_validated_eeg_v01(args.from_eeg_v01)
        if args.eeg_indices:
            indices = [int(x.strip()) for x in args.eeg_indices.split(",") if x.strip()]
        else:
            indices = list(range(len(records)))
        for j, i in enumerate(indices):
            r = records[i]
            feat = r.feature.to(dtype=torch.float32)
            if feat.shape[0] != input_dim:
                raise ValueError(
                    f"记录 {i} feature 维 {feat.shape[0]} 与 adapter input_dim={input_dim} 不一致"
                )
            eid = int(r.emotion_id) if r.emotion_id is not None else int(args.mask_emotion_id)
            stem = _safe_stem(f"{i:04d}_{r.segment_id}")
            out = os.path.join(args.output_dir, f"eeg_{stem}_adapter.wav")
            with torch.no_grad():
                if args.seed is not None:
                    torch.manual_seed(int(args.seed) + j * 1000)
                if args.ablate_use_target_hidden:
                    cond_hidden = hidden_bank[eid : eid + 1].to(device)
                else:
                    cond_hidden = adapter(feat.unsqueeze(0).to(device))
                cond_mask = attn_mask_bank[eid].unsqueeze(0).to(device)
                audio_values = model.generate(
                    inputs=None,
                    encoder_outputs=(cond_hidden,),
                    attention_mask=cond_mask,
                    max_new_tokens=max_new_tokens,
                    guidance_scale=gscale,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
            write_one(audio_values, out)
        return

    for eid in range(4):
        with torch.no_grad():
            if args.ablate_use_target_hidden:
                cond_hidden = hidden_bank[eid : eid + 1].to(device)
            else:
                feature = build_feature(eid, input_dim).unsqueeze(0).to(device)
                cond_hidden = adapter(feature)  # [1, seq_len, 768]
            cond_mask = attn_mask_bank[eid].unsqueeze(0).to(device)  # [1, seq_len]

            audio_values = model.generate(
                inputs=None,
                encoder_outputs=(cond_hidden,),
                attention_mask=cond_mask,
                max_new_tokens=max_new_tokens,
                guidance_scale=gscale,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

        out = os.path.join(args.output_dir, f"{EMOTION_NAMES[eid]}_adapter.wav")
        write_one(audio_values, out)


if __name__ == "__main__":
    main()

