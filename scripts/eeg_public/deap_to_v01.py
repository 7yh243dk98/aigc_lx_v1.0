"""
将 DEAP 预处理 Python 数据（sXX.dat pickle）导出为 eeg_feature_export_v0.1 的 torch.save(list[dict])。

依赖：numpy, torch；与校验共用 scripts/eeg_export/record_v0_1（pydantic）。
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_eeg_export_path() -> None:
    p = str(PROJECT_ROOT / "scripts" / "eeg_export")
    if p not in sys.path:
        sys.path.insert(0, p)


def load_deap_subject(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """返回 data (n_trials, n_ch, n_samples), labels (n_trials, 4) valence,arousal,..."""
    with open(path, "rb") as f:
        obj: Any = pickle.load(f, encoding="latin1")
    if not isinstance(obj, dict) or "data" not in obj or "labels" not in obj:
        raise ValueError(f"非预期 DEAP pickle 结构: {path}")
    data = np.asarray(obj["data"], dtype=np.float64)
    labels = np.asarray(obj["labels"], dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"{path.name}: data 维数应为 3，当前 {data.shape}")
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError(f"{path.name}: labels 应含 valence/arousal 列")
    return data, labels


def trial_mean_std_feature(trial: np.ndarray) -> np.ndarray:
    """trial: (n_ch, n_samples) -> (2*n_ch,) float32"""
    m = trial.mean(axis=1)
    s = trial.std(axis=1)
    out = np.concatenate([m, s]).astype(np.float32)
    return out


def va_to_emotion_id(valence: float, arousal: float, midpoint: float = 5.0) -> int:
    """四象限 → 0..3，与 README_DEAP 表一致。"""
    hv = valence >= midpoint
    ha = arousal >= midpoint
    if hv and ha:
        return 0
    if (not hv) and ha:
        return 1
    if (not hv) and (not ha):
        return 2
    return 3


def discover_subject_files(deap_dir: Path) -> List[Path]:
    files = sorted(deap_dir.glob("s*.dat"))
    if not files:
        # 常见：多一层 data_preprocessed_python
        nested = list(deap_dir.rglob("s*.dat"))
        files = sorted({p for p in nested if re.match(r"s\d+\.dat$", p.name)})
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="DEAP preprocessed Python → eeg_feature_export_v0.1 .pt")
    parser.add_argument(
        "--deap-dir",
        type=Path,
        required=True,
        help="含 s01.dat … 的目录（可指向解压根目录，脚本会尝试递归发现）",
    )
    parser.add_argument("--out", type=Path, required=True, help="输出 .pt 路径")
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help='只处理这些被试，逗号分隔，如 "s01,s02"；默认全部 s*.dat',
    )
    parser.add_argument("--max-trials", type=int, default=None, help="每被试最多前 K 个 trial（试跑）")
    parser.add_argument("--va-midpoint", type=float, default=5.0, help="Valence/Arousal 高低分界（DEAP 1–9）")
    args = parser.parse_args()

    subject_files = discover_subject_files(args.deap_dir)
    if not subject_files:
        raise FileNotFoundError(f"在 {args.deap_dir} 下未找到 s*.dat，请检查 DEAP 解压路径")

    if args.subjects:
        want = {x.strip().lower() for x in args.subjects.split(",") if x.strip()}
        subject_files = [p for p in subject_files if p.stem.lower() in want]

    records: List[dict] = []
    for sub_path in subject_files:
        sid = sub_path.stem.lower()
        data, labels = load_deap_subject(sub_path)
        n_trials = data.shape[0]
        n_use = n_trials if args.max_trials is None else min(n_trials, args.max_trials)
        d_feat = int(trial_mean_std_feature(data[0]).shape[0])

        for t in range(n_use):
            v = float(labels[t, 0])
            a = float(labels[t, 1])
            eid = va_to_emotion_id(v, a, midpoint=args.va_midpoint)
            feat = torch.from_numpy(trial_mean_std_feature(data[t]))
            rec = {
                "dataset": "DEAP",
                "subject_id": sid,
                "segment_id": f"{sid}_trial_{t:02d}",
                "feature": feat,
                "feature_dim": d_feat,
                "feature_dtype": "float32",
                "feature_space": "deap_channel_mean_std_time",
                "emotion_id": eid,
                "trial_id": f"{sid}_trial_{t:02d}",
                "window_start_ms": 0,
                "window_end_ms": int(round(data.shape[2] / 128.0 * 1000)),
                "export_version": "eeg_feature_export_v0.1",
                "snn_version": "deap_baseline_mean_std",
            }
            records.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, args.out)
    print(f"saved {len(records)} records -> {args.out} (feature_dim={records[0]['feature_dim']})")

    _ensure_eeg_export_path()
    from record_v0_1 import EEGExportRecordV01

    for r in records:
        EEGExportRecordV01.model_validate(r)
    print("Pydantic 校验通过")


if __name__ == "__main__":
    main()
