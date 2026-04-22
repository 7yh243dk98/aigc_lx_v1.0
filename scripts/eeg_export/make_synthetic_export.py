"""
生成 v0.1 合成导出（N=16），用于与对方联调格式。
默认写入当前工作目录。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from record_v0_1 import EEGExportRecordV01


def make_records(n: int = 16, d: int = 128) -> list[dict]:
    records: list[dict] = []
    for i in range(n):
        subj = f"subj_{i % 4 + 1:02d}"
        seg = f"{subj}_seg_{i:04d}"
        rec = {
            "dataset": "toy_demo",
            "subject_id": subj,
            "segment_id": seg,
            "feature": torch.randn(d, dtype=torch.float32),
            "feature_dim": d,
            "feature_dtype": "float32",
            "feature_space": "penultimate",
            "emotion_id": i % 4,
            "label": "depression" if i % 2 == 0 else "control",
            "window_start_ms": int(i * 5000),
            "window_end_ms": int(i * 5000 + 2000),
            "snn_version": "snn_v0.1",
            "export_version": "eeg_feature_export_v0.1",
        }
        records.append(rec)
    return records


def save_npz(records: list[dict], path: Path) -> None:
    features = np.stack(
        [r["feature"].cpu().numpy().astype(np.float32, copy=False) for r in records],
        axis=0,
    )
    np.savez_compressed(
        path,
        dataset=np.array([r["dataset"] for r in records], dtype="U64"),
        subject_id=np.array([str(r["subject_id"]) for r in records], dtype="U64"),
        segment_id=np.array([r["segment_id"] for r in records], dtype="U128"),
        label=np.array([r["label"] for r in records], dtype="U32"),
        window_start_ms=np.array([r["window_start_ms"] for r in records], dtype=np.int64),
        window_end_ms=np.array([r["window_end_ms"] for r in records], dtype=np.int64),
        snn_version=np.array([r["snn_version"] for r in records], dtype="U64"),
        feature_dim=np.int64(features.shape[1]),
        feature_dtype=np.array("float32", dtype="U16"),
        feature_space=np.array("penultimate", dtype="U32"),
        export_version=np.array("eeg_feature_export_v0.1", dtype="U32"),
        features=features,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="输出目录（默认当前目录）",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = make_records(n=args.n, d=args.d)
    for r in records:
        EEGExportRecordV01.model_validate(r)

    pt_path = args.out_dir / "synthetic_eeg_export.pt"
    npz_path = args.out_dir / "synthetic_eeg_export.npz"
    torch.save(records, pt_path)
    save_npz(records, npz_path)
    print(f"saved {pt_path}")
    print(f"saved {npz_path}")


if __name__ == "__main__":
    main()
