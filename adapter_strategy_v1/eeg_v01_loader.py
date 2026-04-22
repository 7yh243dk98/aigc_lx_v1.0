"""加载并校验 eeg_feature_export_v0.1 的 torch.save(list_of_dicts)。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_validated_eeg_v01(path: str):
    """
    返回 list[EEGExportRecordV01]（与 scripts/eeg_export/record_v0_1 一致）。
    依赖 pydantic、与校验脚本相同的规则。
    """
    export_dir = PROJECT_ROOT / "scripts" / "eeg_export"
    s = str(export_dir)
    if s not in sys.path:
        sys.path.insert(0, s)
    from record_v0_1 import validate_records_pt

    return validate_records_pt(path)
