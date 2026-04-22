"""
EEG 特征导出 v0.1 — Pydantic 校验模型。
"""
from __future__ import annotations

from typing import Any, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EEGExportRecordV01(BaseModel):
    """单条 EEG conditioning 记录（与 docs/contracts/eeg_feature_export_v0.1.md 一致）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: str = Field(..., min_length=1)
    subject_id: Union[str, int]
    segment_id: str = Field(..., min_length=1)
    feature: torch.Tensor

    trial_id: Optional[str] = None
    label: Optional[str] = None
    # 训练侧可选：与 EMOPIA 四类 T5 目标对齐时用 0..3；纯脑电导出可不填
    emotion_id: Optional[int] = None
    window_start_ms: Optional[int] = None
    window_end_ms: Optional[int] = None
    snn_version: Optional[str] = None
    feature_dim: Optional[int] = None
    feature_dtype: Optional[str] = None
    feature_space: Optional[str] = None
    export_version: Optional[str] = None

    @field_validator("subject_id", mode="before")
    @classmethod
    def subject_to_str(cls, v: Any) -> str:
        return str(v)

    @field_validator("feature", mode="before")
    @classmethod
    def feature_to_float32_1d(cls, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu()
        else:
            t = torch.as_tensor(v)
        if t.ndim != 1:
            raise ValueError(f"feature 必须为 1D，当前 ndim={t.ndim}")
        t = t.to(dtype=torch.float32)
        return t

    @field_validator("emotion_id", mode="before")
    @classmethod
    def emotion_id_range(cls, v: Any) -> Any:
        if v is None:
            return None
        iv = int(v)
        if iv < 0 or iv > 3:
            raise ValueError("emotion_id 必须在 0..3（与 adapter 四类目标一致）")
        return iv

    @model_validator(mode="after")
    def check_feature_dim(self) -> EEGExportRecordV01:
        d = int(self.feature.shape[0])
        if self.feature_dim is not None and int(self.feature_dim) != d:
            raise ValueError(
                f"feature_dim={self.feature_dim} 与 feature.shape[0]={d} 不一致"
            )
        if d != 128 and self.feature_dim is None:
            raise ValueError(
                f"feature 维度为 {d}（非 128）时必须提供 feature_dim={d}"
            )
        if self.feature_dtype is not None and self.feature_dtype != "float32":
            raise ValueError("feature_dtype 应为 float32 或省略")
        return self


def validate_records_pt(path: str) -> list[EEGExportRecordV01]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, list):
        raise ValueError(f"期望 list of dict，得到 {type(raw)}")
    return [EEGExportRecordV01.model_validate(r) for r in raw]
