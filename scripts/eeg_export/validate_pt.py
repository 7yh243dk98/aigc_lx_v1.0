"""校验 torch.save 的 list[dict] 是否符合 v0.1。"""
from __future__ import annotations

import argparse
import sys

from record_v0_1 import validate_records_pt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    try:
        rows = validate_records_pt(args.path)
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {len(rows)} records")
    r0 = rows[0]
    print(f"  first: dataset={r0.dataset} segment_id={r0.segment_id} feature={tuple(r0.feature.shape)} {r0.feature.dtype}")


if __name__ == "__main__":
    main()
