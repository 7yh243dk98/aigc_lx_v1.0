#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/mnt/data/aigc_data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Data root: ${DATA_ROOT}"

mkdir -p "${DATA_ROOT}/datasets" "${DATA_ROOT}/checkpoints" "${DATA_ROOT}/outputs" "${DATA_ROOT}/cache/hf"

move_and_link() {
  local src_rel="$1"
  local dst_abs="$2"
  local src_abs="${PROJECT_ROOT}/${src_rel}"

  mkdir -p "$(dirname "${dst_abs}")"

  if [ -L "${src_abs}" ]; then
    echo "[OK] ${src_rel} is already a symlink."
    return 0
  fi

  if [ -e "${src_abs}" ] && [ ! -e "${dst_abs}" ]; then
    echo "[MOVE] ${src_rel} -> ${dst_abs}"
    mv "${src_abs}" "${dst_abs}"
  elif [ -e "${src_abs}" ] && [ -e "${dst_abs}" ]; then
    echo "[WARN] Both source and destination exist, skip move: ${src_rel}"
  fi

  if [ ! -e "${src_abs}" ]; then
    echo "[LINK] ${src_rel} -> ${dst_abs}"
    ln -s "${dst_abs}" "${src_abs}"
  fi
}

move_and_link "res" "${DATA_ROOT}/datasets/res"
move_and_link "output" "${DATA_ROOT}/outputs/output"
move_and_link "adapter_strategy_v1/checkpoints" "${DATA_ROOT}/checkpoints/adapter_strategy_v1/checkpoints"

cat <<'EOF'

[DONE] Filesystem reorganization finished.
[NEXT] Optional conda setup:
  conda env create -f environment.yml
  conda activate aigc-m-py311

[NEXT] Recommended environment vars (add to ~/.bashrc):
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/mnt/data/aigc_data/cache/hf
  export TRANSFORMERS_CACHE=/mnt/data/aigc_data/cache/hf/transformers
EOF
