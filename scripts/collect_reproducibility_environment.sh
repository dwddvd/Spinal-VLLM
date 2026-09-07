#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUT_DIR="${PROJECT_DIR}/output/reproducibility_environment"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_DIR}/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_DIR}/server_environment_${STAMP}.txt"

mkdir -p "${OUT_DIR}"
cd "${PROJECT_DIR}" || exit 1

{
  echo "# Reproducibility environment"
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "project_dir=${PROJECT_DIR}"
  echo "model_root=${MODEL_ROOT}"
  echo "python_bin=${PYTHON_BIN}"
  echo

  echo "## System"
  uname -a || true
  hostnamectl || true
  free -h || true
  df -h "${PROJECT_DIR}" "${MODEL_ROOT}" || true
  echo

  echo "## GPU"
  nvidia-smi || true
  echo

  echo "## Python"
  "${PYTHON_BIN}" --version || true
  echo

  echo "## Core packages"
  "${PYTHON_BIN}" -m pip show \
    torch torchvision transformers peft accelerate bitsandbytes \
    nnunetv2 dynamic-network-architectures batchgenerators \
    ultralytics numpy scipy scikit-image nibabel pillow pandas tqdm || true
  echo

  echo "## Compact package freeze"
  "${PYTHON_BIN}" -m pip freeze | grep -Ei \
    '^(torch|torchvision|transformers|peft|accelerate|bitsandbytes|nnunetv2|dynamic-network-architectures|batchgenerators|ultralytics|numpy|scipy|scikit-image|nibabel|pillow|pandas|tqdm)=' || true
  echo

  echo "## Git provenance"
  git rev-parse HEAD || true
  git status --short || true
  echo

  echo "## Model configuration files"
  find "${MODEL_ROOT}/Qwen3.5-4B" -maxdepth 1 -type f \
    \( -name 'config.json' -o -name 'generation_config.json' -o -name 'preprocessor_config.json' -o -name 'tokenizer_config.json' \) \
    -print -exec sha256sum {} \; 2>/dev/null || true
  echo

  echo "## Formal adapter configuration"
  ADAPTER_DIR="${PROJECT_DIR}/output/qwen35_4b_stage2_original_splitv1_rebuilt_legacyfmt"
  if [[ -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
    sha256sum "${ADAPTER_DIR}/adapter_config.json" || true
    cat "${ADAPTER_DIR}/adapter_config.json"
  fi
} > "${OUT_FILE}" 2>&1

echo "[INFO] Saved reproducibility environment to ${OUT_FILE}"
