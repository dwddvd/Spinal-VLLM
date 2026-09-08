#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_DIR}/models}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/output/inference_efficiency_benchmark_v1}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1}"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"

BASE_MODEL="${MODEL_ROOT}/Qwen3.5-4B"
ADAPTER_PATH="${PROJECT_DIR}/output/nested_adaptation_selection_v1/fold_0/adapt10_r5/adapter"
PRED_DIR="${PROJECT_DIR}/output/nnunet_stage1_temporal_external_v4_predictions"
LABEL_DIR="${PROJECT_DIR}/nnUNet_raw_external/Dataset504_SpinalLesionTemporalExternalV4/labelsTs"
MANIFEST="${PROJECT_DIR}/nnUNet_raw_external/Dataset504_SpinalLesionTemporalExternalV4/manifest.json"
QWEN_JSON="${PROJECT_DIR}/datasets/temporal_external_v4_reviewed_nooverlap_quality_filtered/data_vl_temporal_external.json"
HIDDEN_JSON="${PROJECT_DIR}/datasets/temporal_external_v4_reviewed_nooverlap_quality_filtered/data_vl_temporal_external_hidden_bbox.json"
PIPELINE_OUT="${OUT_DIR}/pipeline_output"

for path in "${BASE_MODEL}" "${ADAPTER_PATH}" "${PRED_DIR}" "${LABEL_DIR}" \
  "${MANIFEST}" "${QWEN_JSON}" "${HIDDEN_JSON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 1
  fi
done

MONITOR_PID=""
if nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits > "${OUT_DIR}/gpu_baseline.csv" 2> "${OUT_DIR}/gpu_monitor_error.log"; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -l "${MONITOR_INTERVAL}" \
    > "${OUT_DIR}/gpu_monitor.csv" 2>> "${OUT_DIR}/gpu_monitor_error.log" &
  MONITOR_PID=$!
else
  echo "[WARN] NVML monitoring unavailable; timing and CPU memory will still be measured."
  : > "${OUT_DIR}/gpu_monitor.csv"
fi

cleanup() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

date +%s.%N > "${OUT_DIR}/start_epoch.txt"

/usr/bin/time -v -o "${OUT_DIR}/time_verbose.txt" \
  "${PYTHON_BIN}" -m spinal_vllm.nnunet_qwen_remap_pipeline_top3_fusion \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${ADAPTER_PATH}" \
    --pred_dir "${PRED_DIR}" \
    --label_dir "${LABEL_DIR}" \
    --manifest "${MANIFEST}" \
    --qwen_json "${QWEN_JSON}" \
    --hidden_qwen_json "${HIDDEN_JSON}" \
    --output_dir "${PIPELINE_OUT}" \
    --eval_unit qwen_records \
    --image_resize 280 \
    --load_in_4bit \
    --max_new_tokens 128 \
    --seed 42 \
    --coord_mode swap_xy \
    --bbox_strategy largest_component \
    --candidate_topk 3 \
    --candidate_min_area_ratio 0.15 \
    --candidate_border_penalty 0.15 \
    --adjacent_slice_fallback \
    --adjacent_slice_fallback_radius 3 \
  2>&1 | tee "${OUT_DIR}/pipeline.log"

date +%s.%N > "${OUT_DIR}/end_epoch.txt"
cleanup
trap - EXIT

"${PYTHON_BIN}" -m experiments.summarize_inference_benchmark \
  --benchmark_dir "${OUT_DIR}" \
  --pipeline_metrics "${PIPELINE_OUT}/nnunet_qwen_remap_metrics.json" \
  --predictions_csv "${PIPELINE_OUT}/nnunet_qwen_remap_predictions.csv" \
  --output_json "${OUT_DIR}/inference_efficiency_summary.json"

echo "[INFO] Benchmark complete: ${OUT_DIR}/inference_efficiency_summary.json"
