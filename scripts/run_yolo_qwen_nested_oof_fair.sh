#!/usr/bin/env bash
set -euo pipefail

# Same-domain, same-fold, same-adapter YOLO-Qwen versus nnU-Net-Qwen comparison.
# The YOLO detector is trained only on the locked internal split.

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

BASE_MODEL="${BASE_MODEL:-${PROJECT_DIR}/models/Qwen3.5-4B}"
EXTERNAL_DIR="${EXTERNAL_DIR:-${PROJECT_DIR}/datasets/temporal_external_v4_reviewed_nooverlap_quality_filtered}"
FOLD_DIR="${FOLD_DIR:-${PROJECT_DIR}/datasets/temporal_external_v4_nested_selection_v1}"
NESTED_RESULTS="${NESTED_RESULTS:-${PROJECT_DIR}/output/nested_adaptation_selection_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/output/yolo_qwen_nested_oof_fair_v1}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-${PROJECT_DIR}/output/yolo_stage1_splitv1_rebuilt_legacyfmt/yolov8x_img1280_lesion/weights/best.pt}"
YOLO_MODULE="${YOLO_MODULE:-preprocessing.yolo_stage1_detection}"
SEED="${SEED:-42}"
USE_4BIT="${USE_4BIT:-1}"
RUN_FOLDS="${RUN_FOLDS:-0 1 2 3 4}"

for required in \
  "${BASE_MODEL}" \
  "${EXTERNAL_DIR}/data_vl_temporal_external.json" \
  "${EXTERNAL_DIR}/data_vl_temporal_external_hidden_bbox.json" \
  "${FOLD_DIR}/nested_protocol.json" \
  "${NESTED_RESULTS}/oof_summary/oof_selected_patient_predictions.csv" \
  "${YOLO_WEIGHTS}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] Required path not found: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${PROJECT_DIR}/output/logs"
MODEL_ARGS=()
if [[ "${USE_4BIT}" == "1" ]]; then
  MODEL_ARGS+=(--load_in_4bit)
fi

YOLO_SOURCE="${OUTPUT_ROOT}/external_yolo_inference_source"
YOLO_PRED_CSV="${OUTPUT_ROOT}/external_yolo_top10_predictions.csv"
if [[ ! -f "${YOLO_SOURCE}/inference_source_manifest.json" ]]; then
  python -m preprocessing.prepare_yolo_inference_source_from_qwen_json \
    --qwen_json "${EXTERNAL_DIR}/data_vl_temporal_external.json" \
    --output_dir "${YOLO_SOURCE}"
fi
if [[ ! -f "${YOLO_PRED_CSV}" ]]; then
  python -m "${YOLO_MODULE}" predict \
    --weights "${YOLO_WEIGHTS}" \
    --source "${YOLO_SOURCE}" \
    --output_csv "${YOLO_PRED_CSV}" \
    --imgsz 1280 \
    --conf 0.01 \
    --iou 0.7 \
    --device 0 \
    --max_det 10 \
    --top_k 10 \
    --project "${OUTPUT_ROOT}" \
    --name external_predict_top10
fi

for FOLD in ${RUN_FOLDS}; do
  DATA_DIR="${FOLD_DIR}/fold_${FOLD}"
  FOLD_RESULT="${NESTED_RESULTS}/fold_${FOLD}"
  SELECTED_CONFIG="$(python -m experiments.summarize_nested_adaptation_results get-selected --selection_json "${FOLD_RESULT}/selection.json")"
  ADAPTER="${FOLD_RESULT}/${SELECTED_CONFIG}/adapter"
  RUN_DIR="${NESTED_RESULTS}/fold_${FOLD}/yolo_outer_test/${SELECTED_CONFIG}"
  mkdir -p "${RUN_DIR}"
  echo "========== fold_${FOLD}: ${SELECTED_CONFIG} =========="

  if [[ ! -f "${RUN_DIR}/yolo_qwen_final_selections.csv" ]]; then
    python -m spinal_vllm.yolo_qwen_pipeline \
      --base_model "${BASE_MODEL}" \
      --adapter_path "${ADAPTER}" \
      --pred_csv "${YOLO_PRED_CSV}" \
      --qwen_json "${DATA_DIR}/outer_test.json" \
      --hidden_qwen_json "${DATA_DIR}/outer_test_hidden_bbox.json" \
      --output_dir "${RUN_DIR}" \
      --top_ks 3 \
      --thresholds 0.3,0.5 \
      --final_strategies conf_weighted_vote \
      --image_resize 280 \
      "${MODEL_ARGS[@]}" \
      --max_new_tokens 128 \
      --shuffle_eval \
      --seed "${SEED}" \
      --no-region_prompt
  fi

  if [[ ! -f "${RUN_DIR}/yolo_slice_predictions_for_aggregation.csv" ]]; then
    python -m preprocessing.prepare_yolo_predictions_for_aggregation \
      --final_csv "${RUN_DIR}/yolo_qwen_final_selections.csv" \
      --output_csv "${RUN_DIR}/yolo_slice_predictions_for_aggregation.csv" \
      --top_k 3 \
      --strategy conf_weighted_vote
  fi

  if [[ ! -f "${RUN_DIR}/aggregation/aggregation_metrics.json" ]]; then
    python -m spinal_vllm.aggregate_pipeline_predictions \
      --pred_csv "${RUN_DIR}/yolo_slice_predictions_for_aggregation.csv" \
      --output_dir "${RUN_DIR}/aggregation" \
      --strategies quality_weighted_vote \
      --tie_policy uncertain
  fi
done

if [[ "$(echo "${RUN_FOLDS}" | xargs)" == "0 1 2 3 4" ]]; then
  python -m experiments.summarize_yolo_nested_oof_comparison \
    --results_root "${NESTED_RESULTS}" \
    --reference_patient_csv "${NESTED_RESULTS}/oof_summary/oof_selected_patient_predictions.csv" \
    --output_dir "${OUTPUT_ROOT}/oof_summary" \
    --bootstrap_iterations 10000 \
    --seed "${SEED}"
fi

echo "[INFO] Fair YOLO-Qwen OOF comparison complete: ${OUTPUT_ROOT}"
