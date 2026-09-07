#!/usr/bin/env bash
set -euo pipefail

# Selection-adjusted temporal validation on the existing 79-patient cohort.
# Run inside tmux on the Linux server. All stages are resumable: existing
# adapters, predictions, aggregation files, and fold selections are reused.

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"

BASE_MODEL="${BASE_MODEL:-${PROJECT_DIR}/models/Qwen3.5-4B}"
INTERNAL_TRAIN="${INTERNAL_TRAIN:-${PROJECT_DIR}/datasets/qwen_split_v1_rebuilt_legacyfmt/data_vl_train_split_v1.json}"
INTERNAL_VAL="${INTERNAL_VAL:-${PROJECT_DIR}/datasets/qwen_split_v1_rebuilt_legacyfmt/data_vl_val_split_v1.json}"
INTERNAL_ADAPTER="${INTERNAL_ADAPTER:-${PROJECT_DIR}/output/qwen35_4b_stage2_original_splitv1_rebuilt_legacyfmt}"
EXTERNAL_DIR="${EXTERNAL_DIR:-${PROJECT_DIR}/datasets/temporal_external_v4_reviewed_nooverlap_quality_filtered}"
FOLD_DIR="${FOLD_DIR:-${PROJECT_DIR}/datasets/temporal_external_v4_nested_selection_v1}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/output/nested_adaptation_selection_v1}"
NNUNET_RAW="${NNUNET_RAW:-${PROJECT_DIR}/nnUNet_raw_external/Dataset504_SpinalLesionTemporalExternalV4}"
NNUNET_PRED="${NNUNET_PRED:-${PROJECT_DIR}/output/nnunet_stage1_temporal_external_v4_predictions}"
SEED="${SEED:-42}"
RUN_FOLDS="${RUN_FOLDS:-0 1 2 3 4}"
USE_4BIT="${USE_4BIT:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
RESUME_TRAINING="${RESUME_TRAINING:-1}"
MAX_TRAIN_RETRIES="${MAX_TRAIN_RETRIES:-10}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-30}"

if [[ "${USE_4BIT}" != "0" && "${USE_4BIT}" != "1" ]]; then
  echo "[ERROR] USE_4BIT must be 0 or 1, got: ${USE_4BIT}" >&2
  exit 1
fi
if [[ "${RESUME_TRAINING}" != "0" && "${RESUME_TRAINING}" != "1" ]]; then
  echo "[ERROR] RESUME_TRAINING must be 0 or 1, got: ${RESUME_TRAINING}" >&2
  exit 1
fi

MODEL_PRECISION_ARGS=()
if [[ "${USE_4BIT}" == "1" ]]; then
  MODEL_PRECISION_ARGS+=(--load_in_4bit)
fi

mkdir -p "${RESULTS_DIR}" "${PROJECT_DIR}/output/logs"
LOG_FILE="${PROJECT_DIR}/output/logs/nested_adaptation_selection_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========== Nested adaptation selection =========="
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] BASE_MODEL=${BASE_MODEL}"
echo "[INFO] RUN_FOLDS=${RUN_FOLDS}"
echo "[INFO] USE_4BIT=${USE_4BIT}"
echo "[INFO] SAVE_STEPS=${SAVE_STEPS}"
echo "[INFO] SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "[INFO] RESUME_TRAINING=${RESUME_TRAINING}"
echo "[INFO] MAX_TRAIN_RETRIES=${MAX_TRAIN_RETRIES}"
echo "[INFO] RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS}"
echo "[INFO] LOG_FILE=${LOG_FILE}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[ERROR] Required directory not found: $1" >&2
    exit 1
  fi
}

require_file "${INTERNAL_TRAIN}"
require_file "${INTERNAL_VAL}"
require_file "${EXTERNAL_DIR}/data_vl_temporal_external.json"
require_file "${EXTERNAL_DIR}/data_vl_temporal_external_hidden_bbox.json"
require_file "${NNUNET_RAW}/manifest.json"
require_file "${INTERNAL_ADAPTER}/adapter_config.json"
require_dir "${BASE_MODEL}"
require_dir "${NNUNET_RAW}/labelsTs"
require_dir "${NNUNET_PRED}"

if [[ ! -f "${FOLD_DIR}/nested_protocol.json" ]]; then
  python make_nested_adaptation_folds.py \
    --qwen_json "${EXTERNAL_DIR}/data_vl_temporal_external.json" \
    --hidden_qwen_json "${EXTERNAL_DIR}/data_vl_temporal_external_hidden_bbox.json" \
    --output_dir "${FOLD_DIR}" \
    --outer_folds 5 \
    --adapt10_patients 8 \
    --adapt20_patients 15 \
    --seed "${SEED}"
else
  echo "[INFO] Reusing locked fold protocol: ${FOLD_DIR}/nested_protocol.json"
fi

run_pipeline_and_aggregate() {
  local adapter_path="$1"
  local qwen_json="$2"
  local hidden_json="$3"
  local output_dir="$4"

  if [[ ! -f "${output_dir}/nnunet_qwen_remap_predictions.csv" ]]; then
    mkdir -p "${output_dir}"
    python nnunet_qwen_remap_pipeline_top3_fusion.py \
      --base_model "${BASE_MODEL}" \
      --adapter_path "${adapter_path}" \
      --pred_dir "${NNUNET_PRED}" \
      --label_dir "${NNUNET_RAW}/labelsTs" \
      --manifest "${NNUNET_RAW}/manifest.json" \
      --qwen_json "${qwen_json}" \
      --hidden_qwen_json "${hidden_json}" \
      --output_dir "${output_dir}" \
      --eval_unit qwen_records \
      --image_resize 280 \
      "${MODEL_PRECISION_ARGS[@]}" \
      --max_new_tokens 128 \
      --shuffle_eval \
      --seed "${SEED}" \
      --coord_mode swap_xy \
      --bbox_strategy largest_component \
      --candidate_topk 3 \
      --candidate_min_area_ratio 0.15 \
      --candidate_border_penalty 0.15 \
      --adjacent_slice_fallback \
      --adjacent_slice_fallback_radius 3
  else
    echo "[INFO] Reusing predictions: ${output_dir}/nnunet_qwen_remap_predictions.csv"
  fi

  if [[ ! -f "${output_dir}/aggregation/aggregation_metrics.json" ]]; then
    python aggregate_pipeline_predictions.py \
      --pred_csv "${output_dir}/nnunet_qwen_remap_predictions.csv" \
      --output_dir "${output_dir}/aggregation" \
      --strategies quality_weighted_vote \
      --tie_policy uncertain
  else
    echo "[INFO] Reusing aggregation: ${output_dir}/aggregation/aggregation_metrics.json"
  fi
}

for FOLD in ${RUN_FOLDS}; do
  FOLD_NAME="fold_${FOLD}"
  DATA_DIR="${FOLD_DIR}/${FOLD_NAME}"
  FOLD_RESULT="${RESULTS_DIR}/${FOLD_NAME}"
  mkdir -p "${FOLD_RESULT}"
  echo "========== ${FOLD_NAME} =========="

  for CONFIG in adapt10_r5 adapt20_r3; do
    if [[ "${CONFIG}" == "adapt10_r5" ]]; then
      ADAPT_TAG="adapt10"
      REPEAT=5
    else
      ADAPT_TAG="adapt20"
      REPEAT=3
    fi

    MIXED_DIR="${FOLD_RESULT}/${CONFIG}/mixed_dataset"
    ADAPTER_DIR="${FOLD_RESULT}/${CONFIG}/adapter"
    INNER_OUT="${FOLD_RESULT}/${CONFIG}/inner_selection"

    if [[ ! -f "${MIXED_DIR}/mixed_summary.json" ]]; then
      python build_mixed_internal_external_train.py \
        --internal_train_json "${INTERNAL_TRAIN}" \
        --internal_val_json "${INTERNAL_VAL}" \
        --external_adapt_train_json "${DATA_DIR}/${ADAPT_TAG}_train.json" \
        --external_adapt_test_json "${DATA_DIR}/inner_selection.json" \
        --output_dir "${MIXED_DIR}" \
        --external_repeat "${REPEAT}" \
        --seed "$((SEED + FOLD))"
    fi

    if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
      TRAIN_RESUME_ARGS=()
      if [[ "${RESUME_TRAINING}" == "1" ]]; then
        TRAIN_RESUME_ARGS+=(--resume_from_checkpoint auto)
      fi
      TRAIN_CMD=(
        python qwen_stage2_classifier.py train
        --base_model "${BASE_MODEL}"
        --train_json "${MIXED_DIR}/train_mixed.json"
        --val_json "${MIXED_DIR}/internal_val.json"
        --output_dir "${ADAPTER_DIR}"
        --input_mode bbox_prompt
        --prompt_style original
        --answer_style original
        --image_resize 280
        "${MODEL_PRECISION_ARGS[@]}"
        --learning_rate 5e-5
        --num_train_epochs 2
        --per_device_train_batch_size 1
        --per_device_eval_batch_size 1
        --gradient_accumulation_steps 8
        --logging_steps 20
        --eval_steps 500
        --save_steps "${SAVE_STEPS}"
        --save_total_limit "${SAVE_TOTAL_LIMIT}"
        --max_new_tokens 128
        --no_mid_eval_cls_metrics
        --skip_final_eval
        "${TRAIN_RESUME_ARGS[@]}"
        --seed "$((SEED + FOLD))"
      )

      TRAIN_ATTEMPT=1
      while true; do
        echo "[INFO] Training ${FOLD_NAME}/${CONFIG}, attempt ${TRAIN_ATTEMPT}/$((MAX_TRAIN_RETRIES + 1))"
        if PYTHONFAULTHANDLER=1 "${TRAIN_CMD[@]}"; then
          break
        else
          TRAIN_STATUS=$?
        fi
        if (( TRAIN_ATTEMPT > MAX_TRAIN_RETRIES )); then
          echo "[ERROR] Training ${FOLD_NAME}/${CONFIG} failed after $((MAX_TRAIN_RETRIES + 1)) attempts (last status=${TRAIN_STATUS})." >&2
          exit "${TRAIN_STATUS}"
        fi
        echo "[WARN] Training exited with status ${TRAIN_STATUS}; retrying from the latest complete checkpoint in ${RETRY_DELAY_SECONDS}s."
        sleep "${RETRY_DELAY_SECONDS}"
        TRAIN_ATTEMPT=$((TRAIN_ATTEMPT + 1))
      done
    else
      echo "[INFO] Reusing adapter: ${ADAPTER_DIR}"
    fi

    run_pipeline_and_aggregate \
      "${ADAPTER_DIR}" \
      "${DATA_DIR}/inner_selection.json" \
      "${DATA_DIR}/inner_selection_hidden_bbox.json" \
      "${INNER_OUT}"
  done

  if [[ ! -f "${FOLD_RESULT}/selection.json" ]]; then
    python summarize_nested_adaptation_results.py select-fold \
      --fold "${FOLD}" \
      --adapt10_metrics "${FOLD_RESULT}/adapt10_r5/inner_selection/aggregation/aggregation_metrics.json" \
      --adapt20_metrics "${FOLD_RESULT}/adapt20_r3/inner_selection/aggregation/aggregation_metrics.json" \
      --output_json "${FOLD_RESULT}/selection.json"
  fi

  SELECTED_CONFIG="$(python summarize_nested_adaptation_results.py get-selected --selection_json "${FOLD_RESULT}/selection.json")"
  echo "[INFO] ${FOLD_NAME} selected ${SELECTED_CONFIG} before outer-test evaluation"

  run_pipeline_and_aggregate \
    "${FOLD_RESULT}/${SELECTED_CONFIG}/adapter" \
    "${DATA_DIR}/outer_test.json" \
    "${DATA_DIR}/outer_test_hidden_bbox.json" \
    "${FOLD_RESULT}/outer_test/${SELECTED_CONFIG}"

  run_pipeline_and_aggregate \
    "${INTERNAL_ADAPTER}" \
    "${DATA_DIR}/outer_test.json" \
    "${DATA_DIR}/outer_test_hidden_bbox.json" \
    "${FOLD_RESULT}/outer_test/internal_only"
done

if [[ "$(echo "${RUN_FOLDS}" | xargs)" == "0 1 2 3 4" ]]; then
  python summarize_nested_adaptation_results.py summarize \
    --protocol_json "${FOLD_DIR}/nested_protocol.json" \
    --results_root "${RESULTS_DIR}" \
    --output_dir "${RESULTS_DIR}/oof_summary" \
    --bootstrap_iterations 10000 \
    --seed "${SEED}"
else
  echo "[INFO] Partial RUN_FOLDS requested; skipping final OOF summary."
  echo "[INFO] Run all folds or call summarize_nested_adaptation_results.py summarize after completion."
fi

echo "========== Nested experiment complete =========="
echo "[INFO] Results: ${RESULTS_DIR}"
echo "[INFO] Log: ${LOG_FILE}"
