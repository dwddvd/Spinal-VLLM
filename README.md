# Spinal-VLLM

Code for a lesion-localized vision-language workflow that differentiates spinal infection from
tumor on sagittal MRI. The pipeline uses a two-dimensional nnU-Net lesion-proposal subsystem,
adjacent-slice candidate recovery, Qwen3.5-VL classification with predicted bounding-box prompts,
component-weighted fusion, and sequence/patient-level quality-weighted aggregation.

This repository contains the source-code release associated with the manuscript. It intentionally
does not contain patient data, private manifests, reversible identifiers, trained checkpoints, or
institution-specific filesystem paths.

## Workflow

1. Create a patient-disjoint internal split with `make_split_v1.py`.
2. Export lesion-positive 2D records with `export_qwen2d_from_npz_splitv1.py`.
3. Prepare and train the 2D nnU-Net proposal model with `nnunet_stage1_prepare.py` and nnU-Net v2.
4. Fine-tune the Qwen3.5-VL classifier with `qwen_stage2_classifier.py`.
5. Run the leakage-controlled proposal-to-classification pipeline with
   `nnunet_qwen_remap_pipeline_top3_fusion.py`.
6. Aggregate slice evidence with `aggregate_pipeline_predictions.py`.
7. Perform nested limited-sample temporal adaptation with
   `scripts/run_nested_adaptation_selection.sh`.
8. Run the same-patient YOLO comparator with `scripts/run_yolo_qwen_nested_oof_fair.sh`.

## Installation

The final experiments used Python 3.12.12 and the exact packages in `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate spinal-vllm
```

PyTorch CUDA wheels must match the installed driver. If the CUDA 13.0 wheel index is unavailable
for a target platform, install the closest officially supported PyTorch build first and then
install the remaining requirements. The executed server configuration is documented in
`docs/SERVER_ENVIRONMENT.md`.

## Expected private paths

Copy `.env.example` to a private location and set project-specific paths. Do not commit `.env`.
The data interfaces are documented in `docs/DATA_FORMAT.md`.

The final base model used in the study was Qwen3.5-VL 4B. A typical Linux invocation is:

```bash
export PROJECT_DIR="$PWD"
export BASE_MODEL="/path/to/Qwen3.5-4B"
export INTERNAL_TRAIN="/path/to/private/internal_train.json"
export INTERNAL_VAL="/path/to/private/internal_validation.json"
export INTERNAL_ADAPTER="/path/to/private/internal_adapter"
export EXTERNAL_DIR="/path/to/private/temporal_cohort"
export NNUNET_RAW="/path/to/private/nnunet_dataset"
export NNUNET_PRED="/path/to/private/nnunet_predictions"
bash scripts/run_nested_adaptation_selection.sh
```

The nested runner uses five patient-level outer folds. Candidate adaptation configurations are
selected only within the corresponding inner selection set; each patient receives one outer-fold
prediction from a model and configuration that did not use that patient for adaptation or model
selection.

## Main analysis utilities

- `summarize_nested_adaptation_results.py`: fold selection and complete out-of-fold summary.
- `analyze_sequence_balanced_aggregation.py`: sequence-balanced sensitivity analysis.
- `analyze_patient_score_calibration.py`: patient-level score calibration analysis.
- `analyze_multilesion_candidate_subgroups.py`: multi-candidate/multilesion subgroup analysis.
- `make_pipeline_strategy_ablation.py`: proposal and aggregation strategy ablations.
- `summarize_yolo_nested_oof_comparison.py`: paired nnU-Net-versus-YOLO comparison.
- `summarize_inference_benchmark.py`: inference timing and resource summary.

## Data and checkpoints

Human MRI data and trained weights are not publicly distributed. See
`docs/DATA_AND_MODEL_AVAILABILITY.md` for the access rationale and intended request route.

## Reproducibility and safety

Run `scripts/collect_reproducibility_environment.sh` to record the local environment. Before any
public release, confirm that `git status` contains no data, model, output, private mapping, or
absolute clinical path. The supplied `.gitignore` blocks the common sensitive artifacts used by
this project.

## License

The source code is released under the MIT License. External models and libraries remain subject
to their own licenses.
