# Spinal-VLLM

Code for a lesion-localized vision-language workflow that differentiates spinal infection from
tumor on sagittal MRI. The pipeline uses a two-dimensional nnU-Net lesion-proposal subsystem,
adjacent-slice candidate recovery, Qwen3.5-VL classification with predicted bounding-box prompts,
component-weighted fusion, and sequence/patient-level quality-weighted aggregation.

This repository contains the source-code release associated with the manuscript. It intentionally
does not contain patient data, private manifests, reversible identifiers, trained checkpoints, or
institution-specific filesystem paths.

## Workflow

1. Create a patient-disjoint internal split with `preprocessing.make_split_v1`.
2. Export lesion-positive 2D records with `preprocessing.export_qwen2d_from_npz_splitv1`.
3. Prepare and train the 2D nnU-Net proposal model with `preprocessing.nnunet_stage1_prepare` and nnU-Net v2.
4. Fine-tune the Qwen3.5-VL classifier with `spinal_vllm.qwen_stage2_classifier`.
5. Run the leakage-controlled proposal-to-classification pipeline with
   `spinal_vllm.nnunet_qwen_remap_pipeline_top3_fusion`.
6. Aggregate slice evidence with `spinal_vllm.aggregate_pipeline_predictions`.
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

## Repository layout

- `spinal_vllm/`: the four core classification, proposal-fusion, and aggregation modules.
- `preprocessing/`: only the data conversion and proposal-model utilities required by the reported workflow.
- `experiments/`: nested selection, formal comparators, sensitivity analyses, and statistical summaries reported in the manuscript.
- `scripts/`: three end-to-end commands for the primary experiment, YOLO comparison, and efficiency benchmark.

Every retained Python file maps to a reported method, result, comparator, or sensitivity analysis.
Historical fixed-adaptation scripts, exploratory all-slice evaluation, cohort bookkeeping, and
release-only de-identification utilities are intentionally excluded.

## Data and checkpoints

Human MRI data and trained weights are not publicly distributed. See
`docs/DATA_AND_MODEL_AVAILABILITY.md` for the access rationale and intended request route.

## Reproducibility and safety

Before any derivative release, confirm that `git status` contains no data, model, output, private
mapping, or absolute clinical path. The supplied `.gitignore` blocks the common sensitive
artifacts used by this project.

## License

The source code is released under the MIT License. External models and libraries remain subject
to their own licenses.
