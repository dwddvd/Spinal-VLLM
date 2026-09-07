#!/usr/bin/env python
"""Summarize a five-fold YOLO-Qwen OOF baseline against nnU-Net-Qwen OOF."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def binary_metrics(frame):
    gt = frame["gt_label"].astype(str)
    pred = frame["pred_label"].astype(str)
    infection = gt == "infection"
    tumor = gt == "tumor"
    infection_acc = float((pred[infection] == "infection").mean())
    tumor_acc = float((pred[tumor] == "tumor").mean())
    return {
        "total": int(len(frame)),
        "correct": int((gt == pred).sum()),
        "accuracy": float((gt == pred).mean()),
        "balanced_accuracy": (infection_acc + tumor_acc) / 2,
        "infection_sensitivity": infection_acc,
        "tumor_specificity": tumor_acc,
        "confusion": dict(Counter(f"{g}->{p}" for g, p in zip(gt, pred))),
    }


def stratified_indices(labels, rng):
    infection = np.flatnonzero(labels == "infection")
    tumor = np.flatnonzero(labels == "tumor")
    return np.r_[
        rng.choice(infection, len(infection), replace=True),
        rng.choice(tumor, len(tumor), replace=True),
    ]


def bootstrap_comparison(reference, yolo, iterations, seed):
    labels = reference["gt_label"].to_numpy(str)
    rng = np.random.default_rng(seed)
    metric_names = ["accuracy", "balanced_accuracy", "infection_sensitivity", "tumor_specificity"]
    reference_values = {key: [] for key in metric_names}
    yolo_values = {key: [] for key in metric_names}
    deltas = {key: [] for key in metric_names}
    for _ in range(iterations):
        indices = stratified_indices(labels, rng)
        ref_metrics = binary_metrics(reference.iloc[indices])
        yolo_metrics = binary_metrics(yolo.iloc[indices])
        for key in metric_names:
            reference_values[key].append(ref_metrics[key])
            yolo_values[key].append(yolo_metrics[key])
            deltas[key].append(ref_metrics[key] - yolo_metrics[key])
    return reference_values, yolo_values, deltas


def percentile(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def exact_mcnemar(reference_correct, yolo_correct):
    reference_only = int((reference_correct & ~yolo_correct).sum())
    yolo_only = int((~reference_correct & yolo_correct).sum())
    discordant = reference_only + yolo_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(reference_only, yolo_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "nnunet_qwen_only_correct": reference_only,
        "yolo_qwen_only_correct": yolo_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def slice_metrics(frame):
    has_candidate = frame["pred_label"].isin(["infection", "tumor"])
    det_iou = pd.to_numeric(frame.get("det_iou", 0), errors="coerce").fillna(0)
    relaxed = pd.to_numeric(frame.get("relaxed_localization_hit", 0), errors="coerce").fillna(0).astype(bool)
    correct = frame["pred_label"] == frame["gt_label"]
    return {
        "total_slices": int(len(frame)),
        "candidate_coverage": float(has_candidate.mean()),
        "classification_accuracy_given_candidate": float(correct[has_candidate].mean()) if has_candidate.any() else 0.0,
        "det_recall_iou_0.3": float((det_iou >= 0.3).mean()),
        "det_recall_iou_0.5": float((det_iou >= 0.5).mean()),
        "relaxed_localization_recall": float(relaxed.mean()),
        "joint_accuracy_iou_0.5": float(((det_iou >= 0.5) & correct).mean()),
        "joint_accuracy_relaxed": float((relaxed & correct).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--reference_patient_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--strategy", default="quality_weighted_vote")
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.results_root)
    patient_parts = []
    slice_parts = []
    fold_rows = []
    for fold in [int(value) for value in args.folds.split(",") if value.strip()]:
        selection_path = root / f"fold_{fold}" / "selection.json"
        with selection_path.open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
        config = selection["selected_config"]
        run_dir = root / f"fold_{fold}" / "yolo_outer_test" / config
        patient_path = run_dir / "aggregation" / "patient_level_predictions.csv"
        slice_path = run_dir / "yolo_slice_predictions_for_aggregation.csv"
        patient_frame = pd.read_csv(patient_path, dtype={"patient_id": str})
        patient_frame = patient_frame[patient_frame["strategy"] == args.strategy].copy()
        patient_frame["outer_fold"] = fold
        patient_frame["selected_config"] = config
        patient_parts.append(patient_frame)
        slice_frame = pd.read_csv(slice_path, dtype={"patient_id": str})
        slice_frame["outer_fold"] = fold
        slice_frame["selected_config"] = config
        slice_parts.append(slice_frame)
        fold_metric = binary_metrics(patient_frame)
        fold_rows.append({"outer_fold": fold, "selected_config": config, **fold_metric})

    yolo = pd.concat(patient_parts, ignore_index=True).sort_values("patient_id")
    slices = pd.concat(slice_parts, ignore_index=True)
    if yolo["patient_id"].duplicated().any():
        raise ValueError("YOLO OOF patient predictions contain duplicate patient IDs.")
    reference = pd.read_csv(args.reference_patient_csv, dtype={"patient_id": str}).sort_values("patient_id")
    keep_columns = ["patient_id", "gt_label", "pred_label"]
    paired = reference[keep_columns].merge(
        yolo[keep_columns],
        on="patient_id",
        suffixes=("_nnunet_qwen", "_yolo_qwen"),
        validate="one_to_one",
    )
    if len(paired) != len(reference) or len(paired) != len(yolo):
        raise ValueError("YOLO and nnU-Net OOF patient sets are not identical.")
    if not (paired["gt_label_nnunet_qwen"] == paired["gt_label_yolo_qwen"]).all():
        raise ValueError("Ground-truth labels disagree in paired OOF predictions.")

    ref_for_boot = pd.DataFrame(
        {
            "gt_label": paired["gt_label_nnunet_qwen"],
            "pred_label": paired["pred_label_nnunet_qwen"],
        }
    )
    yolo_for_boot = pd.DataFrame(
        {
            "gt_label": paired["gt_label_yolo_qwen"],
            "pred_label": paired["pred_label_yolo_qwen"],
        }
    )
    reference_metrics = binary_metrics(ref_for_boot)
    yolo_metrics = binary_metrics(yolo_for_boot)
    reference_boot, yolo_boot, delta_boot = bootstrap_comparison(
        ref_for_boot,
        yolo_for_boot,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    comparison_rows = []
    for metric in reference_boot:
        ref_low, ref_high = percentile(reference_boot[metric])
        yolo_low, yolo_high = percentile(yolo_boot[metric])
        delta_low, delta_high = percentile(delta_boot[metric])
        comparison_rows.append(
            {
                "metric": metric,
                "nnunet_qwen_estimate": reference_metrics[metric],
                "nnunet_qwen_ci_low": ref_low,
                "nnunet_qwen_ci_high": ref_high,
                "yolo_qwen_estimate": yolo_metrics[metric],
                "yolo_qwen_ci_low": yolo_low,
                "yolo_qwen_ci_high": yolo_high,
                "paired_delta_nnunet_minus_yolo": reference_metrics[metric] - yolo_metrics[metric],
                "paired_delta_ci_low": delta_low,
                "paired_delta_ci_high": delta_high,
            }
        )

    reference_correct = (
        paired["gt_label_nnunet_qwen"].to_numpy() == paired["pred_label_nnunet_qwen"].to_numpy()
    )
    yolo_correct = paired["gt_label_yolo_qwen"].to_numpy() == paired["pred_label_yolo_qwen"].to_numpy()
    result = {
        "comparison_design": (
            "Same 79 temporal OOF patients, identical outer folds, fold-specific nested-selected "
            "Qwen adapters, hidden-bbox prompts, top-3 candidates, and patient-level quality-weighted aggregation."
        ),
        "changed_component": (
            "lesion-proposal subsystem: volume-aware 2D nnU-Net with adjacent-slice fallback "
            "versus strict target-slice 2D YOLOv8x"
        ),
        "nnunet_qwen_patient_metrics": reference_metrics,
        "yolo_qwen_patient_metrics": yolo_metrics,
        "yolo_qwen_slice_metrics": slice_metrics(slices),
        "mcnemar": exact_mcnemar(reference_correct, yolo_correct),
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "method": "patient-level stratified paired percentile bootstrap",
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo.to_csv(output_dir / "yolo_qwen_oof_patient_predictions.csv", index=False, encoding="utf-8-sig")
    slices.to_csv(output_dir / "yolo_qwen_oof_slice_predictions.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output_dir / "paired_patient_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "patient_level_paired_bootstrap_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (output_dir / "yolo_nnunet_fair_oof_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
