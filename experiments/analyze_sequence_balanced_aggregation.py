#!/usr/bin/env python
"""Sensitivity analysis for sequence-balanced patient-level aggregation.

This script does not run nnU-Net or Qwen. It re-aggregates an existing
slice-level nnunet_qwen_remap_predictions.csv using inference-time fields only.
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from spinal_vllm.aggregate_pipeline_predictions import (
    CLS_LABELS,
    aggregate_rows,
    candidate_quality,
    finish_metrics,
    infer_label_from_case,
    row_vote_scores,
    safe_div,
    safe_float,
)


METHOD_SLICE_POOLED = "slice_pooled_quality_weighted"
METHOD_SEQUENCE_BALANCED = "sequence_balanced_quality_weighted"
METHOD_SEQUENCE_MAJORITY = "sequence_label_majority"


def choose_binary_label(infection, tumor, tie_policy="uncertain", atol=1e-12):
    if infection <= 0 and tumor <= 0:
        return "no_candidate"
    if math.isclose(infection, tumor, rel_tol=0.0, abs_tol=atol):
        return tie_policy if tie_policy in CLS_LABELS else "uncertain"
    return "infection" if infection > tumor else "tumor"


def patient_gt_label(rows):
    labels = {
        row.get("gt_label") or infer_label_from_case(row.get("case_id", ""))
        for row in rows
    }
    labels.discard("")
    if len(labels) > 1:
        raise ValueError(f"Inconsistent GT labels within patient: {sorted(labels)}")
    return next(iter(labels), "")


def patient_id_from_row(row):
    patient_id = str(row.get("patient_id", "")).strip()
    if patient_id:
        return patient_id
    parts = str(row.get("case_id", "")).split("_")
    return parts[1] if len(parts) > 1 else ""


def sequence_name_from_row(row):
    seq = str(row.get("seq", "")).strip().upper()
    return seq or "UNKNOWN"


def sequence_evidence(rows, patient_max_area):
    infection = 0.0
    tumor = 0.0
    candidate_slices = 0
    no_candidate_slices = 0

    for row in rows:
        pred = row.get("pred_label", "")
        if pred in CLS_LABELS:
            candidate_slices += 1
        elif pred == "no_candidate":
            no_candidate_slices += 1

        quality = candidate_quality(row, max_area=patient_max_area)
        vote_scores = row_vote_scores(row)
        infection += vote_scores["infection"] * quality
        tumor += vote_scores["tumor"] * quality

    total = infection + tumor
    return {
        "score_infection_raw": infection,
        "score_tumor_raw": tumor,
        "score_total_raw": total,
        "prob_infection": safe_div(infection, total),
        "prob_tumor": safe_div(tumor, total),
        "has_class_evidence": int(total > 0),
        "num_slices": len(rows),
        "num_candidate_slices": candidate_slices,
        "num_no_candidate_slices": no_candidate_slices,
    }


def aggregate_sequence_balanced(patient_id, rows, tie_policy):
    gt_label = patient_gt_label(rows)
    patient_max_area = max(
        (safe_float(row.get("candidate_slice_area"), 0.0) for row in rows),
        default=0.0,
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[sequence_name_from_row(row)].append(row)

    seq_records = []
    valid_records = []
    for seq, seq_rows in sorted(grouped.items()):
        evidence = sequence_evidence(seq_rows, patient_max_area=patient_max_area)
        seq_pred = choose_binary_label(
            evidence["score_infection_raw"],
            evidence["score_tumor_raw"],
            tie_policy=tie_policy,
        )
        record = {
            "patient_id": patient_id,
            "seq": seq,
            "gt_label": gt_label,
            "pred_label": seq_pred,
            **evidence,
        }
        seq_records.append(record)
        if evidence["has_class_evidence"]:
            valid_records.append(record)

    if valid_records:
        balanced_infection = sum(x["prob_infection"] for x in valid_records) / len(valid_records)
        balanced_tumor = sum(x["prob_tumor"] for x in valid_records) / len(valid_records)
        pred_label = choose_binary_label(
            balanced_infection,
            balanced_tumor,
            tie_policy=tie_policy,
        )
        confidence = max(balanced_infection, balanced_tumor)
    else:
        balanced_infection = 0.0
        balanced_tumor = 0.0
        pred_label = "no_candidate"
        confidence = 0.0

    seq_label_counts = Counter(
        x["pred_label"] for x in seq_records if x["pred_label"] in CLS_LABELS
    )
    majority_pred = choose_binary_label(
        seq_label_counts["infection"],
        seq_label_counts["tumor"],
        tie_policy=tie_policy,
    )

    common = {
        "patient_id": patient_id,
        "gt_label": gt_label,
        "seqs": ",".join(sorted(grouped)),
        "num_sequences": len(grouped),
        "num_valid_sequences": len(valid_records),
        "num_slices": len(rows),
        "num_candidate_slices": sum(
            1 for row in rows if row.get("pred_label") in CLS_LABELS
        ),
        "num_no_candidate_slices": sum(
            1 for row in rows if row.get("pred_label") == "no_candidate"
        ),
    }
    balanced = {
        **common,
        "strategy": METHOD_SEQUENCE_BALANCED,
        "pred_label": pred_label,
        "is_correct": int(gt_label == pred_label),
        "confidence": confidence,
        "score_infection": balanced_infection,
        "score_tumor": balanced_tumor,
        "slice_pred_counts": json.dumps(
            dict(Counter(row.get("pred_label", "") for row in rows)),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    majority = {
        **common,
        "strategy": METHOD_SEQUENCE_MAJORITY,
        "pred_label": majority_pred,
        "is_correct": int(gt_label == majority_pred),
        "confidence": safe_div(
            max(seq_label_counts["infection"], seq_label_counts["tumor"]),
            seq_label_counts["infection"] + seq_label_counts["tumor"],
        ),
        "score_infection": float(seq_label_counts["infection"]),
        "score_tumor": float(seq_label_counts["tumor"]),
        "slice_pred_counts": balanced["slice_pred_counts"],
    }
    return balanced, majority, seq_records


def aggregate_slice_pooled(patient_id, rows, tie_policy):
    result = aggregate_rows(
        rows,
        strategy="quality_weighted_vote",
        tie_policy=tie_policy,
    )
    return {
        "strategy": METHOD_SLICE_POOLED,
        "patient_id": patient_id,
        "seqs": ",".join(sorted({sequence_name_from_row(row) for row in rows})),
        "num_sequences": len({sequence_name_from_row(row) for row in rows}),
        "num_valid_sequences": "",
        **result,
    }


def metric_values(rows):
    metrics = finish_metrics(rows)
    return {
        "accuracy": metrics["acc"],
        "balanced_accuracy": metrics["balanced_acc"],
        "infection_accuracy": metrics["by_label"].get("infection", {}).get("acc", 0.0),
        "tumor_accuracy": metrics["by_label"].get("tumor", {}).get("acc", 0.0),
    }


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def bootstrap_comparison(rows_by_method, n_bootstrap, seed):
    methods = list(rows_by_method)
    by_patient = {
        method: {row["patient_id"]: row for row in rows}
        for method, rows in rows_by_method.items()
    }
    patient_ids = sorted(set.intersection(*(set(x) for x in by_patient.values())))
    if not patient_ids:
        raise ValueError("No common patients across aggregation methods.")

    rng = random.Random(seed)
    bootstrap_rows = []
    for iteration in range(n_bootstrap):
        sampled = [rng.choice(patient_ids) for _ in patient_ids]
        result = {"iteration": iteration}
        sampled_metrics = {}
        for method in methods:
            sample_rows = [by_patient[method][patient_id] for patient_id in sampled]
            sampled_metrics[method] = metric_values(sample_rows)
            for metric_name, value in sampled_metrics[method].items():
                result[f"{method}__{metric_name}"] = value

        for metric_name in (
            "accuracy",
            "balanced_accuracy",
            "infection_accuracy",
            "tumor_accuracy",
        ):
            result[f"delta_sequence_balanced_minus_slice_pooled__{metric_name}"] = (
                sampled_metrics[METHOD_SEQUENCE_BALANCED][metric_name]
                - sampled_metrics[METHOD_SLICE_POOLED][metric_name]
            )
        bootstrap_rows.append(result)

    summary = {}
    keys = [key for key in bootstrap_rows[0] if key != "iteration"]
    for key in keys:
        values = [row[key] for row in bootstrap_rows]
        summary[key] = {
            "mean": sum(values) / len(values),
            "ci95_low": percentile(values, 0.025),
            "ci95_high": percentile(values, 0.975),
        }
    return summary, bootstrap_rows


def exact_mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct slice-pooled patient aggregation with sequence-balanced "
            "T1/T2 aggregation using an existing pipeline prediction CSV."
        )
    )
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tie_policy",
        choices=["uncertain", "infection", "tumor"],
        default="uncertain",
    )
    args = parser.parse_args()

    with open(args.pred_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No prediction rows loaded from {args.pred_csv}")

    patient_groups = defaultdict(list)
    for row in rows:
        patient_id = patient_id_from_row(row)
        if not patient_id:
            raise ValueError(f"Unable to infer patient_id from row: {row}")
        patient_groups[patient_id].append(row)

    outputs = {
        METHOD_SLICE_POOLED: [],
        METHOD_SEQUENCE_BALANCED: [],
        METHOD_SEQUENCE_MAJORITY: [],
    }
    sequence_rows = []
    comparison_rows = []

    for patient_id, patient_rows in sorted(patient_groups.items()):
        pooled = aggregate_slice_pooled(patient_id, patient_rows, args.tie_policy)
        balanced, majority, patient_sequence_rows = aggregate_sequence_balanced(
            patient_id,
            patient_rows,
            args.tie_policy,
        )
        outputs[METHOD_SLICE_POOLED].append(pooled)
        outputs[METHOD_SEQUENCE_BALANCED].append(balanced)
        outputs[METHOD_SEQUENCE_MAJORITY].append(majority)
        sequence_rows.extend(patient_sequence_rows)

        comparison_rows.append({
            "patient_id": patient_id,
            "gt_label": balanced["gt_label"],
            "seqs": balanced["seqs"],
            "num_sequences": balanced["num_sequences"],
            "num_valid_sequences": balanced["num_valid_sequences"],
            "num_slices": balanced["num_slices"],
            "num_candidate_slices": balanced["num_candidate_slices"],
            "slice_pooled_pred": pooled["pred_label"],
            "slice_pooled_correct": pooled["is_correct"],
            "slice_pooled_score_infection": pooled["score_infection"],
            "slice_pooled_score_tumor": pooled["score_tumor"],
            "sequence_balanced_pred": balanced["pred_label"],
            "sequence_balanced_correct": balanced["is_correct"],
            "sequence_balanced_score_infection": balanced["score_infection"],
            "sequence_balanced_score_tumor": balanced["score_tumor"],
            "sequence_majority_pred": majority["pred_label"],
            "sequence_majority_correct": majority["is_correct"],
            "pooled_vs_balanced_discordant": int(
                pooled["pred_label"] != balanced["pred_label"]
            ),
        })

    point_metrics = {
        method: finish_metrics(method_rows)
        for method, method_rows in outputs.items()
    }
    bootstrap_summary, bootstrap_rows = bootstrap_comparison(
        outputs,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    pooled_map = {row["patient_id"]: row for row in outputs[METHOD_SLICE_POOLED]}
    balanced_map = {
        row["patient_id"]: row for row in outputs[METHOD_SEQUENCE_BALANCED]
    }
    pooled_only_correct = 0
    balanced_only_correct = 0
    for patient_id in sorted(pooled_map):
        pooled_correct = pooled_map[patient_id]["is_correct"]
        balanced_correct = balanced_map[patient_id]["is_correct"]
        pooled_only_correct += int(pooled_correct and not balanced_correct)
        balanced_only_correct += int(balanced_correct and not pooled_correct)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_csv = output_dir / "patient_aggregation_comparison.csv"
    sequence_csv = output_dir / "sequence_level_evidence.csv"
    bootstrap_csv = output_dir / "paired_bootstrap_samples.csv"
    metrics_json = output_dir / "sequence_balanced_sensitivity_metrics.json"

    write_csv(patient_csv, comparison_rows)
    write_csv(sequence_csv, sequence_rows)
    write_csv(bootstrap_csv, bootstrap_rows)

    metrics = {
        "analysis": "sequence_balanced_aggregation_sensitivity",
        "model_name": args.model_name,
        "pred_csv": str(Path(args.pred_csv)),
        "num_slice_rows": len(rows),
        "num_patients": len(patient_groups),
        "num_sequence_groups": len(sequence_rows),
        "sequence_counts": dict(Counter(row["seq"] for row in sequence_rows)),
        "patients_by_num_valid_sequences": dict(
            Counter(str(row["num_valid_sequences"]) for row in comparison_rows)
        ),
        "point_metrics": point_metrics,
        "paired_bootstrap": {
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "summary": bootstrap_summary,
        },
        "discordance": {
            "num_different_predictions": sum(
                row["pooled_vs_balanced_discordant"] for row in comparison_rows
            ),
            "slice_pooled_only_correct": pooled_only_correct,
            "sequence_balanced_only_correct": balanced_only_correct,
            "exact_mcnemar_p": exact_mcnemar_p(
                pooled_only_correct,
                balanced_only_correct,
            ),
        },
        "decision_rules": {
            METHOD_SLICE_POOLED: (
                "Directly sums quality-weighted infection/tumor evidence over all "
                "available T1/T2 slices for each patient."
            ),
            METHOD_SEQUENCE_BALANCED: (
                "Uses the same patient-level slice quality weights, sums evidence "
                "within each sequence, L1-normalizes infection/tumor evidence within "
                "each valid sequence, and averages valid sequences equally."
            ),
            METHOD_SEQUENCE_MAJORITY: (
                "Converts each sequence to one class label and applies an unweighted "
                "vote across valid sequence labels."
            ),
            "missing_sequence": (
                "No imputation. If only one sequence has valid class evidence, that "
                "sequence determines the patient score. If no sequence has evidence, "
                "the patient prediction is no_candidate."
            ),
            "no_gt_leakage": (
                "GT labels are used only for evaluation. Aggregation decisions do not "
                "use GT bbox, IoU, IoM, GT coverage, prediction precision, or center-hit."
            ),
        },
        "files": {
            "patient_comparison_csv": str(patient_csv),
            "sequence_evidence_csv": str(sequence_csv),
            "bootstrap_samples_csv": str(bootstrap_csv),
        },
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "metrics_json": str(metrics_json),
        "patient_comparison_csv": str(patient_csv),
        "sequence_evidence_csv": str(sequence_csv),
        "bootstrap_samples_csv": str(bootstrap_csv),
        "num_patients": len(patient_groups),
        "num_sequence_groups": len(sequence_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
