#!/usr/bin/env python
"""Select nested adaptation configurations and summarize out-of-fold results."""

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CLS_LABELS = ("infection", "tumor")
STRATEGY = "quality_weighted_vote"


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selection_metrics(path: Path) -> dict:
    data = read_json(path)
    try:
        metrics = data["patient_level"][STRATEGY]
    except KeyError as exc:
        raise KeyError(f"Missing patient_level.{STRATEGY} in {path}") from exc
    return {
        "balanced_acc": float(metrics["balanced_acc"]),
        "acc": float(metrics["acc"]),
        "total": int(metrics["total"]),
        "metrics_path": str(path),
    }


def select_fold(args: argparse.Namespace) -> None:
    candidates = {
        "adapt10_r5": selection_metrics(Path(args.adapt10_metrics)),
        "adapt20_r3": selection_metrics(Path(args.adapt20_metrics)),
    }
    a10 = candidates["adapt10_r5"]
    a20 = candidates["adapt20_r3"]
    if a20["balanced_acc"] > a10["balanced_acc"]:
        selected, reason = "adapt20_r3", "higher balanced accuracy"
    elif a10["balanced_acc"] > a20["balanced_acc"]:
        selected, reason = "adapt10_r5", "higher balanced accuracy"
    elif a20["acc"] > a10["acc"]:
        selected, reason = "adapt20_r3", "balanced accuracy tied; higher accuracy"
    elif a10["acc"] > a20["acc"]:
        selected, reason = "adapt10_r5", "balanced accuracy tied; higher accuracy"
    else:
        selected, reason = "adapt10_r5", "all metrics tied; prespecified lower-adaptation tie-break"

    result = {
        "fold": args.fold,
        "selection_dataset": "shared inner-selection patients",
        "selection_strategy": STRATEGY,
        "primary_metric": "balanced_acc",
        "tie_breaking": ["acc", "prefer adapt10_r5"],
        "candidates": candidates,
        "selected_config": selected,
        "selection_reason": reason,
        "outer_test_metrics_were_not_used": True,
    }
    write_json(Path(args.output_json), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def get_selected(args: argparse.Namespace) -> None:
    value = read_json(Path(args.selection_json)).get("selected_config")
    if value not in {"adapt10_r5", "adapt20_r3"}:
        raise ValueError(f"Invalid selected_config in {args.selection_json}: {value!r}")
    print(value)


def filter_strategy(rows: Iterable[dict]) -> List[dict]:
    selected = [row for row in rows if row.get("strategy") == STRATEGY]
    if not selected:
        raise ValueError(f"No {STRATEGY} rows found")
    return selected


def classification_metrics(rows: Sequence[dict]) -> dict:
    total = len(rows)
    if total == 0:
        raise ValueError("Cannot calculate metrics from zero patients")
    confusion = Counter(f"{row['gt_label']}->{row['pred_label']}" for row in rows)
    by_label = {}
    for label in CLS_LABELS:
        subset = [row for row in rows if row["gt_label"] == label]
        correct = sum(row["pred_label"] == label for row in subset)
        by_label[label] = {
            "correct": correct,
            "total": len(subset),
            "acc": correct / len(subset) if subset else math.nan,
        }

    correct = sum(row["gt_label"] == row["pred_label"] for row in rows)
    infection_tp = confusion["infection->infection"]
    infection_fn = total_by_label(rows, "infection") - infection_tp
    infection_fp = sum(row["gt_label"] == "tumor" and row["pred_label"] == "infection" for row in rows)
    tumor_tn = confusion["tumor->tumor"]
    ppv = safe_div(infection_tp, infection_tp + infection_fp)
    sensitivity = safe_div(infection_tp, infection_tp + infection_fn)
    f1_infection = safe_div(2 * ppv * sensitivity, ppv + sensitivity)
    tumor_precision = safe_div(tumor_tn, tumor_tn + infection_fn)
    tumor_recall = by_label["tumor"]["acc"]
    f1_tumor = safe_div(2 * tumor_precision * tumor_recall, tumor_precision + tumor_recall)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "balanced_accuracy": (by_label["infection"]["acc"] + by_label["tumor"]["acc"]) / 2,
        "infection_accuracy_sensitivity": by_label["infection"]["acc"],
        "tumor_accuracy_specificity": by_label["tumor"]["acc"],
        "ppv_infection": ppv,
        "npv_infection": safe_div(tumor_tn, tumor_tn + infection_fn),
        "f1_infection": f1_infection,
        "macro_f1": (f1_infection + f1_tumor) / 2,
        "no_candidate_or_uncertain": sum(row["pred_label"] not in CLS_LABELS for row in rows),
        "confusion": dict(sorted(confusion.items())),
        "by_label": by_label,
    }


def total_by_label(rows: Sequence[dict], label: str) -> int:
    return sum(row["gt_label"] == label for row in rows)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def stratified_bootstrap_indices(rows: Sequence[dict], rng: random.Random) -> List[int]:
    indices = []
    for label in CLS_LABELS:
        label_indices = [index for index, row in enumerate(rows) if row["gt_label"] == label]
        if not label_indices:
            raise ValueError(f"Bootstrap source has no {label} patients")
        indices.extend(rng.choice(label_indices) for _ in range(len(label_indices)))
    return indices


def bootstrap_single(rows: Sequence[dict], iterations: int, seed: int) -> dict:
    rng = random.Random(seed)
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "infection_accuracy_sensitivity",
        "tumor_accuracy_specificity",
        "f1_infection",
        "macro_f1",
    ]
    draws: Dict[str, List[float]] = {name: [] for name in metric_names}
    for _ in range(iterations):
        sampled = [rows[index] for index in stratified_bootstrap_indices(rows, rng)]
        metrics = classification_metrics(sampled)
        for name in metric_names:
            draws[name].append(metrics[name])
    point = classification_metrics(rows)
    return {
        name: {
            "estimate": point[name],
            "ci_low": percentile(draws[name], 0.025),
            "ci_high": percentile(draws[name], 0.975),
        }
        for name in metric_names
    }


def bootstrap_paired(selected_rows: Sequence[dict], baseline_rows: Sequence[dict], iterations: int, seed: int) -> dict:
    baseline_by_patient = {row["patient_id"]: row for row in baseline_rows}
    if set(baseline_by_patient) != {row["patient_id"] for row in selected_rows}:
        raise ValueError("Selected and baseline OOF patient sets do not match")
    ordered_selected = list(selected_rows)
    ordered_baseline = [baseline_by_patient[row["patient_id"]] for row in ordered_selected]
    for selected, baseline in zip(ordered_selected, ordered_baseline):
        if selected["gt_label"] != baseline["gt_label"]:
            raise ValueError(f"GT mismatch for patient {selected['patient_id']}")

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "infection_accuracy_sensitivity",
        "tumor_accuracy_specificity",
        "f1_infection",
        "macro_f1",
    ]
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = {name: [] for name in metric_names}
    for _ in range(iterations):
        indices = stratified_bootstrap_indices(ordered_selected, rng)
        selected_sample = [ordered_selected[index] for index in indices]
        baseline_sample = [ordered_baseline[index] for index in indices]
        selected_metrics = classification_metrics(selected_sample)
        baseline_metrics = classification_metrics(baseline_sample)
        for name in metric_names:
            draws[name].append(selected_metrics[name] - baseline_metrics[name])

    selected_point = classification_metrics(ordered_selected)
    baseline_point = classification_metrics(ordered_baseline)
    return {
        name: {
            "selected": selected_point[name],
            "baseline": baseline_point[name],
            "delta": selected_point[name] - baseline_point[name],
            "ci_low": percentile(draws[name], 0.025),
            "ci_high": percentile(draws[name], 0.975),
        }
        for name in metric_names
    }


def summarize(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root)
    protocol = read_json(Path(args.protocol_json))
    output_dir = Path(args.output_dir)
    selected_oof: List[dict] = []
    baseline_oof: List[dict] = []
    selected_slice_oof: List[dict] = []
    baseline_slice_oof: List[dict] = []
    fold_rows = []
    outer_seen = set()

    for fold_index in range(int(protocol["outer_folds"])):
        fold_name = f"fold_{fold_index}"
        fold_dir = results_root / fold_name
        selection = read_json(fold_dir / "selection.json")
        selected_config = selection["selected_config"]
        selected_path = fold_dir / "outer_test" / selected_config / "aggregation" / "patient_level_predictions.csv"
        baseline_path = fold_dir / "outer_test" / "internal_only" / "aggregation" / "patient_level_predictions.csv"
        selected_slice_path = (
            fold_dir / "outer_test" / selected_config / "nnunet_qwen_remap_predictions.csv"
        )
        baseline_slice_path = (
            fold_dir / "outer_test" / "internal_only" / "nnunet_qwen_remap_predictions.csv"
        )
        selected_rows = filter_strategy(read_csv(selected_path))
        baseline_rows = filter_strategy(read_csv(baseline_path))
        selected_slice_rows = read_csv(selected_slice_path)
        baseline_slice_rows = read_csv(baseline_slice_path)

        selected_patients = {row["patient_id"] for row in selected_rows}
        baseline_patients = {row["patient_id"] for row in baseline_rows}
        selected_slice_patients = {row["patient_id"] for row in selected_slice_rows}
        baseline_slice_patients = {row["patient_id"] for row in baseline_slice_rows}
        expected_patients = set(protocol["folds"][fold_index]["patient_ids"]["outer_test"])
        if (
            selected_patients != expected_patients
            or baseline_patients != expected_patients
            or selected_slice_patients != expected_patients
            or baseline_slice_patients != expected_patients
        ):
            raise ValueError(f"Outer-test patient mismatch in {fold_name}")
        overlap = outer_seen & selected_patients
        if overlap:
            raise ValueError(f"Patients occur in multiple outer folds: {sorted(overlap)}")
        outer_seen.update(selected_patients)

        selected_augmented = [{**row, "outer_fold": fold_index, "selected_config": selected_config} for row in selected_rows]
        baseline_augmented = [{**row, "outer_fold": fold_index, "selected_config": "internal_only"} for row in baseline_rows]
        selected_slice_augmented = [
            {**row, "outer_fold": fold_index, "selected_config": selected_config}
            for row in selected_slice_rows
        ]
        baseline_slice_augmented = [
            {**row, "outer_fold": fold_index, "selected_config": "internal_only"}
            for row in baseline_slice_rows
        ]
        selected_oof.extend(selected_augmented)
        baseline_oof.extend(baseline_augmented)
        selected_slice_oof.extend(selected_slice_augmented)
        baseline_slice_oof.extend(baseline_slice_augmented)
        selected_metrics = classification_metrics(selected_rows)
        baseline_metrics = classification_metrics(baseline_rows)
        fold_rows.append({
            "fold": fold_index,
            "selected_config": selected_config,
            "selection_reason": selection["selection_reason"],
            "inner_adapt10_balanced_acc": selection["candidates"]["adapt10_r5"]["balanced_acc"],
            "inner_adapt20_balanced_acc": selection["candidates"]["adapt20_r3"]["balanced_acc"],
            "outer_patients": len(selected_rows),
            "outer_infection": total_by_label(selected_rows, "infection"),
            "outer_tumor": total_by_label(selected_rows, "tumor"),
            "selected_outer_accuracy": selected_metrics["accuracy"],
            "selected_outer_balanced_accuracy": selected_metrics["balanced_accuracy"],
            "baseline_outer_accuracy": baseline_metrics["accuracy"],
            "baseline_outer_balanced_accuracy": baseline_metrics["balanced_accuracy"],
        })

    if len(outer_seen) != int(protocol["num_patients"]):
        raise ValueError(f"Expected {protocol['num_patients']} unique OOF patients, got {len(outer_seen)}")

    selected_oof.sort(key=lambda row: row["patient_id"])
    baseline_oof.sort(key=lambda row: row["patient_id"])
    selected_metrics = classification_metrics(selected_oof)
    baseline_metrics = classification_metrics(baseline_oof)
    selected_ci = bootstrap_single(selected_oof, args.bootstrap_iterations, args.seed)
    baseline_ci = bootstrap_single(baseline_oof, args.bootstrap_iterations, args.seed + 1)
    paired = bootstrap_paired(selected_oof, baseline_oof, args.bootstrap_iterations, args.seed + 2)
    selection_counts = Counter(row["selected_config"] for row in fold_rows)

    write_csv(output_dir / "oof_selected_patient_predictions.csv", selected_oof)
    write_csv(output_dir / "oof_internal_baseline_patient_predictions.csv", baseline_oof)
    write_csv(output_dir / "oof_selected_slice_predictions.csv", selected_slice_oof)
    write_csv(output_dir / "oof_internal_baseline_slice_predictions.csv", baseline_slice_oof)
    write_csv(output_dir / "fold_results.csv", fold_rows)

    ci_rows = []
    for model_name, ci_data in (("nested_selected", selected_ci), ("internal_only", baseline_ci)):
        for metric, values in ci_data.items():
            ci_rows.append({"model": model_name, "metric": metric, **values})
    for metric, values in paired.items():
        ci_rows.append({"model": "paired_delta", "metric": metric, **values})
    write_csv(output_dir / "patient_level_bootstrap_ci.csv", ci_rows, sorted({key for row in ci_rows for key in row}))

    summary = {
        "protocol": str(Path(args.protocol_json)),
        "results_root": str(results_root),
        "num_oof_patients": len(selected_oof),
        "selection_counts": dict(sorted(selection_counts.items())),
        "selection_rule": "inner patient-level quality-weighted balanced accuracy; accuracy tie-break; then adapt10_r5",
        "nested_selected_metrics": selected_metrics,
        "internal_only_metrics": baseline_metrics,
        "nested_selected_bootstrap_95ci": selected_ci,
        "internal_only_bootstrap_95ci": baseline_ci,
        "paired_delta_bootstrap_95ci": paired,
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "method": "patient-level stratified percentile bootstrap; paired resampling for deltas",
        },
        "interpretation_boundary": (
            "Selection-adjusted nested temporal validation within the existing same-center cohort; "
            "not a new independent multicenter external cohort."
        ),
        "files": {
            "selected_oof_csv": str(output_dir / "oof_selected_patient_predictions.csv"),
            "baseline_oof_csv": str(output_dir / "oof_internal_baseline_patient_predictions.csv"),
            "selected_slice_oof_csv": str(output_dir / "oof_selected_slice_predictions.csv"),
            "baseline_slice_oof_csv": str(output_dir / "oof_internal_baseline_slice_predictions.csv"),
            "fold_results_csv": str(output_dir / "fold_results.csv"),
            "bootstrap_ci_csv": str(output_dir / "patient_level_bootstrap_ci.csv"),
        },
    }
    write_json(output_dir / "nested_oof_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested adaptation selection and OOF summary utilities.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    select_parser = subparsers.add_parser("select-fold")
    select_parser.add_argument("--fold", type=int, required=True)
    select_parser.add_argument("--adapt10_metrics", required=True)
    select_parser.add_argument("--adapt20_metrics", required=True)
    select_parser.add_argument("--output_json", required=True)
    select_parser.set_defaults(func=select_fold)

    get_parser = subparsers.add_parser("get-selected")
    get_parser.add_argument("--selection_json", required=True)
    get_parser.set_defaults(func=get_selected)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--protocol_json", required=True)
    summary_parser.add_argument("--results_root", required=True)
    summary_parser.add_argument("--output_dir", required=True)
    summary_parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    summary_parser.add_argument("--seed", type=int, default=42)
    summary_parser.set_defaults(func=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
