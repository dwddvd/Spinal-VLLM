#!/usr/bin/env python
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from spinal_vllm.aggregate_pipeline_predictions import aggregate_rows, finish_metrics


CLS_LABELS = {"infection", "tumor"}
COMPONENT_RANGE = range(1, 4)


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def load_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def component_label(row, idx):
    label = (row.get(f"component_{idx}_label") or "").strip()
    return label if label in CLS_LABELS else ""


def component_score(row, idx):
    return safe_float(row.get(f"component_{idx}_score"), 0.0)


def component_weight(row, idx):
    return safe_float(row.get(f"component_{idx}_weight"), component_score(row, idx))


def set_scores(row, infection=0.0, tumor=0.0):
    row["vote_score_infection"] = f"{infection:.12g}"
    row["vote_score_tumor"] = f"{tumor:.12g}"


def set_no_candidate(row, note):
    row["pred_label"] = "no_candidate"
    row["pred_text"] = note
    row["vote_score_infection"] = "0"
    row["vote_score_tumor"] = "0"
    row["valid_component_votes"] = "0"
    row["strict_localization_hit"] = "0"
    row["relaxed_localization_hit"] = "0"
    row["det_iou_qwen_bbox"] = "0"
    row["det_iom_qwen_bbox"] = "0"
    row["center_hit"] = "0"
    return row


def choose_from_scores(infection, tumor):
    if infection > tumor:
        return "infection"
    if tumor > infection:
        return "tumor"
    return "no_candidate"


def variant_current(row):
    return dict(row)


def variant_no_adjacent_fallback(row):
    out = dict(row)
    if row.get("final_candidate_source") == "adjacent_slice_fallback":
        return set_no_candidate(out, "ablation_no_adjacent_fallback")
    return out


def variant_top1_component(row):
    out = dict(row)
    if row.get("pred_label") == "no_candidate":
        return out
    label = component_label(row, 1)
    if not label:
        return set_no_candidate(out, "ablation_top1_no_valid_component")
    score = component_score(row, 1)
    out["pred_label"] = label
    out["pred_text"] = f"ablation_top1_component:{label}@{score:.4g}"
    out["valid_component_votes"] = "1"
    set_scores(out, infection=score if label == "infection" else 0.0, tumor=score if label == "tumor" else 0.0)
    return out


def variant_first_valid(row):
    out = dict(row)
    if row.get("pred_label") == "no_candidate":
        return out
    for idx in COMPONENT_RANGE:
        label = component_label(row, idx)
        if label:
            score = component_score(row, idx)
            out["pred_label"] = label
            out["pred_text"] = f"ablation_first_valid_component_{idx}:{label}@{score:.4g}"
            out["valid_component_votes"] = "1"
            set_scores(out, infection=score if label == "infection" else 0.0, tumor=score if label == "tumor" else 0.0)
            return out
    return set_no_candidate(out, "ablation_first_valid_no_valid_component")


def variant_majority(row):
    out = dict(row)
    if row.get("pred_label") == "no_candidate":
        return out
    counts = Counter()
    score_sums = Counter()
    for idx in COMPONENT_RANGE:
        label = component_label(row, idx)
        if label:
            counts[label] += 1
            score_sums[label] += component_score(row, idx)
    if not counts:
        return set_no_candidate(out, "ablation_majority_no_valid_component")
    if counts["infection"] == counts["tumor"]:
        label = choose_from_scores(score_sums["infection"], score_sums["tumor"])
    else:
        label = counts.most_common(1)[0][0]
    out["pred_label"] = label
    out["pred_text"] = f"ablation_component_majority:{dict(counts)}"
    out["valid_component_votes"] = str(sum(counts.values()))
    set_scores(out, infection=float(score_sums["infection"]), tumor=float(score_sums["tumor"]))
    return out


def variant_unweighted_score(row):
    out = dict(row)
    if row.get("pred_label") == "no_candidate":
        return out
    score_sums = Counter()
    valid = 0
    for idx in COMPONENT_RANGE:
        label = component_label(row, idx)
        if label:
            valid += 1
            score_sums[label] += component_score(row, idx)
    if not valid:
        return set_no_candidate(out, "ablation_unweighted_score_no_valid_component")
    label = choose_from_scores(score_sums["infection"], score_sums["tumor"])
    out["pred_label"] = label
    out["pred_text"] = f"ablation_unweighted_score:{dict(score_sums)}"
    out["valid_component_votes"] = str(valid)
    set_scores(out, infection=float(score_sums["infection"]), tumor=float(score_sums["tumor"]))
    return out


def variant_weighted_score(row):
    out = dict(row)
    if row.get("pred_label") == "no_candidate":
        return out
    score_sums = Counter()
    valid = 0
    for idx in COMPONENT_RANGE:
        label = component_label(row, idx)
        if label:
            valid += 1
            score_sums[label] += component_weight(row, idx)
    if not valid:
        return set_no_candidate(out, "ablation_weighted_score_no_valid_component")
    label = choose_from_scores(score_sums["infection"], score_sums["tumor"])
    out["pred_label"] = label
    out["pred_text"] = f"ablation_weighted_score:{dict(score_sums)}"
    out["valid_component_votes"] = str(valid)
    set_scores(out, infection=float(score_sums["infection"]), tumor=float(score_sums["tumor"]))
    return out


VARIANTS = {
    "current_weighted_top3_fallback_r3": variant_current,
    "no_adjacent_fallback_weighted_top3": variant_no_adjacent_fallback,
    "top1_component_fallback_r3": variant_top1_component,
    "first_valid_component_fallback_r3": variant_first_valid,
    "component_majority_fallback_r3": variant_majority,
    "component_unweighted_score_fallback_r3": variant_unweighted_score,
    "component_weighted_score_fallback_r3": variant_weighted_score,
}


def slice_metrics(rows):
    total = len(rows)
    correct = sum(1 for row in rows if row.get("gt_label") == row.get("pred_label"))
    candidate_rows = [row for row in rows if row.get("pred_label") in CLS_LABELS]
    candidate_correct = sum(1 for row in candidate_rows if row.get("gt_label") == row.get("pred_label"))
    no_candidate = sum(1 for row in rows if row.get("pred_label") == "no_candidate")
    infection_rows = [row for row in rows if row.get("gt_label") == "infection"]
    tumor_rows = [row for row in rows if row.get("gt_label") == "tumor"]
    infection_correct = sum(1 for row in infection_rows if row.get("pred_label") == "infection")
    tumor_correct = sum(1 for row in tumor_rows if row.get("pred_label") == "tumor")
    strict_joint = sum(
        1
        for row in rows
        if row.get("gt_label") == row.get("pred_label") and safe_int(row.get("strict_localization_hit")) == 1
    )
    relaxed_joint = sum(
        1
        for row in rows
        if row.get("gt_label") == row.get("pred_label") and safe_int(row.get("relaxed_localization_hit")) == 1
    )
    det_iou05 = sum(1 for row in rows if safe_float(row.get("det_iou_qwen_bbox")) >= 0.5)
    det_iou03 = sum(1 for row in rows if safe_float(row.get("det_iou_qwen_bbox")) >= 0.3)
    relaxed_loc = sum(1 for row in rows if safe_int(row.get("relaxed_localization_hit")) == 1)
    strict_loc = sum(1 for row in rows if safe_int(row.get("strict_localization_hit")) == 1)
    confusion = Counter(f"{row.get('gt_label')}->{row.get('pred_label')}" for row in rows)
    infection_acc = safe_div(infection_correct, len(infection_rows))
    tumor_acc = safe_div(tumor_correct, len(tumor_rows))
    return {
        "total": total,
        "accuracy": safe_div(correct, total),
        "balanced_accuracy": (infection_acc + tumor_acc) / 2.0,
        "candidate_accuracy": safe_div(candidate_correct, len(candidate_rows)),
        "num_candidate_predictions": len(candidate_rows),
        "num_no_candidate": no_candidate,
        "no_candidate_rate": safe_div(no_candidate, total),
        "infection_accuracy": infection_acc,
        "tumor_accuracy": tumor_acc,
        "det_recall_iou0.3": safe_div(det_iou03, total),
        "det_recall_iou0.5": safe_div(det_iou05, total),
        "strict_localization_recall": safe_div(strict_loc, total),
        "relaxed_localization_recall": safe_div(relaxed_loc, total),
        "joint_acc_iou0.5": safe_div(strict_joint, total),
        "joint_acc_relaxed": safe_div(relaxed_joint, total),
        "confusion": dict(confusion),
    }


def aggregate_by(rows, level, strategy):
    groups = defaultdict(list)
    for row in rows:
        patient_id = row.get("patient_id") or row.get("case_id", "").split("_")[1]
        if level == "sequence":
            key = (patient_id, row.get("seq", ""))
        elif level == "patient":
            key = patient_id
        else:
            raise ValueError(level)
        groups[key].append(row)

    outputs = []
    for key, group_rows in sorted(groups.items()):
        agg = aggregate_rows(group_rows, strategy=strategy, tie_policy="uncertain")
        if level == "sequence":
            patient_id, seq = key
            outputs.append({"patient_id": patient_id, "seq": seq, **agg})
        else:
            outputs.append({"patient_id": key, **agg})
    return finish_metrics(outputs), outputs


def flatten_metric_rows(variant_name, metrics):
    rows = []
    for level, level_metrics in metrics.items():
        rows.append({
            "variant": variant_name,
            "level": level,
            "total": level_metrics.get("total"),
            "accuracy": level_metrics.get("accuracy", level_metrics.get("acc")),
            "balanced_accuracy": level_metrics.get("balanced_accuracy", level_metrics.get("balanced_acc")),
            "candidate_accuracy": level_metrics.get("candidate_accuracy", level_metrics.get("candidate_acc")),
            "no_candidate_rate": level_metrics.get("no_candidate_rate"),
            "infection_accuracy": level_metrics.get("infection_accuracy", ""),
            "tumor_accuracy": level_metrics.get("tumor_accuracy", ""),
            "det_recall_iou0.5": level_metrics.get("det_recall_iou0.5", ""),
            "joint_acc_iou0.5": level_metrics.get("joint_acc_iou0.5", ""),
            "joint_acc_relaxed": level_metrics.get("joint_acc_relaxed", ""),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build strategy ablations from saved component-level pipeline predictions.")
    parser.add_argument("--pred_csv", required=True, help="Final nnunet_qwen_remap_predictions.csv.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS.keys()),
        help="Comma-separated variant names. Use 'all' for every built-in variant.",
    )
    parser.add_argument("--aggregation_strategy", default="quality_weighted_vote")
    args = parser.parse_args()

    variants = list(VARIANTS) if args.variants == "all" else [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [x for x in variants if x not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(VARIANTS)}")

    source_rows = load_rows(Path(args.pred_csv))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "pred_csv": str(Path(args.pred_csv)),
        "output_dir": str(out_dir),
        "aggregation_strategy": args.aggregation_strategy,
        "num_source_rows": len(source_rows),
        "variants": {},
        "notes": {
            "current_weighted_top3_fallback_r3": "Uses the saved final pipeline predictions.",
            "no_adjacent_fallback_weighted_top3": "Rows sourced from adjacent-slice fallback are converted to no_candidate.",
            "top1_component_fallback_r3": "Uses the first/largest component prediction only.",
            "component_majority_fallback_r3": "Votes over up to three component labels without component weights.",
            "component_weighted_score_fallback_r3": "Recomputes weighted component fusion from saved component weights.",
        },
    }
    long_rows = []
    all_variant_rows = []

    for variant_name in variants:
        variant_rows = [VARIANTS[variant_name](row) for row in source_rows]
        variant_csv = out_dir / f"{variant_name}_slice_predictions.csv"
        write_csv(variant_csv, variant_rows, list(variant_rows[0].keys()) if variant_rows else [])

        seq_metrics, seq_outputs = aggregate_by(variant_rows, "sequence", args.aggregation_strategy)
        patient_metrics, patient_outputs = aggregate_by(variant_rows, "patient", args.aggregation_strategy)
        write_csv(out_dir / f"{variant_name}_sequence_predictions.csv", seq_outputs, list(seq_outputs[0].keys()) if seq_outputs else [])
        write_csv(out_dir / f"{variant_name}_patient_predictions.csv", patient_outputs, list(patient_outputs[0].keys()) if patient_outputs else [])

        metrics = {
            "slice": slice_metrics(variant_rows),
            "sequence": seq_metrics,
            "patient": patient_metrics,
        }
        summary["variants"][variant_name] = metrics
        long_rows.extend(flatten_metric_rows(variant_name, metrics))
        all_variant_rows.extend({"variant": variant_name, **row} for row in variant_rows)

    write_csv(
        out_dir / "pipeline_strategy_ablation_long.csv",
        long_rows,
        [
            "variant",
            "level",
            "total",
            "accuracy",
            "balanced_accuracy",
            "candidate_accuracy",
            "no_candidate_rate",
            "infection_accuracy",
            "tumor_accuracy",
            "det_recall_iou0.5",
            "joint_acc_iou0.5",
            "joint_acc_relaxed",
        ],
    )
    write_csv(out_dir / "pipeline_strategy_ablation_all_slice_predictions.csv", all_variant_rows, list(all_variant_rows[0].keys()) if all_variant_rows else [])
    with open(out_dir / "pipeline_strategy_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output_dir": str(out_dir),
        "summary_json": str(out_dir / "pipeline_strategy_ablation_summary.json"),
        "long_csv": str(out_dir / "pipeline_strategy_ablation_long.csv"),
        "variants": variants,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
