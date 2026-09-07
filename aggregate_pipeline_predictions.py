#!/usr/bin/env python
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ["infection", "tumor", "no_candidate", "uncertain"]
CLS_LABELS = ["infection", "tumor"]


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


def infer_label_from_case(case_id):
    case_id = str(case_id)
    if case_id.startswith("infection"):
        return "infection"
    if case_id.startswith("tumor"):
        return "tumor"
    return ""


def choose_label(scores, tie_policy="uncertain"):
    infection = scores.get("infection", 0.0)
    tumor = scores.get("tumor", 0.0)
    no_candidate = scores.get("no_candidate", 0.0)
    if infection == 0 and tumor == 0:
        if no_candidate > 0:
            return "no_candidate"
        return "uncertain"
    if infection > tumor:
        return "infection"
    if tumor > infection:
        return "tumor"
    if tie_policy in CLS_LABELS:
        return tie_policy
    return "uncertain"


def row_vote_scores(row):
    pred = row.get("pred_label", "")
    infection = safe_float(row.get("vote_score_infection"), 0.0)
    tumor = safe_float(row.get("vote_score_tumor"), 0.0)
    if infection <= 0 and tumor <= 0 and pred in CLS_LABELS:
        if pred == "infection":
            infection = 1.0
        else:
            tumor = 1.0
    return {"infection": infection, "tumor": tumor}


def candidate_quality(row, max_area=0.0):
    """Uses only inference-time available fields, not GT-derived IoU/IoM."""
    area = safe_float(row.get("candidate_slice_area"), 0.0)
    area_factor = math.log1p(area) / math.log1p(max_area) if max_area > 0 else 1.0
    distance = abs(safe_float(row.get("candidate_slice_distance"), 0.0))
    distance_factor = 1.0 / (1.0 + distance)
    votes = max(safe_float(row.get("valid_component_votes"), 1.0), 1.0)
    votes_factor = min(votes, 3.0) / 3.0

    source = row.get("final_candidate_source", "")
    if source == "filtered_components":
        source_factor = 1.0
    elif source == "adjacent_slice_fallback":
        source_factor = 0.8
    elif source:
        source_factor = 0.7
    else:
        source_factor = 0.5

    if str(row.get("pred_mask_empty", "")).lower() == "true":
        source_factor *= 0.5
    return area_factor * distance_factor * votes_factor * source_factor


def aggregate_rows(rows, strategy, tie_policy="uncertain"):
    gt_counts = Counter(row.get("gt_label") or infer_label_from_case(row.get("case_id")) for row in rows)
    gt_counts.pop("", None)
    gt_label = gt_counts.most_common(1)[0][0] if gt_counts else ""

    max_area = max((safe_float(row.get("candidate_slice_area"), 0.0) for row in rows), default=0.0)

    if strategy == "majority_vote":
        scores = Counter()
        for row in rows:
            pred = row.get("pred_label", "")
            if pred in CLS_LABELS:
                scores[pred] += 1.0
            elif pred == "no_candidate":
                scores["no_candidate"] += 1.0
        pred_label = choose_label(scores, tie_policy=tie_policy)
        confidence = safe_div(max(scores.get("infection", 0.0), scores.get("tumor", 0.0)), sum(scores.values()))

    elif strategy == "score_weighted_vote":
        scores = Counter()
        no_candidate = 0
        for row in rows:
            if row.get("pred_label") == "no_candidate":
                no_candidate += 1
            vote_scores = row_vote_scores(row)
            scores["infection"] += vote_scores["infection"]
            scores["tumor"] += vote_scores["tumor"]
        scores["no_candidate"] = no_candidate
        pred_label = choose_label(scores, tie_policy=tie_policy)
        confidence = safe_div(max(scores.get("infection", 0.0), scores.get("tumor", 0.0)), scores.get("infection", 0.0) + scores.get("tumor", 0.0))

    elif strategy == "quality_weighted_vote":
        scores = Counter()
        no_candidate = 0
        for row in rows:
            if row.get("pred_label") == "no_candidate":
                no_candidate += 1
            quality = candidate_quality(row, max_area=max_area)
            vote_scores = row_vote_scores(row)
            scores["infection"] += vote_scores["infection"] * quality
            scores["tumor"] += vote_scores["tumor"] * quality
        scores["no_candidate"] = no_candidate
        pred_label = choose_label(scores, tie_policy=tie_policy)
        confidence = safe_div(max(scores.get("infection", 0.0), scores.get("tumor", 0.0)), scores.get("infection", 0.0) + scores.get("tumor", 0.0))

    elif strategy == "best_slice_vote":
        best_row = None
        best_score = -1.0
        for row in rows:
            vote_scores = row_vote_scores(row)
            model_strength = max(vote_scores["infection"], vote_scores["tumor"], 0.0)
            score = candidate_quality(row, max_area=max_area) * max(model_strength, 1.0)
            if score > best_score:
                best_score = score
                best_row = row
        pred_label = best_row.get("pred_label", "uncertain") if best_row else "uncertain"
        if pred_label not in LABELS:
            pred_label = "uncertain"
        confidence = best_score if best_score >= 0 else 0.0
        scores = row_vote_scores(best_row) if best_row else {}

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    pred_counts = Counter(row.get("pred_label", "") for row in rows)
    return {
        "gt_label": gt_label,
        "pred_label": pred_label,
        "is_correct": int(gt_label == pred_label),
        "confidence": confidence,
        "num_slices": len(rows),
        "num_candidate_slices": sum(1 for row in rows if row.get("pred_label") in CLS_LABELS),
        "num_no_candidate_slices": sum(1 for row in rows if row.get("pred_label") == "no_candidate"),
        "score_infection": float(scores.get("infection", 0.0)),
        "score_tumor": float(scores.get("tumor", 0.0)),
        "slice_pred_counts": json.dumps(dict(pred_counts), ensure_ascii=False, sort_keys=True),
    }


def finish_metrics(rows):
    total = len(rows)
    correct = sum(1 for row in rows if row["gt_label"] == row["pred_label"])
    no_candidate = sum(1 for row in rows if row["pred_label"] == "no_candidate")
    uncertain = sum(1 for row in rows if row["pred_label"] == "uncertain")
    candidate_rows = [row for row in rows if row["pred_label"] in CLS_LABELS]
    candidate_correct = sum(1 for row in candidate_rows if row["gt_label"] == row["pred_label"])
    confusion = Counter(f"{row['gt_label']}->{row['pred_label']}" for row in rows)

    by_label = defaultdict(lambda: {"total": 0, "correct": 0})
    by_seq = defaultdict(lambda: {"total": 0, "correct": 0})
    by_label_seq = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        is_correct = int(row["gt_label"] == row["pred_label"])
        label = row.get("gt_label", "")
        seq = row.get("seq", "")
        by_label[label]["total"] += 1
        by_label[label]["correct"] += is_correct
        if seq:
            by_seq[seq]["total"] += 1
            by_seq[seq]["correct"] += is_correct
            by_label_seq[f"{label}|{seq}"]["total"] += 1
            by_label_seq[f"{label}|{seq}"]["correct"] += is_correct

    def finish(group):
        return {
            key: {
                "correct": val["correct"],
                "total": val["total"],
                "acc": safe_div(val["correct"], val["total"]),
            }
            for key, val in sorted(group.items())
        }

    infection_acc = safe_div(by_label["infection"]["correct"], by_label["infection"]["total"])
    tumor_acc = safe_div(by_label["tumor"]["correct"], by_label["tumor"]["total"])
    return {
        "total": total,
        "correct": correct,
        "acc": safe_div(correct, total),
        "balanced_acc": (infection_acc + tumor_acc) / 2.0 if (by_label["infection"]["total"] and by_label["tumor"]["total"]) else 0.0,
        "candidate_acc": safe_div(candidate_correct, len(candidate_rows)),
        "num_candidate_predictions": len(candidate_rows),
        "num_no_candidate": no_candidate,
        "num_uncertain": uncertain,
        "no_candidate_rate": safe_div(no_candidate, total),
        "uncertain_rate": safe_div(uncertain, total),
        "confusion": dict(confusion),
        "by_label": finish(by_label),
        "by_seq": finish(by_seq),
        "by_label_seq": finish(by_label_seq),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate slice-level pipeline predictions into sequence/patient-level decisions.")
    parser.add_argument("--pred_csv", required=True, help="Slice-level nnunet_qwen_remap_predictions.csv.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--strategies",
        default="majority_vote,score_weighted_vote,quality_weighted_vote,best_slice_vote",
        help="Comma-separated strategies.",
    )
    parser.add_argument("--tie_policy", default="uncertain", choices=["uncertain", "infection", "tumor"])
    args = parser.parse_args()

    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    output_dir = Path(args.output_dir)

    with open(args.pred_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    seq_groups = defaultdict(list)
    patient_groups = defaultdict(list)
    for row in rows:
        patient_id = row.get("patient_id") or row.get("case_id", "").split("_")[1]
        seq = row.get("seq", "")
        seq_groups[(patient_id, seq)].append(row)
        patient_groups[patient_id].append(row)

    sequence_outputs = []
    patient_outputs = []
    metrics = {
        "pred_csv": str(Path(args.pred_csv)),
        "strategies": strategies,
        "tie_policy": args.tie_policy,
        "num_slice_rows": len(rows),
        "num_sequence_groups": len(seq_groups),
        "num_patient_groups": len(patient_groups),
        "sequence_level": {},
        "patient_level": {},
        "notes": {
            "quality_weighted_vote": "Uses only inference-time fields: candidate area, candidate-source type, adjacent-slice distance, and component vote count.",
            "best_slice_vote": "Selects the slice with the highest inference-time quality score, then uses its predicted class.",
            "no_gt_leakage": "Aggregation decisions do not use det_iou, det_iom, gt_coverage, pred_precision, center_hit, or GT bbox fields.",
        },
    }

    for strategy in strategies:
        strategy_seq_rows = []
        for (patient_id, seq), group_rows in sorted(seq_groups.items()):
            agg = aggregate_rows(group_rows, strategy=strategy, tie_policy=args.tie_policy)
            strategy_seq_rows.append({
                "strategy": strategy,
                "patient_id": patient_id,
                "seq": seq,
                **agg,
            })
        sequence_outputs.extend(strategy_seq_rows)
        metrics["sequence_level"][strategy] = finish_metrics(strategy_seq_rows)

        strategy_patient_rows = []
        for patient_id, group_rows in sorted(patient_groups.items()):
            agg = aggregate_rows(group_rows, strategy=strategy, tie_policy=args.tie_policy)
            seqs = sorted({row.get("seq", "") for row in group_rows if row.get("seq", "")})
            strategy_patient_rows.append({
                "strategy": strategy,
                "patient_id": patient_id,
                "seqs": ",".join(seqs),
                **agg,
            })
        patient_outputs.extend(strategy_patient_rows)
        metrics["patient_level"][strategy] = finish_metrics(strategy_patient_rows)

    sequence_csv = output_dir / "sequence_level_predictions.csv"
    patient_csv = output_dir / "patient_level_predictions.csv"
    metrics_json = output_dir / "aggregation_metrics.json"

    common_fields = [
        "strategy",
        "patient_id",
        "seq",
        "seqs",
        "gt_label",
        "pred_label",
        "is_correct",
        "confidence",
        "num_slices",
        "num_candidate_slices",
        "num_no_candidate_slices",
        "score_infection",
        "score_tumor",
        "slice_pred_counts",
    ]
    write_csv(sequence_csv, sequence_outputs, [x for x in common_fields if x != "seqs"])
    write_csv(patient_csv, patient_outputs, [x for x in common_fields if x != "seq"])

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "sequence_csv": str(sequence_csv),
        "patient_csv": str(patient_csv),
        "metrics_json": str(metrics_json),
        "num_slice_rows": len(rows),
        "num_sequence_groups": len(seq_groups),
        "num_patient_groups": len(patient_groups),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
