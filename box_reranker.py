import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from qwen_stage2_classifier import LesionRecord, load_records, make_non_empty_bbox
from yolo_qwen_pipeline import CandidateBox, compute_iou, find_candidates, load_candidates


FEATURE_NAMES = [
    "rank",
    "conf",
    "x1_norm",
    "y1_norm",
    "x2_norm",
    "y2_norm",
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "area_norm",
    "aspect_ratio",
    "dist_center_norm",
]


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def require_sklearn():
    try:
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise SystemExit("scikit-learn is required. Install it with: pip install scikit-learn") from exc
    return {
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "roc_auc_score": roc_auc_score,
    }


def feature_vector(record: LesionRecord, candidate: CandidateBox) -> List[float]:
    x1, y1, x2, y2 = make_non_empty_bbox(candidate.bbox, record.width, record.height)
    width = max(float(record.width), 1.0)
    height = max(float(record.height), 1.0)
    box_w = max(float(x2 - x1), 1.0)
    box_h = max(float(y2 - y1), 1.0)
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    dx = cx / width - 0.5
    dy = cy / height - 0.5
    return [
        float(candidate.rank),
        float(candidate.conf),
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height,
        cx / width,
        cy / height,
        box_w / width,
        box_h / height,
        (box_w * box_h) / (width * height),
        box_w / box_h,
        float((dx * dx + dy * dy) ** 0.5),
    ]


def candidate_iou(record: LesionRecord, candidate: CandidateBox) -> float:
    bbox = make_non_empty_bbox(candidate.bbox, record.width, record.height)
    return compute_iou(bbox, record.bbox)


def build_samples(
    detcls_json: str,
    pred_csv: str,
    top_k: int,
    pos_iou: float,
    neg_iou: float,
    include_mid_as_negative: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    records = load_records(detcls_json)
    candidates = load_candidates(pred_csv, top_k)
    features: List[List[float]] = []
    labels: List[int] = []
    rows: List[Dict[str, object]] = []
    skipped_no_candidate = 0
    ignored_mid_iou = 0

    for index, record in enumerate(records):
        image_candidates = find_candidates(candidates, record, index)
        if not image_candidates:
            skipped_no_candidate += 1
            continue
        for candidate in image_candidates:
            iou = candidate_iou(record, candidate)
            if iou >= pos_iou:
                label = 1
            elif iou < neg_iou or include_mid_as_negative:
                label = 0
            else:
                ignored_mid_iou += 1
                continue
            x1, y1, x2, y2 = make_non_empty_bbox(candidate.bbox, record.width, record.height)
            features.append(feature_vector(record, candidate))
            labels.append(label)
            rows.append(
                {
                    "record_index": index,
                    "sample_id": record.sample_id,
                    "image_path": record.image_path,
                    "rank": candidate.rank,
                    "conf": candidate.conf,
                    "iou": iou,
                    "label": label,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    log(f"Built {len(labels)} samples from {detcls_json}; skipped_no_candidate={skipped_no_candidate}; ignored_mid_iou={ignored_mid_iou}.")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), rows


def make_model(model_name: str, seed: int, sklearn):
    if model_name == "extra_trees":
        return sklearn["ExtraTreesClassifier"](
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "random_forest":
        return sklearn["RandomForestClassifier"](
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "gbdt":
        return sklearn["GradientBoostingClassifier"](random_state=seed)
    if model_name == "logreg":
        return sklearn["LogisticRegression"](
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def predict_scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(features).astype(np.float32)


def evaluate_selection(
    model,
    detcls_json: str,
    pred_csv: str,
    top_k: int,
    output_dir: Path,
) -> Dict[str, object]:
    records = load_records(detcls_json)
    candidates = load_candidates(pred_csv, top_k)
    total = 0
    has_candidate = 0
    selected_hits = {0.3: 0, 0.5: 0}
    top1_hits = {0.3: 0, 0.5: 0}
    oracle_hits = {0.3: 0, 0.5: 0}
    rank_counter = Counter()
    rows: List[Dict[str, object]] = []

    for index, record in enumerate(records):
        total += 1
        image_candidates = find_candidates(candidates, record, index)
        if not image_candidates:
            rows.append({"image_path": record.image_path, "selected_rank": "", "selected_score": "", "selected_iou": 0.0})
            continue
        has_candidate += 1
        features = np.asarray([feature_vector(record, candidate) for candidate in image_candidates], dtype=np.float32)
        scores = predict_scores(model, features)
        best_index = int(np.argmax(scores))
        selected = image_candidates[best_index]
        selected_iou = candidate_iou(record, selected)
        top1_iou = candidate_iou(record, image_candidates[0])
        oracle_iou = max(candidate_iou(record, candidate) for candidate in image_candidates)
        rank_counter[str(selected.rank)] += 1
        for threshold in (0.3, 0.5):
            selected_hits[threshold] += int(selected_iou >= threshold)
            top1_hits[threshold] += int(top1_iou >= threshold)
            oracle_hits[threshold] += int(oracle_iou >= threshold)
        x1, y1, x2, y2 = make_non_empty_bbox(selected.bbox, record.width, record.height)
        rows.append(
            {
                "image_path": record.image_path,
                "selected_rank": selected.rank,
                "selected_score": float(scores[best_index]),
                "selected_conf": selected.conf,
                "selected_iou": selected_iou,
                "top1_iou": top1_iou,
                "oracle_iou": oracle_iou,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "box_reranker_selected_boxes.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "selected_rank", "selected_score", "selected_conf", "selected_iou", "top1_iou", "oracle_iou", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "total_images": total,
        "images_with_candidate": has_candidate,
        "candidate_coverage": has_candidate / max(total, 1),
        "selected_det_recall_iou0.3": selected_hits[0.3] / max(total, 1),
        "selected_det_recall_iou0.5": selected_hits[0.5] / max(total, 1),
        "top1_det_recall_iou0.3": top1_hits[0.3] / max(total, 1),
        "top1_det_recall_iou0.5": top1_hits[0.5] / max(total, 1),
        "oracle_det_recall_iou0.3": oracle_hits[0.3] / max(total, 1),
        "oracle_det_recall_iou0.5": oracle_hits[0.5] / max(total, 1),
        "selected_rank": dict(rank_counter),
        "selected_boxes_csv": str(rows_path),
    }
    return metrics


def train(args: argparse.Namespace) -> None:
    sklearn = require_sklearn()
    x_train, y_train, _ = build_samples(
        args.train_json,
        args.train_pred_csv,
        args.top_k,
        args.pos_iou,
        args.neg_iou,
        args.include_mid_as_negative,
    )
    x_val, y_val, _ = build_samples(
        args.val_json,
        args.val_pred_csv,
        args.top_k,
        args.pos_iou,
        args.neg_iou,
        args.include_mid_as_negative,
    )
    model = make_model(args.model, args.seed, sklearn)
    model.fit(x_train, y_train)

    train_scores = predict_scores(model, x_train)
    val_scores = predict_scores(model, x_val)
    try:
        train_auc = float(sklearn["roc_auc_score"](y_train, train_scores))
        val_auc = float(sklearn["roc_auc_score"](y_val, val_scores))
    except ValueError:
        train_auc = None
        val_auc = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "box_reranker.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"model": model, "feature_names": FEATURE_NAMES, "args": vars(args)}, f)

    metrics = evaluate_selection(model, args.val_json, args.val_pred_csv, args.top_k, output_dir)
    metrics.update(
        {
            "model": args.model,
            "feature_names": FEATURE_NAMES,
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "train_label_distribution": {str(k): int(v) for k, v in Counter(y_train.tolist()).items()},
            "val_label_distribution": {str(k): int(v) for k, v in Counter(y_val.tolist()).items()},
            "train_auc": train_auc,
            "val_auc": val_auc,
            "model_path": str(model_path),
        }
    )
    metrics_path = output_dir / "box_reranker_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    log(f"Saved model to {model_path}")
    log(f"Saved metrics to {metrics_path}")


def eval_model(args: argparse.Namespace) -> None:
    require_sklearn()
    with open(args.model_path, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    output_dir = Path(args.output_dir)
    metrics = evaluate_selection(model, args.val_json, args.val_pred_csv, args.top_k, output_dir)
    metrics_path = output_dir / "box_reranker_eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    log(f"Saved metrics to {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight geometry/confidence reranker for YOLO lesion candidate boxes.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train_json", required=True)
    train_parser.add_argument("--train_pred_csv", required=True)
    train_parser.add_argument("--val_json", required=True)
    train_parser.add_argument("--val_pred_csv", required=True)
    train_parser.add_argument("--output_dir", default="output/box_reranker")
    train_parser.add_argument("--top_k", type=int, default=10)
    train_parser.add_argument("--pos_iou", type=float, default=0.5)
    train_parser.add_argument("--neg_iou", type=float, default=0.3)
    train_parser.add_argument("--include_mid_as_negative", action="store_true")
    train_parser.add_argument("--model", choices=["extra_trees", "random_forest", "gbdt", "logreg"], default="extra_trees")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model_path", required=True)
    eval_parser.add_argument("--val_json", required=True)
    eval_parser.add_argument("--val_pred_csv", required=True)
    eval_parser.add_argument("--output_dir", default="output/box_reranker_eval")
    eval_parser.add_argument("--top_k", type=int, default=10)
    eval_parser.set_defaults(func=eval_model)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
