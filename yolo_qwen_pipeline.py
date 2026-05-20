import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen_stage2_classifier import (
    LesionRecord,
    crop_image,
    expand_bbox,
    load_model_and_processor,
    load_records,
    make_non_empty_bbox,
    normalize_label,
    predict_label,
)


@dataclass
class CandidateBox:
    image_path: str
    rank: int
    conf: float
    bbox: Tuple[int, int, int, int]


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def image_key(path: str) -> str:
    return Path(path).name


def clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def compute_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def load_candidates(csv_path: str, top_k: int) -> Dict[str, List[CandidateBox]]:
    by_image: Dict[str, List[CandidateBox]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("rank") or not row.get("x1"):
                continue
            try:
                candidate = CandidateBox(
                    image_path=row["image_path"],
                    rank=int(row["rank"]),
                    conf=float(row["conf"]),
                    bbox=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
                )
            except (KeyError, ValueError):
                continue
            by_image[image_key(candidate.image_path)].append(candidate)

    for key in list(by_image.keys()):
        by_image[key] = sorted(by_image[key], key=lambda x: (x.rank, -x.conf))[:top_k]
    log(f"Loaded YOLO candidates for {len(by_image)} images from {csv_path}.")
    return by_image


def load_qwen(base_model: str, adapter_path: str, load_in_4bit: bool):
    base, processor = load_model_and_processor(base_model, load_in_4bit, gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, processor


def classify_candidate(
    model,
    processor,
    record: LesionRecord,
    candidate: CandidateBox,
    crop_expand_ratio: float,
    input_mode: str,
    max_new_tokens: int,
) -> Tuple[str, Optional[str]]:
    bbox = make_non_empty_bbox(candidate.bbox, record.width, record.height)
    if input_mode == "bbox_prompt":
        return predict_label(
            model,
            processor,
            record.image_path,
            record.seq,
            bbox=bbox,
            input_mode="bbox_prompt",
            max_new_tokens=max_new_tokens,
        )
    crop = crop_image(record.image_path, bbox, crop_expand_ratio)
    return predict_label(
        model,
        processor,
        crop,
        record.seq,
        input_mode="crop",
        max_new_tokens=max_new_tokens,
    )


def evaluate(args: argparse.Namespace) -> None:
    records = load_records(args.val_json)
    candidates = load_candidates(args.pred_csv, args.top_k)
    model, processor = load_qwen(args.base_model, args.adapter_path, args.load_in_4bit)

    total = 0
    has_candidate = 0
    cls_correct = 0
    det_hits = {0.3: 0, 0.5: 0}
    joint_hits = {0.3: 0, 0.5: 0}
    topk_det_hits = {0.3: 0, 0.5: 0}
    selected_rows: List[Dict[str, object]] = []
    patient_stats = defaultdict(lambda: {"total": 0, "joint03": 0, "joint05": 0})

    for record in tqdm(records, desc="Eval YOLO+Qwen", ncols=120):
        total += 1
        patient_stats[record.patient_id]["total"] += 1
        image_candidates = candidates.get(image_key(record.image_path), [])

        if not image_candidates:
            selected_rows.append(
                {
                    "image_path": record.image_path,
                    "gt_label": record.label,
                    "pred_label": "",
                    "pred_text": "",
                    "rank": "",
                    "conf": "",
                    "iou": 0.0,
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                }
            )
            continue

        has_candidate += 1
        for threshold in topk_det_hits:
            topk_det_hits[threshold] += int(any(compute_iou(c.bbox, record.bbox) >= threshold for c in image_candidates))

        selected = image_candidates[0]
        if args.selection == "best_iou":
            selected = max(image_candidates, key=lambda c: compute_iou(c.bbox, record.bbox))

        pred_text, pred_label = classify_candidate(
            model,
            processor,
            record,
            selected,
            crop_expand_ratio=args.crop_expand_ratio,
            input_mode=args.input_mode,
            max_new_tokens=args.max_new_tokens,
        )
        iou = compute_iou(selected.bbox, record.bbox)
        label_ok = pred_label == record.label
        cls_correct += int(label_ok)
        for threshold in det_hits:
            det_ok = iou >= threshold
            det_hits[threshold] += int(det_ok)
            joint_hits[threshold] += int(det_ok and label_ok)
        patient_stats[record.patient_id]["joint03"] += int(iou >= 0.3 and label_ok)
        patient_stats[record.patient_id]["joint05"] += int(iou >= 0.5 and label_ok)

        selected_rows.append(
            {
                "image_path": record.image_path,
                "gt_label": record.label,
                "pred_label": pred_label or "",
                "pred_text": pred_text,
                "rank": selected.rank,
                "conf": selected.conf,
                "iou": iou,
                "x1": selected.bbox[0],
                "y1": selected.bbox[1],
                "x2": selected.bbox[2],
                "y2": selected.bbox[3],
            }
        )

    patient_joint03 = 0
    patient_joint05 = 0
    for stats in patient_stats.values():
        patient_joint03 += int(stats["joint03"] / max(stats["total"], 1) > args.patient_positive_ratio)
        patient_joint05 += int(stats["joint05"] / max(stats["total"], 1) > args.patient_positive_ratio)

    metrics = {
        "total_images": total,
        "images_with_candidate": has_candidate,
        "candidate_coverage": has_candidate / max(total, 1),
        "selection": args.selection,
        "top_k": args.top_k,
        "crop_expand_ratio": args.crop_expand_ratio,
        "selected_det_recall_iou0.3": det_hits[0.3] / max(total, 1),
        "selected_det_recall_iou0.5": det_hits[0.5] / max(total, 1),
        "topk_oracle_det_recall_iou0.3": topk_det_hits[0.3] / max(total, 1),
        "topk_oracle_det_recall_iou0.5": topk_det_hits[0.5] / max(total, 1),
        "cls_acc_on_selected": cls_correct / max(total, 1),
        "joint_acc_iou0.3": joint_hits[0.3] / max(total, 1),
        "joint_acc_iou0.5": joint_hits[0.5] / max(total, 1),
        "total_patients": len(patient_stats),
        "patient_joint_acc_iou0.3": patient_joint03 / max(len(patient_stats), 1),
        "patient_joint_acc_iou0.5": patient_joint05 / max(len(patient_stats), 1),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "pipeline_metrics.json"
    rows_path = output_dir / "pipeline_predictions.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "gt_label", "pred_label", "pred_text", "rank", "conf", "iou", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    log(f"Saved metrics to {metrics_path}")
    log(f"Saved predictions to {rows_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector + Qwen crop classifier pipeline.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--output_dir", default="output/yolo_qwen_pipeline")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--selection", choices=["top1", "best_iou"], default="top1")
    parser.add_argument("--input_mode", choices=["bbox_prompt", "crop"], default="bbox_prompt")
    parser.add_argument("--crop_expand_ratio", type=float, default=0.2)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--patient_positive_ratio", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
