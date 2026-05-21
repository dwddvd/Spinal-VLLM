import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
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


@dataclass
class CandidatePrediction:
    candidate: CandidateBox
    iou: float
    pred_text: str
    pred_label: Optional[str]


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def image_key(path: str) -> str:
    return Path(path).name.lower()


def safe_name(raw: str, fallback: str) -> str:
    name = raw.strip() or fallback
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("._")
    return name or fallback


def candidate_keys(path: str) -> List[str]:
    image_path = Path(path)
    stem = image_path.stem
    suffix = image_path.suffix.lower()
    keys = {
        image_path.name.lower(),
        stem.lower(),
    }
    without_index = re.sub(r"^\d{6}_", "", stem)
    keys.add(without_index.lower())
    if suffix:
        keys.add(f"{without_index}{suffix}".lower())
    return list(keys)


def record_keys(record: LesionRecord, index: int) -> List[str]:
    image_path = Path(record.image_path)
    suffix = image_path.suffix.lower()
    original_stem = image_path.stem
    safe_sample = safe_name(record.sample_id, original_stem)
    yolo_stem = f"{index:06d}_{safe_sample}"
    keys = {
        image_path.name.lower(),
        original_stem.lower(),
        safe_sample.lower(),
        yolo_stem.lower(),
    }
    if suffix:
        keys.add(f"{safe_sample}{suffix}".lower())
        keys.add(f"{yolo_stem}{suffix}".lower())
    return list(keys)


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
    image_names = set()
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
            image_names.add(image_key(candidate.image_path))
            for key in candidate_keys(candidate.image_path):
                by_image[key].append(candidate)

    for key in list(by_image.keys()):
        by_image[key] = sorted(by_image[key], key=lambda x: (x.rank, -x.conf))[:top_k]
    log(f"Loaded YOLO candidates for {len(image_names)} images from {csv_path}.")
    return by_image


def find_candidates(
    candidates: Dict[str, List[CandidateBox]],
    record: LesionRecord,
    index: int,
) -> List[CandidateBox]:
    for key in record_keys(record, index):
        if key in candidates:
            return candidates[key]
    return []


def parse_top_ks(raw: str) -> List[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    values = [value for value in values if value > 0]
    if not values:
        raise ValueError("--top_ks must contain at least one positive integer.")
    return values


def init_metric_bucket() -> Dict[str, object]:
    return {
        "images_with_candidate": 0,
        "det_hits": {0.3: 0, 0.5: 0},
        "joint_hits": {0.3: 0, 0.5: 0},
        "top1_cls_correct": 0,
        "top1_det_hits": {0.3: 0, 0.5: 0},
        "top1_joint_hits": {0.3: 0, 0.5: 0},
        "confusion_any_iou0.3": Counter(),
        "confusion_any_iou0.5": Counter(),
        "confusion_top1": Counter(),
    }


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
    top_ks = parse_top_ks(args.top_ks)
    max_top_k = max(top_ks)
    records = load_records(args.val_json)
    candidates = load_candidates(args.pred_csv, max_top_k)
    model, processor = load_qwen(args.base_model, args.adapter_path, args.load_in_4bit)

    total = 0
    buckets = {k: init_metric_bucket() for k in top_ks}
    selected_rows: List[Dict[str, object]] = []

    for record_index, record in enumerate(tqdm(records, desc="Eval YOLO+Qwen", ncols=120)):
        total += 1
        image_candidates = find_candidates(candidates, record, record_index)

        if not image_candidates:
            for k in top_ks:
                buckets[k]["confusion_top1"][f"{record.label}->missing"] += 1
                for threshold in (0.3, 0.5):
                    buckets[k][f"confusion_any_iou{threshold}"][f"{record.label}->missed"] += 1
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

        predictions: List[CandidatePrediction] = []
        for candidate in image_candidates:
            pred_text, pred_label = classify_candidate(
                model,
                processor,
                record,
                candidate,
                crop_expand_ratio=args.crop_expand_ratio,
                input_mode=args.input_mode,
                max_new_tokens=args.max_new_tokens,
            )
            iou = compute_iou(candidate.bbox, record.bbox)
            predictions.append(CandidatePrediction(candidate=candidate, iou=iou, pred_text=pred_text, pred_label=pred_label))
            selected_rows.append(
                {
                    "image_path": record.image_path,
                    "gt_label": record.label,
                    "pred_label": pred_label or "",
                    "pred_text": pred_text,
                    "rank": candidate.rank,
                    "conf": candidate.conf,
                    "iou": iou,
                    "x1": candidate.bbox[0],
                    "y1": candidate.bbox[1],
                    "x2": candidate.bbox[2],
                    "y2": candidate.bbox[3],
                }
            )

        top1 = predictions[0]
        for k in top_ks:
            subset = predictions[:k]
            bucket = buckets[k]
            bucket["images_with_candidate"] += 1
            bucket["top1_cls_correct"] += int(top1.pred_label == record.label)
            bucket["confusion_top1"][f"{record.label}->{top1.pred_label or 'invalid'}"] += 1
            for threshold in (0.3, 0.5):
                det_ok = any(item.iou >= threshold for item in subset)
                joint_ok = any(item.iou >= threshold and item.pred_label == record.label for item in subset)
                top1_det_ok = top1.iou >= threshold
                top1_joint_ok = top1.iou >= threshold and top1.pred_label == record.label
                bucket["det_hits"][threshold] += int(det_ok)
                bucket["joint_hits"][threshold] += int(joint_ok)
                bucket["top1_det_hits"][threshold] += int(top1_det_ok)
                bucket["top1_joint_hits"][threshold] += int(top1_joint_ok)
                matching = [item for item in subset if item.iou >= threshold]
                if matching:
                    best_match = max(matching, key=lambda item: item.iou)
                    bucket[f"confusion_any_iou{threshold}"][f"{record.label}->{best_match.pred_label or 'invalid'}"] += 1
                else:
                    bucket[f"confusion_any_iou{threshold}"][f"{record.label}->missed"] += 1

    metrics = {
        "total_images": total,
        "top_ks": top_ks,
        "crop_expand_ratio": args.crop_expand_ratio,
        "input_mode": args.input_mode,
        "per_top_k": {},
    }
    for k in top_ks:
        bucket = buckets[k]
        metrics["per_top_k"][f"top{k}"] = {
            "images_with_candidate": bucket["images_with_candidate"],
            "candidate_coverage": bucket["images_with_candidate"] / max(total, 1),
            "det_recall_iou0.3": bucket["det_hits"][0.3] / max(total, 1),
            "det_recall_iou0.5": bucket["det_hits"][0.5] / max(total, 1),
            "joint_acc_iou0.3": bucket["joint_hits"][0.3] / max(total, 1),
            "joint_acc_iou0.5": bucket["joint_hits"][0.5] / max(total, 1),
            "top1_cls_acc": bucket["top1_cls_correct"] / max(total, 1),
            "top1_det_recall_iou0.3": bucket["top1_det_hits"][0.3] / max(total, 1),
            "top1_det_recall_iou0.5": bucket["top1_det_hits"][0.5] / max(total, 1),
            "top1_joint_acc_iou0.3": bucket["top1_joint_hits"][0.3] / max(total, 1),
            "top1_joint_acc_iou0.5": bucket["top1_joint_hits"][0.5] / max(total, 1),
            "confusion_top1": dict(bucket["confusion_top1"]),
            "confusion_any_iou0.3": dict(bucket["confusion_any_iou0.3"]),
            "confusion_any_iou0.5": dict(bucket["confusion_any_iou0.5"]),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "pipeline_topk_metrics.json"
    rows_path = output_dir / "pipeline_candidate_predictions.csv"
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
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector + Qwen classifier pipeline for multiple top-k values.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--output_dir", default="output/yolo_qwen_pipeline")
    parser.add_argument("--top_ks", default="1,3,5,10", help="Comma-separated top-k values to evaluate in one run.")
    parser.add_argument("--top_k", type=int, default=None, help="Deprecated alias for evaluating one top-k value.")
    parser.add_argument("--input_mode", choices=["bbox_prompt", "crop"], default="bbox_prompt")
    parser.add_argument("--crop_expand_ratio", type=float, default=0.2)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k is not None:
        args.top_ks = str(args.top_k)
    evaluate(args)


if __name__ == "__main__":
    main()
