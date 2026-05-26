import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

from qwen_stage2_classifier import load_model_and_processor, make_non_empty_bbox, predict_label


def require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit("nibabel is required. Install it with: pip install nibabel") from exc
    return nib


def load_volume(path: Path) -> np.ndarray:
    nib = require_nibabel()
    return nib.load(str(path)).get_fdata()


def mask_binary(path: Path) -> np.ndarray:
    return (load_volume(path) > 0).astype(np.uint8)


def case_id_from_path(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def load_manifest(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["case_id"]: item for item in items}


def bbox2d(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def iou2d(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def select_slice_and_bbox(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[int, Tuple[int, int, int, int], Tuple[int, int, int, int], float]:
    pred_area_per_slice = pred_mask.sum(axis=(0, 1))
    if int(pred_area_per_slice.sum()) > 0:
        z = int(np.argmax(pred_area_per_slice))
    elif int(gt_mask.sum()) > 0:
        z = int(np.argmax(gt_mask.sum(axis=(0, 1))))
    else:
        z = 0
    pred_bbox = bbox2d(pred_mask[:, :, z])
    gt_bbox = bbox2d(gt_mask[:, :, z])
    return z, pred_bbox, gt_bbox, iou2d(pred_bbox, gt_bbox)


def volume_slice_to_pil(volume: np.ndarray, z: int) -> Image.Image:
    image = volume[:, :, z].astype(np.float32)
    p1, p99 = np.percentile(image, [1, 99])
    if p99 > p1:
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1)
    else:
        image = image / max(float(image.max()), 1.0)
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def load_qwen(base_model: str, adapter_path: str, load_in_4bit: bool):
    base, processor = load_model_and_processor(base_model, load_in_4bit, gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, processor


def evaluate(args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir)
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    manifest = load_manifest(Path(args.manifest))
    model, processor = load_qwen(args.base_model, args.adapter_path, args.load_in_4bit)

    total = 0
    det_hits = {0.3: 0, 0.5: 0}
    joint_hits = {0.3: 0, 0.5: 0}
    cls_correct = 0
    confusion = Counter()
    rows = []

    pred_paths = sorted(pred_dir.glob("*.nii.gz"))
    for pred_path in tqdm(pred_paths, desc="Eval nnU-Net + Qwen", ncols=120):
        case_id = case_id_from_path(pred_path)
        meta = manifest.get(case_id, {})
        gt_label = meta.get("label", "")
        seq = meta.get("seq")
        image_path = image_dir / f"{case_id}_0000.nii.gz"
        label_path = label_dir / f"{case_id}.nii.gz"
        if not image_path.exists() or not label_path.exists() or gt_label not in {"infection", "tumor"}:
            continue

        pred_mask = mask_binary(pred_path)
        gt_mask = mask_binary(label_path)
        image_volume = load_volume(image_path)
        if pred_mask.shape != gt_mask.shape or pred_mask.shape != image_volume.shape:
            raise ValueError(f"Shape mismatch for {case_id}: pred={pred_mask.shape}, gt={gt_mask.shape}, image={image_volume.shape}")

        z, pred_bbox, gt_bbox, iou = select_slice_and_bbox(pred_mask, gt_mask)
        image = volume_slice_to_pil(image_volume, z)
        width, height = image.size
        bbox = make_non_empty_bbox(pred_bbox, width, height)
        pred_text, pred_label = predict_label(
            model,
            processor,
            image,
            seq,
            bbox=bbox,
            input_mode="bbox_prompt",
            max_new_tokens=args.max_new_tokens,
        )

        total += 1
        label_ok = pred_label == gt_label
        cls_correct += int(label_ok)
        confusion[f"{gt_label}->{pred_label or 'invalid'}"] += 1
        for thr in det_hits:
            det_ok = iou >= thr
            det_hits[thr] += int(det_ok)
            joint_hits[thr] += int(det_ok and label_ok)
        rows.append(
            {
                "case_id": case_id,
                "label": gt_label,
                "seq": seq or "",
                "slice_index": z,
                "pred_label": pred_label or "",
                "pred_text": pred_text,
                "bbox_iou": iou,
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "gt_x1": gt_bbox[0],
                "gt_y1": gt_bbox[1],
                "gt_x2": gt_bbox[2],
                "gt_y2": gt_bbox[3],
            }
        )

    metrics = {
        "total_cases": total,
        "det_recall@iou0.3": det_hits[0.3] / max(total, 1),
        "det_recall@iou0.5": det_hits[0.5] / max(total, 1),
        "cls_acc_on_selected": cls_correct / max(total, 1),
        "joint_acc@iou0.3": joint_hits[0.3] / max(total, 1),
        "joint_acc@iou0.5": joint_hits[0.5] / max(total, 1),
        "confusion": dict(confusion),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "nnunet_qwen_predictions.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metrics_path = output_dir / "nnunet_qwen_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net stage1 selected bbox with Qwen stage2 classifier.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", default="output/nnunet_qwen_pipeline")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
