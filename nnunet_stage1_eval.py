import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm


def require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit("nibabel is required. Install it with: pip install nibabel") from exc
    return nib


def load_mask(path: Path) -> np.ndarray:
    nib = require_nibabel()
    data = nib.load(str(path)).get_fdata()
    return (data > 0).astype(np.uint8)


def bbox2d(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def bbox3d(mask: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    xs, ys, zs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(zs.min()), int(xs.max() + 1), int(ys.max() + 1), int(zs.max() + 1)


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


def iou3d(a: Tuple[int, int, int, int, int, int], b: Tuple[int, int, int, int, int, int]) -> float:
    ax1, ay1, az1, ax2, ay2, az2 = a
    bx1, by1, bz1, bx2, by2, bz2 = b
    ix1, iy1, iz1 = max(ax1, bx1), max(ay1, by1), max(az1, bz1)
    ix2, iy2, iz2 = min(ax2, bx2), min(ay2, by2), min(az2, bz2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1) * max(0, iz2 - iz1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1) * max(0, az2 - az1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1) * max(0, bz2 - bz1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int((pred & gt).sum())
    den = int(pred.sum() + gt.sum())
    return 0.0 if den == 0 else 2.0 * inter / den


def selected_slice_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, float, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    if int(pred.sum()) > 0:
        z = int(np.argmax(pred.sum(axis=(0, 1))))
    elif int(gt.sum()) > 0:
        z = int(np.argmax(gt.sum(axis=(0, 1))))
    else:
        z = 0
    pred_box = bbox2d(pred[:, :, z])
    gt_box = bbox2d(gt[:, :, z])
    return z, iou2d(pred_box, gt_box), pred_box, gt_box


def case_id_from_prediction(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def evaluate(args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    hits_2d = {0.3: 0, 0.5: 0}
    hits_3d = {0.3: 0, 0.5: 0}
    dices = []
    missing = []

    pred_paths = sorted(pred_dir.glob("*.nii.gz"))
    for pred_path in tqdm(pred_paths, desc="Eval nnU-Net masks", ncols=120):
        case_id = case_id_from_prediction(pred_path)
        label_path = label_dir / f"{case_id}.nii.gz"
        if not label_path.exists():
            missing.append(case_id)
            continue
        pred = load_mask(pred_path)
        gt = load_mask(label_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {case_id}: pred={pred.shape}, gt={gt.shape}")
        dsc = dice(pred, gt)
        pred_bbox3d = bbox3d(pred)
        gt_bbox3d = bbox3d(gt)
        bbox_iou3d = iou3d(pred_bbox3d, gt_bbox3d)
        z, bbox_iou2d, pred_box2d, gt_box2d = selected_slice_iou(pred, gt)
        for thr in hits_2d:
            hits_2d[thr] += int(bbox_iou2d >= thr)
            hits_3d[thr] += int(bbox_iou3d >= thr)
        dices.append(dsc)
        rows.append(
            {
                "case_id": case_id,
                "dice": dsc,
                "selected_slice": z,
                "selected_slice_bbox_iou": bbox_iou2d,
                "volume_bbox_iou": bbox_iou3d,
                "pred_x1": pred_box2d[0],
                "pred_y1": pred_box2d[1],
                "pred_x2": pred_box2d[2],
                "pred_y2": pred_box2d[3],
                "gt_x1": gt_box2d[0],
                "gt_y1": gt_box2d[1],
                "gt_x2": gt_box2d[2],
                "gt_y2": gt_box2d[3],
            }
        )

    total = len(rows)
    metrics = {
        "pred_dir": str(pred_dir),
        "label_dir": str(label_dir),
        "total_predictions": len(pred_paths),
        "evaluated_cases": total,
        "missing_labels": missing,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "selected_slice_bbox_recall@iou0.3": hits_2d[0.3] / max(total, 1),
        "selected_slice_bbox_recall@iou0.5": hits_2d[0.5] / max(total, 1),
        "volume_bbox_recall@iou0.3": hits_3d[0.3] / max(total, 1),
        "volume_bbox_recall@iou0.5": hits_3d[0.5] / max(total, 1),
    }

    rows_path = output_dir / "nnunet_stage1_eval_cases.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = output_dir / "nnunet_stage1_eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net lesion masks as stage-1 bbox detections.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--output_dir", default="output/nnunet_stage1_eval")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
