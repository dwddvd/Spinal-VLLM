import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from peft import PeftModel
from tqdm import tqdm

from qwen_stage2_classifier import LesionRecord, load_model_and_processor, load_records, make_non_empty_bbox, predict_label
from nnunet_qwen_pipeline import bbox2d, case_id_from_path, iou2d, load_volume, mask_binary, select_slice_and_bbox


def normalize_seq(seq: Optional[str]) -> str:
    if not seq:
        return ""
    seq = seq.upper()
    if "T1" in seq or seq.endswith("_1") or seq == "1":
        return "T1"
    if "T2" in seq or seq.endswith("_2") or seq == "2":
        return "T2"
    return seq


def manifest_seq(meta: Dict) -> str:
    if "seq" in meta:
        return normalize_seq(str(meta["seq"]))
    return normalize_seq(str(meta.get("seq_id", "")))


def extract_slice_index(path: str) -> Optional[int]:
    name = Path(path).stem
    patterns = [
        r"(?:^|[_-])z(\d+)(?:[_-]|$)",
        r"(?:^|[_-])slice[_-]?(\d+)(?:[_-]|$)",
        r"(?:^|[_-])s(\d+)(?:[_-]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.I)
        if match:
            return int(match.group(1))
    return None


def build_qwen_index(records: List[LesionRecord]) -> Dict[Tuple[str, str], List[Tuple[int, LesionRecord]]]:
    index: Dict[Tuple[str, str], List[Tuple[int, LesionRecord]]] = defaultdict(list)
    for record_idx, record in enumerate(records):
        patient_id = record.patient_id
        seq = normalize_seq(record.seq)
        index[(patient_id, seq)].append((record_idx, record))
    return index


def scale_bbox(
    bbox: Tuple[int, int, int, int],
    src_shape_xy: Tuple[int, int],
    dst_width: int,
    dst_height: int,
    coord_mode: str,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    src_w, src_h = src_shape_xy
    if coord_mode == "swap_xy":
        x1, y1, x2, y2 = y1, x1, y2, x2
        src_w, src_h = src_h, src_w
    sx = dst_width / max(float(src_w), 1.0)
    sy = dst_height / max(float(src_h), 1.0)
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def find_qwen_record(
    candidates: List[Tuple[int, LesionRecord]],
    selected_z: int,
    gt_bbox_scaled: Tuple[int, int, int, int],
) -> Tuple[Optional[int], Optional[LesionRecord], str]:
    if not candidates:
        return None, None, "missing_patient_seq"
    slice_matches = [(idx, record) for idx, record in candidates if extract_slice_index(record.image_path) == selected_z]
    if len(slice_matches) == 1:
        return slice_matches[0][0], slice_matches[0][1], "patient_seq_slice"
    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1], "patient_seq_single"
    best_idx, best_record = max(candidates, key=lambda item: iou2d(gt_bbox_scaled, item[1].bbox))
    return best_idx, best_record, "patient_seq_best_gt_bbox"


def load_qwen(base_model: str, adapter_path: str, load_in_4bit: bool):
    base, processor = load_model_and_processor(base_model, load_in_4bit, gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, processor


def evaluate(args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir)
    label_dir = Path(args.label_dir)
    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_items = json.load(f)
    manifest = {item["case_id"]: item for item in manifest_items}

    qwen_records = load_records(args.val_json)
    qwen_index = build_qwen_index(qwen_records)
    model, processor = load_qwen(args.base_model, args.adapter_path, args.load_in_4bit)

    rows = []
    total = 0
    matched = 0
    cls_correct = 0
    det_hits = {0.3: 0, 0.5: 0}
    joint_hits = {0.3: 0, 0.5: 0}
    confusion = Counter()
    match_methods = Counter()

    for pred_path in tqdm(sorted(pred_dir.glob("*.nii.gz")), desc="Eval remapped nnU-Net + Qwen", ncols=120):
        case_id = case_id_from_path(pred_path)
        meta = manifest.get(case_id)
        label_path = label_dir / f"{case_id}.nii.gz"
        if meta is None or not label_path.exists():
            continue
        total += 1
        pred_mask = mask_binary(pred_path)
        gt_mask = mask_binary(label_path)
        selected_z, pred_bbox, gt_bbox, _ = select_slice_and_bbox(pred_mask, gt_mask)
        src_w, src_h = pred_mask.shape[0], pred_mask.shape[1]
        key = (str(meta["patient_id"]), manifest_seq(meta))
        candidates = qwen_index.get(key, [])
        if candidates:
            first_record = candidates[0][1]
            gt_bbox_scaled_for_match = scale_bbox(gt_bbox, (src_w, src_h), first_record.width, first_record.height, args.coord_mode)
        else:
            gt_bbox_scaled_for_match = gt_bbox
        _, record, method = find_qwen_record(candidates, selected_z, gt_bbox_scaled_for_match)
        match_methods[method] += 1
        if record is None:
            rows.append({"case_id": case_id, "match_method": method})
            continue

        matched += 1
        pred_bbox_scaled = scale_bbox(pred_bbox, (src_w, src_h), record.width, record.height, args.coord_mode)
        gt_bbox_scaled = scale_bbox(gt_bbox, (src_w, src_h), record.width, record.height, args.coord_mode)
        bbox = make_non_empty_bbox(pred_bbox_scaled, record.width, record.height)
        det_iou = iou2d(bbox, record.bbox)
        pred_text, pred_label = predict_label(
            model,
            processor,
            record.image_path,
            record.seq,
            bbox=bbox,
            input_mode="bbox_prompt",
            max_new_tokens=args.max_new_tokens,
        )
        label_ok = pred_label == record.label
        cls_correct += int(label_ok)
        confusion[f"{record.label}->{pred_label or 'invalid'}"] += 1
        for thr in det_hits:
            det_ok = det_iou >= thr
            det_hits[thr] += int(det_ok)
            joint_hits[thr] += int(det_ok and label_ok)
        rows.append(
            {
                "case_id": case_id,
                "match_method": method,
                "patient_id": meta.get("patient_id", ""),
                "seq": meta.get("seq", ""),
                "selected_slice": selected_z,
                "qwen_image_path": record.image_path,
                "gt_label": record.label,
                "pred_label": pred_label or "",
                "pred_text": pred_text,
                "det_iou_qwen_bbox": det_iou,
                "pred_x1": bbox[0],
                "pred_y1": bbox[1],
                "pred_x2": bbox[2],
                "pred_y2": bbox[3],
                "qwen_gt_x1": record.bbox[0],
                "qwen_gt_y1": record.bbox[1],
                "qwen_gt_x2": record.bbox[2],
                "qwen_gt_y2": record.bbox[3],
                "scaled_nnunet_gt_iou_to_qwen_gt": iou2d(gt_bbox_scaled, record.bbox),
            }
        )

    metrics = {
        "total_nnunet_cases": total,
        "matched_qwen_records": matched,
        "match_rate": matched / max(total, 1),
        "coord_mode": args.coord_mode,
        "match_methods": dict(match_methods),
        "det_recall@iou0.3": det_hits[0.3] / max(matched, 1),
        "det_recall@iou0.5": det_hits[0.5] / max(matched, 1),
        "cls_acc_on_selected": cls_correct / max(matched, 1),
        "joint_acc@iou0.3": joint_hits[0.3] / max(matched, 1),
        "joint_acc@iou0.5": joint_hits[0.5] / max(matched, 1),
        "confusion": dict(confusion),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "nnunet_qwen_remap_predictions.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metrics_path = output_dir / "nnunet_qwen_remap_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Map nnU-Net bboxes back to original Qwen image domain and evaluate Qwen classification.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--output_dir", default="output/nnunet_qwen_remap_pipeline")
    parser.add_argument("--coord_mode", choices=["xy", "swap_xy"], default="xy")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
