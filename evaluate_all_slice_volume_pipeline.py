import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]


def require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit(
            "nibabel is required. Install it in the environment used for nnU-Net evaluation."
        ) from exc
    return nib


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_shape(shape_text: str) -> Tuple[int, int]:
    text = str(shape_text or "").lower().replace(" ", "")
    if "x" not in text:
        return 0, 0
    parts = text.split("x")
    if len(parts) != 2:
        return 0, 0
    h = parse_int(parts[0])
    w = parse_int(parts[1])
    return w, h


def load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_prediction_rows(path: Optional[Path]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            case_id = row.get("case_id", "")
            selected_slice = parse_int(row.get("selected_slice"), default=-1)
            if case_id and selected_slice >= 0:
                out[(case_id, selected_slice)] = row
    return out


def load_volume(path: Path) -> np.ndarray:
    nib = require_nibabel()
    return nib.load(str(path)).get_fdata()


def mask_binary(path: Path) -> np.ndarray:
    return (load_volume(path) > 0).astype(np.uint8)


def bbox2d(mask: np.ndarray) -> BBox:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area2d(a: BBox, b: BBox) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def iou2d(a: BBox, b: BBox) -> float:
    inter = intersection_area2d(a, b)
    denom = bbox_area(a) + bbox_area(b) - inter
    return inter / denom if denom > 0 else 0.0


def iom2d(a: BBox, b: BBox) -> float:
    inter = intersection_area2d(a, b)
    denom = min(bbox_area(a), bbox_area(b))
    return inter / denom if denom > 0 else 0.0


def gt_coverage_by_pred(pred_bbox: BBox, gt_bbox: BBox) -> float:
    inter = intersection_area2d(pred_bbox, gt_bbox)
    area = bbox_area(gt_bbox)
    return inter / area if area > 0 else 0.0


def pred_precision_to_gt(pred_bbox: BBox, gt_bbox: BBox) -> float:
    inter = intersection_area2d(pred_bbox, gt_bbox)
    area = bbox_area(pred_bbox)
    return inter / area if area > 0 else 0.0


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_hit(pred_bbox: BBox, gt_bbox: BBox) -> bool:
    cx, cy = bbox_center(pred_bbox)
    x1, y1, x2, y2 = gt_bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def relaxed_localization_hit(pred_bbox: BBox, gt_bbox: BBox) -> bool:
    return (
        iou2d(pred_bbox, gt_bbox) >= 0.3
        or iom2d(pred_bbox, gt_bbox) >= 0.7
        or (center_hit(pred_bbox, gt_bbox) and gt_coverage_by_pred(pred_bbox, gt_bbox) >= 0.3)
    )


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    try:
        from scipy import ndimage

        labels, num = ndimage.label(mask > 0)
        components = []
        for idx in range(1, num + 1):
            component = (labels == idx).astype(np.uint8)
            if int(component.sum()) > 0:
                components.append(component)
        components.sort(key=lambda item: int(item.sum()), reverse=True)
        return components
    except ImportError:
        # Small fallback to avoid adding scipy as a hard dependency.
        binary = mask.astype(bool)
        visited = np.zeros(binary.shape, dtype=bool)
        components: List[np.ndarray] = []
        h, w = binary.shape
        for y in range(h):
            for x in range(w):
                if not binary[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                coords = []
                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                comp = np.zeros(binary.shape, dtype=np.uint8)
                ys, xs = zip(*coords)
                comp[list(ys), list(xs)] = 1
                components.append(comp)
        components.sort(key=lambda item: int(item.sum()), reverse=True)
        return components


def component_touches_border(component: np.ndarray) -> bool:
    return bool(
        component[0, :].any()
        or component[-1, :].any()
        or component[:, 0].any()
        or component[:, -1].any()
    )


def component_center_distance_score(bbox: BBox, image_width: int, image_height: int) -> float:
    if bbox_area(bbox) <= 0:
        return 0.0
    cx, cy = bbox_center(bbox)
    dx = abs(cx - image_width / 2.0) / max(image_width / 2.0, 1.0)
    dy = abs(cy - image_height / 2.0) / max(image_height / 2.0, 1.0)
    return max(0.0, 1.0 - math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0))


def component_quality_score(component: np.ndarray, border_penalty: float) -> float:
    h, w = component.shape
    bbox = bbox2d(component)
    area = int(component.sum())
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    fill_ratio = area / max(float(bw * bh), 1.0)
    aspect_ratio = max(bw / bh, bh / bw)
    aspect_score = 1.0 / max(aspect_ratio, 1.0)
    center_score = component_center_distance_score(bbox, w, h)
    score = area * (0.55 + 0.25 * fill_ratio + 0.15 * aspect_score + 0.05 * center_score)
    if component_touches_border(component):
        score *= max(0.0, 1.0 - border_penalty)
    return float(score)


def extract_component_candidates(
    mask_2d: np.ndarray,
    candidate_topk: int,
    min_component_area: int,
    candidate_min_area_ratio: float,
    candidate_border_penalty: float,
) -> List[Dict[str, Any]]:
    components = [c for c in connected_components(mask_2d) if int(c.sum()) >= min_component_area]
    if not components:
        return []
    largest = max(int(c.sum()) for c in components)
    candidates = []
    for comp in components:
        area = int(comp.sum())
        area_ratio = area / max(float(largest), 1.0)
        if area_ratio < candidate_min_area_ratio:
            continue
        candidates.append(
            {
                "bbox": bbox2d(comp),
                "area": area,
                "area_ratio_to_largest": area_ratio,
                "score": component_quality_score(comp, candidate_border_penalty),
                "touches_border": component_touches_border(comp),
            }
        )
    candidates.sort(key=lambda item: (item["score"], item["area"]), reverse=True)
    return candidates[: max(candidate_topk, 1)]


def union_bboxes(bboxes: Iterable[BBox]) -> BBox:
    valid = [bbox for bbox in bboxes if bbox_area(bbox) > 0]
    if not valid:
        return 0, 0, 0, 0
    return (
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    )


def transform_bbox(bbox: BBox, src_width: int, src_height: int, coord_mode: str) -> Tuple[BBox, Tuple[int, int]]:
    x1, y1, x2, y2 = bbox
    points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    def apply(point: Tuple[int, int]) -> Tuple[int, int]:
        x, y = point
        width, height = src_width, src_height
        if coord_mode.startswith("swap_xy"):
            x, y = y, x
            width, height = height, width
        suffix = coord_mode.replace("swap_xy", "").strip("_") if coord_mode.startswith("swap_xy") else coord_mode
        if suffix in {"flip_x", "flip_xy"}:
            x = width - x
        if suffix in {"flip_y", "flip_xy"}:
            y = height - y
        return x, y

    transformed = [apply(point) for point in points]
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]
    out_width, out_height = (src_height, src_width) if coord_mode.startswith("swap_xy") else (src_width, src_height)
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), (out_width, out_height)


def scale_bbox(bbox: BBox, src_shape_xy: Tuple[int, int], dst_width: int, dst_height: int, coord_mode: str) -> BBox:
    src_w, src_h = src_shape_xy
    transformed, (src_w, src_h) = transform_bbox(bbox, src_w, src_h, coord_mode)
    x1, y1, x2, y2 = transformed
    sx = dst_width / max(float(src_w), 1.0)
    sy = dst_height / max(float(src_h), 1.0)
    return int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))


def make_non_empty_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def find_adjacent_slice(pred_mask: np.ndarray, z: int, radius: int, strategy: str) -> Tuple[Optional[int], int, int]:
    candidates: List[Tuple[int, int, int]] = []
    depth = pred_mask.shape[2]
    for offset in range(1, radius + 1):
        for nz in (z - offset, z + offset):
            if 0 <= nz < depth:
                area = int((pred_mask[:, :, nz] > 0).sum())
                if area > 0:
                    candidates.append((nz, offset, area))
    if not candidates:
        return None, 0, 0
    if strategy == "nearest_then_area":
        candidates.sort(key=lambda item: (item[1], -item[2], item[0]))
    elif strategy == "area_then_nearest":
        candidates.sort(key=lambda item: (-item[2], item[1], item[0]))
    else:
        raise ValueError(f"Unsupported adjacent strategy: {strategy}")
    return candidates[0]


def select_candidates_for_slice(pred_mask: np.ndarray, z: int, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target_area = int((pred_mask[:, :, z] > 0).sum())
    selected_z = z
    candidate_slice_distance = 0
    candidate_slice_area = target_area
    pred_mask_empty = target_area == 0
    source = "filtered_components"

    if pred_mask_empty and args.adjacent_slice_fallback:
        found_z, distance, area = find_adjacent_slice(
            pred_mask,
            z,
            args.adjacent_slice_fallback_radius,
            args.adjacent_slice_fallback_strategy,
        )
        if found_z is not None:
            selected_z = found_z
            candidate_slice_distance = distance
            candidate_slice_area = area
            source = "adjacent_slice_fallback"
        else:
            return [], {
                "selected_slice": z,
                "candidate_slice": "",
                "candidate_slice_distance": "",
                "candidate_slice_area": 0,
                "pred_mask_empty": True,
                "filtered_empty": False,
                "used_fallback_component": False,
                "final_candidate_source": "pred_mask_empty",
                "num_raw_components": 0,
                "target_only_candidate": False,
            }

    mask_2d = pred_mask[:, :, selected_z]
    raw_components = connected_components(mask_2d)
    candidates = extract_component_candidates(
        mask_2d,
        args.candidate_topk,
        args.min_component_area,
        args.candidate_min_area_ratio,
        args.candidate_border_penalty,
    )
    filtered_empty = bool(raw_components and not candidates)
    used_fallback_component = False
    if filtered_empty and args.fallback_largest_if_filtered_empty:
        comp = raw_components[0]
        area = int(comp.sum())
        candidates = [
            {
                "bbox": bbox2d(comp),
                "area": area,
                "area_ratio_to_largest": 1.0,
                "score": float(area),
                "touches_border": component_touches_border(comp),
            }
        ]
        source = "largest_component_fallback"
        used_fallback_component = True
    if not candidates:
        source = "filtered_empty" if filtered_empty else "no_raw_components"

    return candidates, {
        "selected_slice": selected_z,
        "candidate_slice": selected_z if candidates else "",
        "candidate_slice_distance": candidate_slice_distance if candidates else "",
        "candidate_slice_area": candidate_slice_area if candidates else 0,
        "pred_mask_empty": pred_mask_empty,
        "filtered_empty": filtered_empty,
        "used_fallback_component": used_fallback_component,
        "final_candidate_source": source,
        "num_raw_components": len(raw_components),
        "target_only_candidate": target_area > 0,
    }


def safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def label_from_row(row: Dict[str, Any]) -> str:
    label = str(row.get("label", "")).strip().lower()
    if label in {"infection", "tumor"}:
        return label
    case_id = str(row.get("case_id", "")).lower()
    if case_id.startswith("infection"):
        return "infection"
    if case_id.startswith("tumor"):
        return "tumor"
    return label


def gt_bbox_from_row(row: Dict[str, Any]) -> BBox:
    return (
        parse_int(row.get("bbox_x1")),
        parse_int(row.get("bbox_y1")),
        parse_int(row.get("bbox_x2")),
        parse_int(row.get("bbox_y2")),
    )


def compute_binary_metrics(prefix: str, rows: List[Dict[str, Any]], candidate_key: str) -> Dict[str, Any]:
    gt_pos = sum(parse_bool(r["gt_positive_slice"]) for r in rows)
    gt_neg = len(rows) - gt_pos
    pred_pos = sum(parse_bool(r[candidate_key]) for r in rows)
    tp = sum(parse_bool(r["gt_positive_slice"]) and parse_bool(r[candidate_key]) for r in rows)
    fp = sum((not parse_bool(r["gt_positive_slice"])) and parse_bool(r[candidate_key]) for r in rows)
    tn = sum((not parse_bool(r["gt_positive_slice"])) and (not parse_bool(r[candidate_key])) for r in rows)
    fn = sum(parse_bool(r["gt_positive_slice"]) and (not parse_bool(r[candidate_key])) for r in rows)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        f"{prefix}_pred_positive_slices": pred_pos,
        f"{prefix}_candidate_recall": recall,
        f"{prefix}_candidate_precision": precision,
        f"{prefix}_candidate_f1": safe_div(2 * precision * recall, precision + recall),
        f"{prefix}_empty_slice_specificity": safe_div(tn, tn + fp),
        f"{prefix}_empty_slice_fpr": safe_div(fp, fp + tn),
        f"{prefix}_tp": tp,
        f"{prefix}_fp": fp,
        f"{prefix}_tn": tn,
        f"{prefix}_fn": fn,
        f"{prefix}_gt_positive_slices": gt_pos,
        f"{prefix}_gt_negative_slices": gt_neg,
    }


def summarize_by_group(rows: List[Dict[str, Any]], group_fields: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
    summaries = []
    for key, items in sorted(grouped.items()):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["num_slices"] = len(items)
        summary.update(compute_binary_metrics("final", items, "final_has_candidate"))
        summaries.append(summary)
    return summaries


def add_nearest_gt_positive_distance(rows: List[Dict[str, Any]]) -> None:
    positive_slices_by_case: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        if parse_bool(row["gt_positive_slice"]):
            positive_slices_by_case[row["case_id"]].append(parse_int(row["slice_idx"]))
    for case_id in positive_slices_by_case:
        positive_slices_by_case[case_id].sort()

    for row in rows:
        z = parse_int(row["slice_idx"])
        positives = positive_slices_by_case.get(row["case_id"], [])
        if not positives:
            distance = ""
            distance_bin = "no_gt_positive_in_case"
        else:
            distance_int = min(abs(z - pz) for pz in positives)
            distance = str(distance_int)
            if distance_int == 0:
                distance_bin = "gt_positive"
            elif distance_int == 1:
                distance_bin = "dist_1"
            elif distance_int == 2:
                distance_bin = "dist_2"
            elif distance_int == 3:
                distance_bin = "dist_3"
            elif distance_int <= 5:
                distance_bin = "dist_4_5"
            else:
                distance_bin = "dist_gt5"
        row["nearest_gt_positive_slice_distance"] = distance
        row["nearest_gt_positive_slice_distance_bin"] = distance_bin


def summarize_negative_distance_bins(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    negative_rows = [row for row in rows if not parse_bool(row["gt_positive_slice"])]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in negative_rows:
        grouped[str(row.get("nearest_gt_positive_slice_distance_bin", ""))].append(row)

    order = {
        "dist_1": 1,
        "dist_2": 2,
        "dist_3": 3,
        "dist_4_5": 4,
        "dist_gt5": 5,
        "no_gt_positive_in_case": 6,
    }
    summaries = []
    for bin_name, items in sorted(grouped.items(), key=lambda kv: order.get(kv[0], 99)):
        n = len(items)
        target_fp = sum(parse_bool(row["target_only_has_candidate"]) for row in items)
        final_fp = sum(parse_bool(row["final_has_candidate"]) for row in items)
        summaries.append(
            {
                "distance_bin": bin_name,
                "num_gt_negative_slices": n,
                "target_only_false_positive": target_fp,
                "target_only_false_positive_rate": safe_div(target_fp, n),
                "with_fallback_false_positive": final_fp,
                "with_fallback_false_positive_rate": safe_div(final_fp, n),
                "interpretation": (
                    "with_fallback counts adjacent-slice rescue as a candidate; this is useful for lesion-positive "
                    "slice rescue but should not be interpreted as the standalone empty-slice detector."
                ),
            }
        )
    return summaries


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_rows = load_manifest_rows(Path(args.slice_manifest))
    if args.manifest_filter:
        before = len(manifest_rows)
        for item in args.manifest_filter:
            if "=" not in item:
                raise ValueError(f"--manifest_filter expects key=value, got: {item}")
            key, value = item.split("=", 1)
            manifest_rows = [row for row in manifest_rows if str(row.get(key, "")) == value]
        print(
            f"[INFO] Filtered manifest rows by {args.manifest_filter}: {before} -> {len(manifest_rows)} rows.",
            flush=True,
        )
    qwen_predictions = load_prediction_rows(Path(args.pipeline_predictions_csv) if args.pipeline_predictions_csv else None)
    if args.restrict_to_pipeline_cases:
        if not qwen_predictions:
            raise ValueError("--restrict_to_pipeline_cases requires --pipeline_predictions_csv")
        allowed_cases = {case_id for case_id, _ in qwen_predictions.keys()}
        before = len(manifest_rows)
        manifest_rows = [row for row in manifest_rows if row.get("case_id", "") in allowed_cases]
        print(
            f"[INFO] Restricted manifest rows by pipeline case IDs: {before} -> {len(manifest_rows)} rows; "
            f"{len(allowed_cases)} cases.",
            flush=True,
        )
    pred_dir = Path(args.pred_dir)

    pred_cache: Dict[str, np.ndarray] = {}
    output_rows: List[Dict[str, Any]] = []
    missing_pred_cases = Counter()
    behavior_counts = Counter()
    loc_values = {
        "iou": [],
        "iom": [],
        "gt_coverage": [],
        "pred_precision": [],
        "center_hit": [],
        "strict_hit": [],
        "relaxed_hit": [],
    }
    cls_counter = Counter()
    end_to_end_confusion = Counter()

    for row in manifest_rows:
        case_id = row["case_id"]
        z = parse_int(row["slice_idx"])
        gt_positive = parse_bool(row.get("has_lesion"))
        gt_label = label_from_row(row) if gt_positive else "no_lesion"
        image_w, image_h = parse_shape(row.get("image_shape", ""))
        if image_w <= 0 or image_h <= 0:
            image_w = image_h = args.default_image_size

        if case_id not in pred_cache:
            pred_path = pred_dir / f"{case_id}.nii.gz"
            if not pred_path.exists():
                missing_pred_cases[case_id] += 1
                continue
            pred_cache[case_id] = mask_binary(pred_path)
        pred_mask = pred_cache[case_id]
        if z < 0 or z >= pred_mask.shape[2]:
            continue

        target_only_candidate = int((pred_mask[:, :, z] > 0).sum()) > 0
        candidates, behavior = select_candidates_for_slice(pred_mask, z, args)
        final_has_candidate = bool(candidates)
        behavior_counts[behavior["final_candidate_source"]] += 1
        behavior_counts["pred_mask_empty"] += int(behavior["pred_mask_empty"])
        behavior_counts["filtered_empty"] += int(behavior["filtered_empty"])
        behavior_counts["used_fallback_component"] += int(behavior["used_fallback_component"])
        behavior_counts["adjacent_slice_fallback"] += int(behavior["final_candidate_source"] == "adjacent_slice_fallback")

        pred_bbox = (0, 0, 0, 0)
        det_iou = det_iom = gt_cov = pred_prec = 0.0
        c_hit = strict_hit = relaxed_hit = False
        if final_has_candidate:
            src_w, src_h = pred_mask.shape[1], pred_mask.shape[0]
            scaled = [
                make_non_empty_bbox(scale_bbox(c["bbox"], (src_w, src_h), image_w, image_h, args.coord_mode), image_w, image_h)
                for c in candidates
            ]
            pred_bbox = make_non_empty_bbox(union_bboxes(scaled), image_w, image_h)
            if gt_positive:
                gt_bbox = gt_bbox_from_row(row)
                det_iou = iou2d(pred_bbox, gt_bbox)
                det_iom = iom2d(pred_bbox, gt_bbox)
                gt_cov = gt_coverage_by_pred(pred_bbox, gt_bbox)
                pred_prec = pred_precision_to_gt(pred_bbox, gt_bbox)
                c_hit = center_hit(pred_bbox, gt_bbox)
                strict_hit = det_iou >= 0.5
                relaxed_hit = relaxed_localization_hit(pred_bbox, gt_bbox)
                loc_values["iou"].append(det_iou)
                loc_values["iom"].append(det_iom)
                loc_values["gt_coverage"].append(gt_cov)
                loc_values["pred_precision"].append(pred_prec)
                loc_values["center_hit"].append(float(c_hit))
                loc_values["strict_hit"].append(float(strict_hit))
                loc_values["relaxed_hit"].append(float(relaxed_hit))

        pred_row = qwen_predictions.get((case_id, z)) or qwen_predictions.get((case_id, behavior["selected_slice"]))
        pred_label = ""
        pred_text = ""
        if pred_row:
            pred_label = str(pred_row.get("pred_label", "")).strip().lower()
            pred_text = str(pred_row.get("pred_text", ""))

        if gt_positive:
            if not final_has_candidate:
                final_label = "no_candidate"
            elif pred_label in {"infection", "tumor"}:
                final_label = pred_label
            else:
                final_label = "disease_candidate"
        else:
            final_label = "no_lesion" if not final_has_candidate else "disease_candidate"

        end_to_end_confusion[f"{gt_label}->{final_label}"] += 1
        if gt_positive and final_has_candidate and pred_label in {"infection", "tumor"}:
            cls_counter["num_cls_eval_records"] += 1
            cls_counter[f"{gt_label}->{pred_label}"] += 1
            cls_counter["correct"] += int(gt_label == pred_label)
            cls_counter[f"{gt_label}_total"] += 1
            cls_counter[f"{gt_label}_correct"] += int(gt_label == pred_label)
        elif gt_positive and final_has_candidate and not pred_label:
            cls_counter["num_missing_qwen_prediction"] += 1

        output_rows.append(
            {
                "case_id": case_id,
                "patient_id": row.get("patient_id", ""),
                "seq": row.get("seq", ""),
                "slice_idx": z,
                "gt_positive_slice": int(gt_positive),
                "gt_label_3class": gt_label,
                "target_only_has_candidate": int(target_only_candidate),
                "final_has_candidate": int(final_has_candidate),
                "final_label_3class": final_label,
                "candidate_source": behavior["final_candidate_source"],
                "selected_slice": behavior["selected_slice"],
                "candidate_slice_distance": behavior["candidate_slice_distance"],
                "num_raw_components": behavior["num_raw_components"],
                "pred_mask_empty": int(behavior["pred_mask_empty"]),
                "filtered_empty": int(behavior["filtered_empty"]),
                "used_fallback_component": int(behavior["used_fallback_component"]),
                "gt_bbox_x1": row.get("bbox_x1", ""),
                "gt_bbox_y1": row.get("bbox_y1", ""),
                "gt_bbox_x2": row.get("bbox_x2", ""),
                "gt_bbox_y2": row.get("bbox_y2", ""),
                "pred_bbox_x1": pred_bbox[0],
                "pred_bbox_y1": pred_bbox[1],
                "pred_bbox_x2": pred_bbox[2],
                "pred_bbox_y2": pred_bbox[3],
                "det_iou": det_iou,
                "det_iom": det_iom,
                "gt_coverage_by_pred": gt_cov,
                "pred_precision_to_gt": pred_prec,
                "center_hit": int(c_hit),
                "strict_localization_hit": int(strict_hit),
                "relaxed_localization_hit": int(relaxed_hit),
                "qwen_pred_label": pred_label,
                "qwen_pred_text": pred_text,
                "image_path": row.get("image_path", ""),
            }
        )

    add_nearest_gt_positive_distance(output_rows)

    gt_pos = sum(parse_bool(r["gt_positive_slice"]) for r in output_rows)
    gt_neg = len(output_rows) - gt_pos
    final_candidate_metrics = compute_binary_metrics("final", output_rows, "final_has_candidate")
    target_candidate_metrics = compute_binary_metrics("target_only", output_rows, "target_only_has_candidate")
    fallback_rescued_positive_slices = max(0, final_candidate_metrics["final_tp"] - target_candidate_metrics["target_only_tp"])
    fallback_added_negative_candidates = max(0, final_candidate_metrics["final_fp"] - target_candidate_metrics["target_only_fp"])
    negative_distance_bins = summarize_negative_distance_bins(output_rows)

    cls_total = cls_counter["num_cls_eval_records"]
    infection_total = cls_counter["infection_total"]
    tumor_total = cls_counter["tumor_total"]
    infection_acc = safe_div(cls_counter["infection_correct"], infection_total)
    tumor_acc = safe_div(cls_counter["tumor_correct"], tumor_total)

    correct_e2e = sum(
        1
        for r in output_rows
        if (
            (r["gt_label_3class"] == "no_lesion" and r["final_label_3class"] == "no_lesion")
            or (r["gt_label_3class"] in {"infection", "tumor"} and r["final_label_3class"] == r["gt_label_3class"])
        )
    )

    metrics = {
        "dataset_summary": {
            "num_total_slices": len(output_rows),
            "num_gt_positive_slices": gt_pos,
            "num_gt_negative_slices": gt_neg,
            "num_cases_with_predictions": len(pred_cache),
            "num_missing_pred_cases": len(missing_pred_cases),
            "missing_pred_cases": dict(missing_pred_cases),
        },
        "target_only_candidate_metrics": target_candidate_metrics,
        "final_candidate_metrics_with_fallback": final_candidate_metrics,
        "paper_interpretation_metrics": {
            "target_only_empty_slice_specificity": target_candidate_metrics["target_only_empty_slice_specificity"],
            "target_only_empty_slice_fpr": target_candidate_metrics["target_only_empty_slice_fpr"],
            "target_only_candidate_recall_on_gt_positive_slices": target_candidate_metrics["target_only_candidate_recall"],
            "with_fallback_candidate_recall_on_gt_positive_slices": final_candidate_metrics["final_candidate_recall"],
            "fallback_rescued_positive_slices": fallback_rescued_positive_slices,
            "fallback_added_negative_candidates": fallback_added_negative_candidates,
            "target_only_role": (
                "Target-slice nnU-Net prediction is the preferred estimate for empty-slice screening, "
                "because it does not borrow lesion proposals from neighboring slices."
            ),
            "adjacent_fallback_role": (
                "Adjacent-slice fallback is a lesion-positive rescue mechanism for clinical/volume-aware reading. "
                "It should not be interpreted as a strict per-slice no-lesion detector."
            ),
        },
        "gt_negative_slice_distance_bins": negative_distance_bins,
        "detection_metrics_on_gt_positive_with_candidate": {
            "num_detection_eval_records": len(loc_values["iou"]),
            "mean_primary_bbox_iou": float(np.mean(loc_values["iou"])) if loc_values["iou"] else 0.0,
            "mean_primary_bbox_iom": float(np.mean(loc_values["iom"])) if loc_values["iom"] else 0.0,
            "mean_gt_coverage": float(np.mean(loc_values["gt_coverage"])) if loc_values["gt_coverage"] else 0.0,
            "mean_pred_precision": float(np.mean(loc_values["pred_precision"])) if loc_values["pred_precision"] else 0.0,
            "center_hit_rate": float(np.mean(loc_values["center_hit"])) if loc_values["center_hit"] else 0.0,
            "det_recall_iou_0.3": safe_div(sum(v >= 0.3 for v in loc_values["iou"]), gt_pos),
            "det_recall_iou_0.5": safe_div(sum(v >= 0.5 for v in loc_values["iou"]), gt_pos),
            "strict_localization_recall": safe_div(sum(loc_values["strict_hit"]), gt_pos),
            "relaxed_localization_recall": safe_div(sum(loc_values["relaxed_hit"]), gt_pos),
        },
        "classification_metrics_given_candidate": {
            "cls_acc_given_candidate": safe_div(cls_counter["correct"], cls_total),
            "infection_acc_given_candidate": infection_acc,
            "tumor_acc_given_candidate": tumor_acc,
            "balanced_acc_given_candidate": (infection_acc + tumor_acc) / 2 if cls_total else 0.0,
            "num_cls_eval_records": cls_total,
            "num_missing_qwen_prediction_for_positive_candidate": cls_counter["num_missing_qwen_prediction"],
            "confusion": {k: v for k, v in cls_counter.items() if "->" in k},
        },
        "end_to_end_3class_metrics": {
            "e2e_3class_or_candidate_acc": safe_div(correct_e2e, len(output_rows)),
            "labels": ["no_lesion", "infection", "tumor", "disease_candidate"],
            "confusion": dict(end_to_end_confusion),
            "note": (
                "GT-negative slices with any candidate are counted as incorrect disease_candidate. "
                "Qwen is not required for negative candidates because any infection/tumor output is a false positive."
            ),
        },
        "pipeline_behavior": {
            "rate_no_candidate": safe_div(final_candidate_metrics["final_fn"] + final_candidate_metrics["final_tn"], len(output_rows)),
            "rate_pred_mask_empty": safe_div(behavior_counts["pred_mask_empty"], len(output_rows)),
            "rate_filtered_empty": safe_div(behavior_counts["filtered_empty"], len(output_rows)),
            "rate_adjacent_slice_fallback": safe_div(behavior_counts["adjacent_slice_fallback"], len(output_rows)),
            "rate_fallback_single_component": safe_div(behavior_counts["used_fallback_component"], len(output_rows)),
            "counts": dict(behavior_counts),
        },
        "by_label_seq": summarize_by_group(output_rows, ["gt_label_3class", "seq"]),
        "by_seq": summarize_by_group(output_rows, ["seq"]),
        "config": {
            "coord_mode": args.coord_mode,
            "candidate_topk": args.candidate_topk,
            "candidate_min_area_ratio": args.candidate_min_area_ratio,
            "candidate_border_penalty": args.candidate_border_penalty,
            "min_component_area": args.min_component_area,
            "adjacent_slice_fallback": args.adjacent_slice_fallback,
            "adjacent_slice_fallback_radius": args.adjacent_slice_fallback_radius,
            "adjacent_slice_fallback_strategy": args.adjacent_slice_fallback_strategy,
            "fallback_largest_if_filtered_empty": args.fallback_largest_if_filtered_empty,
        },
        "files": {
            "slice_manifest": str(Path(args.slice_manifest)),
            "pred_dir": str(Path(args.pred_dir)),
            "pipeline_predictions_csv": args.pipeline_predictions_csv,
            "per_slice_csv": str(Path(args.output_dir) / "all_slice_volume_eval_records.csv"),
        },
        "selection": {
            "restrict_to_pipeline_cases": args.restrict_to_pipeline_cases,
            "manifest_filter": args.manifest_filter,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_csv = output_dir / "all_slice_volume_eval_records.csv"
    if output_rows:
        with records_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
    with (output_dir / "all_slice_volume_eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "all_slice_volume_eval_by_group.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "by_label_seq": metrics["by_label_seq"],
                "by_seq": metrics["by_seq"],
                "gt_negative_slice_distance_bins": negative_distance_bins,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / "all_slice_negative_distance_bins.json").open("w", encoding="utf-8") as f:
        json.dump(negative_distance_bins, f, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all-slice/all-volume candidate behavior for the nnU-Net + Qwen pipeline."
    )
    parser.add_argument("--slice_manifest", required=True, help="All-slice manifest CSV generated from exported volumes.")
    parser.add_argument(
        "--manifest_filter",
        nargs="*",
        default=[],
        help="Optional key=value filters applied to the manifest rows, for example adapt_split=adapt_test.",
    )
    parser.add_argument("--pred_dir", required=True, help="Directory containing nnU-Net predicted masks (*.nii.gz).")
    parser.add_argument(
        "--pipeline_predictions_csv",
        default="",
        help="Optional nnU-Net+Qwen predictions CSV for GT-positive slices; used for classification metrics.",
    )
    parser.add_argument(
        "--restrict_to_pipeline_cases",
        action="store_true",
        help="Evaluate only cases that appear in --pipeline_predictions_csv. Use this for adaptation held-out subsets.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--coord_mode", default="swap_xy", choices=["xy", "swap_xy", "flip_x", "flip_y", "flip_xy", "swap_xy_flip_x", "swap_xy_flip_y", "swap_xy_flip_xy"])
    parser.add_argument("--candidate_topk", type=int, default=3)
    parser.add_argument("--candidate_min_area_ratio", type=float, default=0.15)
    parser.add_argument("--candidate_border_penalty", type=float, default=0.15)
    parser.add_argument("--min_component_area", type=int, default=1)
    parser.add_argument("--adjacent_slice_fallback", action="store_true")
    parser.add_argument("--adjacent_slice_fallback_radius", type=int, default=3)
    parser.add_argument("--adjacent_slice_fallback_strategy", default="nearest_then_area", choices=["nearest_then_area", "area_then_nearest"])
    parser.add_argument("--fallback_largest_if_filtered_empty", action="store_true")
    parser.add_argument("--default_image_size", type=int, default=512)
    args = parser.parse_args()
    metrics = evaluate(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
