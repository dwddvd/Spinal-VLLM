import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from peft import PeftModel
from PIL import Image, ImageDraw
from tqdm import tqdm

from qwen_stage2_classifier import (
    LesionRecord,
    build_messages,
    extract_image_path,
    extract_prompt_text,
    infer_sequence,
    generate_text,
    load_model_and_processor,
    load_records,
    make_non_empty_bbox,
    normalize_label,
    predict_label,
)

BBOX_TOKEN_PATTERN = re.compile(
    r"\[\s*(?:(?:\d+|x1)\s*,\s*(?:\d+|y1)\s*,\s*(?:\d+|x2)\s*,\s*(?:\d+|y2)|MASK)\s*\]",
    flags=re.I,
)
NUMERIC_BBOX_PATTERN = re.compile(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]")

PYCHARM_DEFAULTS = {
    "base_model": "models/Qwen3.5-4B",
    "adapter_path": "outputs/qwen_stage2_adapter",
    "pred_dir": "outputs/nnunet_predictions",
    "label_dir": "data/nnunet/labelsTs",
    "manifest": "data/nnunet/manifest.json",
    "qwen_json": ["data/temporal/records.json"],
    "hidden_qwen_json": ["data/temporal/records_hidden_bbox.json"],
    "output_dir": "outputs/nnunet_qwen_pipeline",
    "eval_unit": "qwen_records",
    "image_resize": 280,
    "load_in_4bit": True,
    "max_new_tokens": 128,
    "shuffle_eval": True,
    "seed": 42,
    "debug_visualize": True,
    "debug_show_plt": True,
    "debug_max_samples": 1,
    "coord_mode": "swap_xy",
    "bbox_strategy": "largest_component",
    "candidate_topk": 3,
    "candidate_min_area_ratio": 0.15,
    "candidate_border_penalty": 0.15
}


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


def intersection_area2d(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def iom2d(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = intersection_area2d(a, b)
    denom = min(bbox_area(a), bbox_area(b))
    return 0.0 if denom <= 0 else inter / denom


def gt_coverage_by_pred(pred_bbox: Tuple[int, int, int, int], gt_bbox: Tuple[int, int, int, int]) -> float:
    inter = intersection_area2d(pred_bbox, gt_bbox)
    gt_area = bbox_area(gt_bbox)
    return 0.0 if gt_area <= 0 else inter / gt_area


def pred_precision_to_gt(pred_bbox: Tuple[int, int, int, int], gt_bbox: Tuple[int, int, int, int]) -> float:
    inter = intersection_area2d(pred_bbox, gt_bbox)
    pred_area = bbox_area(pred_bbox)
    return 0.0 if pred_area <= 0 else inter / pred_area


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_hit(pred_bbox: Tuple[int, int, int, int], gt_bbox: Tuple[int, int, int, int]) -> bool:
    cx, cy = bbox_center(pred_bbox)
    x1, y1, x2, y2 = gt_bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def strict_localization_hit(pred_bbox: Tuple[int, int, int, int], gt_bbox: Tuple[int, int, int, int]) -> bool:
    return iou2d(pred_bbox, gt_bbox) >= 0.5


def relaxed_localization_hit(pred_bbox: Tuple[int, int, int, int], gt_bbox: Tuple[int, int, int, int]) -> bool:
    iou = iou2d(pred_bbox, gt_bbox)
    iom = iom2d(pred_bbox, gt_bbox)
    coverage = gt_coverage_by_pred(pred_bbox, gt_bbox)
    hit_center = center_hit(pred_bbox, gt_bbox)
    return (iou >= 0.3) or (iom >= 0.5) or (coverage >= 0.7) or (hit_center and coverage >= 0.5)


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    try:
        from scipy import ndimage

        labeled, count = ndimage.label(mask > 0)
        components = []
        for label_idx in range(1, count + 1):
            component = labeled == label_idx
            if component.any():
                components.append(component)
        return components
    except ImportError:
        pass

    mask = (mask > 0).astype(np.uint8)
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] > 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            component = np.zeros(mask.shape, dtype=bool)
            ys, xs = zip(*coords)
            component[np.array(ys), np.array(xs)] = True
            components.append(component)
    return components


def expand_bbox(
        bbox: Tuple[int, int, int, int],
        image_width: int,
        image_height: int,
        ratio: float,
) -> Tuple[int, int, int, int]:
    if ratio <= 0 or bbox_area(bbox) <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad_x = int(round(width * ratio))
    pad_y = int(round(height * ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )


def union_bboxes(bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    valid = [bbox for bbox in bboxes if bbox_area(bbox) > 0]
    if not valid:
        return 0, 0, 0, 0
    return (
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    )


def component_touches_border(component: np.ndarray) -> bool:
    return bool(component[0, :].any() or component[-1, :].any() or component[:, 0].any() or component[:, -1].any())


def component_center_distance_score(bbox: Tuple[int, int, int, int], image_width: int, image_height: int) -> float:
    if bbox_area(bbox) <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    image_cx = image_width / 2.0
    image_cy = image_height / 2.0
    dx = (cx - image_cx) / max(image_width / 2.0, 1.0)
    dy = (cy - image_cy) / max(image_height / 2.0, 1.0)
    distance = float((dx * dx + dy * dy) ** 0.5)
    return max(0.0, 1.0 - min(distance, 1.0))


def component_quality_score(
        component: np.ndarray,
        largest_area: int,
        image_width: int,
        image_height: int,
        border_penalty: float,
) -> float:
    bbox = bbox2d(component)
    area = int(component.sum())
    bbox_w = max(1, bbox[2] - bbox[0])
    bbox_h = max(1, bbox[3] - bbox[1])
    area_ratio = area / max(float(largest_area), 1.0)
    fill_ratio = area / max(float(bbox_w * bbox_h), 1.0)
    aspect_ratio = max(bbox_w / bbox_h, bbox_h / bbox_w)
    aspect_penalty = min(max(aspect_ratio - 3.0, 0.0) / 5.0, 1.0)
    center_score = component_center_distance_score(bbox, image_width, image_height)
    border_score = border_penalty if component_touches_border(component) else 0.0
    score = (1.35 * area_ratio) + (0.45 * fill_ratio) + (0.35 * center_score) - (0.35 * aspect_penalty) - border_score
    return float(score)


def extract_component_candidates(
        mask_2d: np.ndarray,
        candidate_topk: int,
        min_component_area: int,
        min_area_ratio: float,
        border_penalty: float,
) -> List[Dict[str, Any]]:
    components = connected_components(mask_2d)
    if not components:
        return []
    components.sort(key=lambda component: int(component.sum()), reverse=True)
    largest_area = max(int(component.sum()) for component in components)
    image_height, image_width = mask_2d.shape
    candidates: List[Dict[str, Any]] = []
    for component in components:
        area = int(component.sum())
        if area < min_component_area:
            continue
        area_ratio = area / max(float(largest_area), 1.0)
        if area_ratio < min_area_ratio:
            continue
        bbox = bbox2d(component)
        score = component_quality_score(component, largest_area, image_width, image_height, border_penalty)
        candidates.append(
            {
                "mask": component,
                "bbox": bbox,
                "area": area,
                "area_ratio_to_largest": area_ratio,
                "touches_border": component_touches_border(component),
                "score": score,
            }
        )
    candidates.sort(key=lambda item: (item["score"], item["area"]), reverse=True)
    return candidates[: max(candidate_topk, 1)]


def slice_component_score(mask_2d: np.ndarray) -> int:
    components = connected_components(mask_2d)
    if not components:
        return 0
    return max(int(component.sum()) for component in components)


def select_slice_index(pred_mask: np.ndarray, gt_mask: np.ndarray, slice_strategy: str) -> int:
    pred_area_per_slice = pred_mask.sum(axis=(0, 1))
    gt_area_per_slice = gt_mask.sum(axis=(0, 1))
    if slice_strategy == "pred_max_area":
        if int(pred_area_per_slice.sum()) > 0:
            return int(np.argmax(pred_area_per_slice))
    elif slice_strategy == "pred_largest_component":
        scores = [slice_component_score(pred_mask[:, :, z]) for z in range(pred_mask.shape[2])]
        if max(scores, default=0) > 0:
            return int(np.argmax(scores))
    elif slice_strategy == "gt_max_area_oracle":
        if int(gt_area_per_slice.sum()) > 0:
            return int(np.argmax(gt_area_per_slice))
    elif slice_strategy == "best_iou_oracle":
        best_z = 0
        best_iou = -1.0
        for z in range(pred_mask.shape[2]):
            score = iou2d(bbox2d(pred_mask[:, :, z]), bbox2d(gt_mask[:, :, z]))
            if score > best_iou:
                best_z = z
                best_iou = score
        return int(best_z)

    if int(gt_area_per_slice.sum()) > 0:
        return int(np.argmax(gt_area_per_slice))
    return 0


def bbox_from_mask_slice(
        mask_2d: np.ndarray,
        bbox_strategy: str,
        component_topk: int,
        min_component_area: int,
) -> Tuple[int, int, int, int]:
    if bbox_strategy == "mask_bbox":
        return bbox2d(mask_2d)

    components = connected_components(mask_2d)
    components = [component for component in components if int(component.sum()) >= min_component_area]
    if not components:
        return 0, 0, 0, 0
    components.sort(key=lambda component: int(component.sum()), reverse=True)
    if bbox_strategy == "largest_component":
        return bbox2d(components[0])
    if bbox_strategy == "topk_components_union":
        selected = components[: max(component_topk, 1)]
        return union_bboxes([bbox2d(component) for component in selected])
    raise ValueError(f"Unsupported bbox_strategy: {bbox_strategy}")


def select_slice_and_bbox(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        slice_strategy: str,
        bbox_strategy: str,
        bbox_expand_ratio: float,
        component_topk: int,
        min_component_area: int,
) -> Tuple[int, Tuple[int, int, int, int], Tuple[int, int, int, int], float]:
    z = select_slice_index(pred_mask, gt_mask, slice_strategy)
    pred_bbox = bbox_from_mask_slice(pred_mask[:, :, z], bbox_strategy, component_topk, min_component_area)
    pred_bbox = expand_bbox(pred_bbox, pred_mask.shape[1], pred_mask.shape[0], bbox_expand_ratio)
    gt_bbox = bbox2d(gt_mask[:, :, z])
    return z, pred_bbox, gt_bbox, iou2d(pred_bbox, gt_bbox)


def find_adjacent_candidate_slice(
        pred_mask: np.ndarray,
        selected_z: int,
        radius: int,
        strategy: str,
) -> Tuple[Optional[int], int, int]:
    if radius <= 0:
        return None, 0, 0
    num_slices = pred_mask.shape[2]
    candidates: List[Tuple[int, int, int]] = []
    for offset in range(1, radius + 1):
        for z in (selected_z - offset, selected_z + offset):
            if not (0 <= z < num_slices):
                continue
            area = int(pred_mask[:, :, z].sum())
            if area > 0:
                candidates.append((z, offset, area))
    if not candidates:
        return None, 0, 0
    if strategy == "nearest_then_area":
        candidates.sort(key=lambda item: (item[1], -item[2], item[0]))
    elif strategy == "max_area_within_radius":
        candidates.sort(key=lambda item: (-item[2], item[1], item[0]))
    else:
        raise ValueError(f"Unsupported adjacent fallback strategy: {strategy}")
    z, distance, area = candidates[0]
    return z, distance, area


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
        r"(?:^|[_-])layer[_-]?(\d+)(?:[_-]|$)",
        r"(?:^|[_-])s(\d+)(?:[_-]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.I)
        if match:
            return int(match.group(1))
    return None


def extract_patient_id(path_or_text: str) -> Optional[str]:
    matches = re.findall(r"\d{6,}", str(path_or_text))
    return matches[-1] if matches else None


def seq_from_path(path: str) -> str:
    text = str(path).upper()
    name = Path(path).stem.upper()
    match = re.match(r"^\d+_([12])(?:[_-]|$)", name)
    if match:
        return "T1" if match.group(1) == "1" else "T2"
    if "T1" in text:
        return "T1"
    if "T2" in text:
        return "T2"
    return ""


def record_patient_id(record: LesionRecord) -> str:
    return str(record.patient_id or "") or extract_patient_id(record.image_path) or extract_patient_id(record.sample_id) or ""


def record_seq(record: LesionRecord) -> str:
    return normalize_seq(record.seq) or seq_from_path(record.image_path)


def record_slice_index(record: LesionRecord) -> Optional[int]:
    if getattr(record, "slice_idx", None) is not None:
        return int(record.slice_idx)
    return extract_slice_index(record.image_path)


def build_qwen_index(records: List[LesionRecord]) -> Dict[Tuple[str, str], List[Tuple[int, LesionRecord]]]:
    index: Dict[Tuple[str, str], List[Tuple[int, LesionRecord]]] = defaultdict(list)
    for record_idx, record in enumerate(records):
        patient_id = record_patient_id(record)
        seq = record_seq(record)
        index[(patient_id, seq)].append((record_idx, record))
    return index


def build_case_index(manifest: Dict[str, Dict]) -> Dict[Tuple[str, str], List[str]]:
    index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for case_id, meta in manifest.items():
        patient_id = str(meta.get("patient_id", ""))
        seq = manifest_seq(meta)
        index[(patient_id, seq)].append(case_id)
    return index


def transform_bbox(
        bbox: Tuple[int, int, int, int],
        src_width: int,
        src_height: int,
        coord_mode: str,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
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
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    out_width, out_height = (src_height, src_width) if coord_mode.startswith("swap_xy") else (src_width, src_height)
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), (out_width, out_height)


def transform_mask(mask_2d: np.ndarray, coord_mode: str) -> np.ndarray:
    transformed = mask_2d
    if coord_mode.startswith("swap_xy"):
        transformed = transformed.T
    suffix = coord_mode.replace("swap_xy", "").strip("_") if coord_mode.startswith("swap_xy") else coord_mode
    if suffix in {"flip_x", "flip_xy"}:
        transformed = np.fliplr(transformed)
    if suffix in {"flip_y", "flip_xy"}:
        transformed = np.flipud(transformed)
    return transformed


def scale_bbox(
        bbox: Tuple[int, int, int, int],
        src_shape_xy: Tuple[int, int],
        dst_width: int,
        dst_height: int,
        coord_mode: str,
) -> Tuple[int, int, int, int]:
    src_w, src_h = src_shape_xy
    (x1, y1, x2, y2), (src_w, src_h) = transform_bbox(bbox, src_w, src_h, coord_mode)
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
    slice_matches = [(idx, record) for idx, record in candidates if record_slice_index(record) == selected_z]
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


def bbox_to_text(bbox: Tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    return f"[{x1},{y1},{x2},{y2}]"


def replace_first_bbox(text: str, replacement: str) -> str:
    return BBOX_TOKEN_PATTERN.sub(replacement, text, count=2)


def replace_nth_bbox(text, replacement, n=2):
    count = 0

    def repl(match):
        nonlocal count
        count += 1
        if count == n:
            return replacement
        return match.group(0)

    return BBOX_TOKEN_PATTERN.sub(repl, text)


def build_region_prompt(prompt: str, bbox_text: str, region_index: int, region_count: int) -> str:
    region_prefix = f"候选病灶区域{region_index}/{region_count}"
    if prompt:
        replaced = replace_nth_bbox(prompt, bbox_text, 2)
        return (
            f"{replaced}\n"
            f"补充说明：当前仅分析{region_prefix}，坐标为{bbox_text}。"
            "请忽略其他区域，仅根据这个框内最主要的病灶表现判断属于感染还是肿瘤；"
            "如果该框不像有效病灶，请输出'无明确病灶'。"
        )
    return (
        "现在你是一个骨科专家，这是一幅脊椎的磁共振图像。"
        f"当前仅分析{region_prefix}，其坐标为{bbox_text}。"
        "请只关注这个框内区域，并判断该区域最可能属于感染还是肿瘤。"
    )


def fuse_component_predictions(component_outputs: List[Dict[str, Any]]) -> Tuple[
    str, Dict[str, float], List[Dict[str, Any]]]:
    vote_scores = {"infection": 0.0, "tumor": 0.0}
    valid_outputs: List[Dict[str, Any]] = []
    for output in component_outputs:
        pred_label = output.get("pred_label")
        if pred_label not in vote_scores:
            continue
        weight = float(output.get("weight", 0.0))
        vote_scores[pred_label] += weight
        valid_outputs.append(output)
    if not valid_outputs:
        return "", vote_scores, valid_outputs
    if vote_scores["infection"] == vote_scores["tumor"]:
        best = max(valid_outputs, key=lambda item: (float(item.get("weight", 0.0)), int(item.get("area", 0))))
        return str(best["pred_label"]), vote_scores, valid_outputs
    final_label = max(vote_scores.items(), key=lambda item: item[1])[0]
    return final_label, vote_scores, valid_outputs


def make_prompt_record(record: LesionRecord, bbox_text: str, sample_id: str) -> Dict:
    prompt = record.prompt_text or ""
    if prompt:
        prompt = replace_first_bbox(prompt, bbox_text)
    else:
        prompt = (
            f"\u73b0\u5728\u4f60\u662f\u4e00\u4e2a\u9aa8\u79d1\u4e13\u5bb6\uff0c"
            f"\u8fd9\u662f\u4e00\u5e45\u810a\u690e\u7684\u78c1\u5171\u632f\u56fe\u50cf\uff0c"
            f"\u75c5\u7076\u4f4d\u7f6e\u4e3a{bbox_text}\u3002"
            f"\u8bf7\u4f60\u5e2e\u6211\u5224\u65ad\u8fd9\u4e2a\u75c5\u7076\u5c5e\u4e8e\u611f\u67d3\u8fd8\u662f\u80bf\u7624\u3002"
        )
    value = f"{prompt}<|vision_start|>{record.image_path}<|vision_end|>"
    return {
        "id": sample_id,
        "conversations": [
            {"from": "user", "value": value},
            {"from": "assistant", "value": record.answer_text or ""},
        ],
    }


def make_replaced_prompt(
        record: LesionRecord,
        bbox: Tuple[int, int, int, int],
        region_index: int = 1,
        region_count: int = 1,
) -> str:
    return build_region_prompt(record.prompt_text or "", bbox_to_text(bbox), region_index, region_count)


def assert_hidden_prompt_is_safe(record: LesionRecord) -> None:
    prompt = record.prompt_text or ""
    if NUMERIC_BBOX_PATTERN.search(prompt):
        raise ValueError(
            f"hidden_qwen_json sample {record.sample_id} still contains a numeric bbox in the prompt. "
            "Please regenerate the hidden JSON before running the pipeline."
        )


def load_record_list(json_paths: List[str]) -> List[LesionRecord]:
    records = []
    for json_path in json_paths:
        records.extend(load_records(json_path))
    return records


def load_prompt_record_list(json_paths: List[str]) -> List[LesionRecord]:
    records: List[LesionRecord] = []
    skipped = 0
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{json_path} must contain a JSON list.")
        for index, item in enumerate(data):
            convs = item.get("conversations", [])
            if len(convs) < 2:
                skipped += 1
                continue
            user_text = convs[0].get("value", "")
            assistant_text = convs[1].get("value", "")
            image_path = extract_image_path(user_text)
            label = normalize_label(assistant_text)
            if image_path is None or label is None or not Path(image_path).exists():
                skipped += 1
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            sample_id = str(item.get("id", "")) or f"sample_{index}"
            patient_id = Path(image_path).name.split("_")[0]
            records.append(
                LesionRecord(
                    sample_id=sample_id,
                    image_path=image_path,
                    bbox=(0, 0, 1, 1),
                    label=label,
                    width=width,
                    height=height,
                    patient_id=patient_id,
                    seq=infer_sequence(user_text),
                    prompt_text=extract_prompt_text(user_text),
                    answer_text=assistant_text.strip(),
                )
            )
        print(f"[INFO] Loaded {len(records)} prompt records so far from {json_path}; skipped {skipped}.", flush=True)
    return records


def pair_gt_and_hidden_records(
        gt_records: List[LesionRecord],
        hidden_records: Optional[List[LesionRecord]],
) -> List[Tuple[LesionRecord, LesionRecord]]:
    if hidden_records is None:
        return [(record, record) for record in gt_records]
    hidden_by_id = {record.sample_id: record for record in hidden_records}
    gt_ids = {record.sample_id for record in gt_records}
    missing = [record.sample_id for record in gt_records if record.sample_id not in hidden_by_id]
    extra = [record.sample_id for record in hidden_records if record.sample_id not in gt_ids]
    if missing or extra:
        preview_missing = missing[:5]
        preview_extra = extra[:5]
        raise ValueError(
            "hidden_qwen_json and qwen_json must contain matching sample ids. "
            f"missing_in_hidden={len(missing)} {preview_missing}; extra_in_hidden={len(extra)} {preview_extra}"
        )
    return [(record, hidden_by_id[record.sample_id]) for record in gt_records]


def choose_case_id(case_ids: List[str], record: LesionRecord) -> Tuple[Optional[str], str]:
    if not case_ids:
        return None, "missing_patient_seq"
    record_case_id = getattr(record, "case_id", None)
    if record_case_id and record_case_id in case_ids:
        return record_case_id, "patient_seq_explicit_case_id"
    if len(case_ids) == 1:
        return case_ids[0], "patient_seq_single_case"
    patient_id = record_patient_id(record)
    seq = record_seq(record)
    seq_id = "1" if seq == "T1" else "2" if seq == "T2" else seq
    for case_id in case_ids:
        if patient_id in case_id and (f"_{seq_id}" in case_id or seq in case_id.upper()):
            return case_id, "patient_seq_case_name"
    return case_ids[0], "patient_seq_first_case"


def load_mask_pair(
        cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
        case_id: str,
        pred_dir: Path,
        label_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    if case_id not in cache:
        pred_path = pred_dir / f"{case_id}.nii.gz"
        label_path = label_dir / f"{case_id}.nii.gz"
        cache[case_id] = (mask_binary(pred_path), mask_binary(label_path))
    return cache[case_id]


def mask_to_resized_bool(mask_2d: np.ndarray, width: int, height: int) -> np.ndarray:
    mask_image = Image.fromarray((mask_2d > 0).astype(np.uint8) * 255)
    mask_image = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.array(mask_image) > 0


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], label: str,
              width: int = 3) -> None:
    x1, y1, x2, y2 = bbox
    for offset in range(width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    draw.text((max(0, x1), max(0, y1 - 18)), label, fill=color)


def overlay_component_boxes(
        image_path: str,
        pred_mask_2d: np.ndarray,
        gt_mask_2d: np.ndarray,
        coord_mode: str,
        component_outputs: List[Dict[str, Any]],
        gt_bbox: Optional[Tuple[int, int, int, int]] = None,
        title: Optional[str] = None,
) -> Image.Image:
    image = overlay_masks(image_path, pred_mask_2d, gt_mask_2d, coord_mode, title=title)
    draw = ImageDraw.Draw(image)
    palette = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 165, 0),
        (255, 0, 255),
        (180, 255, 80),
    ]
    for idx, item in enumerate(component_outputs):
        color = palette[idx % len(palette)]
        bbox = item["bbox_qwen"]
        label = item.get("pred_label") or "invalid"
        weight = float(item.get("weight", 0.0))
        rank = int(item.get("component_rank", idx + 1))
        draw_bbox(draw, bbox, color, f"c{rank}:{label}@{weight:.2f}")
    if gt_bbox is not None:
        draw_bbox(draw, gt_bbox, (255, 0, 0), "gt bbox")
    return image


def overlay_masks(
        image_path: str,
        pred_mask_2d: np.ndarray,
        gt_mask_2d: np.ndarray,
        coord_mode: str,
        pred_bbox: Optional[Tuple[int, int, int, int]] = None,
        gt_bbox: Optional[Tuple[int, int, int, int]] = None,
        title: Optional[str] = None,
        pred_label_text: str = "pred bbox",
        gt_label_text: str = "gt bbox",
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    base = np.array(image).astype(np.float32)
    pred_mask_2d = transform_mask(pred_mask_2d, coord_mode)
    gt_mask_2d = transform_mask(gt_mask_2d, coord_mode)
    pred = mask_to_resized_bool(pred_mask_2d, width, height)
    gt = mask_to_resized_bool(gt_mask_2d, width, height)
    overlay = base.copy()
    overlay[pred] = overlay[pred] * 0.45 + np.array([0, 255, 0], dtype=np.float32) * 0.55
    overlay[gt] = overlay[gt] * 0.45 + np.array([255, 0, 0], dtype=np.float32) * 0.55
    both = pred & gt
    overlay[both] = overlay[both] * 0.35 + np.array([255, 255, 0], dtype=np.float32) * 0.65
    rendered = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(rendered)
    if pred_bbox is not None:
        draw_bbox(draw, pred_bbox, (0, 255, 0), pred_label_text)
    if gt_bbox is not None:
        draw_bbox(draw, gt_bbox, (255, 0, 0), gt_label_text)
    if title:
        draw.rectangle((0, 0, width, 28), fill=(0, 0, 0))
        draw.text((6, 6), title, fill=(255, 255, 255))
    return rendered


def show_debug_plt(images: List[Tuple[str, Image.Image]], metadata: Dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for --debug_show_plt. Install it with: pip install matplotlib") from exc
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
    if cols == 1:
        axes = [axes]
    for axis, (title, image) in zip(axes, images):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(
        f"{metadata['sample_id']} | case={metadata['case_id']} | z={metadata['selected_slice']} | "
        f"gt={metadata['gt_label']} pred={metadata.get('pred_label') or 'pending'}",
        fontsize=11,
    )
    plt.tight_layout()
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    plt.show()


def save_component_debug_visuals(
        args: argparse.Namespace,
        debug_dir: Path,
        debug_index: int,
        record: LesionRecord,
        prompt_record: LesionRecord,
        case_id: str,
        selected_z: int,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        component_output: Dict[str, Any],
        gt_bbox_qwen: Tuple[int, int, int, int],
) -> None:
    if not args.debug_visualize:
        return
    pred_slice = pred_mask[:, :, selected_z]
    gt_slice = gt_mask[:, :, selected_z]
    component_rank = int(component_output.get("component_rank", 0))
    bbox_qwen = component_output["bbox_qwen"]
    pred_label = component_output.get("pred_label") or "invalid"
    pred_text = (component_output.get("pred_text") or "").replace("\n", " ").strip()
    if len(pred_text) > 120:
        pred_text = pred_text[:120] + "..."
    component_dir = debug_dir / f"sample_{debug_index:03d}" / "components"
    component_dir.mkdir(parents=True, exist_ok=True)

    title = (
        f"component {component_rank} | gt={record.label} pred={pred_label} "
        f"| w={float(component_output.get('weight', 0.0)):.3f} "
        f"| area={int(component_output.get('area', 0))}"
    )
    image = overlay_masks(
        prompt_record.image_path,
        pred_slice,
        gt_slice,
        args.coord_mode,
        bbox_qwen,
        gt_bbox_qwen,
        title,
        pred_label_text=f"component_{component_rank}",
        gt_label_text="gt bbox",
    )
    image.save(component_dir / f"component_{component_rank}_slice{selected_z}.png")

    meta = {
        "sample_id": record.sample_id,
        "case_id": case_id,
        "selected_slice": selected_z,
        "component_rank": component_rank,
        "gt_label": record.label,
        "pred_label": pred_label,
        "pred_text": component_output.get("pred_text"),
        "weight": float(component_output.get("weight", 0.0)),
        "score": float(component_output.get("score", 0.0)),
        "area": int(component_output.get("area", 0)),
        "area_ratio_to_largest": float(component_output.get("area_ratio_to_largest", 0.0)),
        "bbox_qwen": list(bbox_qwen),
        "gt_bbox_qwen": list(gt_bbox_qwen),
        "prompt": component_output.get("prompt", ""),
    }
    with (component_dir / f"component_{component_rank}_slice{selected_z}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_debug_visuals(
        args: argparse.Namespace,
        debug_dir: Path,
        debug_index: int,
        record: LesionRecord,
        prompt_record: LesionRecord,
        case_id: str,
        selected_z: int,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        pred_bbox_qwen: Tuple[int, int, int, int],
        gt_bbox_qwen: Tuple[int, int, int, int],
        prompt: str,
        pred_text: Optional[str] = None,
        pred_label: Optional[str] = None,
        stage: str = "before_qwen",
        primary_bbox_qwen: Optional[Tuple[int, int, int, int]] = None,
        component_outputs: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if not (args.debug_visualize or args.debug_show_plt) or debug_index > args.debug_max_samples:
        return
    pred_slice = pred_mask[:, :, selected_z]
    gt_slice = gt_mask[:, :, selected_z]
    component_outputs = component_outputs or []

    mask_title = f"mask overlay | case={case_id} z={selected_z} | green=pred red=gt yellow=overlap"
    mask_image = overlay_masks(prompt_record.image_path, pred_slice, gt_slice, args.coord_mode, title=mask_title)

    bbox_title = f"bbox overlay | pred={bbox_to_text(pred_bbox_qwen)} gt={bbox_to_text(gt_bbox_qwen)}"
    bbox_image = overlay_masks(
        prompt_record.image_path,
        pred_slice,
        gt_slice,
        args.coord_mode,
        pred_bbox_qwen,
        gt_bbox_qwen,
        bbox_title,
    )

    qwen_boxes_title = "qwen component boxes"
    qwen_boxes_image = overlay_component_boxes(
        prompt_record.image_path,
        np.zeros_like(pred_slice),
        np.zeros_like(gt_slice),
        args.coord_mode,
        component_outputs,
        gt_bbox_qwen,
        qwen_boxes_title,
    )

    primary_image = None
    if primary_bbox_qwen is not None:
        primary_title = f"primary component bbox | bbox={bbox_to_text(primary_bbox_qwen)}"
        primary_image = overlay_masks(
            prompt_record.image_path,
            pred_slice,
            gt_slice,
            args.coord_mode,
            primary_bbox_qwen,
            gt_bbox_qwen,
            primary_title,
            pred_label_text="primary bbox",
            gt_label_text="gt bbox",
        )

    final_image = None
    if pred_text is not None:
        vote_preview = pred_text.replace("\\n", " ").strip()
        if len(vote_preview) > 120:
            vote_preview = vote_preview[:120] + "..."
        final_title = (
            f"final={pred_label or 'invalid'} | detIoU(union)={iou2d(pred_bbox_qwen, gt_bbox_qwen):.3f} | "
            f"{vote_preview}"
        )
        final_image = overlay_component_boxes(
            prompt_record.image_path,
            pred_slice,
            gt_slice,
            args.coord_mode,
            component_outputs,
            gt_bbox_qwen,
            final_title,
        )

    metadata = {
        "stage": stage,
        "sample_id": record.sample_id,
        "case_id": case_id,
        "selected_slice": selected_z,
        "image_path": prompt_record.image_path,
        "gt_label": record.label,
        "pred_label": pred_label,
        "pred_text": pred_text,
        "pred_bbox_qwen": list(pred_bbox_qwen),
        "gt_bbox_qwen": list(gt_bbox_qwen),
        "component_boxes_qwen": [list(item["bbox_qwen"]) for item in component_outputs],
        "prompt": prompt,
        "note": "Final visualization uses per-component boxes. pred_bbox_qwen is only the union box for detection-style IoU bookkeeping.",
    }
    if args.debug_show_plt:
        images = [
            ("01 mask overlay", mask_image),
            ("02 bbox overlay", bbox_image),
            ("03 Qwen component boxes", qwen_boxes_image),
        ]
        if primary_image is not None:
            images.append(("04 primary component", primary_image))
        if component_outputs:
            for item in component_outputs:
                component_title = (
                    f"c{item['component_rank']} | {item.get('pred_label') or 'invalid'} "
                    f"| w={float(item.get('weight', 0.0)):.3f}"
                )
                component_image = overlay_masks(
                    prompt_record.image_path,
                    pred_slice,
                    gt_slice,
                    args.coord_mode,
                    item["bbox_qwen"],
                    gt_bbox_qwen,
                    component_title,
                    pred_label_text=f"c{item['component_rank']}",
                    gt_label_text="gt bbox",
                )
                images.append((f"component_{item['component_rank']}", component_image))
        if final_image is not None:
            images.append(("final result", final_image))
        show_debug_plt(images, metadata)

    if args.debug_visualize:
        sample_dir = debug_dir / f"{debug_index:04d}_{record.sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        mask_image.save(sample_dir / "01_mask_overlay.png")
        bbox_image.save(sample_dir / "02_bbox_overlay.png")
        qwen_boxes_image.save(sample_dir / "03_qwen_component_boxes.png")
        if primary_image is not None:
            primary_image.save(sample_dir / "04_primary_component_bbox.png")
        if final_image is not None:
            final_image.save(sample_dir / "05_final_result_components.png")
        with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        with (sample_dir / "prompt.txt").open("w", encoding="utf-8") as f:
            f.write(prompt)
        if args.debug_pause:
            input(f"[DEBUG] Saved visual checks to {sample_dir}. Press Enter to continue...")


def select_bbox_for_record(

        record: LesionRecord,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        args: argparse.Namespace,
) -> Tuple[int, List[Dict[str, Any]], Tuple[int, int, int, int], float, str, Dict[str, Any]]:
    record_z = record_slice_index(record)
    if record_z is not None and 0 <= record_z < pred_mask.shape[2]:
        selected_z = record_z
        slice_method = "qwen_record_slice"
    else:
        selected_z = select_slice_index(pred_mask, gt_mask, args.slice_strategy)
        slice_method = args.slice_strategy

    pred_slice = pred_mask[:, :, selected_z]
    gt_bbox = bbox2d(gt_mask[:, :, selected_z])
    pred_slice_nonzero = int(pred_slice.sum())
    behavior = {
        "pred_mask_empty": pred_slice_nonzero == 0,
        "filtered_empty": False,
        "used_fallback_component": False,
        "final_candidate_source": "",
        "num_raw_components": len(connected_components(pred_slice)),
    }

    if pred_slice_nonzero == 0:
        fallback_z, fallback_distance, fallback_area = find_adjacent_candidate_slice(
            pred_mask,
            selected_z,
            radius=args.adjacent_slice_fallback_radius if args.adjacent_slice_fallback else 0,
            strategy=args.adjacent_slice_fallback_strategy,
        )
        if fallback_z is None:
            behavior["final_candidate_source"] = "pred_mask_empty"
            behavior["candidate_slice"] = selected_z
            behavior["candidate_slice_distance"] = 0
            behavior["candidate_slice_area"] = 0
            return selected_z, [], gt_bbox, 0.0, slice_method, behavior
        pred_slice = pred_mask[:, :, fallback_z]
        pred_slice_nonzero = int(pred_slice.sum())
        behavior["final_candidate_source"] = "adjacent_slice_fallback"
        behavior["candidate_slice"] = fallback_z
        behavior["candidate_slice_distance"] = fallback_distance
        behavior["candidate_slice_area"] = fallback_area
        behavior["num_raw_components"] = len(connected_components(pred_slice))
    else:
        behavior["candidate_slice"] = selected_z
        behavior["candidate_slice_distance"] = 0
        behavior["candidate_slice_area"] = pred_slice_nonzero

    candidate_components = extract_component_candidates(
        pred_slice,
        candidate_topk=args.candidate_topk,
        min_component_area=args.min_component_area,
        min_area_ratio=args.candidate_min_area_ratio,
        border_penalty=args.candidate_border_penalty,
    )

    if candidate_components:
        for candidate in candidate_components:
            candidate["bbox"] = expand_bbox(candidate["bbox"], pred_mask.shape[1], pred_mask.shape[0],
                                            args.bbox_expand_ratio)
        if behavior["final_candidate_source"] != "adjacent_slice_fallback":
            behavior["final_candidate_source"] = "filtered_components"
    else:
        behavior["filtered_empty"] = True
        fallback_bbox = bbox_from_mask_slice(pred_slice, args.bbox_strategy, args.component_topk,
                                             args.min_component_area)
        fallback_bbox = expand_bbox(fallback_bbox, pred_mask.shape[1], pred_mask.shape[0], args.bbox_expand_ratio)
        fallback_area = bbox_area(fallback_bbox)
        if fallback_area > 0:
            behavior["used_fallback_component"] = True
            if behavior["final_candidate_source"] == "adjacent_slice_fallback":
                behavior["final_candidate_source"] = "adjacent_slice_largest_component_fallback"
            else:
                behavior["final_candidate_source"] = "largest_component_fallback"
            candidate_components = [
                {
                    "mask": pred_slice > 0,
                    "bbox": fallback_bbox,
                    "area": fallback_area,
                    "area_ratio_to_largest": 1.0,
                    "touches_border": False,
                    "score": 1.0,
                }
            ]
        else:
            if behavior["final_candidate_source"] == "adjacent_slice_fallback":
                behavior["final_candidate_source"] = "adjacent_slice_fallback_empty"
            else:
                behavior["final_candidate_source"] = "fallback_empty"
            return selected_z, [], gt_bbox, 0.0, slice_method, behavior

    union_bbox = union_bboxes([candidate["bbox"] for candidate in candidate_components])
    return selected_z, candidate_components, gt_bbox, iou2d(union_bbox, gt_bbox), slice_method, behavior


def evaluate_qwen_records(args: argparse.Namespace) -> None:
    pred_dir = Path(args.pred_dir)
    label_dir = Path(args.label_dir)
    with Path(args.manifest).open("r", encoding="utf-8") as f:
        manifest_items = json.load(f)
    manifest = {item["case_id"]: item for item in manifest_items}
    case_index = build_case_index(manifest)

    qwen_records = load_record_list(args.qwen_json)
    hidden_records = load_prompt_record_list(args.hidden_qwen_json) if args.hidden_qwen_json else None
    eval_records = pair_gt_and_hidden_records(qwen_records, hidden_records)
    if args.shuffle_eval:
        rng = random.Random(args.seed)
        rng.shuffle(eval_records)
        print(f"[INFO] Shuffled Qwen eval records with seed={args.seed}.", flush=True)
    if args.limit_eval > 0:
        eval_records = eval_records[: args.limit_eval]
        print(f"[INFO] Limited eval records to {len(eval_records)} samples.", flush=True)

    model, processor = load_qwen(args.base_model, args.adapter_path, args.load_in_4bit)

    rows = []
    hidden_json_records = []
    pred_bbox_json_records = []
    mask_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    total = 0
    matched = 0
    cls_correct = 0
    qwen_invoked = 0
    det_hits = {0.3: 0, 0.5: 0}
    joint_hits = {0.3: 0, 0.5: 0}
    strict_loc_hits = 0
    relaxed_loc_hits = 0
    strict_joint_hits = 0
    relaxed_joint_hits = 0
    sum_det_iou = 0.0
    sum_det_iom = 0.0
    sum_gt_coverage = 0.0
    sum_pred_precision = 0.0
    sum_center_hit = 0.0
    confusion = Counter()
    match_methods = Counter()
    slice_methods = Counter()
    behavior_counts = Counter()
    debug_dir = Path(args.debug_dir) if args.debug_dir else Path(args.output_dir) / "debug_visuals"
    debug_saved = 0

    progress = tqdm(eval_records, desc="Eval nnU-Net bbox filled Qwen prompts", ncols=160)
    for record, prompt_record in progress:
        total += 1
        if args.hidden_qwen_json:
            assert_hidden_prompt_is_safe(prompt_record)
        key = (record_patient_id(record), record_seq(record))
        case_id, method = choose_case_id(case_index.get(key, []), record)
        match_methods[method] += 1
        if case_id is None:
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "match_method": method,
                    "patient_id": key[0],
                    "seq": key[1],
                    "gt_label": record.label,
                    "qwen_image_path": record.image_path,
                }
            )
            continue

        pred_mask, gt_mask = load_mask_pair(mask_cache, case_id, pred_dir, label_dir)
        selected_z, candidate_components, gt_bbox, slice_iou, slice_method, behavior = select_bbox_for_record(
            record,
            pred_mask,
            gt_mask,
            args,
        )
        slice_methods[slice_method] += 1
        behavior_counts["pred_mask_empty"] += int(behavior["pred_mask_empty"])
        behavior_counts["filtered_empty"] += int(behavior["filtered_empty"])
        behavior_counts["used_fallback_component"] += int(behavior["used_fallback_component"])
        behavior_counts[f"candidate_source_{behavior['final_candidate_source']}"] += 1
        src_w, src_h = pred_mask.shape[1], pred_mask.shape[0]

        scaled_candidates: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidate_components, start=1):
            scaled_bbox = scale_bbox(candidate["bbox"], (src_w, src_h), record.width, record.height, args.coord_mode)
            scaled_bbox = make_non_empty_bbox(scaled_bbox, record.width, record.height)
            weight = max(float(candidate["score"]), 0.0) * max(float(candidate["area_ratio_to_largest"]), 0.05)
            scaled_candidates.append(
                {
                    **candidate,
                    "component_rank": index,
                    "bbox_qwen": scaled_bbox,
                    "weight": weight,
                }
            )

        if not scaled_candidates:
            matched += 1
            confusion[f"{record.label}->no_candidate"] += 1
            rows.append(
                {
                    "case_id": case_id,
                    "sample_id": record.sample_id,
                    "match_method": method,
                    "patient_id": key[0],
                    "seq": key[1],
                    "selected_slice": selected_z,
                    "candidate_slice": behavior["candidate_slice"],
                    "candidate_slice_distance": behavior["candidate_slice_distance"],
                    "candidate_slice_area": behavior["candidate_slice_area"],
                    "slice_method": slice_method,
                    "pred_mask_empty": behavior["pred_mask_empty"],
                    "filtered_empty": behavior["filtered_empty"],
                    "used_fallback_component": behavior["used_fallback_component"],
                    "final_candidate_source": behavior["final_candidate_source"],
                    "num_raw_components": behavior["num_raw_components"],
                    "gt_label": record.label,
                    "pred_label": "no_candidate",
                    "pred_text": "",
                    "note": "No candidate components available; Qwen was skipped.",
                }
            )
            continue

        fusion_bbox = make_non_empty_bbox(union_bboxes([item["bbox_qwen"] for item in scaled_candidates]), record.width,
                                          record.height)
        primary_bbox = scaled_candidates[0]["bbox_qwen"]
        gt_bbox_scaled = scale_bbox(gt_bbox, (src_w, src_h), record.width, record.height, args.coord_mode)
        det_iou = iou2d(fusion_bbox, record.bbox)
        det_iom = iom2d(fusion_bbox, record.bbox)
        det_gt_coverage = gt_coverage_by_pred(fusion_bbox, record.bbox)
        det_pred_precision = pred_precision_to_gt(fusion_bbox, record.bbox)
        det_center_hit = center_hit(fusion_bbox, record.bbox)
        det_strict_hit = strict_localization_hit(fusion_bbox, record.bbox)
        det_relaxed_hit = relaxed_localization_hit(fusion_bbox, record.bbox)

        component_outputs: List[Dict[str, Any]] = []
        should_debug = (args.debug_visualize or args.debug_show_plt) and debug_saved < args.debug_max_samples
        debug_prompt = ""
        # if should_debug:
        #     debug_saved += 1
        #     debug_prompt = make_replaced_prompt(prompt_record, primary_bbox, 1, len(scaled_candidates))
        #     save_debug_visuals(
        #         args=args,
        #         debug_dir=debug_dir,
        #         debug_index=debug_saved,
        #         record=record,
        #         prompt_record=prompt_record,
        #         case_id=case_id,
        #         selected_z=selected_z,
        #         pred_mask=pred_mask,
        #         gt_mask=gt_mask,
        #         pred_bbox_qwen=fusion_bbox,
        #         gt_bbox_qwen=record.bbox,
        #         prompt=debug_prompt,
        #         stage="before_qwen",
        #         primary_bbox_qwen=primary_bbox,
        #         component_outputs=scaled_candidates,
        #     )

        for candidate in scaled_candidates:
            prompt = make_replaced_prompt(
                prompt_record,
                candidate["bbox_qwen"],
                candidate["component_rank"],
                len(scaled_candidates),
            )
            pred_text = generate_text(
                model,
                processor,
                build_messages(prompt, prompt_record.image_path, args.image_resize),
                max_new_tokens=args.max_new_tokens,
            )
            pred_label = normalize_label(pred_text)
            component_outputs.append(
                {
                    **candidate,
                    "prompt": prompt,
                    "pred_text": pred_text,
                    "pred_label": pred_label,
                }
            )

        if should_debug and args.debug_visualize:
            for component_output in component_outputs:
                save_component_debug_visuals(
                    args=args,
                    debug_dir=debug_dir,
                    debug_index=debug_saved,
                    record=record,
                    prompt_record=prompt_record,
                    case_id=case_id,
                    selected_z=selected_z,
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                    component_output=component_output,
                    gt_bbox_qwen=record.bbox,
                )

        final_label, vote_scores, valid_outputs = fuse_component_predictions(component_outputs)
        best_output = max(component_outputs, key=lambda item: (float(item["weight"]), int(item["area"])))
        final_text = " || ".join(
            f"c{item['component_rank']}:{item.get('pred_label') or 'invalid'}@w={item['weight']:.3f}"
            for item in component_outputs
        )
        if not final_label:
            final_label = best_output.get("pred_label") or ""
        label_ok = final_label == record.label
        qwen_invoked += 1

        if should_debug:
            save_debug_visuals(
                args=args,
                debug_dir=debug_dir,
                debug_index=debug_saved,
                record=record,
                prompt_record=prompt_record,
                case_id=case_id,
                selected_z=selected_z,
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                pred_bbox_qwen=fusion_bbox,
                gt_bbox_qwen=record.bbox,
                prompt=debug_prompt,
                pred_text=final_text,
                pred_label=final_label,
                stage="after_qwen",
                primary_bbox_qwen=primary_bbox,
                component_outputs=component_outputs,
            )

        matched += 1
        cls_correct += int(label_ok)
        strict_loc_hits += int(det_strict_hit)
        relaxed_loc_hits += int(det_relaxed_hit)
        strict_joint_hits += int(det_strict_hit and label_ok)
        relaxed_joint_hits += int(det_relaxed_hit and label_ok)
        sum_det_iou += det_iou
        sum_det_iom += det_iom
        sum_gt_coverage += det_gt_coverage
        sum_pred_precision += det_pred_precision
        sum_center_hit += int(det_center_hit)
        if final_label in {"infection", "tumor"}:
            confusion[f"{record.label}->{final_label}"] += 1
        else:
            confusion["invalid"] += 1
        for thr in det_hits:
            det_ok = det_iou >= thr
            det_hits[thr] += int(det_ok)
            joint_hits[thr] += int(det_ok and label_ok)

        preview = final_text.replace("\n", " ").replace("\r", " ").strip()
        if len(preview) > 48:
            preview = preview[:48] + "..."
        progress.set_postfix(
            {"gt": record.label, "pred": final_label or "invalid", "acc": f"{cls_correct / max(matched, 1):.4f}",
             "text": preview})

        row = {
            "case_id": case_id,
            "sample_id": record.sample_id,
            "match_method": method,
            "patient_id": key[0],
            "seq": key[1],
            "selected_slice": selected_z,
            "candidate_slice": behavior["candidate_slice"],
            "candidate_slice_distance": behavior["candidate_slice_distance"],
            "candidate_slice_area": behavior["candidate_slice_area"],
            "slice_method": slice_method,
            "slice_iou_in_nnunet_space": slice_iou,
            "bbox_strategy": args.bbox_strategy,
            "bbox_expand_ratio": args.bbox_expand_ratio,
            "component_topk": args.component_topk,
            "candidate_topk": args.candidate_topk,
            "candidate_min_area_ratio": args.candidate_min_area_ratio,
            "candidate_border_penalty": args.candidate_border_penalty,
            "fusion_mode": args.fusion_mode,
            "min_component_area": args.min_component_area,
            "pred_mask_empty": behavior["pred_mask_empty"],
            "filtered_empty": behavior["filtered_empty"],
            "used_fallback_component": behavior["used_fallback_component"],
            "final_candidate_source": behavior["final_candidate_source"],
            "num_raw_components": behavior["num_raw_components"],
            "qwen_image_path": record.image_path,
            "prompt_image_path": prompt_record.image_path,
            "uses_hidden_qwen_json": bool(args.hidden_qwen_json),
            "gt_label": record.label,
            "pred_label": final_label or "",
            "pred_text": final_text,
            "valid_component_votes": len(valid_outputs),
            "det_iou_qwen_bbox": det_iou,
            "det_iom_qwen_bbox": det_iom,
            "gt_coverage_by_pred_bbox": det_gt_coverage,
            "pred_precision_to_gt_bbox": det_pred_precision,
            "center_hit": int(det_center_hit),
            "strict_localization_hit": int(det_strict_hit),
            "relaxed_localization_hit": int(det_relaxed_hit),
            "scaled_nnunet_gt_iou_to_qwen_gt": iou2d(gt_bbox_scaled, record.bbox),
            "pred_x1": fusion_bbox[0],
            "pred_y1": fusion_bbox[1],
            "pred_x2": fusion_bbox[2],
            "pred_y2": fusion_bbox[3],
            "primary_pred_x1": primary_bbox[0],
            "primary_pred_y1": primary_bbox[1],
            "primary_pred_x2": primary_bbox[2],
            "primary_pred_y2": primary_bbox[3],
            "vote_score_infection": vote_scores["infection"],
            "vote_score_tumor": vote_scores["tumor"],
            "qwen_gt_x1": record.bbox[0],
            "qwen_gt_y1": record.bbox[1],
            "qwen_gt_x2": record.bbox[2],
            "qwen_gt_y2": record.bbox[3],
        }
        for item in component_outputs:
            prefix = f"component_{item['component_rank']}"
            bbox = item["bbox_qwen"]
            row[f"{prefix}_label"] = item.get("pred_label") or ""
            row[f"{prefix}_text"] = item["pred_text"]
            row[f"{prefix}_weight"] = item["weight"]
            row[f"{prefix}_score"] = item["score"]
            row[f"{prefix}_area"] = item["area"]
            row[f"{prefix}_area_ratio"] = item["area_ratio_to_largest"]
            row[f"{prefix}_touches_border"] = item["touches_border"]
            row[f"{prefix}_x1"] = bbox[0]
            row[f"{prefix}_y1"] = bbox[1]
            row[f"{prefix}_x2"] = bbox[2]
            row[f"{prefix}_y2"] = bbox[3]
        rows.append(row)

        hidden_json_records.append(make_prompt_record(prompt_record, "[x1,y1,x2,y2]", f"hidden_gt_{matched:06d}"))
        pred_bbox_json_records.append(
            make_prompt_record(prompt_record, bbox_to_text(fusion_bbox), f"nnunet_pred_{matched:06d}"))

    metrics = {
        "eval_unit": "qwen_records",
        "total_qwen_records": total,
        "matched_qwen_records": matched,
        "match_rate": matched / max(total, 1),
        "coord_mode": args.coord_mode,
        "slice_strategy": args.slice_strategy,
        "bbox_strategy": args.bbox_strategy,
        "bbox_expand_ratio": args.bbox_expand_ratio,
        "component_topk": args.component_topk,
        "candidate_topk": args.candidate_topk,
        "candidate_min_area_ratio": args.candidate_min_area_ratio,
        "candidate_border_penalty": args.candidate_border_penalty,
        "min_component_area": args.min_component_area,
        "fusion_mode": args.fusion_mode,
        "adjacent_slice_fallback": args.adjacent_slice_fallback,
        "adjacent_slice_fallback_radius": args.adjacent_slice_fallback_radius,
        "adjacent_slice_fallback_strategy": args.adjacent_slice_fallback_strategy,
        "shuffle_eval": args.shuffle_eval,
        "seed": args.seed if args.shuffle_eval else None,
        "match_methods": dict(match_methods),
        "slice_methods": dict(slice_methods),
        "det_recall@iou0.3_over_matched": det_hits[0.3] / max(matched, 1),
        "det_recall@iou0.5_over_matched": det_hits[0.5] / max(matched, 1),
        "det_recall@iou0.3_over_total": det_hits[0.3] / max(total, 1),
        "det_recall@iou0.5_over_total": det_hits[0.5] / max(total, 1),
        "mean_det_iou_over_matched": sum_det_iou / max(matched, 1),
        "mean_det_iom_over_matched": sum_det_iom / max(matched, 1),
        "mean_gt_coverage_over_matched": sum_gt_coverage / max(matched, 1),
        "mean_pred_precision_over_matched": sum_pred_precision / max(matched, 1),
        "center_hit_rate_over_matched": sum_center_hit / max(matched, 1),
        "strict_localization_recall_over_matched": strict_loc_hits / max(matched, 1),
        "relaxed_localization_recall_over_matched": relaxed_loc_hits / max(matched, 1),
        "strict_localization_recall_over_total": strict_loc_hits / max(total, 1),
        "relaxed_localization_recall_over_total": relaxed_loc_hits / max(total, 1),
        "cls_acc_on_selected": cls_correct / max(qwen_invoked, 1),
        "joint_acc@iou0.3_over_matched": joint_hits[0.3] / max(matched, 1),
        "joint_acc@iou0.5_over_matched": joint_hits[0.5] / max(matched, 1),
        "joint_acc@iou0.3_over_total": joint_hits[0.3] / max(total, 1),
        "joint_acc@iou0.5_over_total": joint_hits[0.5] / max(total, 1),
        "joint_acc_strict_over_matched": strict_joint_hits / max(matched, 1),
        "joint_acc_relaxed_over_matched": relaxed_joint_hits / max(matched, 1),
        "joint_acc_strict_over_total": strict_joint_hits / max(total, 1),
        "joint_acc_relaxed_over_total": relaxed_joint_hits / max(total, 1),
        "num_cls_eval_records": qwen_invoked,
        "confusion": dict(confusion),
        "pipeline_behavior": {
            "rate_no_candidate": (confusion["infection->no_candidate"] + confusion["tumor->no_candidate"]) / max(total, 1),
            "rate_fallback_single_component": behavior_counts["used_fallback_component"] / max(total, 1),
            "rate_pred_mask_empty": behavior_counts["pred_mask_empty"] / max(total, 1),
            "rate_filtered_empty": behavior_counts["filtered_empty"] / max(total, 1),
            "rate_adjacent_slice_fallback": (
                behavior_counts["candidate_source_adjacent_slice_fallback"]
                + behavior_counts["candidate_source_adjacent_slice_largest_component_fallback"]
            ) / max(total, 1),
            "num_used_fallback_component": behavior_counts["used_fallback_component"],
            "num_pred_mask_empty": behavior_counts["pred_mask_empty"],
            "num_filtered_empty": behavior_counts["filtered_empty"],
            "num_adjacent_slice_fallback": (
                behavior_counts["candidate_source_adjacent_slice_fallback"]
                + behavior_counts["candidate_source_adjacent_slice_largest_component_fallback"]
            ),
            "candidate_source_counts": {k: v for k, v in behavior_counts.items() if k.startswith("candidate_source_")},
        },
        "qwen_records_loaded": len(qwen_records),
        "hidden_qwen_records_loaded": len(hidden_records) if hidden_records is not None else 0,
        "uses_hidden_qwen_json": bool(args.hidden_qwen_json),
        "gt_qwen_json": args.qwen_json,
        "hidden_qwen_json": args.hidden_qwen_json or [],
        "nnunet_patient_seq_keys": len(case_index),
        "debug_visualize": args.debug_visualize,
        "debug_show_plt": args.debug_show_plt,
        "debug_dir": str(debug_dir) if args.debug_visualize else "",
        "debug_saved_samples": debug_saved,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "nnunet_qwen_remap_predictions.csv"
    preferred = [
        "case_id",
        "sample_id",
        "match_method",
        "patient_id",
        "seq",
        "selected_slice",
        "candidate_slice",
        "candidate_slice_distance",
        "candidate_slice_area",
        "slice_method",
        "slice_iou_in_nnunet_space",
        "bbox_strategy",
        "bbox_expand_ratio",
        "component_topk",
        "candidate_topk",
        "candidate_min_area_ratio",
        "candidate_border_penalty",
        "fusion_mode",
        "min_component_area",
        "qwen_image_path",
        "prompt_image_path",
        "uses_hidden_qwen_json",
        "gt_label",
        "pred_label",
        "pred_text",
        "valid_component_votes",
        "det_iou_qwen_bbox",
        "det_iom_qwen_bbox",
        "gt_coverage_by_pred_bbox",
        "pred_precision_to_gt_bbox",
        "center_hit",
        "strict_localization_hit",
        "relaxed_localization_hit",
        "scaled_nnunet_gt_iou_to_qwen_gt",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
        "primary_pred_x1",
        "primary_pred_y1",
        "primary_pred_x2",
        "primary_pred_y2",
        "vote_score_infection",
        "vote_score_tumor",
        "qwen_gt_x1",
        "qwen_gt_y1",
        "qwen_gt_x2",
        "qwen_gt_y2",
    ]
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        all_fields = sorted({key for row in rows for key in row.keys()})
        fieldnames = [key for key in preferred if key in all_fields] + [key for key in all_fields if
                                                                        key not in preferred]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hidden_json_path = output_dir / "qwen_val_hidden_gt_bbox.json"
    pred_json_path = output_dir / "qwen_val_nnunet_pred_bbox.json"
    with hidden_json_path.open("w", encoding="utf-8") as f:
        json.dump(hidden_json_records, f, ensure_ascii=False, indent=2)
    with pred_json_path.open("w", encoding="utf-8") as f:
        json.dump(pred_bbox_json_records, f, ensure_ascii=False, indent=2)
    metrics["predictions_csv"] = str(rows_path)
    metrics["hidden_gt_bbox_json"] = str(hidden_json_path)
    metrics["nnunet_pred_bbox_json"] = str(pred_json_path)
    metrics_path = output_dir / "nnunet_qwen_remap_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    if args.eval_unit == "qwen_records":
        evaluate_qwen_records(args)
        return

    pred_dir = Path(args.pred_dir)
    label_dir = Path(args.label_dir)
    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_items = json.load(f)
    manifest = {item["case_id"]: item for item in manifest_items}

    qwen_records = []
    for json_path in args.qwen_json:
        qwen_records.extend(load_records(json_path))
    qwen_index = build_qwen_index(qwen_records)
    model, processor = load_qwen(args.base_model, args.adapter_path, args.load_in_4bit)

    rows = []
    hidden_json_records = []
    pred_bbox_json_records = []
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
        selected_z, pred_bbox, gt_bbox, slice_iou = select_slice_and_bbox(
            pred_mask,
            gt_mask,
            slice_strategy=args.slice_strategy,
            bbox_strategy=args.bbox_strategy,
            bbox_expand_ratio=args.bbox_expand_ratio,
            component_topk=args.component_topk,
            min_component_area=args.min_component_area,
        )
        src_w, src_h = pred_mask.shape[1], pred_mask.shape[0]
        key = (str(meta["patient_id"]), manifest_seq(meta))
        candidates = qwen_index.get(key, [])
        if candidates:
            first_record = candidates[0][1]
            gt_bbox_scaled_for_match = scale_bbox(gt_bbox, (src_w, src_h), first_record.width, first_record.height,
                                                  args.coord_mode)
        else:
            gt_bbox_scaled_for_match = gt_bbox
        _, record, method = find_qwen_record(candidates, selected_z, gt_bbox_scaled_for_match)
        match_methods[method] += 1
        if record is None:
            rows.append(
                {
                    "case_id": case_id,
                    "match_method": method,
                    "patient_id": meta.get("patient_id", ""),
                    "seq": meta.get("seq", ""),
                    "qwen_candidates_for_patient": sum(
                        len(items) for (patient, _), items in qwen_index.items() if patient == str(meta["patient_id"])),
                }
            )
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
            prompt_style=args.prompt_style,
            image_resize=args.image_resize,
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
                "sample_id": record.sample_id,
                "match_method": method,
                "patient_id": meta.get("patient_id", ""),
                "seq": meta.get("seq", ""),
                "selected_slice": selected_z,
                "slice_iou_in_nnunet_space": slice_iou,
                "slice_strategy": args.slice_strategy,
                "bbox_strategy": args.bbox_strategy,
                "bbox_expand_ratio": args.bbox_expand_ratio,
                "component_topk": args.component_topk,
                "min_component_area": args.min_component_area,
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
        sample_id = f"nnunet_pred_{matched:06d}"
        hidden_json_records.append(make_prompt_record(record, "[x1,y1,x2,y2]", f"hidden_gt_{matched:06d}"))
        pred_bbox_json_records.append(make_prompt_record(record, bbox_to_text(bbox), sample_id))

    metrics = {
        "total_nnunet_cases": total,
        "matched_qwen_records": matched,
        "match_rate": matched / max(total, 1),
        "coord_mode": args.coord_mode,
        "slice_strategy": args.slice_strategy,
        "bbox_strategy": args.bbox_strategy,
        "bbox_expand_ratio": args.bbox_expand_ratio,
        "component_topk": args.component_topk,
        "min_component_area": args.min_component_area,
        "match_methods": dict(match_methods),
        "det_recall@iou0.3": det_hits[0.3] / max(matched, 1),
        "det_recall@iou0.5": det_hits[0.5] / max(matched, 1),
        "cls_acc_on_selected": cls_correct / max(matched, 1),
        "joint_acc@iou0.3": joint_hits[0.3] / max(matched, 1),
        "joint_acc@iou0.5": joint_hits[0.5] / max(matched, 1),
        "confusion": dict(confusion),
        "qwen_records_loaded": len(qwen_records),
        "qwen_patient_seq_keys": len(qwen_index),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "nnunet_qwen_remap_predictions.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        preferred = [
            "case_id",
            "sample_id",
            "match_method",
            "patient_id",
            "seq",
            "selected_slice",
            "slice_iou_in_nnunet_space",
            "slice_strategy",
            "bbox_strategy",
            "bbox_expand_ratio",
            "component_topk",
            "min_component_area",
            "qwen_image_path",
            "gt_label",
            "pred_label",
            "pred_text",
            "det_iou_qwen_bbox",
            "scaled_nnunet_gt_iou_to_qwen_gt",
            "pred_x1",
            "pred_y1",
            "pred_x2",
            "pred_y2",
            "qwen_gt_x1",
            "qwen_gt_y1",
            "qwen_gt_x2",
            "qwen_gt_y2",
            "qwen_candidates_for_patient",
        ]
        all_fields = sorted({key for row in rows for key in row.keys()})
        fieldnames = [key for key in preferred if key in all_fields] + [key for key in all_fields if
                                                                        key not in preferred]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metrics_path = output_dir / "nnunet_qwen_remap_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    hidden_json_path = output_dir / "qwen_val_hidden_gt_bbox.json"
    pred_json_path = output_dir / "qwen_val_nnunet_pred_bbox.json"
    with hidden_json_path.open("w", encoding="utf-8") as f:
        json.dump(hidden_json_records, f, ensure_ascii=False, indent=2)
    with pred_json_path.open("w", encoding="utf-8") as f:
        json.dump(pred_bbox_json_records, f, ensure_ascii=False, indent=2)
    metrics["hidden_gt_bbox_json"] = str(hidden_json_path)
    metrics["nnunet_pred_bbox_json"] = str(pred_json_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map nnU-Net bboxes back to original Qwen image domain and evaluate Qwen classification.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--qwen_json",
        "--val_json",
        nargs="+",
        required=True,
        help="Ground-truth Qwen JSON used only for labels, GT bboxes, image metadata, and evaluation.",
    )
    parser.add_argument(
        "--hidden_qwen_json",
        nargs="+",
        default=None,
        help="Optional Qwen JSON with true bbox hidden. In qwen_records mode this prompt is filled with nnU-Net bbox for model input.",
    )
    parser.add_argument("--output_dir", default="output/nnunet_qwen_remap_pipeline")
    parser.add_argument(
        "--eval_unit",
        choices=["nnunet_cases", "qwen_records"],
        default="qwen_records",
        help="qwen_records is the paper-facing setting: evaluate every 2D Qwen sample and fill its prompt with nnU-Net bbox.",
    )
    parser.add_argument(
        "--coord_mode",
        choices=["xy", "flip_x", "flip_y", "flip_xy", "swap_xy", "swap_xy_flip_x", "swap_xy_flip_y", "swap_xy_flip_xy"],
        default="xy",
    )
    parser.add_argument(
        "--slice_strategy",
        choices=["pred_max_area", "pred_largest_component", "gt_max_area_oracle", "best_iou_oracle"],
        default="pred_max_area",
        help="How to choose the sagittal slice from the nnU-Net mask. Oracle modes are for diagnosis only.",
    )
    parser.add_argument(
        "--bbox_strategy",
        choices=["mask_bbox", "largest_component", "topk_components_union"],
        default="mask_bbox",
        help="How to convert the selected mask slice into a 2D bbox.",
    )
    parser.add_argument("--bbox_expand_ratio", type=float, default=0.0,
                        help="Expand bbox in nnU-Net space before remapping.")
    parser.add_argument("--component_topk", type=int, default=2, help="Used by topk_components_union.")
    parser.add_argument("--min_component_area", type=int, default=1,
                        help="Ignore connected components smaller than this many pixels.")
    parser.add_argument("--candidate_topk", type=int, default=3,
                        help="Maximum number of filtered components sent to Qwen for per-region classification.")
    parser.add_argument("--candidate_min_area_ratio", type=float, default=0.15,
                        help="Discard components smaller than this ratio of the largest component on the selected slice.")
    parser.add_argument("--candidate_border_penalty", type=float, default=0.15,
                        help="Penalty applied when a component touches the slice border during candidate ranking.")
    parser.add_argument("--fusion_mode", choices=["weighted_vote"], default="weighted_vote",
                        help="How to fuse per-component Qwen predictions.")
    parser.add_argument(
        "--adjacent_slice_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the current Qwen slice has an empty predicted mask, borrow candidates from nearby slices "
            "in the same nnU-Net case while keeping the current Qwen image for classification."
        ),
    )
    parser.add_argument(
        "--adjacent_slice_fallback_radius",
        type=int,
        default=2,
        help="Search radius in slices for --adjacent_slice_fallback.",
    )
    parser.add_argument(
        "--adjacent_slice_fallback_strategy",
        choices=["nearest_then_area", "max_area_within_radius"],
        default="nearest_then_area",
        help="How to select a non-empty adjacent slice when the current slice prediction is empty.",
    )
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--prompt_style", choices=["clean", "legacy"], default="clean")
    parser.add_argument("--image_resize", type=int, default=0)
    parser.add_argument("--shuffle_eval", action="store_true",
                        help="Shuffle Qwen eval records before running inference.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by --shuffle_eval.")
    parser.add_argument("--limit_eval", type=int, default=0,
                        help="Only evaluate the first N records after optional shuffle. Useful for PyCharm debugging.")
    parser.add_argument("--debug_visualize", action="store_true",
                        help="Save per-sample visual checks for mask, bbox, Qwen input, and final prediction.")
    parser.add_argument("--debug_show_plt", action="store_true",
                        help="Show per-sample visual checks with matplotlib instead of only saving images.")
    parser.add_argument("--debug_dir", default="",
                        help="Directory for debug visual outputs. Defaults to output_dir/debug_visuals.")
    parser.add_argument("--debug_max_samples", type=int, default=10, help="Maximum number of samples to visualize.")
    parser.add_argument("--debug_pause", action="store_true", help="Pause after saving each debug visualization stage.")
    return parser


def pycharm_default_argv() -> List[str]:
    argv = [
        "--base_model",
        PYCHARM_DEFAULTS["base_model"],
        "--adapter_path",
        PYCHARM_DEFAULTS["adapter_path"],
        "--pred_dir",
        PYCHARM_DEFAULTS["pred_dir"],
        "--label_dir",
        PYCHARM_DEFAULTS["label_dir"],
        "--manifest",
        PYCHARM_DEFAULTS["manifest"],
        "--output_dir",
        PYCHARM_DEFAULTS["output_dir"],
        "--eval_unit",
        PYCHARM_DEFAULTS["eval_unit"],
        "--image_resize",
        str(PYCHARM_DEFAULTS["image_resize"]),
        "--max_new_tokens",
        str(PYCHARM_DEFAULTS["max_new_tokens"]),
        "--seed",
        str(PYCHARM_DEFAULTS["seed"]),
        "--debug_max_samples",
        str(PYCHARM_DEFAULTS["debug_max_samples"]),
        "--limit_eval",
        str(PYCHARM_DEFAULTS["debug_max_samples"]),
        "--coord_mode",
        str(PYCHARM_DEFAULTS["coord_mode"]),
        "--bbox_strategy",
        str(PYCHARM_DEFAULTS["bbox_strategy"]),
        "--candidate_topk",
        str(PYCHARM_DEFAULTS.get("candidate_topk", 3)),
        "--candidate_min_area_ratio",
        str(PYCHARM_DEFAULTS.get("candidate_min_area_ratio", 0.15)),
        "--candidate_border_penalty",
        str(PYCHARM_DEFAULTS.get("candidate_border_penalty", 0.15)),
    ]
    argv.append("--val_json")
    argv.extend(PYCHARM_DEFAULTS["qwen_json"])
    argv.append("--hidden_qwen_json")
    argv.extend(PYCHARM_DEFAULTS["hidden_qwen_json"])
    if PYCHARM_DEFAULTS.get("load_in_4bit", False):
        argv.append("--load_in_4bit")
    if PYCHARM_DEFAULTS.get("shuffle_eval", False):
        argv.append("--shuffle_eval")
    if PYCHARM_DEFAULTS.get("debug_visualize", False):
        argv.append("--debug_visualize")
    if PYCHARM_DEFAULTS.get("debug_show_plt", False):
        argv.append("--debug_show_plt")
    return argv


def main() -> None:
    import sys

    parser = build_parser()
    if len(sys.argv) == 1:
        print("[INFO] No command-line arguments detected; using PYCHARM_DEFAULTS for local debugging.", flush=True)
        args = parser.parse_args(pycharm_default_argv())
    else:
        args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
