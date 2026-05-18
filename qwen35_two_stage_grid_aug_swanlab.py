import os
import re
import json
import random
import argparse
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from datasets import Dataset
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from swanlab.integration.transformers import SwanLabCallback
import swanlab


# =====================================
# Basic utils
# =====================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str):
    print(f"[INFO] {msg}", flush=True)


def extract_image_path(user_text: str) -> Optional[str]:
    m = re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", user_text, flags=re.S)
    return m.group(1).strip() if m else None


def extract_bbox_from_text(text: str) -> Optional[Tuple[int, int, int, int]]:
    patterns = [
        r"bbox\s*[:：]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r'"bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
        r"位置\s*[:：]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.I | re.S)
        if m:
            return tuple(map(int, m.groups()))
    return None


def extract_label_from_text(text: str) -> Optional[str]:
    low = text.strip().lower()

    m = re.search(r"label\s*[:：]\s*([^\s,，。\]\}]+)", text, flags=re.I)
    if m:
        val = m.group(1).strip().lower()
        if ("感染" in val) or ("infection" in val):
            return "感染"
        if ("肿瘤" in val) or ("tumor" in val) or ("tumour" in val):
            return "肿瘤"

    if ("感染" in text) or ("infection" in low):
        return "感染"
    if ("肿瘤" in text) or ("tumor" in low) or ("tumour" in low):
        return "肿瘤"
    return None


def infer_seq_from_prompt(text: str) -> Optional[str]:
    patterns = [
        r"序列为\s*([Tt][12](?:WI)?)",
        r"([Tt][12](?:WI)?)\s*序列",
        r"\b([Tt][12](?:WI)?)\b",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).upper()
    return None


def clamp_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def maybe_expand_bbox(bbox: Tuple[int, int, int, int], w: int, h: int, ratio: float = 0.1):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * ratio))
    pad_y = int(round(bh * ratio))
    return clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), w, h)


def crop_by_bbox(image: Image.Image, bbox: Tuple[int, int, int, int], expand_ratio: float = 0.1) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = maybe_expand_bbox(bbox, w, h, expand_ratio)
    return image.crop((x1, y1, x2, y2))


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def make_deterministic_rng(text: str, seed: int, tag: str = "default") -> random.Random:
    raw = f"{tag}::{seed}::{text}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    seed_int = int(digest[:8], 16)
    return random.Random(seed_int)


# =====================================
# Grid helpers
# =====================================

def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_to_grid_id(cx: float, cy: float, w: int, h: int, grid_size: int = 3) -> int:
    col = min(int(cx / max(w, 1) * grid_size), grid_size - 1)
    row = min(int(cy / max(h, 1) * grid_size), grid_size - 1)
    return row * grid_size + col + 1


def bbox_to_grid_id(bbox: Tuple[int, int, int, int], w: int, h: int, grid_size: int = 3) -> int:
    cx, cy = bbox_center(bbox)
    return point_to_grid_id(cx, cy, w, h, grid_size)


def grid_id_to_bbox(grid_id: int, w: int, h: int, grid_size: int = 3) -> Tuple[int, int, int, int]:
    idx = grid_id - 1
    row = idx // grid_size
    col = idx % grid_size

    cell_w = w / grid_size
    cell_h = h / grid_size

    x1 = int(round(col * cell_w))
    y1 = int(round(row * cell_h))
    x2 = int(round((col + 1) * cell_w))
    y2 = int(round((row + 1) * cell_h))

    return clamp_bbox((x1, y1, x2, y2), w, h)


def grid_id_to_center(grid_id: int, w: int, h: int, grid_size: int = 3) -> Tuple[int, int]:
    idx = grid_id - 1
    row = idx // grid_size
    col = idx % grid_size

    cell_w = w / grid_size
    cell_h = h / grid_size

    cx = int(round((col + 0.5) * cell_w))
    cy = int(round((row + 0.5) * cell_h))
    return cx, cy


def extract_region_from_text(text: str, grid_size: int = 3) -> Optional[int]:
    m = re.search(r"region\s*[:：]\s*r(\d+)", text, flags=re.I)
    if not m:
        return None
    rid = int(m.group(1))
    max_region = grid_size * grid_size
    if 1 <= rid <= max_region:
        return rid
    return None


def compute_region_classification_metrics(
    gt_regions: List[int],
    pred_regions: List[Optional[int]],
    num_classes: int,
) -> Dict[str, float]:
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    total = len(gt_regions)
    correct = 0

    for gt, pred in zip(gt_regions, pred_regions):
        gt_idx = gt - 1
        if pred is None or pred < 1 or pred > num_classes:
            continue
        pred_idx = pred - 1
        confusion[gt_idx][pred_idx] += 1
        if gt_idx == pred_idx:
            correct += 1

    metrics = {}
    metrics["det_region_acc"] = correct / max(total, 1)

    recalls = []
    precisions = []

    for c in range(num_classes):
        tp = confusion[c][c]
        fn = sum(confusion[c][j] for j in range(num_classes) if j != c)
        fp = sum(confusion[i][c] for i in range(num_classes) if i != c)

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)

        recalls.append(recall)
        precisions.append(precision)

        metrics[f"det_region_recall_r{c+1}"] = recall
        metrics[f"det_region_precision_r{c+1}"] = precision

    metrics["det_region_macro_recall"] = sum(recalls) / num_classes
    metrics["det_region_macro_precision"] = sum(precisions) / num_classes
    return metrics


def print_region_metrics_table(metrics: Dict[str, float], grid_size: int = 3, prefix: str = "det_region_"):
    print("[INFO] 各区域验证指标：")
    print(f"{'Region':<10}{'Recall':<15}{'Precision':<15}")
    for rid in range(1, grid_size * grid_size + 1):
        recall = metrics.get(f"{prefix}recall_r{rid}", 0.0)
        precision = metrics.get(f"{prefix}precision_r{rid}", 0.0)
        print(f"{('r' + str(rid)):<10}{recall:<15.6f}{precision:<15.6f}")


# =====================================
# Detector augmentation helpers
# =====================================

def shift_image_and_bbox(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    dx: int,
    dy: int,
    fill_color=0,
):
    w, h = image.size
    shifted = Image.new(image.mode, (w, h), color=fill_color)
    shifted.paste(image, (dx, dy))

    x1, y1, x2, y2 = bbox
    new_bbox = clamp_bbox((x1 + dx, y1 + dy, x2 + dx, y2 + dy), w, h)
    return shifted, new_bbox


def random_shift_image_and_bbox(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    max_shift_ratio: float = 0.15,
    fill_color=0,
):
    w, h = image.size
    max_dx = int(round(w * max_shift_ratio))
    max_dy = int(round(h * max_shift_ratio))

    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    return shift_image_and_bbox(image, bbox, dx, dy, fill_color=fill_color)


def relocate_image_to_target_grid(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_grid_id: int,
    grid_size: int = 3,
    jitter_ratio: float = 0.15,
    fill_color=0,
):
    w, h = image.size
    cx, cy = bbox_center(bbox)

    target_cx, target_cy = grid_id_to_center(target_grid_id, w, h, grid_size=grid_size)

    cell_w = w / grid_size
    cell_h = h / grid_size
    jitter_x = int(round(cell_w * jitter_ratio))
    jitter_y = int(round(cell_h * jitter_ratio))

    target_cx += random.randint(-jitter_x, jitter_x)
    target_cy += random.randint(-jitter_y, jitter_y)

    dx = int(round(target_cx - cx))
    dy = int(round(target_cy - cy))

    return shift_image_and_bbox(image, bbox, dx, dy, fill_color=fill_color)


# =====================================
# Visualization / debug helpers
# =====================================

def draw_bbox_and_grid(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    grid_size: int = 3,
    label_text: Optional[str] = None,
):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    w, h = img.size
    x1, y1, x2, y2 = bbox

    bbox_color = "red" if img.mode != "L" else 255
    grid_color = "yellow" if img.mode != "L" else 180
    text_color = "white" if img.mode != "L" else 255

    draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=3)

    for i in range(1, grid_size):
        x = int(round(i * w / grid_size))
        y = int(round(i * h / grid_size))
        draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)

    if label_text:
        draw.text((10, 10), label_text, fill=text_color)

    return img


def make_side_by_side(image_left: Image.Image, image_right: Image.Image) -> Image.Image:
    w, h = image_left.size
    canvas = Image.new(image_left.mode, (w * 2, h))
    canvas.paste(image_left, (0, 0))
    canvas.paste(image_right, (w, 0))
    return canvas


def draw_bbox_and_grid_with_region(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    region_id: Optional[int],
    grid_size: int = 3,
    label_text: Optional[str] = None,
):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    w, h = img.size
    x1, y1, x2, y2 = bbox

    bbox_color = "red" if img.mode != "L" else 255
    grid_color = "yellow" if img.mode != "L" else 180
    text_color = "white" if img.mode != "L" else 255

    draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=3)

    for i in range(1, grid_size):
        x = int(round(i * w / grid_size))
        y = int(round(i * h / grid_size))
        draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)

    text_lines = []
    if label_text:
        text_lines.append(label_text)
    if region_id is not None:
        text_lines.append(f"region:r{region_id}")

    if text_lines:
        draw.text((10, 10), "\n".join(text_lines), fill=text_color)

    return img


def export_det_debug_pair(
    before_img: Image.Image,
    before_bbox: Tuple[int, int, int, int],
    before_region: int,
    after_img: Image.Image,
    after_bbox: Tuple[int, int, int, int],
    after_region: int,
    out_path: str,
    grid_size: int = 3,
    extra_text_before: str = "before",
    extra_text_after: str = "after",
):
    left = draw_bbox_and_grid_with_region(
        before_img,
        before_bbox,
        before_region,
        grid_size=grid_size,
        label_text=extra_text_before,
    )
    right = draw_bbox_and_grid_with_region(
        after_img,
        after_bbox,
        after_region,
        grid_size=grid_size,
        label_text=extra_text_after,
    )
    merged = make_side_by_side(left, right)
    merged.save(out_path)


def summarize_region_counter(counter: Counter, grid_size: int) -> Dict[str, Dict[str, float]]:
    total = sum(counter.values())
    out = {}
    for rid in range(1, grid_size * grid_size + 1):
        n = counter.get(rid, 0)
        out[f"r{rid}"] = {
            "count": int(n),
            "ratio": float(n / total) if total > 0 else 0.0
        }
    return out


# =====================================
# Data record
# =====================================

@dataclass
class Record:
    sample_id: str
    image_path: str
    bbox_px: Tuple[int, int, int, int]
    label: str
    seq: Optional[str]
    width: int
    height: int
    patient_id: str


def parse_record(item: Dict) -> Optional[Record]:
    convs = item.get("conversations", [])
    if len(convs) < 2:
        return None

    user_text = convs[0].get("value", "")
    assistant_text = convs[1].get("value", "")

    image_path = extract_image_path(user_text)
    bbox = extract_bbox_from_text(assistant_text) or extract_bbox_from_text(user_text)
    label = extract_label_from_text(assistant_text)
    seq = infer_seq_from_prompt(user_text)

    if image_path is None or bbox is None or label is None:
        return None
    if not os.path.exists(image_path):
        return None

    with Image.open(image_path) as img:
        w, h = img.size

    bbox = clamp_bbox(bbox, w, h)
    patient_id = os.path.basename(image_path).split("_")[0]
    sample_id = item.get("id", "") or f"sample_{os.path.basename(image_path)}"

    return Record(
        sample_id=sample_id,
        image_path=image_path,
        bbox_px=bbox,
        label=label,
        seq=seq,
        width=w,
        height=h,
        patient_id=patient_id,
    )


def load_records(json_path: str) -> List[Record]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    skipped = 0
    for item in data:
        rec = parse_record(item)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    log(f"Loaded from {json_path}: {len(records)}, skipped: {skipped}")
    return records


# =====================================
# Prompts / answers
# =====================================

def build_det_prompt(seq: Optional[str], grid_size: int = 3) -> str:
    seq_part = f"，序列为{seq}" if seq else ""
    max_region = grid_size * grid_size
    region_list = ",".join([f"r{i}" for i in range(1, max_region + 1)])
    return (
        f"现在你是一个骨科专家，这是一幅脊椎的磁共振图像{seq_part},该图像中可能包含了数个病灶。"
        f"请观察整张图像，并着重关注脊柱部分，然后请判断图中最主要病灶位于哪个网格区域。"
        f"整张图像被划分为{grid_size}x{grid_size}网格区域，"
        f"区域编号为：{region_list}。"
        f"请严格按照以下格式回答：region:r?"
    )


def build_det_answer(bbox_px: Tuple[int, int, int, int], w: int, h: int, grid_size: int = 3) -> str:
    region_id = bbox_to_grid_id(bbox_px, w, h, grid_size=grid_size)
    return f"region:r{region_id}"


def build_cls_prompt(seq: Optional[str]) -> str:
    seq_part = f"，序列为{seq}" if seq else ""
    return (
        f"现在你是一个骨科专家，这是一幅脊椎病灶区域的磁共振图像{seq_part}。"
        f"请根据该病灶区域判断它属于感染还是肿瘤。"
        f"请严格按照以下格式回答：label:感染/肿瘤"
    )


def build_cls_answer(label: str) -> str:
    return f"label:{label}"


# =====================================
# Balanced detector dataset expansion
# =====================================

def build_region_to_records(records: List[Record], grid_size: int) -> Dict[int, List[Record]]:
    region_to_records = defaultdict(list)
    for rec in records:
        rid = bbox_to_grid_id(rec.bbox_px, rec.width, rec.height, grid_size=grid_size)
        region_to_records[rid].append(rec)
    return region_to_records


def choose_balance_target(
    records: List[Record],
    grid_size: int,
    det_balance_target: int,
    det_balance_target_mode: str,
) -> int:
    counts = Counter()
    for rec in records:
        rid = bbox_to_grid_id(rec.bbox_px, rec.width, rec.height, grid_size=grid_size)
        counts[rid] += 1

    vals = [counts.get(r, 0) for r in range(1, grid_size * grid_size + 1)]

    if det_balance_target > 0:
        return det_balance_target

    if det_balance_target_mode == "max":
        return max(vals)
    if det_balance_target_mode == "mean":
        return int(round(sum(vals) / max(len(vals), 1)))
    if det_balance_target_mode == "median":
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        if n == 0:
            return 0
        if n % 2 == 1:
            return vals_sorted[n // 2]
        return int(round((vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2))
    if det_balance_target_mode == "p75":
        vals_sorted = sorted(vals)
        if not vals_sorted:
            return 0
        idx = int(round((len(vals_sorted) - 1) * 0.75))
        return vals_sorted[idx]
    if det_balance_target_mode == "fixed":
        return det_balance_target

    raise ValueError(f"Unknown det_balance_target_mode: {det_balance_target_mode}")


def expand_detector_records_balanced(
    records: List[Record],
    grid_size: int,
    target_per_region: int,
    seed: int = 42,
) -> Tuple[List[Dict], Dict]:
    """
    1. 每个区域最多保留 target_per_region 条原始样本
    2. 超出的原始样本作为 donor，优先迁移到缺样区域
    3. donor 不足时，再少量复制补齐
    """
    region_to_records = build_region_to_records(records, grid_size=grid_size)

    original_counter = Counter()
    for rid in range(1, grid_size * grid_size + 1):
        original_counter[rid] = len(region_to_records.get(rid, []))

    keep_rows: List[Dict] = []
    donor_pool: List[Record] = []
    final_counter = Counter()
    donor_used_counter = Counter()
    added_copy_counter = Counter()

    # Step 1. 保留最多 target_per_region 条，剩余进入 donor_pool
    for rid in range(1, grid_size * grid_size + 1):
        recs = list(region_to_records.get(rid, []))
        rng = make_deterministic_rng(f"keep_region_{rid}", seed, tag="det_balance_keep")
        rng.shuffle(recs)

        keep_num = min(len(recs), target_per_region)
        keep_recs = recs[:keep_num]
        extra_recs = recs[keep_num:]

        for rec in keep_recs:
            keep_rows.append({
                "sample_id": rec.sample_id,
                "image_path": rec.image_path,
                "bbox_px": list(rec.bbox_px),
                "label": rec.label,
                "seq": rec.seq,
                "width": rec.width,
                "height": rec.height,
                "patient_id": rec.patient_id,
                "det_aug_mode": "none",
                "det_target_region": rid,
                "det_aug_uid": f"{rec.sample_id}::orig_keep::{rid}",
            })
            final_counter[rid] += 1

        donor_pool.extend(extra_recs)

    # Step 2. 统计缺口
    deficits = []
    for rid in range(1, grid_size * grid_size + 1):
        need = max(0, target_per_region - final_counter[rid])
        deficits.extend([rid] * need)

    rng_def = make_deterministic_rng("deficit_assign", seed, tag="det_balance_deficit")
    rng_def.shuffle(deficits)

    # Step 3. 优先使用 donor_pool
    rng_donor = make_deterministic_rng("donor_pool", seed, tag="det_balance_donor")
    rng_donor.shuffle(donor_pool)

    relocated_rows: List[Dict] = []
    used_donor_num = min(len(donor_pool), len(deficits))

    for i in range(used_donor_num):
        src = donor_pool[i]
        target_region = deficits[i]

        relocated_rows.append({
            "sample_id": src.sample_id,
            "image_path": src.image_path,
            "bbox_px": list(src.bbox_px),
            "label": src.label,
            "seq": src.seq,
            "width": src.width,
            "height": src.height,
            "patient_id": src.patient_id,
            "det_aug_mode": "relocate_to_target",
            "det_target_region": target_region,
            "det_aug_uid": f"{src.sample_id}::donor_to::{target_region}::{i}",
        })
        final_counter[target_region] += 1
        donor_used_counter[target_region] += 1

    remain_deficits = deficits[used_donor_num:]

    # Step 4. donor 不足时再复制补齐
    extra_copy_rows: List[Dict] = []
    all_source_records = list(records)

    if len(remain_deficits) > 0 and len(all_source_records) > 0:
        rng_copy = make_deterministic_rng("copy_fill", seed, tag="det_balance_copy")
        for j, target_region in enumerate(remain_deficits):
            src = all_source_records[rng_copy.randrange(len(all_source_records))]
            extra_copy_rows.append({
                "sample_id": src.sample_id,
                "image_path": src.image_path,
                "bbox_px": list(src.bbox_px),
                "label": src.label,
                "seq": src.seq,
                "width": src.width,
                "height": src.height,
                "patient_id": src.patient_id,
                "det_aug_mode": "relocate_to_target",
                "det_target_region": target_region,
                "det_aug_uid": f"{src.sample_id}::copy_to::{target_region}::{j}",
            })
            final_counter[target_region] += 1
            added_copy_counter[target_region] += 1

    expanded_rows = keep_rows + relocated_rows + extra_copy_rows

    stats = {
        "target_per_region": target_per_region,
        "original_distribution": {
            f"r{rid}": int(original_counter[rid]) for rid in range(1, grid_size * grid_size + 1)
        },
        "kept_distribution": {
            f"r{rid}": int(min(original_counter[rid], target_per_region))
            for rid in range(1, grid_size * grid_size + 1)
        },
        "donor_relocated_distribution": {
            f"r{rid}": int(donor_used_counter[rid]) for rid in range(1, grid_size * grid_size + 1)
        },
        "extra_copy_distribution": {
            f"r{rid}": int(added_copy_counter[rid]) for rid in range(1, grid_size * grid_size + 1)
        },
        "final_distribution": {
            f"r{rid}": int(final_counter[rid]) for rid in range(1, grid_size * grid_size + 1)
        },
        "num_original_records": len(records),
        "num_kept_records": len(keep_rows),
        "num_donor_relocated_records": len(relocated_rows),
        "num_extra_copy_records": len(extra_copy_rows),
        "num_final_records": len(expanded_rows),
        "num_donor_pool": len(donor_pool),
        "num_unfilled_by_donor": max(0, len(deficits) - len(donor_pool)),
    }
    return expanded_rows, stats


def print_balanced_expansion_stats(stats: Dict, grid_size: int):
    print("[INFO] 原始 detector 训练集区域分布：")
    for rid in range(1, grid_size * grid_size + 1):
        print(f"  r{rid}: {stats['original_distribution'].get(f'r{rid}', 0)}")

    print("[INFO] 保留为原区域样本数：")
    for rid in range(1, grid_size * grid_size + 1):
        print(f"  r{rid}: {stats['kept_distribution'].get(f'r{rid}', 0)}")

    print("[INFO] 使用 donor 重分配到各目标区域的样本数：")
    for rid in range(1, grid_size * grid_size + 1):
        print(f"  r{rid}: {stats['donor_relocated_distribution'].get(f'r{rid}', 0)}")

    print("[INFO] donor 不足时额外复制补齐到各目标区域的样本数：")
    for rid in range(1, grid_size * grid_size + 1):
        print(f"  r{rid}: {stats['extra_copy_distribution'].get(f'r{rid}', 0)}")

    print("[INFO] 最终均衡后 detector 训练集区域分布：")
    for rid in range(1, grid_size * grid_size + 1):
        print(f"  r{rid}: {stats['final_distribution'].get(f'r{rid}', 0)}")

    total = max(stats["num_final_records"], 1)
    print("[INFO] 最终均衡后 detector 训练集区域占比：")
    for rid in range(1, grid_size * grid_size + 1):
        cnt = stats["final_distribution"].get(f"r{rid}", 0)
        print(f"  r{rid}: {cnt} ({cnt / total:.4%})")

    print(f"[INFO] target_per_region = {stats['target_per_region']}")
    print(f"[INFO] 原始样本数 = {stats['num_original_records']}")
    print(f"[INFO] 保留原样本数 = {stats['num_kept_records']}")
    print(f"[INFO] donor 重分配样本数 = {stats['num_donor_relocated_records']}")
    print(f"[INFO] donor 不足额外复制数 = {stats['num_extra_copy_records']}")
    print(f"[INFO] donor_pool 总数 = {stats['num_donor_pool']}")
    print(f"[INFO] 最终样本数 = {stats['num_final_records']}")


def save_balanced_expansion_stats(stats: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "train_det_balanced_distribution_before_training.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 均衡增强统计已保存到: {save_path}")


# =====================================
# Dataset builder
# =====================================

class TwoStageDatasetBuilder:
    def __init__(
        self,
        processor,
        task: str,
        max_length: int,
        crop_expand_ratio: float = 0.1,
        grid_size: int = 3,
        det_shift_max_ratio: float = 0.15,
        aug_seed: int = 42,
        det_debug_aug: bool = False,
        det_debug_aug_dir: Optional[str] = None,
        det_debug_aug_num: int = 50,
        det_debug_print: bool = False,
    ):
        self.processor = processor
        self.task = task
        self.max_length = max_length
        self.crop_expand_ratio = crop_expand_ratio
        self.grid_size = grid_size
        self.det_shift_max_ratio = det_shift_max_ratio
        self.aug_seed = aug_seed
        self.tokenizer = processor.tokenizer

        self.det_debug_aug = det_debug_aug
        self.det_debug_aug_dir = det_debug_aug_dir
        self.det_debug_aug_num = det_debug_aug_num
        self.det_debug_print = det_debug_print
        self.det_debug_saved = 0

        if self.det_debug_aug and self.det_debug_aug_dir is not None:
            os.makedirs(self.det_debug_aug_dir, exist_ok=True)

    def _maybe_export_det_debug(
        self,
        sample_id: str,
        det_aug_mode: str,
        det_target_region: Optional[int],
        before_img: Image.Image,
        before_bbox: Tuple[int, int, int, int],
        before_region: int,
        after_img: Image.Image,
        after_bbox: Tuple[int, int, int, int],
        after_region: int,
        answer_before: str,
        answer_after: str,
    ):
        if not self.det_debug_aug:
            return
        if self.det_debug_aug_dir is None:
            return
        if self.det_debug_saved >= self.det_debug_aug_num:
            return
        if det_aug_mode == 'none':
            return

        safe_id = str(sample_id).replace("/", "_").replace(":", "_")
        out_name = f"{self.det_debug_saved:04d}_{safe_id}_{det_aug_mode}_r{before_region}_to_r{after_region}.png"
        out_path = os.path.join(self.det_debug_aug_dir, out_name)

        export_det_debug_pair(
            before_img=before_img,
            before_bbox=before_bbox,
            before_region=before_region,
            after_img=after_img,
            after_bbox=after_bbox,
            after_region=after_region,
            out_path=out_path,
            grid_size=self.grid_size,
            extra_text_before=f"before | {answer_before}",
            extra_text_after=f"after | {answer_after}",
        )

        if self.det_debug_print:
            print(
                f"[DET-AUG-DEBUG] sample_id={sample_id} "
                f"mode={det_aug_mode} "
                f"target_region={det_target_region} "
                f"before={answer_before} "
                f"after={answer_after} "
                f"saved={out_path}"
            )

        self.det_debug_saved += 1

    def _build_messages_and_answer(self, example: Dict):
        sample_id = example["sample_id"]
        image_path = example["image_path"]
        label = example["label"]
        bbox = tuple(example["bbox_px"])
        seq = example["seq"]
        width = example["width"]
        height = example["height"]

        det_aug_mode = example.get("det_aug_mode", "none")
        det_target_region = example.get("det_target_region", None)
        det_aug_uid = example.get("det_aug_uid", f"{sample_id}::default")

        if self.task == "det":
            prompt = build_det_prompt(seq, grid_size=self.grid_size)

            img = Image.open(image_path).convert("RGB")
            orig_img = img.copy()
            orig_bbox = bbox
            orig_region = bbox_to_grid_id(orig_bbox, width, height, grid_size=self.grid_size)
            orig_answer = build_det_answer(orig_bbox, width, height, grid_size=self.grid_size)

            if det_aug_mode == "relocate_to_target":
                rng = make_deterministic_rng(det_aug_uid, self.aug_seed, tag="det_balance_relocate")
                old_state = random.getstate()
                random.seed(rng.randint(0, 10**9))
                img, bbox = relocate_image_to_target_grid(
                    img,
                    bbox,
                    target_grid_id=int(det_target_region),
                    grid_size=self.grid_size,
                    jitter_ratio=0.2,
                    fill_color=0,
                )
                random.setstate(old_state)

            elif det_aug_mode == "random_shift":
                rng = make_deterministic_rng(det_aug_uid, self.aug_seed, tag="det_random_shift")
                old_state = random.getstate()
                random.seed(rng.randint(0, 10**9))
                img, bbox = random_shift_image_and_bbox(
                    img,
                    bbox,
                    max_shift_ratio=self.det_shift_max_ratio,
                    fill_color=0,
                )
                random.setstate(old_state)

            aug_region = bbox_to_grid_id(bbox, width, height, grid_size=self.grid_size)
            answer = build_det_answer(bbox, width, height, grid_size=self.grid_size)

            self._maybe_export_det_debug(
                sample_id=sample_id,
                det_aug_mode=det_aug_mode,
                det_target_region=det_target_region,
                before_img=orig_img,
                before_bbox=orig_bbox,
                before_region=orig_region,
                after_img=img.copy(),
                after_bbox=bbox,
                after_region=aug_region,
                answer_before=orig_answer,
                answer_after=answer,
            )

            image_obj = img

        elif self.task == "cls":
            prompt = build_cls_prompt(seq)
            answer = build_cls_answer(label)
            img = Image.open(image_path).convert("RGB")
            image_obj = crop_by_bbox(img, bbox, expand_ratio=self.crop_expand_ratio)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_obj},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages, answer

    def __call__(self, example: Dict) -> Dict:
        messages, answer = self._build_messages_and_answer(example)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        response = self.tokenizer(answer, add_special_tokens=False, return_tensors=None)

        prompt_ids = inputs["input_ids"][0].tolist()
        prompt_mask = inputs["attention_mask"][0].tolist()
        resp_ids = response["input_ids"]
        resp_mask = response["attention_mask"]

        eos_id = self.tokenizer.eos_token_id
        input_ids = prompt_ids + resp_ids + [eos_id]
        attention_mask = prompt_mask + resp_mask + [1]
        labels = [-100] * len(prompt_ids) + resp_ids + [eos_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs["pixel_values"][0].tolist(),
            "image_grid_thw": inputs["image_grid_thw"][0].tolist(),
        }
        return out


class VLDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def _to_tensor(self, x, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        return torch.tensor(x, dtype=dtype)

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = [self._to_tensor(f["input_ids"], torch.long) for f in features]
        attention_mask_list = [self._to_tensor(f["attention_mask"], torch.long) for f in features]
        labels_list = [self._to_tensor(f["labels"], torch.long) for f in features]
        pixel_values_list = [self._to_tensor(f["pixel_values"], torch.float32) for f in features]
        image_grid_thw_list = [self._to_tensor(f["image_grid_thw"], torch.long) for f in features]

        max_len = max(x.size(0) for x in input_ids_list)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
            l = input_ids.size(0)
            pad_len = max_len - l

            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros((pad_len,), dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_labels.append(labels)

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
            "pixel_values": torch.stack(pixel_values_list),
            "image_grid_thw": torch.stack(image_grid_thw_list),
        }
        return batch


def make_hf_dataset_from_records(records: List[Record], grid_size: int) -> Dataset:
    rows = []
    for r in records:
        orig_region = bbox_to_grid_id(r.bbox_px, r.width, r.height, grid_size=grid_size)
        rows.append(
            {
                "sample_id": r.sample_id,
                "image_path": r.image_path,
                "bbox_px": list(r.bbox_px),
                "label": r.label,
                "seq": r.seq,
                "width": r.width,
                "height": r.height,
                "patient_id": r.patient_id,
                "det_aug_mode": "none",
                "det_target_region": orig_region,
                "det_aug_uid": f"{r.sample_id}::orig",
            }
        )
    return Dataset.from_list(rows)


def make_hf_dataset_from_rows(rows: List[Dict]) -> Dataset:
    return Dataset.from_list(rows)


# =====================================
# Model helpers
# =====================================

def load_model_and_processor(model_name_or_path: str, load_in_4bit: bool, gradient_checkpointing: bool):
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, processor


def add_lora(model, r: int, alpha: int, dropout: float):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# =====================================
# Prediction
# =====================================

def build_messages(prompt: str, image_obj):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_text(model, processor, messages, max_new_tokens: int = 64) -> str:
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = model.get_input_embeddings().weight.device
    model_inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    prompt_len = model_inputs["input_ids"].shape[1]
    trimmed = generated_ids[:, prompt_len:]
    out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out[0].strip()


def predict_region(det_model, det_processor, image_path: str, seq: Optional[str], max_new_tokens: int = 16,
                   grid_size: int = 3):
    prompt = build_det_prompt(seq, grid_size=grid_size)
    messages = build_messages(prompt, image_path)
    text = generate_text(det_model, det_processor, messages, max_new_tokens=max_new_tokens)
    region_id = extract_region_from_text(text, grid_size=grid_size)
    return text, region_id


def predict_label(cls_model, cls_processor, image_path: str, bbox_px: Tuple[int, int, int, int], seq: Optional[str],
                  max_new_tokens: int = 16):
    img = Image.open(image_path).convert("RGB")
    crop = crop_by_bbox(img, bbox_px, expand_ratio=0.1)
    prompt = build_cls_prompt(seq)
    messages = build_messages(prompt, crop)
    text = generate_text(cls_model, cls_processor, messages, max_new_tokens=max_new_tokens)
    label = extract_label_from_text(text)
    return text, label


# =====================================
# Evaluation
# =====================================

def evaluate_detector(model, processor, records: List[Record], log_image_num: int = 20, grid_size: int = 3):
    total = 0
    iou03 = 0
    iou05 = 0
    image_logs = []

    gt_regions = []
    pred_regions = []

    pbar = tqdm(records, desc="Eval Detector", ncols=140)
    for rec in pbar:
        total += 1

        pred_text, pred_region = predict_region(
            model, processor, rec.image_path, rec.seq, max_new_tokens=16, grid_size=grid_size
        )

        gt_region = bbox_to_grid_id(rec.bbox_px, rec.width, rec.height, grid_size=grid_size)

        gt_regions.append(gt_region)
        pred_regions.append(pred_region)

        iou = 0.0
        region_correct = int(pred_region == gt_region)

        if pred_region is not None:
            pred_bbox_px = grid_id_to_bbox(pred_region, rec.width, rec.height, grid_size=grid_size)
            iou = compute_iou(pred_bbox_px, rec.bbox_px)
            iou03 += int(iou >= 0.3)
            iou05 += int(iou >= 0.5)

        running_acc = sum(int(g == p) for g, p in zip(gt_regions, pred_regions)) / max(len(gt_regions), 1)

        pbar.set_postfix(
            acc=f"{running_acc:.4f}",
            hit=f"{region_correct}",
            iou03=f"{iou03 / max(total, 1):.4f}",
            iou05=f"{iou05 / max(total, 1):.4f}",
            iou=f"{iou:.3f}",
            gt=f"r{gt_region}",
            pred=f"r{pred_region}" if pred_region is not None else "None",
        )

        if len(image_logs) < log_image_num:
            try:
                image_logs.append(
                    swanlab.Image(
                        rec.image_path,
                        caption=(
                            f"pred={pred_text} | gt_region=r{gt_region} | "
                            f"pred_region={('r' + str(pred_region)) if pred_region is not None else 'None'} | "
                            f"gt_bbox={list(rec.bbox_px)} | iou={iou:.3f}"
                        )
                    )
                )
            except Exception:
                pass

    cls_metrics = compute_region_classification_metrics(
        gt_regions=gt_regions,
        pred_regions=pred_regions,
        num_classes=grid_size * grid_size,
    )

    metrics = {
        **cls_metrics,
        "det_iou03": iou03 / max(total, 1),
        "det_iou05": iou05 / max(total, 1),
        "det_total": total,
    }
    return metrics, image_logs


def evaluate_classifier(model, processor, records: List[Record], log_image_num: int = 20):
    total = 0
    correct = 0
    image_logs = []

    pbar = tqdm(records, desc="Eval Classifier", ncols=120)
    for rec in pbar:
        total += 1
        pred_text, pred_label = predict_label(model, processor, rec.image_path, rec.bbox_px, rec.seq)
        correct += int(pred_label == rec.label)
        pbar.set_postfix(acc=f"{correct / max(total, 1):.4f}", gt=rec.label, pred=str(pred_label))

        if len(image_logs) < log_image_num:
            try:
                image_logs.append(
                    swanlab.Image(
                        rec.image_path,
                        caption=f"pred={pred_text} | gt_label={rec.label}"
                    )
                )
            except Exception:
                pass

    metrics = {
        "cls_acc": correct / max(total, 1),
        "cls_total": total,
    }
    return metrics, image_logs


def evaluate_pipeline(det_model, det_processor, cls_model, cls_processor, records: List[Record],
                      log_image_num: int = 20, grid_size: int = 3):
    total = 0
    det03 = 0
    det05 = 0
    cls_acc = 0
    joint03 = 0
    joint05 = 0

    patient_stats = {}
    image_logs = []

    gt_regions = []
    pred_regions = []

    pbar = tqdm(records, desc="Eval Two-Stage Pipeline", ncols=140)
    for rec in pbar:
        total += 1

        det_text, pred_region = predict_region(
            det_model, det_processor, rec.image_path, rec.seq, max_new_tokens=16, grid_size=grid_size
        )

        gt_region = bbox_to_grid_id(rec.bbox_px, rec.width, rec.height, grid_size=grid_size)
        gt_regions.append(gt_region)
        pred_regions.append(pred_region)

        pred_bbox_px = None
        pred_label = None
        cls_text = ""
        iou = 0.0

        if pred_region is not None:
            pred_bbox_px = grid_id_to_bbox(pred_region, rec.width, rec.height, grid_size=grid_size)
            iou = compute_iou(pred_bbox_px, rec.bbox_px)
            det03 += int(iou >= 0.3)
            det05 += int(iou >= 0.5)

            cls_text, pred_label = predict_label(cls_model, cls_processor, rec.image_path, pred_bbox_px, rec.seq)
            cls_acc += int(pred_label == rec.label)
            joint03 += int((iou >= 0.3) and (pred_label == rec.label))
            joint05 += int((iou >= 0.5) and (pred_label == rec.label))

        if rec.patient_id not in patient_stats:
            patient_stats[rec.patient_id] = {"total": 0, "joint03": 0}
        patient_stats[rec.patient_id]["total"] += 1
        patient_stats[rec.patient_id]["joint03"] += int((iou >= 0.3) and (pred_label == rec.label))

        running_acc = sum(int(g == p) for g, p in zip(gt_regions, pred_regions)) / max(len(gt_regions), 1)

        pbar.set_postfix(
            region_acc=f"{running_acc:.4f}",
            det03=f"{det03 / max(total, 1):.4f}",
            cls=f"{cls_acc / max(total, 1):.4f}",
            j03=f"{joint03 / max(total, 1):.4f}",
            iou=f"{iou:.3f}",
            gt=rec.label,
            pred=str(pred_label),
        )

        if len(image_logs) < log_image_num:
            try:
                image_logs.append(
                    swanlab.Image(
                        rec.image_path,
                        caption=(
                            f"det_pred={det_text} | gt_region=r{gt_region} | "
                            f"pred_region={('r' + str(pred_region)) if pred_region is not None else 'None'} | "
                            f"cls_pred={cls_text} | gt_bbox={list(rec.bbox_px)} | "
                            f"gt_label={rec.label} | iou={iou:.3f}"
                        )
                    )
                )
            except Exception:
                pass

    patient_joint03 = 0
    for _, st in patient_stats.items():
        if st["joint03"] / max(st["total"], 1) > 0.5:
            patient_joint03 += 1

    region_metrics = compute_region_classification_metrics(
        gt_regions=gt_regions,
        pred_regions=pred_regions,
        num_classes=grid_size * grid_size,
    )

    metrics = {
        "pipeline_region_acc": region_metrics["det_region_acc"],
        "pipeline_region_macro_recall": region_metrics["det_region_macro_recall"],
        "pipeline_region_macro_precision": region_metrics["det_region_macro_precision"],
        "pipeline_det_iou03": det03 / max(total, 1),
        "pipeline_det_iou05": det05 / max(total, 1),
        "pipeline_cls_acc": cls_acc / max(total, 1),
        "pipeline_joint_iou03": joint03 / max(total, 1),
        "pipeline_joint_iou05": joint05 / max(total, 1),
        "pipeline_patient_joint_iou03": patient_joint03 / max(len(patient_stats), 1),
        "pipeline_total_images": total,
        "pipeline_total_patients": len(patient_stats),
    }

    for k, v in region_metrics.items():
        if k.startswith("det_region_recall_") or k.startswith("det_region_precision_"):
            metrics["pipeline_" + k.replace("det_region_", "region_")] = v

    return metrics, image_logs


# =====================================
# Training entry
# =====================================

def train_stage(
        task: str,
        model_name_or_path: str,
        train_json: str,
        val_json: str,
        output_dir: str,
        max_length: int,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        gradient_accumulation_steps: int,
        num_train_epochs: float,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        logging_steps: int,
        save_steps: int,
        eval_steps: int,
        save_total_limit: int,
        load_in_4bit: bool,
        gradient_checkpointing: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        crop_expand_ratio: float,
        swanlab_project: str,
        swanlab_experiment: str,
        grid_size: int,
        det_shift_max_ratio: float,
        det_balance_enable: bool,
        det_balance_target: int,
        det_balance_target_mode: str,
        seed: int,
        det_debug_aug: bool,
        det_debug_aug_dir: Optional[str],
        det_debug_aug_num: int,
        det_debug_print: bool,
):
    log(f"Loading records for task={task} ...")
    train_records = load_records(train_json)
    val_records = load_records(val_json)

    if task == "det" and det_balance_enable:
        log("Building balanced detector training dataset ...")
        target_per_region = choose_balance_target(
            records=train_records,
            grid_size=grid_size,
            det_balance_target=det_balance_target,
            det_balance_target_mode=det_balance_target_mode,
        )
        expanded_rows, balance_stats = expand_detector_records_balanced(
            records=train_records,
            grid_size=grid_size,
            target_per_region=target_per_region,
            seed=seed,
        )
        print_balanced_expansion_stats(balance_stats, grid_size=grid_size)
        save_balanced_expansion_stats(balance_stats, output_dir=output_dir)
        train_dataset_raw = make_hf_dataset_from_rows(expanded_rows)
        effective_train_size = len(expanded_rows)
    else:
        train_dataset_raw = make_hf_dataset_from_records(train_records, grid_size=grid_size)
        effective_train_size = len(train_records)

    val_dataset_raw = make_hf_dataset_from_records(val_records, grid_size=grid_size)

    log("Loading model and processor ...")
    model, processor = load_model_and_processor(
        model_name_or_path=model_name_or_path,
        load_in_4bit=load_in_4bit,
        gradient_checkpointing=gradient_checkpointing,
    )
    model = add_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    builder = TwoStageDatasetBuilder(
        processor=processor,
        task=task,
        max_length=max_length,
        crop_expand_ratio=crop_expand_ratio,
        grid_size=grid_size,
        det_shift_max_ratio=det_shift_max_ratio,
        aug_seed=seed,
        det_debug_aug=det_debug_aug if task == "det" else False,
        det_debug_aug_dir=det_debug_aug_dir if task == "det" else None,
        det_debug_aug_num=det_debug_aug_num,
        det_debug_print=det_debug_print if task == "det" else False,
    )

    train_dataset_raw = train_dataset_raw.shuffle()

    train_dataset = train_dataset_raw.map(
        builder,
        remove_columns=train_dataset_raw.column_names
    )
    val_dataset = val_dataset_raw.map(
        builder,
        remove_columns=val_dataset_raw.column_names
    )

    collator = VLDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    swanlab.init(
        project=swanlab_project,
        experiment_name=swanlab_experiment,
        config={
            "stage": task,
            "base_model": model_name_or_path,
            "train_json": train_json,
            "val_json": val_json,
            "output_dir": output_dir,
            "max_length": max_length,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "load_in_4bit": load_in_4bit,
            "gradient_checkpointing": gradient_checkpointing,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "crop_expand_ratio": crop_expand_ratio,
            "grid_size": grid_size,
            "train_records": len(train_records),
            "effective_train_records": effective_train_size,
            "val_records": len(val_records),
            "det_balance_enable": det_balance_enable if task == "det" else False,
            "det_balance_target": det_balance_target if task == "det" else 0,
            "det_balance_target_mode": det_balance_target_mode if task == "det" else "none",
            "det_shift_max_ratio": det_shift_max_ratio,
            "seed": seed,
            "det_debug_aug": det_debug_aug if task == "det" else False,
            "det_debug_aug_num": det_debug_aug_num if task == "det" else 0,
            "det_debug_print": det_debug_print if task == "det" else False,
        },
    )

    swanlab_callback = SwanLabCallback(
        project=swanlab_project,
        experiment_name=swanlab_experiment,
        config={
            "stage": task,
            "base_model": model_name_or_path,
            "train_records": len(train_records),
            "effective_train_records": effective_train_size,
            "val_records": len(val_records),
            "grid_size": grid_size,
            "det_balance_enable": det_balance_enable if task == "det" else False,
            "det_balance_target": det_balance_target if task == "det" else 0,
            "det_balance_target_mode": det_balance_target_mode if task == "det" else "none",
            "seed": seed,
        },
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        bf16=False,
        fp16=not load_in_4bit,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[swanlab_callback],
    )

    log(f"Start training stage: {task}")
    trainer.train()

    log("Saving final adapter ...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    log("Running stage evaluation ...")
    if task == "det":
        metrics, image_logs = evaluate_detector(model, processor, val_records, grid_size=grid_size)
        print_region_metrics_table(metrics, grid_size=grid_size, prefix="det_region_")
    elif task == "cls":
        metrics, image_logs = evaluate_classifier(model, processor, val_records)
    else:
        raise ValueError(task)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    swanlab.log(metrics)
    if image_logs:
        swanlab.log({"Prediction": image_logs})
    swanlab.finish()


# =====================================
# Main
# =====================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="train_det",
        choices=[
            "train_det",
            "train_cls",
            "eval_pipeline",
        ]
    )

    parser.add_argument("--base_model", type=str, default="/home/dwd/桌面/qwen_models/Qwen3.5-0.8B")

    parser.add_argument("--train_json", type=str,
                        default="/home/dwd/桌面/Spinal-qwen-finetune/datasets/train_output/data_detcls_vl.json")
    parser.add_argument("--val_json", type=str,
                        default="/home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_detcls_vl.json")

    parser.add_argument("--det_output_dir", type=str,
                        default="/home/dwd/桌面/Spinal-qwen-finetune/output/qwen35_stage1_det_balanced")
    parser.add_argument("--cls_output_dir", type=str,
                        default="/home/dwd/桌面/Spinal-qwen-finetune/output/qwen35_stage2_cls")

    parser.add_argument("--det_adapter_path", type=str, default=None)
    parser.add_argument("--cls_adapter_path", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--eval_steps", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--crop_expand_ratio", type=float, default=0.1)
    parser.add_argument("--grid_size", type=int, default=3)
    parser.add_argument("--det_shift_max_ratio", type=float, default=0.15)

    parser.add_argument("--det_balance_enable", action="store_true", default=True)
    parser.add_argument("--det_balance_target", type=int, default=1200)
    parser.add_argument(
        "--det_balance_target_mode",
        type=str,
        default="fixed",
        choices=["fixed", "max", "mean", "median", "p75"]
    )

    parser.add_argument("--det_debug_aug", action="store_true", default=True)
    parser.add_argument("--det_debug_print", action="store_true", default=True)
    parser.add_argument(
        "--det_debug_aug_dir",
        type=str,
        default="/home/dwd/桌面/Spinal-qwen-finetune/output/det_aug_debug"
    )
    parser.add_argument("--det_debug_aug_num", type=int, default=30)

    parser.add_argument("--swanlab_project", type=str, default="Qwen3.5-VL-LoRA")
    parser.add_argument("--swanlab_experiment", type=str, default="two-stage-balanced")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == "train_det":
        os.makedirs(args.det_output_dir, exist_ok=True)
        train_stage(
            task="det",
            model_name_or_path=args.base_model,
            train_json=args.train_json,
            val_json=args.val_json,
            output_dir=args.det_output_dir,
            max_length=args.max_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=args.gradient_checkpointing,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            crop_expand_ratio=args.crop_expand_ratio,
            swanlab_project=args.swanlab_project,
            swanlab_experiment=f"{args.swanlab_experiment}-stage1-det",
            grid_size=args.grid_size,
            det_shift_max_ratio=args.det_shift_max_ratio,
            det_balance_enable=args.det_balance_enable,
            det_balance_target=args.det_balance_target,
            det_balance_target_mode=args.det_balance_target_mode,
            seed=args.seed,
            det_debug_aug=args.det_debug_aug,
            det_debug_aug_dir=args.det_debug_aug_dir,
            det_debug_aug_num=args.det_debug_aug_num,
            det_debug_print=args.det_debug_print,
        )

    elif args.mode == "train_cls":
        os.makedirs(args.cls_output_dir, exist_ok=True)
        train_stage(
            task="cls",
            model_name_or_path=args.base_model,
            train_json=args.train_json,
            val_json=args.val_json,
            output_dir=args.cls_output_dir,
            max_length=args.max_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=args.gradient_checkpointing,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            crop_expand_ratio=args.crop_expand_ratio,
            swanlab_project=args.swanlab_project,
            swanlab_experiment=f"{args.swanlab_experiment}-stage2-cls",
            grid_size=args.grid_size,
            det_shift_max_ratio=args.det_shift_max_ratio,
            det_balance_enable=False,
            det_balance_target=0,
            det_balance_target_mode="fixed",
            seed=args.seed,
            det_debug_aug=False,
            det_debug_aug_dir=args.det_debug_aug_dir,
            det_debug_aug_num=args.det_debug_aug_num,
            det_debug_print=False,
        )

    elif args.mode == "eval_pipeline":
        if args.det_adapter_path is None:
            args.det_adapter_path = args.det_output_dir
        if args.cls_adapter_path is None:
            args.cls_adapter_path = args.cls_output_dir

        swanlab.init(
            project=args.swanlab_project,
            experiment_name=f"{args.swanlab_experiment}-pipeline-eval",
            config={
                "base_model": args.base_model,
                "val_json": args.val_json,
                "det_adapter_path": args.det_adapter_path,
                "cls_adapter_path": args.cls_adapter_path,
                "grid_size": args.grid_size,
            },
        )

        log("Loading validation records ...")
        val_records = load_records(args.val_json)

        log("Loading detector base model ...")
        det_base, det_processor = load_model_and_processor(
            model_name_or_path=args.base_model,
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=False,
        )
        det_model = PeftModel.from_pretrained(det_base, args.det_adapter_path)
        det_model.eval()

        log("Loading classifier base model ...")
        cls_base, cls_processor = load_model_and_processor(
            model_name_or_path=args.base_model,
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=False,
        )
        cls_model = PeftModel.from_pretrained(cls_base, args.cls_adapter_path)
        cls_model.eval()

        metrics, image_logs = evaluate_pipeline(
            det_model, det_processor, cls_model, cls_processor, val_records, grid_size=args.grid_size
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        swanlab.log(metrics)
        if image_logs:
            swanlab.log({"Prediction": image_logs})
        swanlab.finish()


if __name__ == "__main__":
    main()