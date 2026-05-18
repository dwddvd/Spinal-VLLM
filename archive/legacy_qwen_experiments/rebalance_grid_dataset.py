import os
import re
import json
import math
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


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


def parse_sample_region(item: Dict, grid_size: int = 3) -> Optional[int]:
    convs = item.get("conversations", [])
    if len(convs) < 2:
        return None

    user_text = convs[0].get("value", "")
    assistant_text = convs[1].get("value", "")

    image_path = extract_image_path(user_text)
    bbox = extract_bbox_from_text(assistant_text) or extract_bbox_from_text(user_text)
    if image_path is None or bbox is None:
        return None
    if not os.path.exists(image_path):
        return None

    try:
        from PIL import Image
        with Image.open(image_path) as img:
            w, h = img.size
    except Exception:
        return None

    bbox = clamp_bbox(bbox, w, h)
    region_id = bbox_to_grid_id(bbox, w, h, grid_size=grid_size)
    return region_id


def summarize_distribution(counter: Counter, num_regions: int) -> Dict[str, Dict[str, float]]:
    total = sum(counter.values())
    out = {}
    for rid in range(1, num_regions + 1):
        n = counter.get(rid, 0)
        out[f"r{rid}"] = {
            "count": n,
            "ratio": (n / total) if total > 0 else 0.0,
        }
    return out


def rebalance_by_oversampling(
    data: List[Dict],
    regions: List[int],
    target_count: Optional[int] = None,
    random_seed: int = 42,
) -> List[Dict]:
    """
    过采样到 target_count。
    若 target_count=None，则默认补到最大类别数。
    """
    rng = random.Random(random_seed)

    by_region = defaultdict(list)
    for item, rid in zip(data, regions):
        by_region[rid].append(item)

    max_count = max(len(v) for v in by_region.values())
    if target_count is None:
        target_count = max_count

    balanced = []
    for rid in sorted(by_region.keys()):
        items = by_region[rid]
        cur = len(items)

        if cur == 0:
            continue

        if cur >= target_count:
            sampled = items[:target_count]
        else:
            sampled = list(items)
            extra = [rng.choice(items) for _ in range(target_count - cur)]
            sampled.extend(extra)

        balanced.extend(sampled)

    rng.shuffle(balanced)
    return balanced


def add_resampled_ids(data: List[Dict]) -> List[Dict]:
    """
    给过采样后的重复样本补唯一 id，避免重复 id。
    """
    seen = Counter()
    new_data = []

    for item in data:
        item = json.loads(json.dumps(item, ensure_ascii=False))
        base_id = item.get("id", "sample")
        seen[base_id] += 1
        if seen[base_id] > 1:
            item["id"] = f"{base_id}_aug{seen[base_id]-1}"
        new_data.append(item)

    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        default="/home/dwd/桌面/Spinal-qwen-finetune/datasets/train_output/data_detcls_vl.json",
    )
    parser.add_argument(
        "--balanced_output_json",
        type=str,
        default="/home/dwd/桌面/Spinal-qwen-finetune/datasets/train_output/data_detcls_vl_grid_balanced.json",
    )
    parser.add_argument(
        "--stats_output_json",
        type=str,
        default="/home/dwd/桌面/Spinal-qwen-finetune/datasets/train_output/data_detcls_vl_grid_stats.json",
    )
    parser.add_argument(
        "--balanced_stats_output_json",
        type=str,
        default="/home/dwd/桌面/Spinal-qwen-finetune/datasets/train_output/data_detcls_vl_grid_balanced_stats.json",
    )
    parser.add_argument("--grid_size", type=int, default=3)
    parser.add_argument(
        "--target_count",
        type=int,
        default=None,
        help="每个区域过采样到多少；默认补到最大类数量",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_items = []
    valid_regions = []
    skipped = []

    for idx, item in enumerate(data):
        rid = parse_sample_region(item, grid_size=args.grid_size)
        if rid is None:
            skipped.append(idx)
            continue
        valid_items.append(item)
        valid_regions.append(rid)

    num_regions = args.grid_size * args.grid_size
    counter = Counter(valid_regions)
    stats = {
        "total_input": len(data),
        "total_valid": len(valid_items),
        "total_skipped": len(skipped),
        "grid_size": args.grid_size,
        "distribution": summarize_distribution(counter, num_regions),
    }

    with open(args.stats_output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[INFO] 原始分布：")
    for rid in range(1, num_regions + 1):
        n = counter.get(rid, 0)
        ratio = n / max(len(valid_items), 1)
        print(f"  r{rid}: {n} ({ratio:.2%})")

    balanced = rebalance_by_oversampling(
        data=valid_items,
        regions=valid_regions,
        target_count=args.target_count,
        random_seed=args.seed,
    )
    balanced = add_resampled_ids(balanced)

    with open(args.balanced_output_json, "w", encoding="utf-8") as f:
        json.dump(balanced, f, ensure_ascii=False, indent=2)

    balanced_regions = []
    for item in balanced:
        rid = parse_sample_region(item, grid_size=args.grid_size)
        if rid is not None:
            balanced_regions.append(rid)

    balanced_counter = Counter(balanced_regions)
    balanced_stats = {
        "total_balanced": len(balanced),
        "grid_size": args.grid_size,
        "target_count": args.target_count if args.target_count is not None else max(counter.values()),
        "distribution": summarize_distribution(balanced_counter, num_regions),
    }

    with open(args.balanced_stats_output_json, "w", encoding="utf-8") as f:
        json.dump(balanced_stats, f, ensure_ascii=False, indent=2)

    print("[INFO] 平衡后分布：")
    for rid in range(1, num_regions + 1):
        n = balanced_counter.get(rid, 0)
        ratio = n / max(len(balanced), 1)
        print(f"  r{rid}: {n} ({ratio:.2%})")

    print(f"[INFO] 原始统计已保存到: {args.stats_output_json}")
    print(f"[INFO] 平衡训练集已保存到: {args.balanced_output_json}")
    print(f"[INFO] 平衡后统计已保存到: {args.balanced_stats_output_json}")


if __name__ == "__main__":
    main()