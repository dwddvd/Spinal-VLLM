import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - fallback for environments without scipy
    ndimage = None


DEFAULT_INFECTION_DIR = "data/private/infection_npz"
DEFAULT_TUMOR_DIR = "data/private/tumor_npz"
DEFAULT_SPLIT_CSV = "datasets/splits/spinal_split_v1.csv"
DEFAULT_OUTPUT_DIR = "datasets/qwen_split_v1_rebuilt"
DEFAULT_JSON_IMAGE_PREFIX = "data/processed/qwen_internal"
DEFAULT_MIN_MASK_AREA = 20

INFECTION_ZH = "感染"
TUMOR_ZH = "肿瘤"


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def read_csv_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resize_mask_volume(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    resized = []
    for z in range(mask.shape[0]):
        pil_mask = Image.fromarray(mask[z])
        pil_mask = pil_mask.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        resized.append(np.asarray(pil_mask, dtype=mask.dtype))
    return np.stack(resized, axis=0)


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "image" not in data.files or "mask" not in data.files:
        raise ValueError(f"{path} must contain image and mask arrays; got keys={list(data.files)}")
    image = data["image"]
    mask = data["mask"]
    if image.ndim != 3 or mask.ndim != 3:
        raise ValueError(f"{path} image/mask must be 3D [D,H,W]; got image={image.shape}, mask={mask.shape}")
    if image.shape[0] != mask.shape[0]:
        raise ValueError(f"{path} image/mask depth mismatch: image={image.shape}, mask={mask.shape}")
    if image.shape[1:] != mask.shape[1:]:
        mask = resize_mask_volume(mask, image.shape[1:])
    return image.astype(np.float32), (mask > 0).astype(np.uint8)


def normalize_slice_to_uint8(image_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_2d)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    vals = arr[finite]
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo:
        out = np.zeros(arr.shape, dtype=np.uint8)
        out[finite] = 0
        return out
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    scaled[~finite] = 0
    return scaled


def bbox2d(mask_2d: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    # Keep the original legacy prompt convention:
    # x2/y2 are inclusive max coordinates instead of exclusive bounds.
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def largest_component_bbox(mask_2d: np.ndarray) -> Tuple[int, int, int, int]:
    if int(mask_2d.sum()) <= 0:
        return 0, 0, 0, 0
    if ndimage is None:
        return bbox2d(mask_2d)
    labeled, num = ndimage.label(mask_2d > 0)
    if num <= 1:
        return bbox2d(mask_2d)
    sizes = ndimage.sum(mask_2d > 0, labeled, range(1, num + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return bbox2d(labeled == largest_label)


def json_image_path(prefix: str, split_name: str, filename: str) -> str:
    prefix = prefix.rstrip("/\\")
    return f"{prefix}/{split_name}/image/{filename}"


def prompt_for(seq_id: str, bbox: Tuple[int, int, int, int]) -> str:
    seq_name = "T1" if seq_id == "1" else "T2" if seq_id == "2" else f"seq{seq_id}"
    x1, y1, x2, y2 = bbox
    return (
        f"现在你是一个骨科专家，这是一幅脊椎的磁共振图像,序列为{seq_name}。"
        f"该图像中可能包含了数个病灶，然后我会将最大病灶的坐标位置按照[x1,y1,x2,y2]的格式给出："
        f"该张图片中的最大病灶在[{x1},{y1},{x2},{y2}]这个位置。"
        f"请你帮我判断这个病灶属于感染还是肿瘤。"
    )


def answer_for(label: str) -> str:
    return f"这个病灶的类型为{INFECTION_ZH if label == 'infection' else TUMOR_ZH}。"


def hide_first_bbox(text: str, replacement: str) -> str:
    return re.sub(
        r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]",
        replacement,
        text,
        count=1,
    )


def build_records_with_hidden(records: List[dict], replacement: str = "[x1,y1,x2,y2]") -> List[dict]:
    hidden = []
    for record in records:
        copied = json.loads(json.dumps(record, ensure_ascii=False))
        convs = copied.get("conversations", [])
        if convs:
            convs[0]["value"] = hide_first_bbox(convs[0]["value"], replacement)
        hidden.append(copied)
    return hidden


def write_json(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_dataset(args: argparse.Namespace) -> None:
    split_rows = read_csv_rows(args.split_csv)
    required = {"patient_id", "seq_id", "label", "case_id", "split", "npz_name"}
    missing = required - set(split_rows[0].keys() if split_rows else [])
    if missing:
        raise ValueError(f"{args.split_csv} missing columns: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    train_img_dir = output_dir / "train" / "image"
    val_img_dir = output_dir / "val" / "image"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    train_records: List[dict] = []
    val_records: List[dict] = []
    manifest_rows: List[dict] = []
    counts = Counter()

    for row in split_rows:
        label = row["label"].strip()
        split_name = row["split"].strip().lower()
        patient_id = row["patient_id"].strip()
        seq_id = row["seq_id"].strip()
        npz_name = row["npz_name"].strip()
        npz_dir = Path(args.infection_dir if label == "infection" else args.tumor_dir)
        npz_path = npz_dir / npz_name
        image_vol, mask_vol = load_npz(npz_path)

        label_idx = 0 if label == "infection" else 1
        out_img_dir = train_img_dir if split_name == "train" else val_img_dir

        for slice_idx in range(image_vol.shape[0]):
            image_2d = normalize_slice_to_uint8(image_vol[slice_idx])
            mask_2d = mask_vol[slice_idx]
            mask_area = int((mask_2d > 0).sum())
            has_lesion = int(mask_area >= args.min_mask_area)
            x1, y1, x2, y2 = largest_component_bbox(mask_2d)

            filename = f"{patient_id}_{seq_id}_cls_{label_idx}_layer_{slice_idx}.jpg"
            save_path = out_img_dir / filename
            Image.fromarray(image_2d).save(save_path, quality=95)

            image_path_for_json = json_image_path(args.json_image_prefix, split_name, filename)
            manifest_rows.append(
                {
                    "patient_id": patient_id,
                    "seq_id": seq_id,
                    "label": label,
                    "case_id": row["case_id"].strip(),
                    "split": split_name,
                    "npz_name": npz_name,
                    "npz_path": str(npz_path),
                    "slice_idx": slice_idx,
                    "image_path": image_path_for_json,
                    "local_image_path": str(save_path.resolve()),
                    "mask_area": mask_area,
                    "has_lesion": has_lesion,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "image_shape": f"{image_2d.shape[0]}x{image_2d.shape[1]}",
                }
            )

            counts[f"{split_name}_all_slices"] += 1
            counts[f"{split_name}_lesion_slices"] += has_lesion
            counts[f"{label}_{split_name}_all_slices"] += 1
            counts[f"{label}_{split_name}_lesion_slices"] += has_lesion

            if not has_lesion:
                continue

            record = {
                "id": "",
                "conversations": [
                    {
                        "from": "user",
                        "value": f"{prompt_for(seq_id, (x1, y1, x2, y2))} <|vision_start|>{image_path_for_json}<|vision_end|>",
                    },
                    {
                        "from": "assistant",
                        "value": answer_for(label),
                    },
                ],
                "slice_manifest": {
                    "patient_id": patient_id,
                    "seq_id": seq_id,
                    "slice_idx": slice_idx,
                    "mask_area": mask_area,
                    "case_id": row["case_id"].strip(),
                },
            }
            if split_name == "train":
                train_records.append(record)
            else:
                val_records.append(record)

    for idx, record in enumerate(train_records, start=1):
        record["id"] = f"split_v1_train_{idx:06d}"
    for idx, record in enumerate(val_records, start=1):
        record["id"] = f"split_v1_val_{idx:06d}"

    hidden_train = build_records_with_hidden(train_records)
    hidden_val = build_records_with_hidden(val_records)

    manifest_path = output_dir / "slice_manifest.csv"
    fieldnames = [
        "patient_id",
        "seq_id",
        "label",
        "case_id",
        "split",
        "npz_name",
        "npz_path",
        "slice_idx",
        "image_path",
        "local_image_path",
        "mask_area",
        "has_lesion",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "image_shape",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    train_json = output_dir / "data_vl_train_split_v1.json"
    val_json = output_dir / "data_vl_val_split_v1.json"
    hidden_train_json = output_dir / "data_vl_train_split_v1_hidden_bbox.json"
    hidden_val_json = output_dir / "data_vl_val_split_v1_hidden_bbox.json"
    write_json(train_json, train_records)
    write_json(val_json, val_records)
    write_json(hidden_train_json, hidden_train)
    write_json(hidden_val_json, hidden_val)

    summary = {
        "split_csv": str(Path(args.split_csv).resolve()),
        "infection_dir": str(Path(args.infection_dir).resolve()),
        "tumor_dir": str(Path(args.tumor_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "json_image_prefix": args.json_image_prefix,
        "prompt_template": "legacy_original",
        "bbox_coordinate_mode": "inclusive_xyxy",
        "manifest_csv": str(manifest_path.resolve()),
        "train_json": str(train_json.resolve()),
        "val_json": str(val_json.resolve()),
        "hidden_train_json": str(hidden_train_json.resolve()),
        "hidden_val_json": str(hidden_val_json.resolve()),
        "min_mask_area": args.min_mask_area,
        "counts": dict(counts),
        "train_records": len(train_records),
        "val_records": len(val_records),
    }
    summary_path = output_dir / "export_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export authoritative 2D Qwen images and JSON from split_v1 + original 3D npz volumes."
    )
    parser.add_argument("--infection_dir", default=DEFAULT_INFECTION_DIR)
    parser.add_argument("--tumor_dir", default=DEFAULT_TUMOR_DIR)
    parser.add_argument("--split_csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json_image_prefix", default=DEFAULT_JSON_IMAGE_PREFIX)
    parser.add_argument("--min_mask_area", type=int, default=DEFAULT_MIN_MASK_AREA)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    export_dataset(args)


if __name__ == "__main__":
    main()
