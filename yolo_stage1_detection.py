import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_TRAIN_JSON = "/home/dwd/桌面/Spinal-qwen-finetune/datasets/train_output/data_detcls_vl.json"
DEFAULT_VAL_JSON = "/home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_detcls_vl.json"


@dataclass
class DetectionRecord:
    sample_id: str
    image_path: Path
    bbox_xyxy: Tuple[int, int, int, int]
    width: int
    height: int


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def extract_image_path(user_text: str) -> Optional[str]:
    match = re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", user_text, flags=re.S)
    return match.group(1).strip() if match else None


def extract_bbox_from_text(text: str) -> Optional[Tuple[int, int, int, int]]:
    patterns = [
        r"bbox\s*[:\uFF1A]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r'"bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return tuple(map(int, match.groups()))
    return None


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


def xyxy_to_yolo(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    return cx / width, cy / height, box_w / width, box_h / height


def yolo_to_xyxy(
    xywh: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    cx, cy, box_w, box_h = xywh
    px_w = box_w * width
    px_h = box_h * height
    x1 = int(round(cx * width - px_w / 2.0))
    y1 = int(round(cy * height - px_h / 2.0))
    x2 = int(round(cx * width + px_w / 2.0))
    y2 = int(round(cy * height + px_h / 2.0))
    return clamp_bbox((x1, y1, x2, y2), width, height)


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


def safe_stem(sample_id: str, image_path: Path, index: int) -> str:
    raw = sample_id.strip() or image_path.stem
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    raw = raw.strip("._")
    if not raw:
        raw = image_path.stem
    return f"{index:06d}_{raw}"


def load_records(json_path: Path, image_root: Optional[Path] = None) -> List[DetectionRecord]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a JSON list.")

    records: List[DetectionRecord] = []
    skipped: List[Dict[str, str]] = []

    for index, item in enumerate(data):
        convs = item.get("conversations", [])
        if len(convs) < 2:
            skipped.append({"index": str(index), "reason": "missing conversations"})
            continue
        user_text = convs[0].get("value", "")
        assistant_text = convs[1].get("value", "")
        image_text = extract_image_path(user_text)
        bbox = extract_bbox_from_text(assistant_text) or extract_bbox_from_text(user_text)
        if image_text is None or bbox is None:
            skipped.append({"index": str(index), "reason": "missing image path or bbox"})
            continue

        image_path = Path(image_text)
        if not image_path.is_absolute() and image_root is not None:
            image_path = image_root / image_path
        if not image_path.exists():
            skipped.append({"index": str(index), "reason": f"image not found: {image_path}"})
            continue
        if image_path.suffix.lower() not in IMAGE_EXTS:
            skipped.append({"index": str(index), "reason": f"unsupported image extension: {image_path}"})
            continue

        with Image.open(image_path) as img:
            width, height = img.size
        bbox = clamp_bbox(bbox, width, height)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            skipped.append({"index": str(index), "reason": f"empty bbox: {bbox}"})
            continue

        records.append(
            DetectionRecord(
                sample_id=str(item.get("id", "")),
                image_path=image_path,
                bbox_xyxy=bbox,
                width=width,
                height=height,
            )
        )

    log(f"Loaded {len(records)} records from {json_path}; skipped {len(skipped)}.")
    if skipped:
        skipped_path = json_path.with_suffix(".stage1_yolo_skipped.json")
        with skipped_path.open("w", encoding="utf-8") as f:
            json.dump(skipped, f, ensure_ascii=False, indent=2)
        log(f"Skipped details written to {skipped_path}")
    return records


def copy_or_link_image(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_split(
    records: Iterable[DetectionRecord],
    split: str,
    output_dir: Path,
    link_images: bool,
) -> int:
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for index, record in enumerate(records):
        stem = safe_stem(record.sample_id, record.image_path, index)
        image_dst = image_dir / f"{stem}{record.image_path.suffix.lower()}"
        label_dst = label_dir / f"{stem}.txt"

        copy_or_link_image(record.image_path, image_dst, link=link_images)
        x, y, w, h = xyxy_to_yolo(record.bbox_xyxy, record.width, record.height)
        label_dst.write_text(f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n", encoding="utf-8")
        count += 1
    return count


def write_dataset_yaml(output_dir: Path) -> Path:
    yaml_path = output_dir / "spinal_lesion.yaml"
    yaml_text = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: lesion\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def prepare_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root).resolve() if args.image_root else None

    train_records = load_records(Path(args.train_json), image_root=image_root)
    val_records = load_records(Path(args.val_json), image_root=image_root)

    train_count = write_split(train_records, "train", output_dir, link_images=args.link_images)
    val_count = write_split(val_records, "val", output_dir, link_images=args.link_images)
    yaml_path = write_dataset_yaml(output_dir)

    summary = {
        "train_json": str(args.train_json),
        "val_json": str(args.val_json),
        "train_images": train_count,
        "val_images": val_count,
        "class_names": ["lesion"],
        "yaml": str(yaml_path),
    }
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"YOLO dataset ready: {output_dir}")
    log(f"Dataset yaml: {yaml_path}")


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install it first with: pip install ultralytics"
        ) from exc
    return YOLO


def train_detector(args: argparse.Namespace) -> None:
    YOLO = require_ultralytics()
    model = YOLO(args.model)
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        pretrained=True,
        single_cls=True,
        amp=not args.no_amp,
    )
    log(f"Training finished: {results}")


def validate_detector(args: argparse.Namespace) -> None:
    YOLO = require_ultralytics()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        split="val",
        single_cls=True,
    )
    log(f"Validation finished: {metrics}")


def predict_detector(args: argparse.Namespace) -> None:
    YOLO = require_ultralytics()
    model = YOLO(args.weights)
    source = Path(args.source)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        save=args.save_images,
        project=args.project,
        name=args.name,
    )
    for result in results:
        image_path = Path(result.path)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            rows.append(
                {
                    "image_path": str(image_path),
                    "rank": "",
                    "conf": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                }
            )
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        order = confs.argsort()[::-1][: args.top_k]
        for rank, box_index in enumerate(order, start=1):
            x1, y1, x2, y2 = xyxy[box_index].round().astype(int).tolist()
            rows.append(
                {
                    "image_path": str(image_path),
                    "rank": rank,
                    "conf": float(confs[box_index]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "rank", "conf", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)
    log(f"Predictions written to {output_csv}")


def recall_at_iou(args: argparse.Namespace) -> None:
    YOLO = require_ultralytics()
    records = load_records(Path(args.val_json), image_root=Path(args.image_root).resolve() if args.image_root else None)
    model = YOLO(args.weights)
    thresholds = [float(x) for x in args.thresholds.split(",")]
    hits = {thr: 0 for thr in thresholds}
    total = 0

    for record in records:
        total += 1
        results = model.predict(
            source=str(record.image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )
        best_iou = 0.0
        boxes = results[0].boxes if results else None
        if boxes is not None and len(boxes) > 0:
            for pred in boxes.xyxy.detach().cpu().numpy():
                pred_box = tuple(map(int, pred.round().tolist()))
                best_iou = max(best_iou, compute_iou(pred_box, record.bbox_xyxy))
        for thr in thresholds:
            hits[thr] += int(best_iou >= thr)

    metrics = {
        f"recall@iou{thr:g}": hits[thr] / max(total, 1)
        for thr in thresholds
    }
    metrics["total"] = total
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1 YOLO lesion detector for spinal MRI.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare = subparsers.add_parser("prepare", help="Convert detcls JSON files to a YOLO dataset.")
    prepare.add_argument("--train_json", default=DEFAULT_TRAIN_JSON)
    prepare.add_argument("--val_json", default=DEFAULT_VAL_JSON)
    prepare.add_argument("--output_dir", default="datasets/yolo_spinal_lesion")
    prepare.add_argument("--image_root", default=None)
    prepare.add_argument("--link_images", action="store_true", help="Use hard links instead of copies when possible.")
    prepare.set_defaults(func=prepare_dataset)

    train = subparsers.add_parser("train", help="Train a single-class YOLO lesion detector.")
    train.add_argument("--data_yaml", default="datasets/yolo_spinal_lesion/spinal_lesion.yaml")
    train.add_argument("--model", default="yolov8m.pt")
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default=0)
    train.add_argument("--project", default="output/yolo_stage1")
    train.add_argument("--name", default="yolov8m_lesion")
    train.add_argument("--patience", type=int, default=40)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--no_amp", action="store_true", help="Disable Ultralytics AMP checks/downloads.")
    train.set_defaults(func=train_detector)

    val = subparsers.add_parser("val", help="Run YOLO validation metrics.")
    val.add_argument("--weights", required=True)
    val.add_argument("--data_yaml", default="datasets/yolo_spinal_lesion/spinal_lesion.yaml")
    val.add_argument("--imgsz", type=int, default=640)
    val.add_argument("--batch", type=int, default=16)
    val.add_argument("--device", default=0)
    val.add_argument("--conf", type=float, default=0.001)
    val.add_argument("--iou", type=float, default=0.6)
    val.set_defaults(func=validate_detector)

    predict = subparsers.add_parser("predict", help="Export top-k lesion boxes for downstream Qwen classification.")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--output_csv", default="output/yolo_stage1/predictions.csv")
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--conf", type=float, default=0.05)
    predict.add_argument("--iou", type=float, default=0.5)
    predict.add_argument("--device", default=0)
    predict.add_argument("--max_det", type=int, default=10)
    predict.add_argument("--top_k", type=int, default=3)
    predict.add_argument("--save_images", action="store_true")
    predict.add_argument("--project", default="output/yolo_stage1")
    predict.add_argument("--name", default="predict")
    predict.set_defaults(func=predict_detector)

    recall = subparsers.add_parser("recall", help="Compute high-recall IoU metrics against the detcls JSON boxes.")
    recall.add_argument("--weights", required=True)
    recall.add_argument("--val_json", default=DEFAULT_VAL_JSON)
    recall.add_argument("--image_root", default=None)
    recall.add_argument("--imgsz", type=int, default=640)
    recall.add_argument("--conf", type=float, default=0.05)
    recall.add_argument("--iou", type=float, default=0.5)
    recall.add_argument("--device", default=0)
    recall.add_argument("--max_det", type=int, default=10)
    recall.add_argument("--thresholds", default="0.3,0.5")
    recall.set_defaults(func=recall_at_iou)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
