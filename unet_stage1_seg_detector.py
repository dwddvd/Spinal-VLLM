import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_INFECTION_DIR = r"H:\Lab\Bone\dataset\infection dataset\npz"
DEFAULT_TUMOR_DIR = r"H:\Lab\Bone\dataset\tumor_fuse_mask_remove margin"


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def parse_case(path: Path, label: str) -> Dict[str, object]:
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Expected filename like patientid_1.npz or patientid_2.npz, got: {path.name}")
    patient_id, seq_id = stem.rsplit("_", 1)
    seq = "T1Sag" if seq_id == "1" else "T2Sag" if seq_id == "2" else f"seq{seq_id}"
    return {"patient_id": patient_id, "seq_id": seq_id, "seq": seq, "label": label, "npz_path": str(path)}


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "image" not in data.files or "mask" not in data.files:
        raise ValueError(f"{path} must contain image and mask arrays; got keys={data.files}")
    image = data["image"]
    mask = data["mask"]
    if image.ndim != 3 or mask.ndim != 3:
        raise ValueError(f"{path} image/mask must be 3D [D,H,W]; got image={image.shape}, mask={mask.shape}")
    if image.shape[0] != mask.shape[0]:
        raise ValueError(f"{path} image/mask depth mismatch: image={image.shape}, mask={mask.shape}")
    if image.shape[1:] != mask.shape[1:]:
        mask = resize_mask_volume(mask, image.shape[1:])
    return image, mask


def resize_mask_volume(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    resized = []
    for z in range(mask.shape[0]):
        pil_mask = Image.fromarray(mask[z])
        pil_mask = pil_mask.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        resized.append(np.asarray(pil_mask, dtype=mask.dtype))
    return np.stack(resized, axis=0)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    p1, p99 = np.percentile(image, [1, 99])
    if p99 > p1:
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1)
    else:
        image = image / max(float(image.max()), 1.0)
    return image.astype(np.float32)


def mask_to_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def compute_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def list_cases(infection_dir: str, tumor_dir: str) -> List[Dict[str, object]]:
    cases = []
    for path in sorted(Path(infection_dir).glob("*.npz")):
        cases.append(parse_case(path, "infection"))
    for path in sorted(Path(tumor_dir).glob("*.npz")):
        cases.append(parse_case(path, "tumor"))
    return cases


def stratified_patient_split(cases: List[Dict[str, object]], val_ratio: float, seed: int) -> Dict[str, str]:
    patient_labels: Dict[str, str] = {}
    for case in cases:
        patient_labels[str(case["patient_id"])] = str(case["label"])
    by_label: Dict[str, List[str]] = {"infection": [], "tumor": []}
    for patient_id, label in patient_labels.items():
        by_label[label].append(patient_id)
    rng = random.Random(seed)
    split = {}
    for label, patient_ids in by_label.items():
        rng.shuffle(patient_ids)
        val_count = max(1, int(round(len(patient_ids) * val_ratio)))
        val_set = set(patient_ids[:val_count])
        for patient_id in patient_ids:
            split[patient_id] = "val" if patient_id in val_set else "train"
    return split


def inspect(args: argparse.Namespace) -> None:
    cases = list_cases(args.infection_dir, args.tumor_dir)
    summary = {"total_npz": len(cases), "by_label_seq": {}}
    for case in cases:
        key = f"{case['label']}_{case['seq']}"
        summary["by_label_seq"][key] = summary["by_label_seq"].get(key, 0) + 1
    for case in cases[: args.num_examples]:
        image, mask = load_npz(str(case["npz_path"]))
        print(
            json.dumps(
                {
                    **case,
                    "image_shape": image.shape,
                    "image_dtype": str(image.dtype),
                    "image_min": float(image.min()),
                    "image_max": float(image.max()),
                    "mask_shape": mask.shape,
                    "mask_dtype": str(mask.dtype),
                    "mask_min": float(mask.min()),
                    "mask_max": float(mask.max()),
                    "mask_positive_voxels": int((mask > 0).sum()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prepare(args: argparse.Namespace) -> None:
    cases = list_cases(args.infection_dir, args.tumor_dir)
    split = stratified_patient_split(cases, args.val_ratio, args.seed)
    output_dir = Path(args.output_dir)
    slice_dir = output_dir / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for case in tqdm(cases, desc="Prepare UNet slices", ncols=120):
        image, mask = load_npz(str(case["npz_path"]))
        image = normalize_image(image)
        mask = mask_to_binary(mask)
        patient_id = str(case["patient_id"])
        case_split = split[patient_id]
        for z in range(image.shape[0]):
            if args.keep_positive_only and int(mask[z].sum()) == 0:
                continue
            out_name = f"{patient_id}_{case['seq_id']}_z{z:03d}.npz"
            out_path = slice_dir / case_split / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, image=image[z].astype(np.float32), mask=mask[z].astype(np.uint8))
            rel_slice_path = out_path.relative_to(output_dir).as_posix()
            x1, y1, x2, y2 = bbox_from_mask(mask[z])
            rows.append(
                {
                    "split": case_split,
                    "label": case["label"],
                    "patient_id": patient_id,
                    "seq": case["seq"],
                    "seq_id": case["seq_id"],
                    "slice_index": z,
                    "slice_path": rel_slice_path,
                    "source_npz": case["npz_path"],
                    "has_mask": int(mask[z].sum() > 0),
                    "mask_area": int(mask[z].sum()),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "label", "patient_id", "seq", "seq_id", "slice_index", "slice_path", "source_npz", "has_mask", "mask_area", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "infection_dir": args.infection_dir,
        "tumor_dir": args.tumor_dir,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "total_slices": len(rows),
        "train_slices": sum(row["split"] == "train" for row in rows),
        "val_slices": sum(row["split"] == "val" for row in rows),
        "positive_slices": sum(int(row["has_mask"]) for row in rows),
    }
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


class SliceDataset(Dataset):
    def __init__(self, manifest_path: str, split: str, image_size: int):
        self.manifest_dir = Path(manifest_path).resolve().parent
        self.image_size = image_size
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.rows = [row for row in csv.DictReader(f) if row["split"] == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        slice_path = Path(row["slice_path"])
        if not slice_path.is_absolute():
            slice_path = self.manifest_dir / slice_path
        data = np.load(slice_path)
        image = torch.from_numpy(data["image"].astype(np.float32))[None, ...]
        mask = torch.from_numpy(data["mask"].astype(np.float32))[None, ...]
        if self.image_size > 0 and image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
            mask = F.interpolate(mask[None], size=(self.image_size, self.image_size), mode="nearest")[0]
        return image, mask


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def dice_loss(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    train_set = SliceDataset(args.manifest, "train", args.image_size)
    val_set = SliceDataset(args.manifest, "val", args.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = SmallUNet(base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for image, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120):
            image = image.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = bce(logits, mask) + dice_loss(logits, mask)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * image.size(0)
        train_loss /= max(len(train_set), 1)
        val_dice = evaluate_dice(model, val_loader, device)
        log(f"epoch={epoch} train_loss={train_loss:.5f} val_dice={val_dice:.5f}")
        ckpt = {"model": model.state_dict(), "base_channels": args.base_channels, "epoch": epoch, "val_dice": val_dice}
        torch.save(ckpt, output_dir / "last.pt")
        if val_dice > best_val:
            best_val = val_dice
            torch.save(ckpt, output_dir / "best.pt")


def evaluate_dice(model, loader, device) -> float:
    model.eval()
    dices = []
    with torch.no_grad():
        for image, mask in loader:
            image = image.to(device)
            mask = mask.to(device)
            pred = (torch.sigmoid(model(image)) >= 0.5).float()
            num = 2 * (pred * mask).sum(dim=(2, 3))
            den = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
            valid = den > 0
            if valid.any():
                dices.extend(((num[valid] + 1e-6) / (den[valid] + 1e-6)).detach().cpu().tolist())
    return float(np.mean(dices)) if dices else 0.0


def load_model(weights: str, device: torch.device) -> SmallUNet:
    ckpt = torch.load(weights, map_location=device)
    model = SmallUNet(base=int(ckpt.get("base_channels", 32))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = load_model(args.weights, device)
    cases = list_cases(args.infection_dir, args.tumor_dir)
    rows = []
    hits = {0.3: 0, 0.5: 0}
    total_with_mask = 0

    for case in tqdm(cases, desc="Predict UNet bboxes", ncols=120):
        image, mask = load_npz(str(case["npz_path"]))
        image = normalize_image(image)
        mask = mask_to_binary(mask)
        best = {"score": -1.0, "slice_index": 0, "bbox": (0, 0, 0, 0)}
        for z in range(image.shape[0]):
            tensor = torch.from_numpy(image[z].astype(np.float32))[None, None].to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
            pred_mask = prob >= args.threshold
            if int(pred_mask.sum()) == 0:
                score = float(prob.max())
                bbox = (0, 0, 0, 0)
            else:
                score = float(prob[pred_mask].mean() * np.sqrt(pred_mask.sum()))
                bbox = bbox_from_mask(pred_mask.astype(np.uint8))
            if score > best["score"]:
                best = {"score": score, "slice_index": z, "bbox": bbox}
        gt_mask = mask[int(best["slice_index"])]
        gt_bbox = bbox_from_mask(gt_mask)
        iou = compute_iou(best["bbox"], gt_bbox)
        if int(mask.sum()) > 0:
            total_with_mask += 1
            for thr in hits:
                hits[thr] += int(iou >= thr)
        rows.append(
            {
                "npz_path": case["npz_path"],
                "label": case["label"],
                "patient_id": case["patient_id"],
                "seq": case["seq"],
                "rank": 1,
                "conf": best["score"],
                "slice_index": best["slice_index"],
                "iou_on_selected_slice": iou,
                "x1": best["bbox"][0],
                "y1": best["bbox"][1],
                "x2": best["bbox"][2],
                "y2": best["bbox"][3],
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["npz_path", "label", "patient_id", "seq", "rank", "conf", "slice_index", "iou_on_selected_slice", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "total_cases": len(cases),
        "total_with_mask": total_with_mask,
        "selected_slice_bbox_recall@iou0.3": hits[0.3] / max(total_with_mask, 1),
        "selected_slice_bbox_recall@iou0.5": hits[0.5] / max(total_with_mask, 1),
        "output_csv": str(output_csv),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def resize_tensor_pair(image: torch.Tensor, mask: torch.Tensor, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if image_size > 0 and image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(image[None], size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask[None], size=(image_size, image_size), mode="nearest")[0]
    return image, mask


def predict_manifest(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = load_model(args.weights, device)
    manifest_path = Path(args.manifest)
    manifest_dir = manifest_path.resolve().parent
    with manifest_path.open("r", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == args.split]

    output_rows = []
    dice_scores = []
    hits = {0.3: 0, 0.5: 0}
    total_positive = 0

    for row in tqdm(rows, desc=f"Predict UNet {args.split} slices", ncols=120):
        slice_path = Path(row["slice_path"])
        if not slice_path.is_absolute():
            slice_path = manifest_dir / slice_path
        data = np.load(slice_path)
        image = torch.from_numpy(data["image"].astype(np.float32))[None, ...]
        mask = torch.from_numpy(data["mask"].astype(np.float32))[None, ...]
        image, mask = resize_tensor_pair(image, mask, args.image_size)
        with torch.no_grad():
            prob = torch.sigmoid(model(image[None].to(device)))[0, 0].detach().cpu().numpy()
        pred_mask = (prob >= args.threshold).astype(np.uint8)
        gt_mask = mask[0].numpy().astype(np.uint8)
        pred_bbox = bbox_from_mask(pred_mask)
        gt_bbox = bbox_from_mask(gt_mask)
        iou = compute_iou(pred_bbox, gt_bbox)
        pred_area = int(pred_mask.sum())
        gt_area = int(gt_mask.sum())
        if gt_area > 0:
            total_positive += 1
            for thr in hits:
                hits[thr] += int(iou >= thr)
            den = pred_area + gt_area
            dice_scores.append((2.0 * int((pred_mask & gt_mask).sum()) + 1e-6) / (den + 1e-6))
        output_rows.append(
            {
                **row,
                "pred_area": pred_area,
                "pred_x1": pred_bbox[0],
                "pred_y1": pred_bbox[1],
                "pred_x2": pred_bbox[2],
                "pred_y2": pred_bbox[3],
                "bbox_iou": iou,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f"unet_{args.split}_slice_predictions.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(output_rows[0].keys()) if output_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    metrics = {
        "split": args.split,
        "total_slices": len(rows),
        "positive_slices": total_positive,
        "threshold": args.threshold,
        "image_size": args.image_size,
        "mean_dice_positive_slices": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "slice_bbox_recall@iou0.3": hits[0.3] / max(total_positive, 1),
        "slice_bbox_recall@iou0.5": hits[0.5] / max(total_positive, 1),
        "predictions_csv": str(rows_path),
    }
    metrics_path = output_dir / f"unet_{args.split}_manifest_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    log(f"Saved predictions to {rows_path}")
    log(f"Saved metrics to {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2D UNet stage-1 lesion segmentation detector from 3D npz image/mask volumes.")
    sub = parser.add_subparsers(dest="mode", required=True)

    inspect_p = sub.add_parser("inspect")
    inspect_p.add_argument("--infection_dir", default=DEFAULT_INFECTION_DIR)
    inspect_p.add_argument("--tumor_dir", default=DEFAULT_TUMOR_DIR)
    inspect_p.add_argument("--num_examples", type=int, default=4)
    inspect_p.set_defaults(func=inspect)

    prepare_p = sub.add_parser("prepare")
    prepare_p.add_argument("--infection_dir", default=DEFAULT_INFECTION_DIR)
    prepare_p.add_argument("--tumor_dir", default=DEFAULT_TUMOR_DIR)
    prepare_p.add_argument("--output_dir", default="datasets/unet_stage1_seg")
    prepare_p.add_argument("--val_ratio", type=float, default=0.2)
    prepare_p.add_argument("--seed", type=int, default=42)
    prepare_p.add_argument("--keep_positive_only", action="store_true")
    prepare_p.set_defaults(func=prepare)

    train_p = sub.add_parser("train")
    train_p.add_argument("--manifest", default="datasets/unet_stage1_seg/manifest.csv")
    train_p.add_argument("--output_dir", default="output/unet_stage1_seg")
    train_p.add_argument("--epochs", type=int, default=80)
    train_p.add_argument("--batch_size", type=int, default=8)
    train_p.add_argument("--image_size", type=int, default=512)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight_decay", type=float, default=1e-4)
    train_p.add_argument("--base_channels", type=int, default=32)
    train_p.add_argument("--workers", type=int, default=2)
    train_p.add_argument("--device", default="cuda:0")
    train_p.set_defaults(func=train)

    predict_p = sub.add_parser("predict")
    predict_p.add_argument("--weights", required=True)
    predict_p.add_argument("--infection_dir", default=DEFAULT_INFECTION_DIR)
    predict_p.add_argument("--tumor_dir", default=DEFAULT_TUMOR_DIR)
    predict_p.add_argument("--output_csv", default="output/unet_stage1_seg/predictions.csv")
    predict_p.add_argument("--threshold", type=float, default=0.5)
    predict_p.add_argument("--device", default="cuda:0")
    predict_p.set_defaults(func=predict)

    predict_manifest_p = sub.add_parser("predict_manifest")
    predict_manifest_p.add_argument("--weights", required=True)
    predict_manifest_p.add_argument("--manifest", default="datasets/unet_stage1_seg/manifest.csv")
    predict_manifest_p.add_argument("--output_dir", default="output/unet_stage1_seg_manifest_eval")
    predict_manifest_p.add_argument("--split", choices=["train", "val"], default="val")
    predict_manifest_p.add_argument("--threshold", type=float, default=0.5)
    predict_manifest_p.add_argument("--image_size", type=int, default=512)
    predict_manifest_p.add_argument("--device", default="cuda:0")
    predict_manifest_p.set_defaults(func=predict_manifest)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
