import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_INFECTION_DIR = r"H:\Lab\Bone\dataset\infection dataset\npz"
DEFAULT_TUMOR_DIR = r"H:\Lab\Bone\dataset\tumor_fuse_mask_remove_margin"
DEFAULT_NNUNET_RAW = "nnUNet_raw"
INFECTION_ZH = "\u611f\u67d3"
TUMOR_ZH = "\u80bf\u7624"


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit("nibabel is required. Install it with: pip install nibabel") from exc
    return nib


def parse_case(path: Path, label: str) -> Dict[str, str]:
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Expected filename like patientid_1.npz or patientid_2.npz, got: {path.name}")
    patient_id, seq_id = stem.rsplit("_", 1)
    seq = "T1Sag" if seq_id == "1" else "T2Sag" if seq_id == "2" else f"seq{seq_id}"
    return {
        "patient_id": patient_id,
        "seq_id": seq_id,
        "seq": seq,
        "label": label,
        "case_id": f"{label}_{patient_id}_{seq_id}",
        "npz_path": str(path),
    }


def list_cases(infection_dir: str, tumor_dir: str) -> List[Dict[str, str]]:
    cases = []
    for path in sorted(Path(infection_dir).glob("*.npz")):
        cases.append(parse_case(path, "infection"))
    for path in sorted(Path(tumor_dir).glob("*.npz")):
        cases.append(parse_case(path, "tumor"))
    return cases


def extract_patient_id(text: str) -> Optional[str]:
    matches = re.findall(r"\d{6,}", str(text))
    return matches[-1] if matches else None


def extract_image_path(text: str) -> Optional[str]:
    match = re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", text, flags=re.S)
    return match.group(1).strip() if match else None


def infer_sequence(text: str) -> Optional[str]:
    match = re.search(r"\b([Tt][12](?:WI)?)\b", text)
    return match.group(1).upper() if match else None


def normalize_label(text: str) -> Optional[str]:
    low = str(text).lower()
    if "infection" in low or INFECTION_ZH in str(text):
        return "infection"
    if "tumor" in low or "tumour" in low or TUMOR_ZH in str(text):
        return "tumor"
    return None


def normalize_seq(seq: Optional[str]) -> str:
    if not seq:
        return ""
    seq = seq.upper()
    if "T1" in seq or seq.endswith("_1") or seq == "1":
        return "T1"
    if "T2" in seq or seq.endswith("_2") or seq == "2":
        return "T2"
    return seq


def seq_from_path(path: str) -> str:
    text = str(path).upper()
    name = Path(path).stem.upper()
    if re.search(r"(?:^|[_-])1(?:[_-]|$)", name) or "T1" in text:
        return "T1"
    if re.search(r"(?:^|[_-])2(?:[_-]|$)", name) or "T2" in text:
        return "T2"
    return ""


def case_seq(seq_id: str) -> str:
    return "T1" if seq_id == "1" else "T2" if seq_id == "2" else normalize_seq(seq_id)


def load_qwen_patient_seq_set(json_path: str) -> set:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = set()
    skipped = 0
    for item in data:
        convs = item.get("conversations", [])
        if not convs:
            skipped += 1
            continue
        user_text = convs[0].get("value", "")
        assistant_text = convs[1].get("value", "") if len(convs) > 1 else ""
        image_path = extract_image_path(user_text) or ""
        label = normalize_label(assistant_text)
        patient_id = extract_patient_id(image_path) or extract_patient_id(str(item.get("id", "")))
        seq = normalize_seq(infer_sequence(user_text)) or seq_from_path(image_path)
        if patient_id is None or label is None:
            skipped += 1
            continue
        result.add((patient_id, seq))
    log(f"Loaded {len(result)} patient/sequence keys from {json_path}; skipped {skipped}.")
    return result


def qwen_json_split(cases: List[Dict[str, str]], train_json: str, val_json: str, missing_policy: str) -> Dict[str, str]:
    train_keys = load_qwen_patient_seq_set(train_json)
    val_keys = load_qwen_patient_seq_set(val_json)
    split = {}
    stats = {"train": 0, "val": 0, "missing": 0, "conflict": 0}
    for case in cases:
        key = (case["patient_id"], case_seq(case["seq_id"]))
        in_train = key in train_keys
        in_val = key in val_keys
        if in_val and not in_train:
            split[case["case_id"]] = "val"
            stats["val"] += 1
        elif in_train and not in_val:
            split[case["case_id"]] = "train"
            stats["train"] += 1
        elif in_train and in_val:
            split[case["case_id"]] = "val"
            stats["conflict"] += 1
        else:
            stats["missing"] += 1
            if missing_policy != "skip":
                split[case["case_id"]] = missing_policy
    log(f"Qwen split assignment stats: {stats}")
    return split


def resize_mask_volume(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    resized = []
    for z in range(mask.shape[0]):
        pil_mask = Image.fromarray(mask[z])
        pil_mask = pil_mask.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        resized.append(np.asarray(pil_mask, dtype=mask.dtype))
    return np.stack(resized, axis=0)


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
    return image.astype(np.float32), (mask > 0).astype(np.uint8)


def to_nnunet_xyz(volume_dhw: np.ndarray) -> np.ndarray:
    return np.transpose(volume_dhw, (2, 1, 0))


def stratified_patient_split(cases: List[Dict[str, str]], val_ratio: float, seed: int) -> Dict[str, str]:
    patient_labels = {case["patient_id"]: case["label"] for case in cases}
    by_label: Dict[str, List[str]] = {"infection": [], "tumor": []}
    for patient_id, label in patient_labels.items():
        by_label[label].append(patient_id)
    rng = random.Random(seed)
    split = {}
    for _, patient_ids in by_label.items():
        rng.shuffle(patient_ids)
        val_count = max(1, int(round(len(patient_ids) * val_ratio)))
        val_ids = set(patient_ids[:val_count])
        for patient_id in patient_ids:
            split[patient_id] = "val" if patient_id in val_ids else "train"
    return split


def write_dataset_json(output_dir: Path, dataset_id: int, dataset_name: str, train_count: int, test_count: int) -> None:
    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": train_count,
        "numTest": test_count,
        "file_ending": ".nii.gz",
        "dataset_name": dataset_name,
        "overwrite_image_reader_writer": "NibabelIO",
        "description": "Spinal lesion segmentation from npz image/mask volumes; each patient-sequence is a single-channel case.",
        "reference": "",
        "licence": "",
        "release": "0.1",
        "dataset_id": dataset_id,
    }
    with (output_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)


def prepare(args: argparse.Namespace) -> None:
    nib = require_nibabel()
    dataset_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    output_dir = Path(args.nnunet_raw).resolve() / dataset_name
    images_tr = output_dir / "imagesTr"
    labels_tr = output_dir / "labelsTr"
    images_ts = output_dir / "imagesTs"
    labels_ts = output_dir / "labelsTs"
    for directory in (images_tr, labels_tr, images_ts, labels_ts):
        directory.mkdir(parents=True, exist_ok=True)

    cases = list_cases(args.infection_dir, args.tumor_dir)
    if args.qwen_train_json and args.qwen_val_json:
        split = qwen_json_split(cases, args.qwen_train_json, args.qwen_val_json, args.missing_policy)
    else:
        patient_split = stratified_patient_split(cases, args.val_ratio, args.seed)
        split = {case["case_id"]: patient_split[case["patient_id"]] for case in cases}
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    manifest = []
    train_count = 0
    val_count = 0

    for case in tqdm(cases, desc="Convert npz to nnU-Net", ncols=120):
        if case["case_id"] not in split:
            continue
        image, mask = load_npz(case["npz_path"])
        case_id = case["case_id"]
        case_split = split[case_id]
        image_nifti = nib.Nifti1Image(to_nnunet_xyz(image), affine)
        mask_nifti = nib.Nifti1Image(to_nnunet_xyz(mask).astype(np.uint8), affine)
        if case_split == "train":
            nib.save(image_nifti, images_tr / f"{case_id}_0000.nii.gz")
            nib.save(mask_nifti, labels_tr / f"{case_id}.nii.gz")
            train_count += 1
        else:
            nib.save(image_nifti, images_ts / f"{case_id}_0000.nii.gz")
            nib.save(mask_nifti, labels_ts / f"{case_id}.nii.gz")
            val_count += 1
        manifest.append({**case, "split": case_split, "case_id": case_id})

    write_dataset_json(output_dir, args.dataset_id, args.dataset_name, train_count, val_count)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    summary = {
        "dataset": dataset_name,
        "output_dir": str(output_dir),
        "train_cases": train_count,
        "val_cases_as_imagesTs": val_count,
        "manifest": str(manifest_path),
        "note": "labelsTs is written for local evaluation/export, but nnU-Net training uses imagesTr/labelsTr.",
    }
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert spinal lesion npz image/mask volumes to nnU-Net v2 raw dataset format.")
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_p = sub.add_parser("prepare")
    prepare_p.add_argument("--infection_dir", default=DEFAULT_INFECTION_DIR)
    prepare_p.add_argument("--tumor_dir", default=DEFAULT_TUMOR_DIR)
    prepare_p.add_argument("--nnunet_raw", default=DEFAULT_NNUNET_RAW, help="Path to nnUNet_raw directory. Defaults to a project-local relative directory.")
    prepare_p.add_argument("--dataset_id", type=int, default=501)
    prepare_p.add_argument("--dataset_name", default="SpinalLesionSeq")
    prepare_p.add_argument("--val_ratio", type=float, default=0.2)
    prepare_p.add_argument("--seed", type=int, default=42)
    prepare_p.add_argument("--qwen_train_json", default=None)
    prepare_p.add_argument("--qwen_val_json", default=None)
    prepare_p.add_argument("--missing_policy", choices=["train", "val", "skip"], default="train")
    prepare_p.set_defaults(func=prepare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
