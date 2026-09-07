import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


DEFAULT_INFECTION_DIR = "data/private/infection_npz"
DEFAULT_TUMOR_DIR = "data/private/tumor_npz"
DEFAULT_OUTPUT_CSV = "datasets/splits/spinal_split_v1.csv"


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


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
        "npz_name": path.name,
    }


def list_cases(infection_dir: str, tumor_dir: str) -> List[Dict[str, str]]:
    cases = []
    for path in sorted(Path(infection_dir).glob("*.npz")):
        cases.append(parse_case(path, "infection"))
    for path in sorted(Path(tumor_dir).glob("*.npz")):
        cases.append(parse_case(path, "tumor"))
    if not cases:
        raise SystemExit("No npz files found. Please check --infection_dir and --tumor_dir.")
    return cases


def build_patient_split(cases: List[Dict[str, str]], val_ratio: float, seed: int) -> Dict[str, str]:
    patient_labels = {}
    for case in cases:
        patient_id = case["patient_id"]
        label = case["label"]
        if patient_id in patient_labels and patient_labels[patient_id] != label:
            raise ValueError(f"Patient {patient_id} appears in multiple labels: {patient_labels[patient_id]} and {label}")
        patient_labels[patient_id] = label

    by_label = defaultdict(list)
    for patient_id, label in patient_labels.items():
        by_label[label].append(patient_id)

    rng = random.Random(seed)
    split = {}
    for label in sorted(by_label):
        patient_ids = sorted(by_label[label])
        rng.shuffle(patient_ids)
        val_count = max(1, int(round(len(patient_ids) * val_ratio)))
        val_ids = set(patient_ids[:val_count])
        for patient_id in patient_ids:
            split[patient_id] = "val" if patient_id in val_ids else "train"
    return split


def write_split(cases: List[Dict[str, str]], patient_split: Dict[str, str], output_csv: str) -> Dict[str, object]:
    rows = []
    for case in sorted(cases, key=lambda item: (item["label"], item["patient_id"], item["seq_id"])):
        rows.append({**case, "split": patient_split[case["patient_id"]]})

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "seq_id", "seq", "label", "case_id", "split", "npz_name"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    patient_counts = Counter()
    case_counts = Counter()
    label_split_counts = Counter()
    seen_patients = set()
    for row in rows:
        case_counts[row["split"]] += 1
        label_split_counts[f'{row["label"]}_{row["split"]}'] += 1
        patient_key = row["patient_id"]
        if patient_key not in seen_patients:
            seen_patients.add(patient_key)
            patient_counts[row["split"]] += 1

    summary = {
        "output_csv": str(output_path.resolve()),
        "total_cases": len(rows),
        "total_patients": len(seen_patients),
        "patient_counts": dict(patient_counts),
        "case_counts": dict(case_counts),
        "label_split_counts": dict(label_split_counts),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_json"] = str(summary_path.resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the fixed patient-level split_v1 CSV for spinal lesion experiments.")
    parser.add_argument("--infection_dir", default=DEFAULT_INFECTION_DIR)
    parser.add_argument("--tumor_dir", default=DEFAULT_TUMOR_DIR)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = list_cases(args.infection_dir, args.tumor_dir)
    split = build_patient_split(cases, args.val_ratio, args.seed)
    summary = write_split(cases, split, args.output_csv)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
