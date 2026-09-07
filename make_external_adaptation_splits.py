#!/usr/bin/env python
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def write_json(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: List[dict], extra_fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = list(rows[0].keys()) if rows else []
    fields = base_fields + [x for x in extra_fields if x not in base_fields]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key_record(record: dict) -> str:
    return str(record.get("id") or "")


def record_patient(record: dict) -> str:
    patient_id = record.get("patient_id")
    if patient_id:
        return str(patient_id)
    manifest = record.get("slice_manifest") or {}
    patient_id = manifest.get("patient_id")
    if patient_id:
        return str(patient_id)
    image_path = ""
    convs = record.get("conversations") or []
    if convs:
        image_path = convs[0].get("value", "")
    name = Path(image_path.split("<|vision_start|>")[-1].split("<|vision_end|>")[0]).name
    parts = name.split("_")
    return parts[1] if len(parts) >= 3 and parts[0] in {"infection", "tumor"} else parts[0]


def record_label(record: dict) -> str:
    label = record.get("label")
    if label:
        return str(label)
    manifest = record.get("slice_manifest") or {}
    label = manifest.get("label")
    if label:
        return str(label)
    convs = record.get("conversations") or []
    assistant = convs[1].get("value", "") if len(convs) > 1 else ""
    if "感染" in assistant or "infection" in assistant.lower():
        return "infection"
    if "肿瘤" in assistant or "tumor" in assistant.lower():
        return "tumor"
    return "unknown"


def record_seq(record: dict) -> str:
    seq = record.get("seq")
    if seq:
        return str(seq)
    manifest = record.get("slice_manifest") or {}
    seq = manifest.get("seq")
    return str(seq or "")


def choose_adapt_patients(records: List[dict], ratio: float, seed: int) -> Tuple[set, Dict[str, dict]]:
    patient_labels: Dict[str, Counter] = defaultdict(Counter)
    patient_seq: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        patient = record_patient(record)
        patient_labels[patient][record_label(record)] += 1
        patient_seq[patient][record_seq(record)] += 1

    label_to_patients: Dict[str, List[str]] = defaultdict(list)
    patient_meta: Dict[str, dict] = {}
    for patient, label_counts in patient_labels.items():
        label = label_counts.most_common(1)[0][0]
        label_to_patients[label].append(patient)
        patient_meta[patient] = {
            "label": label,
            "num_records": sum(label_counts.values()),
            "label_counts": dict(label_counts),
            "seq_counts": dict(patient_seq[patient]),
        }

    rng = random.Random(seed)
    selected = set()
    for label, patients in sorted(label_to_patients.items()):
        patients = sorted(patients)
        rng.shuffle(patients)
        n_select = int(round(len(patients) * ratio))
        if ratio > 0 and patients:
            n_select = max(1, n_select)
        n_select = min(n_select, len(patients))
        selected.update(patients[:n_select])
    return selected, patient_meta


def summarize_records(records: List[dict]) -> dict:
    return {
        "records": len(records),
        "patients": len({record_patient(x) for x in records}),
        "label_counts": dict(Counter(record_label(x) for x in records)),
        "seq_counts": dict(Counter(record_seq(x) for x in records)),
        "label_seq_counts": dict(Counter(f"{record_label(x)}|{record_seq(x)}" for x in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level temporal external adaptation splits.")
    parser.add_argument("--qwen_json", required=True)
    parser.add_argument("--hidden_qwen_json", required=True)
    parser.add_argument("--slice_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ratios", default="0.05,0.10,0.20")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    qwen_json = Path(args.qwen_json)
    hidden_qwen_json = Path(args.hidden_qwen_json)
    slice_manifest = Path(args.slice_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_json(qwen_json)
    hidden_records = read_json(hidden_qwen_json)
    manifest_rows = read_manifest(slice_manifest)

    hidden_by_id = {key_record(x): x for x in hidden_records}
    if len(hidden_by_id) != len(hidden_records):
        raise ValueError("Hidden Qwen JSON has duplicated ids; cannot align safely.")

    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    summary = {
        "qwen_json": str(qwen_json),
        "hidden_qwen_json": str(hidden_qwen_json),
        "slice_manifest": str(slice_manifest),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "source_summary": summarize_records(records),
        "splits": {},
    }

    for ratio in ratios:
        tag = f"adapt{int(round(ratio * 100)):02d}"
        adapt_patients, patient_meta = choose_adapt_patients(records, ratio, args.seed)
        test_patients = {record_patient(x) for x in records} - adapt_patients

        adapt_records = [x for x in records if record_patient(x) in adapt_patients]
        test_records = [x for x in records if record_patient(x) in test_patients]
        adapt_hidden_records = [hidden_by_id[key_record(x)] for x in adapt_records if key_record(x) in hidden_by_id]
        test_hidden_records = [hidden_by_id[key_record(x)] for x in test_records if key_record(x) in hidden_by_id]

        split_rows = []
        for row in manifest_rows:
            patient = str(row.get("patient_id", ""))
            row = dict(row)
            row["adapt_split"] = "adapt_train" if patient in adapt_patients else "adapt_test"
            row["adapt_ratio"] = tag
            split_rows.append(row)

        write_json(output_dir / f"{tag}_train.json", adapt_records)
        write_json(output_dir / f"{tag}_train_hidden_bbox.json", adapt_hidden_records)
        write_json(output_dir / f"{tag}_test.json", test_records)
        write_json(output_dir / f"{tag}_test_hidden_bbox.json", test_hidden_records)
        write_manifest(output_dir / f"{tag}_split_manifest.csv", split_rows, ["adapt_split", "adapt_ratio"])

        selected_patient_meta = {p: patient_meta[p] for p in sorted(adapt_patients)}
        summary["splits"][tag] = {
            "ratio": ratio,
            "adapt_patients": len(adapt_patients),
            "test_patients": len(test_patients),
            "adapt_patient_ids": sorted(adapt_patients),
            "adapt_patient_meta": selected_patient_meta,
            "train_json": str(output_dir / f"{tag}_train.json"),
            "train_hidden_bbox_json": str(output_dir / f"{tag}_train_hidden_bbox.json"),
            "test_json": str(output_dir / f"{tag}_test.json"),
            "test_hidden_bbox_json": str(output_dir / f"{tag}_test_hidden_bbox.json"),
            "split_manifest": str(output_dir / f"{tag}_split_manifest.csv"),
            "train_summary": summarize_records(adapt_records),
            "test_summary": summarize_records(test_records),
        }

    with (output_dir / "adaptation_split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
