#!/usr/bin/env python
"""Create leakage-controlled patient-level folds for adaptation selection.

Each outer fold contains four mutually controlled roles:
  * outer_test: never used for training or configuration selection in that fold
  * inner_selection: shared validation patients for both candidate configurations
  * adapt20_train: 15 target-domain adaptation patients
  * adapt10_train: an 8-patient subset of adapt20_train

The seven adapt20-only patients are deliberately excluded from the shared inner
selection set so adapt10_r5 and adapt20_r3 are compared on identical patients.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


LABELS = ("infection", "tumor")


def read_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_id(record: dict) -> str:
    return str(record.get("id") or "")


def record_patient(record: dict) -> str:
    patient_id = record.get("patient_id") or (record.get("slice_manifest") or {}).get("patient_id")
    if patient_id:
        return str(patient_id)
    case_id = str(record.get("case_id") or "")
    parts = case_id.split("_")
    if len(parts) >= 3 and parts[0] in LABELS:
        return parts[1]
    raise ValueError(f"Cannot determine patient_id for record {record_id(record)!r}")


def record_label(record: dict) -> str:
    label = record.get("label") or (record.get("slice_manifest") or {}).get("label")
    if label in LABELS:
        return str(label)
    case_id = str(record.get("case_id") or "")
    if case_id.startswith("infection_"):
        return "infection"
    if case_id.startswith("tumor_"):
        return "tumor"
    raise ValueError(f"Cannot determine label for record {record_id(record)!r}")


def patient_metadata(records: Sequence[dict]) -> Dict[str, dict]:
    labels: Dict[str, Counter] = defaultdict(Counter)
    record_counts: Counter = Counter()
    seq_counts: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        patient = record_patient(record)
        labels[patient][record_label(record)] += 1
        record_counts[patient] += 1
        seq = str(record.get("seq") or (record.get("slice_manifest") or {}).get("seq") or "")
        if seq:
            seq_counts[patient][seq] += 1

    metadata = {}
    for patient, counts in labels.items():
        if len(counts) != 1:
            raise ValueError(f"Patient {patient} has inconsistent labels: {dict(counts)}")
        metadata[patient] = {
            "label": next(iter(counts)),
            "num_records": record_counts[patient],
            "seq_counts": dict(seq_counts[patient]),
        }
    return metadata


def make_stratified_outer_folds(metadata: Dict[str, dict], n_folds: int, seed: int) -> List[Set[str]]:
    folds: List[Set[str]] = [set() for _ in range(n_folds)]
    by_label: Dict[str, List[str]] = defaultdict(list)
    for patient, meta in metadata.items():
        by_label[meta["label"]].append(patient)

    for label_index, label in enumerate(LABELS):
        patients = sorted(by_label[label])
        rng = random.Random(seed + 1009 * (label_index + 1))
        rng.shuffle(patients)
        for index, patient in enumerate(patients):
            folds[index % n_folds].add(patient)
    return folds


def allocate_label_counts(available: Counter, total: int) -> Dict[str, int]:
    available_total = sum(available.values())
    if total > available_total:
        raise ValueError(f"Requested {total} patients from only {available_total}")

    raw = {label: total * available[label] / available_total for label in LABELS}
    counts = {label: min(int(raw[label]), available[label]) for label in LABELS}
    remaining = total - sum(counts.values())
    order = sorted(LABELS, key=lambda label: (raw[label] - int(raw[label]), available[label]), reverse=True)
    while remaining:
        progressed = False
        for label in order:
            if counts[label] < available[label]:
                counts[label] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Unable to allocate stratified patient counts")
    return counts


def stratified_sample(
    patients: Iterable[str],
    metadata: Dict[str, dict],
    count: int,
    seed: int,
) -> Set[str]:
    patients = set(patients)
    by_label: Dict[str, List[str]] = defaultdict(list)
    for patient in patients:
        by_label[metadata[patient]["label"]].append(patient)
    targets = allocate_label_counts(Counter({label: len(by_label[label]) for label in LABELS}), count)

    selected: Set[str] = set()
    for label_index, label in enumerate(LABELS):
        candidates = sorted(by_label[label])
        rng = random.Random(seed + 7919 * (label_index + 1))
        rng.shuffle(candidates)
        selected.update(candidates[: targets[label]])
    if len(selected) != count:
        raise RuntimeError(f"Expected {count} selected patients, got {len(selected)}")
    return selected


def subset_records(records: Sequence[dict], patients: Set[str]) -> List[dict]:
    return [record for record in records if record_patient(record) in patients]


def summarize_patients(patients: Set[str], metadata: Dict[str, dict]) -> dict:
    label_counts = Counter(metadata[patient]["label"] for patient in patients)
    return {
        "patients": len(patients),
        "patient_label_counts": dict(sorted(label_counts.items())),
        "records": sum(metadata[patient]["num_records"] for patient in patients),
    }


def write_assignment_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["fold", "patient_id", "label", "role", "in_adapt10", "in_adapt20", "num_records", "seq_counts"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create nested patient-level temporal adaptation folds.")
    parser.add_argument("--qwen_json", required=True)
    parser.add_argument("--hidden_qwen_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--adapt10_patients", type=int, default=8)
    parser.add_argument("--adapt20_patients", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    qwen_path = Path(args.qwen_json)
    hidden_path = Path(args.hidden_qwen_json)
    output_dir = Path(args.output_dir)
    protocol_path = output_dir / "nested_protocol.json"
    if protocol_path.exists() and not args.overwrite:
        raise FileExistsError(f"{protocol_path} already exists. Use --overwrite only before any model run.")
    if args.adapt10_patients >= args.adapt20_patients:
        raise ValueError("adapt10_patients must be smaller than adapt20_patients")

    records = read_json(qwen_path)
    hidden_records = read_json(hidden_path)
    record_ids = [record_id(record) for record in records]
    if any(not value for value in record_ids):
        raise ValueError("Every Qwen record must have a non-empty id")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Qwen JSON contains duplicate record ids")
    hidden_by_id = {record_id(record): record for record in hidden_records}
    if len(hidden_by_id) != len(hidden_records):
        raise ValueError("Hidden Qwen JSON contains duplicate record ids")
    missing_hidden = [record_id(record) for record in records if record_id(record) not in hidden_by_id]
    if missing_hidden:
        raise ValueError(f"Missing {len(missing_hidden)} hidden records; first={missing_hidden[0]!r}")
    extra_hidden = sorted(set(hidden_by_id) - set(record_ids))
    if extra_hidden:
        raise ValueError(f"Hidden JSON has {len(extra_hidden)} unmatched records; first={extra_hidden[0]!r}")

    metadata = patient_metadata(records)
    all_patients = set(metadata)
    if len(all_patients) < args.outer_folds * 2:
        raise ValueError("Too few patients for requested outer folds")
    outer_folds = make_stratified_outer_folds(metadata, args.outer_folds, args.seed)

    assignment_rows: List[dict] = []
    fold_summaries = []
    for fold_index, outer_test in enumerate(outer_folds):
        fold_name = f"fold_{fold_index}"
        fold_dir = output_dir / fold_name
        outer_development = all_patients - outer_test
        adapt20 = stratified_sample(
            outer_development,
            metadata,
            args.adapt20_patients,
            args.seed + 10000 * (fold_index + 1),
        )
        adapt10 = stratified_sample(
            adapt20,
            metadata,
            args.adapt10_patients,
            args.seed + 20000 * (fold_index + 1),
        )
        inner_selection = outer_development - adapt20

        if not adapt10 <= adapt20:
            raise AssertionError("adapt10 must be nested within adapt20")
        role_sets = [outer_test, adapt20, inner_selection]
        if any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError(f"Patient leakage detected in {fold_name}")
        if set().union(*role_sets) != all_patients:
            raise AssertionError(f"Patient coverage mismatch in {fold_name}")
        for subset in (outer_test, inner_selection, adapt10, adapt20):
            labels = {metadata[patient]["label"] for patient in subset}
            if labels != set(LABELS):
                raise ValueError(f"{fold_name} subset lacks a class: {labels}")

        subsets = {
            "adapt10_train": adapt10,
            "adapt20_train": adapt20,
            "inner_selection": inner_selection,
            "outer_test": outer_test,
        }
        for subset_name, patients in subsets.items():
            subset = subset_records(records, patients)
            hidden_subset = [hidden_by_id[record_id(record)] for record in subset]
            write_json(fold_dir / f"{subset_name}.json", subset)
            write_json(fold_dir / f"{subset_name}_hidden_bbox.json", hidden_subset)

        for patient in sorted(all_patients):
            if patient in outer_test:
                role = "outer_test"
            elif patient in adapt20:
                role = "adapt20_train"
            else:
                role = "inner_selection"
            assignment_rows.append({
                "fold": fold_index,
                "patient_id": patient,
                "label": metadata[patient]["label"],
                "role": role,
                "in_adapt10": int(patient in adapt10),
                "in_adapt20": int(patient in adapt20),
                "num_records": metadata[patient]["num_records"],
                "seq_counts": json.dumps(metadata[patient]["seq_counts"], sort_keys=True),
            })

        fold_summary = {
            "fold": fold_index,
            "fold_name": fold_name,
            "seed": args.seed,
            "adapt10_is_subset_of_adapt20": True,
            "sets": {name: summarize_patients(patients, metadata) for name, patients in subsets.items()},
            "patient_ids": {name: sorted(patients) for name, patients in subsets.items()},
            "selection_rule": {
                "primary": "patient_level.quality_weighted_vote.balanced_acc",
                "tie_break_1": "patient_level.quality_weighted_vote.acc",
                "tie_break_2": "prefer adapt10_r5",
            },
        }
        write_json(fold_dir / "fold_summary.json", fold_summary)
        fold_summaries.append(fold_summary)

    outer_test_union = set().union(*outer_folds)
    outer_test_total = sum(len(fold) for fold in outer_folds)
    if outer_test_union != all_patients or outer_test_total != len(all_patients):
        raise AssertionError("Each patient must appear in exactly one outer test fold")

    protocol = {
        "protocol_version": "nested_adaptation_selection_v1",
        "qwen_json": str(qwen_path),
        "hidden_qwen_json": str(hidden_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "outer_folds": args.outer_folds,
        "num_patients": len(all_patients),
        "num_records": len(records),
        "patient_label_counts": dict(sorted(Counter(meta["label"] for meta in metadata.values()).items())),
        "candidate_configurations": {
            "adapt10_r5": {"adaptation_patients": args.adapt10_patients, "external_repeat": 5},
            "adapt20_r3": {"adaptation_patients": args.adapt20_patients, "external_repeat": 3},
        },
        "selection_metric": "patient-level quality-weighted balanced accuracy on shared inner-selection patients",
        "tie_breaking": ["patient-level accuracy", "prefer adapt10_r5"],
        "outer_endpoint": "out-of-fold patient-level quality-weighted vote",
        "leakage_control": [
            "outer-test patients are excluded from adaptation training and configuration selection within their fold",
            "adapt10 and adapt20 candidates use the same inner-selection patients",
            "adapt10 adaptation patients are nested within the adapt20 adaptation pool",
            "each patient appears in exactly one outer-test fold",
        ],
        "folds": fold_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(protocol_path, protocol)
    write_assignment_csv(output_dir / "patient_fold_assignments.csv", assignment_rows)
    print(json.dumps({
        "protocol": str(protocol_path),
        "assignments": str(output_dir / "patient_fold_assignments.csv"),
        "patients": len(all_patients),
        "records": len(records),
        "outer_fold_sizes": [len(fold) for fold in outer_folds],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
