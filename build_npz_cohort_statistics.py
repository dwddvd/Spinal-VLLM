#!/usr/bin/env python3
"""Build manuscript-ready cohort statistics directly from finalized NPZ volumes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_seq(value: str) -> str:
    value = value.upper()
    if "T1" in value:
        return "T1"
    if "T2" in value:
        return "T2"
    return value or "unknown"


def scalar_text(value: np.ndarray | str) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    return str(value)


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_npz(npz_path: Path, min_mask_area: int) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        if "image" not in data.files or "mask" not in data.files:
            raise ValueError(f"Missing image/mask arrays: {npz_path}")
        image = data["image"]
        mask = data["mask"]
        metadata = {
            key: scalar_text(data[key])
            for key in ("label", "seq", "patient_id", "case_id")
            if key in data.files
        }

    if image.ndim != 3:
        raise ValueError(f"Expected DHW arrays, got {image.shape}")
    if mask.ndim != 3 or image.shape[0] != mask.shape[0]:
        raise ValueError(f"Depth/dimension mismatch: image={image.shape}, mask={mask.shape}")
    original_mask_shape = tuple(int(v) for v in mask.shape)
    mask_resized_to_image = image.shape[1:] != mask.shape[1:]
    if mask_resized_to_image:
        target_h, target_w = image.shape[1:]
        mask = np.stack([
            np.asarray(
                Image.fromarray(mask[z]).resize(
                    (target_w, target_h), resample=Image.Resampling.NEAREST
                ),
                dtype=mask.dtype,
            )
            for z in range(mask.shape[0])
        ])

    mask_area = np.count_nonzero(mask, axis=(1, 2)).astype(np.int64)
    raw_positive = mask_area > 0
    analysis_positive = mask_area >= min_mask_area
    positive_areas = mask_area[analysis_positive]
    return {
        "depth": int(image.shape[0]),
        "height": int(image.shape[1]),
        "width": int(image.shape[2]),
        "image_dtype": str(image.dtype),
        "mask_dtype": str(mask.dtype),
        "original_mask_shape": "x".join(str(v) for v in original_mask_shape),
        "mask_resized_to_image": int(mask_resized_to_image),
        "raw_positive_slices": int(raw_positive.sum()),
        "analysis_positive_slices": int(analysis_positive.sum()),
        "analysis_negative_slices": int((~analysis_positive).sum()),
        "small_mask_slices_1_to_threshold_minus_1": int(((mask_area > 0) & (mask_area < min_mask_area)).sum()),
        "mask_voxels": int(np.count_nonzero(mask)),
        "positive_slice_mask_area_mean": float(positive_areas.mean()) if positive_areas.size else 0.0,
        "positive_slice_mask_area_median": float(np.median(positive_areas)) if positive_areas.size else 0.0,
        "positive_slice_mask_area_min": int(positive_areas.min()) if positive_areas.size else 0,
        "positive_slice_mask_area_max": int(positive_areas.max()) if positive_areas.size else 0,
        "metadata": metadata,
        "positive_slice_areas": positive_areas.tolist(),
    }


def internal_cases(args) -> list[dict]:
    rows = read_csv(args.internal_split_csv)
    cases = []
    for row in rows:
        label = row["label"].strip().lower()
        root = args.internal_infection_dir if label == "infection" else args.internal_tumor_dir
        cases.append({
            "cohort": "internal",
            "split": row["split"].strip().lower(),
            "patient_id": row["patient_id"].strip(),
            "case_id": row["case_id"].strip(),
            "label": label,
            "seq": normalize_seq(row["seq"]),
            "npz_path": root / row["npz_name"].strip(),
        })
    return cases


def external_cases(args) -> list[dict]:
    manifest_rows = read_csv(args.external_slice_manifest)
    by_case: dict[str, dict] = {}
    for row in manifest_rows:
        case_id = row["case_id"].strip()
        by_case.setdefault(case_id, row)
    cases = []
    for case_id, row in sorted(by_case.items()):
        cases.append({
            "cohort": "temporal_external",
            "split": "temporal_validation",
            "patient_id": row["patient_id"].strip(),
            "case_id": case_id,
            "label": row["label"].strip().lower(),
            "seq": normalize_seq(row["seq"]),
            "npz_path": args.external_npz_dir / f"{case_id}.npz",
        })
    return cases


def process_cases(cases: list[dict], min_mask_area: int) -> tuple[list[dict], list[dict], list[float]]:
    rows = []
    issues = []
    all_positive_areas: list[float] = []
    for idx, case in enumerate(cases, 1):
        path = case["npz_path"]
        if not path.exists():
            issues.append({**{k: str(v) for k, v in case.items()}, "issue": "missing_npz"})
            continue
        try:
            stats = summarize_npz(path, min_mask_area)
        except Exception as exc:
            issues.append({**{k: str(v) for k, v in case.items()}, "issue": f"load_error: {exc}"})
            continue
        meta = stats.pop("metadata")
        positive_areas = stats.pop("positive_slice_areas")
        all_positive_areas.extend(positive_areas)
        if int(stats["mask_resized_to_image"]) == 1:
            issues.append({
                **{k: str(v) for k, v in case.items()},
                "issue": (
                    f'mask_resized_nearest_neighbor: original={stats["original_mask_shape"]} '
                    f'target={stats["depth"]}x{stats["height"]}x{stats["width"]}'
                ),
            })
        for key in ("patient_id", "case_id", "label", "seq"):
            if key in meta and meta[key] and str(case[key]) != meta[key]:
                issues.append({
                    **{k: str(v) for k, v in case.items()},
                    "issue": f"npz_metadata_mismatch_{key}: index={case[key]} npz={meta[key]}",
                })
        rows.append({
            **{k: str(v) for k, v in case.items()},
            **stats,
            "npz_size_bytes": path.stat().st_size,
        })
        if idx % 100 == 0 or idx == len(cases):
            print(f"[INFO] Processed {idx}/{len(cases)} NPZ volumes")
    return rows, issues, all_positive_areas


def patient_rows(volume_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    issues = []
    for row in volume_rows:
        grouped[(row["cohort"], row["split"], row["patient_id"])].append(row)
    output = []
    for (cohort, split, patient_id), rows in sorted(grouped.items()):
        labels = sorted({r["label"] for r in rows})
        if len(labels) != 1:
            issues.append({"cohort": cohort, "split": split, "patient_id": patient_id, "issue": f"multiple_labels: {labels}"})
        seq_counts = Counter(r["seq"] for r in rows)
        output.append({
            "cohort": cohort,
            "split": split,
            "patient_id": patient_id,
            "label": ";".join(labels),
            "num_volumes": len(rows),
            "num_T1_volumes": seq_counts.get("T1", 0),
            "num_T2_volumes": seq_counts.get("T2", 0),
            "has_T1": int(seq_counts.get("T1", 0) > 0),
            "has_T2": int(seq_counts.get("T2", 0) > 0),
            "has_paired_T1_T2": int(seq_counts.get("T1", 0) > 0 and seq_counts.get("T2", 0) > 0),
            "total_slices": sum(int(r["depth"]) for r in rows),
            "lesion_positive_slices": sum(int(r["analysis_positive_slices"]) for r in rows),
            "lesion_negative_slices": sum(int(r["analysis_negative_slices"]) for r in rows),
            "mask_voxels": sum(int(r["mask_voxels"]) for r in rows),
        })
    return output, issues


def count_rows(rows: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def group_summary(volume_rows: list[dict], patient_data: list[dict], positive_areas: list[float]) -> list[dict]:
    groups = [
        ("internal_train", "internal", "train"),
        ("internal_validation", "internal", "val"),
        ("temporal_validation", "temporal_external", "temporal_validation"),
    ]
    output = []
    for name, cohort, split in groups:
        vr = [r for r in volume_rows if r["cohort"] == cohort and r["split"] == split]
        pr = [r for r in patient_data if r["cohort"] == cohort and r["split"] == split]
        labels = Counter(r["label"] for r in pr)
        seqs = Counter(r["seq"] for r in vr)
        sizes = Counter(f'{r["height"]}x{r["width"]}' for r in vr)
        group_areas = []
        for r in vr:
            # Volume-level medians are retained here; exact slice-level distribution is reported globally.
            if int(r["analysis_positive_slices"]) > 0:
                group_areas.append(float(r["positive_slice_mask_area_median"]))
        output.append({
            "dataset_group": name,
            "patients": len(pr),
            "infection_patients": labels.get("infection", 0),
            "tumor_patients": labels.get("tumor", 0),
            "patients_with_paired_T1_T2": sum(int(r["has_paired_T1_T2"]) for r in pr),
            "patients_missing_T1": sum(1 - int(r["has_T1"]) for r in pr),
            "patients_missing_T2": sum(1 - int(r["has_T2"]) for r in pr),
            "volumes": len(vr),
            "T1_volumes": seqs.get("T1", 0),
            "T2_volumes": seqs.get("T2", 0),
            "all_slices": sum(int(r["depth"]) for r in vr),
            "lesion_positive_slices": sum(int(r["analysis_positive_slices"]) for r in vr),
            "lesion_negative_slices": sum(int(r["analysis_negative_slices"]) for r in vr),
            "small_mask_slices_1_to_19": sum(int(r["small_mask_slices_1_to_threshold_minus_1"]) for r in vr),
            "mask_voxels": sum(int(r["mask_voxels"]) for r in vr),
            "image_matrix_distribution": json.dumps(dict(sorted(sizes.items())), ensure_ascii=False),
            "median_of_volume_positive_slice_area_medians": float(np.median(group_areas)) if group_areas else None,
        })
    return output


def external_acquisition_summary(path: Path | None, final_case_ids: set[str]) -> dict:
    if path is None or not path.exists():
        return {"available": False}
    rows = [r for r in read_csv(path) if r.get("case_id", "").strip() in final_case_ids]
    dates = sorted(r["study_date"].strip() for r in rows if r.get("study_date", "").strip())
    return {
        "available": True,
        "records": len(rows),
        "study_date_min": dates[0] if dates else None,
        "study_date_max": dates[-1] if dates else None,
        "rows_distribution": count_rows(rows, "rows") if rows else {},
        "columns_distribution": count_rows(rows, "columns") if rows else {},
        "slice_thickness_distribution": count_rows(rows, "slice_thickness") if rows else {},
        "series_description_distribution": count_rows(rows, "series_description") if rows else {},
        "note": "These fields come from the finalized external manifest, not from the NPZ payload.",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal_split_csv", type=Path, default=Path("datasets/splits/spinal_split_v1.csv"))
    parser.add_argument("--internal_infection_dir", type=Path, default=Path("data/private/infection_npz"))
    parser.add_argument("--internal_tumor_dir", type=Path, default=Path("data/private/tumor_npz"))
    parser.add_argument("--external_npz_dir", type=Path, default=Path("datasets/temporal_external_v4_reviewed_nooverlap_quality_filtered/npz"))
    parser.add_argument("--external_slice_manifest", type=Path, default=Path("datasets/temporal_external_v4_reviewed_nooverlap_quality_filtered/slice_manifest.csv"))
    parser.add_argument("--external_final_manifest", type=Path, default=Path("datasets/splits/temporal_external_candidates_v7_reviewed_nooverlap_quality_filtered.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/cohort_npz_statistics"))
    parser.add_argument("--min_mask_area", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    indexed_cases = internal_cases(args) + external_cases(args)
    volume_rows, issues, positive_areas = process_cases(indexed_cases, args.min_mask_area)
    patients, patient_issues = patient_rows(volume_rows)
    issues.extend(patient_issues)
    summary_rows = group_summary(volume_rows, patients, positive_areas)

    internal_patients = {r["patient_id"] for r in patients if r["cohort"] == "internal"}
    external_patients = {r["patient_id"] for r in patients if r["cohort"] == "temporal_external"}
    patient_overlap = sorted(internal_patients & external_patients)
    final_external_cases = {r["case_id"] for r in volume_rows if r["cohort"] == "temporal_external"}

    summary = {
        "analysis_definition": {
            "unit_hierarchy": ["deidentified_patient", "MRI_volume_or_sequence", "2D_slice"],
            "lesion_positive_slice": f"mask area >= {args.min_mask_area} pixels",
            "mask_coordinate_domain": "pixel-level masks on sagittal T1-weighted images; T1 and T2 were jointly reviewed during annotation",
            "demographic_metadata": "not retained in the de-identified NPZ modeling dataset",
        },
        "dataset_groups": summary_rows,
        "totals": {
            "patients": len(patients),
            "volumes": len(volume_rows),
            "slices": sum(int(r["depth"]) for r in volume_rows),
            "lesion_positive_slices": sum(int(r["analysis_positive_slices"]) for r in volume_rows),
            "lesion_negative_slices": sum(int(r["analysis_negative_slices"]) for r in volume_rows),
        },
        "patient_overlap_internal_vs_temporal": {
            "count": len(patient_overlap),
            "patient_ids": patient_overlap,
        },
        "external_acquisition_manifest": external_acquisition_summary(args.external_final_manifest, final_external_cases),
        "global_positive_slice_mask_area": describe(positive_areas),
        "quality_control": {
            "indexed_cases": len(indexed_cases),
            "successfully_loaded_volumes": len(volume_rows),
            "issues": len(issues),
            "mask_resized_to_image_volumes": sum(int(r["mask_resized_to_image"]) for r in volume_rows),
        },
        "source_files": {
            "internal_split_csv": str(args.internal_split_csv.resolve()),
            "internal_infection_dir": str(args.internal_infection_dir.resolve()),
            "internal_tumor_dir": str(args.internal_tumor_dir.resolve()),
            "external_npz_dir": str(args.external_npz_dir.resolve()),
            "external_slice_manifest": str(args.external_slice_manifest.resolve()),
            "external_final_manifest": str(args.external_final_manifest.resolve()),
        },
    }

    write_csv(args.output_dir / "dataset_group_summary.csv", summary_rows)
    write_csv(args.output_dir / "patient_level_statistics.csv", patients)
    write_csv(args.output_dir / "volume_level_statistics.csv", volume_rows)
    write_csv(args.output_dir / "quality_control_issues.csv", issues)
    with (args.output_dir / "cohort_npz_statistics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "workbook_data.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "patient_columns": list(patients[0]) if patients else [],
                "patient_rows": patients,
                "volume_columns": list(volume_rows[0]) if volume_rows else [],
                "volume_rows": volume_rows,
                "issue_columns": list(issues[0]) if issues else [],
                "issue_rows": issues,
            },
            f,
            ensure_ascii=False,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
