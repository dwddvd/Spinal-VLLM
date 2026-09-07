#!/usr/bin/env python3
"""Create a de-identified release copy of the temporal external dataset metadata.

The source dataset is never modified. The script creates:
1. public patient/case/series identifiers;
2. whitelisted public CSV manifests;
3. sanitized Qwen JSON files;
4. an optional renamed copy of JPEG, mask, and NPZ assets;
5. a private identifier mapping that must not be published.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


VISION_RE = re.compile(r"(<\|vision_start\|>)(.*?)(<\|vision_end\|>)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def year_only(value: str) -> str:
    match = re.match(r"^\s*(\d{4})", value or "")
    return match.group(1) if match else ""


def seq_suffix(seq: str) -> str:
    normalized = (seq or "").upper().replace(" ", "")
    if normalized.startswith("T1"):
        return "T1"
    if normalized.startswith("T2"):
        return "T2"
    return re.sub(r"[^A-Z0-9]+", "", normalized) or "SEQ"


def safe_asset_name(public_case_id: str, original_name: str) -> str:
    match = re.search(r"(?:_cls_\d+)?_layer_(\d+)", original_name)
    if match:
        return f"{public_case_id}_layer_{int(match.group(1)):04d}{Path(original_name).suffix.lower()}"
    return f"{public_case_id}{Path(original_name).suffix.lower()}"


def build_identifier_maps(
    final_rows: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    patient_ids = sorted({row["patient_id"].strip() for row in final_rows if row.get("patient_id", "").strip()})
    patient_map = {patient_id: f"EXT{index:04d}" for index, patient_id in enumerate(patient_ids, 1)}

    case_map: dict[str, str] = {}
    series_map: dict[str, str] = {}
    for index, row in enumerate(
        sorted(final_rows, key=lambda r: (r.get("patient_id", ""), r.get("seq", ""), r.get("case_id", ""))),
        1,
    ):
        original_case = row.get("case_id", "").strip()
        original_patient = row.get("patient_id", "").strip()
        public_patient = patient_map[original_patient]
        public_case = f"{public_patient}_{seq_suffix(row.get('seq', ''))}"
        if original_case:
            case_map[original_case] = public_case
        original_uid = row.get("series_uid", "").strip()
        if original_uid:
            series_map[original_uid] = f"EXTSER{index:04d}"
    return patient_map, case_map, series_map


def sanitize_final_manifest(
    rows: list[dict[str, str]],
    patient_map: dict[str, str],
    case_map: dict[str, str],
    series_map: dict[str, str],
) -> list[dict[str, Any]]:
    public_rows = []
    for row in rows:
        patient_id = row.get("patient_id", "").strip()
        case_id = row.get("case_id", "").strip()
        series_uid = row.get("series_uid", "").strip()
        public_rows.append(
            {
                "public_patient_id": patient_map.get(patient_id, ""),
                "public_case_id": case_map.get(case_id, ""),
                "public_series_id": series_map.get(series_uid, ""),
                "label": row.get("label", ""),
                "seq": row.get("seq", ""),
                "study_year": year_only(row.get("study_date", "")),
                "series_description": row.get("series_description", ""),
                "protocol_name": row.get("protocol_name", ""),
                "num_dicom_files": row.get("num_dicom_files", ""),
                "rows": row.get("rows", ""),
                "columns": row.get("columns", ""),
                "slice_thickness": row.get("slice_thickness", ""),
                "pixel_spacing": row.get("pixel_spacing", ""),
                "label_source_category": row.get("label_source", ""),
                "recommended_use": row.get("recommended_use", ""),
                "exclusion_category": (
                    row.get("manual_exclude_reason", "")
                    or row.get("final_overlap_reason", "")
                    or row.get("review_note", "")
                ),
            }
        )
    return public_rows


def resolve_asset(source_dir: Path, subdir: str, raw_path: str) -> Path:
    candidate = source_dir / subdir / Path(raw_path).name
    if candidate.exists():
        return candidate
    path = Path(raw_path)
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not resolve asset: {raw_path}")


def sanitize_slice_manifest(
    rows: list[dict[str, str]],
    patient_map: dict[str, str],
    case_map: dict[str, str],
    series_map: dict[str, str],
    source_dir: Path,
    output_dir: Path,
    public_asset_prefix: str,
    copy_assets: bool,
) -> tuple[list[dict[str, Any]], dict[str, str], Counter]:
    public_rows: list[dict[str, Any]] = []
    image_path_map: dict[str, str] = {}
    copy_counts: Counter = Counter()
    copied_npz: set[str] = set()

    for row in rows:
        original_patient = row.get("patient_id", "").strip()
        original_case = row.get("case_id", "").strip()
        public_patient = patient_map.get(original_patient, "")
        public_case = case_map.get(original_case, "")
        if not public_case:
            continue

        original_image = row.get("image_path", "") or row.get("local_image_path", "")
        original_mask = row.get("mask_path", "")
        original_npz = row.get("npz_path", "")
        public_image_name = safe_asset_name(public_case, Path(original_image).name)
        public_mask_name = safe_asset_name(public_case, Path(original_mask).name)
        public_npz_name = f"{public_case}.npz"

        relative_image = f"assets/image/{public_image_name}"
        relative_mask = f"assets/mask/{public_mask_name}"
        relative_npz = f"assets/npz/{public_npz_name}"
        public_image_path = f"{public_asset_prefix.rstrip('/')}/{relative_image}" if public_asset_prefix else relative_image
        public_mask_path = f"{public_asset_prefix.rstrip('/')}/{relative_mask}" if public_asset_prefix else relative_mask
        public_npz_path = f"{public_asset_prefix.rstrip('/')}/{relative_npz}" if public_asset_prefix else relative_npz

        for raw_image_path in {row.get("image_path", ""), row.get("local_image_path", "")}:
            if raw_image_path:
                image_path_map[raw_image_path] = public_image_path
                image_path_map[Path(raw_image_path).name] = public_image_path

        if copy_assets:
            image_source = resolve_asset(source_dir, "image", original_image)
            mask_source = resolve_asset(source_dir, "mask", original_mask)
            image_target = output_dir / relative_image
            mask_target = output_dir / relative_mask
            image_target.parent.mkdir(parents=True, exist_ok=True)
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_source, image_target)
            shutil.copy2(mask_source, mask_target)
            copy_counts["image"] += 1
            copy_counts["mask"] += 1

            if public_case not in copied_npz:
                npz_source = resolve_asset(source_dir, "npz", original_npz)
                npz_target = output_dir / relative_npz
                npz_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(npz_source, npz_target)
                copied_npz.add(public_case)
                copy_counts["npz"] += 1

        public_rows.append(
            {
                "public_case_id": public_case,
                "public_patient_id": public_patient,
                "label": row.get("label", ""),
                "seq": row.get("seq", ""),
                "slice_idx": row.get("slice_idx", ""),
                "image_path": public_image_path,
                "mask_path": public_mask_path,
                "npz_path": public_npz_path,
                "mask_area": row.get("mask_area", ""),
                "has_lesion": row.get("has_lesion", ""),
                "bbox_x1": row.get("bbox_x1", ""),
                "bbox_y1": row.get("bbox_y1", ""),
                "bbox_x2": row.get("bbox_x2", ""),
                "bbox_y2": row.get("bbox_y2", ""),
                "image_shape": row.get("image_shape", ""),
                "public_series_id": series_map.get(row.get("series_uid", "").strip(), ""),
                "series_description": row.get("series_description", ""),
                "label_source_category": row.get("label_source", ""),
                "mask_transform": row.get("mask_transform", ""),
            }
        )

    return public_rows, image_path_map, copy_counts


def sanitize_qwen_records(
    records: list[dict[str, Any]],
    image_path_map: dict[str, str],
    patient_map: dict[str, str],
    case_map: dict[str, str],
    series_map: dict[str, str],
) -> tuple[list[dict[str, Any]], int]:
    forbidden_keys = {
        "patient_name",
        "case_folder",
        "series_dir",
        "label_path",
        "review_overlay_path",
        "local_image_path",
    }

    def public_image_path(raw_path: str) -> str:
        return image_path_map.get(raw_path) or image_path_map.get(Path(raw_path).name, "")

    def sanitize_value(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {
                child_key: sanitize_value(child_value, child_key)
                for child_key, child_value in value.items()
                if child_key not in forbidden_keys
            }
        if isinstance(value, list):
            return [sanitize_value(item, key) for item in value]
        if not isinstance(value, str):
            return value

        if key == "patient_id":
            return patient_map.get(value, "")
        if key == "case_id":
            return case_map.get(value, "")
        if key == "series_uid":
            return series_map.get(value, "")
        if key in {"image_path", "mask_path", "npz_path"}:
            mapped = public_image_path(value) if key == "image_path" else ""
            return mapped
        if key == "study_date":
            return year_only(value)

        sanitized_text = value
        for original_case, public_case in case_map.items():
            sanitized_text = sanitized_text.replace(original_case, public_case)
        for original_patient, public_patient in patient_map.items():
            sanitized_text = sanitized_text.replace(original_patient, public_patient)
        return sanitized_text

    sanitized = []
    missing_paths = 0
    for index, record in enumerate(records, 1):
        new_record = sanitize_value(json.loads(json.dumps(record, ensure_ascii=False)))
        new_record["id"] = f"external_public_{index:06d}"
        raw_image_path = str(record.get("image_path", ""))
        if raw_image_path:
            mapped_image_path = public_image_path(raw_image_path)
            if mapped_image_path:
                new_record["image_path"] = mapped_image_path
            else:
                new_record["image_path"] = "UNRESOLVED_PUBLIC_IMAGE_PATH"
                missing_paths += 1
        original_messages = record.get("conversations", [])
        for message_index, message in enumerate(new_record.get("conversations", [])):
            if message_index < len(original_messages):
                value = str(original_messages[message_index].get("value", ""))
            else:
                value = str(message.get("value", ""))

            def replace_path(match: re.Match[str]) -> str:
                nonlocal missing_paths
                raw_path = match.group(2)
                public_path = image_path_map.get(raw_path) or image_path_map.get(Path(raw_path).name)
                if not public_path:
                    missing_paths += 1
                    public_path = "UNRESOLVED_PUBLIC_IMAGE_PATH"
                return f"{match.group(1)}{public_path}{match.group(3)}"

            sanitized_message = VISION_RE.sub(replace_path, value)
            for original_case, public_case in case_map.items():
                sanitized_message = sanitized_message.replace(original_case, public_case)
            for original_patient, public_patient in patient_map.items():
                sanitized_message = sanitized_message.replace(original_patient, public_patient)
            message["value"] = sanitized_message
        sanitized.append(new_record)
    return sanitized, missing_paths


def audit_public_json(
    records: list[dict[str, Any]],
    original_patient_ids: set[str],
    original_case_ids: set[str],
) -> dict[str, Any]:
    forbidden_keys = {
        "patient_name",
        "case_folder",
        "series_dir",
        "label_path",
        "review_overlay_path",
        "local_image_path",
    }
    findings: Counter = Counter()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in forbidden_keys:
                    findings[f"forbidden_key:{key}"] += 1
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            if re.search(r"(?:[A-Za-z]:\\|/home/|/mnt/)", value):
                findings["absolute_path"] += 1
            if any(patient_id and patient_id in value for patient_id in original_patient_ids):
                findings["raw_patient_id"] += 1
            if any(case_id and case_id in value for case_id in original_case_ids):
                findings["raw_case_id"] += 1

    walk(records)
    return {
        "finding_counts": dict(findings),
        "passed": not findings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_manifest", type=Path, required=True)
    parser.add_argument("--slice_manifest", type=Path, required=True)
    parser.add_argument("--qwen_json", type=Path, required=True)
    parser.add_argument("--hidden_qwen_json", type=Path, required=True)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_mapping_csv", type=Path, required=True)
    parser.add_argument(
        "--public_asset_prefix",
        default="",
        help="Optional public path prefix written into manifests/JSON, for example datasets/temporal_external_public_v1",
    )
    parser.add_argument("--copy_assets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Use --overwrite to continue.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_rows = read_csv(args.final_manifest)
    slice_rows = read_csv(args.slice_manifest)
    patient_map, case_map, series_map = build_identifier_maps(final_rows)

    private_mapping_rows = []
    for row in final_rows:
        private_mapping_rows.append(
            {
                "original_patient_id": row.get("patient_id", ""),
                "public_patient_id": patient_map.get(row.get("patient_id", ""), ""),
                "original_case_id": row.get("case_id", ""),
                "public_case_id": case_map.get(row.get("case_id", ""), ""),
                "original_series_uid": row.get("series_uid", ""),
                "public_series_id": series_map.get(row.get("series_uid", ""), ""),
            }
        )
    write_csv(
        args.private_mapping_csv,
        private_mapping_rows,
        [
            "original_patient_id",
            "public_patient_id",
            "original_case_id",
            "public_case_id",
            "original_series_uid",
            "public_series_id",
        ],
    )

    public_final_rows = sanitize_final_manifest(final_rows, patient_map, case_map, series_map)
    write_csv(
        args.output_dir / "temporal_external_public_manifest.csv",
        public_final_rows,
        list(public_final_rows[0].keys()),
    )

    public_slice_rows, image_path_map, copy_counts = sanitize_slice_manifest(
        slice_rows,
        patient_map,
        case_map,
        series_map,
        args.source_dir,
        args.output_dir,
        args.public_asset_prefix,
        args.copy_assets,
    )
    write_csv(
        args.output_dir / "slice_manifest_public.csv",
        public_slice_rows,
        list(public_slice_rows[0].keys()),
    )

    qwen_public, qwen_missing = sanitize_qwen_records(
        read_json(args.qwen_json), image_path_map, patient_map, case_map, series_map
    )
    hidden_public, hidden_missing = sanitize_qwen_records(
        read_json(args.hidden_qwen_json), image_path_map, patient_map, case_map, series_map
    )
    write_json(args.output_dir / "data_vl_temporal_external_public.json", qwen_public)
    write_json(args.output_dir / "data_vl_temporal_external_hidden_bbox_public.json", hidden_public)
    qwen_content_audit = audit_public_json(qwen_public, set(patient_map), set(case_map))
    hidden_content_audit = audit_public_json(hidden_public, set(patient_map), set(case_map))

    forbidden_fields = {
        "patient_name",
        "case_folder",
        "series_dir",
        "label_path",
        "review_overlay_path",
        "patient_id",
        "case_id",
        "series_uid",
        "study_date",
    }
    release_columns = set(public_final_rows[0]) | set(public_slice_rows[0])
    report = {
        "source_final_manifest": str(args.final_manifest),
        "source_slice_manifest": str(args.slice_manifest),
        "output_dir": str(args.output_dir),
        "private_mapping_csv": str(args.private_mapping_csv),
        "private_mapping_must_not_be_published": True,
        "copy_assets": args.copy_assets,
        "patients": len(patient_map),
        "cases": len(case_map),
        "series": len(series_map),
        "public_final_manifest_rows": len(public_final_rows),
        "public_slice_manifest_rows": len(public_slice_rows),
        "public_qwen_records": len(qwen_public),
        "public_hidden_qwen_records": len(hidden_public),
        "unresolved_qwen_image_paths": qwen_missing,
        "unresolved_hidden_qwen_image_paths": hidden_missing,
        "copy_counts": dict(copy_counts),
        "forbidden_release_columns_present": sorted(forbidden_fields & release_columns),
        "qwen_json_content_audit": qwen_content_audit,
        "hidden_qwen_json_content_audit": hidden_content_audit,
        "release_safe_metadata": (
            not (forbidden_fields & release_columns)
            and qwen_content_audit["passed"]
            and hidden_content_audit["passed"]
            and qwen_missing == 0
            and hidden_missing == 0
        ),
        "exact_study_dates_removed": True,
        "study_year_retained": True,
        "manual_required_checks": [
            "Keep the private mapping outside every manuscript/public release folder.",
            "Visually inspect exported JPEG files for burned-in identifiers.",
            "Do not publish the source manifests or raw local paths.",
            "Confirm institutional/ethics approval before sharing image assets.",
        ],
    }
    write_json(args.output_dir / "sanitization_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
