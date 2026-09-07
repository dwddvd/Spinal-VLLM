#!/usr/bin/env python
"""Filter YOLO-Qwen final selections and add inference-time aggregation fields."""

import argparse
import csv
import json
from pathlib import Path


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--strategy", default="conf_weighted_vote")
    args = parser.parse_args()

    with open(args.final_csv, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if int(float(row.get("top_k", 0))) == args.top_k and row.get("strategy") == args.strategy
    ]
    seen = set()
    duplicates = []
    output_rows = []
    for row in selected:
        sample_id = row.get("sample_id", "")
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
        x1 = as_float(row.get("pred_x1"))
        y1 = as_float(row.get("pred_y1"))
        x2 = as_float(row.get("pred_x2"))
        y2 = as_float(row.get("pred_y2"))
        has_candidate = row.get("pred_label") in {"infection", "tumor"}
        row["candidate_slice_area"] = max(x2 - x1 + 1, 0) * max(y2 - y1 + 1, 0) if has_candidate else 0
        row["candidate_slice_distance"] = 0
        row["valid_component_votes"] = 1 if has_candidate else 0
        row["final_candidate_source"] = "filtered_components" if has_candidate else "pred_mask_empty"
        row["pred_mask_empty"] = str(not has_candidate).lower()
        output_rows.append(row)
    if duplicates:
        raise ValueError(f"Duplicate sample IDs after filtering: {duplicates[:5]}")
    if not output_rows:
        raise ValueError("No rows matched the requested top_k and strategy.")

    fields = []
    for row in output_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        json.dumps(
            {
                "final_csv": args.final_csv,
                "output_csv": str(output_path),
                "top_k": args.top_k,
                "strategy": args.strategy,
                "num_input_rows": len(rows),
                "num_output_rows": len(output_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
