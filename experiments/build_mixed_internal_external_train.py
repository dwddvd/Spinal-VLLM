#!/usr/bin/env python
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import List


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


def label_of(record: dict) -> str:
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


def seq_of(record: dict) -> str:
    seq = record.get("seq")
    if seq:
        return str(seq)
    manifest = record.get("slice_manifest") or {}
    return str(manifest.get("seq") or "")


def retag(record: dict, prefix: str, index: int, source: str) -> dict:
    item = dict(record)
    item["id"] = f"{prefix}_{index:06d}"
    item["source_dataset"] = source
    return item


def summarize(records: List[dict]) -> dict:
    return {
        "records": len(records),
        "label_counts": dict(Counter(label_of(x) for x in records)),
        "seq_counts": dict(Counter(seq_of(x) for x in records)),
        "label_seq_counts": dict(Counter(f"{label_of(x)}|{seq_of(x)}" for x in records)),
        "source_counts": dict(Counter(str(x.get("source_dataset", "unknown")) for x in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed internal+external Qwen training dataset.")
    parser.add_argument("--internal_train_json", required=True)
    parser.add_argument("--internal_val_json", required=True)
    parser.add_argument("--external_adapt_train_json", required=True)
    parser.add_argument("--external_adapt_test_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--external_repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    internal_train = read_json(Path(args.internal_train_json))
    internal_val = read_json(Path(args.internal_val_json))
    external_train = read_json(Path(args.external_adapt_train_json))
    external_test = read_json(Path(args.external_adapt_test_json))

    mixed = []
    for index, record in enumerate(internal_train):
        mixed.append(retag(record, "mixed_internal", index, "internal_train"))
    offset = len(mixed)
    for rep in range(args.external_repeat):
        for index, record in enumerate(external_train):
            mixed.append(retag(record, f"mixed_external_r{rep + 1}", offset + rep * len(external_train) + index, "external_adapt"))

    rng = random.Random(args.seed)
    rng.shuffle(mixed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_mixed.json"
    internal_val_path = output_dir / "internal_val.json"
    external_test_path = output_dir / "external_test.json"
    summary_path = output_dir / "mixed_summary.json"

    write_json(train_path, mixed)
    write_json(internal_val_path, internal_val)
    write_json(external_test_path, external_test)

    summary = {
        "internal_train_json": args.internal_train_json,
        "internal_val_json": args.internal_val_json,
        "external_adapt_train_json": args.external_adapt_train_json,
        "external_adapt_test_json": args.external_adapt_test_json,
        "output_dir": str(output_dir),
        "external_repeat": args.external_repeat,
        "seed": args.seed,
        "train_mixed_json": str(train_path),
        "internal_val_json_out": str(internal_val_path),
        "external_test_json": str(external_test_path),
        "internal_train_summary": summarize(internal_train),
        "external_adapt_train_summary": summarize(external_train),
        "mixed_train_summary": summarize(mixed),
        "internal_val_summary": summarize(internal_val),
        "external_test_summary": summarize(external_test),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
