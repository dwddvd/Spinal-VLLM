#!/usr/bin/env python
"""Create a YOLO inference directory whose filenames preserve Qwen sample IDs."""

import argparse
import json
import os
import shutil
from pathlib import Path


def extract_image_path(item):
    image_path = item.get("image_path")
    if image_path:
        return str(image_path)
    conversations = item.get("conversations", [])
    if conversations:
        text = str(conversations[0].get("value", ""))
        start = text.find("<|vision_start|>")
        end = text.find("<|vision_end|>")
        if start >= 0 and end > start:
            return text[start + len("<|vision_start|>") : end].strip()
    return ""


def remap(path_text, source_prefix, local_prefix):
    normalized = str(path_text).replace("\\", "/")
    source = str(source_prefix or "").replace("\\", "/").rstrip("/")
    if source and local_prefix and normalized.startswith(source):
        suffix = normalized[len(source) :].lstrip("/")
        return Path(local_prefix) / Path(*suffix.split("/"))
    return Path(path_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_image_prefix", default="")
    parser.add_argument("--local_image_prefix", default="")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.qwen_json, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    manifest = []
    missing = []
    for index, item in enumerate(records):
        sample_id = str(item.get("id") or f"sample_{index:06d}")
        source = remap(
            extract_image_path(item),
            args.source_image_prefix,
            args.local_image_prefix,
        )
        if not source.is_file():
            missing.append({"sample_id": sample_id, "source": str(source)})
            continue
        target = output_dir / f"{index:06d}_{sample_id}{source.suffix.lower() or '.jpg'}"
        if not target.exists():
            if args.copy:
                shutil.copy2(source, target)
            else:
                try:
                    os.symlink(source.resolve(), target)
                except OSError:
                    shutil.copy2(source, target)
        manifest.append(
            {
                "sample_id": sample_id,
                "source_image": str(source),
                "yolo_image": str(target),
            }
        )

    summary = {
        "qwen_json": args.qwen_json,
        "output_dir": str(output_dir),
        "num_records": len(records),
        "num_prepared": len(manifest),
        "num_missing": len(missing),
        "link_mode": "copy" if args.copy else "symlink_with_copy_fallback",
    }
    with (output_dir / "inference_source_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": manifest, "missing": missing}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing:
        raise FileNotFoundError(f"{len(missing)} source images were missing; see inference_source_manifest.json")


if __name__ == "__main__":
    main()
