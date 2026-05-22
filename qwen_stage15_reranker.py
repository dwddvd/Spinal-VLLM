import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

from qwen_stage2_classifier import (
    VLDataCollator,
    add_lora,
    load_model_and_processor,
    load_records,
    make_non_empty_bbox,
    set_seed,
)
from yolo_qwen_pipeline import CandidateBox, compute_iou, find_candidates, load_candidates


YES_ZH = "\u662f"
NO_ZH = "\u5426"


@dataclass
class RerankRecord:
    sample_id: str
    image_path: str
    gt_bbox: Tuple[int, int, int, int]
    cand_bbox: Tuple[int, int, int, int]
    rank: int
    conf: float
    iou: float
    label: str
    seq: Optional[str] = None


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def normalize_rerank_label(text: str) -> Optional[str]:
    text = text.strip()
    low = text.lower()
    if text.startswith(YES_ZH) or low.startswith("yes") or low.startswith("true"):
        return "yes"
    if text.startswith(NO_ZH) or low.startswith("no") or low.startswith("false"):
        return "no"
    if YES_ZH in text and NO_ZH not in text:
        return "yes"
    if NO_ZH in text and YES_ZH not in text:
        return "no"
    return None


def answer_text(label: str) -> str:
    if label == "yes":
        return YES_ZH
    if label == "no":
        return NO_ZH
    raise ValueError(f"Unknown rerank label: {label}")


def build_prompt(seq: Optional[str], bbox: Tuple[int, int, int, int]) -> str:
    seq_text = f"\u5e8f\u5217\u4e3a{seq}\u3002" if seq else ""
    x1, y1, x2, y2 = bbox
    return (
        f"\u8fd9\u662f\u4e00\u5e45\u810a\u690e\u7684\u78c1\u5171\u632f\u56fe\u50cf\uff0c{seq_text}"
        f"\u5019\u9009\u6846\u4f4d\u7f6e\u6309\u7167[x1,y1,x2,y2]\u683c\u5f0f\u4e3a[{x1},{y1},{x2},{y2}]\u3002"
        "\u8bf7\u5224\u65ad\u8be5\u5019\u9009\u6846\u662f\u5426\u8986\u76d6\u771f\u5b9e\u75c5\u7076\u533a\u57df\u3002"
        "\u5982\u679c\u5019\u9009\u6846\u4e3b\u8981\u8986\u76d6\u75c5\u7076\uff0c\u8f93\u51fa\u662f\uff1b"
        "\u5982\u679c\u4e0d\u662f\u75c5\u7076\u6216\u660e\u663e\u504f\u79bb\u75c5\u7076\uff0c\u8f93\u51fa\u5426\u3002"
        "\u53ea\u8f93\u51fa\u662f\u6216\u5426\u3002"
    )


def record_to_dict(record: RerankRecord) -> Dict:
    return {
        "sample_id": record.sample_id,
        "image_path": record.image_path,
        "gt_bbox": list(record.gt_bbox),
        "cand_bbox": list(record.cand_bbox),
        "rank": record.rank,
        "conf": record.conf,
        "iou": record.iou,
        "label": record.label,
        "seq": record.seq,
    }


def dict_to_record(item: Dict) -> RerankRecord:
    return RerankRecord(
        sample_id=str(item["sample_id"]),
        image_path=str(item["image_path"]),
        gt_bbox=tuple(item["gt_bbox"]),
        cand_bbox=tuple(item["cand_bbox"]),
        rank=int(item["rank"]),
        conf=float(item["conf"]),
        iou=float(item["iou"]),
        label=str(item["label"]),
        seq=item.get("seq"),
    )


def load_rerank_records(path: str) -> List[RerankRecord]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = [dict_to_record(item) for item in data]
    log(f"Loaded {len(records)} rerank records from {path}.")
    return records


def label_distribution(records: List[RerankRecord]) -> Dict[str, int]:
    counts = Counter(record.label for record in records)
    return {"yes": int(counts.get("yes", 0)), "no": int(counts.get("no", 0))}


def balance_records(records: List[RerankRecord], seed: int, max_neg_per_pos: int) -> List[RerankRecord]:
    positives = [record for record in records if record.label == "yes"]
    negatives = [record for record in records if record.label == "no"]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if positives and max_neg_per_pos > 0:
        negatives = negatives[: len(positives) * max_neg_per_pos]
    balanced = positives + negatives
    rng.shuffle(balanced)
    return balanced


def prepare(args: argparse.Namespace) -> None:
    lesion_records = load_records(args.detcls_json)
    candidates = load_candidates(args.pred_csv, args.top_k)
    rng = random.Random(args.seed)
    rerank_records: List[RerankRecord] = []
    skipped_no_candidates = 0
    ignored_mid_iou = 0

    for index, lesion in enumerate(lesion_records):
        image_candidates = find_candidates(candidates, lesion, index)
        if not image_candidates:
            skipped_no_candidates += 1
            continue

        positives: List[RerankRecord] = []
        negatives: List[RerankRecord] = []
        for candidate in image_candidates:
            bbox = make_non_empty_bbox(candidate.bbox, lesion.width, lesion.height)
            iou = compute_iou(bbox, lesion.bbox)
            if iou >= args.pos_iou:
                label = "yes"
            elif iou < args.neg_iou:
                label = "no"
            else:
                ignored_mid_iou += 1
                continue
            record = RerankRecord(
                sample_id=lesion.sample_id,
                image_path=lesion.image_path,
                gt_bbox=lesion.bbox,
                cand_bbox=bbox,
                rank=candidate.rank,
                conf=candidate.conf,
                iou=iou,
                label=label,
                seq=lesion.seq,
            )
            if label == "yes":
                positives.append(record)
            else:
                negatives.append(record)

        if args.keep_all_pos:
            rerank_records.extend(positives)
        elif positives:
            rerank_records.append(max(positives, key=lambda record: record.iou))

        if args.neg_per_image >= 0 and negatives:
            rng.shuffle(negatives)
            rerank_records.extend(negatives[: args.neg_per_image])
        elif args.neg_per_image < 0:
            rerank_records.extend(negatives)

    if args.balance:
        rerank_records = balance_records(rerank_records, args.seed, args.max_neg_per_pos)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([record_to_dict(record) for record in rerank_records], f, ensure_ascii=False, indent=2)

    summary = {
        "detcls_json": args.detcls_json,
        "pred_csv": args.pred_csv,
        "top_k": args.top_k,
        "pos_iou": args.pos_iou,
        "neg_iou": args.neg_iou,
        "total": len(rerank_records),
        "label_distribution": label_distribution(rerank_records),
        "skipped_no_candidates": skipped_no_candidates,
        "ignored_mid_iou": ignored_mid_iou,
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"Saved rerank data to {output_path}")
    log(f"Saved summary to {summary_path}")


class RerankDatasetBuilder:
    def __init__(self, processor, max_length: int, image_resize: int):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.image_resize = image_resize

    def __call__(self, record: RerankRecord) -> Dict:
        image_content = {"type": "image", "image": record.image_path}
        if self.image_resize > 0:
            image_content["resized_height"] = self.image_resize
            image_content["resized_width"] = self.image_resize
        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": build_prompt(record.seq, record.cand_bbox)},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        response = self.tokenizer(answer_text(record.label), add_special_tokens=False, return_tensors=None)

        prompt_ids = inputs["input_ids"][0].tolist()
        prompt_mask = inputs["attention_mask"][0].tolist()
        resp_ids = response["input_ids"]
        resp_mask = response["attention_mask"]
        eos_id = self.tokenizer.eos_token_id

        input_ids = prompt_ids + resp_ids + [eos_id]
        attention_mask = prompt_mask + resp_mask + [1]
        labels = [-100] * len(prompt_ids) + resp_ids + [eos_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]

        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 3 and pixel_values.size(0) == 1:
            pixel_values = pixel_values.squeeze(0)
        image_grid_thw = inputs["image_grid_thw"]
        if image_grid_thw.dim() >= 2 and image_grid_thw.size(0) == 1:
            image_grid_thw = image_grid_thw.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values.tolist(),
            "image_grid_thw": image_grid_thw.tolist(),
        }


class RerankLazyDataset(torch.utils.data.Dataset):
    def __init__(self, records: List[RerankRecord], builder: RerankDatasetBuilder):
        self.records = records
        self.builder = builder

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        return self.builder(self.records[index])


def generate_text(model, processor, image_path: str, seq: Optional[str], bbox: Tuple[int, int, int, int], image_resize: int, max_new_tokens: int) -> str:
    image_content = {"type": "image", "image": image_path}
    if image_resize > 0:
        image_content["resized_height"] = image_resize
        image_content["resized_width"] = image_resize
    messages = [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": build_prompt(seq, bbox)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = model.get_input_embeddings().weight.device
    model_inputs = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    prompt_len = model_inputs["input_ids"].shape[1]
    text = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text[0].strip()


def load_qwen(base_model: str, adapter_path: Optional[str], load_in_4bit: bool, gradient_checkpointing: bool):
    model, processor = load_model_and_processor(base_model, load_in_4bit, gradient_checkpointing)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model, processor


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    train_records = load_rerank_records(args.train_json)
    val_records = load_rerank_records(args.val_json)
    if args.limit_train > 0:
        train_records = train_records[: args.limit_train]
    if args.limit_val > 0:
        val_records = val_records[: args.limit_val]
    log(f"Train distribution: {label_distribution(train_records)}")
    log(f"Val distribution: {label_distribution(val_records)}")

    model, processor = load_qwen(args.base_model, adapter_path=None, load_in_4bit=args.load_in_4bit, gradient_checkpointing=args.gradient_checkpointing)
    model = add_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    builder = RerankDatasetBuilder(processor, args.max_length, args.image_resize)
    train_dataset = RerankLazyDataset(train_records, builder)
    val_dataset = RerankLazyDataset(val_records, builder)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        fp16=not args.load_in_4bit,
        bf16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VLDataCollator(processor.tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    log(f"Saved reranker to {args.output_dir}")


def eval_rerank_records(args: argparse.Namespace) -> None:
    records = load_rerank_records(args.val_json)
    model, processor = load_qwen(args.base_model, args.adapter_path, args.load_in_4bit, gradient_checkpointing=False)
    model.eval()
    correct = 0
    confusion = Counter()
    rows: List[Dict[str, object]] = []
    for record in tqdm(records, desc="Eval reranker candidates", ncols=120):
        text = generate_text(model, processor, record.image_path, record.seq, record.cand_bbox, args.image_resize, args.max_new_tokens)
        pred = normalize_rerank_label(text)
        correct += int(pred == record.label)
        confusion[f"{record.label}->{pred or 'invalid'}"] += 1
        rows.append({**record_to_dict(record), "pred_text": text, "pred_label": pred or ""})

    metrics = {
        "total": len(records),
        "acc": correct / max(len(records), 1),
        "confusion": dict(confusion),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reranker_candidate_eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "reranker_candidate_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "image_path", "gt_bbox", "cand_bbox", "rank", "conf", "iou", "label", "seq", "pred_text", "pred_label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def eval_select(args: argparse.Namespace) -> None:
    lesion_records = load_records(args.detcls_json)
    candidates = load_candidates(args.pred_csv, args.top_k)
    model, processor = load_qwen(args.base_model, args.adapter_path, args.load_in_4bit, gradient_checkpointing=False)
    model.eval()

    total = 0
    has_candidate = 0
    selected_hits = {0.3: 0, 0.5: 0}
    oracle_hits = {0.3: 0, 0.5: 0}
    selected_rows: List[Dict[str, object]] = []

    for index, lesion in enumerate(tqdm(lesion_records, desc="Eval reranker selection", ncols=120)):
        total += 1
        image_candidates = find_candidates(candidates, lesion, index)
        if not image_candidates:
            selected_rows.append({"image_path": lesion.image_path, "selected_rank": "", "pred_label": "", "selected_iou": 0.0})
            continue
        has_candidate += 1
        predictions = []
        for candidate in image_candidates:
            bbox = make_non_empty_bbox(candidate.bbox, lesion.width, lesion.height)
            text = generate_text(model, processor, lesion.image_path, lesion.seq, bbox, args.image_resize, args.max_new_tokens)
            pred = normalize_rerank_label(text)
            iou = compute_iou(bbox, lesion.bbox)
            predictions.append((candidate, bbox, text, pred, iou))
        yes_predictions = [item for item in predictions if item[3] == "yes"]
        selected = yes_predictions[0] if yes_predictions else predictions[0]
        for threshold in (0.3, 0.5):
            selected_hits[threshold] += int(selected[4] >= threshold)
            oracle_hits[threshold] += int(any(item[4] >= threshold for item in predictions))
        selected_rows.append(
            {
                "image_path": lesion.image_path,
                "selected_rank": selected[0].rank,
                "selected_conf": selected[0].conf,
                "pred_text": selected[2],
                "pred_label": selected[3] or "",
                "selected_iou": selected[4],
                "x1": selected[1][0],
                "y1": selected[1][1],
                "x2": selected[1][2],
                "y2": selected[1][3],
            }
        )

    metrics = {
        "total_images": total,
        "images_with_candidate": has_candidate,
        "candidate_coverage": has_candidate / max(total, 1),
        "selected_det_recall_iou0.3": selected_hits[0.3] / max(total, 1),
        "selected_det_recall_iou0.5": selected_hits[0.5] / max(total, 1),
        "oracle_det_recall_iou0.3": oracle_hits[0.3] / max(total, 1),
        "oracle_det_recall_iou0.5": oracle_hits[0.5] / max(total, 1),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reranker_selection_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / "reranker_selected_boxes.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "selected_rank", "selected_conf", "pred_text", "pred_label", "selected_iou", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1.5 Qwen reranker for selecting the best YOLO lesion candidate.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--detcls_json", required=True)
    prepare_parser.add_argument("--pred_csv", required=True)
    prepare_parser.add_argument("--output_json", required=True)
    prepare_parser.add_argument("--top_k", type=int, default=10)
    prepare_parser.add_argument("--pos_iou", type=float, default=0.5)
    prepare_parser.add_argument("--neg_iou", type=float, default=0.3)
    prepare_parser.add_argument("--neg_per_image", type=int, default=3, help="Use -1 to keep all negative candidates.")
    prepare_parser.add_argument("--keep_all_pos", action="store_true")
    prepare_parser.add_argument("--balance", action="store_true", default=True)
    prepare_parser.add_argument("--no_balance", action="store_false", dest="balance")
    prepare_parser.add_argument("--max_neg_per_pos", type=int, default=3)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.set_defaults(func=prepare)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--base_model", required=True)
    train_parser.add_argument("--train_json", required=True)
    train_parser.add_argument("--val_json", required=True)
    train_parser.add_argument("--output_dir", default="output/qwen_stage15_reranker")
    train_parser.add_argument("--image_resize", type=int, default=280)
    train_parser.add_argument("--max_length", type=int, default=8192)
    train_parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    train_parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    train_parser.add_argument("--num_train_epochs", type=float, default=2.0)
    train_parser.add_argument("--learning_rate", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=0.0)
    train_parser.add_argument("--warmup_ratio", type=float, default=0.03)
    train_parser.add_argument("--logging_steps", type=int, default=10)
    train_parser.add_argument("--save_steps", type=int, default=100)
    train_parser.add_argument("--eval_steps", type=int, default=300)
    train_parser.add_argument("--save_total_limit", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--limit_train", type=int, default=0)
    train_parser.add_argument("--limit_val", type=int, default=0)
    train_parser.add_argument("--load_in_4bit", action="store_true")
    train_parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    train_parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    train_parser.add_argument("--lora_r", type=int, default=64)
    train_parser.add_argument("--lora_alpha", type=int, default=16)
    train_parser.add_argument("--lora_dropout", type=float, default=0.05)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--base_model", required=True)
    eval_parser.add_argument("--adapter_path", required=True)
    eval_parser.add_argument("--val_json", required=True)
    eval_parser.add_argument("--output_dir", default="output/qwen_stage15_reranker_eval")
    eval_parser.add_argument("--image_resize", type=int, default=280)
    eval_parser.add_argument("--load_in_4bit", action="store_true")
    eval_parser.add_argument("--max_new_tokens", type=int, default=8)
    eval_parser.set_defaults(func=eval_rerank_records)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--base_model", required=True)
    select_parser.add_argument("--adapter_path", required=True)
    select_parser.add_argument("--detcls_json", required=True)
    select_parser.add_argument("--pred_csv", required=True)
    select_parser.add_argument("--output_dir", default="output/qwen_stage15_reranker_select")
    select_parser.add_argument("--top_k", type=int, default=10)
    select_parser.add_argument("--image_resize", type=int, default=280)
    select_parser.add_argument("--load_in_4bit", action="store_true")
    select_parser.add_argument("--max_new_tokens", type=int, default=8)
    select_parser.set_defaults(func=eval_select)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
