import os
import re
import json
import random
import argparse
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from swanlab.integration.transformers import SwanLabCallback
import swanlab


# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str):
    print(f"[INFO] {msg}", flush=True)


def extract_image_path(user_text: str) -> Optional[str]:
    m = re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", user_text, flags=re.S)
    return m.group(1).strip() if m else None


def extract_bbox_from_text(text: str) -> Optional[Tuple[int, int, int, int]]:
    """
    适配：
    bbox:[174,296,303,398],label:感染
    同时保留对旧格式的兼容
    """
    patterns = [
        r"bbox\s*[:：]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r'"bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
        r"位置\s*[:：]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.S | re.I)
        if m:
            return tuple(map(int, m.groups()))
    return None


def extract_label_from_text(text: str) -> Optional[str]:
    """
    适配：
    bbox:[...],label:感染
    """
    low = text.strip().lower()

    # 优先匹配 label 字段
    m = re.search(r"label\s*[:：]\s*([^\s,，。\]\}]+)", text, flags=re.I)
    if m:
        val = m.group(1).strip().lower()
        if ("感染" in val) or ("infection" in val):
            return "感染"
        if ("肿瘤" in val) or ("tumor" in val) or ("tumour" in val):
            return "肿瘤"

    if ("感染" in text) or ("infection" in low):
        return "感染"
    if ("肿瘤" in text) or ("tumor" in low) or ("tumour" in low):
        return "肿瘤"
    return None


def infer_seq_from_prompt(text: str) -> Optional[str]:
    patterns = [
        r"序列为\s*([Tt][12](?:WI)?)",
        r"([Tt][12](?:WI)?)\s*序列",
        r"\b([Tt][12](?:WI)?)\b",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).upper()
    return None


def clamp_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def pixel_to_norm1000(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    return (
        int(round(x1 / max(w, 1) * 1000)),
        int(round(y1 / max(h, 1) * 1000)),
        int(round(x2 / max(w, 1) * 1000)),
        int(round(y2 / max(h, 1) * 1000)),
    )


def norm1000_to_pixel(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    px = (
        int(round(x1 / 1000 * w)),
        int(round(y1 / 1000 * h)),
        int(round(x2 / 1000 * w)),
        int(round(y2 / 1000 * h)),
    )
    return clamp_bbox(px, w, h)


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def maybe_jitter_bbox(bbox: Tuple[int, int, int, int], w: int, h: int, jitter: int) -> Tuple[int, int, int, int]:
    if jitter <= 0:
        return clamp_bbox(bbox, w, h)
    x1, y1, x2, y2 = bbox
    x1 += random.randint(-jitter, jitter)
    y1 += random.randint(-jitter, jitter)
    x2 += random.randint(-jitter, jitter)
    y2 += random.randint(-jitter, jitter)
    return clamp_bbox((x1, y1, x2, y2), w, h)


def build_prompt_templates(seq: Optional[str]) -> List[str]:
    seq_part = f"，序列为{seq}" if seq else ""
    return [
        f"现在你是一个骨科专家，这是一幅脊椎的磁共振图像{seq_part}。请观察整张图像，找出图中最主要病灶的大概位置，并判断该病灶属于感染还是肿瘤。请严格按照以下格式回答：bbox:[x1,y1,x2,y2],label:感染/肿瘤",
        f"请分析这张脊柱MRI图像{seq_part}，给出主病灶的大概框坐标，并判断它属于感染还是肿瘤。请只输出：bbox:[x1,y1,x2,y2],label:感染/肿瘤",
        f"观察这张脊椎磁共振图像{seq_part}，定位最主要病灶，并输出病灶类别。输出格式固定为：bbox:[x1,y1,x2,y2],label:感染/肿瘤",
    ]


def build_answer(norm_bbox: Tuple[int, int, int, int], label: str) -> str:
    x1, y1, x2, y2 = norm_bbox
    return f"bbox:[{x1},{y1},{x2},{y2}],label:{label}"


@dataclass
class SampleRecord:
    sample_id: str
    image_path: str
    prompt: str
    answer: str
    label: str
    bbox_px: Tuple[int, int, int, int]
    bbox_norm: Tuple[int, int, int, int]
    width: int
    height: int
    patient_id: str


def parse_sample(item: Dict, jitter_px: int = 0, random_prompt: bool = True) -> Optional[SampleRecord]:
    convs = item.get("conversations", [])
    if len(convs) < 2:
        return None

    user_text = convs[0].get("value", "")
    assistant_text = convs[1].get("value", "")

    image_path = extract_image_path(user_text)
    bbox_px = extract_bbox_from_text(assistant_text) or extract_bbox_from_text(user_text)
    label = extract_label_from_text(assistant_text)
    seq = infer_seq_from_prompt(user_text)

    if image_path is None or bbox_px is None or label is None:
        return None
    if not os.path.exists(image_path):
        return None

    with Image.open(image_path) as img:
        w, h = img.size

    bbox_px = maybe_jitter_bbox(bbox_px, w, h, jitter_px)
    bbox_norm = pixel_to_norm1000(bbox_px, w, h)

    templates = build_prompt_templates(seq)
    prompt = random.choice(templates) if random_prompt else templates[0]
    answer = build_answer(bbox_norm, label)

    sample_id = item.get("id", "") or f"sample_{os.path.basename(image_path)}"
    patient_id = os.path.basename(image_path).split("_")[0]

    return SampleRecord(
        sample_id=sample_id,
        image_path=image_path,
        prompt=prompt,
        answer=answer,
        label=label,
        bbox_px=bbox_px,
        bbox_norm=bbox_norm,
        width=w,
        height=h,
        patient_id=patient_id,
    )


# =========================
# Dataset / Collator
# =========================

class DetClsDatasetBuilder:
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = processor.tokenizer

    def __call__(self, example: Dict) -> Dict:
        prompt = example["prompt"]
        image_path = example["image_path"]
        answer = example["answer"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
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

        response = self.tokenizer(answer, add_special_tokens=False, return_tensors=None)

        prompt_ids = inputs["input_ids"][0].tolist()
        prompt_mask = inputs["attention_mask"][0].tolist()
        resp_ids = response["input_ids"]
        resp_mask = response["attention_mask"]

        input_ids = prompt_ids + resp_ids + [self.tokenizer.eos_token_id]
        attention_mask = prompt_mask + resp_mask + [1]
        labels = [-100] * len(prompt_ids) + resp_ids + [self.tokenizer.eos_token_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": inputs["pixel_values"][0],
            "image_grid_thw": inputs["image_grid_thw"][0],
        }
        return out


class VLDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def _to_tensor(self, x, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        return torch.tensor(x, dtype=dtype)

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 先统一转 tensor
        input_ids_list = [self._to_tensor(f["input_ids"], torch.long) for f in features]
        attention_mask_list = [self._to_tensor(f["attention_mask"], torch.long) for f in features]
        labels_list = [self._to_tensor(f["labels"], torch.long) for f in features]
        pixel_values_list = [self._to_tensor(f["pixel_values"], torch.float32) for f in features]
        image_grid_thw_list = [self._to_tensor(f["image_grid_thw"], torch.long) for f in features]

        max_len = max(x.size(0) for x in input_ids_list)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
            l = input_ids.size(0)
            pad_len = max_len - l

            if pad_len > 0:
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]
                )
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros((pad_len,), dtype=torch.long)]
                )
                labels = torch.cat(
                    [labels, torch.full((pad_len,), -100, dtype=torch.long)]
                )

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_labels.append(labels)

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
            "pixel_values": torch.stack(pixel_values_list),
            "image_grid_thw": torch.stack(image_grid_thw_list),
        }
        return batch


# =========================
# Prediction / Evaluation
# =========================

def parse_prediction_text(text: str) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
    bbox = extract_bbox_from_text(text)
    label = extract_label_from_text(text)
    return bbox, label


def build_messages(prompt: str, image_path: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def predict_one(model, processor, prompt: str, image_path: str, max_new_tokens: int = 64) -> str:
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    messages = build_messages(prompt, image_path)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = model.get_input_embeddings().weight.device
    model_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    prompt_len = model_inputs["input_ids"].shape[1]
    trimmed = generated_ids[:, prompt_len:]
    out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out[0].strip()


def evaluate_generation(model, processor, records: List[SampleRecord], max_new_tokens=64, log_image_num=20):
    total = 0
    cls_correct = 0
    det_correct_03 = 0
    det_correct_05 = 0
    joint_correct_03 = 0
    joint_correct_05 = 0
    patient_stats = defaultdict(lambda: {"joint03_total": 0, "joint03_correct": 0})
    image_logs = []

    pbar = tqdm(records, desc="Evaluating", ncols=140)

    for rec in pbar:
        total += 1
        try:
            pred_text = predict_one(model, processor, rec.prompt, rec.image_path, max_new_tokens=max_new_tokens)
        except Exception as e:
            pred_text = f"[ERROR] {repr(e)}"

        pred_bbox_norm, pred_label = parse_prediction_text(pred_text)
        cls_ok = int(pred_label == rec.label)
        cls_correct += cls_ok

        iou = 0.0
        det03_ok = 0
        det05_ok = 0
        joint03_ok = 0
        joint05_ok = 0

        if pred_bbox_norm is not None:
            pred_bbox_px = norm1000_to_pixel(pred_bbox_norm, rec.width, rec.height)
            iou = compute_iou(pred_bbox_px, rec.bbox_px)
            det03_ok = int(iou >= 0.3)
            det05_ok = int(iou >= 0.5)
            joint03_ok = int((iou >= 0.3) and (pred_label == rec.label))
            joint05_ok = int((iou >= 0.5) and (pred_label == rec.label))
            det_correct_03 += det03_ok
            det_correct_05 += det05_ok
            joint_correct_03 += joint03_ok
            joint_correct_05 += joint05_ok

        patient_stats[rec.patient_id]["joint03_total"] += 1
        patient_stats[rec.patient_id]["joint03_correct"] += joint03_ok

        pbar.set_postfix({
            "cls": f"{cls_correct/max(total,1):.4f}",
            "j03": f"{joint_correct_03/max(total,1):.4f}",
            "iou": f"{iou:.3f}",
            "gt": rec.label,
            "pred": str(pred_label)[:10],
        })

        if len(image_logs) < log_image_num:
            try:
                image_logs.append(
                    swanlab.Image(
                        rec.image_path,
                        caption=(
                            f"pred={pred_text} | gt_bbox_norm={list(rec.bbox_norm)} | "
                            f"gt_label={rec.label} | iou={iou:.3f}"
                        ),
                    )
                )
            except Exception:
                pass

    patient_joint03_correct = 0
    total_patients = len(patient_stats)
    for _, stats in patient_stats.items():
        acc = stats["joint03_correct"] / max(stats["joint03_total"], 1)
        if acc > 0.5:
            patient_joint03_correct += 1

    metrics = {
        "eval_cls_acc": cls_correct / max(total, 1),
        "eval_det_acc_iou03": det_correct_03 / max(total, 1),
        "eval_det_acc_iou05": det_correct_05 / max(total, 1),
        "eval_joint_acc_iou03": joint_correct_03 / max(total, 1),
        "eval_joint_acc_iou05": joint_correct_05 / max(total, 1),
        "eval_patient_joint_acc_iou03": patient_joint03_correct / max(total_patients, 1),
        "eval_total_images": total,
        "eval_total_patients": total_patients,
    }
    return metrics, image_logs


# =========================
# Main
# =========================

def load_records(json_path: str, jitter_px: int = 0, random_prompt: bool = True) -> List[SampleRecord]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    skipped = 0
    for item in data:
        rec = parse_sample(item, jitter_px=jitter_px, random_prompt=random_prompt)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
    log(f"Loaded records from {json_path}: {len(records)}, skipped: {skipped}")
    return records


def make_hf_dataset(records: List[SampleRecord]) -> Dataset:
    rows = []
    for r in records:
        rows.append({
            "sample_id": r.sample_id,
            "image_path": r.image_path,
            "prompt": r.prompt,
            "answer": r.answer,
            "label": r.label,
        })
    return Dataset.from_list(rows)


def load_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForImageTextToText.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, processor


def add_lora(model, args):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/dwd/桌面/qwen_models/Qwen3.5-0.8B")
    parser.add_argument("--train_json", type=str, default="/home/dwd/桌面/Spinal-qwen-finetune/datasets/train_output/data_detcls_vl.json")
    parser.add_argument("--val_json", type=str, default="/home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_detcls_vl.json")
    parser.add_argument("--output_dir", type=str, default="/home/dwd/桌面/Spinal-qwen-finetune/output/qwen35_detcls_lora")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter_px", type=int, default=3)
    parser.add_argument("--eval_gen_max_new_tokens", type=int, default=64)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--swanlab_project", type=str, default="Qwen3.5-VL-LoRA")
    parser.add_argument("--swanlab_experiment", type=str, default="spine-detcls-finetune")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    log("Loading and converting train records...")
    train_records = load_records(args.train_json, jitter_px=args.jitter_px, random_prompt=True)

    log("Loading and converting val records...")
    val_records = load_records(args.val_json, jitter_px=0, random_prompt=False)

    log(f"Train records: {len(train_records)} | Val records: {len(val_records)}")

    log("Loading model and processor...")
    model, processor = load_model_and_processor(args)
    model = add_lora(model, args)

    builder = DetClsDatasetBuilder(processor, args.max_length)
    train_dataset = make_hf_dataset(train_records).map(
        builder,
        remove_columns=["sample_id", "image_path", "prompt", "answer", "label"]
    )
    val_dataset = make_hf_dataset(val_records).map(
        builder,
        remove_columns=["sample_id", "image_path", "prompt", "answer", "label"]
    )

    collator = VLDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    swanlab_callback = SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
        config={
            "model": args.model_name_or_path,
            "train_dataset": args.train_json,
            "val_dataset": args.val_json,
            "task": "main-lesion coarse detection + classification",
            "output_format": "bbox:[x1,y1,x2,y2],label:感染/肿瘤",
            "train_records": len(train_records),
            "val_records": len(val_records),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "jitter_px": args.jitter_px,
            "bbox_space": "normalized_0_1000",
        },
    )

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
        bf16=False,
        fp16=not args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[swanlab_callback],
    )

    log("Starting training...")
    trainer.train()

    log("Saving final adapter...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    log("Running generation evaluation...")
    gen_metrics, image_logs = evaluate_generation(
        model,
        processor,
        val_records,
        max_new_tokens=args.eval_gen_max_new_tokens,
    )
    print(json.dumps(gen_metrics, ensure_ascii=False, indent=2))
    swanlab.log(gen_metrics)
    if image_logs:
        swanlab.log({"Prediction": image_logs})
    swanlab.finish()
    log("Done.")


if __name__ == "__main__":
    main()