import os
import re
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from datasets import Dataset
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from swanlab.integration.transformers import SwanLabCallback
import swanlab


# =====================================
# Basic utils
# =====================================

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
    patterns = [
        r"bbox\s*[:：]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r'"bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
        r"位置\s*[:：]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.I | re.S)
        if m:
            return tuple(map(int, m.groups()))
    return None


def extract_label_from_text(text: str) -> Optional[str]:
    low = text.strip().lower()

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


def maybe_expand_bbox(bbox: Tuple[int, int, int, int], w: int, h: int, ratio: float = 0.1):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * ratio))
    pad_y = int(round(bh * ratio))
    return clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), w, h)


def crop_by_bbox(image: Image.Image, bbox: Tuple[int, int, int, int], expand_ratio: float = 0.1) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = maybe_expand_bbox(bbox, w, h, expand_ratio)
    return image.crop((x1, y1, x2, y2))


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
    return 0.0 if union <= 0 else inter / union


# =====================================
# Vertical 3-region helpers
# =====================================

VERTICAL_REGIONS = ["upper", "middle", "lower"]


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_to_vertical_region(bbox: Tuple[int, int, int, int], h: int) -> str:
    _, cy = bbox_center(bbox)
    y_ratio = cy / max(h, 1)
    if y_ratio < 1 / 3:
        return "upper"
    elif y_ratio < 2 / 3:
        return "middle"
    else:
        return "lower"


def vertical_region_to_bbox(region: str, w: int, h: int) -> Tuple[int, int, int, int]:
    if region == "upper":
        return clamp_bbox((0, 0, w, int(round(h / 3))), w, h)
    elif region == "middle":
        return clamp_bbox((0, int(round(h / 3)), w, int(round(2 * h / 3))), w, h)
    else:
        return clamp_bbox((0, int(round(2 * h / 3)), w, h), w, h)


def extract_region_from_text(text: str) -> Optional[str]:
    m = re.search(r"region\s*[:：]\s*(upper|middle|lower)", text, flags=re.I)
    if not m:
        return None
    return m.group(1).lower()


def compute_vertical_region_metrics(
    gt_regions: List[str],
    pred_regions: List[Optional[str]],
) -> Dict[str, float]:
    region_to_idx = {r: i for i, r in enumerate(VERTICAL_REGIONS)}
    num_classes = len(VERTICAL_REGIONS)

    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    total = len(gt_regions)
    correct = 0

    for gt, pred in zip(gt_regions, pred_regions):
        gt_idx = region_to_idx[gt]
        if pred is None or pred not in region_to_idx:
            continue
        pred_idx = region_to_idx[pred]
        confusion[gt_idx][pred_idx] += 1
        if gt_idx == pred_idx:
            correct += 1

    metrics = {"det_region_acc": correct / max(total, 1)}

    recalls = []
    precisions = []

    for region in VERTICAL_REGIONS:
        c = region_to_idx[region]
        tp = confusion[c][c]
        fn = sum(confusion[c][j] for j in range(num_classes) if j != c)
        fp = sum(confusion[i][c] for i in range(num_classes) if i != c)

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)

        recalls.append(recall)
        precisions.append(precision)

        metrics[f"det_region_recall_{region}"] = recall
        metrics[f"det_region_precision_{region}"] = precision

    metrics["det_region_macro_recall"] = sum(recalls) / num_classes
    metrics["det_region_macro_precision"] = sum(precisions) / num_classes
    return metrics


# =====================================
# Data record
# =====================================

@dataclass
class Record:
    sample_id: str
    image_path: str
    bbox_px: Tuple[int, int, int, int]
    label: str
    seq: Optional[str]
    width: int
    height: int
    patient_id: str


def parse_record(item: Dict) -> Optional[Record]:
    convs = item.get("conversations", [])
    if len(convs) < 2:
        return None

    user_text = convs[0].get("value", "")
    assistant_text = convs[1].get("value", "")

    image_path = extract_image_path(user_text)
    bbox = extract_bbox_from_text(assistant_text) or extract_bbox_from_text(user_text)
    label = extract_label_from_text(assistant_text)
    seq = infer_seq_from_prompt(user_text)

    if image_path is None or bbox is None or label is None:
        return None
    if not os.path.exists(image_path):
        return None

    with Image.open(image_path) as img:
        w, h = img.size

    bbox = clamp_bbox(bbox, w, h)
    patient_id = os.path.basename(image_path).split("_")[0]
    sample_id = item.get("id", "") or f"sample_{os.path.basename(image_path)}"

    return Record(
        sample_id=sample_id,
        image_path=image_path,
        bbox_px=bbox,
        label=label,
        seq=seq,
        width=w,
        height=h,
        patient_id=patient_id,
    )


def load_records(json_path: str) -> List[Record]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    skipped = 0
    for item in data:
        rec = parse_record(item)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    log(f"Loaded from {json_path}: {len(records)}, skipped: {skipped}")
    return records


# =====================================
# Stage-1 Detector prompts/answers
# =====================================

def build_det_prompt(seq: Optional[str]) -> str:
    seq_part = f"，序列为{seq}" if seq else ""
    return (
        f"现在你是一个骨科专家，这是一幅脊椎的磁共振图像{seq_part}，图像中可能包含数个病灶。"
        f"请先重点关注脊柱主体区域，包括椎体、椎间盘及其邻近区域，不要被周围无关软组织干扰。"
        f"然后判断图中最主要病灶位于脊柱的上部、中部还是下部。"
        f"这里的上中下是按照脊柱主体在图像中的纵向位置划分，而不是按照整张图像背景划分。"
        f"请严格按照以下格式回答：region:upper / region:middle / region:lower"
    )


def build_det_answer(bbox_px: Tuple[int, int, int, int], h: int) -> str:
    region = bbox_to_vertical_region(bbox_px, h)
    return f"region:{region}"


# =====================================
# Stage-2 Classifier prompts/answers
# =====================================

def build_cls_prompt(seq: Optional[str]) -> str:
    seq_part = f"，序列为{seq}" if seq else ""
    return (
        f"现在你是一个骨科专家，这是一幅脊椎病灶区域的磁共振图像{seq_part}。"
        f"请根据该病灶区域判断它属于感染还是肿瘤。"
        f"请严格按照以下格式回答：label:感染/肿瘤"
    )


def build_cls_answer(label: str) -> str:
    return f"label:{label}"


# =====================================
# Dataset builder
# =====================================

class TwoStageDatasetBuilder:
    def __init__(self, processor, task: str, max_length: int, crop_expand_ratio: float = 0.1):
        self.processor = processor
        self.task = task
        self.max_length = max_length
        self.crop_expand_ratio = crop_expand_ratio
        self.tokenizer = processor.tokenizer

    def _build_messages_and_answer(self, example: Dict):
        image_path = example["image_path"]
        label = example["label"]
        bbox = tuple(example["bbox_px"])
        seq = example["seq"]
        height = example["height"]

        if self.task == "det":
            prompt = build_det_prompt(seq)
            answer = build_det_answer(bbox, height)
            image_obj = image_path
        elif self.task == "cls":
            prompt = build_cls_prompt(seq)
            answer = build_cls_answer(label)
            img = Image.open(image_path).convert("RGB")
            image_obj = crop_by_bbox(img, bbox, expand_ratio=self.crop_expand_ratio)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_obj},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages, answer

    def __call__(self, example: Dict) -> Dict:
        messages, answer = self._build_messages_and_answer(example)

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

        eos_id = self.tokenizer.eos_token_id
        input_ids = prompt_ids + resp_ids + [eos_id]
        attention_mask = prompt_mask + resp_mask + [1]
        labels = [-100] * len(prompt_ids) + resp_ids + [eos_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs["pixel_values"][0].tolist(),
            "image_grid_thw": inputs["image_grid_thw"][0].tolist(),
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
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros((pad_len,), dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

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


def make_hf_dataset(records: List[Record]) -> Dataset:
    rows = []
    for r in records:
        rows.append(
            {
                "sample_id": r.sample_id,
                "image_path": r.image_path,
                "bbox_px": list(r.bbox_px),
                "label": r.label,
                "seq": r.seq,
                "width": r.width,
                "height": r.height,
                "patient_id": r.patient_id,
            }
        )
    return Dataset.from_list(rows)


# =====================================
# Model helpers
# =====================================

def load_model_and_processor(model_name_or_path: str, load_in_4bit: bool, gradient_checkpointing: bool):
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, processor


def add_lora(model, r: int, alpha: int, dropout: float):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# =====================================
# Prediction
# =====================================

def build_messages(prompt: str, image_obj):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_text(model, processor, messages, max_new_tokens: int = 64) -> str:
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = model.get_input_embeddings().weight.device
    model_inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

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


def predict_region(det_model, det_processor, image_path: str, seq: Optional[str], max_new_tokens: int = 16):
    prompt = build_det_prompt(seq)
    messages = build_messages(prompt, image_path)
    text = generate_text(det_model, det_processor, messages, max_new_tokens=max_new_tokens)
    region = extract_region_from_text(text)
    return text, region


def predict_label(cls_model, cls_processor, image_path: str, bbox_px: Tuple[int, int, int, int], seq: Optional[str],
                  max_new_tokens: int = 16):
    img = Image.open(image_path).convert("RGB")
    crop = crop_by_bbox(img, bbox_px, expand_ratio=0.1)
    prompt = build_cls_prompt(seq)
    messages = build_messages(prompt, crop)
    text = generate_text(cls_model, cls_processor, messages, max_new_tokens=max_new_tokens)
    label = extract_label_from_text(text)
    return text, label


# =====================================
# Evaluation
# =====================================

def evaluate_detector(model, processor, records: List[Record], log_image_num: int = 20):
    total = 0
    iou03 = 0
    iou05 = 0
    image_logs = []

    gt_regions = []
    pred_regions = []

    pbar = tqdm(records, desc="Eval Detector", ncols=140)
    for rec in pbar:
        total += 1

        pred_text, pred_region = predict_region(
            model, processor, rec.image_path, rec.seq, max_new_tokens=16
        )

        gt_region = bbox_to_vertical_region(rec.bbox_px, rec.height)

        gt_regions.append(gt_region)
        pred_regions.append(pred_region)

        iou = 0.0
        region_correct = int(pred_region == gt_region)

        if pred_region is not None:
            pred_bbox_px = vertical_region_to_bbox(pred_region, rec.width, rec.height)
            iou = compute_iou(pred_bbox_px, rec.bbox_px)
            iou03 += int(iou >= 0.3)
            iou05 += int(iou >= 0.5)

        running_acc = sum(int(g == p) for g, p in zip(gt_regions, pred_regions)) / max(len(gt_regions), 1)

        pbar.set_postfix(
            acc=f"{running_acc:.4f}",
            hit=f"{region_correct}",
            iou03=f"{iou03 / max(total, 1):.4f}",
            iou05=f"{iou05 / max(total, 1):.4f}",
            iou=f"{iou:.3f}",
            gt=gt_region,
            pred=pred_region if pred_region is not None else "None",
        )

        if len(image_logs) < log_image_num:
            try:
                image_logs.append(
                    swanlab.Image(
                        rec.image_path,
                        caption=(
                            f"pred={pred_text} | gt_region={gt_region} | "
                            f"pred_region={pred_region if pred_region is not None else 'None'} | "
                            f"gt_bbox={list(rec.bbox_px)} | iou={iou:.3f}"
                        )
                    )
                )
            except Exception:
                pass

    cls_metrics = compute_vertical_region_metrics(
        gt_regions=gt_regions,
        pred_regions=pred_regions,
    )

    metrics = {
        **cls_metrics,
        "det_iou03": iou03 / max(total, 1),
        "det_iou05": iou05 / max(total, 1),
        "det_total": total,
    }
    return metrics, image_logs


def evaluate_classifier(model, processor, records: List[Record], log_image_num: int = 20):
    total = 0
    correct = 0
    image_logs = []

    pbar = tqdm(records, desc="Eval Classifier", ncols=120)
    for rec in pbar:
        total += 1
        pred_text, pred_label = predict_label(model, processor, rec.image_path, rec.bbox_px, rec.seq)
        correct += int(pred_label == rec.label)
        pbar.set_postfix(acc=f"{correct / max(total, 1):.4f}", gt=rec.label, pred=str(pred_label))

        if len(image_logs) < log_image_num:
            try:
                image_logs.append(
                    swanlab.Image(
                        rec.image_path,
                        caption=f"pred={pred_text} | gt_label={rec.label}"
                    )
                )
            except Exception:
                pass

    metrics = {
        "cls_acc": correct / max(total, 1),
        "cls_total": total,
    }
    return metrics, image_logs


def evaluate_pipeline(det_model, det_processor, cls_model, cls_processor, records: List[Record],
                      log_image_num: int = 20):
    total = 0
    det03 = 0
    det05 = 0
    cls_acc = 0
    joint03 = 0
    joint05 = 0

    patient_stats = {}
    image_logs = []

    gt_regions = []
    pred_regions = []

    pbar = tqdm(records, desc="Eval Two-Stage Pipeline", ncols=140)
    for rec in pbar:
        total += 1

        det_text, pred_region = predict_region(
            det_model, det_processor, rec.image_path, rec.seq, max_new_tokens=16
        )

        gt_region = bbox_to_vertical_region(rec.bbox_px, rec.height)
        gt_regions.append(gt_region)
        pred_regions.append(pred_region)

        pred_bbox_px = None
        pred_label = None
        cls_text = ""
        iou = 0.0

        if pred_region is not None:
            pred_bbox_px = vertical_region_to_bbox(pred_region, rec.width, rec.height)
            iou = compute_iou(pred_bbox_px, rec.bbox_px)
            det03 += int(iou >= 0.3)
            det05 += int(iou >= 0.5)

            cls_text, pred_label = predict_label(cls_model, cls_processor, rec.image_path, pred_bbox_px, rec.seq)
            cls_acc += int(pred_label == rec.label)
            joint03 += int((iou >= 0.3) and (pred_label == rec.label))
            joint05 += int((iou >= 0.5) and (pred_label == rec.label))

        if rec.patient_id not in patient_stats:
            patient_stats[rec.patient_id] = {"total": 0, "joint03": 0}
        patient_stats[rec.patient_id]["total"] += 1
        patient_stats[rec.patient_id]["joint03"] += int((iou >= 0.3) and (pred_label == rec.label))

        running_acc = sum(int(g == p) for g, p in zip(gt_regions, pred_regions)) / max(len(gt_regions), 1)

        pbar.set_postfix(
            region_acc=f"{running_acc:.4f}",
            det03=f"{det03 / max(total, 1):.4f}",
            cls=f"{cls_acc / max(total, 1):.4f}",
            j03=f"{joint03 / max(total, 1):.4f}",
            iou=f"{iou:.3f}",
            gt=rec.label,
            pred=str(pred_label),
        )

        if len(image_logs) < log_image_num:
            try:
                image_logs.append(
                    swanlab.Image(
                        rec.image_path,
                        caption=(
                            f"det_pred={det_text} | gt_region={gt_region} | "
                            f"pred_region={pred_region if pred_region is not None else 'None'} | "
                            f"cls_pred={cls_text} | gt_bbox={list(rec.bbox_px)} | "
                            f"gt_label={rec.label} | iou={iou:.3f}"
                        )
                    )
                )
            except Exception:
                pass

    patient_joint03 = 0
    for _, st in patient_stats.items():
        if st["joint03"] / max(st["total"], 1) > 0.5:
            patient_joint03 += 1

    region_metrics = compute_vertical_region_metrics(
        gt_regions=gt_regions,
        pred_regions=pred_regions,
    )

    metrics = {
        "pipeline_region_acc": region_metrics["det_region_acc"],
        "pipeline_region_macro_recall": region_metrics["det_region_macro_recall"],
        "pipeline_region_macro_precision": region_metrics["det_region_macro_precision"],
        "pipeline_det_iou03": det03 / max(total, 1),
        "pipeline_det_iou05": det05 / max(total, 1),
        "pipeline_cls_acc": cls_acc / max(total, 1),
        "pipeline_joint_iou03": joint03 / max(total, 1),
        "pipeline_joint_iou05": joint05 / max(total, 1),
        "pipeline_patient_joint_iou03": patient_joint03 / max(len(patient_stats), 1),
        "pipeline_total_images": total,
        "pipeline_total_patients": len(patient_stats),
    }

    for k, v in region_metrics.items():
        if k.startswith("det_region_recall_") or k.startswith("det_region_precision_"):
            metrics["pipeline_" + k.replace("det_region_", "region_")] = v

    return metrics, image_logs


# =====================================
# Training entry
# =====================================

def train_stage(
        task: str,
        model_name_or_path: str,
        train_json: str,
        val_json: str,
        output_dir: str,
        max_length: int,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        gradient_accumulation_steps: int,
        num_train_epochs: float,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        logging_steps: int,
        save_steps: int,
        eval_steps: int,
        save_total_limit: int,
        load_in_4bit: bool,
        gradient_checkpointing: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        crop_expand_ratio: float,
        swanlab_project: str,
        swanlab_experiment: str,
):
    log(f"Loading records for task={task} ...")
    train_records = load_records(train_json)
    val_records = load_records(val_json)

    log("Loading model and processor ...")
    model, processor = load_model_and_processor(
        model_name_or_path=model_name_or_path,
        load_in_4bit=load_in_4bit,
        gradient_checkpointing=gradient_checkpointing,
    )
    model = add_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    builder = TwoStageDatasetBuilder(
        processor=processor,
        task=task,
        max_length=max_length,
        crop_expand_ratio=crop_expand_ratio,
    )

    train_dataset = make_hf_dataset(train_records).map(
        builder,
        remove_columns=["sample_id", "image_path", "bbox_px", "label", "seq", "width", "height", "patient_id"]
    )
    val_dataset = make_hf_dataset(val_records).map(
        builder,
        remove_columns=["sample_id", "image_path", "bbox_px", "label", "seq", "width", "height", "patient_id"]
    )

    collator = VLDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    swanlab.init(
        project=swanlab_project,
        experiment_name=swanlab_experiment,
        config={
            "stage": task,
            "base_model": model_name_or_path,
            "train_json": train_json,
            "val_json": val_json,
            "output_dir": output_dir,
            "max_length": max_length,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "load_in_4bit": load_in_4bit,
            "gradient_checkpointing": gradient_checkpointing,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "crop_expand_ratio": crop_expand_ratio,
            "vertical_regions": VERTICAL_REGIONS,
            "train_records": len(train_records),
            "val_records": len(val_records),
        },
    )

    swanlab_callback = SwanLabCallback(
        project=swanlab_project,
        experiment_name=swanlab_experiment,
        config={
            "stage": task,
            "base_model": model_name_or_path,
            "train_records": len(train_records),
            "val_records": len(val_records),
            "vertical_regions": VERTICAL_REGIONS,
        },
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        bf16=False,
        fp16=not load_in_4bit,
        gradient_checkpointing=gradient_checkpointing,
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

    log(f"Start training stage: {task}")
    trainer.train()

    log("Saving final adapter ...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    log("Running stage evaluation ...")
    if task == "det":
        metrics, image_logs = evaluate_detector(model, processor, val_records)
    elif task == "cls":
        metrics, image_logs = evaluate_classifier(model, processor, val_records)
    else:
        raise ValueError(task)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    swanlab.log(metrics)
    if image_logs:
        swanlab.log({"Prediction": image_logs})
    swanlab.finish()


# =====================================
# Main
# =====================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train_det",
                        choices=["train_det", "train_cls", "eval_pipeline"])

    parser.add_argument("--base_model", type=str, default="/home/dwd/桌面/qwen_models/Qwen3.5-0.8B")

    parser.add_argument("--train_json", type=str,
                        default="/home/dwd/桌面/Spinal-qwen-finetune/datasets/train_output/data_detcls_vl.json")
    parser.add_argument("--val_json", type=str,
                        default="/home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_detcls_vl.json")

    parser.add_argument("--det_output_dir", type=str,
                        default="/home/dwd/桌面/Spinal-qwen-finetune/output/qwen35_stage1_det_vertical3")
    parser.add_argument("--cls_output_dir", type=str,
                        default="/home/dwd/桌面/Spinal-qwen-finetune/output/qwen35_stage2_cls")

    parser.add_argument("--det_adapter_path", type=str, default=None)
    parser.add_argument("--cls_adapter_path", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--eval_steps", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--crop_expand_ratio", type=float, default=0.1)

    parser.add_argument("--swanlab_project", type=str, default="Qwen3.5-VL-LoRA")
    parser.add_argument("--swanlab_experiment", type=str, default="two-stage-vertical3")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == "train_det":
        os.makedirs(args.det_output_dir, exist_ok=True)
        train_stage(
            task="det",
            model_name_or_path=args.base_model,
            train_json=args.train_json,
            val_json=args.val_json,
            output_dir=args.det_output_dir,
            max_length=args.max_length,
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
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=args.gradient_checkpointing,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            crop_expand_ratio=args.crop_expand_ratio,
            swanlab_project=args.swanlab_project,
            swanlab_experiment=f"{args.swanlab_experiment}-stage1-det",
        )

    elif args.mode == "train_cls":
        os.makedirs(args.cls_output_dir, exist_ok=True)
        train_stage(
            task="cls",
            model_name_or_path=args.base_model,
            train_json=args.train_json,
            val_json=args.val_json,
            output_dir=args.cls_output_dir,
            max_length=args.max_length,
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
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=args.gradient_checkpointing,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            crop_expand_ratio=args.crop_expand_ratio,
            swanlab_project=args.swanlab_project,
            swanlab_experiment=f"{args.swanlab_experiment}-stage2-cls",
        )

    elif args.mode == "eval_pipeline":
        if args.det_adapter_path is None:
            args.det_adapter_path = args.det_output_dir
        if args.cls_adapter_path is None:
            args.cls_adapter_path = args.cls_output_dir

        swanlab.init(
            project=args.swanlab_project,
            experiment_name=f"{args.swanlab_experiment}-pipeline-eval",
            config={
                "base_model": args.base_model,
                "val_json": args.val_json,
                "det_adapter_path": args.det_adapter_path,
                "cls_adapter_path": args.cls_adapter_path,
                "vertical_regions": VERTICAL_REGIONS,
            },
        )

        log("Loading validation records ...")
        val_records = load_records(args.val_json)

        log("Loading detector base model ...")
        det_base, det_processor = load_model_and_processor(
            model_name_or_path=args.base_model,
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=False,
        )
        det_model = PeftModel.from_pretrained(det_base, args.det_adapter_path)
        det_model.eval()

        log("Loading classifier base model ...")
        cls_base, cls_processor = load_model_and_processor(
            model_name_or_path=args.base_model,
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=False,
        )
        cls_model = PeftModel.from_pretrained(cls_base, args.cls_adapter_path)
        cls_model.eval()

        metrics, image_logs = evaluate_pipeline(
            det_model, det_processor, cls_model, cls_processor, val_records
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        swanlab.log(metrics)
        if image_logs:
            swanlab.log({"Prediction": image_logs})
        swanlab.finish()


if __name__ == "__main__":
    main()