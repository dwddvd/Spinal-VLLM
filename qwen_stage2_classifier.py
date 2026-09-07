import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments


INFECTION_ZH = "\u611f\u67d3"
TUMOR_ZH = "\u80bf\u7624"
INFECTION_MOJIBAKE = "\u93b0\u71b8\u714b"
TUMOR_MOJIBAKE = "\u9472\u8de8\u69eb"


PYCHARM_DEFAULTS = {
    "mode": "train",
    "base_model": "models/Qwen3.5-4B",
    "train_json": "data/internal/train.json",
    "val_json": "data/internal/validation.json",
    "output_dir": "outputs/qwen_stage2_adapter",
    "input_mode": "bbox_prompt",
    "load_in_4bit": True,
    "limit_train": 0,
    "limit_val": 0,
}


@dataclass
class LesionRecord:
    sample_id: str
    image_path: str
    bbox: Tuple[int, int, int, int]
    label: str
    width: int
    height: int
    patient_id: str
    seq: Optional[str] = None
    prompt_text: Optional[str] = None
    answer_text: Optional[str] = None
    case_id: Optional[str] = None
    slice_idx: Optional[int] = None


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def slugify_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_image_path(text: str) -> Optional[str]:
    match = re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", text, flags=re.S)
    return match.group(1).strip() if match else None


def extract_prompt_text(text: str) -> str:
    return text.split("<|vision_start|>")[0].strip()


def extract_bbox(text: str) -> Optional[Tuple[int, int, int, int]]:
    patterns = [
        r"bbox\s*[:\uFF1A]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r'"bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return tuple(map(int, match.groups()))
    return None


def normalize_label(text: str) -> Optional[str]:
    low = text.lower()
    if "infection" in low or INFECTION_ZH in text or INFECTION_MOJIBAKE in text:
        return "infection"
    if "tumor" in low or "tumour" in low or TUMOR_ZH in text or TUMOR_MOJIBAKE in text:
        return "tumor"
    return None


def label_to_zh(label: str) -> str:
    if label == "infection":
        return INFECTION_ZH
    if label == "tumor":
        return TUMOR_ZH
    raise ValueError(f"Unknown label: {label}")


def infer_sequence(text: str) -> Optional[str]:
    match = re.search(r"\b([Tt][12](?:WI)?)\b", text)
    return match.group(1).upper() if match else None


def clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def make_non_empty_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    if x2 <= x1:
        if x1 < width - 1:
            x2 = x1 + 1
        else:
            x1 = max(0, x1 - 1)
    if y2 <= y1:
        if y1 < height - 1:
            y2 = y1 + 1
        else:
            y1 = max(0, y1 - 1)
    return x1, y1, x2, y2


def expand_bbox(bbox: Tuple[int, int, int, int], width: int, height: int, ratio: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(round(box_w * ratio))
    pad_y = int(round(box_h * ratio))
    return make_non_empty_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def crop_image(image_path: str, bbox: Tuple[int, int, int, int], expand_ratio: float) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = expand_bbox(bbox, width, height, expand_ratio)
    return image.crop((x1, y1, x2, y2))


def load_records(json_path: str) -> List[LesionRecord]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a JSON list.")

    records: List[LesionRecord] = []
    skipped = 0
    for index, item in enumerate(data):
        convs = item.get("conversations", [])
        if len(convs) < 2:
            skipped += 1
            continue
        user_text = convs[0].get("value", "")
        assistant_text = convs[1].get("value", "")
        image_path = extract_image_path(user_text)
        bbox_value = item.get("bbox")
        bbox = tuple(bbox_value) if isinstance(bbox_value, list) and len(bbox_value) == 4 else None
        bbox = bbox or extract_bbox(assistant_text) or extract_bbox(user_text)
        label = item.get("label") or normalize_label(assistant_text)
        if image_path is None or bbox is None or label is None or not os.path.exists(image_path):
            skipped += 1
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        if width < 2 or height < 2:
            skipped += 1
            continue
        bbox = make_non_empty_bbox(bbox, width, height)
        sample_id = str(item.get("id", "")) or f"sample_{index}"
        patient_id = str(item.get("patient_id") or item.get("slice_manifest", {}).get("patient_id") or Path(image_path).name.split("_")[0])
        seq = item.get("seq") or item.get("slice_manifest", {}).get("seq") or infer_sequence(user_text)
        slice_idx = item.get("slice_idx", item.get("slice_manifest", {}).get("slice_idx"))
        try:
            slice_idx = int(slice_idx) if slice_idx is not None and str(slice_idx) != "" else None
        except (TypeError, ValueError):
            slice_idx = None
        records.append(
            LesionRecord(
                sample_id=sample_id,
                image_path=image_path,
                bbox=bbox,
                label=label,
                width=width,
                height=height,
                patient_id=patient_id,
                seq=seq,
                prompt_text=extract_prompt_text(user_text),
                answer_text=assistant_text.strip(),
                case_id=item.get("case_id") or item.get("slice_manifest", {}).get("case_id"),
                slice_idx=slice_idx,
            )
        )
    log(f"Loaded {len(records)} records from {json_path}; skipped {skipped}.")
    return records


def build_prompt(
    seq: Optional[str],
    bbox: Optional[Tuple[int, int, int, int]] = None,
    input_mode: str = "bbox_prompt",
    prompt_style: str = "clean",
) -> str:
    seq_text = f"\u5e8f\u5217\u4e3a{seq}\u3002" if seq else ""
    if input_mode == "bbox_prompt":
        if bbox is None:
            raise ValueError("bbox_prompt mode requires bbox.")
        x1, y1, x2, y2 = bbox
        if prompt_style == "legacy":
            return (
                f"\u73b0\u5728\u4f60\u662f\u4e00\u4e2a\u9aa8\u79d1\u4e13\u5bb6\uff0c"
                f"\u8fd9\u662f\u4e00\u5e45\u810a\u690e\u7684\u78c1\u5171\u632f\u56fe\u50cf\uff0c{seq_text}"
                f"\u8be5\u56fe\u50cf\u4e2d\u53ef\u80fd\u5305\u542b\u4e86\u6570\u4e2a\u75c5\u7076\uff0c"
                f"\u7136\u540e\u6211\u4f1a\u5c06\u6700\u5927\u75c5\u7076\u7684\u5750\u6807\u4f4d\u7f6e"
                f"\u6309\u7167[x1,y1,x2,y2]\u7684\u683c\u5f0f\u7ed9\u51fa\uff1a"
                f"\u8be5\u5f20\u56fe\u7247\u4e2d\u7684\u6700\u5927\u75c5\u7076\u5728[{x1},{y1},{x2},{y2}]\u8fd9\u4e2a\u4f4d\u7f6e\u3002"
                f"\u8bf7\u4f60\u5e2e\u6211\u5224\u65ad\u8fd9\u4e2a\u75c5\u7076\u5c5e\u4e8e{INFECTION_ZH}\u8fd8\u662f{TUMOR_ZH}\u3002"
            )
        return (
            f"\u8fd9\u662f\u4e00\u5e45\u810a\u690e\u7684\u78c1\u5171\u632f\u56fe\u50cf\uff0c{seq_text}"
            f"\u5176\u4e2d\u75c5\u7076\u7684\u4f4d\u7f6e\u6309\u7167[x1,y1,x2,y2]\u7684\u683c\u5f0f\u5728"
            f"[{x1},{y1},{x2},{y2}]\u3002"
            f"\u8bf7\u5224\u65ad\u8fd9\u4e2a\u75c5\u7076\u5c5e\u4e8e{INFECTION_ZH}\u8fd8\u662f{TUMOR_ZH}\u3002"
            f"\u53ea\u8f93\u51fa{INFECTION_ZH}\u6216{TUMOR_ZH}\u3002"
        )
    return (
        f"\u8fd9\u662f\u4e00\u5e45\u810a\u690e\u78c1\u5171\u632f\u56fe\u50cf\u4e2d\u7684\u75c5\u7076\u533a\u57df\uff0c{seq_text}"
        f"\u8bf7\u5224\u65ad\u8fd9\u4e2a\u75c5\u7076\u5c5e\u4e8e{INFECTION_ZH}\u8fd8\u662f{TUMOR_ZH}\u3002"
        f"\u53ea\u8f93\u51fa{INFECTION_ZH}\u6216{TUMOR_ZH}\u3002"
    )


def build_answer(label: str, answer_style: str = "short", original_answer: Optional[str] = None) -> str:
    if answer_style == "original" and original_answer:
        return original_answer
    if answer_style == "legacy":
        return f"\u8fd9\u4e2a\u75c5\u7076\u7684\u7c7b\u578b\u4e3a{label_to_zh(label)}\u3002"
    return label_to_zh(label)


def label_distribution(records: List[LesionRecord]) -> Dict[str, int]:
    counts = Counter(r.label for r in records)
    return {"infection": int(counts.get("infection", 0)), "tumor": int(counts.get("tumor", 0))}


def balance_records(records: List[LesionRecord], seed: int) -> List[LesionRecord]:
    by_label: Dict[str, List[LesionRecord]] = {"infection": [], "tumor": []}
    for record in records:
        if record.label in by_label:
            by_label[record.label].append(record)

    max_count = max(len(items) for items in by_label.values())
    rng = random.Random(seed)
    balanced: List[LesionRecord] = []
    for label, items in by_label.items():
        if not items:
            continue
        balanced.extend(items)
        needed = max_count - len(items)
        balanced.extend(rng.choice(items) for _ in range(needed))
    rng.shuffle(balanced)
    return balanced


class QwenCropDatasetBuilder:
    def __init__(
        self,
        processor,
        max_length: int,
        crop_expand_ratio: float,
        input_mode: str,
        image_resize: int,
        prompt_style: str,
        answer_style: str,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.crop_expand_ratio = crop_expand_ratio
        self.input_mode = input_mode
        self.image_resize = image_resize
        self.prompt_style = prompt_style
        self.answer_style = answer_style

    def __call__(self, example: Dict) -> Dict:
        bbox = tuple(example["bbox"])
        if self.input_mode == "bbox_prompt":
            image_content = {"type": "image", "image": example["image_path"]}
            if self.image_resize > 0:
                image_content["resized_height"] = self.image_resize
                image_content["resized_width"] = self.image_resize
            if self.prompt_style == "original" and example.get("prompt_text"):
                prompt = example["prompt_text"]
            else:
                style = "legacy" if self.prompt_style == "original" else self.prompt_style
                prompt = build_prompt(example.get("seq"), bbox=bbox, input_mode=self.input_mode, prompt_style=style)
        else:
            crop = crop_image(example["image_path"], bbox, self.crop_expand_ratio)
            image_content = {"type": "image", "image": crop}
            prompt = build_prompt(example.get("seq"), bbox=None, input_mode=self.input_mode, prompt_style=self.prompt_style)
        answer = build_answer(example["label"], self.answer_style, example.get("answer_text"))
        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
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


class QwenLazyDataset(torch.utils.data.Dataset):
    def __init__(self, records: List[LesionRecord], builder: QwenCropDatasetBuilder):
        self.records = records
        self.builder = builder

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        example = {
            "sample_id": record.sample_id,
            "image_path": record.image_path,
            "bbox": list(record.bbox),
            "label": record.label,
            "seq": record.seq,
            "prompt_text": record.prompt_text,
            "answer_text": record.answer_text,
        }
        return self.builder(example)


class VLDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    @staticmethod
    def to_tensor(value, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [self.to_tensor(f["input_ids"], torch.long) for f in features]
        attention_mask = [self.to_tensor(f["attention_mask"], torch.long) for f in features]
        labels = [self.to_tensor(f["labels"], torch.long) for f in features]
        pixel_values = [self.to_tensor(f["pixel_values"], torch.float32) for f in features]
        image_grid_thw = [self.to_tensor(f["image_grid_thw"], torch.long) for f in features]
        for value in pixel_values:
            if value.dim() != 2:
                raise ValueError(f"Expected pixel_values to be 2D [num_patches, hidden], got shape {tuple(value.shape)}")

        max_len = max(x.size(0) for x in input_ids)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ids, mask, label in zip(input_ids, attention_mask, labels):
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros((pad_len,), dtype=torch.long)])
                label = torch.cat([label, torch.full((pad_len,), -100, dtype=torch.long)])
            batch_input_ids.append(ids)
            batch_attention_mask.append(mask)
            batch_labels.append(label)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.stack(image_grid_thw),
        }


def load_model_and_processor(model_name_or_path: str, load_in_4bit: bool, gradient_checkpointing: bool):
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **kwargs)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, processor


def resolve_resume_checkpoint(value: Optional[str], output_dir: str) -> Optional[str]:
    if not value:
        return None

    if value.lower() != "auto":
        checkpoint = Path(value).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Resume checkpoint directory not found: {checkpoint}")
        return str(checkpoint)

    output_path = Path(output_dir)
    valid_checkpoints = []
    incomplete_checkpoints = []
    for checkpoint in output_path.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
        if not match or not checkpoint.is_dir():
            continue
        required_files = [
            checkpoint / "trainer_state.json",
            checkpoint / "optimizer.pt",
            checkpoint / "scheduler.pt",
            checkpoint / "adapter_config.json",
        ]
        has_adapter_weights = any(
            (checkpoint / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        )
        if all(path.is_file() for path in required_files) and has_adapter_weights:
            valid_checkpoints.append((int(match.group(1)), checkpoint))
        else:
            incomplete_checkpoints.append(checkpoint.name)

    if incomplete_checkpoints:
        log(
            "Ignoring incomplete checkpoints: "
            + ", ".join(sorted(incomplete_checkpoints))
        )
    if not valid_checkpoints:
        log(f"No complete checkpoint found in {output_path}; starting training from step 0.")
        return None

    _, checkpoint = max(valid_checkpoints, key=lambda item: item[0])
    log(f"Resuming training from complete checkpoint: {checkpoint}")
    return str(checkpoint)


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


def build_messages(prompt: str, image, image_resize: int = 0) -> List[Dict]:
    image_content = {"type": "image", "image": image}
    if image_resize > 0:
        image_content["resized_height"] = image_resize
        image_content["resized_width"] = image_resize
    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_text(model, processor, messages: List[Dict], max_new_tokens: int) -> str:
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
    model_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    prompt_len = model_inputs["input_ids"].shape[1]
    text = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text[0].strip()


def predict_label(
    model,
    processor,
    image,
    seq: Optional[str],
    bbox: Optional[Tuple[int, int, int, int]] = None,
    input_mode: str = "crop",
    max_new_tokens: int = 16,
    prompt_style: str = "clean",
    image_resize: int = 0,
) -> Tuple[str, Optional[str]]:
    text = generate_text(
        model,
        processor,
        build_messages(build_prompt(seq, bbox=bbox, input_mode=input_mode, prompt_style=prompt_style), image, image_resize),
        max_new_tokens=max_new_tokens,
    )
    return text, normalize_label(text)


def evaluate_gt(
    model,
    processor,
    records: List[LesionRecord],
    crop_expand_ratio: float,
    input_mode: str,
    max_new_tokens: int,
    prompt_style: str,
    image_resize: int,
    output_csv: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    correct = 0
    total = 0
    confusion = {"infection->infection": 0, "infection->tumor": 0, "tumor->infection": 0, "tumor->tumor": 0, "invalid": 0}
    rows = []
    progress = tqdm(records, desc="Eval Qwen GT crops", ncols=160) if show_progress else records
    for record in progress:
        if input_mode == "bbox_prompt":
            image = record.image_path
            bbox = record.bbox
        else:
            image = crop_image(record.image_path, record.bbox, crop_expand_ratio)
            bbox = None
        if prompt_style == "original" and record.prompt_text:
            text = generate_text(
                model,
                processor,
                build_messages(record.prompt_text, image, image_resize),
                max_new_tokens=max_new_tokens,
            )
            pred = normalize_label(text)
        else:
            text, pred = predict_label(
                model,
                processor,
                image,
                record.seq,
                bbox=bbox,
                input_mode=input_mode,
                max_new_tokens=max_new_tokens,
                prompt_style=prompt_style,
                image_resize=image_resize,
            )
        total += 1
        correct += int(pred == record.label)
        if pred in {"infection", "tumor"}:
            confusion[f"{record.label}->{pred}"] += 1
        else:
            confusion["invalid"] += 1
        preview = text.replace("\n", " ").replace("\r", " ").strip()
        if len(preview) > 36:
            preview = preview[:36] + "..."
        if show_progress:
            progress.set_postfix(
                {
                    "gt": record.label,
                    "pred": pred or "invalid",
                    "acc": f"{correct / max(total, 1):.4f}",
                    "invalid": confusion["invalid"],
                    "text": preview,
                },
                refresh=True,
            )
        if output_csv:
            x1, y1, x2, y2 = record.bbox
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "patient_id": record.patient_id,
                    "seq": record.seq or "",
                    "image_path": record.image_path,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "gt_label": record.label,
                    "pred_label": pred or "invalid",
                    "correct": int(pred == record.label),
                    "generated_text": text,
                    "prompt_style": prompt_style,
                }
            )
    metrics = {"cls_acc": correct / max(total, 1), "total": total}
    metrics.update(confusion)
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id",
            "patient_id",
            "seq",
            "image_path",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "gt_label",
            "pred_label",
            "correct",
            "generated_text",
            "prompt_style",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log(f"Saved per-sample predictions to {output_path}")
    return metrics


class MidEvalGTCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        val_records: List[LesionRecord],
        crop_expand_ratio: float,
        input_mode: str,
        max_new_tokens: int,
        prompt_style: str,
        image_resize: int,
        seed: int,
        max_samples: int,
        shuffle_records: bool,
    ) -> None:
        records = list(val_records)
        if shuffle_records:
            rng = random.Random(seed)
            rng.shuffle(records)
        if max_samples > 0:
            records = records[:max_samples]
        self.processor = processor
        self.eval_records = records
        self.crop_expand_ratio = crop_expand_ratio
        self.input_mode = input_mode
        self.max_new_tokens = max_new_tokens
        self.prompt_style = prompt_style
        self.image_resize = image_resize

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        metrics = kwargs.get("metrics")
        if model is None or not self.eval_records:
            return control
        was_training = model.training
        model.eval()
        eval_metrics = evaluate_gt(
            model,
            self.processor,
            self.eval_records,
            self.crop_expand_ratio,
            self.input_mode,
            self.max_new_tokens,
            self.prompt_style,
            self.image_resize,
            output_csv=None,
            show_progress=False,
        )
        prefixed = {
            "mid_eval_cls_acc": eval_metrics.get("cls_acc", 0.0),
            "mid_eval_total": eval_metrics.get("total", 0),
            "mid_eval_invalid": eval_metrics.get("invalid", 0),
            "mid_eval_inf_inf": eval_metrics.get("infection->infection", 0),
            "mid_eval_inf_tum": eval_metrics.get("infection->tumor", 0),
            "mid_eval_tum_inf": eval_metrics.get("tumor->infection", 0),
            "mid_eval_tum_tum": eval_metrics.get("tumor->tumor", 0),
        }
        if isinstance(metrics, dict):
            metrics.update(prefixed)
        log(
            f"Mid-eval step {state.global_step}: "
            f"cls_acc={prefixed['mid_eval_cls_acc']:.4f}, "
            f"invalid={prefixed['mid_eval_invalid']}, "
            f"total={prefixed['mid_eval_total']}"
        )
        if was_training:
            model.train()
        return control


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_records = load_records(args.train_json)
    val_records = load_records(args.val_json)
    if args.limit_train > 0:
        train_records = train_records[: args.limit_train]
        log(f"Debug limit applied to train records: {len(train_records)}")
    if args.limit_val > 0:
        val_records = val_records[: args.limit_val]
        log(f"Debug limit applied to val records: {len(val_records)}")
    log(f"Train label distribution before balancing: {label_distribution(train_records)}")
    log(f"Val label distribution: {label_distribution(val_records)}")
    if args.balance_train:
        train_records = balance_records(train_records, args.seed)
        log(f"Train label distribution after balancing: {label_distribution(train_records)}")

    model, processor = load_model_and_processor(args.base_model, args.load_in_4bit, args.gradient_checkpointing)
    if args.init_adapter_path:
        log(f"Initializing trainable LoRA adapter from {args.init_adapter_path}")
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)
        model.print_trainable_parameters()
    else:
        model = add_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    builder = QwenCropDatasetBuilder(
        processor,
        args.max_length,
        args.crop_expand_ratio,
        args.input_mode,
        args.image_resize,
        args.prompt_style,
        args.answer_style,
    )
    train_dataset = QwenLazyDataset(train_records, builder)
    val_dataset = QwenLazyDataset(val_records, builder)
    log("Using lazy preprocessing dataset; images are processed batch-by-batch during training.")

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
        callbacks=[
            MidEvalGTCallback(
                processor=processor,
                val_records=val_records,
                crop_expand_ratio=args.crop_expand_ratio,
                input_mode=args.input_mode,
                max_new_tokens=args.max_new_tokens,
                prompt_style=args.prompt_style,
                image_resize=args.image_resize,
                seed=args.seed,
                max_samples=args.mid_eval_max_samples,
                shuffle_records=args.mid_eval_shuffle,
            )
        ] if args.mid_eval_cls_metrics else None,
    )
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.skip_final_eval:
        log("Skipped post-training generative evaluation as requested; adapter and processor were saved.")
        return

    eval_tag = f"{args.input_mode}_eval_{args.prompt_style}_prompt"
    eval_slug = slugify_name(eval_tag)
    predictions_path = Path(args.output_dir) / "gt_crop_eval_predictions.csv"
    predictions_alias_path = Path(args.output_dir) / f"{eval_slug}_predictions.csv"
    metrics = evaluate_gt(
        model,
        processor,
        val_records,
        args.crop_expand_ratio,
        args.input_mode,
        args.max_new_tokens,
        args.prompt_style,
        args.image_resize,
        str(predictions_path),
    )
    metrics["eval_input_mode"] = args.input_mode
    metrics["eval_prompt_style"] = args.prompt_style
    metrics["eval_predictions_csv"] = str(predictions_path)
    metrics["eval_predictions_alias_csv"] = str(predictions_alias_path)
    metrics_path = Path(args.output_dir) / "gt_crop_eval_metrics.json"
    metrics_alias_path = Path(args.output_dir) / f"{eval_slug}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with metrics_alias_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as src, predictions_alias_path.open("w", encoding="utf-8") as dst:
            dst.write(src.read())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    log(f"Saved metrics to {metrics_path}")
    log(f"Saved alias metrics to {metrics_alias_path}")


def eval_gt(args: argparse.Namespace) -> None:
    records = load_records(args.val_json)
    if args.shuffle_eval:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        log(f"Shuffled eval records with seed={args.seed}.")
    base_model, processor = load_model_and_processor(args.base_model, args.load_in_4bit, False)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    metrics = evaluate_gt(
        model,
        processor,
        records,
        args.crop_expand_ratio,
        args.input_mode,
        args.max_new_tokens,
        args.prompt_style,
        args.image_resize,
        args.output_csv,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean Qwen crop classifier for spinal lesion infection/tumor classification.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--base_model", required=True)
    train_parser.add_argument("--train_json", required=True)
    train_parser.add_argument("--val_json", required=True)
    train_parser.add_argument("--output_dir", default="output/qwen_stage2_cls_gtbox")
    train_parser.add_argument("--init_adapter_path", default=None, help="Optional existing LoRA adapter to continue fine-tuning from.")
    train_parser.add_argument("--input_mode", choices=["bbox_prompt", "crop"], default="bbox_prompt")
    train_parser.add_argument("--prompt_style", choices=["clean", "legacy", "original"], default="clean")
    train_parser.add_argument("--answer_style", choices=["short", "legacy", "original"], default="short")
    train_parser.add_argument("--image_resize", type=int, default=280)
    train_parser.add_argument("--crop_expand_ratio", type=float, default=0.2)
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
    train_parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint directory to resume from, or 'auto' to use the latest complete checkpoint in output_dir.",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--limit_train", type=int, default=0)
    train_parser.add_argument("--limit_val", type=int, default=0)
    train_parser.add_argument("--balance_train", action="store_true", default=True)
    train_parser.add_argument("--no_balance_train", action="store_false", dest="balance_train")
    train_parser.add_argument("--load_in_4bit", action="store_true")
    train_parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    train_parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    train_parser.add_argument("--lora_r", type=int, default=64)
    train_parser.add_argument("--lora_alpha", type=int, default=16)
    train_parser.add_argument("--lora_dropout", type=float, default=0.05)
    train_parser.add_argument("--max_new_tokens", type=int, default=16)
    train_parser.add_argument("--mid_eval_cls_metrics", action="store_true", default=True)
    train_parser.add_argument("--no_mid_eval_cls_metrics", action="store_false", dest="mid_eval_cls_metrics")
    train_parser.add_argument("--mid_eval_max_samples", type=int, default=200)
    train_parser.add_argument("--mid_eval_shuffle", action="store_true", default=True)
    train_parser.add_argument("--no_mid_eval_shuffle", action="store_false", dest="mid_eval_shuffle")
    train_parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="Save the trained adapter without running the post-training generative validation pass.",
    )
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval_gt")
    eval_parser.add_argument("--base_model", required=True)
    eval_parser.add_argument("--adapter_path", required=True)
    eval_parser.add_argument("--val_json", required=True)
    eval_parser.add_argument("--input_mode", choices=["bbox_prompt", "crop"], default="bbox_prompt")
    eval_parser.add_argument("--prompt_style", choices=["clean", "legacy", "original"], default="clean")
    eval_parser.add_argument("--image_resize", type=int, default=280)
    eval_parser.add_argument("--crop_expand_ratio", type=float, default=0.2)
    eval_parser.add_argument("--load_in_4bit", action="store_true")
    eval_parser.add_argument("--max_new_tokens", type=int, default=16)
    eval_parser.add_argument("--output_csv", default=None)
    eval_parser.add_argument("--shuffle_eval", action="store_true")
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.set_defaults(func=eval_gt)

    return parser


def pycharm_default_argv() -> List[str]:
    argv = [
        PYCHARM_DEFAULTS["mode"],
        "--base_model",
        PYCHARM_DEFAULTS["base_model"],
        "--train_json",
        PYCHARM_DEFAULTS["train_json"],
        "--val_json",
        PYCHARM_DEFAULTS["val_json"],
        "--output_dir",
        PYCHARM_DEFAULTS["output_dir"],
        "--input_mode",
        PYCHARM_DEFAULTS["input_mode"],
    ]
    if PYCHARM_DEFAULTS.get("load_in_4bit", False):
        argv.append("--load_in_4bit")
    if PYCHARM_DEFAULTS.get("limit_train", 0) > 0:
        argv.extend(["--limit_train", str(PYCHARM_DEFAULTS["limit_train"])])
    if PYCHARM_DEFAULTS.get("limit_val", 0) > 0:
        argv.extend(["--limit_val", str(PYCHARM_DEFAULTS["limit_val"])])
    return argv


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        log("No command-line arguments detected; using PYCHARM_DEFAULTS for local debugging.")
        args = parser.parse_args(pycharm_default_argv())
    else:
        args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
