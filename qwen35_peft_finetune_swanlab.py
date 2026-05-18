import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import swanlab
from swanlab.integration.transformers import SwanLabCallback

VISION_RE = re.compile(r"<\|vision_start\|>(.*?)<\|vision_end\|>", re.DOTALL)


def parse_sample(item: Dict[str, Any]) -> Dict[str, str]:
    conversations = item["conversations"]
    user_msg = None
    assistant_msg = None
    for turn in conversations:
        if turn.get("from") == "user" and user_msg is None:
            user_msg = turn.get("value", "")
        elif turn.get("from") == "assistant" and assistant_msg is None:
            assistant_msg = turn.get("value", "")

    if user_msg is None or assistant_msg is None:
        raise ValueError(f"Bad sample: missing user/assistant turn in item {item.get('id')}")

    m = VISION_RE.search(user_msg)
    if m is None:
        raise ValueError(f"Bad sample: missing vision tags in item {item.get('id')}")

    image_path = m.group(1).strip()
    prompt_text = VISION_RE.sub("", user_msg).strip()

    return {
        "id": item.get("id", ""),
        "image_path": image_path,
        "prompt": prompt_text,
        "answer": assistant_msg.strip(),
    }


class SpineVLJsonDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            records: List[Dict[str, str]],
            processor,
            image_root: Optional[str] = None,
            max_length: int = 2048,
    ):
        self.records = records
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_root = image_root
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.records)

    def _resolve_image_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        if self.image_root is None:
            return p
        return os.path.join(self.image_root, p)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image_path = self._resolve_image_path(rec["image_path"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": rec["prompt"]},
                    {"type": "image", "image": image},
                ],
            }
        ]

        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": rec["prompt"]},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": rec["answer"]},
                ],
            },
        ]

        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        full_inputs = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        prompt_len = prompt_inputs["input_ids"].shape[1]

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            prompt_len = min(prompt_len, self.max_length)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        for k in ["pixel_values", "image_grid_thw", "pixel_attention_mask"]:
            if k in full_inputs:
                out[k] = full_inputs[k]

        return out


@dataclass
class QwenVLDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenizer = self.processor.tokenizer

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for x in labels:
            if x.shape[0] < max_len:
                pad = torch.full((max_len - x.shape[0],), -100, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            padded_labels.append(x)
        batch["labels"] = torch.stack(padded_labels, dim=0)

        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)
        if "pixel_attention_mask" in features[0]:
            batch["pixel_attention_mask"] = torch.cat([f["pixel_attention_mask"] for f in features], dim=0)

        return batch


def train_val_split(records: List[Dict[str, str]], val_ratio: float = 0.05, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(records), generator=g).tolist()
    n_val = max(1, int(len(records) * val_ratio)) if len(records) > 10 else 0
    val_idx = set(indices[:n_val])
    train_records = [records[i] for i in range(len(records)) if i not in val_idx]
    val_records = [records[i] for i in range(len(records)) if i in val_idx]
    return train_records, val_records


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["lm_head"],
    )


def load_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if args.load_in_4bit else None,
        quantization_config=quantization_config,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def resolve_image_path(p: str, image_root: Optional[str]) -> str:
    if os.path.isabs(p):
        return p
    if image_root is None:
        return p
    return os.path.join(image_root, p)


def build_messages(prompt: str, image_path: str):
    image = Image.open(image_path).convert("RGB")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
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

    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def normalize_label(text: str) -> str:
    t = (text or "").strip().lower()
    if "感染" in t or "infection" in t:
        return "感染"
    if "肿瘤" in t or "tumor" in t or "tumour" in t:
        return "肿瘤"
    return (text or "").strip()


def patient_id_from_path(image_path: str) -> str:
    base = os.path.basename(image_path)
    return base.split("_")[0] if "_" in base else os.path.splitext(base)[0]


def evaluate_generation(model, processor, records: List[Dict[str, str]], image_root: Optional[str],
                        max_new_tokens: int = 64, max_log_images: int = 50) -> Tuple[Dict[str, float], List[Any]]:
    model.eval()
    total = 0
    correct = 0
    patient_stats: Dict[str, Dict[str, int]] = {}
    logged_images = []

    for rec in records:
        image_path = resolve_image_path(rec["image_path"], image_root)
        pred_text = predict_one(model, processor, rec["prompt"], image_path, max_new_tokens=max_new_tokens)
        pred_label = normalize_label(pred_text)
        true_label = normalize_label(rec["answer"])

        total += 1
        is_correct = int(pred_label == true_label)
        correct += is_correct

        pid = patient_id_from_path(image_path)
        if pid not in patient_stats:
            patient_stats[pid] = {"total": 0, "correct": 0}
        patient_stats[pid]["total"] += 1
        patient_stats[pid]["correct"] += is_correct

        if len(logged_images) < max_log_images:
            caption = f"pred={pred_text} | pred_norm={pred_label} | gt={rec['answer']} | gt_norm={true_label}"
            logged_images.append(swanlab.Image(image_path, caption=caption))

    image_acc = (correct / total) if total > 0 else 0.0

    correct_patients = 0
    for _, stats in patient_stats.items():
        if stats["total"] > 0 and (stats["correct"] / stats["total"]) > 0.5:
            correct_patients += 1
    patient_acc = (correct_patients / len(patient_stats)) if patient_stats else 0.0

    metrics = {
        "eval_gen_image_acc": image_acc,
        "eval_gen_patient_acc": patient_acc,
        "eval_gen_num_images": float(total),
        "eval_gen_num_patients": float(len(patient_stats)),
    }
    return metrics, logged_images


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--eval_json", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--swanlab_project", type=str, default="Qwen3.5-VL-LoRA")
    parser.add_argument("--swanlab_experiment", type=str, default="spine-vl-finetune")
    parser.add_argument("--eval_gen_max_new_tokens", type=int, default=64)

    args = parser.parse_args()

    # ===== 如果你是直接在 PyCharm 里点运行，没有传参，就自动补默认值 =====
    if args.model_name_or_path is None:
        args.model_name_or_path = "/home/dwd/桌面/qwen_models/Qwen3.5-0.8B"
    if args.train_json is None:
        args.train_json = "/home/dwd/桌面/Spinal-qwen-finetune/datasets/train_output/data_vl.json"
    if args.eval_json is None:
        args.eval_json = "/home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_vl.json"
    if args.output_dir is None:
        args.output_dir = "/home/dwd/桌面/Spinal-qwen-finetune/output/qwen35_lora_spine"

    # 调试时默认打开
    if not args.load_in_4bit:
        args.load_in_4bit = True
    if not args.gradient_checkpointing:
        args.gradient_checkpointing = True

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    train_records = [parse_sample(x) for x in train_data]

    with open(args.eval_json, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    val_records = [parse_sample(x) for x in eval_data]
    # train_records, val_records = train_val_split(records, val_ratio=args.val_ratio, seed=args.seed)

    print(f"Total samples: {len(train_records) + len(val_records)}")
    print(f"Train samples: {len(train_records)}")
    print(f"Val samples: {len(val_records)}")

    if train_records:
        print("Example parsed sample:")
        print(train_records[0])

    model, processor = load_model_and_processor(args)

    train_dataset = SpineVLJsonDataset(
        train_records,
        processor=processor,
        image_root=args.image_root,
        max_length=args.max_length,
    )
    eval_dataset = SpineVLJsonDataset(
        val_records,
        processor=processor,
        image_root=args.image_root,
        max_length=args.max_length,
    ) if len(val_records) > 0 else None

    data_collator = QwenVLDataCollator(processor=processor)

    bf16_flag = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    swanlab_callback = SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
        config={
            "model_name_or_path": args.model_name_or_path,
            "train_json": args.train_json,
            "output_dir": args.output_dir,
            "num_total_samples": len(train_records) + len(val_records),
            "num_train_samples": len(train_records),
            "num_val_samples": len(val_records),
            "max_length": args.max_length,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "load_in_4bit": args.load_in_4bit,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "val_ratio": args.val_ratio,
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
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=bf16_flag,
        fp16=(torch.cuda.is_available() and not bf16_flag),
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        logging_strategy="steps",
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
    )

    trainer.train()

    if eval_dataset is not None:
        loss_metrics = trainer.evaluate()
        print("Loss-based eval metrics:", loss_metrics)
        swanlab.log(loss_metrics)

        gen_metrics, image_logs = evaluate_generation(
            trainer.model,
            processor,
            val_records,
            image_root=args.image_root,
            max_new_tokens=args.eval_gen_max_new_tokens,
        )
        print("Generation-based eval metrics:", gen_metrics)
        swanlab.log(gen_metrics)
        if image_logs:
            swanlab.log({"eval_predictions": image_logs})

    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved adapter and processor to: {args.output_dir}")

    swanlab.finish()


if __name__ == "__main__":
    main()
