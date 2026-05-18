import os
import json
import argparse
from collections import defaultdict

import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText
import swanlab


def log(msg: str):
    print(f"[INFO] {msg}", flush=True)


def parse_record(example):
    conversation = example["conversations"]
    user_text = conversation[0]["value"]
    label = conversation[1]["value"].strip()

    image_path = user_text.split("<|vision_start|>")[1].split("<|vision_end|>")[0].strip()
    prompt = user_text.split("<|vision_start|>")[0].strip()

    return {
        "image_path": image_path,
        "prompt": prompt,
        "label": label,
    }


def build_messages(prompt, image_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def normalize_label(text: str):
    text = text.strip().lower()
    if ("感染" in text) or ("infection" in text):
        return "感染"
    if ("肿瘤" in text) or ("tumor" in text) or ("tumour" in text):
        return "肿瘤"
    return text.strip()


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


def evaluate_generation(model, processor, records, max_new_tokens=64, log_image_num=20):
    total_count = 0
    correct_count = 0
    patient_result_dict = defaultdict(lambda: {"total_count": 0, "correct_count": 0})
    image_logs = []

    pbar = tqdm(records, desc="Evaluating", ncols=120)

    for rec in pbar:
        image_path = rec["image_path"]
        prompt = rec["prompt"]
        label = rec["label"]

        try:
            pred_text = predict_one(
                model,
                processor,
                prompt,
                image_path,
                max_new_tokens=max_new_tokens
            )
        except Exception as e:
            pred_text = f"[ERROR] {repr(e)}"

        pred_norm = normalize_label(pred_text)
        label_norm = normalize_label(label)

        total_count += 1

        patient_id = os.path.basename(image_path).split("_")[0]
        patient_result_dict[patient_id]["total_count"] += 1

        if pred_norm == label_norm:
            correct_count += 1
            patient_result_dict[patient_id]["correct_count"] += 1

        image_acc = correct_count / total_count if total_count > 0 else 0.0

        pbar.set_postfix({
            "img_acc": f"{image_acc:.4f}",
            "pred": pred_norm[:12],
            "gt": label_norm[:12],
        })

        if len(image_logs) < log_image_num:
            try:
                image_logs.append(
                    swanlab.Image(
                        image_path,
                        caption=f"pred={pred_text} | pred_norm={pred_norm} | gt={label}"
                    )
                )
            except Exception:
                pass

    image_acc = correct_count / total_count if total_count > 0 else 0.0

    correct_ids = 0
    total_ids = len(patient_result_dict)
    for pid, stats in patient_result_dict.items():
        acc = stats["correct_count"] / stats["total_count"]
        if acc > 0.5:
            correct_ids += 1

    patient_acc = correct_ids / total_ids if total_ids > 0 else 0.0

    metrics = {
        "eval_gen_image_acc": image_acc,
        "eval_gen_patient_acc": patient_acc,
        "eval_gen_total_images": total_count,
        "eval_gen_total_patients": total_ids,
        "eval_gen_correct_images": correct_count,
        "eval_gen_correct_patients": correct_ids,
    }
    return metrics, image_logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--eval_json", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--swanlab_project", type=str, default="Qwen3.5-VL-LoRA")
    parser.add_argument("--swanlab_experiment", type=str, default="eval-only")
    args = parser.parse_args()

    # ===== PyCharm 直接运行时的默认参数 =====
    if args.base_model is None:
        args.base_model = "/home/dwd/桌面/qwen_models/Qwen3.5-0.8B"
    if args.adapter_path is None:
        args.adapter_path = "/home/dwd/桌面/Spinal-qwen-finetune/output/qwen35_lora_spine/checkpoint-3378"
    if args.eval_json is None:
        args.eval_json = "/home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_vl.json"

    log("Initializing SwanLab...")
    swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
        config=vars(args),
    )

    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    log("Loading base model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    log("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    try:
        emb_device = model.get_input_embeddings().weight.device
        log(f"Model embedding device: {emb_device}")
    except Exception as e:
        log(f"Cannot inspect embedding device: {repr(e)}")

    log(f"Loading eval json: {args.eval_json}")
    with open(args.eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    log(f"Loaded eval samples: {len(data)}")
    records = [parse_record(x) for x in data]

    log("Starting generation evaluation...")
    metrics, image_logs = evaluate_generation(
        model,
        processor,
        records,
        max_new_tokens=args.max_new_tokens,
    )

    log(f"Evaluation done. Metrics: {metrics}")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    swanlab.log(metrics)
    if len(image_logs) > 0:
        swanlab.log({"Prediction": image_logs})

    swanlab.finish()
    log("Finished.")


if __name__ == "__main__":
    main()