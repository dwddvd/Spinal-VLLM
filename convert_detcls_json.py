import os
import re
import json
import argparse
from typing import Optional, Tuple, List, Dict


def extract_image_path(user_text: str) -> Optional[str]:
    """
    从 <|vision_start|> ... <|vision_end|> 中提取图像路径
    """
    m = re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", user_text, flags=re.S)
    if not m:
        return None
    return m.group(1).strip()


def extract_bbox(user_text: str) -> Optional[Tuple[int, int, int, int]]:
    """
    从原始 prompt 中提取病灶框，例如:
    [174,296,303,398]
    """
    patterns = [
        r"最大病灶.*?\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r"病灶.*?位置.*?\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r"bbox\s*[:：]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
    ]

    for p in patterns:
        m = re.search(p, user_text, flags=re.S | re.I)
        if m:
            x1, y1, x2, y2 = map(int, m.groups())
            return x1, y1, x2, y2
    return None


def extract_label(assistant_text: str) -> Optional[str]:
    """
    从 assistant 文本中提取类别：感染 / 肿瘤
    """
    text = assistant_text.strip().lower()

    if "感染" in assistant_text or "infection" in text:
        return "感染"
    if "肿瘤" in assistant_text or "tumor" in text or "tumour" in text:
        return "肿瘤"
    return None


def extract_mri_sequence(user_text: str) -> Optional[str]:
    """
    尝试提取 MRI 序列，比如 T1 / T2 / T1WI / T2WI
    """
    patterns = [
        r"序列为\s*([Tt][12](?:WI)?)",
        r"([Tt][12](?:WI)?)\s*序列",
        r"\b([Tt][12](?:WI)?)\b",
    ]
    for p in patterns:
        m = re.search(p, user_text)
        if m:
            return m.group(1).upper()
    return None


def build_new_prompt(image_path: str, mri_seq: Optional[str] = None) -> str:
    """
    构造新的 user prompt：不给框，只给图像，让模型输出位置+类别
    """
    seq_text = f"，序列为{mri_seq}" if mri_seq else ""

    prompt = (
        f"现在你是一个骨科专家，这是一幅脊椎的磁共振图像{seq_text}。"
        f"请观察整张图像，找出图中最主要病灶的大概位置，并判断该病灶属于感染还是肿瘤。"
        f"请严格按照以下格式回答：bbox:[x1,y1,x2,y2],label:感染/肿瘤 "
        f"<|vision_start|>{image_path}<|vision_end|>"
    )
    return prompt


def build_new_answer(bbox: Tuple[int, int, int, int], label: str) -> str:
    x1, y1, x2, y2 = bbox
    return f"bbox:[{x1},{y1},{x2},{y2}],label:{label}"


def convert_record(item: Dict, keep_original_id: bool = True) -> Optional[Dict]:
    """
    将单条原始样本转换为新的检测+分类样本
    """
    if "conversations" not in item or len(item["conversations"]) < 2:
        return None

    user_turn = item["conversations"][0]
    assistant_turn = item["conversations"][1]

    user_text = user_turn.get("value", "")
    assistant_text = assistant_turn.get("value", "")

    image_path = extract_image_path(user_text)
    bbox = extract_bbox(user_text)
    label = extract_label(assistant_text)
    mri_seq = extract_mri_sequence(user_text)

    if image_path is None or bbox is None or label is None:
        return None

    new_item = {
        "id": item.get("id") if keep_original_id else None,
        "conversations": [
            {
                "from": "user",
                "value": build_new_prompt(image_path, mri_seq),
            },
            {
                "from": "assistant",
                "value": build_new_answer(bbox, label),
            },
        ],
    }

    if new_item["id"] is None:
        new_item["id"] = ""

    return new_item


def convert_dataset(
    input_path: str,
    output_path: str,
    keep_original_id: bool = True,
    check_image_exists: bool = False,
):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是 list。")

    converted: List[Dict] = []
    skipped = []

    for idx, item in enumerate(data):
        new_item = convert_record(item, keep_original_id=keep_original_id)

        if new_item is None:
            skipped.append({"index": idx, "reason": "missing image_path / bbox / label"})
            continue

        if check_image_exists:
            user_text = new_item["conversations"][0]["value"]
            image_path = extract_image_path(user_text)
            if image_path is None or (not os.path.exists(image_path)):
                skipped.append({"index": idx, "reason": f"image not found: {image_path}"})
                continue

        if not new_item["id"]:
            new_item["id"] = f"sample_{len(converted) + 1}"

        converted.append(new_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 输入样本数: {len(data)}")
    print(f"[INFO] 成功转换: {len(converted)}")
    print(f"[INFO] 跳过样本: {len(skipped)}")
    print(f"[INFO] 输出文件: {output_path}")

    if skipped:
        skipped_path = os.path.splitext(output_path)[0] + "_skipped.json"
        with open(skipped_path, "w", encoding="utf-8") as f:
            json.dump(skipped, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 跳过记录已保存到: {skipped_path}")


def main():
    parser = argparse.ArgumentParser(description="将原始纯分类 MRI VLM JSON 转换为 检测+分类 JSON")
    parser.add_argument("--input_json", type=str, default=None, help="原始 json 路径")
    parser.add_argument("--output_json", type=str, default=None, help="输出新 json 路径")
    parser.add_argument("--check_image_exists", action="store_true", help="是否检查图像路径存在")
    parser.add_argument("--new_id", action="store_true", help="不保留原始 id，重新生成 id")
    args = parser.parse_args()

    # ===== 如果你是直接在 PyCharm 里点运行，没有传参，就自动补默认值 =====
    if args.input_json is None:
        args.input_json = "/home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_vl.json"
    if args.output_json is None:
        args.output_json = "/home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_detcls_vl.json"

    convert_dataset(
        input_path=args.input_json,
        output_path=args.output_json,
        keep_original_id=not args.new_id,
        check_image_exists=args.check_image_exists,
    )


if __name__ == "__main__":
    main()