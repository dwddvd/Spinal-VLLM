import mimetypes

import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

image_path = "/home/dwd/桌面/Spinal qwen-finetune/datasets/train_output/image/0000450553_1_0_cls_1_layer_2.jpg"

mime_type, _ = mimetypes.guess_type(image_path)
if mime_type is None:
    mime_type = "image/png"

with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text",
             "text": "You are a radiology assistant. Describe only visible imaging findings. Do not infer diagnosis unless strongly supported by the image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            }
        ]
    }
]

resp = client.chat.completions.create(
    model="/home/dwd/桌面/qwen_models/Qwen3.5-0.8B@main",
    messages=messages,
    max_tokens=512,  # 比之前大一点
    temperature=0.2,  # 医学场景建议低一点
    top_p=0.9,
)

print("finish_reason:", resp.choices[0].finish_reason)
print(resp.choices[0].message.content)
