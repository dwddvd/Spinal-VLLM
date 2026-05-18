import os.path
import random

import torch

from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json


def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径
    prompt = input_content.split("<|vision_start|>")[0]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


model_name = 'Qwen2.5-VL-7B-Instruct-AWQ'

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(f"/home/dwd/桌面/qwen/{model_name}", use_fast=False,
                                          trust_remote_code=True)
processor = AutoProcessor.from_pretrained(f"/home/dwd/桌面/qwen/{model_name}")

checkpoint_path = f"./output/{model_name}"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    f"/home/dwd/桌面/qwen/{model_name}",
    device_map="auto",
    # torch_dtype=torch.bfloat16,
    torch_dtype=torch.float16,
    trust_remote_code=True, )

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取json文件
# 拆分成训练集和测试集，保存为data_vl_train.json和data_vl_test.json
train_json_path = "/home/dwd/桌面/qwen-finetune/datasets/train_output/data_vl.json"
with open(train_json_path, 'r') as f:
    data = json.load(f)
    train_data = data

test_json_path = "/home/dwd/桌面/qwen-finetune/datasets/val_output/data_vl.json"

# swanlab.init()
# ====================训练模式===================
train_ds = Dataset.from_json(train_json_path)
train_dataset = train_ds.map(process_func)
print(train_dataset[0])
# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
peft_model = get_peft_model(model, config)

# 配置训练参数
args = TrainingArguments(
    output_dir=checkpoint_path,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project=f"{model_name}-finetune",
    experiment_name=f"{model_name}-bone-inf-tumor",
    config={
        "model": f"https://modelscope.cn/models/Qwen/{model_name}",
        "dataset": "Bone Infection Tumor",
        "github": "https://github.com/datawhalechina/self-llm",
        "prompt": '这是一幅脊椎的磁共振图像,序列为{mri_cls}。'
                  '其中病灶的位置按照[x1,y1,x2,y2]的格式在[{x1},{y1},{x2},{y2}]。'
                  '请你帮我判断这个病灶属于感染还是肿瘤。',
        "train_data_number": len(train_data),
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

# 开启模型训练
trainer.train()

# ====================测试模式===================
# 配置测试参数

val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)


checkpoint_name = sorted(os.listdir(checkpoint_path))[-1]
# 获取测试模型
val_peft_model = PeftModel.from_pretrained(model,
                                           model_id=f"{checkpoint_path}/{checkpoint_name}",
                                           config=val_config)

# 读取测试数据
with open(test_json_path, 'r') as f:
    test_dataset = json.load(f)

# 打乱列表
random.shuffle(test_dataset)

test_image_list = []
# 初始化统计变量
correct_count = 0  # 预测正确的数量
total_count = 0  # 总数量

patient_result_dict = {}
import tqdm

test_bar = tqdm.tqdm(test_dataset)
for item in test_bar:
    total_count += 1
    input_image_prompt = item["conversations"][0]["value"]
    # 去掉前后的<|vision_start|>和<|vision_end|>
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    prompt = input_image_prompt.split("<|vision_start|>")[0]

    patient_id = os.path.split(origin_image_path)[-1].split("_")[0]
    if patient_id not in patient_result_dict:
        patient_result_dict[patient_id] = {'total_count': 0, 'correct_count': 0}

    patient_result_dict[patient_id]['total_count'] += 1

    label = item["conversations"][1]["value"]
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": origin_image_path
            },
            {
                "type": "text",
                "text": prompt
            }
        ]}]

    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": f"{response}", 'label': label})
    if f"{response}" == label:
        patient_result_dict[patient_id]['correct_count'] += 1
        correct_count += 1
    # 计算准确率
    accuracy = correct_count / total_count
    test_bar.set_description(f'acc:{accuracy}')
    # print(messages[-1])

    test_image_list.append(swanlab.Image(origin_image_path, caption=response))

# # 遍历列表并比较 content 和 label
# for item in test_image_list:
#     if item['content'] == item['label']:
#         correct_count += 1

# 计算准确率
accuracy = correct_count / total_count

# 输出结果
print(f"单张图像预测正确的数量: {correct_count}")
print(f"单张图像总数量: {total_count}")
print(f"单张图像准确率: {accuracy:.2%}")

# 初始化统计变量
correct_ids = 0  # 判断正确的 id 数量
total_ids = len(patient_result_dict)  # 总 id 数量

# 遍历字典并计算每个 id 的准确率
for id, stats in patient_result_dict.items():
    correct_count = stats['correct_count']
    total_count = stats['total_count']

    # 计算准确率
    accuracy = correct_count / total_count

    # 判断是否算作正确
    if accuracy > 0.5:
        correct_ids += 1

# 计算整体准确率
overall_accuracy = correct_ids / total_ids
# 输出结果
print(f"判断正确的病例数量: {correct_ids}")
print(f"总病例数量: {total_ids}")
print(f"整体病例准确率: {overall_accuracy:.2%}")

swanlab.log({"Prediction": test_image_list})

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()
