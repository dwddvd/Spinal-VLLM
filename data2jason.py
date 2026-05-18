# 导入所需的库
import json

import numpy as np
from modelscope.msdatasets import MsDataset
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from imblearn.over_sampling import RandomOverSampler
from scipy.ndimage import label
import tqdm

def convert_mri_cls(label):
    if label == '2':
        return 'T2'
    elif label == '1':
        return 'T1'
    elif label == '0':
        return 'CT'


def split_train_test(folder_path, test_size):
    # 获取所有npz文件
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

    # 提取病人ID（假设文件名前10位是病人ID）
    patient_ids = sorted(list(set([f[:10] for f in file_list])))  # 去重并排序

    # 按90%训练集和10%验证集划分病人ID
    train_ids, val_ids = train_test_split(patient_ids, test_size=test_size, random_state=42, shuffle=False)

    # 根据病人ID分配文件到训练集和验证集
    train_files = [os.path.join(folder_path, f) for f in file_list if f[:10] in train_ids]
    val_files = [os.path.join(folder_path, f) for f in file_list if f[:10] in val_ids]

    return train_files, val_files


def process_inf(folder_path, save_path):
    image_paths_list = []
    instruction_list = []
    captions_list = []
    cls = 0

    folder_bar = tqdm.tqdm(folder_path)
    # 遍历文件夹中的所有npz文件
    for file_path in folder_bar:
        if file_path.endswith('.npz'):
            filename = os.path.basename(file_path).split('.')[0]
            mri_cls = convert_mri_cls(filename.split('_')[1])

            # 加载npz文件
            data = np.load(file_path)
            image = data['image']  # 假设image的键是'image'
            mask = data['mask']  # 假设mask的键是'mask'

            try:
                assert image.shape[0] == mask.shape[0], f"{filename}'s image and mask shapes do not match"
            except AssertionError:
                print(f"{filename}'s image and mask shape do not match")
                continue

            # 遍历第一个维度（层）
            for i in range(image.shape[0]):
                layer_image = image[i]
                layer_mask = mask[i]

                # 检查mask是否有病灶（即是否有非零值）
                if np.any(layer_mask):
                    # 使用连通性检测找出不同的肿瘤区域
                    labeled_mask, num_features = label(layer_mask)

                    # # 如果没有连通区域，跳过
                    # if num_features > 1:
                    #     print('Find!')

                    # 找到每个连通区域的大小
                    region_sizes = [np.sum(labeled_mask == label_idx) for label_idx in range(1, num_features + 1)]

                    # 选择最大的连通区域
                    largest_region_label = np.argmax(region_sizes) + 1  # +1 因为标签从1开始
                    largest_region_mask = (labeled_mask == largest_region_label)

                    # 找到最大连通区域的边界框
                    rows = np.any(largest_region_mask, axis=1)
                    cols = np.any(largest_region_mask, axis=0)
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]

                    # 保存image层为jpg图像
                    img = Image.fromarray(layer_image)
                    image_path = os.path.join(save_path, f'{filename}_cls_{cls}_layer_{i}.jpg')
                    img.save(image_path)

                    instruction = (f'现在你是一个骨科专家，这是一幅脊椎的磁共振图像,序列为{mri_cls}。'
                                   f'该图像中可能包含了数个病灶，然后我会将最大病灶的坐标位置按照[x1,y1,x2,y2]的格式给出：'
                                   f'该张图片中的最大病灶在[{x1},{y1},{x2},{y2}]这个位置。'
                                   f'请你帮我判断这个病灶属于感染还是肿瘤。')
                    caption = '这个病灶的类型为感染。'

                    image_paths_list.append(image_path)
                    instruction_list.append(instruction)
                    captions_list.append(caption)

    return image_paths_list, instruction_list, captions_list

def process_tumor(folder_path, save_path):
    image_paths_list = []
    instruction_list = []
    captions_list = []
    cls = 1

    folder_bar = tqdm.tqdm(folder_path)
    # 遍历文件夹中的所有npz文件
    for file_path in folder_bar:
        if file_path.endswith('.npz'):
            filename = os.path.basename(file_path).split('.')[0]
            if filename.split('_')[1] not in ['1', '2']:
                continue
            mri_cls = convert_mri_cls(filename.split('_')[1])

            # 加载npz文件
            data = np.load(file_path)
            image = data['image']  # 假设image的键是'image'
            mask = data['mask']  # 假设mask的键是'mask'

            try:
                assert image.shape[0] == mask.shape[0], f"{filename}'s image and mask shapes do not match"
            except AssertionError:
                print(f"{filename}'s image and mask shape do not match")
                continue

            # 遍历第一个维度（层）
            for i in range(image.shape[0]):
                layer_image = image[i]
                layer_mask = mask[i]

                # 检查mask是否有病灶（即是否有非零值）
                if np.any(layer_mask):
                    # 使用连通性检测找出不同的肿瘤区域
                    labeled_mask, num_features = label(layer_mask)

                    # # 如果没有连通区域，跳过
                    # if num_features >1:
                    #     print('Find!')

                    # 找到每个连通区域的大小
                    region_sizes = [np.sum(labeled_mask == label_idx) for label_idx in range(1, num_features + 1)]

                    # 选择最大的连通区域
                    largest_region_label = np.argmax(region_sizes) + 1  # +1 因为标签从1开始
                    largest_region_mask = (labeled_mask == largest_region_label)

                    # 找到最大连通区域的边界框
                    rows = np.any(largest_region_mask, axis=1)
                    cols = np.any(largest_region_mask, axis=0)
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]

                    # 保存image层为jpg图像
                    img = Image.fromarray(layer_image)
                    image_path = os.path.join(save_path, f'{filename}_cls_{cls}_layer_{i}.jpg')
                    img.save(image_path)

                    # 生成instruction和caption
                    instruction = (f'这是一幅脊椎的磁共振图像,序列为{mri_cls}。'
                                   f'其中病灶的位置按照[x1,y1,x2,y2]的格式在[{x1},{y1},{x2},{y2}]。'
                                   f'请你帮我判断这个病灶属于感染还是肿瘤。')
                    caption = '这个病灶的类型为肿瘤。'

                    # 添加到结果列表
                    image_paths_list.append(image_path)
                    instruction_list.append(instruction)
                    captions_list.append(caption)

    return image_paths_list, instruction_list, captions_list

infection_root = '/media/dwd/本地项目/Lab/Bone/dataset/infection dataset/npz/'
tumor_root = '/media/dwd/本地项目/Lab/Bone/dataset/tumor'
dataset_root = '/home/dwd/桌面/qwen-finetune/datasets'

# 定义输出文件夹
output_train_dir = os.path.join(dataset_root, 'train_output')
output_val_dir = os.path.join(dataset_root, 'val_output')
# 定义image输出文件夹
output_train_image_dir = os.path.join(output_train_dir, 'image')
output_val_image_dir = os.path.join(output_val_dir, 'image')

# 创建输出文件夹（如果不存在）
os.makedirs(output_train_image_dir, exist_ok=True)
os.makedirs(output_val_image_dir, exist_ok=True)

test_size = 0.1

val_image_paths_list = []
val_instruction_list = []
val_captions_list = []

train_files, val_files = split_train_test(infection_root, test_size)

inf_image_paths_list, inf_instruction_list, inf_captions_list = process_inf(train_files, output_train_image_dir)

image_paths_list, instruction_list, captions_list = process_inf(val_files, output_val_image_dir)
val_image_paths_list.extend(image_paths_list)
val_instruction_list.extend(instruction_list)
val_captions_list.extend(captions_list)


train_files, val_files = split_train_test(tumor_root, test_size)

tumor_image_paths_list, tumor_instruction_list, tumor_captions_list = process_tumor(train_files, output_train_image_dir)

image_paths_list, instruction_list, captions_list = process_tumor(val_files, output_val_image_dir)
val_image_paths_list.extend(image_paths_list)
val_instruction_list.extend(instruction_list)
val_captions_list.extend(captions_list)

# 合并数据
train_image_paths_list = inf_image_paths_list + tumor_image_paths_list
train_instruction_list = inf_instruction_list + tumor_instruction_list
train_captions_list = inf_captions_list + tumor_captions_list

# 创建标签列表（inf 为 0，tumor 为 1）
labels = [0] * len(inf_image_paths_list) + [1] * len(tumor_image_paths_list)

# 创建索引列表
indices = np.arange(len(labels)).reshape(-1, 1)

# 使用 RandomOverSampler 进行过采样
oversampler = RandomOverSampler(random_state=42)
indices_resampled, labels_resampled = oversampler.fit_resample(indices, labels)

# 根据过采样后的索引重建平衡的数据集
balanced_image_paths_list = [train_image_paths_list[idx[0]] for idx in indices_resampled]
balanced_instruction_list = [train_instruction_list[idx[0]] for idx in indices_resampled]
balanced_captions_list = [train_captions_list[idx[0]] for idx in indices_resampled]


# 将图片路径和描述保存为CSV文件
train_df = pd.DataFrame({
    'image_path': balanced_image_paths_list,
    'instruction': balanced_instruction_list,
    'caption': balanced_captions_list
})

train_conversations = []

# 添加对话数据
for i in range(len(train_df)):
    train_conversations.append({
        "id": f"identity_{i + 1}",
        "conversations": [
            {
                "from": "user",
                "value": f"{train_df.iloc[i]['instruction']} <|vision_start|>{train_df.iloc[i]['image_path']}<|vision_end|>"
            },
            {
                "from": "assistant",
                "value": train_df.iloc[i]['caption']
            }
        ]
    })

# 保存为Json
with open('/home/dwd/桌面/qwen-finetune/datasets/train_output/data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(train_conversations, f , ensure_ascii=False, indent=2)


# 将图片路径和描述保存为CSV文件
val_df = pd.DataFrame({
    'image_path': val_image_paths_list,
    'instruction': val_instruction_list,
    'caption': val_captions_list
})

val_conversations = []

# 添加对话数据
for i in range(len(val_df)):
    val_conversations.append({
        "id": f"identity_{i + 1}",
        "conversations": [
            {
                "from": "user",
                "value": f"{val_df.iloc[i]['instruction']} <|vision_start|>{val_df.iloc[i]['image_path']}<|vision_end|>"
            },
            {
                "from": "assistant",
                "value": val_df.iloc[i]['caption']
            }
        ]
    })

# 保存为Json
with open('/home/dwd/桌面/qwen-finetune/datasets/val_output/data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(val_conversations, f, ensure_ascii=False, indent=2)
