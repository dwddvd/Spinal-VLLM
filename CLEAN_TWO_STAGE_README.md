# Clean Two-Stage Training Flow

Recommended paper pipeline:

```text
YOLOv8x detects lesion boxes -> Qwen classifies cropped lesion regions -> joint metrics
```

## 1. Train Qwen On GT Crops

```bash
python qwen_stage2_classifier.py train \
  --base_model /home/dwd/桌面/qwen_models/Qwen3.5-0.8B \
  --train_json /home/dwd/桌面/Spinal-qwen-finetune/datasets/train_output/data_detcls_vl.json \
  --val_json /home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_detcls_vl.json \
  --output_dir /home/dwd/桌面/Spinal-qwen-finetune/output/qwen_stage2_cls_gtbox \
  --crop_expand_ratio 0.2 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --load_in_4bit
```

This trains only the infection/tumor crop classifier. It does not do detection.

## 2. Re-evaluate Qwen On GT Crops

```bash
python qwen_stage2_classifier.py eval_gt \
  --base_model /home/dwd/桌面/qwen_models/Qwen3.5-0.8B \
  --adapter_path /home/dwd/桌面/Spinal-qwen-finetune/output/qwen_stage2_cls_gtbox \
  --val_json /home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_detcls_vl.json \
  --crop_expand_ratio 0.2 \
  --load_in_4bit
```

Expected key metric:

```text
cls_acc
```

## 3. Export YOLO Top-k Boxes

```bash
python yolo_stage1_detection.py predict \
  --weights /home/dwd/桌面/Spinal-qwen-finetune/runs/detect/output/yolo_stage1/yolov8x_img1280_lesion-2/weights/best.pt \
  --source /home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/image \
  --imgsz 1280 \
  --conf 0.01 \
  --iou 0.5 \
  --max_det 10 \
  --top_k 5 \
  --output_csv /home/dwd/桌面/Spinal-qwen-finetune/output/yolo_stage1/val_yolov8x1280_top5.csv
```

## 4. Evaluate YOLO + Qwen Pipeline

```bash
python yolo_qwen_pipeline.py \
  --base_model /home/dwd/桌面/qwen_models/Qwen3.5-0.8B \
  --adapter_path /home/dwd/桌面/Spinal-qwen-finetune/output/qwen_stage2_cls_gtbox \
  --val_json /home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_detcls_vl.json \
  --pred_csv /home/dwd/桌面/Spinal-qwen-finetune/output/yolo_stage1/val_yolov8x1280_top5.csv \
  --output_dir /home/dwd/桌面/Spinal-qwen-finetune/output/yolo_qwen_pipeline/yolov8x1280_top5_qwen \
  --top_k 5 \
  --selection top1 \
  --crop_expand_ratio 0.2 \
  --load_in_4bit
```

Main paper metrics:

```text
selected_det_recall_iou0.3
selected_det_recall_iou0.5
cls_acc_on_selected
joint_acc_iou0.3
joint_acc_iou0.5
patient_joint_acc_iou0.3
patient_joint_acc_iou0.5
```

For an upper-bound analysis of top-k detector quality, run:

```bash
python yolo_qwen_pipeline.py \
  --base_model /home/dwd/桌面/qwen_models/Qwen3.5-0.8B \
  --adapter_path /home/dwd/桌面/Spinal-qwen-finetune/output/qwen_stage2_cls_gtbox \
  --val_json /home/dwd/桌面/Spinal-qwen-finetune/datasets/val_output/data_detcls_vl.json \
  --pred_csv /home/dwd/桌面/Spinal-qwen-finetune/output/yolo_stage1/val_yolov8x1280_top5.csv \
  --output_dir /home/dwd/桌面/Spinal-qwen-finetune/output/yolo_qwen_pipeline/yolov8x1280_top5_qwen_oracle \
  --top_k 5 \
  --selection best_iou \
  --crop_expand_ratio 0.2 \
  --load_in_4bit
```

Do not report `best_iou` as the final deployable result. It is useful as an oracle upper bound for analysis.
