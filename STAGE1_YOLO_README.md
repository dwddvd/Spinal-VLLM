# Stage-1 YOLO Lesion Detector

This is the recommended first-stage detector for stable paper metrics:

```text
MRI image -> YOLO single-class lesion detector -> top-k boxes -> Qwen crop classifier
```

The detector uses one class only:

```text
0: lesion
```

## 1. Prepare YOLO Data

```bash
python yolo_stage1_detection.py prepare \
  --train_json /path/to/datasets/train_output/data_detcls_vl.json \
  --val_json /path/to/datasets/val_output/data_detcls_vl.json \
  --output_dir datasets/yolo_spinal_lesion
```

This creates:

```text
datasets/yolo_spinal_lesion/
  images/train/
  images/val/
  labels/train/
  labels/val/
  spinal_lesion.yaml
  prepare_summary.json
```

If image paths in JSON are relative, add:

```bash
--image_root /path/to/project_or_dataset_root
```

## 2. Train

Start with YOLOv8m:

```bash
python yolo_stage1_detection.py train \
  --data_yaml datasets/yolo_spinal_lesion/spinal_lesion.yaml \
  --model yolov8m.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --project output/yolo_stage1 \
  --name yolov8m_lesion
```

If the server is slow when Ultralytics runs AMP checks and downloads a small `*n.pt` model, disable AMP:

```bash
python yolo_stage1_detection.py train \
  --data_yaml datasets/yolo_spinal_lesion/spinal_lesion.yaml \
  --model ./yolov8m.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --project output/yolo_stage1 \
  --name yolov8m_lesion \
  --no_amp
```

For small data, also try:

```bash
--model yolov8s.pt
```

For stronger final numbers if GPU memory allows:

```bash
--model yolov8l.pt
```

## 3. Validate

```bash
python yolo_stage1_detection.py val \
  --weights output/yolo_stage1/yolov8m_lesion/weights/best.pt \
  --data_yaml datasets/yolo_spinal_lesion/spinal_lesion.yaml
```

Report:

```text
mAP@0.5
mAP@0.5:0.95
precision
recall
```

## 4. High-Recall Check

For a two-stage system, recall is more important than first-stage precision:

```bash
python yolo_stage1_detection.py recall \
  --weights output/yolo_stage1/yolov8m_lesion/weights/best.pt \
  --val_json /path/to/datasets/val_output/data_detcls_vl.json \
  --conf 0.05 \
  --max_det 10 \
  --thresholds 0.3,0.5
```

## 5. Export Top-k Boxes For Qwen

```bash
python yolo_stage1_detection.py predict \
  --weights output/yolo_stage1/yolov8m_lesion/weights/best.pt \
  --source /path/to/val/images_or_one_image \
  --conf 0.05 \
  --top_k 3 \
  --output_csv output/yolo_stage1/predictions.csv
```

The CSV columns are:

```text
image_path,rank,conf,x1,y1,x2,y2
```

Feed these boxes to the existing Qwen crop classifier, preferably with 10%-30% bbox expansion.
