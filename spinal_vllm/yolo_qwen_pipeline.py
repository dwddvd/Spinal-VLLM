import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from peft import PeftModel
from PIL import Image
from tqdm import tqdm

from .qwen_stage2_classifier import (
    LesionRecord,
    build_messages,
    extract_image_path,
    extract_prompt_text,
    infer_sequence,
    generate_text,
    load_model_and_processor,
    make_non_empty_bbox,
    normalize_label,
)


BBox = Tuple[int, int, int, int]
BBOX_TOKEN_PATTERN = re.compile(
    r"\[\s*(?:(?:\d+|x1)\s*,\s*(?:\d+|y1)\s*,\s*(?:\d+|x2)\s*,\s*(?:\d+|y2)|MASK)\s*\]",
    flags=re.I,
)
NUMERIC_BBOX_PATTERN = re.compile(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]")


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bbox_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area2d(a: BBox, b: BBox) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def iou2d(a: BBox, b: BBox) -> float:
    inter = intersection_area2d(a, b)
    denom = bbox_area(a) + bbox_area(b) - inter
    return inter / denom if denom > 0 else 0.0


def iom2d(a: BBox, b: BBox) -> float:
    inter = intersection_area2d(a, b)
    denom = min(bbox_area(a), bbox_area(b))
    return inter / denom if denom > 0 else 0.0


def gt_coverage_by_pred(pred_bbox: BBox, gt_bbox: BBox) -> float:
    inter = intersection_area2d(pred_bbox, gt_bbox)
    area = bbox_area(gt_bbox)
    return inter / area if area > 0 else 0.0


def pred_precision_to_gt(pred_bbox: BBox, gt_bbox: BBox) -> float:
    inter = intersection_area2d(pred_bbox, gt_bbox)
    area = bbox_area(pred_bbox)
    return inter / area if area > 0 else 0.0


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_hit(pred_bbox: BBox, gt_bbox: BBox) -> bool:
    cx, cy = bbox_center(pred_bbox)
    x1, y1, x2, y2 = gt_bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def relaxed_localization_hit(pred_bbox: BBox, gt_bbox: BBox) -> bool:
    iou = iou2d(pred_bbox, gt_bbox)
    iom = iom2d(pred_bbox, gt_bbox)
    coverage = gt_coverage_by_pred(pred_bbox, gt_bbox)
    return (iou >= 0.3) or (iom >= 0.5) or (coverage >= 0.7) or (center_hit(pred_bbox, gt_bbox) and coverage >= 0.5)


def bbox_to_text(bbox: BBox) -> str:
    return f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"


def remap_path(path_text: str, server_prefix: str = "", local_prefix: str = "") -> str:
    path_text = str(path_text or "")
    if server_prefix and local_prefix:
        normalized_path = path_text.replace("\\", "/")
        normalized_server = server_prefix.replace("\\", "/").rstrip("/")
        if normalized_path.startswith(normalized_server):
            suffix = normalized_path[len(normalized_server):].lstrip("/")
            return str(Path(local_prefix) / Path(*suffix.split("/")))
    return path_text


def load_qwen_records(json_path: str, server_image_prefix: str = "", local_image_prefix: str = "") -> List[LesionRecord]:
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
        image_path_raw = extract_image_path(user_text)
        image_path = remap_path(image_path_raw or "", server_image_prefix, local_image_prefix)
        bbox_value = item.get("bbox")
        bbox = tuple(bbox_value) if isinstance(bbox_value, list) and len(bbox_value) == 4 else None
        if bbox is None:
            match = NUMERIC_BBOX_PATTERN.search(user_text) or NUMERIC_BBOX_PATTERN.search(assistant_text)
            if match:
                bbox = tuple(parse_int(v) for v in re.findall(r"\d+", match.group(0))[:4])
        label = item.get("label") or normalize_label(assistant_text)
        if image_path_raw is None or bbox is None or label is None or not os.path.exists(image_path):
            skipped += 1
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        sample_id = str(item.get("id", "")) or f"sample_{index}"
        slice_manifest = item.get("slice_manifest", {}) if isinstance(item.get("slice_manifest"), dict) else {}
        patient_id = str(item.get("patient_id") or slice_manifest.get("patient_id") or Path(image_path).name.split("_")[0])
        seq = item.get("seq") or slice_manifest.get("seq") or infer_sequence(user_text)
        slice_idx = item.get("slice_idx", slice_manifest.get("slice_idx"))
        try:
            slice_idx = int(slice_idx) if slice_idx is not None and str(slice_idx) != "" else None
        except (TypeError, ValueError):
            slice_idx = None
        records.append(
            LesionRecord(
                sample_id=sample_id,
                image_path=image_path,
                bbox=make_non_empty_bbox(tuple(bbox), width, height),
                label=label,
                width=width,
                height=height,
                patient_id=patient_id,
                seq=seq,
                prompt_text=extract_prompt_text(user_text),
                answer_text=assistant_text.strip(),
                case_id=item.get("case_id") or slice_manifest.get("case_id"),
                slice_idx=slice_idx,
            )
        )
    print(f"[INFO] Loaded {len(records)} records from {json_path}; skipped {skipped}.", flush=True)
    return records


def sample_id_from_yolo_path(path_text: str) -> str:
    stem = Path(str(path_text)).stem
    match = re.match(r"^\d{6}_(.+)$", stem)
    return match.group(1) if match else stem


def replace_first_bbox(text: str, bbox: BBox) -> str:
    return BBOX_TOKEN_PATTERN.sub(bbox_to_text(bbox), text, count=2)


def replace_nth_bbox(text: str, bbox: BBox, n: int = 2) -> str:
    count = 0

    def repl(match):
        nonlocal count
        count += 1
        return bbox_to_text(bbox) if count == n else match.group(0)

    replaced = BBOX_TOKEN_PATTERN.sub(repl, text)
    if count < n:
        return replace_first_bbox(text, bbox)
    return replaced


def make_candidate_prompt(prompt_text: str, bbox: BBox, rank: int, total: int, region_prompt: bool) -> str:
    if not region_prompt:
        return replace_first_bbox(prompt_text, bbox)
    replaced = replace_nth_bbox(prompt_text, bbox, n=2)
    return (
        f"{replaced}\n"
        f"补充说明：当前仅分析候选病灶区域{rank}/{total}，坐标为{bbox_to_text(bbox)}。"
        "请忽略其他区域，仅根据这个框内最主要的病灶表现判断属于感染还是肿瘤；"
        "如果该框不像有效病灶，请输出'无明确病灶'。"
    )


def load_prompt_records(
    json_path: str,
    gt_records_by_id: Dict[str, LesionRecord],
    server_image_prefix: str = "",
    local_image_prefix: str = "",
) -> Dict[str, LesionRecord]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a JSON list.")
    out: Dict[str, LesionRecord] = {}
    skipped = 0
    for index, item in enumerate(data):
        convs = item.get("conversations", [])
        if len(convs) < 2:
            skipped += 1
            continue
        sample_id = str(item.get("id", "")) or f"sample_{index}"
        gt = gt_records_by_id.get(sample_id)
        if gt is None:
            skipped += 1
            continue
        user_text = convs[0].get("value", "")
        assistant_text = convs[1].get("value", "")
        image_path = remap_path(extract_image_path(user_text) or gt.image_path, server_image_prefix, local_image_prefix)
        out[sample_id] = LesionRecord(
            sample_id=sample_id,
            image_path=image_path,
            bbox=gt.bbox,
            label=gt.label,
            width=gt.width,
            height=gt.height,
            patient_id=gt.patient_id,
            seq=gt.seq,
            prompt_text=extract_prompt_text(user_text),
            answer_text=assistant_text.strip(),
            case_id=gt.case_id,
            slice_idx=gt.slice_idx,
        )
    print(f"[INFO] Loaded {len(out)} hidden prompt records from {json_path}; skipped {skipped}.", flush=True)
    return out


def assert_hidden_prompt_is_safe(records: Dict[str, LesionRecord]) -> None:
    bad = [sample_id for sample_id, record in records.items() if NUMERIC_BBOX_PATTERN.search(record.prompt_text or "")]
    if bad:
        raise ValueError(
            "hidden_qwen_json still contains numeric bbox prompts. "
            f"Examples: {bad[:5]}. Please use *_hidden_bbox.json for pipeline inference."
        )


def load_yolo_candidates(pred_csv: str, records_by_id: Dict[str, LesionRecord]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(pred_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("sample_id") or sample_id_from_yolo_path(row.get("image_path", ""))
            record = records_by_id.get(sample_id)
            if record is None:
                continue
            bbox = make_non_empty_bbox(
                (
                    parse_int(row.get("x1")),
                    parse_int(row.get("y1")),
                    parse_int(row.get("x2")),
                    parse_int(row.get("y2")),
                ),
                record.width,
                record.height,
            )
            rank = parse_int(row.get("rank"), default=len(grouped[sample_id]) + 1)
            grouped[sample_id].append(
                {
                    "sample_id": sample_id,
                    "rank": rank,
                    "conf": parse_float(row.get("conf")),
                    "bbox": bbox,
                    "source_image_path": row.get("image_path", ""),
                }
            )
    for sample_id in grouped:
        grouped[sample_id].sort(key=lambda item: (item["rank"], -item["conf"]))
    print(f"[INFO] Loaded YOLO candidates for {len(grouped)} Qwen records from {pred_csv}.", flush=True)
    return grouped


def load_qwen(base_model: str, adapter_path: str, load_in_4bit: bool):
    base_model_obj, processor = load_model_and_processor(base_model, load_in_4bit, gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base_model_obj, adapter_path)
    model.eval()
    return model, processor


def choose_final_candidate(candidates: List[Dict[str, Any]], strategy: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
    valid = [c for c in candidates if c.get("pred_label") in {"infection", "tumor"}]
    scores = {"infection": 0.0, "tumor": 0.0}
    if not candidates:
        return None, scores
    if strategy == "top1_conf":
        return candidates[0], scores
    if strategy == "first_valid":
        return (valid[0] if valid else candidates[0]), scores
    if strategy == "majority_vote":
        counts = Counter(c["pred_label"] for c in valid)
        if not counts:
            return candidates[0], scores
        winner, _ = max(counts.items(), key=lambda item: (item[1], scores.get(item[0], 0.0)))
        selected = max([c for c in valid if c["pred_label"] == winner], key=lambda c: c["conf"])
        return selected, {label: float(counts.get(label, 0)) for label in scores}
    if strategy == "conf_weighted_vote":
        for cand in valid:
            scores[cand["pred_label"]] += float(cand.get("conf", 0.0))
        if not valid:
            return candidates[0], scores
        if scores["infection"] == scores["tumor"]:
            return max(valid, key=lambda c: c["conf"]), scores
        winner = max(scores.items(), key=lambda item: item[1])[0]
        selected = max([c for c in valid if c["pred_label"] == winner], key=lambda c: c["conf"])
        return selected, scores
    raise ValueError(f"Unsupported final strategy: {strategy}")


def candidate_metrics(candidates: List[Dict[str, Any]], gt_label: str, gt_bbox: BBox, thresholds: List[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for thr in thresholds:
        out[f"any_det_recall_iou{thr}"] = int(any(c["det_iou"] >= thr for c in candidates))
        out[f"any_joint_iou{thr}"] = int(any(c["det_iou"] >= thr and c.get("pred_label") == gt_label for c in candidates))
    out["any_relaxed_loc"] = int(any(c["relaxed_localization_hit"] for c in candidates))
    out["any_relaxed_joint"] = int(any(c["relaxed_localization_hit"] and c.get("pred_label") == gt_label for c in candidates))
    return out


def evaluate(args: argparse.Namespace) -> None:
    qwen_records = load_qwen_records(args.qwen_json, args.server_image_prefix, args.local_image_prefix)
    records_by_id = {record.sample_id: record for record in qwen_records}
    hidden_records = (
        load_prompt_records(args.hidden_qwen_json, records_by_id, args.server_image_prefix, args.local_image_prefix)
        if args.hidden_qwen_json
        else records_by_id
    )
    if args.hidden_qwen_json:
        missing = [sample_id for sample_id in records_by_id if sample_id not in hidden_records]
        if missing:
            raise ValueError(f"hidden_qwen_json missing {len(missing)} sample ids, examples={missing[:5]}")
        assert_hidden_prompt_is_safe(hidden_records)

    eval_records = list(qwen_records)
    if args.shuffle_eval:
        rng = random.Random(args.seed)
        rng.shuffle(eval_records)
        print(f"[INFO] Shuffled eval records with seed={args.seed}.", flush=True)
    if args.limit_eval > 0:
        eval_records = eval_records[: args.limit_eval]
        print(f"[INFO] Limited eval records to {len(eval_records)} samples.", flush=True)

    yolo_candidates = load_yolo_candidates(args.pred_csv, records_by_id)
    top_ks = [int(item) for item in str(args.top_ks).split(",") if item.strip()]
    thresholds = [float(item) for item in str(args.thresholds).split(",") if item.strip()]
    strategies = [item.strip() for item in str(args.final_strategies).split(",") if item.strip()]
    max_k = max(top_ks)

    model, processor = load_qwen(args.base_model, args.adapter_path, args.load_in_4bit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    hidden_json_records: List[Dict[str, Any]] = []
    pred_bbox_json_records: List[Dict[str, Any]] = []
    per_sample_candidates: Dict[str, List[Dict[str, Any]]] = {}

    progress = tqdm(eval_records, desc="Eval YOLO bbox filled Qwen prompts", ncols=160)
    for record in progress:
        prompt_record = hidden_records[record.sample_id]
        candidates = yolo_candidates.get(record.sample_id, [])[:max_k]
        evaluated: List[Dict[str, Any]] = []
        for cand in candidates:
            prompt = make_candidate_prompt(
                prompt_record.prompt_text or "",
                cand["bbox"],
                cand["rank"],
                len(candidates),
                region_prompt=args.region_prompt,
            )
            pred_text = generate_text(
                model,
                processor,
                build_messages(prompt, prompt_record.image_path, args.image_resize),
                max_new_tokens=args.max_new_tokens,
            )
            pred_label = normalize_label(pred_text)
            det_iou = iou2d(cand["bbox"], record.bbox)
            det_iom = iom2d(cand["bbox"], record.bbox)
            gt_cov = gt_coverage_by_pred(cand["bbox"], record.bbox)
            pred_prec = pred_precision_to_gt(cand["bbox"], record.bbox)
            c_hit = center_hit(cand["bbox"], record.bbox)
            relaxed_hit = relaxed_localization_hit(cand["bbox"], record.bbox)
            item = {
                **cand,
                "pred_text": pred_text,
                "pred_label": pred_label or "",
                "det_iou": det_iou,
                "det_iom": det_iom,
                "gt_coverage_by_pred": gt_cov,
                "pred_precision_to_gt": pred_prec,
                "center_hit": int(c_hit),
                "relaxed_localization_hit": int(relaxed_hit),
            }
            evaluated.append(item)
            candidate_rows.append(
                {
                    "sample_id": record.sample_id,
                    "patient_id": record.patient_id,
                    "seq": record.seq or "",
                    "gt_label": record.label,
                    "rank": cand["rank"],
                    "conf": cand["conf"],
                    "pred_label": pred_label or "",
                    "pred_text": pred_text,
                    "det_iou": det_iou,
                    "det_iom": det_iom,
                    "gt_coverage_by_pred": gt_cov,
                    "pred_precision_to_gt": pred_prec,
                    "center_hit": int(c_hit),
                    "relaxed_localization_hit": int(relaxed_hit),
                    "pred_x1": cand["bbox"][0],
                    "pred_y1": cand["bbox"][1],
                    "pred_x2": cand["bbox"][2],
                    "pred_y2": cand["bbox"][3],
                    "gt_x1": record.bbox[0],
                    "gt_y1": record.bbox[1],
                    "gt_x2": record.bbox[2],
                    "gt_y2": record.bbox[3],
                    "qwen_image_path": record.image_path,
                    "prompt_image_path": prompt_record.image_path,
                    "source_image_path": cand.get("source_image_path", ""),
                }
            )
        per_sample_candidates[record.sample_id] = evaluated
        preview = evaluated[0]["pred_text"].replace("\n", " ").strip() if evaluated else "no_candidate"
        if len(preview) > 48:
            preview = preview[:48] + "..."
        progress.set_postfix({"gt": record.label, "n": len(evaluated), "top1": evaluated[0]["pred_label"] if evaluated else "none", "text": preview})

    metrics: Dict[str, Any] = {
        "total_records": len(eval_records),
        "qwen_records_loaded": len(qwen_records),
        "yolo_candidate_records": len(yolo_candidates),
        "top_ks": top_ks,
        "thresholds": thresholds,
        "final_strategies": strategies,
        "input_source": "YOLO predicted bboxes filled into hidden-bbox Qwen prompts",
        "region_prompt": args.region_prompt,
        "shuffle_eval": args.shuffle_eval,
        "seed": args.seed if args.shuffle_eval else None,
        "per_top_k": {},
    }

    for top_k in top_ks:
        subset_rows = []
        any_hits = Counter()
        candidate_count = 0
        images_with_candidate = 0
        mean_values = Counter()
        for record in eval_records:
            candidates = per_sample_candidates.get(record.sample_id, [])[:top_k]
            if candidates:
                images_with_candidate += 1
                candidate_count += len(candidates)
            any_metric = candidate_metrics(candidates, record.label, record.bbox, thresholds)
            any_hits.update(any_metric)
            if candidates:
                best_iou = max(c["det_iou"] for c in candidates)
                best_iom = max(c["det_iom"] for c in candidates)
                mean_values["best_iou_sum"] += best_iou
                mean_values["best_iom_sum"] += best_iom

        top_metrics: Dict[str, Any] = {
            "images_with_candidate": images_with_candidate,
            "candidate_coverage": images_with_candidate / max(len(eval_records), 1),
            "num_candidate_predictions": candidate_count,
            "mean_best_iou": mean_values["best_iou_sum"] / max(images_with_candidate, 1),
            "mean_best_iom": mean_values["best_iom_sum"] / max(images_with_candidate, 1),
            "candidate_pool": {},
            "final_selection": {},
        }
        for thr in thresholds:
            top_metrics["candidate_pool"][f"det_recall@iou{thr}"] = any_hits[f"any_det_recall_iou{thr}"] / max(len(eval_records), 1)
            top_metrics["candidate_pool"][f"joint_oracle@iou{thr}"] = any_hits[f"any_joint_iou{thr}"] / max(len(eval_records), 1)
        top_metrics["candidate_pool"]["relaxed_localization_recall"] = any_hits["any_relaxed_loc"] / max(len(eval_records), 1)
        top_metrics["candidate_pool"]["relaxed_joint_oracle"] = any_hits["any_relaxed_joint"] / max(len(eval_records), 1)

        for strategy in strategies:
            cls_correct = 0
            selected_count = 0
            det_hits = Counter()
            joint_hits = Counter()
            strict_hits = 0
            relaxed_hits = 0
            strict_joint = 0
            relaxed_joint = 0
            sum_det_iou = 0.0
            sum_det_iom = 0.0
            sum_gt_cov = 0.0
            sum_pred_prec = 0.0
            center_hits = 0
            confusion = Counter()
            selected_rank = Counter()

            for record in eval_records:
                candidates = per_sample_candidates.get(record.sample_id, [])[:top_k]
                selected, vote_scores = choose_final_candidate(candidates, strategy)
                if selected is None:
                    confusion[f"{record.label}->no_candidate"] += 1
                    selected_rank["no_candidate"] += 1
                    final_rows.append(
                        {
                            "sample_id": record.sample_id,
                            "patient_id": record.patient_id,
                            "seq": record.seq or "",
                            "top_k": top_k,
                            "strategy": strategy,
                            "gt_label": record.label,
                            "pred_label": "no_candidate",
                            "selected_rank": "",
                            "selected_conf": 0.0,
                            "vote_score_infection": 0.0,
                            "vote_score_tumor": 0.0,
                            "det_iou": 0.0,
                            "det_iom": 0.0,
                            "qwen_image_path": record.image_path,
                        }
                    )
                    continue
                selected_count += 1
                pred_label = selected.get("pred_label") or "invalid"
                label_ok = pred_label == record.label
                cls_correct += int(label_ok)
                confusion[f"{record.label}->{pred_label}"] += 1
                selected_rank[str(selected["rank"])] += 1
                det_iou = selected["det_iou"]
                det_iom = selected["det_iom"]
                sum_det_iou += det_iou
                sum_det_iom += det_iom
                sum_gt_cov += selected["gt_coverage_by_pred"]
                sum_pred_prec += selected["pred_precision_to_gt"]
                center_hits += int(selected["center_hit"])
                strict_hit = det_iou >= 0.5
                relaxed_hit = bool(selected["relaxed_localization_hit"])
                strict_hits += int(strict_hit)
                relaxed_hits += int(relaxed_hit)
                strict_joint += int(strict_hit and label_ok)
                relaxed_joint += int(relaxed_hit and label_ok)
                for thr in thresholds:
                    det_hits[thr] += int(det_iou >= thr)
                    joint_hits[thr] += int(det_iou >= thr and label_ok)
                final_rows.append(
                    {
                        "sample_id": record.sample_id,
                        "patient_id": record.patient_id,
                        "seq": record.seq or "",
                        "top_k": top_k,
                        "strategy": strategy,
                        "gt_label": record.label,
                        "pred_label": pred_label,
                        "pred_text": selected.get("pred_text", ""),
                        "selected_rank": selected["rank"],
                        "selected_conf": selected["conf"],
                        "vote_score_infection": vote_scores["infection"],
                        "vote_score_tumor": vote_scores["tumor"],
                        "det_iou": det_iou,
                        "det_iom": det_iom,
                        "gt_coverage_by_pred": selected["gt_coverage_by_pred"],
                        "pred_precision_to_gt": selected["pred_precision_to_gt"],
                        "center_hit": selected["center_hit"],
                        "strict_localization_hit": int(strict_hit),
                        "relaxed_localization_hit": int(relaxed_hit),
                        "pred_x1": selected["bbox"][0],
                        "pred_y1": selected["bbox"][1],
                        "pred_x2": selected["bbox"][2],
                        "pred_y2": selected["bbox"][3],
                        "gt_x1": record.bbox[0],
                        "gt_y1": record.bbox[1],
                        "gt_x2": record.bbox[2],
                        "gt_y2": record.bbox[3],
                        "qwen_image_path": record.image_path,
                    }
                )

            strategy_metrics: Dict[str, Any] = {
                "selected_count": selected_count,
                "cls_acc_on_selected": cls_correct / max(selected_count, 1),
                "mean_det_iou": sum_det_iou / max(selected_count, 1),
                "mean_det_iom": sum_det_iom / max(selected_count, 1),
                "mean_gt_coverage": sum_gt_cov / max(selected_count, 1),
                "mean_pred_precision": sum_pred_prec / max(selected_count, 1),
                "center_hit_rate": center_hits / max(selected_count, 1),
                "strict_localization_recall": strict_hits / max(len(eval_records), 1),
                "relaxed_localization_recall": relaxed_hits / max(len(eval_records), 1),
                "joint_acc_strict": strict_joint / max(len(eval_records), 1),
                "joint_acc_relaxed": relaxed_joint / max(len(eval_records), 1),
                "confusion": dict(confusion),
                "selected_rank": dict(selected_rank),
            }
            for thr in thresholds:
                strategy_metrics[f"selected_det_recall@iou{thr}"] = det_hits[thr] / max(len(eval_records), 1)
                strategy_metrics[f"selected_joint_acc@iou{thr}"] = joint_hits[thr] / max(len(eval_records), 1)
            top_metrics["final_selection"][strategy] = strategy_metrics

        metrics["per_top_k"][f"top{top_k}"] = top_metrics

    candidate_csv = output_dir / "yolo_qwen_candidate_predictions.csv"
    final_csv = output_dir / "yolo_qwen_final_selections.csv"
    metrics_json = output_dir / "yolo_qwen_pipeline_metrics.json"

    if candidate_rows:
        with candidate_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(candidate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(candidate_rows)
    if final_rows:
        all_fields = []
        seen = set()
        for row in final_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    all_fields.append(key)
        with final_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(final_rows)

    metrics["files"] = {
        "candidate_predictions_csv": str(candidate_csv),
        "final_selections_csv": str(final_csv),
        "metrics_json": str(metrics_json),
        "pred_csv": args.pred_csv,
        "qwen_json": args.qwen_json,
        "hidden_qwen_json": args.hidden_qwen_json,
    }
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[INFO] Saved metrics to {metrics_json}", flush=True)
    print(f"[INFO] Saved candidate predictions to {candidate_csv}", flush=True)
    print(f"[INFO] Saved final selections to {final_csv}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO proposal + Qwen classifier baseline with hidden-bbox prompts."
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument(
        "--pred_csv",
        required=True,
        help="YOLO top-k prediction CSV exported by preprocessing.yolo_stage1_detection predict.",
    )
    parser.add_argument("--qwen_json", required=True, help="GT Qwen JSON with real bbox, used only for labels/evaluation.")
    parser.add_argument("--hidden_qwen_json", required=True, help="Hidden-bbox Qwen JSON used to construct inference prompts.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--server_image_prefix",
        default="",
        help="Optional prefix in JSON image paths to replace, for example /path/to/original/project.",
    )
    parser.add_argument(
        "--local_image_prefix",
        default="",
        help="Optional local prefix replacing --server_image_prefix, for example /path/to/current/project.",
    )
    parser.add_argument("--top_ks", default="1,3,5,10")
    parser.add_argument("--thresholds", default="0.3,0.5")
    parser.add_argument(
        "--final_strategies",
        default="top1_conf,first_valid,majority_vote,conf_weighted_vote",
        help="Comma-separated final candidate strategies.",
    )
    parser.add_argument("--image_resize", type=int, default=280)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--shuffle_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_eval", type=int, default=0)
    parser.add_argument(
        "--region_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append candidate-region instructions for multi-candidate evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
