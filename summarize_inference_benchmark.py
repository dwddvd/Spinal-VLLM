import argparse
import csv
import json
import re
from pathlib import Path


def read_float(path: Path) -> float:
    return float(path.read_text(encoding="utf-8").strip())


def parse_time_verbose(path: Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    rss_kb = values.get("Maximum resident set size (kbytes)")
    return {
        "maximum_resident_set_size_mb": float(rss_kb) / 1024 if rss_kb else None,
        "user_time_seconds": _first_number(values.get("User time (seconds)")),
        "system_time_seconds": _first_number(values.get("System time (seconds)")),
        "cpu_percent": values.get("Percent of CPU this job got"),
    }


def _first_number(value):
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def parse_gpu_csv(path: Path, baseline_path: Path) -> dict:
    baseline = None
    baseline_rows = (
        list(csv.reader(baseline_path.open(encoding="utf-8")))
        if baseline_path.exists()
        else []
    )
    if baseline_rows and len(baseline_rows[0]) >= 2:
        baseline = float(baseline_rows[0][1].strip())

    memory_values = []
    utilization_values = []
    power_values = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 4:
                    continue
                try:
                    memory_values.append(float(row[1].strip()))
                    utilization_values.append(float(row[2].strip()))
                    power_values.append(float(row[3].strip()))
                except ValueError:
                    continue

    peak_memory = max(memory_values) if memory_values else None
    return {
        "sampling_interval_seconds": 1,
        "num_samples": len(memory_values),
        "baseline_gpu_memory_mib": baseline,
        "peak_total_gpu_memory_mib": peak_memory,
        "peak_incremental_gpu_memory_mib": (
            peak_memory - baseline if peak_memory is not None and baseline is not None else None
        ),
        "mean_gpu_utilization_percent": (
            sum(utilization_values) / len(utilization_values) if utilization_values else None
        ),
        "peak_gpu_utilization_percent": max(utilization_values) if utilization_values else None,
        "mean_power_watts": sum(power_values) / len(power_values) if power_values else None,
        "peak_power_watts": max(power_values) if power_values else None,
    }


def read_prediction_counts(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    patient_ids = {
        (row.get("patient_id") or row.get("patient") or "").strip()
        for row in rows
        if (row.get("patient_id") or row.get("patient") or "").strip()
    }
    return {"num_slice_records": len(rows), "num_unique_patients": len(patient_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    parser.add_argument("--pipeline_metrics", type=Path, required=True)
    parser.add_argument("--predictions_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    elapsed = read_float(args.benchmark_dir / "end_epoch.txt") - read_float(
        args.benchmark_dir / "start_epoch.txt"
    )
    counts = read_prediction_counts(args.predictions_csv)
    metrics = json.loads(args.pipeline_metrics.read_text(encoding="utf-8"))
    cpu = parse_time_verbose(args.benchmark_dir / "time_verbose.txt")
    gpu = parse_gpu_csv(
        args.benchmark_dir / "gpu_monitor.csv",
        args.benchmark_dir / "gpu_baseline.csv",
    )

    slices = counts["num_slice_records"]
    patients = counts["num_unique_patients"]
    summary = {
        "benchmark_scope": (
            "Cold-start end-to-end pipeline benchmark on the complete temporal cohort "
            "using the fold-0 nested-selected adapt10_r5 adapter."
        ),
        "hardware_scope": "Single NVIDIA RTX PRO 6000 Blackwell GPU.",
        "includes": [
            "Qwen model and LoRA adapter loading",
            "nnU-Net mask loading and candidate extraction",
            "component-wise Qwen inference",
            "CSV and JSON output writing",
        ],
        "excludes": [
            "nnU-Net segmentation prediction generation",
            "DICOM-to-NIfTI preprocessing",
        ],
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
        "num_slice_records": slices,
        "num_unique_patients": patients,
        "slices_per_second": slices / elapsed if elapsed and slices else None,
        "seconds_per_slice": elapsed / slices if slices else None,
        "seconds_per_patient": elapsed / patients if patients else None,
        "cpu_memory": cpu,
        "gpu": gpu,
        "gpu_monitoring_available": bool(gpu.get("num_samples")),
        "pipeline_behavior": metrics.get("pipeline_behavior", {}),
        "source_files": {
            "pipeline_metrics": str(args.pipeline_metrics),
            "predictions_csv": str(args.predictions_csv),
            "time_verbose": str(args.benchmark_dir / "time_verbose.txt"),
            "gpu_monitor": str(args.benchmark_dir / "gpu_monitor.csv"),
        },
        "reporting_note": (
            "This is a representative engineering benchmark, not a clinical latency "
            "claim. GPU memory is sampled device-wide and reported relative to the "
            "pre-run baseline when NVML monitoring is available; GPU telemetry is "
            "unavailable in this run because NVML could not be initialized."
            if not gpu.get("num_samples")
            else
            "This is a representative engineering benchmark, not a clinical latency "
            "claim. GPU memory is sampled device-wide and reported relative to the "
            "pre-run baseline."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
