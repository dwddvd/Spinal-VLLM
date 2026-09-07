#!/usr/bin/env python
"""Patient-level discrimination and calibration analysis for paired OOF models."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-6


def normalized_infection_score(frame):
    infection = pd.to_numeric(frame["score_infection"], errors="coerce").fillna(0.0).to_numpy(float)
    tumor = pd.to_numeric(frame["score_tumor"], errors="coerce").fillna(0.0).to_numpy(float)
    total = infection + tumor
    return np.divide(infection, total, out=np.full_like(total, 0.5), where=total > 0)


def roc_curve_auc(y, p):
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    positives = max(int(y.sum()), 1)
    negatives = max(int((1 - y).sum()), 1)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def precision_recall_ap(y, p):
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / max(int(y.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall_curve = np.r_[0.0, recall]
    precision_curve = np.r_[1.0, precision]
    ap = float(np.sum((recall_curve[1:] - recall_curve[:-1]) * precision_curve[1:]))
    return recall_curve, precision_curve, ap


def calibration_fit(y, p):
    x = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(100):
        linear = np.clip(design @ beta, -30, 30)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(fitted * (1 - fitted), EPS, None)
        gradient = design.T @ (y - fitted)
        information = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return None, None
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 20:
        return None, None
    return float(beta[0]), float(beta[1])


def quantile_calibration(y, p, n_bins=10):
    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, min(n_bins, len(order)))
    rows = []
    ece = 0.0
    for index, group in enumerate(groups, start=1):
        if not len(group):
            continue
        mean_pred = float(p[group].mean())
        observed = float(y[group].mean())
        weight = len(group) / len(y)
        ece += weight * abs(observed - mean_pred)
        rows.append(
            {
                "bin": index,
                "n": int(len(group)),
                "mean_predicted_infection_probability": mean_pred,
                "observed_infection_fraction": observed,
                "absolute_calibration_error": abs(observed - mean_pred),
            }
        )
    return float(ece), rows


def classification_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "sensitivity_infection": float(sensitivity),
        "specificity_tumor": float(specificity),
        "tp_infection": tp,
        "fn_infection": fn,
        "tn_tumor": tn,
        "fp_tumor": fp,
    }


def score_metrics(y, p, n_bins=10):
    _, _, roc_auc = roc_curve_auc(y, p)
    _, _, average_precision = precision_recall_ap(y, p)
    clipped = np.clip(p, EPS, 1 - EPS)
    intercept, slope = calibration_fit(y, clipped)
    ece, calibration_rows = quantile_calibration(y, p, n_bins=n_bins)
    metrics = {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "ece_equal_frequency": ece,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        **classification_metrics(y, p),
    }
    return metrics, calibration_rows


def percentile_ci(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def stratified_indices(y, rng):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    return np.r_[
        rng.choice(positive, size=len(positive), replace=True),
        rng.choice(negative, size=len(negative), replace=True),
    ]


def bootstrap_analysis(y, selected, baseline, iterations, seed, n_bins):
    rng = np.random.default_rng(seed)
    metric_names = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "ece_equal_frequency",
        "accuracy",
        "balanced_accuracy",
        "sensitivity_infection",
        "specificity_tumor",
    ]
    selected_boot = {key: [] for key in metric_names}
    baseline_boot = {key: [] for key in metric_names}
    delta_boot = {key: [] for key in metric_names}
    for _ in range(iterations):
        indices = stratified_indices(y, rng)
        selected_metrics, _ = score_metrics(y[indices], selected[indices], n_bins=n_bins)
        baseline_metrics, _ = score_metrics(y[indices], baseline[indices], n_bins=n_bins)
        for key in metric_names:
            selected_boot[key].append(selected_metrics[key])
            baseline_boot[key].append(baseline_metrics[key])
            delta_boot[key].append(selected_metrics[key] - baseline_metrics[key])
    return selected_boot, baseline_boot, delta_boot


def curve_rows(y, p, model):
    fpr, tpr, _ = roc_curve_auc(y, p)
    recall, precision, _ = precision_recall_ap(y, p)
    rows = []
    for x, yy in zip(fpr, tpr):
        rows.append({"model": model, "curve": "roc", "x": float(x), "y": float(yy)})
    for x, yy in zip(recall, precision):
        rows.append({"model": model, "curve": "pr", "x": float(x), "y": float(yy)})
    return rows


def draw_figure(output_dir, y, selected, baseline, metrics, calibration):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    colors = {"Nested-selected OOF": "#2C6E9B", "Internal-only OOF": "#8A8F98"}
    probabilities = {"Nested-selected OOF": selected, "Internal-only OOF": baseline}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    for model, p in probabilities.items():
        fpr, tpr, _ = roc_curve_auc(y, p)
        recall, precision, _ = precision_recall_ap(y, p)
        axes[0].plot(fpr, tpr, color=colors[model], lw=1.8, label=f"{model} ({metrics[model]['roc_auc']:.3f})")
        axes[1].plot(recall, precision, color=colors[model], lw=1.8, label=f"{model} ({metrics[model]['average_precision']:.3f})")
        cal = calibration[model]
        axes[2].plot(
            [row["mean_predicted_infection_probability"] for row in cal],
            [row["observed_infection_fraction"] for row in cal],
            marker="o",
            ms=3.5,
            color=colors[model],
            lw=1.5,
            label=f"{model} (Brier {metrics[model]['brier_score']:.3f})",
        )

    axes[0].plot([0, 1], [0, 1], ls="--", lw=0.8, color="#B8BDC5")
    axes[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="a  ROC discrimination", xlim=(0, 1), ylim=(0, 1.02))
    prevalence = float(y.mean())
    axes[1].axhline(prevalence, ls="--", lw=0.8, color="#B8BDC5")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="b  Precision-recall", xlim=(0, 1), ylim=(0, 1.02))
    axes[2].plot([0, 1], [0, 1], ls="--", lw=0.8, color="#B8BDC5")
    axes[2].set(
        xlabel="Predicted infection probability",
        ylabel="Observed infection fraction",
        title="c  Calibration",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    for axis in axes:
        axis.legend(loc="lower right", fontsize=6)
        axis.grid(color="#E7E9ED", lw=0.5)
    fig.suptitle("Nested temporal adaptation improves patient-level discrimination", fontsize=9, y=1.03)
    fig.tight_layout()
    base = output_dir / "patient_score_discrimination_calibration"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_csv", required=True)
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration_bins", type=int, default=10)
    args = parser.parse_args()

    selected_frame = pd.read_csv(args.selected_csv, dtype={"patient_id": str})
    baseline_frame = pd.read_csv(args.baseline_csv, dtype={"patient_id": str})
    required = {"patient_id", "gt_label", "score_infection", "score_tumor"}
    for name, frame in [("selected", selected_frame), ("baseline", baseline_frame)]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

    selected_frame = selected_frame.drop_duplicates("patient_id").copy()
    baseline_frame = baseline_frame.drop_duplicates("patient_id").copy()
    merged = selected_frame.merge(
        baseline_frame,
        on="patient_id",
        how="inner",
        suffixes=("_selected", "_baseline"),
        validate="one_to_one",
    ).sort_values("patient_id")
    if len(merged) != len(selected_frame) or len(merged) != len(baseline_frame):
        raise ValueError("Selected and baseline CSVs do not contain identical patient sets.")
    if not (merged["gt_label_selected"] == merged["gt_label_baseline"]).all():
        raise ValueError("Ground-truth labels disagree between paired patient rows.")

    y = (merged["gt_label_selected"].to_numpy() == "infection").astype(int)
    selected_scores = normalized_infection_score(
        merged.rename(
            columns={
                "score_infection_selected": "score_infection",
                "score_tumor_selected": "score_tumor",
            }
        )
    )
    baseline_scores = normalized_infection_score(
        merged.rename(
            columns={
                "score_infection_baseline": "score_infection",
                "score_tumor_baseline": "score_tumor",
            }
        )
    )
    model_scores = {
        "Nested-selected OOF": selected_scores,
        "Internal-only OOF": baseline_scores,
    }
    metrics = {}
    calibration = {}
    for model, scores in model_scores.items():
        metrics[model], calibration[model] = score_metrics(y, scores, n_bins=args.calibration_bins)

    selected_boot, baseline_boot, delta_boot = bootstrap_analysis(
        y,
        selected_scores,
        baseline_scores,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
        n_bins=args.calibration_bins,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for metric in selected_boot:
        selected_estimate = metrics["Nested-selected OOF"][metric]
        baseline_estimate = metrics["Internal-only OOF"][metric]
        summary_rows.append(
            {
                "metric": metric,
                "nested_selected_estimate": selected_estimate,
                "nested_selected_ci_low": percentile_ci(selected_boot[metric])[0],
                "nested_selected_ci_high": percentile_ci(selected_boot[metric])[1],
                "internal_only_estimate": baseline_estimate,
                "internal_only_ci_low": percentile_ci(baseline_boot[metric])[0],
                "internal_only_ci_high": percentile_ci(baseline_boot[metric])[1],
                "paired_delta_selected_minus_internal": selected_estimate - baseline_estimate,
                "paired_delta_ci_low": percentile_ci(delta_boot[metric])[0],
                "paired_delta_ci_high": percentile_ci(delta_boot[metric])[1],
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    calibration_status = pd.DataFrame(
        [
            {
                "model": model,
                "calibration_intercept": values["calibration_intercept"],
                "calibration_slope": values["calibration_slope"],
                "status": (
                    "estimable"
                    if values["calibration_intercept"] is not None
                    else "not estimable: boundary vote scores caused quasi-complete separation"
                ),
            }
            for model, values in metrics.items()
        ]
    )

    patient_scores = pd.DataFrame(
        {
            "patient_id": merged["patient_id"],
            "gt_label": merged["gt_label_selected"],
            "infection_probability_nested_selected": selected_scores,
            "infection_probability_internal_only": baseline_scores,
            "pred_label_nested_selected": np.where(selected_scores >= 0.5, "infection", "tumor"),
            "pred_label_internal_only": np.where(baseline_scores >= 0.5, "infection", "tumor"),
            "outer_fold": merged.get("outer_fold_selected", ""),
            "selected_config": merged.get("selected_config_selected", ""),
        }
    )
    calibration_rows = []
    for model, rows in calibration.items():
        for row in rows:
            calibration_rows.append({"model": model, **row})
    calibration_frame = pd.DataFrame(calibration_rows)
    curves_frame = pd.DataFrame(
        curve_rows(y, selected_scores, "Nested-selected OOF")
        + curve_rows(y, baseline_scores, "Internal-only OOF")
    )

    summary_frame.to_csv(output_dir / "patient_score_metrics_bootstrap.csv", index=False, encoding="utf-8-sig")
    patient_scores.to_csv(output_dir / "patient_continuous_scores.csv", index=False, encoding="utf-8-sig")
    calibration_frame.to_csv(output_dir / "calibration_curve_bins.csv", index=False, encoding="utf-8-sig")
    curves_frame.to_csv(output_dir / "roc_pr_curve_points.csv", index=False, encoding="utf-8-sig")

    result = {
        "analysis_population": {
            "patients": int(len(y)),
            "infection": int(y.sum()),
            "tumor": int((1 - y).sum()),
            "positive_class": "infection",
        },
        "score_definition": "score_infection / (score_infection + score_tumor)",
        "nested_selected_oof": metrics["Nested-selected OOF"],
        "internal_only_oof": metrics["Internal-only OOF"],
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "method": "patient-level stratified paired percentile bootstrap",
        },
        "calibration": {
            "bins": args.calibration_bins,
            "method": "equal-frequency bins; logistic recalibration intercept and slope",
            "intercept_slope_status": "Not reported when boundary vote scores cause quasi-complete separation.",
        },
        "interpretation_boundary": (
            "Selection-adjusted same-center temporal OOF analysis. Scores are relative model vote "
            "weights, not prospectively calibrated clinical probabilities."
        ),
    }
    with (output_dir / "patient_score_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)

    with pd.ExcelWriter(output_dir / "patient_score_analysis.xlsx", engine="openpyxl") as writer:
        summary_frame.to_excel(writer, sheet_name="Metrics_95CI", index=False)
        patient_scores.to_excel(writer, sheet_name="Patient_scores", index=False)
        calibration_frame.to_excel(writer, sheet_name="Calibration_bins", index=False)
        calibration_status.to_excel(writer, sheet_name="Calibration_status", index=False)
        curves_frame.to_excel(writer, sheet_name="Curve_points", index=False)

    figure_base = draw_figure(output_dir, y, selected_scores, baseline_scores, metrics, calibration)
    result["figure_base"] = figure_base
    with (output_dir / "patient_score_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
