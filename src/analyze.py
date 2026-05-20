import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

TIME_SIGS = ["2/4", "3/4", "4/4", "5/4", "6/8"]


def _normalize_sig(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})


def load_data(input_glob: str) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for csv_path in glob.glob(input_glob):
        df = pd.read_csv(csv_path)
        if "model" not in df.columns:
            model = os.path.basename(os.path.dirname(csv_path.rstrip("/")))
            df["model"] = model

        if "human_ear_time_sig" not in df.columns:
            df["human_ear_time_sig"] = np.nan

        for c in ["desired_time_sig", "actual_time_sig", "human_ear_time_sig", "prompt_type", "model"]:
            if c in df.columns:
                df[c] = _normalize_sig(df[c])

        if "confidence" not in df.columns:
            df["confidence"] = np.nan

        df["source_csv"] = csv_path
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No CSV found with pattern: {input_glob}")

    data = pd.concat(dfs, ignore_index=True)
    required = ["model", "desired_time_sig", "actual_time_sig", "prompt_type", "confidence", "human_ear_time_sig"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return data


def apply_truth_mode(data: pd.DataFrame, mode: str, threshold: float) -> pd.DataFrame:
    out = data.copy()
    desired = out["desired_time_sig"]
    actual = out["actual_time_sig"]
    human = out["human_ear_time_sig"]
    conf = pd.to_numeric(out["confidence"], errors="coerce")

    if mode == "madmom":
        out["truth_time_sig"] = actual
        out["truth_source"] = "madmom"
    elif mode == "human":
        out["truth_time_sig"] = np.where(human.notna(), human, actual)
        out["truth_source"] = np.where(human.notna(), "human", "madmom_fallback")
    elif mode == "hybrid":
        use_human = (conf < threshold) & human.notna()
        out["truth_time_sig"] = np.where(use_human, human, actual)
        out["truth_source"] = np.where(use_human, "human_low_conf", "madmom")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    out["truth_time_sig"] = _normalize_sig(out["truth_time_sig"])
    out["correct_eval"] = (desired == out["truth_time_sig"]).astype(float)
    return out


def plot_confusion(data: pd.DataFrame, outdir: str, mode: str) -> str:
    models = data["model"].dropna().unique().tolist()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        df_m = data[data["model"] == model]
        cm = (
            pd.crosstab(df_m["desired_time_sig"], df_m["truth_time_sig"])
            .reindex(index=TIME_SIGS, columns=TIME_SIGS, fill_value=0)
        )
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5, cbar=False, square=True)
        ax.set_title(model)
        ax.set_xlabel("Truth Label")
        ax.set_ylabel("Desired")

    plt.suptitle(f"Confusion Matrix (mode={mode})", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, f"confusion_matrix_{mode}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rq1_prompt_type(data: pd.DataFrame, outdir: str, mode: str) -> str:
    acc_by_type = (
        data.groupby(["model", "desired_time_sig", "prompt_type"], dropna=False)["correct_eval"]
        .mean()
        .reset_index()
    )

    g = sns.catplot(
        data=acc_by_type,
        x="desired_time_sig",
        y="correct_eval",
        hue="prompt_type",
        col="model",
        kind="bar",
        palette={"cultural": "#4C72B0", "formal": "#DD8452"},
        height=5,
        aspect=1.1,
        order=TIME_SIGS,
    )
    g.set_axis_labels("Time Signature", "Accuracy")
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1))
    plt.suptitle(f"RQ1: Accuracy by Prompt Type (mode={mode})", y=1.02, fontsize=14)
    path = os.path.join(outdir, f"rq1_prompt_type_{mode}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    return path


def plot_collapse_to_44(data: pd.DataFrame, outdir: str, mode: str) -> str:
    non_44 = data[data["desired_time_sig"] != "4/4"].copy()
    non_44["collapsed_to_44"] = non_44["truth_time_sig"] == "4/4"

    collapse_rate = (
        non_44.groupby(["model", "desired_time_sig"], dropna=False)["collapsed_to_44"]
        .mean()
        .reset_index()
    )

    g = sns.catplot(
        data=collapse_rate,
        x="desired_time_sig",
        y="collapsed_to_44",
        col="model",
        kind="bar",
        color="#c44e52",
        height=5,
        aspect=1.1,
        order=[s for s in TIME_SIGS if s != "4/4"],
    )
    g.set_axis_labels("Desired Time Signature", "Collapse-to-4/4 Rate")
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1))
    plt.suptitle(f"RQ2: Collapse to 4/4 Rate (mode={mode})", y=1.02, fontsize=14)
    path = os.path.join(outdir, f"rq2_collapse_{mode}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    return path


def build_metrics(data: pd.DataFrame, mode: str, threshold: float) -> pd.DataFrame:
    rows = []
    for model, df_m in data.groupby("model"):
        n = len(df_m)
        overall_acc = float(df_m["correct_eval"].mean()) if n else np.nan
        coverage_human = float(df_m["human_ear_time_sig"].notna().mean()) if n else np.nan

        c_mask = df_m["prompt_type"] == "cultural"
        f_mask = df_m["prompt_type"] == "formal"
        c_acc = float(df_m.loc[c_mask, "correct_eval"].mean()) if c_mask.any() else np.nan
        f_acc = float(df_m.loc[f_mask, "correct_eval"].mean()) if f_mask.any() else np.nan

        non44 = df_m[df_m["desired_time_sig"] != "4/4"]
        collapse_44 = float((non44["truth_time_sig"] == "4/4").mean()) if len(non44) else np.nan

        if "confidence" in df_m.columns:
            low_conf_mask = pd.to_numeric(df_m["confidence"], errors="coerce") < threshold
            low_conf_acc = float(df_m.loc[low_conf_mask, "correct_eval"].mean()) if low_conf_mask.any() else np.nan
            high_conf_acc = float(df_m.loc[~low_conf_mask, "correct_eval"].mean()) if (~low_conf_mask).any() else np.nan
        else:
            low_conf_acc = np.nan
            high_conf_acc = np.nan

        row = {
            "mode": mode,
            "confidence_threshold": threshold,
            "model": model,
            "n_samples": n,
            "overall_accuracy": overall_acc,
            "cultural_accuracy": c_acc,
            "formal_accuracy": f_acc,
            "cultural_minus_formal": c_acc - f_acc if pd.notna(c_acc) and pd.notna(f_acc) else np.nan,
            "collapse_to_44_rate_non44": collapse_44,
            "human_label_coverage": coverage_human,
            "low_conf_accuracy": low_conf_acc,
            "high_conf_accuracy": high_conf_acc,
        }

        for ts in TIME_SIGS:
            mask = df_m["desired_time_sig"] == ts
            row[f"acc_{ts.replace('/', '_')}"] = float(df_m.loc[mask, "correct_eval"].mean()) if mask.any() else np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze time-signature results with selectable truth label policy.")
    parser.add_argument("--input-glob", default="audio/*/results.csv", help="Glob for model result CSV files")
    parser.add_argument("--truth-mode", choices=["madmom", "human", "hybrid"], default="madmom")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Used only for hybrid mode")
    parser.add_argument("--outdir", default="figures", help="Directory to save plots")
    parser.add_argument("--metrics-csv", default=None, help="Output path for performance metrics CSV")
    parser.add_argument("--save-eval-csv", default=None, help="Optional output path for merged evaluated rows")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw = load_data(args.input_glob)
    data = apply_truth_mode(raw, args.truth_mode, args.confidence_threshold)

    cm_path = plot_confusion(data, args.outdir, args.truth_mode)
    rq1_path = plot_rq1_prompt_type(data, args.outdir, args.truth_mode)
    collapse_path = plot_collapse_to_44(data, args.outdir, args.truth_mode)

    metrics = build_metrics(data, args.truth_mode, args.confidence_threshold)
    metrics_csv = args.metrics_csv or os.path.join(args.outdir, f"model_performance_{args.truth_mode}.csv")
    metrics.to_csv(metrics_csv, index=False)

    if args.save_eval_csv:
        data.to_csv(args.save_eval_csv, index=False)

    print(f"Loaded {len(data)} rows | models: {sorted(data['model'].dropna().unique().tolist())}")
    print(f"Truth mode: {args.truth_mode}")
    if args.truth_mode == "hybrid":
        print(f"Hybrid threshold: confidence < {args.confidence_threshold} -> human label")
    print(f"Saved: {cm_path}")
    print(f"Saved: {rq1_path}")
    print(f"Saved: {collapse_path}")
    print(f"Saved metrics CSV: {metrics_csv}")


if __name__ == "__main__":
    main()
