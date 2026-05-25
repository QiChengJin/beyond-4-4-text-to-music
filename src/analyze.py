import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

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
    prompt_order = ["cultural", "formal"]
    models = data["model"].dropna().unique().tolist()

    # Aggregate counts for point estimate + Fisher exact test
    agg = (
        data.groupby(["model", "desired_time_sig", "prompt_type"], dropna=False)["correct_eval"]
        .agg(k="sum", n="count")
        .reset_index()
    )
    agg["acc"] = np.where(agg["n"] > 0, agg["k"] / agg["n"], np.nan)

    # Ensure all combinations exist so zero-accuracy bars (e.g., 5/4) are rendered and labeled.
    full_index = pd.MultiIndex.from_product(
        [models, TIME_SIGS, prompt_order],
        names=["model", "desired_time_sig", "prompt_type"],
    )
    agg = (
        agg.set_index(["model", "desired_time_sig", "prompt_type"])
        .reindex(full_index)
        .reset_index()
    )
    agg["k"] = agg["k"].fillna(0)
    agg["n"] = agg["n"].fillna(0)
    agg["acc"] = np.where(agg["n"] > 0, agg["k"] / agg["n"], 0.0)

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    colors = {"cultural": "#1B4F72", "formal": "#85929E"}
    bar_w = 0.38
    x = np.arange(len(TIME_SIGS))

    model_pvals = {}
    for ax, model in zip(axes, models):
        m = agg[agg["model"] == model].copy()
        m = m.set_index(["desired_time_sig", "prompt_type"]).sort_index()

        # Fisher exact test on overall cultural vs formal (collapsed across time signatures)
        d_m = data[data["model"] == model]
        c_ok = int((d_m[d_m["prompt_type"] == "cultural"]["correct_eval"] == 1).sum())
        c_bad = int((d_m[d_m["prompt_type"] == "cultural"]["correct_eval"] == 0).sum())
        f_ok = int((d_m[d_m["prompt_type"] == "formal"]["correct_eval"] == 1).sum())
        f_bad = int((d_m[d_m["prompt_type"] == "formal"]["correct_eval"] == 0).sum())
        _, p = fisher_exact([[c_ok, c_bad], [f_ok, f_bad]])
        sig_text = "not significant" if p >= 0.05 else "significant"
        model_pvals[model] = p
        # Per-time-signature Fisher p-values
        per_ts_pvals = {}
        for ts in TIME_SIGS:
            d_ts = d_m[d_m["desired_time_sig"] == ts]
            c_ok_ts = int((d_ts[d_ts["prompt_type"] == "cultural"]["correct_eval"] == 1).sum())
            c_bad_ts = int((d_ts[d_ts["prompt_type"] == "cultural"]["correct_eval"] == 0).sum())
            f_ok_ts = int((d_ts[d_ts["prompt_type"] == "formal"]["correct_eval"] == 1).sum())
            f_bad_ts = int((d_ts[d_ts["prompt_type"] == "formal"]["correct_eval"] == 0).sum())
            _, p_ts = fisher_exact([[c_ok_ts, c_bad_ts], [f_ok_ts, f_bad_ts]])
            per_ts_pvals[ts] = float(p_ts)
        for j, pt in enumerate(prompt_order):
            y = []
            ns = []
            for ts in TIME_SIGS:
                row = m.loc[(ts, pt)]
                yv = float(row["acc"])
                y.append(yv)
                ns.append(int(row["n"]))

            xpos = x + (j - 0.5) * bar_w
            ax.bar(xpos, y, width=bar_w, color=colors[pt], label=pt if model == models[0] else None)

            # Annotate 0% (or n=0) to avoid "empty bar looks like missing data"
            for xi, yv, n in zip(xpos, y, ns):
                if n == 0:
                    ax.text(xi, 0.015, "n=0", ha="center", va="bottom", fontsize=8, color="#666666")
                elif yv == 0:
                    ax.text(xi, 0.015, "0%", ha="center", va="bottom", fontsize=8, color="#333333")

        # Annotate per-meter effect size as simple nearby percentage text.
        # Skip tiny gaps to reduce visual noise.
        min_gap_to_show = 0.03  # 3 percentage points
        for i, ts in enumerate(TIME_SIGS):
            yc = float(m.loc[(ts, "cultural"), "acc"])
            yf = float(m.loc[(ts, "formal"), "acc"])
            gap = abs(yc - yf)
            if gap < min_gap_to_show:
                continue
            gap_pct = gap * 100.0
            x_left = i - bar_w / 2
            x_right = i + bar_w / 2
            y_top = min(max(yc, yf) + 0.045, 1.08)
            # |--| connector
            ax.plot([x_left, x_right], [y_top, y_top], color="#333333", linewidth=1.1)
            ax.plot([x_left, x_left], [y_top - 0.014, y_top], color="#333333", linewidth=1.1)
            ax.plot([x_right, x_right], [y_top - 0.014, y_top], color="#333333", linewidth=1.1)
            ax.text(
                i,
                min(y_top + 0.01, 1.10),
                f"{gap_pct:.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#333333",
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(TIME_SIGS)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("Time Signature")
        ax.set_title(f"{model}\np={p:.3f}, {sig_text}")
        for i, ts in enumerate(TIME_SIGS):
            ax.text(
                i,
                1.04,
                f"p={per_ts_pvals[ts]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444444",
            )

    axes[0].set_ylabel("Accuracy")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in prompt_order]
    fig.legend(handles, prompt_order, title="prompt_type", loc="center right")
    plt.suptitle(
        f"RQ1: Accuracy by Prompt Type (mode={mode})\n"
        "No significant difference between cultural and formal prompts "
        f"(overall Fisher: Lyria p={model_pvals.get('Lyria', float('nan')):.3f}, "
        f"Suno p={model_pvals.get('Suno', float('nan')):.3f})",
        y=1.03,
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    path = os.path.join(outdir, f"rq1_prompt_type_{mode}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_collapse_to_44(data: pd.DataFrame, outdir: str, mode: str) -> str:
    non_44 = data[data["desired_time_sig"] != "4/4"].copy()
    order = [s for s in TIME_SIGS if s != "4/4"]

    # Decompose outcomes for each desired meter:
    # 1) collapsed_to_44
    # 2) correct
    # 3) other_non44_error = 1 - (1) - (2)
    rows = []
    for (model, desired), g in non_44.groupby(["model", "desired_time_sig"], dropna=False):
        collapse_44 = float((g["truth_time_sig"] == "4/4").mean())
        correct = float((g["truth_time_sig"] == g["desired_time_sig"]).mean())
        other_non44_error = max(0.0, 1.0 - collapse_44 - correct)
        rows.append(
            {
                "model": model,
                "desired_time_sig": desired,
                "collapse_44": collapse_44,
                "correct": correct,
                "other_non44_error": other_non44_error,
            }
        )

    dec = pd.DataFrame(rows)
    models = dec["model"].dropna().unique().tolist()

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        d = (
            dec[dec["model"] == model]
            .set_index("desired_time_sig")
            .reindex(order)
            .fillna(0.0)
        )
        x = np.arange(len(order))

        # Bottom-up red: collapsed to 4/4
        ax.bar(x, d["collapse_44"], color="#c44e52", label="Collapse to 4/4")
        # Middle gray: wrong non-4/4 outcomes
        ax.bar(
            x,
            d["other_non44_error"],
            bottom=d["collapse_44"],
            color="#bdbdbd",
            label="Collapse to non-4/4 (wrong)",
        )
        # Top-down blue: correct desired meter
        ax.bar(
            x,
            d["correct"],
            bottom=1.0 - d["correct"],
            color="#4C72B0",
            label="Correct desired meter",
        )

        # Percentage labels for each chunk (skip very tiny chunks to reduce clutter)
        min_label_h = 0.06
        for i, ts in enumerate(order):
            r = float(d.loc[ts, "collapse_44"])
            g = float(d.loc[ts, "other_non44_error"])
            b = float(d.loc[ts, "correct"])

            if r >= min_label_h:
                ax.text(i, r / 2, f"{r*100:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            if g >= min_label_h:
                ax.text(i, r + g / 2, f"{g*100:.0f}%", ha="center", va="center", color="black", fontsize=8)
            if b >= min_label_h:
                ax.text(i, 1.0 - b / 2, f"{b*100:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

            # Emphasize 5/4 and 6/8 collapse-to-4/4 in red segment
            if ts in ("5/4", "6/8"):
                y_in_red = max(0.04, r - 0.04)
                ax.text(
                    i,
                    y_in_red,
                    f"collapse {r*100:.0f}%",
                    ha="center",
                    va="top",
                    color="white",
                    fontsize=11,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.22",
                        facecolor="#8b1a1a",
                        edgecolor="white",
                        linewidth=0.8,
                        alpha=0.95,
                    ),
                )

        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_ylim(0, 1)
        ax.set_title(model)
        ax.set_xlabel("Desired Time Signature")
        for tick in ax.get_xticklabels():
            if tick.get_text() in ("5/4", "6/8"):
                tick.set_color("#8b1a1a")
                tick.set_fontweight("bold")

    axes[0].set_ylabel("Proportion")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
    plt.suptitle(f"RQ2: Outcome Decomposition (mode={mode})", y=1.10, fontsize=14)
    plt.tight_layout()
    path = os.path.join(outdir, f"rq2_collapse_{mode}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
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
