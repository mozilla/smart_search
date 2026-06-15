from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# folder structure: results/<profile>/evaluation_results/*_detailed_metrics.csv
CANDIDATE_METRICS = [
    "precision@2",
    "recall@2",
    "ndcg@2",
    "reciprocal_rank",
    "average_precision",
]


def get_model_name(stem):
    s = stem.replace("_traditional_eval_detailed_metrics", "")
    s = s.replace("_traditional_eval_aggregate_metrics", "")
    return s.replace("_", " ")


def load_all_detailed(base_dir):

    csvs = sorted(base_dir.glob("*/evaluation_results/*_detailed_metrics.csv"))
    if not csvs:
        raise FileNotFoundError(f"No detailed metrics CSVs found under {base_dir}/<profile>/evaluation_results/")

    dfs = []
    for p in csvs:

        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            continue

        ## results/<profile>/evaluation_results/<file>
        profile = p.parent.parent.name
        model = get_model_name(p.stem)
        df["profile"] = profile
        df["model"] = model
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No CSVs could be loaded successfully.")
    all_detailed = pd.concat(dfs, ignore_index=True)

    metric_cols = [c for c in CANDIDATE_METRICS if c in all_detailed.columns]
    if not metric_cols:
        raise ValueError("None of the expected metric columns were found in the CSVs.")
    return all_detailed, metric_cols


def compute_micro_macro(all_detailed, metric_cols):

    # per-query: mean across all queries
    micro = (all_detailed.groupby("model", as_index=False)[metric_cols].mean())
    micro["overall_mean"] = micro[metric_cols].mean(axis=1)

    # per-profile: mean per (model, profile), then mean/std across profiles
    by_profile = (all_detailed.groupby(["model", "profile"], as_index=False)[metric_cols].mean())
    macro_mean = (by_profile.groupby("model", as_index=False)[metric_cols].mean())
    macro_std = (by_profile.groupby("model", as_index=False)[metric_cols].std())
    macro = macro_mean.merge(macro_std, on="model", suffixes=("", "_std"))
    macro["overall_mean"] = macro[metric_cols].mean(axis=1)
    return micro, macro


def plot_grouped_bars_micro(micro, metric_cols, out_png, title="Model Comparison"):

    models = micro["model"].tolist()
    x_metrics = metric_cols
    x = np.arange(len(x_metrics))
    bar_w = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        y = micro.loc[micro["model"] == m, x_metrics].values.flatten().tolist()
        ax.bar(x + i * bar_w, y, width=bar_w, label=m)
        for j, val in enumerate(y):
            ax.text(x[j] + i * bar_w, min(val + 0.02, 1.02), f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + (len(models) - 1) * bar_w / 2)
    ax.set_xticklabels(x_metrics, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def generate_plots(base_dir):
    path = Path(base_dir)
    all_detailed, metric_cols = load_all_detailed(path)
    micro, macro = compute_micro_macro(all_detailed, metric_cols)
    plot_grouped_bars_micro(micro, metric_cols, Path(f"{base_dir}/per_query_grouped_bar.png"), title="Model Comparison (Per-query average across all queries)")
    plot_grouped_bars_micro(macro, metric_cols, Path(f"{base_dir}/per_profile_grouped_bar.png"), title="Model Comparison (Per-profile average across all profiles)")
    print("Saved plots: per_query_grouped_bar.png, per_profile_grouped_bar.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="results")
    args = ap.parse_args()
    generate_plots(base_dir=args.base_dir)


