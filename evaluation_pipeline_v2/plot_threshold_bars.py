"""Bar chart of Recall@1 and Precision per query/doc bucket for a single config.

Intended for the exact-match-only baseline (threshold-independent), but works on
any experiment log at a chosen threshold. Useful as the reference the semantic
recall/precision-vs-threshold curves are compared against.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_driver.jobs.plot_threshold_results import parse_log

BUCKET_ORDER = [
    ("shortQ", "shortD"),
    ("shortQ", "longD"),
    ("longQ", "shortD"),
    ("longQ", "longD"),
]


def plot_bars(log_path, threshold, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = parse_log(log_path)
    if threshold is None:
        threshold = sorted(df["threshold"].unique())[0]
    df = df[df["threshold"] == threshold]

    labels, recalls, precisions = [], [], []
    for qt, dt in BUCKET_ORDER:
        row = df[(df["query_type"] == qt) & (df["doc_type"] == dt)]
        if row.empty:
            continue
        labels.append(f"{qt}\n→ {dt}")
        recalls.append(float(row["recall"].iloc[0]))
        precisions.append(
            float(row["precision"].iloc[0]) if row["precision"].notna().iloc[0] else 0.0
        )

    import numpy as np

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_r = ax.bar(x - width / 2, recalls, width, label="Recall@1", color="tab:blue")
    bars_p = ax.bar(x + width / 2, precisions, width, label="Precision", color="tab:orange")

    for bars in (bars_r, bars_p):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved bar chart to {out_path}")

    print(f"\n{'bucket':<18}{'recall':>10}{'precision':>12}")
    for lbl, r, p in zip(labels, recalls, precisions):
        print(f"{lbl.replace(chr(10), ' '):<18}{r:>10.3f}{p:>12.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold block to plot (default: the only/first one).",
    )
    parser.add_argument("--out", type=Path, default=Path("threshold_bars.png"))
    parser.add_argument(
        "--title", default="Exact-match-only baseline (semantic off): Recall@1 & Precision"
    )
    args = parser.parse_args()
    plot_bars(args.log, args.threshold, args.out, args.title)


if __name__ == "__main__":
    main()
