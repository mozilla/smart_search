"""Overlay two threshold-experiment logs (e.g. semantic-only vs URL-bar combined).

Draws one figure per metric where each query_type->doc_type bucket gets a single
color, the first log is drawn dashed and the second solid, so the effect of the
second pipeline (e.g. adding exact-match history) is visible per bucket.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .plot_threshold_results import make_pivot, parse_log

BUCKET_COLORS = {
    ("longQ", "longD"): "tab:blue",
    ("longQ", "shortD"): "tab:orange",
    ("shortQ", "longD"): "tab:green",
    ("shortQ", "shortD"): "tab:red",
}

METRIC_YLABEL = {"recall": "Mean Recall@1", "precision": "Mean Precision"}


def plot_comparison(log_a, label_a, log_b, label_b, metric, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pivot_a = make_pivot(parse_log(log_a), metric)
    pivot_b = make_pivot(parse_log(log_b), metric)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for (qt, dt), color in BUCKET_COLORS.items():
        if (qt, dt) in pivot_a.columns:
            s = pivot_a[(qt, dt)]
            ax.plot(s.index, s.values, color=color, linestyle="--", marker="o", alpha=0.9)
        if (qt, dt) in pivot_b.columns:
            s = pivot_b[(qt, dt)]
            ax.plot(s.index, s.values, color=color, linestyle="-", marker="s", alpha=0.9)

    # Two legends: one for bucket color, one for line style (which run).
    bucket_handles = [
        Line2D([], [], color=c, marker="o", linestyle="-", label=f"{qt} → {dt}")
        for (qt, dt), c in BUCKET_COLORS.items()
    ]
    style_handles = [
        Line2D([], [], color="gray", linestyle="--", marker="o", label=label_a),
        Line2D([], [], color="gray", linestyle="-", marker="s", label=label_b),
    ]
    legend1 = ax.legend(handles=bucket_handles, title="query_type → doc_type", loc="lower right")
    ax.add_artist(legend1)
    ax.legend(handles=style_handles, title="pipeline", loc="upper left")

    ax.set_title(title)
    ax.set_xlabel("Distance Threshold")
    ax.set_ylabel(METRIC_YLABEL[metric])
    ax.grid(True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved {metric} comparison to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_a", type=Path, help="First log (drawn dashed).")
    parser.add_argument("log_b", type=Path, help="Second log (drawn solid).")
    parser.add_argument("--label-a", default="semantic only")
    parser.add_argument("--label-b", default="URL bar (combined)")
    parser.add_argument("--metric", choices=["recall", "precision"], default="recall")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    out = args.out or Path(f"comparison_{args.metric}.png")
    title = args.title or f"{METRIC_YLABEL[args.metric]} vs Threshold: {args.label_a} vs {args.label_b}"
    plot_comparison(args.log_a, args.label_a, args.log_b, args.label_b, args.metric, out, title)


if __name__ == "__main__":
    main()
