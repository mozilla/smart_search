"""Plot metrics from a urlbar_displacement_experiment CSV.

Metrics (computed from pooled or per-bucket counters):
  recall     = correct / recall_total
  precision  = rel_retrieved / retrieved
  eviction   = evicted / exact_total      (displacement: exact results pushed out of top-N)
  rank_delta = rank_delta_sum / rank_delta_n  (mean rank increase of surviving exact results)

Use --sweep-label to label the x-axis ("Distance Threshold" or "Query length (chars)").
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

UNBOUND = 10_000

METRICS = {
    "recall": ("correct", "recall_total", "Mean Recall@1"),
    "precision": ("rel_retrieved", "retrieved", "Mean Precision"),
    "eviction": ("evicted", "exact_total", "Exact-match eviction rate (displacement)"),
    "rank_delta": ("rank_delta_sum", "rank_delta_n", "Mean rank increase of exact match"),
}


def load_rows(csv_path: Path):
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def sweep_value(raw: str) -> float:
    v = float(raw)
    return v


def aggregate(rows, metric, pooled):
    num_f, den_f, _ = METRICS[metric]
    # series[label][sweep_key] = (num, den)
    series = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r in rows:
        key = sweep_value(r["sweep"])
        label = "all" if pooled else f'{r["query_type"]}→{r["doc_type"]}'
        cell = series[label][key]
        cell[0] += float(r[num_f])
        cell[1] += float(r[den_f])
    # to sorted (x, y) with y = num/den (None if den==0)
    out = {}
    for label, by_key in series.items():
        xs = sorted(by_key)
        pts = [(k, (by_key[k][0] / by_key[k][1]) if by_key[k][1] else None) for k in xs]
        out[label] = pts
    return out


def fmt_x(v):
    return "unbound" if v >= UNBOUND else (f"{v:g}")


def plot(csv_path, metric, out_path, title, sweep_label, pooled):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(csv_path)
    series = aggregate(rows, metric, pooled)
    _, _, ylabel = METRICS[metric]

    # Map possibly-large UNBOUND x to a compact tick position.
    all_keys = sorted({sweep_value(r["sweep"]) for r in rows})
    xpos = {k: (max(k2 for k2 in all_keys if k2 < UNBOUND) + 2 if k >= UNBOUND else k) for k in all_keys}

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, pts in sorted(series.items()):
        xs = [xpos[k] for k, y in pts if y is not None]
        ys = [y for _, y in pts if y is not None]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xticks([xpos[k] for k in all_keys])
    ax.set_xticklabels([fmt_x(k) for k in all_keys])
    ax.set_xlabel(sweep_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    if len(series) > 1:
        ax.legend(title="query_type → doc_type")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved {metric} plot to {out_path}")

    # Also print the pooled table for the chosen metric.
    for label, pts in sorted(series.items()):
        print(f"\n{label}:")
        for k, y in pts:
            print(f"  {fmt_x(k):>8}: {'n/a' if y is None else f'{y:.3f}'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--metric", choices=list(METRICS), default="eviction")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--sweep-label", default="Distance Threshold")
    p.add_argument("--pooled", action="store_true", help="Pool all buckets into a single line.")
    args = p.parse_args()

    out = args.out or Path(f"displacement_{args.metric}.png")
    _, _, ylabel = METRICS[args.metric]
    title = args.title or f"{ylabel} vs {args.sweep_label}"
    plot(args.csv, args.metric, out, title, args.sweep_label, args.pooled)


if __name__ == "__main__":
    main()
