"""Plot Recall@1 vs distance threshold from a threshold-experiment log.

Mirrors smart_search/notebooks/threshold_analysis_from_log_results.ipynb: it
parses the log produced by ``ml_driver.jobs.threshold_experiment`` with the same
regexes, groups by (query_type, doc_type, threshold), and draws the
recall-vs-threshold line plot (one line per query_type -> doc_type).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Same block-level pattern the notebook uses. The optional Precision suffix is
# emitted by ml_driver.jobs.threshold_experiment and ignored by the notebook.
TEST_BLOCK_PATTERN = re.compile(
    r"Running (?P<queryType>\w+) → (?P<docType>\w+) test at threshold (?P<threshold>[\d.]+).*?"
    r"(Matched|No match).*?"
    r"Threshold [\d.]+ \| (?P<profile>profile_.*?\.json) \| "
    r"Recall@1: (?P<correct>\d+)/(?P<total>\d+) \((?P<accuracy>[\d.]+)%\)"
    r"(?: \| Precision: (?P<relRetrieved>\d+)/(?P<retrieved>\d+) \((?P<precision>[\d.]+)%\))?",
    re.DOTALL,
)

VALID_QUERY_TYPES = ["shortQ", "longQ"]
VALID_DOC_TYPES = ["shortD", "longD"]


def parse_log(log_path: Path) -> pd.DataFrame:
    log_text = log_path.read_text(encoding="utf-8")
    rows = []
    for block in TEST_BLOCK_PATTERN.finditer(log_text):
        correct = int(block.group("correct"))
        total = int(block.group("total"))
        retrieved = block.group("retrieved")
        rel_retrieved = block.group("relRetrieved")
        rows.append(
            {
                "query_type": block.group("queryType").strip(),
                "doc_type": block.group("docType").strip(),
                "threshold": float(block.group("threshold")),
                "profile": block.group("profile").strip(),
                "correct": correct,
                "total": total,
                "recall": correct / total if total else 0.0,
                "precision": (
                    int(rel_retrieved) / int(retrieved)
                    if retrieved and int(retrieved)
                    else None
                ),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No test blocks parsed from {log_path}")
    df = df[df["query_type"].isin(VALID_QUERY_TYPES) & df["doc_type"].isin(VALID_DOC_TYPES)]
    return df


def make_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df[metric].isna().all():
        raise SystemExit(
            f"No '{metric}' values in log. Re-run threshold_experiment to emit Precision."
        )
    grouped = (
        df.groupby(["query_type", "doc_type", "threshold"])
        .agg(value=(metric, "mean"))
        .reset_index()
    )
    return grouped.pivot_table(
        index="threshold", columns=["query_type", "doc_type"], values="value"
    )


def plot(pivot: pd.DataFrame, out_path: Path, title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = pivot.plot(marker="o", figsize=(10, 6))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Distance Threshold")
    ax.grid(True)
    ax.legend(title="query_type → doc_type")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


METRIC_LABELS = {
    "recall": ("Mean Recall@1", "Recall vs Threshold for Query/Document Types"),
    "precision": ("Mean Precision", "Precision vs Threshold for Query/Document Types"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to the threshold-experiment log.")
    parser.add_argument("--metric", choices=["recall", "precision"], default="recall")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    ylabel, default_title = METRIC_LABELS[args.metric]
    out = args.out or Path(f"threshold_{args.metric}.png")
    title = args.title or default_title

    df = parse_log(args.log)
    pivot = make_pivot(df, args.metric)
    print(pivot.to_string())
    plot(pivot, out, title, ylabel)


if __name__ == "__main__":
    main()
