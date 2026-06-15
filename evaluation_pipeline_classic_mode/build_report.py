"""Assemble a self-contained HTML report from threshold-experiment artifacts.

Embeds PNG charts as base64 (so the file is portable) and renders Recall@1 /
Precision tables straight from the experiment logs using the same parser as the
plotters. Section layout is defined in ``main`` and references explicit paths,
so it can be re-pointed at any run's outputs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .plot_threshold_results import make_pivot, parse_log

UNBOUND = 10_000


def embed_image(path: Path) -> str:
    if not path.exists():
        return f"<p><em>[missing image: {html.escape(str(path))}]</em></p>"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.name)}" />'


def pivot_table_html(log_path: Path, metric: str) -> str:
    if not log_path.exists():
        return f"<p><em>[missing log: {html.escape(str(log_path))}]</em></p>"
    pivot = make_pivot(parse_log(log_path), metric)
    pct = (pivot * 100).round(1)
    pct.index.name = "threshold"
    pct.columns = [f"{qt}→{dt}" for qt, dt in pct.columns]
    return pct.to_html(
        classes="data", border=0, float_format=lambda v: f"{v:.1f}%", na_rep="—"
    )


def baseline_table_html(log_path: Path) -> str:
    """Single-row baseline table: recall & precision per bucket (one threshold)."""
    if not log_path.exists():
        return f"<p><em>[missing log: {html.escape(str(log_path))}]</em></p>"
    df = parse_log(log_path)
    t = sorted(df["threshold"].unique())[0]
    df = df[df["threshold"] == t]
    rows = []
    for _, r in df.iterrows():
        bucket = f"{r['query_type']}→{r['doc_type']}"
        rec = f"{r['recall'] * 100:.1f}%"
        prec = "—" if pd.isna(r["precision"]) else f"{r['precision'] * 100:.1f}%"
        rows.append((bucket, rec, prec))
    body = "".join(
        f"<tr><td>{html.escape(b)}</td><td>{rc}</td><td>{pc}</td></tr>" for b, rc, pc in rows
    )
    return (
        '<table class="data" border="0"><thead><tr>'
        "<th>bucket</th><th>Recall@1</th><th>Precision</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def displacement_table_html(csv_path: Path, key_label: str, key_is_length: bool) -> str:
    """Pool a urlbar_displacement_experiment CSV across buckets and render
    recall / precision / eviction / exact-match prevalence per sweep key."""
    if not csv_path.exists():
        return f"<p><em>[missing csv: {html.escape(str(csv_path))}]</em></p>"
    agg: dict = defaultdict(lambda: defaultdict(float))
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            a = agg[float(r["sweep"])]
            for col in ("n_queries", "correct", "recall_total", "rel_retrieved",
                        "retrieved", "n_with_exact", "exact_total", "evicted"):
                a[col] += float(r[col])

    def rate(n, d):
        return f"{n / d * 100:.1f}%" if d else "—"

    def fmt_key(k):
        if key_is_length:
            return "unbound" if k >= UNBOUND else f"{k:g}"
        return f"{k:g}"

    rows = []
    for k in sorted(agg):
        a = agg[k]
        rows.append(
            f"<tr><td>{fmt_key(k)}</td>"
            f"<td>{rate(a['correct'], a['recall_total'])}</td>"
            f"<td>{rate(a['rel_retrieved'], a['retrieved'])}</td>"
            f"<td>{rate(a['evicted'], a['exact_total'])}</td>"
            f"<td>{int(a['n_with_exact'])}/{int(a['n_queries'])}</td></tr>"
        )
    return (
        '<table class="data" border="0"><thead><tr>'
        f"<th>{html.escape(key_label)}</th><th>Recall@1</th><th>Precision</th>"
        "<th>Eviction@3</th><th>queries w/ exact match</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 1100px; padding: 32px 24px; color: #1a1a1a; line-height: 1.5; }
h1 { border-bottom: 3px solid #0060df; padding-bottom: 8px; }
h2 { margin-top: 40px; border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #0a3d91; }
h3 { margin-top: 24px; color: #333; }
img { max-width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 6px; margin: 8px 0; }
table.data { border-collapse: collapse; margin: 12px 0 20px; font-size: 14px; }
table.data th, table.data td { border: 1px solid #d0d0d0; padding: 5px 12px; text-align: right; }
table.data th { background: #f0f4fa; }
table.data td:first-child, table.data th:first-child { text-align: left; }
.meta { background: #f7f7f9; border-left: 4px solid #0060df; padding: 10px 16px; border-radius: 4px; }
.charts { display: flex; flex-wrap: wrap; gap: 16px; }
.charts > div { flex: 1 1 480px; }
.tables { display: flex; flex-wrap: wrap; gap: 28px; }
.note { color: #666; font-size: 13px; }
"""


def build_report(out_path: Path, title: str, intro_html: str, sections: list) -> None:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        intro_html,
    ]
    for sec in sections:
        parts.append(f"<h2>{html.escape(sec['title'])}</h2>")
        if sec.get("desc"):
            parts.append(sec["desc"])
        charts = sec.get("charts", [])
        if charts:
            parts.append('<div class="charts">')
            for caption, img in charts:
                parts.append(
                    f"<div><h3>{html.escape(caption)}</h3>{embed_image(Path(img))}</div>"
                )
            parts.append("</div>")
        tables = sec.get("tables", [])
        if tables:
            parts.append('<div class="tables">')
            for caption, table_html in tables:
                parts.append(f"<div><h3>{html.escape(caption)}</h3>{table_html}</div>")
            parts.append("</div>")
    parts.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote report to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/semantic_history_report.html"))
    args = parser.parse_args()
    d = args.artifact_dir

    intro = """
    <div class="meta">
    <strong>Pipeline:</strong> Mozilla/history-search-retrieval dataset → loaded into a
    Firefox build's Places via <code>ml_driver</code> → embeddings generated →
    <code>PlacesSemanticHistoryManager.infer()</code> (semantic) and the URL-bar
    providers manager (combined). <strong>Scale:</strong> 5 personas, up to 40
    queries each, 150 docs/persona. <strong>Metrics:</strong> Recall@1 = top-1
    result is a qrels-relevant doc; Precision = relevant retrieved / retrieved
    (micro-averaged), over the threshold-filtered result set. Buckets are
    queryType (short/long by length) × docType (short/long by title length).
    </div>
    """

    sections = [
        {
            "title": "1. Semantic history search (static model) — threshold sweep",
            "desc": "<p>Recall@1 rises with the distance threshold; precision falls as the "
            "second nearest neighbor starts clearing the bar.</p>",
            "charts": [
                ("Recall@1 vs threshold", d / "threshold_recall.png"),
                ("Precision vs threshold", d / "threshold_precision.png"),
            ],
            "tables": [
                ("Recall@1", pivot_table_html(d / "threshold_results.log", "recall")),
                ("Precision", pivot_table_html(d / "threshold_results.log", "precision")),
            ],
        },
        {
            "title": "2. Exact-match-only baseline (semantic off — current default)",
            "desc": "<p>The generic history provider requires all query tokens to appear in a "
            "history entry; on natural-language queries it almost never fires, so recall is "
            "≈0. This is the do-nothing baseline that motivates semantic search. "
            "<span class='note'>(Precision is undefined where no results were returned; shown as 0.)</span></p>",
            "charts": [("Recall@1 & Precision per bucket", d / "exact_only_bars.png")],
            "tables": [("Baseline", baseline_table_html(d / "exact_only.log"))],
        },
        {
            "title": "3. URL bar (semantic + exact-match history, combined via muxer)",
            "desc": "<p>The real address-bar pipeline: <code>UrlbarProviderPlaces</code> + "
            "<code>UrlbarProviderSemanticHistorySearch</code>, combined and ranked by the muxer "
            "(by frecency, deduped by URL). Combining exact-match nudges recall up at no "
            "precision cost vs semantic-only.</p>",
            "charts": [
                ("Recall: semantic only vs combined", d / "cmp_recall.png"),
                ("Precision: semantic only vs combined", d / "cmp_precision.png"),
            ],
            "tables": [
                ("Combined Recall@1", pivot_table_html(d / "urlbar_results.log", "recall")),
                ("Combined Precision", pivot_table_html(d / "urlbar_results.log", "precision")),
            ],
        },
        {
            "title": "4. Embedding model: static (512-dim) vs contextual (384-dim)",
            "desc": "<p>Set via <code>places.semanticHistory.embeddingType</code>. The contextual "
            "model reaches high recall at a tighter (more precise) threshold than static, "
            "trading some precision back past ~0.6.</p>",
            "charts": [
                ("Recall: static vs contextual", d / "cmp_emb_recall.png"),
                ("Precision: static vs contextual", d / "cmp_emb_precision.png"),
            ],
            "tables": [
                ("Static — Recall@1", pivot_table_html(d / "embtype_static.log", "recall")),
                ("Contextual — Recall@1", pivot_table_html(d / "embtype_contextual.log", "recall")),
                ("Static — Precision", pivot_table_html(d / "embtype_static.log", "precision")),
                ("Contextual — Precision", pivot_table_html(d / "embtype_contextual.log", "precision")),
            ],
        },
        {
            "title": "5. Displacement: do semantic results push exact matches out of view?",
            "desc": "<p>Displacement = <strong>eviction@3</strong>: of the exact/token-match history "
            "results, the fraction pushed out of the combined top-3 by the semantic results "
            "(which rank by frecency, not relevance). On <em>full</em> queries this is ~0 — the "
            "exact-match provider needs all query tokens in a title, so it fires on only 2/200 "
            "queries; there is nothing to displace. The effect appears while typing (next section).</p>",
            "charts": [("Eviction vs distance threshold (full queries)", d / "disp_threshold_eviction.png")],
            "tables": [
                ("Per threshold (pooled)", displacement_table_html(
                    d / "displacement_threshold.csv", "threshold", key_is_length=False)),
            ],
        },
        {
            "title": "6. As-you-type: precision / recall / displacement vs query length",
            "desc": "<p>All query/doc lengths pooled into one dataset; semantic threshold fixed at 0.5, "
            "visible top-3. The query is truncated to N characters to simulate typing. "
            "<strong>Displacement is an early-typing problem:</strong> at 5–8 chars the token "
            "provider returns many history hits and the 2 semantic results evict ~42–46% of them "
            "from the top-3, while precision is very low (semantic on a half-typed query is mostly "
            "noise). By ≥12 chars eviction is 0 and precision climbs above 0.8. Implication: gate "
            "semantic by typed length (raise <code>suggest.semanticHistory.minLength</code>) and/or "
            "reserve slots for exact matches, rather than tuning the distance threshold.</p>",
            "charts": [
                ("Eviction (displacement) vs query length", d / "trunc_eviction.png"),
                ("Precision vs query length", d / "trunc_precision.png"),
                ("Recall@1 vs query length", d / "trunc_recall.png"),
            ],
            "tables": [
                ("Per query length (pooled)", displacement_table_html(
                    d / "displacement_truncation.csv", "chars typed", key_is_length=True)),
            ],
        },
    ]

    build_report(args.out, "Semantic History Search — Threshold Analysis", intro, sections)


if __name__ == "__main__":
    main()
