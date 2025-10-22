
import os
import glob
import pandas as pd
from pathlib import Path
from html import escape
from datetime import datetime

def get_model_name(path):
    name = os.path.basename(path)
    tok = "_traditional_eval_"
    return name.split(tok)[0] if tok in name else name.replace(".csv", "")

def df_to_html_table(df):
    if df.empty:
        return "<p><em>No data.</em></p>"
    cols = list(df.columns)
    for pin in ["profile", "model_name"]:
        if pin in cols:
            cols.insert(0, cols.pop(cols.index(pin)))
    df = df[cols]
    ths = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    trs = []
    for _, row in df.iterrows():
        tds = "".join(f"<td>{escape(str(v))}</td>" for v in row.values)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{''.join(trs)}</tbody></table>"

def build_aggregate_report(base_dir="results", out_html="eval_report_aggregate_only.html"):
    style = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
      h1 { margin-bottom: 0; }
      .subtitle { color: #555; margin-top: 4px; margin-bottom: 24px; }
      h2 { border-top: 2px solid #ddd; padding-top: 10px; margin-top: 30px; }
      table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 14px; }
      th { background: #f7f7f7; text-align: left; }
      code { background: #f4f4f4; padding: 1px 4px; border-radius: 4px; }
    </style>
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Aggregate Evaluation Report</title>",
        style,
        "</head><body>",
        "<h1>Aggregate Evaluation Report</h1>",
        f"<div class='subtitle'>Generated at {escape(generated_at)}</div>",
        f"<div class='subtitle'>Scanning <code>{escape(os.path.abspath(base_dir))}</code></div>",
    ]

    parts += [
        "<h2>Overview</h2>",
        "<table style='width:100%; border-collapse:collapse'><tr>",
        f"<td style='width:50%; padding-right:8px; vertical-align:top;'>"
        f"<img src='per_profile_grouped_bar.png' style='width:100%; height:auto; border:1px solid #eee; border-radius:8px;'></td>",
        f"<td style='width:50%; padding-left:8px; vertical-align:top;'>"
        f"<img src='per_query_grouped_bar.png' style='width:100%; height:auto; border:1px solid #eee; border-radius:8px;'></td>",
        "</tr></table>",
    ]

    base = Path(base_dir)
    profiles = sorted([p.name for p in base.iterdir()
                       if (base / p.name).is_dir() and (base / p.name / "evaluation_results").exists()]) if base.exists() else []

    if not profiles:
        parts.append("<p><em>No profiles found (expected results/&lt;profile&gt;/evaluation_results/).</em></p>")
    else:
        for profile in profiles:
            parts.append(f"<h2>Profile: {escape(profile)}</h2>")
            eval_dir = base / profile / "evaluation_results"
            csv_paths = sorted(glob.glob(str(eval_dir / "*_traditional_eval_aggregate_metrics.csv")))
            if not csv_paths:
                parts.append("<p><em>No aggregate CSVs for this profile.</em></p>")
                continue
            for p in csv_paths:
                try:
                    df = pd.read_csv(p)
                    if "profile" not in df.columns:
                        df["profile"] = profile
                    if "model_name" not in df.columns:
                        df["model_name"] = get_model_name(p)
                    parts.append(f"<h3>Model: {escape(get_model_name(p))}</h3>")
                    parts.append(df_to_html_table(df))
                    parts.append(f"<p style='color:#666;font-size:12px;'>Source: {escape(p)}</p>")
                except Exception as e:
                    parts.append(f"<p style='color:#b00;'>Failed to read {escape(p)}: {escape(str(e))}</p>")

    parts.append("</body></html>")
    Path(out_html).write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved to {out_html}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="results")
    ap.add_argument("--out_html", type=str, default="eval_report.html")
    args = ap.parse_args()
    build_aggregate_report(base_dir=args.base_dir, out_html=args.out_html)
