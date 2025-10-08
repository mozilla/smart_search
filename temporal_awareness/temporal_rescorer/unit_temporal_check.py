
HISTORY_CSV = "./data/identical_with_diff_time_history.csv"
GOLDEN_CSV = "./data/identical_with_diff_time_golden_queries.csv"
CURRENT_DT = "2025-09-29"
ALPHA_TEMPORAL = 1.0
LANG = "en"
TOP_K = 200
SAVE_DIR = "./unit_temporal_check_results"

import csv
from pathlib import Path
import pandas as pd
from parser import parse_time_window
from ranking import rank_results


def load_history_csv(path):
    df = pd.read_csv(path)
    header = ["url", "title", "description", "frecency", "last_visit_date", "last_visit_date_str"]
    return df[header].to_dict(orient="records")

def load_golden_group(path):
    df = pd.read_csv(path)
    group = {}
    for q, sub in df.groupby("search_query"):
        group[q] = set(sub["url"].tolist())
    return group

def main():
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    details_dir = save_dir / "per_query_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    incorrect_details_dir = save_dir / "per_query_details_incorrect"
    incorrect_details_dir.mkdir(parents=True, exist_ok=True)

    histories = load_history_csv(HISTORY_CSV)
    queries = load_golden_group(GOLDEN_CSV)

    agg_rows = []

    for query in sorted(queries.keys()):

        gt_results = queries[query]
        parse_ok = parse_time_window(query, CURRENT_DT, LANG) is not None

        ranked = rank_results(
            histories=histories,
            query=query,
            current_dt=CURRENT_DT,
            alpha_temporal=ALPHA_TEMPORAL,
            lang=LANG,
        )
        ranked_urls = [r["url"] for r in ranked]

        # unit test: top-W set must equal in-window set
        W = len(gt_results)
        topW = ranked_urls[:W]
        topW_set = set(topW)
        in_topW = len(topW_set & gt_results)

        if topW_set == gt_results:
            unit_pass = True
            unit_reason = ""
        else:
            unit_pass = False
            if parse_ok is False:
                unit_reason = "parser failed to resolve window"
            else:
                out_in_topW = len([u for u in topW if u not in gt_results])
                unit_reason = f"boundary/ranking: {in_topW}/{W} in-window in top-W; {out_in_topW} out-of-window ahead"

        agg_rows.append({
            "query": query,
            "ground_truth_count": len(gt_results),
            "can_parse_window": parse_ok,
            "unit_pass": unit_pass,
            "unit_reason": unit_reason,
            "in_topW": in_topW,
            "W": W,
        })

        ranks = ranked[:TOP_K]
        ranks.sort(key=lambda r: r["title"])
        if ranks:
            with open(details_dir / f"{query.replace(' ','_').replace('/','-')}_top{TOP_K}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(ranks[0].keys()))
                w.writeheader()
                w.writerows(ranks)
            if not unit_pass:
                with open(incorrect_details_dir / f"{query.replace(' ', '_').replace('/', '-')}_top{TOP_K}.csv", "w", newline="",
                          encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(ranks[0].keys()))
                    w.writeheader()
                    w.writerows(ranks)


    agg_df = pd.DataFrame(agg_rows).sort_values(by=["unit_pass","query"], ascending=[True, True])
    agg_df.to_csv(save_dir / "aggregate_unit.csv", index=False)


    total = len(agg_df)
    passed = int((agg_df["unit_pass"] == True).sum()) if total else 0
    print(f"\nUnit pass rate: {passed}/{total} ({(passed/total*100 if total else 0):.1f}%)")
    if total and (passed < total):
        print("\nFailures:")
        for _, r in agg_df[agg_df['unit_pass'] == False].iterrows():
            print(f" - {r['query']}: {r['unit_reason']} (W={r['W']}, in_topW={r['in_topW']})")
    print(f"\nSaved to file: {save_dir/'aggregate_unit.csv'}")
    print(f"Details in folder: {details_dir}")
    print(f"Details for incorrect results in folder: {incorrect_details_dir}")

if __name__ == "__main__":
    main()
