"""URL-bar displacement / as-you-type experiment.

Measures how the semantic results interact with the exact/token-match history
results in the combined URL-bar pipeline, by running BOTH the exact-only query
(E) and the combined query (C) for every query and comparing them.

Metrics per query:
  - recall@1   : C[0] is a qrels-relevant doc.
  - precision  : relevant in C / |C| (over the combined returned set).
  - eviction@N : of the exact-match results E, how many are NOT in the combined
                 top-N (pushed out of view by semantic results).
  - rank delta : mean rank increase of exact-match results that survived.

Two sweep modes:
  - threshold  : vary the semantic distance threshold (exact-match is threshold-
                 independent, so E is computed once per query).
  - truncation : fix the threshold and truncate the query to N characters
                 (5,8,12,... and unbound) to simulate the user typing.

Output is a CSV of raw counters per (sweep, query_type, doc_type); pooling and
rate computation happen in plot_displacement.py.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ml_driver.firefox.driver import FirefoxDriver
from .threshold_experiment import (
    DEFAULT_THRESHOLDS,
    build_profile_plan,
    load_dataset_tables,
    select_profiles,
)

logger = logging.getLogger("urlbar-displacement")

# Sentinel sweep key for an untruncated ("unbound") query length.
UNBOUND = 10_000


def _counter() -> Dict[str, float]:
    return {
        "n_queries": 0,
        "correct": 0,
        "recall_total": 0,
        "rel_retrieved": 0,
        "retrieved": 0,
        "n_with_exact": 0,
        "exact_total": 0,
        "evicted": 0,
        "rank_delta_sum": 0.0,
        "rank_delta_n": 0,
    }


def _score(combined: List[Dict[str, Any]], exact: List[Dict[str, Any]], relevant, visible_n):
    """Compute per-query counters for combined results C and exact results E."""
    c_urls = [r.get("url") for r in combined]
    e_urls = [r.get("url") for r in exact]

    correct = 1 if c_urls and c_urls[0] in relevant else 0
    rel_retrieved = sum(1 for u in c_urls if u in relevant)

    evicted = sum(1 for u in e_urls if u not in c_urls[:visible_n])
    rank_delta_sum = 0.0
    rank_delta_n = 0
    for rank_e, u in enumerate(e_urls):
        if u in c_urls:
            rank_delta_sum += c_urls.index(u) - rank_e
            rank_delta_n += 1

    return {
        "correct": correct,
        "recall_total": 1,
        "rel_retrieved": rel_retrieved,
        "retrieved": len(c_urls),
        "n_with_exact": 1 if e_urls else 0,
        "exact_total": len(e_urls),
        "evicted": evicted,
        "rank_delta_sum": rank_delta_sum,
        "rank_delta_n": rank_delta_n,
    }


def _accumulate(agg, key, scored):
    c = agg[key]
    c["n_queries"] += 1
    for k, v in scored.items():
        c[k] += v


def run_experiment(
    firefox: FirefoxDriver,
    tables: Dict[str, Any],
    profiles: List[str],
    *,
    sweep: str,
    thresholds: List[float],
    truncations: List[int],
    fixed_threshold: float,
    visible_n: int,
    variant: str,
    max_queries: Optional[int],
    max_docs: Optional[int],
    include_temporal: bool,
    include_multi: bool,
) -> Dict[Tuple[Any, str, str], Dict[str, float]]:
    # agg[(sweep_key, qt, dt)] = counters
    agg: Dict[Tuple[Any, str, str], Dict[str, float]] = defaultdict(_counter)

    for idx, profile in enumerate(profiles):
        plan = build_profile_plan(
            tables, profile, variant, max_queries, max_docs, include_temporal, include_multi
        )
        if plan is None:
            logger.warning("Profile %s has no usable queries; skipping.", profile)
            continue

        if idx > 0:
            firefox.semantic_history_cleanup()

        logger.info("Profile %s: loading %d docs, %d queries", profile, len(plan["docs"]), len(plan["queries"]))
        row_limit = max(10000, len(plan["docs"]) + 100)
        load_res = firefox.semantic_history_load_profile(plan["docs"], row_limit=row_limit, distance_threshold=1.0)
        if not load_res.get("ok"):
            logger.warning("Profile %s load skipped: %s", profile, load_res.get("reason"))
            continue
        firefox.urlbar_semantic_setup(threshold=max(thresholds + [fixed_threshold]))

        for spec in plan["queries"]:
            query = spec["query"]
            relevant = spec["relevant_urls"]
            bucket = (spec["query_type"], spec["doc_type"])

            if sweep == "threshold":
                # Exact match is threshold-independent: compute E once.
                exact = firefox.urlbar_query(query, exact_only=True).get("results", [])
                for t in thresholds:
                    firefox.urlbar_set_threshold(t)
                    combined = firefox.urlbar_query(query, exact_only=False).get("results", [])
                    _accumulate(agg, (t, *bucket), _score(combined, exact, relevant, visible_n))
            else:  # truncation
                firefox.urlbar_set_threshold(fixed_threshold)
                for length in truncations:
                    q = query if length >= len(query) else query[:length]
                    if not q.strip():
                        continue
                    exact = firefox.urlbar_query(q, exact_only=True).get("results", [])
                    combined = firefox.urlbar_query(q, exact_only=False).get("results", [])
                    key = UNBOUND if length >= UNBOUND else length
                    _accumulate(agg, (key, *bucket), _score(combined, exact, relevant, visible_n))
        logger.info("  profile %s done", profile)

    return agg


def write_csv(agg, out_path: Path, sweep: str) -> None:
    fields = [
        "sweep", "query_type", "doc_type", "n_queries", "correct", "recall_total",
        "rel_retrieved", "retrieved", "n_with_exact", "exact_total", "evicted",
        "rank_delta_sum", "rank_delta_n",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for (sweep_key, qt, dt), c in sorted(agg.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
            w.writerow(
                [sweep_key, qt, dt, c["n_queries"], c["correct"], c["recall_total"],
                 c["rel_retrieved"], c["retrieved"], c["n_with_exact"], c["exact_total"],
                 c["evicted"], c["rank_delta_sum"], c["rank_delta_n"]]
            )
    logger.info("Wrote %s (sweep=%s, %d rows)", out_path, sweep, len(agg))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--firefox-bin", type=Path, default=None)
    p.add_argument("--profile-path", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("displacement.csv"))
    p.add_argument("--sweep", choices=["threshold", "truncation"], default="threshold")
    p.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_THRESHOLDS)
    p.add_argument("--truncations", type=int, nargs="*", default=[5, 8, 12, 14, 16, UNBOUND])
    p.add_argument("--fixed-threshold", type=float, default=0.5, help="Threshold used in truncation sweep.")
    p.add_argument("--visible-n", type=int, default=3, help="Top-N considered 'visible' for eviction.")
    p.add_argument("--max-profiles", type=int, default=5)
    p.add_argument("--profiles", nargs="*", default=None)
    p.add_argument("--max-queries", type=int, default=40)
    p.add_argument("--max-docs", type=int, default=150)
    p.add_argument("--variant", default="base")
    p.add_argument("--include-temporal", action="store_true")
    p.add_argument("--include-multi", action="store_true", default=True)
    p.add_argument("--no-include-multi", dest="include_multi", action="store_false")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", dest="headless", action="store_false")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="[%(name)s] %(message)s")

    tables = load_dataset_tables(cache_dir=args.cache_dir)
    profiles = select_profiles(tables, args.max_profiles, args.profiles, args.variant)
    if not profiles:
        raise SystemExit("No matching profiles found in dataset.")
    logger.info("Sweep=%s, profiles=%s", args.sweep, profiles)

    firefox = FirefoxDriver(
        model="", auth_config={}, headless=args.headless, firefox_bin=args.firefox_bin,
        profile_path=args.profile_path, prompt_path="", skip_fastly=False,
    )
    try:
        agg = run_experiment(
            firefox, tables, profiles,
            sweep=args.sweep, thresholds=args.thresholds, truncations=args.truncations,
            fixed_threshold=args.fixed_threshold, visible_n=args.visible_n, variant=args.variant,
            max_queries=args.max_queries, max_docs=args.max_docs,
            include_temporal=args.include_temporal, include_multi=args.include_multi,
        )
    finally:
        firefox.quit()

    write_csv(agg, args.out, args.sweep)


if __name__ == "__main__":
    main()
