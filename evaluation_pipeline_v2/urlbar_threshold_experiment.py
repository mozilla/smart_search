"""URL-bar semantic + exact-match history threshold sweep.

Same dataset and Recall@1 / Precision metrics as
``evaluation_pipeline_v2.threshold_experiment``, but the query is run the way a user
typing in the address bar would experience it: through the real
UrlbarProvidersManager pipeline, restricted to the history providers
(UrlbarProviderPlaces for exact/token matching + UrlbarProviderSemanticHistorySearch
for semantic history), with the muxer combining and deduping their results.

The semantic distance threshold is swept exactly as before; the difference is
that exact-match history results are now mixed in, so the resulting
recall/precision-vs-threshold charts show how exact matching changes things.

Output is written in the same log format, so plot_threshold_results.py works
unchanged.
"""

from __future__ import annotations

import argparse
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
    write_log,
)

logger = logging.getLogger("urlbar-threshold-experiment")


def run_experiment(
    firefox: FirefoxDriver,
    tables: Dict[str, Any],
    profiles: List[str],
    thresholds: List[float],
    variant: str,
    max_queries: Optional[int],
    max_docs: Optional[int],
    include_temporal: bool,
    include_multi: bool,
    exact_only: bool = False,
) -> Dict[Tuple[float, str, str], List[Dict[str, Any]]]:
    """Run the sweep through the URL bar pipeline.

    With ``exact_only=True`` the semantic provider is disabled (no embeddings),
    leaving the generic exact/token-match history provider only -- the system's
    default behaviour. This is threshold-independent, so a single pseudo-
    threshold of 0.0 is used.
    """
    results: Dict[Tuple[float, str, str], List[Dict[str, Any]]] = defaultdict(list)
    sweep = [0.0] if exact_only else thresholds

    for idx, profile in enumerate(profiles):
        plan = build_profile_plan(
            tables, profile, variant, max_queries, max_docs, include_temporal, include_multi
        )
        if plan is None:
            logger.warning("Profile %s has no usable queries; skipping.", profile)
            continue

        # Isolate each persona: clear the prior profile's history + vector DB.
        # (In exact-only mode the embed=False load clears history itself.)
        if idx > 0 and not exact_only:
            firefox.semantic_history_cleanup()

        logger.info(
            "Profile %s: loading %d docs, %d queries%s",
            profile,
            len(plan["docs"]),
            len(plan["queries"]),
            " (exact-match only)" if exact_only else "",
        )
        row_limit = max(10000, len(plan["docs"]) + 100)
        # Load at a permissive threshold; the urlbar wrapper applies the sweep.
        # For the exact-only baseline, skip embedding entirely.
        load_res = firefox.semantic_history_load_profile(
            plan["docs"], row_limit=row_limit, distance_threshold=1.0, embed=not exact_only
        )
        if not load_res.get("ok"):
            logger.warning("Profile %s load skipped: %s", profile, load_res.get("reason"))
            continue
        logger.info("  embedded %s/%s docs", load_res.get("embedded"), load_res.get("inserted"))

        if exact_only:
            firefox.urlbar_exact_setup()
        else:
            setup = firefox.urlbar_semantic_setup(threshold=max(thresholds))
            if not setup.get("ok"):
                logger.warning("Profile %s urlbar setup failed: %s", profile, setup.get("reason"))
                continue

        for threshold in sweep:
            if not exact_only:
                firefox.urlbar_set_threshold(threshold)
            for spec in plan["queries"]:
                qres = firefox.urlbar_query(spec["query"], exact_only=exact_only)
                hits = qres.get("results", [])
                relevant = spec["relevant_urls"]
                matched = bool(hits) and hits[0].get("url") in relevant
                rel_retrieved = sum(1 for r in hits if r.get("url") in relevant)
                results[(threshold, spec["query_type"], spec["doc_type"])].append(
                    {
                        "query": spec["query"],
                        "matched": matched,
                        "results": hits,
                        "retrieved": len(hits),
                        "rel_retrieved": rel_retrieved,
                    }
                )
            logger.info("  threshold %s done", threshold)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firefox-bin", type=Path, default=None)
    parser.add_argument("--profile-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("urlbar_threshold_test.log"))
    parser.add_argument("--max-profiles", type=int, default=3)
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--max-docs", type=int, default=800)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--include-temporal", action="store_true")
    parser.add_argument("--include-multi", action="store_true", default=True)
    parser.add_argument("--no-include-multi", dest="include_multi", action="store_false")
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Disable semantic; exact/token-match history only (the default-behaviour baseline).",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="[%(name)s] %(message)s")

    tables = load_dataset_tables(cache_dir=args.cache_dir)
    profiles = select_profiles(tables, args.max_profiles, args.profiles, args.variant)
    if not profiles:
        raise SystemExit("No matching profiles found in dataset.")
    logger.info("Running profiles: %s", profiles)

    firefox = FirefoxDriver(
        model="",
        auth_config={},
        headless=args.headless,
        firefox_bin=args.firefox_bin,
        profile_path=args.profile_path,
        prompt_path="",
        skip_fastly=False,
    )
    try:
        results = run_experiment(
            firefox,
            tables,
            profiles,
            args.thresholds,
            args.variant,
            args.max_queries,
            args.max_docs,
            args.include_temporal,
            args.include_multi,
            exact_only=args.exact_only,
        )
    finally:
        firefox.quit()

    write_log(results, [0.0] if args.exact_only else args.thresholds, args.out)


if __name__ == "__main__":
    main()
