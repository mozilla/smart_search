"""Top-k evaluation of the Smart Window hybrid history search.

Where ``threshold_experiment`` drives ``PlacesSemanticHistoryManager.infer()`` --
pure semantic, hard-capped at ``k=2``, so only Recall@1 is meaningful -- this
drives ``SearchBrowsingHistory.searchBrowsingHistory()``, which returns up to
``historyLimit`` results. That makes nDCG@10, recall@10 and MRR@10 computable in
the browser, and hence directly comparable to the offline numbers.

Two things to keep in mind when reading the output.

**This measures the product pipeline, not the embedding model.** The hybrid search
runs a semantic leg and a Places keyword leg and fuses them with Reciprocal Rank
Fusion, topping up from a domain fallback if the fused set is short. A model
difference can therefore be masked: if keyword matching already surfaces the gold
URL, RRF ranks it well whichever model ran. The ``keyword-only`` arm exists to
quantify that -- run it and diff.

**Places keyword search matches the URL, not only the title**
(``nsNavHistory.cpp``: ``h.url, page_title, tags``). The corpus keeps its real
upstream URLs while only title and description are translated, so on the French
arm 39% of gold pairs share a token with an English URL (65% on English). That is
free cross-language matching a genuinely French history would not provide. It is
why the keyword-only baseline is not optional.

Usage (from the smart_search repo root):

    python -m evaluation_pipeline_classic_mode.topk_experiment \\
        --firefox-bin "$FF" --language fr \\
        --embedding-types static static-en-fr contextual keyword-only \\
        --max-profiles 5 --max-queries 40 --out results/topk.log
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Allow non-Mozilla model hubs (the en-fr model lives under a personal account).
# Must precede the driver import.
os.environ.setdefault("MOZ_ALLOW_EXTERNAL_ML_HUB", "1")

from ml_driver.firefox.driver import FirefoxDriver  # noqa: E402

from .threshold_experiment import (  # noqa: E402
    EMBEDDING_CONFIGS,
    apply_french_overlay,
    build_profile_plan,
    load_dataset_tables,
    select_profiles,
)

logger = logging.getLogger("topk-experiment")

# Measured behaviour of the semantic leg on this corpus: at 0.6 (the product
# default) only ~0.5 results per query clear the distance filter, so top-k metrics
# would measure the filter rather than the ranking. By 0.9 the result set fills
# historyLimit and stops changing -- 1.2 and 1.5 are identical to 0.9 -- so the
# informative range is 0.6 to 0.9 and there is no point sweeping higher.
DEFAULT_THRESHOLDS = [0.6, 0.75, 0.9]
DEFAULT_K = [1, 3, 5, 10]

# The keyword-only baseline is an arm, not an engine: it gates the semantic leg
# off, so whatever model is configured is irrelevant to the result.
KEYWORD_ONLY = "keyword-only"


# ---------------------------------------------------------------- datasets
def load_shortquery_tables(overlay_dir: Path, language: str) -> Dict[str, pd.DataFrame]:
    """Load the creative-hobbyist short-query set into the upstream table shape.

    Built by multilingual_embeddings_eval/scripts/build_fr_shortquery_eval.py. Its
    queries average 4.1 words against the HF set's 6.4, which matters because
    Places keyword search requires *every* term to match url/title/tags -- on the
    long queries the keyword leg returns nothing at all, so RRF fusion cannot be
    observed.

    `select_profiles` and `build_profile_plan` expect the upstream columns, so
    `profile`, `variant`, `is_temporal`, `is_multi` and `rank` are synthesised.
    Only one profile and one gold doc per query exist here, so those are constants.
    """
    docs = pd.read_parquet(overlay_dir / "docs.parquet").copy()
    queries = pd.read_parquet(overlay_dir / "queries.parquet").copy()
    qrels = pd.read_parquet(overlay_dir / "qrels.parquet").copy()

    suffix = "fr" if language == "fr" else "en"
    docs["title"] = docs[f"title_{suffix}"].fillna("")
    docs["description"] = docs[f"description_{suffix}"].fillna("")
    queries["search_query"] = queries[f"query_{suffix}"]

    # qrels carries only query_id/doc_id/relevance, so its profile is taken from
    # the single profile this set contains.
    profile_id = docs["profile_id"].iloc[0]
    qrels["profile_id"] = profile_id
    for frame in (docs, queries, qrels):
        frame["profile"] = frame["profile_id"]
        frame["variant"] = "base"
    queries["is_temporal"] = False
    queries["is_multi"] = False
    qrels["rank"] = 1

    logger.info(
        "Short-query set (%s text): %d docs, %d queries, %d qrels, profile=%s",
        "French" if suffix == "fr" else "English",
        len(docs), len(queries), len(qrels), docs.profile.iloc[0],
    )
    return {"docs": docs, "queries": queries, "qrels": qrels}


# ---------------------------------------------------------------- metrics
def dcg(gains: List[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_urls: List[str], relevant: set, k: int) -> float:
    gains = [1.0 if u in relevant else 0.0 for u in ranked_urls[:k]]
    ideal = [1.0] * min(len(relevant), k)
    denom = dcg(ideal)
    return dcg(gains) / denom if denom else 0.0


def recall_at_k(ranked_urls: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_urls[:k]) & relevant) / len(relevant)


def precision_at_k(ranked_urls: List[str], relevant: set, k: int) -> float:
    """Fraction of the top-k that are relevant.

    Read this as an over-retrieval / clutter signal, NOT as classic IR precision.
    This dataset labels a single gold document per query, so precision@k is capped
    at 1/k -- 0.10 at k=10 -- and falls as the threshold loosens simply because
    more results come back. For choosing a threshold, prefer the marginal view:
    how much recall each additional returned result buys.
    """
    shown = ranked_urls[:k]
    if not shown:
        return 0.0
    return len([u for u in shown if u in relevant]) / len(shown)


def mrr_at_k(ranked_urls: List[str], relevant: set, k: int) -> float:
    for i, u in enumerate(ranked_urls[:k]):
        if u in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------- run
def run_experiment(
    firefox: FirefoxDriver,
    tables: Dict[str, pd.DataFrame],
    profiles: List[str],
    thresholds: List[float],
    ks: List[int],
    variant: str,
    max_queries: Optional[int],
    max_docs: Optional[int],
    include_temporal: bool,
    include_multi: bool,
    embedding_config: str,
    history_limit: int,
) -> Dict[str, Any]:
    keyword_only = embedding_config == KEYWORD_ONLY
    config = {} if keyword_only else EMBEDDING_CONFIGS[embedding_config]
    embedding_type = config.get("embedding_type")
    region = config.get("region")
    expected_model_id = config.get("model_id")
    common_components = config.get("common_components")
    dtype = config.get("dtype")

    # For the baseline arm the corpus still has to be embedded (the load path is
    # shared), but the search itself runs with the semantic leg gated off.
    load_type = embedding_type or "static"
    load_region = region or "US"

    per_query: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    diag_seen: List[Dict[str, Any]] = []

    for profile in profiles:
        plan = build_profile_plan(
            tables, profile, variant, max_queries, max_docs,
            include_temporal, include_multi,
        )
        if plan is None:
            logger.warning("Profile %s has no usable queries; skipping.", profile)
            continue

        logger.info(
            "Profile %s: loading %d docs, %d queries (arm=%s)",
            profile, len(plan["docs"]), len(plan["queries"]), embedding_config,
        )
        row_limit = max(10000, len(plan["docs"]) + 100)
        load_res = firefox.semantic_history_load_profile(
            plan["docs"],
            row_limit=row_limit,
            distance_threshold=max(thresholds),
            embedding_type=load_type,
            region=load_region,
            common_components=common_components,
            dtype=dtype,
        )
        if not load_res.get("ok"):
            logger.warning("Profile %s load skipped: %s", profile, load_res.get("reason"))
            continue
        logger.info(
            "  embedded %s/%s (dim=%s, modelId=%s, region=%s, described=%s, cc=%s, "
            "dtype=%s)",
            load_res.get("embedded"), load_res.get("inserted"),
            load_res.get("embeddingSize"), load_res.get("modelId"),
            load_res.get("region"), load_res.get("described"),
            load_res.get("commonComponents"),
            load_res.get("dtype"),
        )
        # dtype is invisible in embeddingSize -- every dtype of this model is
        # 384-dim -- so the quantization comparison needs this check.
        if dtype is not None and load_res.get("dtype") != dtype:
            logger.error(
                "dtype=%s requested but the pipeline loaded %s; this run is not a "
                "valid quantization comparison", dtype, load_res.get("dtype"),
            )
        if load_res.get("described") == 0 and any(
            d.get("description") for d in plan["docs"]
        ):
            logger.error(
                "descriptions were supplied but none persisted; the corpus was "
                "embedded from titles only and this run is not comparable"
            )
        if expected_model_id and load_res.get("modelId") != expected_model_id:
            logger.error(
                "modelId %s != expected %s for arm=%s -- wrong model ran",
                load_res.get("modelId"), expected_model_id, embedding_config,
            )

        for threshold in thresholds:
            for spec in plan["queries"]:
                res = firefox.search_browsing_history(
                    {
                        "searchTerm": spec["query"],
                        "historyLimit": history_limit,
                        "featureGate": not keyword_only,
                        "distanceThreshold": threshold,
                    }
                )
                # The impl returns {count, results:[{url,title,relevanceScore,...}]}.
                # relevanceScore is the fused RRF score, not an embedding distance.
                payload = res.get("result") or {}
                rows = payload.get("results") or []
                ranked = [r.get("url") for r in rows if r.get("url")]
                relevant = set(spec["relevant_urls"])
                per_query[threshold].append(
                    {
                        "query": spec["query"],
                        "returned": len(ranked),
                        "queryType": spec["query_type"],
                        "docType": spec["doc_type"],
                        **{f"nDCG@{k}": ndcg_at_k(ranked, relevant, k) for k in ks},
                        **{f"recall@{k}": recall_at_k(ranked, relevant, k) for k in ks},
                        **{f"MRR@{k}": mrr_at_k(ranked, relevant, k) for k in ks},
                        **{f"P@{k}": precision_at_k(ranked, relevant, k) for k in ks},
                    }
                )
                if len(diag_seen) < 1:
                    diag_seen.append({k: res.get(k) for k in (
                        "featureGatePref", "distanceThresholdPref",
                        "isEnabledForSmartWindow", "hasSufficientEntries",
                        "embeddedRows", "modelId", "embeddingSize",
                        "commonComponents", "appLocale", "region",
                        "diagnosticsError",
                    )})

    # Preflight: a hybrid arm whose semantic leg was gated off silently measures
    # keyword-only. Fail loudly rather than let it into a comparison.
    if diag_seen:
        d = diag_seen[0]
        logger.info("  diagnostics: %s", json.dumps(d, default=str))
        if not keyword_only:
            if not d.get("isEnabledForSmartWindow"):
                logger.error(
                    "isEnabledForSmartWindow is false (featureGate=%s, locale=%s, "
                    "region=%s) -- the semantic leg did NOT run; this arm is "
                    "keyword-only and must not be compared as hybrid",
                    d.get("featureGatePref"), d.get("appLocale"), d.get("region"),
                )
            if not d.get("hasSufficientEntries"):
                logger.error(
                    "hasSufficientEntries is false (embeddedRows=%s) -- semantic "
                    "leg skipped",
                    d.get("embeddedRows"),
                )
        elif d.get("isEnabledForSmartWindow"):
            logger.error(
                "keyword-only arm still reports isEnabledForSmartWindow=true; the "
                "baseline may include semantic results"
            )
    return {"per_query": per_query, "diagnostics": diag_seen[0] if diag_seen else {}}


def write_log(results: Dict[str, Any], thresholds: List[float], ks: List[int],
              out_path: Path, arm: str, language: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_query = results["per_query"]
    lines = [
        f"# arm={arm} language={language}",
        f"# diagnostics={json.dumps(results['diagnostics'], default=str)}",
    ]
    metrics = ([f"nDCG@{k}" for k in ks] + [f"recall@{k}" for k in ks]
               + [f"MRR@{k}" for k in ks] + [f"P@{k}" for k in ks])
    for threshold in thresholds:
        rows = per_query.get(threshold, [])
        if not rows:
            continue
        agg = {m: sum(r[m] for r in rows) / len(rows) for m in metrics}
        avg_returned = sum(r["returned"] for r in rows) / len(rows)
        lines.append(
            f"INFO Threshold {threshold:g} | n={len(rows)} | avg_returned="
            f"{avg_returned:.2f} | "
            + " | ".join(f"{m}: {agg[m]:.4f}" for m in metrics)
        )
        # per-bucket breakdown, mirroring the threshold experiment's shape
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            buckets[f"{r['queryType']}_{r['docType']}"].append(r)
        for name in sorted(buckets):
            b = buckets[name]
            agg_b = {m: sum(r[m] for r in b) / len(b) for m in metrics}
            lines.append(
                f"INFO Threshold {threshold:g} | {name} | n={len(b)} | "
                + " | ".join(f"{m}: {agg_b[m]:.4f}" for m in metrics)
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote log to %s (%d lines)", out_path, len(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--firefox-bin", type=Path, default=None)
    parser.add_argument("--profile-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("topk.log"))
    parser.add_argument("--max-profiles", type=int, default=5)
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--max-queries", type=int, default=40)
    parser.add_argument("--max-docs", type=int, default=400)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--k", type=int, nargs="*", default=DEFAULT_K)
    parser.add_argument("--history-limit", type=int, default=15)
    parser.add_argument(
        "--embedding-types", nargs="*", default=["static-en-fr", KEYWORD_ONLY],
        choices=sorted(EMBEDDING_CONFIGS) + [KEYWORD_ONLY],
        help="Arms to run. '%s' gates the semantic leg off for a Places-only "
             "baseline." % KEYWORD_ONLY,
    )
    parser.add_argument("--language", choices=["en", "fr", "en-paired"], default="en")
    parser.add_argument(
        "--dataset", choices=["history", "shortquery"], default="history",
        help="history: Mozilla/history-search-retrieval (long queries, ~6.4 words; "
             "the keyword leg returns nothing on these). shortquery: the "
             "creative-hobbyist golden set (~4.1 words), where the keyword leg "
             "contributes and RRF fusion is observable. With shortquery, "
             "--language en and en-paired are equivalent (one corpus, no subsetting).",
    )
    parser.add_argument(
        "--fr-overlay-dir", type=Path,
        default=Path.home() / "Documents/GitHub/multilingual_embeddings_eval"
                             "/datasets/fr_history_search",
    )
    parser.add_argument(
        "--shortquery-dir", type=Path,
        default=Path.home() / "Documents/GitHub/multilingual_embeddings_eval"
                             "/datasets/fr_shortquery_search",
    )
    parser.add_argument("--include-temporal", action="store_true")
    parser.add_argument("--include-multi", action="store_true", default=True)
    parser.add_argument("--no-include-multi", dest="include_multi", action="store_false")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(name)s] %(message)s",
    )

    if args.dataset == "shortquery":
        tables = load_shortquery_tables(args.shortquery_dir, args.language)
    else:
        tables = load_dataset_tables(cache_dir=args.cache_dir)
        if args.language in ("fr", "en-paired"):
            tables = apply_french_overlay(
                tables, args.fr_overlay_dir, translate=args.language == "fr"
            )
    profiles = select_profiles(tables, args.max_profiles, args.profiles, args.variant)
    if not profiles:
        raise SystemExit("No matching profiles found in dataset.")
    logger.info("Running profiles: %s", profiles)

    firefox = FirefoxDriver(
        model="", auth_config={}, headless=args.headless,
        firefox_bin=args.firefox_bin, profile_path=args.profile_path,
        prompt_path="", skip_fastly=False,
    )
    try:
        for index, arm in enumerate(args.embedding_types):
            desc = ("Places keyword only (semantic gated off)" if arm == KEYWORD_ONLY
                    else EMBEDDING_CONFIGS[arm]["description"])
            logger.info("=== Arm: %s (%s) ===", arm, desc)
            # Switching model in a live session hangs once the vector table is
            # populated; a restart sidesteps it. See threshold_experiment.
            if index > 0:
                logger.info("Restarting Firefox before switching arm...")
                firefox.restart()
            results = run_experiment(
                firefox, tables, profiles, args.thresholds, args.k, args.variant,
                args.max_queries, args.max_docs, args.include_temporal,
                args.include_multi, arm, args.history_limit,
            )
            parts = [args.out.stem, arm]
            if args.language != "en":
                parts.append(args.language)
            if args.dataset != "history":
                parts.append(args.dataset)
            out_path = args.out.with_name("_".join(parts) + args.out.suffix)
            write_log(results, args.thresholds, args.k, out_path, arm, args.language)
            firefox.semantic_history_cleanup()
    finally:
        firefox.quit()


if __name__ == "__main__":
    main()
