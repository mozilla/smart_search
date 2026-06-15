"""Semantic-history-search threshold sweep experiment.

Reproduces the experiment analysed by
``smart_search/notebooks/threshold_analysis_from_log_results.ipynb`` but sources
its data from the ``Mozilla/history-search-retrieval`` HuggingFace dataset and
runs the retrieval through a real Firefox build via ``PlacesSemanticHistoryManager``
(driven by :class:`ml_driver.firefox.driver.FirefoxDriver`).

For each selected persona profile we:

1. Load the persona's history docs into Firefox Places (``insertMany``) and
   generate embeddings.
2. For each distance threshold, re-instantiate the semantic manager at that
   threshold and run every query through ``infer``.
3. Score Recall@1 (is the top-1 result a qrels-relevant doc?) and bucket each
   query into ``shortQ``/``longQ`` x ``shortD``/``longD`` by length.

Results are written to a log file in the exact format the notebook parses, so
the existing threshold-analysis notebook works unchanged against the output.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml_driver.firefox.driver import FirefoxDriver

logger = logging.getLogger("threshold-experiment")

HF_REPO_ID = "Mozilla/history-search-retrieval"
HF_FILES = {
    "docs": "data/v1/docs.parquet",
    "queries": "data/v1/queries.parquet",
    "qrels": "data/v1/qrels.parquet",
}

DEFAULT_THRESHOLDS = [0.25, 0.4, 0.5, 0.6, 0.75]

# Length cutoffs used to bucket queries/docs into the notebook's short/long axes.
SHORT_QUERY_MAX_CHARS = 30
SHORT_DOC_MAX_CHARS = 30


def _fmt_threshold(value: float) -> str:
    """Format a threshold the way the notebook regex expects (0.25, 0.4, ...)."""
    return f"{value:g}"


def load_dataset_tables(cache_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Download and load the docs/queries/qrels parquet tables."""
    from huggingface_hub import hf_hub_download

    tables: Dict[str, pd.DataFrame] = {}
    for name, filename in HF_FILES.items():
        logger.info("Downloading %s (%s)...", name, filename)
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        tables[name] = pd.read_parquet(path)
        logger.info("  %s rows: %d", name, len(tables[name]))
    return tables


def _query_type(search_query: str) -> str:
    return "shortQ" if len(search_query.strip()) < SHORT_QUERY_MAX_CHARS else "longQ"


def _doc_type(title: str) -> str:
    return "shortD" if len((title or "").strip()) < SHORT_DOC_MAX_CHARS else "longD"


def select_profiles(
    tables: Dict[str, pd.DataFrame],
    max_profiles: Optional[int],
    profiles: Optional[List[str]],
    variant: str,
) -> List[str]:
    queries = tables["queries"]
    queries = queries[queries["variant"] == variant]
    available = sorted(queries["profile"].unique().tolist())
    if profiles:
        chosen = [p for p in profiles if p in available]
        missing = sorted(set(profiles) - set(available))
        if missing:
            logger.warning("Requested profiles not in dataset: %s", missing)
    else:
        chosen = available
    if max_profiles is not None:
        chosen = chosen[:max_profiles]
    return chosen


def build_profile_plan(
    tables: Dict[str, pd.DataFrame],
    profile: str,
    variant: str,
    max_queries: Optional[int],
    max_docs: Optional[int],
    include_temporal: bool,
    include_multi: bool,
) -> Optional[Dict[str, Any]]:
    """Assemble the docs to load and the queries to run for one persona.

    Returns ``None`` if the persona has no usable queries.
    """
    docs = tables["docs"]
    queries = tables["queries"]
    qrels = tables["qrels"]

    p_queries = queries[(queries["profile"] == profile) & (queries["variant"] == variant)].copy()
    if not include_temporal:
        p_queries = p_queries[~p_queries["is_temporal"].astype(bool)]
    if not include_multi:
        p_queries = p_queries[~p_queries["is_multi"].astype(bool)]
    p_queries = p_queries.reset_index(drop=True)
    if max_queries is not None:
        p_queries = p_queries.head(max_queries)
    if p_queries.empty:
        return None

    p_docs = docs[docs["profile"] == profile].copy()
    # doc_id -> row, and url lookup for scoring infer results (which return url).
    doc_by_id = {row["doc_id"]: row for _, row in p_docs.iterrows()}

    query_ids = set(p_queries["query_id"])
    p_qrels = qrels[(qrels["query_id"].isin(query_ids)) & (qrels["relevance"] > 0)]

    # Relevant docs per query (urls), and the rank-1 golden doc (for doc-type bucket).
    relevant_urls: Dict[str, set] = defaultdict(set)
    golden_doc: Dict[str, Any] = {}
    for _, r in p_qrels.iterrows():
        doc = doc_by_id.get(r["doc_id"])
        if doc is None:
            continue
        relevant_urls[r["query_id"]].add(doc["url"])
        if r["rank"] == 1 or r["query_id"] not in golden_doc:
            golden_doc[r["query_id"]] = doc

    # Keep only queries that have at least one relevant doc present in this profile.
    usable = [q for _, q in p_queries.iterrows() if relevant_urls.get(q["query_id"])]
    if not usable:
        return None

    # Build the doc set: all relevant (golden + others) docs, plus distractors up
    # to max_docs so golden docs are always retrievable.
    relevant_doc_ids = set(p_qrels["doc_id"])
    relevant_docs = p_docs[p_docs["doc_id"].isin(relevant_doc_ids)]
    distractor_docs = p_docs[~p_docs["doc_id"].isin(relevant_doc_ids)]
    if max_docs is not None:
        remaining = max(0, max_docs - len(relevant_docs))
        distractor_docs = distractor_docs.head(remaining)
    load_docs = pd.concat([relevant_docs, distractor_docs]).drop_duplicates("doc_id")

    now_ms = int(time.time() * 1000)
    payload_docs: List[Dict[str, Any]] = []
    for _, d in load_docs.iterrows():
        last_visit_ms = int(d["last_visit_date"]) // 1000 if pd.notna(d["last_visit_date"]) else now_ms
        # insertMany rejects future visit dates.
        last_visit_ms = min(last_visit_ms, now_ms - 1000)
        payload_docs.append(
            {
                "url": d["url"],
                "title": d["title"] or "",
                "description": (d["description"] or "") if pd.notna(d.get("description")) else "",
                "last_visit_ms": last_visit_ms,
            }
        )

    query_specs = []
    for q in usable:
        qid = q["query_id"]
        query_specs.append(
            {
                "query": q["search_query"],
                "relevant_urls": relevant_urls[qid],
                "query_type": _query_type(q["search_query"]),
                "doc_type": _doc_type(golden_doc[qid]["title"]),
            }
        )

    return {"profile": profile, "docs": payload_docs, "queries": query_specs}


def run_experiment(
    firefox: FirefoxDriver,
    tables: Dict[str, pd.DataFrame],
    profiles: List[str],
    thresholds: List[float],
    variant: str,
    max_queries: Optional[int],
    max_docs: Optional[int],
    include_temporal: bool,
    include_multi: bool,
    embedding_type: Optional[str] = None,
) -> Dict[Tuple[float, str, str], List[Dict[str, Any]]]:
    """Run the sweep, returning per-(threshold, queryType, docType) records."""
    # results[(threshold, qt, dt)] = [ {query, matched, results}, ... ]
    results: Dict[Tuple[float, str, str], List[Dict[str, Any]]] = defaultdict(list)
    expected_dim = {"static": 512, "contextual": 384}.get(embedding_type or "")

    for profile in profiles:
        plan = build_profile_plan(
            tables, profile, variant, max_queries, max_docs, include_temporal, include_multi
        )
        if plan is None:
            logger.warning("Profile %s has no usable queries; skipping.", profile)
            continue

        logger.info(
            "Profile %s: loading %d docs, %d queries%s",
            profile,
            len(plan["docs"]),
            len(plan["queries"]),
            f" (embeddingType={embedding_type})" if embedding_type else "",
        )
        row_limit = max(10000, len(plan["docs"]) + 100)
        load_res = firefox.semantic_history_load_profile(
            plan["docs"],
            row_limit=row_limit,
            distance_threshold=max(thresholds),
            embedding_type=embedding_type,
        )
        if not load_res.get("ok"):
            logger.warning("Profile %s load skipped: %s", profile, load_res.get("reason"))
            continue
        emb_size = load_res.get("embeddingSize")
        logger.info(
            "  embedded %s/%s docs (embeddingSize=%s)",
            load_res.get("embedded"),
            load_res.get("inserted"),
            emb_size,
        )
        if expected_dim is not None and emb_size != expected_dim:
            logger.warning(
                "embeddingSize %s != expected %s for embeddingType=%s -- pref may not have applied",
                emb_size,
                expected_dim,
                embedding_type,
            )

        try:
            for threshold in thresholds:
                firefox.semantic_history_set_threshold(threshold, row_limit=row_limit)
                for spec in plan["queries"]:
                    infer_res = firefox.semantic_history_infer(spec["query"])
                    hits = infer_res.get("results", [])
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
                logger.info("  threshold %s done", _fmt_threshold(threshold))
        finally:
            firefox.semantic_history_cleanup()

    return results


def write_log(
    results: Dict[Tuple[float, str, str], List[Dict[str, Any]]],
    thresholds: List[float],
    out_path: Path,
) -> None:
    """Write results in the format threshold_analysis_from_log_results.ipynb parses."""
    query_types = ["shortQ", "longQ"]
    doc_types = ["shortD", "longD"]

    lines: List[str] = []
    # Notebook iterates blocks in file order; group by (qt, dt) then threshold.
    for qt in query_types:
        for dt in doc_types:
            for threshold in thresholds:
                records = results.get((threshold, qt, dt), [])
                if not records:
                    continue
                t_str = _fmt_threshold(threshold)
                lines.append(f"Running {qt} → {dt} test at threshold {t_str}")
                correct = 0
                for rec in records:
                    status = "Matched" if rec["matched"] else "No match"
                    if rec["matched"]:
                        correct += 1
                    lines.append(f"{status} query: {rec['query']}")
                    lines.append(f"INFO results: {json.dumps(rec['results'], ensure_ascii=False)}")
                total = len(records)
                accuracy = (correct / total * 100) if total else 0.0
                # Precision (micro-averaged over the threshold-filtered retrieved
                # set): relevant retrieved / total retrieved across the block.
                rel_retrieved = sum(r.get("rel_retrieved", 0) for r in records)
                retrieved = sum(r.get("retrieved", 0) for r in records)
                precision = (rel_retrieved / retrieved * 100) if retrieved else 0.0
                profile_name = f"profile_{qt}_{dt}_df.json"
                lines.append(
                    f"INFO Threshold {t_str} | {profile_name} | "
                    f"Recall@1: {correct}/{total} ({accuracy:.2f}%) | "
                    f"Precision: {rel_retrieved}/{retrieved} ({precision:.2f}%)"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote log to %s (%d lines)", out_path, len(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firefox-bin", type=Path, default=None, help="Path to a Firefox build with ML enabled.")
    parser.add_argument("--profile-path", type=Path, default=None, help="Firefox profile directory to reuse.")
    parser.add_argument("--out", type=Path, default=Path("browser_ml_semantic_history_search_threshold_test.log"))
    parser.add_argument("--max-profiles", type=int, default=3, help="Number of personas to run.")
    parser.add_argument("--profiles", nargs="*", default=None, help="Explicit persona names (overrides --max-profiles count).")
    parser.add_argument("--max-queries", type=int, default=20, help="Max queries per persona.")
    parser.add_argument("--max-docs", type=int, default=800, help="Max docs loaded per persona (relevant docs always kept).")
    parser.add_argument("--variant", default="base", help="Dataset variant to use.")
    parser.add_argument("--thresholds", type=float, nargs="*", default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--embedding-types",
        nargs="*",
        default=["static"],
        choices=["static", "contextual"],
        help="Embedding model(s) to run, sequentially (one shared DB). "
        "static=512-dim, contextual=384-dim. One log per type is written.",
    )
    parser.add_argument("--include-temporal", action="store_true", help="Include time-relative queries (not handled by infer).")
    parser.add_argument("--include-multi", action="store_true", default=True, help="Include multi-label queries.")
    parser.add_argument("--no-include-multi", dest="include_multi", action="store_false")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache dir.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="[%(name)s] %(message)s")

    tables = load_dataset_tables(cache_dir=args.cache_dir)
    profiles = select_profiles(tables, args.max_profiles, args.profiles, args.variant)
    if not profiles:
        raise SystemExit("No matching profiles found in dataset.")
    logger.info("Running profiles: %s", profiles)

    # Semantic history search only needs a chrome-context Firefox with
    # browser.ml.enable + places.semanticHistory.featureGate (both set by the
    # driver) and the ML RemoteSettings model. It does not use Smart Window, so
    # skip_fastly=False avoids requiring the FASTLY_TOKEN env var.
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
        # Run each embedding type sequentially; there is a single shared vector
        # DB, and switching places.semanticHistory.embeddingType drops/recreates
        # the embedding table at the next load.
        for embedding_type in args.embedding_types:
            logger.info("=== Embedding type: %s ===", embedding_type)
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
                embedding_type=embedding_type,
            )
            if len(args.embedding_types) > 1:
                out_path = args.out.with_name(
                    f"{args.out.stem}_{embedding_type}{args.out.suffix}"
                )
            else:
                out_path = args.out
            write_log(results, args.thresholds, out_path)
            # Clean up between embedding types so the next load starts fresh.
            firefox.semantic_history_cleanup()
    finally:
        firefox.quit()


if __name__ == "__main__":
    main()
