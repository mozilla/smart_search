import sqlite3
import argparse
import os
import logging
import pandas as pd
from kg import (
    extract_tags_for_queries,
    infer_topics,
    fetch_entity_relations_with_keywords
)
from metrics import run_traditional_eval


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
DATA_PATH = "data/"
KG_DB_PATH = f"{DATA_PATH}/ml_kg.db"

def get_sql_connection_for_validation():
    """
    Conects you with KG data built by kg_builder.py
    Attaches the places DB to that
    returns a connection
    """
    connection = sqlite3.connect(KG_DB_PATH, check_same_thread=False)
    places_db_path = f"{DATA_PATH}/places.sqlite"
    connection.execute(f"ATTACH DATABASE '{places_db_path}' AS places_db;")
    return connection

def fetch_ground_truths(conn):
    val_cursor = conn.cursor()

    val_data = val_cursor.execute(
        """
        SELECT ih.input AS keyword,
               p.url_hash,
               ih.use_count,
               p.url
          FROM moz_inputhistory ih
          JOIN moz_places p
            ON ih.place_id = p.id
          WHERE input != ''
          ORDER BY keyword, use_count DESC
        """
    ).fetchall()
    val_cursor.close()
    val_columns = ['keyword', 'url_hash', 'use_count', 'url']
    actuals_df = pd.DataFrame(val_data, columns=val_columns)
    return actuals_df.groupby('keyword')['url_hash'].agg(list).reset_index()

def perform_traditional_evals(conn, to_be_predicted_queries, k, use_tags=True, use_topics=True):
    eval_rows = []
    for idx, row in to_be_predicted_queries.iterrows():
        if (idx+1) % 50 == 0:
            print(f" {idx+1} queries evaluated")
        search_query = row['keyword']
        relevant_docs = row['url_hash']
        
        search_tags = extract_tags_for_queries([search_query])[0] if use_tags else []
        search_topics = infer_topics([search_query], pbar=False)[0] if use_topics else []
        # print(f"search_tags = {search_tags}")
        # print(f"search_topics = {search_topics}")
        results = fetch_entity_relations_with_keywords(conn, search_query, search_tags, search_topics).head(k)
        retrieved_docs = []
        if len(results) > 0:
            retrieved_docs = results['url_hash'].values.tolist()
            eval_row = run_traditional_eval(idx, search_query, relevant_docs, retrieved_docs, retrieved_distances=None, k=k)
            eval_rows.append(eval_row)
    return pd.DataFrame(eval_rows)

def get_url_hash_batch(golden_queries, conn):
    urls = tuple(golden_queries['url'].tolist())

    # Query all URL hashes in one go
    query = f"""
    SELECT url, url_hash
    FROM moz_places
    WHERE url IN ({','.join(['?'] * len(urls))})
    """
    cursor = conn.cursor()
    results = cursor.execute(query, urls).fetchall()
    return pd.DataFrame(results, columns=["url", "url_hash"])

def main(golden_queries_file, k=2):
    conn = get_sql_connection_for_validation()
    if os.path.exists(golden_queries_file):
        golden_queries = pd.read_csv(golden_queries_file, usecols=['search_query', 'url'])
        golden_queries['search_query'] = golden_queries['search_query'].str.lower()
        logger.info(f"Number of golden queries = {len(golden_queries)}")
        logger.info(golden_queries.head().T)

        url_hashes = get_url_hash_batch(golden_queries, conn)
        queries_with_ground_truth = golden_queries.merge(url_hashes, on='url', how='inner')
        queries_with_ground_truth = queries_with_ground_truth.groupby('search_query')['url_hash'].agg(list).reset_index()\
                                               .rename(columns={'search_query': 'keyword'})
    else:
        logger.warn(f"{golden_queries_file} does not exist and hence using the places DB")
        queries_with_ground_truth = fetch_ground_truths(conn)
        
    os.makedirs(f"{DATA_PATH}/kg_results", exist_ok=True)
    metric_cols = [f'precision@{k}',f'recall@{k}',f'ndcg@{k}','reciprocal_rank','average_precision']

    logger.info("\n Use keywords + topics + tags")
    golden_eval_df = perform_traditional_evals(conn, queries_with_ground_truth, k=k)
    if golden_eval_df is None or golden_eval_df.empty:
        logger.warning("No matches using keywords + topics + tag. Please check.")
    else:
        logger.info(golden_eval_df[metric_cols].mean())
        logger.info(len(golden_eval_df))
        golden_eval_df.to_csv(f"{DATA_PATH}/kg_results/golden_eval_df_keywords_topics_tags.csv", index=False)

    logger.info("\n Use keywords + tags and no topics")
    golden_eval_df = perform_traditional_evals(conn, queries_with_ground_truth, k=k, use_topics=False)
    if golden_eval_df is None or golden_eval_df.empty:
        logger.warning("No matches using keywords + tags and no topics. Please check.")
    else:
        logger.info(golden_eval_df[metric_cols].mean())
        logger.info(len(golden_eval_df))
        golden_eval_df.to_csv(f"{DATA_PATH}/kg_results/golden_eval_df_keywords_tags.csv", index=False)

    logger.info("\n Use keywords + topics and no tags")
    golden_eval_df = perform_traditional_evals(conn, queries_with_ground_truth, k=k, use_tags=False)
    if golden_eval_df is None or golden_eval_df.empty:
        logger.warning("No matches using keywords + topics and no tags. Please check.")
    else:
        logger.info(golden_eval_df[metric_cols].mean())
        logger.info(len(golden_eval_df))
        golden_eval_df.to_csv(f"{DATA_PATH}/kg_results/golden_eval_df_keywords_topics.csv", index=False)

    logger.info("\n Use keywords only")
    golden_eval_df = perform_traditional_evals(conn, queries_with_ground_truth,
                                               k=k, use_tags=False, use_topics=False)
    if golden_eval_df is None or golden_eval_df.empty:
        logger.warning("No matches using  keywords only. Please check.")
    else:
        logger.info(golden_eval_df[metric_cols].mean())
        logger.info(len(golden_eval_df))
        golden_eval_df.to_csv(f"{DATA_PATH}/kg_results/golden_eval_df_keywords.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run KG Validator")
    parser.add_argument("--golden_queries_file",
                        type=str,
                        default=f"{DATA_PATH}/chidam_golden_query.csv",
                        help="Relative path to the golden queries CSV file"
                        )
    args = parser.parse_args()
    main(golden_queries_file=args.golden_queries_file)
