import sqlite3
import pandas as pd
import logging
from kg import (
    extract_tags_for_queries,
    infer_topics,
    fetch_entity_relations_with_keywords
)
from metrics import run_traditional_eval
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
DATA_PATH = "data/"
KG_DB_PATH = f"{DATA_PATH}/ml_kg.db"
## Override with your golden queries
GOLDEN_QUERIES_FILE = f"{DATA_PATH}/chidam_golden_query.csv"

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

def perform_traditional_evals(conn, to_be_predicted_queries, use_tags=True, use_topics=True):
    eval_rows = []
    for idx, row in to_be_predicted_queries.iterrows():
        if (idx+1) % 50 == 0:
            print(f" {idx+1} queries evaluated")
        search_query = search_keyword = row['keyword']
        # print(f"search_keyword = {search_keyword}")
        relevant_docs = row['url_hash']
        
        search_tags = extract_tags_for_queries([search_query])[0] if use_tags else []
        search_topics = infer_topics([search_query], pbar=False)[0] if use_topics else []
        # print(f"search_tags = {search_tags}")
        # print(f"search_topics = {search_topics}")
        results = fetch_entity_relations_with_keywords(conn, search_keyword, search_tags, search_topics).head(2)
        retrieved_docs = []
        if len(results) > 0:
            retrieved_docs = results['url_hash'].values.tolist()
            eval_row = run_traditional_eval(idx, search_keyword, relevant_docs, retrieved_docs, retrieved_distances=None, k=2)
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

def main():
    conn = get_sql_connection_for_validation()
    # queries_with_ground_truth = fetch_ground_truths(conn)
    
    

    if os.path.exists(GOLDEN_QUERIES_FILE):
        golden_queries = pd.read_csv(GOLDEN_QUERIES_FILE, usecols=['search_query', 'url'])
        logger.info(f"Number of golden queries = {len(golden_queries)}")
        logger.info(golden_queries.head().T)

        url_hashes = get_url_hash_batch(golden_queries, conn)
        queries_with_ground_truth = golden_queries.merge(url_hashes, on='url', how='inner')
        queries_with_ground_truth = queries_with_ground_truth.groupby('search_query')['url_hash'].agg(list).reset_index()\
                                               .rename(columns={'search_query': 'keyword'})
    
    else:
        queries_with_ground_truth = fetch_ground_truths(conn)
        
    logger.info("\n Use keywords + topics + tags")
    golden_eval_df = perform_traditional_evals(conn, queries_with_ground_truth)
    logger.info(golden_eval_df[['precision@2','recall@2','ndcg@2','reciprocal_rank','average_precision']].mean())
    logger.info(len(golden_eval_df))
    os.makedirs(f"{DATA_PATH}/kg_results", exist_ok=True)
    golden_eval_df.to_csv(f"{DATA_PATH}/kg_results/golden_eval_df_keywords_topics_tags.csv", index=False)

    logger.info("\n Use keywords + tags and no topics")
    golden_eval_df = perform_traditional_evals(conn, queries_with_ground_truth, use_topics=False)
    logger.info(golden_eval_df[['precision@2','recall@2','ndcg@2','reciprocal_rank','average_precision']].mean())
    logger.info(len(golden_eval_df))
    golden_eval_df.to_csv(f"{DATA_PATH}/kg_results/golden_eval_df_keywords_tags.csv", index=False)

    logger.info("\n Use keywords + topics and no tags")
    golden_eval_df = perform_traditional_evals(conn, queries_with_ground_truth, use_tags=False)
    logger.info(golden_eval_df[['precision@2','recall@2','ndcg@2','reciprocal_rank','average_precision']].mean())
    logger.info(len(golden_eval_df))
    golden_eval_df.to_csv(f"{DATA_PATH}/kg_results/golden_eval_df_keywords_topics.csv", index=False)

    logger.info("\n Use keywords only")
    golden_eval_df = perform_traditional_evals(conn, queries_with_ground_truth, use_tags=False, use_topics=False)
    logger.info(golden_eval_df[['precision@2','recall@2','ndcg@2','reciprocal_rank','average_precision']].mean())
    logger.info(len(golden_eval_df))
    golden_eval_df.to_csv(f"{DATA_PATH}/kg_results/golden_eval_df_keywords.csv", index=False)


if __name__ == '__main__':
    main()