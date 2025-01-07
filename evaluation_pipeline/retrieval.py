

import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import tracemalloc
import time

import pandas as pd
import numpy as np
import sqlite3
import sqlite_vec
from typing import List
import struct
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import sys
import argparse


# Add the parent directory of `evaluation_pipeline` to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constants import EMBEDDING_MODELS_DICT
from src.feature_extractor import FeatureExtractor




# Configure logging
logging.basicConfig(
    filename="performance.log", 
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


def log_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        end_time = time.time()
        
        logging.info(f"Function: {func.__name__}")
        logging.info(f"Execution Time: {end_time - start_time:.2f} seconds")
        logging.info(f"Current Memory Usage: {current / 10**6:.2f} MB")
        logging.info(f"Peak Memory Usage: {peak / 10**6:.2f} MB")
        
        tracemalloc.stop()
        return result
    return wrapper

def format_size(size_in_bytes):
    """Convert bytes to a human-readable format (KB, MB, GB, etc.)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024


def process_golden_queries(golden_query_file_path):
    golden_df = pd.read_csv(golden_query_file_path=golden_query_file_path)
    query_ids = {query: str(hash(query)) for query in golden_df['search_query'].unique()}
    return query_ids



@log_performance
def process_history(row_limit, history_file_path):
    browsing_history = pd.read_csv(history_file_path).head(row_limit)
    browsing_history['last_visit_date'] = pd.to_datetime(browsing_history['last_visit_date'], unit='us')
    # fill empty last_visit_date with default value "1970-01-01"
    browsing_history['last_visit_date'] = browsing_history['last_visit_date'].fillna(pd.to_datetime("1970-01-01"))
    browsing_history['combined_text'] = browsing_history['title'].fillna('') + " " + browsing_history['description'].fillna('')
    browsing_history['combined_text_url'] = browsing_history['title'].fillna('') + " " + browsing_history['description'].fillna('') + browsing_history['url'].fillna('')
    browsing_history = browsing_history.loc[browsing_history['combined_text'] != ''].reset_index(drop=True)

    print(len(browsing_history))

    return browsing_history

@log_performance
def create_embeddings(history, embeddings_model_dict):
    texts = history['combined_text'].values.tolist()
    embeddings_dict = {}
    embeddings_sizes = {}

    for model in embeddings_model_dict.keys():
        if model == 'nomic-ai/nomic-embed-text-v1.5': 
            prefix = 'search_document: '
            texts = [prefix + text for text in texts]
        fe = FeatureExtractor(embeddings_model_dict, model_name=model)
        embeddings_dict[model] = fe.get_embeddings(texts)
        print(model, embeddings_dict[model].shape)
        embeddings_sizes[model] = embeddings_dict[model].shape[1]
    return embeddings_dict, embeddings_sizes

@log_performance
def create_db():
    # vector database
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    sqlite_version, vec_version = db.execute(
        "select sqlite_version(), vec_version()"
    ).fetchone()
    print(f"sqlite_version={sqlite_version}, vec_version={vec_version}")
    return db

def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)

@log_performance
def create_embeddings_table_in_vector_db(db, model_name, embeddings_sizes, embeddings_dict):
    EMBEDDING_SIZE = embeddings_sizes[model_name]
    items = []
    for idx, vec in enumerate(embeddings_dict[model_name]):
        items.append((idx, list(vec)))
    model_name_normalized = model_name.replace("/","_").replace("-","_").replace(".","_")
    db.execute(f"CREATE VIRTUAL TABLE vec_items_{model_name_normalized} USING vec0(embedding float[{EMBEDDING_SIZE}])")

    with db:
        for item in items:
            db.execute(
                f"INSERT INTO vec_items_{model_name_normalized}(rowid, embedding) VALUES (?, ?)",
                [item[0], serialize_f32(item[1])],
            )
    return db


# retrieval
@log_performance
def query_and_result(query, db, model_name, threshold, k):
    model_name_normalized = model_name.replace("/","_").replace("-","_").replace(".","_") 
    if model_name == 'nomic-ai/nomic-embed-text-v1.5': 
        query = 'search_query: ' + query 

    # create embedding from query
    fe = FeatureExtractor(EMBEDDING_MODELS_DICT, model_name=model_name)
    query_embedding = fe.get_embeddings([query])[0]
    # using cosine distance
    rows = db.execute(
    f"""
      SELECT
        rowid,
        vec_distance_cosine(embedding, ?) AS cosine_distance
      FROM vec_items_{model_name_normalized}
      ORDER BY cosine_distance
      LIMIT {k}
    """,
    [serialize_f32(query_embedding)],
    ).fetchall()

    results = []

    for row in rows:
        rowid = row[0]  # Get the row ID
        distance = row[1]  # Get the cosine distance

        # Skip rows where distance > threshold
        #if distance > threshold:
        #    continue

        print("doing search now")
        # Step 2: Query additional details for the matching row from search_data
        res = db.execute(
            """
            SELECT rowid, title, url, combined_text
            FROM search_data
            WHERE rowid = ?
            """,
            (rowid,)
        ).fetchone()

        # Add the result to the results list
        if res:
            results.append({
                "id": res[0],
                "title": res[1],
                "url": res[2],
                "combined_text": res[3],
                "distance": distance,
            })

    return results

def get_table_size(connection, table_name):
    """Get the approximate size of a table in SQLite."""
    with connection:
        cursor = connection.cursor()
        # Get page size and page count
        cursor.execute("PRAGMA page_count;")
        page_count = cursor.fetchone()[0]

        cursor.execute("PRAGMA page_size;")
        page_size = cursor.fetchone()[0]
        
        total_db_size = page_count * page_size

        # If dbstat is available, use it for more precision
        try:
            cursor.execute(f"""
                SELECT SUM(pgsize) AS table_size
                FROM dbstat
                WHERE name = ?;
            """, (table_name,))
            result = cursor.fetchone()
            if result and result[0]:
                return result[0]
        except sqlite3.DatabaseError:
            print("dbstat is not available; estimating based on database size.")

        # Fallback to database size as an estimate
        return total_db_size



@log_performance
def load_history_in_db(db, browsing_history):
    db.execute('''CREATE TABLE IF NOT EXISTS search_data (
        url TEXT,
        title TEXT,
        combined_text TEXT
        )
        ''')
    for _, row in browsing_history.iterrows():
            db.execute("""
                INSERT INTO search_data (url, title, combined_text)
                VALUES (?, ?, ?)
            """, ( row['url'], row['title'], row['combined_text'])
            )



def create_ground_truth(history, query_ids):
    queries = []
    relevant_ids = []
    for query in history['search_query'].unique():
        query_id = query_ids[query]
        queries.append(query_id)
        ground_truths = []
        for i, row in history.iterrows():
            relevant = 1 if query == row['search_query'] else 0
            if relevant == 1:
                ground_truths.append(int(i))
        relevant_ids.append(ground_truths)

    return dict(zip(queries, relevant_ids))

def load_ground_truth_from_golden(db, golden_df_file_path):
    golden_df = pd.read_csv(golden_df_file_path)
    print(golden_df)
    query_ids = {query: str(hash(query)) for query in golden_df['search_query'].unique()}
    db.execute('''CREATE TABLE IF NOT EXISTS ground_truth (
        search_query TEXT,
        url TEXT
        )
        ''')
    with db:
       for _, row in golden_df.iterrows():
            db.execute("""
                INSERT INTO ground_truth (search_query, url)
                VALUES (?, ?)
            """, ( row['search_query'], row['url'])
            )

    # join to search query to get doc ID for ground truth URL
    results = db.execute(
        '''SELECT a.search_query, a.url, b.rowid
        FROM ground_truth a
        left join search_data b
        on a.url = b.url'''
    ).fetchall()

    print(results)

    ground_truth = {}
    for query, url, id_ in results:
        query_id = query_ids[query]
        if query not in ground_truth:
            ground_truth[query_id] = []
        ground_truth[query_id].append(id_)

    return ground_truth, query_ids




@log_performance
def run_history_in_vector_db(row_limit, history_file_path, golden_set_file_path):

    browsing_history = process_history(row_limit, history_file_path=history_file_path)

    # create vector DB
    db = create_db()


    # load in history for joining later
    load_history_in_db(db, browsing_history)


    # if a golden set is not provided, assume it's with the history
    if not golden_set_file_path:
        query_ids = {query: str(hash(query)) for query in browsing_history['search_query'].unique()}
        browsing_history['query_id'] = browsing_history['search_query'].map(query_ids)
        ground_truth = create_ground_truth(browsing_history, query_ids)
    else:
        print("Getting doc ids for history")
        ground_truth, query_ids = load_ground_truth_from_golden(db, golden_df_file_path=golden_set_file_path)
        print(ground_truth)


    # create embeddings for candidate models
    print("Generating Embeddings")
    embeddings_dict, embeddings_sizes = create_embeddings(browsing_history, embeddings_model_dict=EMBEDDING_MODELS_DICT)

    # loop through each model/embedding type and store in db
    for model_name in embeddings_dict.keys():
        model_name_normalized = model_name.replace("/","_").replace("-","_").replace(".","_")

        # create table for embeddings for model
        create_embeddings_table_in_vector_db(db, model_name, embeddings_sizes=embeddings_sizes, embeddings_dict=embeddings_dict)

        table_size = get_table_size(db, table_name=model_name_normalized)
        logging.info(f"{model_name_normalized} table size: {table_size}")
        total_db_size_human_readable = format_size(table_size)
        logging.info(f"Table size {model_name_normalized}: {total_db_size_human_readable}")

    return query_ids, db, ground_truth

@log_performance
def run_retrieval(query_ids, db, model_name, threshold, k):
    # loop through each query
    retreival_dict = {}
    query_lookup = {}
    for query in query_ids.keys():
        # for later identifying query
        query_id = query_ids[query]
        query_lookup[query_id] = query
        # perform retrieval
        results = query_and_result(query, db=db,
        model_name=model_name, threshold=threshold, k=k)
        retreival_dict[query_id] = results
    return retreival_dict, query_lookup


def convert_dict_to_df(retrieval_dict, query_lookup, ground_truth, model_name, k):
    rows = []
    print(len(retrieval_dict.keys()))
    print(ground_truth)
    print(len(ground_truth.keys()))
    assert len(retrieval_dict.keys()) == len(ground_truth.keys())
    for query_id, retrievals in retrieval_dict.items():
        # Flatten each retrieval into a single row with column names based on retrieval index
        row = {'query_id': query_id}
        print(query_id)
        retrieved_ids = []  # List to collect all retrieved IDs
        for i, retrieval in enumerate(retrievals, start=1):
            row[f'retrieval_{i}_id'] = retrieval.get('id')
            row[f'retrieval_{i}_title'] = retrieval.get('title')
            row[f'retrieval_{i}_url'] = retrieval.get('url')
            row[f'retrieval_{i}_combined_text'] = retrieval.get('combined_text')
            row[f'retrieval_{i}_distance'] = retrieval.get('distance')
            retrieved_ids.append(retrieval.get('id'))
            # Collect the ID for the list

        row['retrieved_ids'] = retrieved_ids
        row['model_name'] = model_name
        row['query'] = query_lookup[query_id]
        row['relevant_docs'] = ground_truth[query_id]
        row['k'] = k
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main(model_name, k, threshold, history_file_path, golden_path=None, row_limit=100):
    model_name_normalized = model_name.replace("/","_").replace("-","_").replace(".","_")
    query_ids, db, ground_truth = run_history_in_vector_db(row_limit, history_file_path=history_file_path, golden_set_file_path=golden_path)
    print(ground_truth)
    retrieval_results, query_lookup = run_retrieval(query_ids, db, model_name, threshold, k)
    # reshape & save to df and csv
    df = convert_dict_to_df(retrieval_dict=retrieval_results, query_lookup=query_lookup, ground_truth=ground_truth, model_name=model_name, k=k)
    df.to_csv(f"results/{model_name_normalized}_results.csv")
    return retrieval_results


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run the retrieval pipeline with specified parameters.")

     # Add arguments
    parser.add_argument("model_name", type=str, help="Name of the model to use.")
    parser.add_argument("k", type=int, help="Top-K results to retrieve.")
    parser.add_argument("threshold", type=float, help="Threshold for retrieval.")
    parser.add_argument("history_file_path", type=str, help="Path to the browsing history file.")
    parser.add_argument("--golden_path", type=str, default=None, help="Path to the golden query set file (optional).")
    parser.add_argument("--row_limit", type=int, default=100, help="Whether to limit rows from browsing history upon load")
    # Parse arguments
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(
        model_name=args.model_name,
        k=args.k,
        threshold=args.threshold,
        history_file_path=args.history_file_path,
        golden_path=args.golden_path,
        row_limit=args.row_limit
    )



