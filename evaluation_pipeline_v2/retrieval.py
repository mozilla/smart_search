
from sentence_transformers import SentenceTransformer

import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import tracemalloc
import time
from tqdm.auto import tqdm
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
import shutil
from pathlib import Path
import sys
import argparse
import pickle
import json
from datasets import load_dataset, DatasetDict

# Add the parent directory of `evaluation_pipeline` to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.constants import EMBEDDING_MODELS_DICT
# from src.feature_extractor import FeatureExtractor

from model2vec import StaticModel


# (model name, name for load, type)
MODELS = {
    "Xenova/all-MiniLM-L6-v2": ("all-MiniLM-L6-v2", "transformer"),
    "minishlab/potion-retrieval-32M": ("minishlab/potion-retrieval-32M", "static"),
}

def get_model(model_name):
    print(f"Loading model: {model_name}...")
    if model_name not in MODELS:
        raise ValueError(f"{model_name} is not a valid model name")
    load_name, model_type = MODELS[model_name]
    if model_type == "transformer":  # fix: was "transformers"
        return SentenceTransformer(load_name)
    elif model_type == "static":
        return StaticModel.from_pretrained(load_name)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


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
    golden_df = pd.read_csv(golden_query_file_path)
    query_ids = {query: str(hash(query)) for query in golden_df['search_query'].unique()}
    return query_ids


def get_normalized_name(name):
    return name.replace("/", "_").replace("-", "_").replace(".", "_")


@log_performance
def process_history(row_limit, history_file_path):
    browsing_history = pd.read_csv(history_file_path).head(row_limit)
    browsing_history['last_visit_date'] = pd.to_datetime(browsing_history['last_visit_date'], unit='us')
    # fill empty last_visit_date with default value "1970-01-01"
    browsing_history['last_visit_date'] = browsing_history['last_visit_date'].fillna(pd.to_datetime("1970-01-01"))
    browsing_history['combined_text'] = browsing_history['title'].fillna('') + " " + browsing_history['description'].fillna('')
    browsing_history['combined_text_url'] = browsing_history['title'].fillna('') + " " + browsing_history['description'].fillna('') + browsing_history['url'].fillna('')
    browsing_history = browsing_history.loc[browsing_history['combined_text'] != ''].reset_index(drop=True)

    print("history processed, total size:", len(browsing_history))

    return browsing_history


@log_performance
def process_history_from_hf(profile_docs):
    df = profile_docs.copy()

    df['last_visit_date'] = pd.to_datetime(df['last_visit_date'], unit='us')
    df['last_visit_date'] = df['last_visit_date'].fillna(pd.to_datetime("1970-01-01"))

    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['url'] = df['url'].fillna('')

    df['combined_text'] = (df['title'] + ' ' + df['description']).str.strip()
    df['combined_text_url'] = (df['title'] + ' ' + df['description'] + ' ' + df['url']).str.strip()
    df = df.loc[df['combined_text'] != ''].reset_index(drop=True)

    print("history processed, total size:", len(df))
    return df


def load_ground_truth_from_hf(db, profile_query_pairs):

    golden_df = profile_query_pairs #pd.read_csv(golden_df_file_path)
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
            """, (row['search_query'], row['url'])
                       )

    # join to search query to get doc ID for ground truth URL
    results = db.execute(
        '''SELECT a.search_query, a.url, b.rowid
        FROM ground_truth a
        left join search_data b
        on a.url = b.url'''
    ).fetchall()

    ground_truth = {}
    ground_truth_urls = {}
    for query, url, id_ in results:
        query_id = query_ids[query]
        if query_id not in ground_truth:
            ground_truth[query_id] = []
            ground_truth_urls[query_id] = []

        ground_truth[query_id].append(id_)
        ground_truth_urls[query_id].append(url)

    return ground_truth, query_ids, ground_truth_urls


@log_performance
def create_embedding(history, model_name, save_path):
    texts = history['combined_text'].values.tolist()

    model_name_normalized = get_normalized_name(model_name)
    model_path = f"{save_path}/embeddings__{model_name_normalized}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_embedding = pickle.load(f)
            embedding_size = model_embedding.shape[1]
            print(f"Found embedding file for model {model_name}, loaded it from {model_path}")
    else:
        if model_name == 'nomic-ai/nomic-embed-text-v1.5':
            prefix = 'search_document: '
            texts = [prefix + text for text in texts]
        # fe = FeatureExtractor(EMBEDDING_MODELS_DICT, model_name=model_name)
        model = get_model(model_name)
        # model_embedding = fe.get_embeddings(texts)
        model_embedding = model.encode(texts)
        embedding_size = model_embedding.shape[1]

        with open(model_path, "wb") as f:
            pickle.dump(model_embedding, f)

    print("model embeddings size:", model_embedding.shape)

    with open(f"{save_path}/embedding_size__{model_name_normalized}.txt", "w") as f:
        f.write(json.dumps(embedding_size, indent=2))

    return model_embedding, embedding_size

@log_performance
def create_db():
    # vector database
    #db = sqlite3.connect(":memory:")
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
def create_embeddings_table_in_vector_db(db, model_name, embedding_size, model_embedding):
    print("Creating table...")
    EMBEDDING_SIZE = embedding_size
    items = []
    for idx, vec in enumerate(model_embedding):
        items.append((idx, list(vec)))
    model_name_normalized = get_normalized_name(model_name)
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
def query_and_result(model, query, db, model_name, threshold, k):
    model_name_normalized = get_normalized_name(model_name)
    if model_name == 'nomic-ai/nomic-embed-text-v1.5':
        query = 'search_query: ' + query
   
    query_embedding = model.encode([query])[0]
    # using cosine distance
    rows = db.execute(
    f"""
      SELECT
        a.rowid,
        vec_distance_cosine(embedding, ?) AS cosine_distance
      FROM vec_items_{model_name_normalized} a
      inner join search_data b
      on a.rowid = b. rowid

      where b.url not like '%google.com/search?%'
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
        if distance > threshold:
            print("Distance threshold exceeded")
            continue

        # print("doing search now")
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
        description TEXT,
        combined_text TEXT
        )
        ''')
    for idx, row in browsing_history.iterrows():
            db.execute("""
                INSERT INTO search_data (rowid, url, title, description, combined_text)
                VALUES (?, ?, ?, ?, ?)
            """, (idx, row['url'], row['title'], row['description'],  row['combined_text'])
            )
    browsing_history.to_sql("full_history", db, if_exists="replace", index=True)  # Creates the table automatically



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

    ground_truth = {}
    ground_truth_urls = {}
    for query, url, id_ in results:
        query_id = query_ids[query]
        if query_id not in ground_truth:
            ground_truth[query_id] = []
            ground_truth_urls[query_id] = []
            
        ground_truth[query_id].append(id_)
        ground_truth_urls[query_id].append(url)

    return ground_truth, query_ids, ground_truth_urls


def load_data(repo):
    docs = load_dataset(repo, name="docs", download_mode="force_redownload")["train"]
    queries = load_dataset(repo, name="queries", download_mode="force_redownload")["train"]
    qrels = load_dataset(repo, name="qrels", download_mode="force_redownload")["train"]
    return DatasetDict(docs=docs, queries=queries, qrels=qrels)

@log_performance
def run_history_in_vector_db(save_path, model_name, profile, profile_id, hf_repo):

    data = load_data(repo=hf_repo)
    docs, queries, qrels = data["docs"].to_pandas(), data["queries"].to_pandas(), data["qrels"].to_pandas()

    qrels_with_profile = qrels.merge(
        queries[['query_id', 'profile']],
        on='query_id',
        how='left',
        validate='m:1'
    )

    if profile_id:
        profile_key = "profile_id"
        profile_val = profile_id
    elif profile:
        profile_key = "profile"
        profile_val = profile
    else:
        raise Exception("profile_id or profile must be specified")

    profile_docs = docs[docs[profile_key] == profile_val]
    profile_queries = queries[queries[profile_key] == profile_val]
    profile_qrels = qrels_with_profile[qrels_with_profile[profile_key] == profile_val]
    profile_query_pairs = profile_queries.merge(profile_qrels[["query_id", "doc_id", "relevance", "rank"]], on="query_id").merge(docs[["doc_id", "url"]], on="doc_id").drop_duplicates()

    browsing_history = process_history_from_hf(profile_docs)

    # create vector DB
    db = create_db()


    # load in history for joining later
    load_history_in_db(db, browsing_history)

    # print("Getting doc ids for history")
    ground_truth, query_ids, ground_truth_urls = load_ground_truth_from_hf(db, profile_query_pairs)


    # create embeddings for candidate models
    print(f"Creating embeddings for model {model_name}...")
    start_time = time.time()
    model_embedding, embedding_size = create_embedding(browsing_history, model_name=model_name, save_path=save_path)
    print(f"Done! Creating embeddings for model {model_name} took {time.time() - start_time:.2f} seconds.")

    model_name_normalized = get_normalized_name(model_name)

    # create table for embeddings for model
    create_embeddings_table_in_vector_db(db, model_name, embedding_size=embedding_size, model_embedding=model_embedding)

    table_size = get_table_size(db, table_name=model_name_normalized)
    logging.info(f"{model_name_normalized} table size: {table_size}")
    total_db_size_human_readable = format_size(table_size)
    logging.info(f"Table size {model_name_normalized}: {total_db_size_human_readable}")

    return query_ids, db, ground_truth, ground_truth_urls

@log_performance
def run_retrieval(model, query_ids, db, model_name, threshold, k):
    # loop through each query
    retreival_dict = {}
    query_lookup = {}
    # for query in query_ids.keys():
    for query in tqdm(query_ids.keys(), desc="Retrieving", unit="query", total=len(query_ids)):
        # for later identifying query
        query_id = query_ids[query]
        query_lookup[query_id] = query
        # perform retrieval
        results = query_and_result(model, query, db=db,
        model_name=model_name, threshold=threshold, k=k)
        retreival_dict[query_id] = results
    return retreival_dict, query_lookup


def convert_dict_to_df(retrieval_dict, query_lookup, ground_truth, ground_truth_urls, model_name, k):
    rows = []
    for query_id, retrievals in retrieval_dict.items():
        # Flatten each retrieval into a single row with column names based on retrieval index
        row = {'query_id': str(query_id)}
        retrieved_ids = []  # List to collect all retrieved IDs
        retrieved_distances = [] # collect the distances
        for i, retrieval in enumerate(retrievals, start=1):
            row[f'retrieval_{i}_id'] = retrieval.get('id')
            row[f'retrieval_{i}_title'] = retrieval.get('title')
            row[f'retrieval_{i}_url'] = retrieval.get('url')
            row[f'retrieval_{i}_combined_text'] = retrieval.get('combined_text')
            row[f'retrieval_{i}_distance'] = retrieval.get('distance')
            retrieved_ids.append(retrieval.get('id'))
            retrieved_distances.append(retrieval.get('distance'))
            # Collect the ID for the list

        row['retrieved_ids'] = retrieved_ids
        row['retrieved_distances'] = retrieved_distances
        row['model_name'] = model_name
        row['query'] = query_lookup[query_id]
        row['relevant_docs'] = ground_truth[query_id]
        row['relevant_urls'] = ground_truth_urls[query_id]
        row['k'] = k
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def convert_dict_to_csv(retrieval_dict, query_lookup, ground_truth, ground_truth_urls, model_name, k):

    rows = []
    for query_id, retrievals in retrieval_dict.items():

        t_rows = [{'query_id': str(query_id), "query": query_lookup[query_id]}]

        for i in range(len(ground_truth[query_id])):
            if i >= len(t_rows):
                t_rows.append({})
            t_rows[i]["relevant_doc"] = ground_truth[query_id][i]
            t_rows[i]["relevant_url"] = ground_truth_urls[query_id][i]

        for i, retrieval in enumerate(retrievals):
            if i >= len(t_rows):
                t_rows.append({})
            t_rows[i]["retrieved_id"] = retrieval.get('id')
            t_rows[i]["retrieved_title"] = retrieval.get('title')
            t_rows[i]["retrieved_url"] = retrieval.get('url')
            t_rows[i]["retrieved_combined_text"] = retrieval.get('combined_text')
            t_rows[i]["retrieved_combined_distance"] = retrieval.get('distance')

        for i in range(k):
            if i >= len(t_rows):
                t_rows.append({})
            t_rows[i]["rank"] = i+1

        rows.extend(t_rows)

    df = pd.DataFrame(rows)
    return df



def convert_dict_to_json(retrieval_dict, query_lookup, ground_truth, ground_truth_urls, model_name, k):

    dic = {}
    for query_id, retrievals in retrieval_dict.items():

        dic[str(query_id)] = {
            'query_id': str(query_id),
            "query": query_lookup[query_id],
            "relevants": [{"relevant_doc": d, "relevant_url": u} for d, u in zip(ground_truth[query_id], ground_truth_urls[query_id])],
            "retrievals": retrievals,
            "return_size_k": k,
            "model_name": model_name,
        }

    return dic




def main(model_name, k, threshold, save_path, profile, profile_id, hf_repo):

    os.makedirs(Path(save_path), exist_ok=True)

    model_name_normalized = get_normalized_name(model_name)
    # Configure logging
    # logging.basicConfig(
    # filename=f"performance_{model_name_normalized}.log",
    # level=logging.INFO,
    # format="%(asctime)s - %(message)s"
    # )
    query_ids, db, ground_truth, ground_truth_urls = run_history_in_vector_db(save_path=save_path, model_name=model_name, profile=profile, profile_id=profile_id, hf_repo=hf_repo)
    # fe = FeatureExtractor(EMBEDDING_MODELS_DICT, model_name=model_name)
    model = get_model(model_name)
    retrieval_results, query_lookup = run_retrieval(model, query_ids, db, model_name, threshold, k)
    # reshape & save to df and csv
    print("Saving to files...")
    df = convert_dict_to_df(retrieval_dict=retrieval_results, query_lookup=query_lookup, ground_truth=ground_truth, ground_truth_urls=ground_truth_urls, model_name=model_name, k=k)
    df2 = convert_dict_to_csv(retrieval_dict=retrieval_results, query_lookup=query_lookup, ground_truth=ground_truth, ground_truth_urls=ground_truth_urls, model_name=model_name, k=k)
    dic3 = convert_dict_to_json(retrieval_dict=retrieval_results, query_lookup=query_lookup, ground_truth=ground_truth, ground_truth_urls=ground_truth_urls, model_name=model_name, k=k)
    time_stamp = int(time.time())

    df.to_csv(f"{save_path}/result__{model_name_normalized}.csv", index=False)
    df2.to_csv(f"{save_path}/result_rank__{model_name_normalized}.csv", index=False)
    with open(f"{save_path}/result_json__{model_name_normalized}.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dic3, indent=2, ensure_ascii=False))
    print("Saved to files.")

    return db, retrieval_results, df

def validate_profile_args(args, parser):
    # require at least one
    if args.profile is None and args.profile_id is None:
        parser.error("You must provide either --profile or --profile_id.")


if __name__ == "__main__":
     # Create the argument parser
     parser = argparse.ArgumentParser(description="Run the retrieval pipeline with specified parameters.")
      # Add arguments
     parser.add_argument("--model_name", type=str,default='Xenova/all-MiniLM-L6-v2', help="Name of the model to use.")
     parser.add_argument("--k", type=int, default=2, help="Top-K results to retrieve.")
     parser.add_argument("--threshold", type=float, default=10.0, help="Threshold for retrieval.")
     parser.add_argument("--save_path", type=str, default=None, help="Path for save results.")
     parser.add_argument("--profile", type=str, default=None, help="Profile name in dataset.")
     parser.add_argument("--profile_id", type=str, default=None, help="Profile ID in dataset. If provided with --profile, this takes precedence.")
     parser.add_argument("--hf_repo", type=str, default=None, help="Hugging Face dataset repo (e.g., org/dataset).")

     # Parse arguments
     args = parser.parse_args()
     validate_profile_args(args, parser)
     # Call the main function with parsed arguments
     main(
         model_name=args.model_name,
         k=args.k,
         threshold=args.threshold,
         save_path=args.save_path,
         profile=args.profile,
         profile_id=args.profile_id,
         hf_repo=args.hf_repo,
     )

