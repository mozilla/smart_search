from urllib.parse import urlparse, urlunparse
import tracemalloc
import logging
import time
from typing import List
import struct
import tldextract
from urllib.parse import urlparse, unquote
import sqlite3
import ast
import pandas as pd
import re
import unicodedata

def clean_text(text):
    """Lowercases, removes extra spaces, special characters, and normalizes Unicode."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()  # Convert to lowercase & trim spaces
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[\t\n\r]', ' ', text)  # Replace newlines/tabs with space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = unicodedata.normalize("NFKD", text)  # Normalize Unicode
    return text


def normalize_url(url):
    parsed = urlparse(url)
    # Remove fragments and queries, keep scheme + netloc + path
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def preprocess_url(url, keep_subdomain=False):
    """Extracts and normalizes domain from a URL."""
    parsed_url = urlparse(url)
    domain_parts = tldextract.extract(parsed_url.netloc)

    # Extract root domain or full domain (subdomain.example.com)
    domain = f"{domain_parts.domain}.{domain_parts.suffix}" if not keep_subdomain else parsed_url.netloc
    return unquote(domain)  # Decode URL encoding (e.g., %20 → " ")


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


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)



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




def convert_to_list(value):
    if isinstance(value, str):
        value = value.replace('np.int64(', '').replace(')', '')
        value = value.replace('np.float64(', '').replace(')', '')
        return ast.literal_eval(value)
    return value



def get_combined_texts_uniform_k(df, k):
    # Identify retrieval columns and sort them numerically
    retrieval_cols = sorted(
        [col for col in df.columns if 'retrieval_' in col and '_combined_text' in col],
        key=lambda x: int(x.split('_')[1])
    )

    # Extract relevant retrieval columns as a NumPy array
    retrieval_matrix = df[retrieval_cols].to_numpy()

    # Slice the matrix up to `k` columns for all rows
    sliced_matrix = retrieval_matrix[:, :k]

    # Convert to a list of lists
    result = sliced_matrix.tolist()
    return result


def convert_dict_to_df(retrieval_dict, ground_truth, norm_ground_truth, model_run_details):

    rows = []

    for query, retrievals in retrieval_dict.items():
        # Flatten each retrieval into a single row with column names based on retrieval index
        row = {'query': query}
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

        row['retrieved_ids'] = retrieved_ids
        row['retrieved_distances'] = retrieved_distances
        row['model_name'] = model_run_details['model_name']
        row['relevant_docs'] = ground_truth[query]['doc_ids']
        row['relevant_urls'] = ground_truth[query]['urls']
        row['relevant_combined_text'] = ground_truth[query]['combined_texts']
        row['norm_relevant_docs'] = norm_ground_truth[query]['doc_ids']
        row['norm_relevant_urls'] = norm_ground_truth[query]['norm_urls']
        row['norm_relevant_combined_text'] = norm_ground_truth[query]['combined_texts']
        row['k'] = model_run_details['k']
        rows.append(row)

    df = pd.DataFrame(rows)

    return df


def build_model_run_details_dict(history_file_path, model_name, features, quantized_model, distance_measure, pooling, preprocess, row_limit, k, binary_quantization=False, binary_quantization_coarse_filter=100):

    dataset = history_file_path.split('/')[-1].replace('.csv','')

    model_name_normalized = model_name.replace("/","_").replace("-","_").replace(".","_")

    feature_string = "_".join(features)

    suffix = '_quantized' * quantized_model
    binary_quantization_suffix = '_binary_quantized' * binary_quantization
    model_run_details  = {'model_name': model_name,
                          'model_name_normalized': model_name_normalized,
                          'quantized': quantized_model,
                          'suffix': suffix,
                          'distance_measure': distance_measure,
                          'pooling': pooling,
                          'row_limit': row_limit,
                          'features': features,
                          'feature_string': feature_string,
                          'preprocess': preprocess,
                          'k': k,
                          'embedding_file_name': f"{dataset}_{model_name_normalized}_{suffix}_{pooling}_{distance_measure}_{feature_string}_{preprocess*'preprocess'}_{row_limit}",
                          'retrieval_file_name': f"{dataset}_{model_name_normalized}_{suffix}_{pooling}_{distance_measure}_{feature_string}_{preprocess*'preprocess'}__top{k}_{row_limit}{binary_quantization_suffix}",
                          'dataset': dataset,
                          'binary_quantization': binary_quantization,
                          'binary_quantization_coarse_filter':binary_quantization_coarse_filter
                           }

    return model_run_details




def plot_retrieval(file_path):
     import pandas as pd
     import matplotlib.pyplot as plt
     import textwrap

     df = pd.read_csv(file_path)
     
     wrapped_labels = ["\n".join(textwrap.wrap(q, width=20)) for q in df["query"]]
     
     # Dynamically combine all retrieval_*_url columns into one
     retrieval_url_columns = [col for col in df.columns if "retrieval_" in col and "_url" in col]
     df["retrieved_urls"] = df[retrieval_url_columns].apply(lambda row: [url for url in row if pd.notna(url)], axis=1)
     
     retrieval_combined_text_columns = [col for col in df.columns if "retrieval_" in col and "_combined_text" in col]
     df["retrieved_texts"] = df[retrieval_combined_text_columns].apply(lambda row: [combined_text for combined_text in row if pd.notna(combined_text)], axis=1)
     
     # Split queries into two halves
     mid_index = len(df) // 2
     df1 = df.iloc[:mid_index].reset_index(drop=True)  # First half
     df2 = df.iloc[mid_index:].reset_index(drop=True)  # Second half
     
     # Create the visualization with two subplots
     fig, axes = plt.subplots(ncols=2, figsize=(12, 5), sharex=True, sharey=False)
     
     for ax, df_subset in zip(axes, [df1, df2]):
         ax.set_yticks(range(len(df_subset)))
         ax.set_yticklabels(df_subset["query"], fontsize=8, ha="right")
     
         x_labels = ["Top 1", "Top 2", "Top 3"]
         ax.set_xticks(range(3))
         ax.set_xticklabels(x_labels, fontsize=8)
     
         box_height = 0.4
         box_width = 0.3
     
         for i, row in df_subset.iterrows():
             retrieved_urls = row["retrieved_urls"][:3]
             relevant_urls = row.get("relevant_urls", [])  
     
             colors = ["blue" if url in relevant_urls else "gray" for url in retrieved_urls]
     
             for j, color in enumerate(colors):
                 ax.add_patch(plt.Rectangle((j - box_width / 2, i - box_height / 2), box_width, box_height, color=color))
     
         ax.set_xlim(-0.5, 2.5)
         ax.set_ylim(-0.5, len(df_subset) - 0.5)
     
     
     # Set a shared title
     fig.suptitle("Information Retrieval Results", fontsize=12)
     plt.grid(False)
     plt.tight_layout()
     plt.show()