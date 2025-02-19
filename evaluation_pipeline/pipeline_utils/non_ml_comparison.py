from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from urllib.parse import urlparse, urlunparse
import nltk
from pipeline_utils.features import process_history
from rank_bm25 import BM25Okapi
import pandas as pd
import re
from urllib.parse import unquote
import gc
from pipeline_utils.utils import build_model_run_details_dict, convert_dict_to_df
from collections import defaultdict





def normalize_url(url):
    parsed = urlparse(url)
    # Remove fragments and queries, keep scheme + netloc + path
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))


def get_relevant_docs(history, url_type, query_file_path):
    golden_queries = pd.read_csv(query_file_path)

    if 'query' not in golden_queries.columns:
        golden_queries['query'] = golden_queries['search_query']

    golden_queries['norm_url'] = golden_queries['url'].apply(normalize_url)
    golden_queries = golden_queries[['query','url','norm_url']]

    relevant_docs = pd.merge(
        golden_queries,
        history,
        on=url_type,
        how='inner',
        suffixes=('_left', '_right')
    )


    relevant_docs = relevant_docs.rename(columns={
         "url_left": "url",
         "norm_url_left": "norm_url",
    })


    relevant_docs = relevant_docs[['query', 'url', 'norm_url', 'index_col', 'title', 'description', 'combined_text']]


    golden_query_df = pd.DataFrame(relevant_docs.groupby('query').agg({
            'url': list,
            'norm_url': list,
            'index_col': list,
            'title': list,
            'description': list,
            'combined_text': list,
        }
    )).reset_index()

    return golden_query_df



def process_data(history_file_path, model_run_details):

    processed_history, processed_history_file_path = process_history(
        row_limit=model_run_details['row_limit'],
        history_file_path=history_file_path,
        preprocess=model_run_details['preprocess'],
        features=model_run_details['features'],
    )

    processed_history['index_col'] = processed_history.index

    return processed_history

def create_ground_truth(golden_query_df, url_type):

    # Each query_id maps to a dict of lists
    ground_truth_dict = defaultdict(lambda: {
        "doc_ids": [],
        f"{url_type}s": [],
        "combined_texts": []
    })

    for query, index_col, url, combined_text in zip(
        golden_query_df["query"],
        golden_query_df["index_col"],
        golden_query_df[f"{url_type}"],
        golden_query_df["combined_text"],
        ):
        ground_truth_dict[query]["doc_ids"].extend(index_col)
        ground_truth_dict[query][f"{url_type}s"].extend(url)
        ground_truth_dict[query]["combined_texts"].extend(combined_text)

    return ground_truth_dict


# TF-IDF Retrieval
def tfidf_retrieval(history,  golden_query_df, model_run_details):

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(history["combined_text"])
    results = {}

    for i, row in golden_query_df.iterrows():

        query_matrix = vectorizer.transform(
            [row['query']]
            )

        if model_run_details['distance_measure'] =='cosine':

            similarities = cosine_similarity(
                query_matrix,
                tfidf_matrix).flatten()

            distances = 1 - similarities

            ranked_indices = similarities.argsort()[::-1][:model_run_details['k']]

        elif model_run_details['distance_measure'] == 'L2':

            distances = euclidean_distances(
                query_matrix,
                tfidf_matrix).flatten()

            ranked_indices = distances.argsort()[:model_run_details['k']]


        retrievals = [
            {
                "id": history.iloc[idx]["index_col"],  # Use original index from `history`
                "title": history.iloc[idx]["title"],
                "url": history.iloc[idx]["url"],
                "combined_text": history.iloc[idx]["combined_text"],
                "distance": distances[idx]
            }
            for idx in ranked_indices
        ]
        results[row['query']] = retrievals

    return results

# BM25 Retrieval
def bm25_retrieval(history, golden_query_df, model_run_details):
    nltk.download("punkt")

    tokenized_corpus = [word_tokenize(
        doc.lower()
        ) for doc in history["combined_text"]]

    bm25 = BM25Okapi(tokenized_corpus)
    results = {}

    for i, row in golden_query_df.iterrows():

        tokenized_query = word_tokenize(
            row['query'].lower()
            )

        scores = bm25.get_scores(tokenized_query)

        ranked_indices = np.argsort(scores)[::-1][:model_run_details['k']]

        retrievals = [
            {
                "id": history.iloc[idx]["index_col"],  # Use original index from `history`
                "title": history.iloc[idx]["title"],
                "url": history.iloc[idx]["url"],
                "combined_text": history.iloc[idx]["combined_text"],
                "distance": scores[idx]
            }
            for idx in ranked_indices
        ]
        results[row['query']] =  retrievals

    return results


def exact_match_retrieval(history, golden_query_df, model_run_details):
    """
    Exact Match Retrieval: Matches documents where the query terms exactly match the document terms.
    
    Args:
        history (pd.DataFrame): DataFrame containing the documents with `combined_text` and metadata (e.g., title, URL, index_col).
        unique_search_query_df (pd.DataFrame): DataFrame with search queries and associated metadata.
        top_k (int): Number of top results to retrieve for each query.

    Returns:
        list[dict]: A list of dictionaries containing the retrieval results for each query.
    """
    results = {}

    for i, row in golden_query_df.iterrows():

        query = row['query'].lower()

        # Filter documents in the history DataFrame where `combined_text` contains the exact query terms
        matches = history[history['combined_text'].str.contains(rf'\b{query}\b', case=False, na=False)]

        # Limit to top_k results
        matches = matches.head(
            model_run_details['k']
            )

        # Format the retrieval results
        retrievals = [
            {
                "id": doc["index_col"],
                "title": doc["title"],
                "url": doc["url"],
                "combined_text": doc["combined_text"],
                "distance": 0  # Exact match has no distance concept, so we use 0 as a placeholder
            }
            for _, doc in matches.iterrows()
        ]

        results[row['query']] =  retrievals

    return results


def exact_match_overlap(history, golden_query_df, model_run_details):
    results = {}
    for i, row in golden_query_df.iterrows():
        query_terms = set(row['query'].lower().split())  # Convert query to a set of words

        history['overlap_count'] = history['combined_text'].apply(
            lambda text: len(query_terms.intersection(set(str(text).lower().split())))
        )

        k = model_run_details['k']
        print(k)

        # Keep only rows where there is at least one matching word, then sort by overlap & frecency
        matches = history[history['overlap_count'] > 0].sort_values(by=['overlap_count', 'frecency'], ascending=[False, False])[:k]
        retrievals = [
            {
                "id": doc["index_col"],
                "title": doc["title"],
                "url": doc["url"],
                "combined_text": doc["combined_text"],
                "distance": 0  # Exact match has no distance concept, so we use 0 as a placeholder
            }
            for _, doc in matches.iterrows()
        ]

        results[row['query']] =  retrievals

        return results
    

def exact_match_terms_retrieval_frecency(history, golden_query_df, model_run_details):
    """
    Exact Match Retrieval: Matches documents where the query terms exactly match the document terms.
    
    Args:
        history (pd.DataFrame): DataFrame containing the documents with `combined_text` and metadata (e.g., title, URL, index_col).
        unique_search_query_df (pd.DataFrame): DataFrame with search queries and associated metadata.
        top_k (int): Number of top results to retrieve for each query.

    Returns:
        list[dict]: A list of dictionaries containing the retrieval results for each query.
    """
    results = {}

    for i, row in golden_query_df.iterrows():
        query = row['query'].lower()
        query_terms = query.split()  # Split query into words
        pattern = r"\b(" + "|".join(query_terms) + r")\b"
        # Escape regex special characters to ensure plain text search
        escaped_pattern = re.escape(pattern)
        matches = history[history['combined_text'].str.contains(escaped_pattern, case=False, na=False,regex=True)].sort_values(by='frecency', ascending=False)

        # Limit to top_k results
        matches = matches.head(model_run_details['k'])
        # Format the retrieval results
        retrievals = [
            {
                "id": doc["index_col"],
                "title": doc["title"],
                "url": doc["url"],
                "combined_text": doc["combined_text"],
                "distance": 0  # Exact match has no distance concept, so we use 0 as a placeholder
            }
            for _, doc in matches.iterrows()
        ]

        results[row['query']] =  retrievals

    return results


def fixup_uri_spec(url: str, match_behavior: str) -> str:
    """Normalize URL by removing prefixes and unescaping."""
    url = unquote(url)  # Decode URL-encoded characters

    # Remove common URL prefixes
    if url.startswith("http://"):
        url = url[7:]
    elif url.startswith("https://"):
        url = url[8:]
    elif url.startswith("ftp://"):
        url = url[6:]

    if match_behavior == "MATCH_ANYWHERE_UNMODIFIED":
        return url

    return url


def find_anywhere(token: str, source: str) -> bool:
    """Finds token anywhere in the source string (case-insensitive)."""
    return token.lower() in  str(source).lower()  # Convert NaN/float to string


def find_on_boundary(token: str, source: str) -> bool:
    """Finds token only on word boundaries (case-insensitive)."""
    return bool(re.search(r'\b' + re.escape(token) + r'\b', str(source), re.IGNORECASE))


def get_search_function(match_behavior: str):
    """Returns the appropriate search function."""
    if match_behavior in ["MATCH_ANYWHERE", "MATCH_ANYWHERE_UNMODIFIED"]:
        return find_anywhere
    return find_on_boundary


def match_autocomplete_pandas(df, search_string, match_behavior,  k):
    """
    Searches a Pandas DataFrame of moz_places using filtering and matching logic.

    Args:
        df: Pandas DataFrame containing the browser history.
        search_string (str): User's search input.
        match_behavior (str): Matching type (MATCH_ANYWHERE, MATCH_BOUNDARY).


    Returns:
        Filtered Pandas DataFrame with matching results.
    """
    search_tokens = search_string.lower().split()
    search_function = get_search_function(match_behavior)

    # Normalize URLs
    df["fixed_url"] = df["url"].apply(lambda x: fixup_uri_spec(x, match_behavior))
    if 'id' not in df.columns:
        df['id'] = df.index
    if 'frecency' not in df.columns:
        df['frecency'] = 0


    # Apply search logic
    def matches(row):
        return all(
            search_function(token, row["fixed_url"]) or
            search_function(token, row["title"])
            for token in search_tokens
        )

    result_df = df[df.apply(matches, axis=1)][["id", "url", "title", "combined_text","frecency"]].copy()

    return result_df.sort_values(by=["frecency", "id"], ascending=[False, False]).head(k)



def current_suggest_match(history, golden_query_df, model_run_details, match_behavior):
    results = {}
    for i, row in golden_query_df.iterrows():
        search_query = row['query']
        results_df = match_autocomplete_pandas(history, search_string=search_query, match_behavior=match_behavior, k=model_run_details['k'])
        retrievals = [
            {
                "id": doc["id"],
                "title": doc["title"],
                "url": doc["url"],
                "combined_text": doc["combined_text"],
                "distance": 0  # Exact match has no distance concept, so we use 0 as a placeholder
            }
            for _, doc in results_df.iterrows()
        ]
        results[row['query']] =  retrievals
    return results



def save_results(retrieval_results, model_run_details, ground_truth, norm_ground_truth, save_dir):

    result_df = convert_dict_to_df(
        retrieval_results,
        ground_truth,
        norm_ground_truth,
        model_run_details
    )

    result_df['model_run_details'] = [model_run_details] * len(result_df)

    save_file_name = f'{save_dir}'+model_run_details['retrieval_file_name']+'.csv'

    result_df.to_csv(save_file_name)

    return save_file_name


def non_ml_baseline_pipeline(history_file_path, save_dir, query_file_path, model_run_details):

    history = process_data(
        history_file_path,
        model_run_details
    )


    golden_query_df = get_relevant_docs(
        history,
        url_type='url',
        query_file_path=query_file_path
    )


    ground_truth = create_ground_truth(
        golden_query_df,
        url_type='url'
    )


    norm_golden_query_df = get_relevant_docs(
        history,
        url_type='norm_url',
        query_file_path=query_file_path
    )


    norm_ground_truth = create_ground_truth(
        norm_golden_query_df,
        url_type='norm_url'
    )

    tfidf_results = tfidf_retrieval(
        history,
        golden_query_df,
        model_run_details
    )


    bm25_results = bm25_retrieval(
        history,
        golden_query_df,
        model_run_details
    )


    exact_match_results = exact_match_retrieval(
        history,
        golden_query_df,
        model_run_details
    )

    exact_match_results_frecency = exact_match_terms_retrieval_frecency(
        history,
        golden_query_df,
        model_run_details
    )


    # current suggest implementation
    suggest_match_anywhere_results = current_suggest_match(
        history,
        golden_query_df,
        model_run_details,
        match_behavior='MATCH_ANYWHERE'
    )


    suggest_match_boundary_results = current_suggest_match(
        history,
        golden_query_df,
        model_run_details,
        match_behavior='MATCH_BOUNDARY'
    )

    exact_match_overlap_results = exact_match_overlap(
        history,
        golden_query_df,
        model_run_details
    )


    all_results = [tfidf_results,
                   bm25_results,
                   exact_match_results,
                   exact_match_results_frecency,
                   suggest_match_anywhere_results,
                   suggest_match_boundary_results,
                   exact_match_overlap_results
                   ]

    # retrieval
    model_names = ['tfidf', 'bm25','exact_match','exact_match_frecency','suggest_match_anywhere','suggest_match_boundary','exact_match_overlap']

    results_dict = dict(zip(model_names, all_results))

    features = model_run_details['features']
    preprocess = model_run_details['preprocess']
    row_limit = model_run_details['row_limit']
    k=model_run_details['k']

    # save results
    saved_files = []
    for model_name, retrieval_results in results_dict.items():

        model_run_details = build_model_run_details_dict(history_file_path, model_name, features, quantized_model=False, distance_measure='', pooling='', preprocess=preprocess, row_limit=row_limit, k=k)
        model_run_details['embedding_dim'] = 0


        save_file_name = save_results(
            retrieval_results,
            model_run_details,
            ground_truth,
            norm_ground_truth,
            save_dir
        )
        saved_files.append(save_file_name)

    del results_dict
    del history
    del golden_query_df
    gc.collect()
    return saved_files


def main(history_file_path, save_dir, query_file_path, model_run_details):
    saved_files = non_ml_baseline_pipeline(history_file_path, save_dir, query_file_path, model_run_details)


if __name__ == "__main__":
    main()
