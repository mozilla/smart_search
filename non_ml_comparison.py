from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from urllib.parse import urlparse, urlunparse
import nltk
from evaluation_pipeline.retrieval import process_history, convert_dict_to_df
from rank_bm25 import BM25Okapi
import argparse


# Ensure NLTK tokenizer data is available
nltk.download("punkt")

def normalize_url(url):
    parsed = urlparse(url)
    # Remove fragments and queries, keep scheme + netloc + path
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))



def process_data(history_file_path, query_file_path):
    # Define Corpus and Queries
    history = process_history(row_limit=10001, history_file_path=history_file_path)
    history['norm_url'] = history['url'].apply(normalize_url)
    query_set = pd.read_csv(query_file_path)
    query_set['norm_url'] = query_set['url'].apply(normalize_url)
    history['index_col'] = history.index
    query_list = query_set[['search_query','url','norm_url']]
    relevant_docs = pd.merge(query_list, history, on='url', how='inner')
    norm_relevant_docs = pd.merge(query_list, history, on='norm_url', how='inner')
    relevant_docs = relevant_docs[['search_query','url','index_col','title','description','combined_text']]
    unique_search_query_df = pd.DataFrame(relevant_docs.groupby('search_query').agg({
        'url': list,   # Combine all URLs for the same query into a list
        'index_col': list,  # Optionally include the index as a list
        'title': list,
        'description': list,
        'combined_text': list,
    }).reset_index())

    norm_relevant_docs = norm_relevant_docs[['search_query','norm_url','index_col','title','description','combined_text']]

    norm_unique_search_query_df = pd.DataFrame(norm_relevant_docs.groupby('search_query').agg({
        'norm_url': list,
        'index_col': list,  # Optionally include the index as a list
        'title': list,
        'description': list,
        'combined_text': list,
    }).reset_index())

    return history, unique_search_query_df, query_list, query_set, norm_unique_search_query_df

def create_ground_truth(query_list, unique_search_query_df, norm_unique_search_query_df, history, query_set):
    query_ids = {query: str(hash(query)) for query in query_list['search_query'].unique()}
    unique_search_query_df['query_id'] = unique_search_query_df['search_query'].map(query_ids)
    norm_unique_search_query_df['query_id'] = norm_unique_search_query_df['search_query'].map(query_ids)
    query_lookup = {}
    for query in unique_search_query_df['search_query'].unique():
        print(query)
        query_id = query_ids[query]
        query_lookup[query_id] = query
    documents = history['combined_text'].to_list()
    queries = query_set['search_query'].unique()

    ground_truth = dict(zip(unique_search_query_df['query_id'], unique_search_query_df['index_col']))
    ground_truth_urls = dict(zip(unique_search_query_df['query_id'], unique_search_query_df['url']))
    norm_ground_truth = dict(zip(norm_unique_search_query_df['query_id'], norm_unique_search_query_df['index_col']))
    norm_ground_truth_urls = dict(zip(norm_unique_search_query_df['query_id'], norm_unique_search_query_df['norm_url']))
    return ground_truth, ground_truth_urls, unique_search_query_df, query_ids, documents, queries, query_lookup, norm_ground_truth, norm_ground_truth_urls



# TF-IDF Retrieval
def tfidf_retrieval(history, unique_search_query_df, top_k):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(history["combined_text"])
    results = {}

    for i, row in unique_search_query_df.iterrows():
        query_id = row['query_id']
        query_matrix = vectorizer.transform([row['search_query']])
        similarities = cosine_similarity(query_matrix, tfidf_matrix).flatten()
        distances = 1 - similarities
        ranked_indices = similarities.argsort()[::-1][:top_k]


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
        results[query_id] = retrievals
    return results

# BM25 Retrieval
def bm25_retrieval(history, unique_search_query_df, top_k):
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in history["combined_text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    results = {}

    for i, row in unique_search_query_df.iterrows():
        query_id = row['query_id']
        tokenized_query = word_tokenize(row['search_query'].lower())
        scores = bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

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
        results[query_id] =  retrievals
    return results


def exact_match_retrieval(history, unique_search_query_df, top_k):
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

    for i, row in unique_search_query_df.iterrows():
        query_id = row['query_id']
        query = row['search_query'].lower()
       # query_terms = query.split()  # Split query into words
        #pattern = r"\b(" + "|".join(query_terms) + r")\b"
        #matches = history[history['combined_text'].str.contains(pattern, case=False, na=False, regex=True)]

        # Filter documents in the history DataFrame where `combined_text` contains the exact query terms
        matches = history[history['combined_text'].str.contains(rf'\b{query}\b', case=False, na=False)]

        # Limit to top_k results
        matches = matches.head(top_k)
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

        results[query_id] =  retrievals

    return results



def exact_match_terms_retrieval_frecency(history, unique_search_query_df, top_k):
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

    for i, row in unique_search_query_df.iterrows():
        query_id = row['query_id']
        query = row['search_query'].lower()
        query_terms = query.split()  # Split query into words
        pattern = r"\b(" + "|".join(query_terms) + r")\b"
        matches = history[history['combined_text'].str.contains(pattern, case=False, na=False,regex=True)].sort_values(by='frecency', ascending=False)

        # Filter documents in the history DataFrame where `combined_text` contains the exact query terms
#        matches = history[history['combined_text'].str.contains(rf'\b{query}\b', case=False, na=False)]

        # Limit to top_k results
        matches = matches.head(top_k)
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

        results[query_id] =  retrievals

    return results


# Run and Combine TF-IDF and BM25 Results
def main(k, history_file_path, query_file_path):
    history, unique_search_query_df, query_list, query_set,  norm_unique_search_query_df = process_data(history_file_path, query_file_path)
    ground_truth, ground_truth_urls, unique_search_query_df, query_ids, documents, queries, query_lookup, norm_ground_truth, norm_ground_truth_urls = create_ground_truth(query_list, unique_search_query_df,  norm_unique_search_query_df, history, query_set)

    
    # retrieval
    tfidf_results = tfidf_retrieval(history, unique_search_query_df, top_k=k)
    bm25_results = bm25_retrieval(history, unique_search_query_df, top_k=k)
    # Perform Exact Match Retrieval
    exact_match_results = exact_match_retrieval(history, unique_search_query_df, top_k=k)
    # exact match sort by freceny desc
    exact_match_results_frecency = exact_match_terms_retrieval_frecency(history, unique_search_query_df, top_k=k)
    
    # Create Output DataFrames
    # tfidf
    tfidf_df = convert_dict_to_df(tfidf_results, query_lookup, ground_truth=ground_truth, ground_truth_urls=ground_truth_urls,  norm_ground_truth=norm_ground_truth, norm_ground_truth_urls=norm_ground_truth_urls, model_name="TF-IDF", k=k)

    # bm25
    bm25_df = convert_dict_to_df(bm25_results,query_lookup, ground_truth=ground_truth, ground_truth_urls=ground_truth_urls, norm_ground_truth=norm_ground_truth, norm_ground_truth_urls=norm_ground_truth_urls, model_name="BM25", k=k)

    # exact match
    exact_match_df = convert_dict_to_df(exact_match_results,query_lookup, ground_truth=ground_truth, ground_truth_urls=ground_truth_urls, norm_ground_truth=norm_ground_truth, norm_ground_truth_urls=norm_ground_truth_urls,model_name="exact_match", k=k)

    # exact match (sort by frecency desc)
    exact_match_results_frecency_df = convert_dict_to_df(exact_match_results_frecency,query_lookup, ground_truth=ground_truth, ground_truth_urls=ground_truth_urls, norm_ground_truth=norm_ground_truth, norm_ground_truth_urls=norm_ground_truth_urls,model_name="exact_match_terms_freceny", k=k)

    
    # Save Results to CSV
    tfidf_df.to_csv("results/tfidf_retrieval_results.csv", index=False)
    bm25_df.to_csv("results/bm25_retrieval_results.csv", index=False)
    exact_match_df.to_csv("results/exact_match_retrieval_results.csv", index=False)
    exact_match_results_frecency_df.to_csv("results/exact_match_frecency_retrieval_results.csv", index=False)


if __name__ == "__main__":
     # Create the argument parser
     parser = argparse.ArgumentParser(description="Non ML retrieval")
      # Add arguments
     parser.add_argument("--history_file_path", default="output.csv", type=str, help="File path for browsing history csv")
     parser.add_argument("--golden_path", default="golden_query_set.csv",type=str, help="Path for golden query set")
     parser.add_argument("--k", type=int, default=2, help="Top-K results used in retrieval being evaluated.")

     args = parser.parse_args()
     # Call the main function with parsed arguments
     main(
         k=args.k,
         history_file_path= args.history_file_path,
         query_file_path=args.golden_path
     )


