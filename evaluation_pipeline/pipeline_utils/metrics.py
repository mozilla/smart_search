
import numpy as np
import json
import re
from sklearn.metrics import ndcg_score


def calc_precision_at_k(relevant_docs, retrieved_docs, k):
    """
    Compute Precision@K.
    Parameters:
        relevant_docs (set): Set of relevant document IDs.
        retrieved_docs (list): List of retrieved document IDs in ranked order.
        k (int): Number of top results to consider.
    Returns:
        float: Precision@K score.
    """
    retrieved_at_k = retrieved_docs[:k]
    intersection = set(retrieved_at_k) & set(relevant_docs)
    return len(intersection) / float(k)


def calc_recall_at_k(relevant_docs, retrieved_docs, k):
    """
    Compute Recall@K.
    Parameters:
        relevant_docs (set): Set of relevant document IDs.
        retrieved_docs (list): List of retrieved document IDs in ranked order.
        k (int): Number of top results to consider.
    Returns:
        float: Recall@K score.
    """
    retrieved_at_k = retrieved_docs[:k]
    intersection = set(retrieved_at_k) & set(relevant_docs)
    return len(intersection) / len(relevant_docs)


def calc_ndcg(relevant_docs, retrieved_docs, k, score_type='rank',  retrieved_distances=None):
    if len(retrieved_docs) <2:
        return 0.00
    y_true = np.array([[1 if doc in relevant_docs else 0 for doc in retrieved_docs]])
    if score_type == 'rank':
        y_scores = np.array([[1 / (rank + 1) for rank in range(len(retrieved_docs))]])
    elif score_type == 'distance':
        y_scores = np.array([[1 - dist for dist in retrieved_distances]])
    ndcg = ndcg_score(y_true, y_scores, k=k)
    return ndcg

def calc_reciprocal_rank(relevant_docs, retrieved_docs):
    """
    Compute Reciprocal Rank (RR).
    Parameters:
        relevant_docs (set): Set of relevant document IDs.
        retrieved_docs (list): List of retrieved document IDs in ranked order.
    Returns:
        float: MRR score.
        Can be used to compute mean reciprocal rank for number of queries Q
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1 / rank
    return 0.0


def calc_average_precision(relevant_docs, retrieved_docs, k):
    """
    Compute Average Precision (AP).
    Parameters:
        relevant_docs (list): List of relevant document IDs.
        retrieved_docs (list): List of retrieved document IDs in ranked order.
        k: average precision from 1 to k 
    Returns:
        float: AP score.
        Can be used to calculate mean average precision for number of queries Q
    """
    if len(retrieved_docs) < 2:
        return 0.0

    total_precision = 0.0

    for i in range(1, len(retrieved_docs)+1): 
        precision = calc_precision_at_k(relevant_docs, retrieved_docs, k=i)
        total_precision += precision

    if k > len(retrieved_docs):
        print(f"k is higher than retrieval {len(retrieved_docs)}")

    elif k < len(retrieved_docs):
        print(f"k is lower than retrieval {len(retrieved_docs)}")

    average_precision = total_precision / float(k)

    return average_precision


def format_judge_response(answer):
    try:
        formatted_answer = json.loads(answer[0]['generated_text'])
    except:
        try:
            print("cleaning json")
            formatted_answer = json.loads(answer[0]['generated_text'].replace("\n",""))
        except:
            try:
                print("trying another way")
                formatted_answer = json.loads(re.sub(r'```json|```', '', answer[0]['generated_text'].replace("\n","")).strip())
            except:
                formatted_answer = answer[0]['generated_text']
    return formatted_answer



def calc_jaccard_index(text_a, text_b):
    try:
        text_a = text_a.lower()
        text_b = text_b.lower()
        # Tokenize into sets of words
        set_a = set(text_a.split())
        set_b = set(text_b.split())
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union
    except:
        return 0.00

def calc_overlap_coefficient(text_a, text_b):
    try:
        text_a = text_a.lower()
        text_b = text_b.lower()
        set1, set2 = set(text_a.split()), set(text_b.split())
        intersection = len(set1 & set2)
        min_length = min(len(set1), len(set2))
        return intersection / min_length
    except:
        return 0.00



def run_traditional_eval(query_id, query, relevant_docs, retrieved_docs, retrieved_distances, retrieved_texts, relevant_texts, k):
    row = {'query_id': query_id}
    row['query'] = query
    # calcuate traditional IR metrics
    precision = calc_precision_at_k(relevant_docs, retrieved_docs, k)
    recall = calc_recall_at_k(relevant_docs, retrieved_docs, k)
    ndcg = calc_ndcg(relevant_docs, retrieved_docs,score_type='rank',retrieved_distances=retrieved_distances, k=k)
    reciprocal_rank = calc_reciprocal_rank(relevant_docs, retrieved_docs)
    average_precision = calc_average_precision(relevant_docs, retrieved_docs, k=k)
    jaccard_indices = [calc_jaccard_index(query, text) for text in retrieved_texts]
    overlap_coefficients = [calc_overlap_coefficient(query, text) for text in retrieved_texts]
    query_overlap_coefficients = np.mean([calc_overlap_coefficient(query, text) for text in relevant_texts])

    # store in row
    row['retrieved_ids'] = retrieved_docs
    row['relevant_docs'] = relevant_docs
    row[f'precision@{k}'] = precision
    row[f'recall@{k}'] = recall
    row[f'ndcg@{k}'] = ndcg
    row['reciprocal_rank'] = reciprocal_rank
    row['average_precision'] = average_precision
    row['jaccard_indices'] = jaccard_indices
    row['overlap_coefficients'] = overlap_coefficients
    row['query_overlap_coefficients'] = query_overlap_coefficients


    return row
