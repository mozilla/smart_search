import pandas as pd
import onnxruntime as ort
import numpy as np
import os
import sys
import pandas as pd
from typing import List
import ast
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from textwrap import wrap
import math
import wandb
import re
import json

# Add the parent directory of `evaluation_pipeline` to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constants import EMBEDDING_MODELS_DICT
from src.feature_extractor import FeatureExtractor
from src.llm_judge import llm_as_judge


def convert_to_list(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


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


def calc_ndcg(relevant_docs, retrieved_docs, k=None):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).
    Parameters:
        relevant_docs (set): Set of relevant document IDs.
        retrieved_docs (list): List of retrieved document IDs in ranked order.
        k (int, optional): Number of top results to consider. If None, use all retrieved_docs.
    Returns:
        float: NDCG score.
    """
    def dcg(relevance_scores):
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    retrieved_at_k = retrieved_docs[:k] if k else retrieved_docs
    relevance_scores = [1 if doc in relevant_docs else 0 for doc in retrieved_at_k]
    actual_dcg = dcg(relevance_scores)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg(ideal_relevance_scores)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def calc_mean_reciprocal_rank(relevant_docs, retrieved_docs):
    """
    Compute Mean Reciprocal Rank (MRR).
    Parameters:
        relevant_docs (set): Set of relevant document IDs.
        retrieved_docs (list): List of retrieved document IDs in ranked order.
    Returns:
        float: MRR score.
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1 / rank
    return 0.0


def calc_mean_average_precision(relevant_docs, retrieved_docs):
    """
    Compute Mean Average Precision (MAP).
    Parameters:
        relevant_docs (set): Set of relevant document IDs.
        retrieved_docs (list): List of retrieved document IDs in ranked order.
    Returns:
        float: MAP score.
    """
    relevant_retrieved = 0
    precision_sum = 0.0

    for k, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            relevant_retrieved += 1
            precision_sum += relevant_retrieved / k

    return precision_sum / len(relevant_docs) if relevant_docs else 0.0

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


def run_llm_judge(judge, query_id, query, retrieved_texts, k):
    row = {'query_id': query_id}
    row['query'] = query
    # call llm
    decisions = []
    for retrieved_text in retrieved_texts:
        try:
            llm_judge_response = judge.evaluation_prompt(query, retrieved_text)
            # clean up JSON format from LLM
            response = format_judge_response(llm_judge_response)
            decisions.append(response['binary_decision'])
        except:
            decisions.append(0)
    # store results
    row['decisions'] = decisions
    row[f'on_topic_number@{k}'] = sum(decisions)
    row[f'on_topic_rate@{k}'] = sum(decisions) / float(k)
    return row


def run_traditional_eval(query_id, query, relevant_docs, retrieved_docs, k):
    row = {'query_id': query_id}
    row['query'] = query
    # calcuate traditional IR metrics
    precision = calc_precision_at_k(relevant_docs, retrieved_docs, k)
    recall = calc_recall_at_k(relevant_docs, retrieved_docs, k)
    ndcg = calc_ndcg(relevant_docs, retrieved_docs, k=k)
    mean_reciprocal_rank = calc_mean_reciprocal_rank(relevant_docs, retrieved_docs)
    mean_average_precision = calc_mean_average_precision(relevant_docs, retrieved_docs)
    # store in row
    row['retrieved_ids'] = retrieved_docs
    row['relevant_docs'] = relevant_docs
    row[f'precision@{k}'] = precision
    row[f'recall@{k}'] = recall
    row[f'ndcg@{k}'] = ndcg
    row['mean_reciprocal_rank'] = mean_reciprocal_rank
    row['mean_average_precision'] = mean_average_precision
    return row




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



def load_retrieved(file_path):
    df = pd.read_csv(file_path,
            converters={
                'relevant_docs': convert_to_list,
                'retrieved_ids': convert_to_list,
                'combined_text': convert_to_list,
    })
    k = df['k'].max()
    df['combined_text'] = get_combined_texts_uniform_k(df, k)
    return df, k

def vectorized_evaluation(row, k):
    return run_traditional_eval(
        query_id=row['query_id'],
        query = row['query'],
        relevant_docs=row['relevant_docs'],
        retrieved_docs=row['retrieved_ids'],
        k=k
    )

def vectorized_llm_evaluation(row, k, judge):
    return run_llm_judge(
        judge=judge,
        query_id=row['query_id'],
        query = row['query'],
        retrieved_texts=row['combined_text'],
        k=k
    )

def run_vectorized_traditional_eval(df, k):
    # Apply the function row-wise, passing k as a constant
    df['evaluation'] = df.apply(lambda row: vectorized_evaluation(row, k), axis=1)
    # Return the evaluations as a DataFrame
    return pd.DataFrame(df['evaluation'].tolist())



def run_vectorized_llm_eval(df, k, judge):
    # Apply the function row-wise, passing k as a constant
    df['llm_evaluation'] = df.apply(lambda row: vectorized_llm_evaluation(row, k, judge), axis=1)
    # Return the evaluations as a DataFrame
    return pd.DataFrame(df['llm_evaluation'].tolist())



def eval_pipeline(run_llm_judge: bool, file_path):
    retrieval_df, k = load_retrieved(file_path=file_path)
    results_df = run_vectorized_traditional_eval(retrieval_df, k)
    dfs_to_return = []
    dfs_to_return.append(results_df)
    df_labels = []
    df_labels.append("traditional_eval")
    if run_llm_judge:
        judge = llm_as_judge()
        llm_results_df = run_vectorized_llm_eval(retrieval_df, k, judge)
        dfs_to_return.append(llm_results_df)
        df_labels.append("llm_eval")

    for df, label in zip(dfs_to_return, df_labels):
        df.to_csv(f"{label}.csv")






def visualize_embeddings(model_name, queries):
    fe = FeatureExtractor(EMBEDDING_MODELS_DICT, model_name=model_name)
    model_name_normalized = model_name.replace("/","_").replace("-","_").replace(".","_")
    embeddings = []
    for query in queries:
        query_embeddings = fe.get_embeddings([query])[0]
        embeddings.append(query_embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity = 25)
    reduced_embeddings = tsne.fit_transform(np.array(embeddings))

    # Wrap text to a maximum width
    wrapped_labels = [ "\n".join(wrap(label, width=25)) for label in queries ]  # 15 characters per line


    # Plot with wrapped labels
    fig, ax = plt.subplots(figsize=(10, 7)) 

    plt.scatter([x[0] for x in reduced_embeddings], [x[1] for x in reduced_embeddings], c='orange', alpha=0.6)

    for i, txt in enumerate(wrapped_labels):
        ax.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),fontsize=7,)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f't-SNE Visualization of Embeddings for {model_name}')
    try: 
        wandb.log({"chart": wandb.Image(fig)})
    except: 
        pass 
    plt.savefig(f'figs/t_sne_embeddings_{model_name_normalized}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()



def main(use_llm_judge, file_path):
    eval_pipeline(run_llm_judge = use_llm_judge, file_path=file_path)


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline with specified parameters.")

    parser.add_argument("-f", type=str, help="Path to the file to be processed")

    parser.add_argument(
    "-llm",
    type=lambda x: str(x).lower() in ['true', '1', 'yes'],
    default=False,
    help="Whether or not to use LLM judge (default: False)"
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    main(use_llm_judge=args.llm, file_path=args.f)


