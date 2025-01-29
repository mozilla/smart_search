import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import ast
import numpy as np


def convert_to_mean(value):
    if isinstance(value, str):
        value = value.replace('np.int64(', '').replace(')', '')
        value = value.replace('np.float64(', '').replace(')', '')
        return np.mean(ast.literal_eval(value))
    return value


def load_models(embedding_size_path):
    with open(embedding_size_path, "rb") as f:
        embeddings_sizes= pickle.load(f)
    sizes = []
    model_names = []
    for k, v in embeddings_sizes.items():
        normalized_k = k.replace("/","_").replace("-","_").replace(".","_")
        model_names.append(normalized_k)
        sizes.append(v)

    model_size_dict = dict(zip(model_names, sizes))
    return model_size_dict



def load_results(label, eval_results_dir, metrics, model_size_dict):
    #  List to store DataFrames
    dataframes = []

    # Iterate over files in the directory
    for file in os.listdir(eval_results_dir):
        if label == "llm":
             if file.endswith(".csv") and 'detailed' and 'llm' in file:  # Check if the file is a CSV
                file_path = os.path.join(eval_results_dir, file)
                dataframes.append(pd.read_csv(file_path))
        elif label == "normalized_eval":
            if file.endswith(".csv") and 'detailed' in file and 'norm' in file and 'traditional' in file:  # Check if the file is a CSV
                file_path = os.path.join(eval_results_dir, file)
                dataframes.append(pd.read_csv(file_path,
                converters = {
                        'jaccard_indices': convert_to_mean,
                        'overlap_coefficients': convert_to_mean,
                    }))
        else:
             if file.endswith(".csv") and 'detailed' in file and 'traditional' in file and "norm" not in file:  # Check if the file is a CSV
                file_path = os.path.join(eval_results_dir, file)
                print(file_path)
                dataframes.append(pd.read_csv(file_path,
                converters = {
                        'jaccard_indices': convert_to_mean,
                        'overlap_coefficients': convert_to_mean,
                    }))

    # combined all dfs into one
    all_data = pd.concat(dataframes, ignore_index=True)
    print(all_data)

    # Group by `model_name`
    base_metrics = ['model_name']
    base_metrics.extend(metrics)
    grouped = all_data[base_metrics].groupby("model_name")

    # Create the aggregation dictionary based on metric list entered
    agg_dict = {f"{metric}": "mean" for metric in metrics}

    # Aggregate Metrics
    agg_metrics = grouped.agg(agg_dict).reset_index()

    agg_metrics['model_name_normalized'] = agg_metrics['model_name'].replace("/","_").replace("-","_").replace(".","_")
    agg_metrics['embedding_size'] = agg_metrics['model_name_normalized'].map(model_size_dict).fillna(0)
    print(f"Result Type: {label}\n ------")
    print(agg_metrics)

    directories = [f"{eval_results_dir}/result_analysis",f"{eval_results_dir}/figs"]

    for directory in directories:
        if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory created: {directory}")

    # save as csv


    agg_metrics.to_csv(f'{eval_results_dir}/result_analysis/metric_results_{label}.csv')

    return agg_metrics


def plot_metrics(label, metric, agg_metrics, color_map, eval_results_dir):

    # Sort the DataFrame by the metric in descending order
    sorted_data = agg_metrics.sort_values(by=metric, ascending=True)

    # Map colors to the sorted data
    sorted_colors = sorted_data["embedding_size"].map(color_map)

    # Bar plot
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_data["model_name"], sorted_data[metric], color=sorted_colors)

    # Create a custom legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[size]) for size in color_map]
    labels = [f"{size}" for size in color_map]
    plt.legend(handles, labels, title="Embedding Dim", loc="upper center", bbox_to_anchor=(0.5, -0.2))

    # Customize plot
    plt.title(f"{metric} by Model Name - {label}")
    plt.xlabel(metric)
    plt.ylabel("Model Name")
    plt.tight_layout()
    plt.savefig(f"{eval_results_dir}/figs/{label}_{metric}_by_model_name.png")



def main(llm, k, embedding_sizes_path, eval_results_dir):
    labels = ['traditional_eval','normalized_eval']
    if llm:
        labels.append('llm')
    print(labels)

    # may need to adjust if new embedding dims are added
    color_map = {
        384.: "blue",  # Color for embedding size 348
        768.: "green",  # Color for embedding size 768
        0.: "pink",
    }
    model_size_dict = load_models(embedding_sizes_path)

    # metric sets
    traditional_metrics = [f'precision@{k}', f'recall@{k}', f'ndcg@{k}', 'reciprocal_rank',
               'average_precision','jaccard_indices','overlap_coefficients']
    llm_metrics = [f'on_topic_number@{k}', f'on_topic_rate@{k}']


    for label in labels:
        if label == 'llm':
             # llm judge metrics
             agg_metrics = load_results(label,  eval_results_dir=eval_results_dir, metrics=llm_metrics, model_size_dict=model_size_dict)
             for metric in llm_metrics:
                plot_metrics(label, metric, agg_metrics, color_map, eval_results_dir)
        else:
             # traditional metrics
             agg_metrics = load_results(label, eval_results_dir=eval_results_dir, metrics=traditional_metrics, model_size_dict=model_size_dict)
             for metric in traditional_metrics:
                plot_metrics(label, metric, agg_metrics, color_map, eval_results_dir)



if __name__ == "__main__":
     # Create the argument parser
     parser = argparse.ArgumentParser(description="Compare model results")
      # Add arguments
     parser.add_argument("--eval_results_dir", default="evaluation_results/", type=str, help="Eval results directory")
     parser.add_argument("--embedding_sizes_path", type=str, help="Path to embedding sizes dict.")
     parser.add_argument("--k", type=int, default=2, help="Top-K results used in retrieval being evaluated.")
     parser.add_argument("--llm", type=bool, default=False, help="whether LLM judge reuslts are included")

     args = parser.parse_args()
     # Call the main function with parsed arguments
     main(
         llm=args.llm,
         k=args.k,
         embedding_sizes_path=args.embedding_sizes_path,
         eval_results_dir = args.eval_results_dir
     )

