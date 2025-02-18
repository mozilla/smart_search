import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import ast
import numpy as np
import scipy.stats as stats
#import scikit_posthocs as sp
import plotly.express as px
import plotly.graph_objects as go

def convert_to_mean(value):
    if isinstance(value, str):
        value = value.replace('np.int64(', '').replace(')', '')
        value = value.replace('np.float64(', '').replace(')', '')
        return np.mean(ast.literal_eval(value))
    return value


def extract_variants(row):
     model_runs = ast.literal_eval(row['model_run_details'])  # Convert string to dictionary
     return pd.Series({
         'quantized': model_runs.get('quantized', None),
         'distance': model_runs.get('distance_measure', None),
         'pooling': model_runs.get('pooling', None),
         'feature_string': model_runs.get('feature_string', None),
         'preprocess': model_runs.get('preprocess', None),
         'k': model_runs.get('k', None),
     })

def load_results(label, eval_results_dir, metrics, k):

    #  List to store DataFrames
    dataframes = []

    # Iterate over files in the directory
    for file in os.listdir(eval_results_dir):


        if label == "llm":

             if file.endswith(".csv") and 'detailed' and 'llm' in file and f'top{k}' in file:  # Check if the file is a CSV
                try:
                    file_path = os.path.join(eval_results_dir, file)
                    df = pd.read_csv(file_path)
                    df['file_path'] = file_path
                    dataframes.append(df)
                except:
                    pass
        elif label == "normalized_eval":

            if file.endswith(".csv") and 'detailed' in file and 'norm' in file and 'traditional' in file and  f'top{k}' in file:  # Check if the file is a CSV
                try:
                     file_path = os.path.join(eval_results_dir, file)
                     df = pd.read_csv(file_path)
                     df['file_path'] = file_path
                     dataframes.append(df)
                except:
                    pass
        else:

             if file.endswith(".csv") and 'detailed' in file and 'traditional' in file and f'top{k}' in file and "norm" not in file:  # Check if the file is a CSV
                try:
                     file_path = os.path.join(eval_results_dir, file)
                     
                     df = pd.read_csv(file_path)
                     df['file_path'] = file_path
                     dataframes.append(df)
                except:
                    pass


    # combined all dfs into one
    all_data = pd.concat(dataframes, ignore_index=True)
    model_names = list(all_data['model_name'].unique())

    file_paths = list(all_data['file_path'].unique())

    embedding_sizes = [
        all_data[all_data['model_name'] == model]['embedding_dim'].iloc[0] if not all_data[all_data['model_name'] == model].empty else 0
        for model in model_names
    ]

    variants = [all_data[all_data['file_path'] == file]['model_run_details'].iloc[0] for file in file_paths]

    variant_dict = dict(zip(file_paths, variants))
    model_size_dict = dict(zip(model_names, embedding_sizes))

    # Group by `model_name`
    base_metrics = ['model_name', "file_path"]
    base_metrics.extend(metrics)
    grouped = all_data[base_metrics].groupby(["file_path",'model_name'])

    # Create the aggregation dictionary based on metric list entered
    agg_dict = {f"{metric}": "mean" for metric in metrics}

    # Aggregate Metrics
    agg_metrics = grouped.agg(agg_dict).reset_index()
    agg_metrics['model_name_normalized'] = agg_metrics['model_name'].replace("/","_").replace("-","_").replace(".","_")
    agg_metrics['embedding_size'] = agg_metrics['model_name'].map(model_size_dict).fillna(0)
    agg_metrics['model_run_details'] = agg_metrics['file_path'].map(variant_dict).fillna(0)


    # Apply the function to all rows
    agg_metrics = agg_metrics.join(agg_metrics.apply(extract_variants, axis=1))

    directories = [f"{eval_results_dir}/result_analysis",f"{eval_results_dir}/figs"]

    for directory in directories:
        if not os.path.exists(directory):
                os.makedirs(directory)
                

    # save as csv
    agg_metrics.to_csv(f'{eval_results_dir}/result_analysis/metric_results_{label}_{k}.csv')

    return agg_metrics, all_data



def precision_recall_tradeoff(label,k, agg_metrics, eval_results_dir):

    agg_metrics[f"recall@{k}_jitter"] = agg_metrics[f"recall@{k}"] + np.random.uniform(-0.003, 0.003, agg_metrics.shape[0])

    agg_metrics[f"precision@{k}_jitter"] = agg_metrics[f"precision@{k}"] + np.random.uniform(-0.003, 0.003, agg_metrics.shape[0])

    agg_metrics["ml_or_baseline"] = agg_metrics.apply(
        lambda row: "ml" if (row["embedding_size"] > 0 or "KG" in row["model_name"]) else "baseline",
        axis=1
    )


    fig = px.scatter(
        agg_metrics,
        x=f"recall@{k}_jitter",
        y=f"precision@{k}_jitter",
        color="model_name",  # Assign color based on Model
        symbol="ml_or_baseline",
        hover_data={
            "model_name": True,
            "distance": True,
            "pooling": True,
            "feature_string": True,
            "preprocess": True,
            "k": True,
            "quantized":True,
            f"recall@{k}": True,
            f"precision@{k}": True,
            f"precision@{k}_jitter": False,
            f"recall@{k}_jitter": False,
            "embedding_size": True,


            },  # Tooltip details
        title=f"Precision vs. Recall (Interactive Model Variants) at {k} {label}"

    )

    # Improve layout
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        hovermode="closest",

    )
    fig.update_traces(marker=dict(size=11, opacity=.7))  # Adjust the size as needed

    fig.write_html(f"{eval_results_dir}/figs/{label}_precision_recall_{k}_interactive.html")
    fig.show()




def plot_text_length_vs_metric(all_data, metric, label, k, eval_results_dir,  length_type="words"):
    """
    Plots an interactive scatter plot of text length (words or characters) vs. a metric.
    
    Args:
        texts (list of str): List of text strings.
        metric_values (list of float): Corresponding metric values.
        metric_name (str): Name of the metric for labeling.
        length_type (str): "words" for word count, "characters" for character count.
    """
    texts = all_data['query'].values.tolist()

    if length_type == "words":
        lengths = [len(text.split()) for text in texts]
        x_label = "Word Count"
    elif length_type == "characters":
        lengths = [len(text) for text in texts]
        x_label = "Character Count"
    else:
        raise ValueError("length_type must be 'words' or 'characters'")


    all_data['length'] = lengths
    all_data['length_type'] = x_label


    fig = px.scatter(
        all_data,
        x="length",
        y=metric,
        color="model_name",
        hover_data={"query": True, "length": True, "length_type":True, metric: True, "model_name": True},
        title=f"{x_label} vs {metric}"
    )

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=metric,
        hovermode="closest"
    )
    fig.write_html(f"{eval_results_dir}/figs/{length_type}_{metric}_{label}_{k}_interactive.html")
    fig.show()



def plot_bar(label, metric, k, agg_metrics, eval_results_dir):
    agg_metrics = agg_metrics.sort_values(by=metric, ascending=False)

    agg_metrics["x_numeric"] = agg_metrics["model_name"].astype("category").cat.codes
    agg_metrics["ml_or_baseline"] = agg_metrics["embedding_size"].apply(lambda x: "ml" if x > 0 else "baseline")

    # Create bar chart for ML models
    fig = px.bar(
        agg_metrics,
        x="x_numeric",
        y=metric,
        color="model_name",  # Groups bars by category
        text="model_name",  # Display model names inside bars
        title=f"{metric}: {label} Evaluation",
    )


    # Apply Tufte-style layout
    fig.update_layout(
        plot_bgcolor="white",  # White background for clarity
        xaxis=dict(
            showgrid=False,  # Remove gridlines
            zeroline=False,  # Remove zero line
            showline=True,  # Keep axis line
            linecolor="black",  # Make axis line visible
            mirror=True,  # Mirror axes for a boxed look
            tickmode="array",
            tickvals=agg_metrics["x_numeric"],
            ticktext=agg_metrics["model_name"],
            title="Model",
        ),
        yaxis=dict(
            showgrid=True,  # Light gridlines to aid readability
            gridcolor="lightgray",  # Subtle gridlines
            zeroline=False,  # Remove zero line
            showline=True,  # Keep axis line
            linecolor="black",  # Make axis line visible
            mirror=True,  # Mirror axes
            tickmode="auto",
            ticks="outside",
            tickcolor="black",
        ),
        yaxis_title=metric,
        hovermode="x unified",  # Show all values at the same x position
        legend=dict(
            bordercolor="black",
            borderwidth=0.5,
            font=dict(size=12),
            x=1.02,  # Moves legend outside plot
            y=1,
            xanchor="left",
            yanchor="top",
        ),
        font=dict(
            family="Arial",
            size=14,
            color="black",
        ),

        margin=dict(l=40, r=120, t=40, b=40),  # Extra right margin for legend,
        showlegend=False  # Hide legend for minimalism
    )


    # Show figure
    fig.show()
    fig.write_html(f"{eval_results_dir}/figs/{label}_{metric}_interactive_bar.html")

def plot_interactive_metrics(label, metric, k, agg_metrics, eval_results_dir):
    """ Generates an interactive scatter plot with hover details """

     # Ensure the model_name is treated as a categorical variable and preserve order
    agg_metrics["model_name"] = agg_metrics["model_name"].astype("category")
    agg_metrics["x_numeric"] = agg_metrics["model_name"].cat.codes  # Assign categorical indices
    model_order = agg_metrics["model_name"].cat.categories  # Get ordered labels

    np.random.seed(42)
    agg_metrics["x_jittered"] = agg_metrics["x_numeric"] + np.random.uniform(-0.1, 0.1, size=len(agg_metrics))  # Small jitter
    agg_metrics["ml_or_baseline"] = agg_metrics["embedding_size"].apply(lambda x: "ml" if x > 0 else "baseline")

    fig = px.scatter(
        agg_metrics,
        x="x_jittered",
        y=metric,
        color="model_name",
        symbol="ml_or_baseline",
        hover_data={
            metric: True,
            "model_name": True,
            "distance": True,
            "pooling": True,
            "feature_string": True,
            "preprocess": True,
            "k": True,
            "quantized":True,
            "x_jittered":False,
            "embedding_size": True,
        },

        title=f"Model Variants Performance at {k} - {metric} - {label}"
    )


    fig.update_layout(
        plot_bgcolor="white",  # White background for clarity
        xaxis=dict(
            showgrid=False,  # Remove gridlines
            zeroline=False,  # Remove zero line
            showline=True,  # Keep axis line
            linecolor="black",  # Make axis line visible
            mirror=True,  # Mirror axes for a boxed look
         #   tickmode="auto",
            ticks="outside",
            tickcolor="black",
            tickmode="array",
            tickvals=agg_metrics["x_numeric"].unique(),
            ticktext=model_order,
            title="Model",
        ),
        yaxis=dict(
            showgrid=False,  # Remove gridlines
            zeroline=False,  # Remove zero line
            showline=True,  # Keep axis line
            linecolor="black",  # Make axis line visible
            mirror=True,  # Mirror axes
            tickmode="auto",
            ticks="outside",
            tickcolor="black",
        ),
        yaxis_title=metric,
        hovermode="closest",
        legend=dict(
            bordercolor="black",
            borderwidth=0.5,
            font=dict(size=12),
            x=1.7,
            y=1,
            xanchor="right",
            yanchor="top",
        ),
        font=dict(
            family="Arial",  # Simple, readable font
            size=14,
            color="black",
        ),
        margin=dict(l=40, r=150, t=40, b=80),
        showlegend=False  # Hide legend for minimalism
    )


    fig.update_traces(marker=dict(size=11, opacity=0.7))  # Adjust the size as needed


    fig.write_html(f"{eval_results_dir}/figs/{label}_{metric}_interactive.html")
    fig.show()


def plot_metrics(label, metric, k, agg_metrics, color_map, eval_results_dir):

    if label == "jaccard_index" or label == "overlap_coefficient":
        pass
    else: pass
    #    stat_sig_test(df, metric)


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
    # Increase the marker size

    # Customize plot
    plt.title(f"{metric} by Model Name at {k}- {label}")
    plt.xlabel(metric)
    plt.ylabel("Model Name")
    plt.tight_layout()
    plt.savefig(f"{eval_results_dir}/figs/{label}_{metric}_by_model_name.png")



def main(llm, k,  eval_results_dir):

    labels = ['traditional_eval','normalized_eval']

    if llm:
        labels.append('llm')

    color_map = {
        384.: "blue",  # Color for embedding size 348
        768.: "green",  # Color for embedding size 768
        0.: "pink",
    }

    # metric sets
    traditional_metrics = [f'precision@{k}', f'recall@{k}']
    llm_metrics = [f'on_topic_rate@{k}']


    for label in labels:
        if label == 'llm':
             # llm judge metrics
             agg_metrics, all_data =  load_results(label,  eval_results_dir=eval_results_dir, metrics=llm_metrics, k=k)
             for metric in llm_metrics:
                plot_bar(label, metric, k, agg_metrics, eval_results_dir)
        else:
             # traditional metrics
             agg_metrics, all_data =  load_results(label, eval_results_dir=eval_results_dir, metrics=traditional_metrics, k=k)
             for metric in traditional_metrics:
                 plot_bar(label, metric, k, agg_metrics, eval_results_dir)



if __name__ == "__main__":
      # Create the argument parser
      parser = argparse.ArgumentParser(description="Compare model results")
       # Add arguments
      parser.add_argument("--eval_results_dir", default="evaluation_results/", type=str, help="Eval results directory")
      parser.add_argument("--k", type=int, default=2, help="Top-K results used in retrieval being evaluated.")
      parser.add_argument("--llm", type=bool, default=False, help="whether LLM judge reuslts are included")

      args = parser.parse_args()
      # Call the main function with parsed arguments
      main(
          llm=args.llm,
          k=args.k,
          eval_results_dir = args.eval_results_dir
      )

