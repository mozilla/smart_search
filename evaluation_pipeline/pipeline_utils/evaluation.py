import pandas as pd
import numpy as np
import os
from typing import List
import ast
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from textwrap import wrap
import wandb
from dotenv import load_dotenv
from pipeline_utils.feature_extractor_optimized import FeatureExtractor
from pipeline_utils.llm_judge import llm_as_judge
from pipeline_utils.metrics import format_judge_response, run_traditional_eval
from pipeline_utils.utils import convert_to_list, get_combined_texts_uniform_k


def run_llm_judge(judge, query, retrieved_texts, k):
    row = {'query': query}

    # call llm
    decisions = []
    for retrieved_text in retrieved_texts:
        try:
            llm_judge_response = judge.evaluation_prompt(
                query,
                retrieved_text
            )

            response = format_judge_response(
                llm_judge_response
            )

            decisions.append(
                response['binary_decision']
            )
        except:
            decisions.append(0)
    # store results
    row['decisions'] = decisions
    row[f'on_topic_number@{k}'] = sum(decisions)
    row[f'on_topic_rate@{k}'] = sum(decisions) / float(k)
    return row


def load_retrieved(file_path):

    df = pd.read_csv(file_path,
            converters={
                'relevant_docs': convert_to_list,
                'retrieved_ids': convert_to_list,
                'combined_text': convert_to_list,
                'retrieved_distances': convert_to_list,
                'norm_relevant_docs': convert_to_list,
                }
    )

    model_run_details = ast.literal_eval(
        df['model_run_details'].iloc[0]
    )

    k = model_run_details['k']

    df['combined_text'] = get_combined_texts_uniform_k(
        df,
        model_run_details['k']
    )
    return df, k, model_run_details

def vectorized_evaluation(row, k, url_type):
    if url_type == "norm":
        relevant_prefix = "norm_"
    elif url_type == "traditional":
        relevant_prefix = ""
    else:
        ValueError("Only running for norm or traditional")

    evaluated_query = run_traditional_eval(
        query_id = row['query'],
        query = row['query'],
        relevant_docs=row[f'{relevant_prefix}relevant_docs'],
        retrieved_docs=row['retrieved_ids'],
        retrieved_distances = row['retrieved_distances'],
        retrieved_texts = row['combined_text'],
        relevant_texts =  row[f'{relevant_prefix}relevant_combined_text'],
        k=k
    )
    return evaluated_query


def vectorized_llm_evaluation(row, k, judge):
    return run_llm_judge(
        judge=judge,
        query = row['query'],
        retrieved_texts=row['combined_text'],
        k=k
    )

def run_vectorized_traditional_eval(df, k):

    df['evaluation'] = df.apply(
        lambda row: vectorized_evaluation(
            row,
            k,
            url_type='traditional'
            ),
        axis=1
        )

    df['norm_evaluation'] = df.apply(
        lambda row: vectorized_evaluation(
            row,
            k,
            url_type='norm'
            ),
        axis=1
    )

    evaluation_df = pd.DataFrame(
        df['evaluation'].tolist()
    )

    norm_evaluation_df = pd.DataFrame(
        df['norm_evaluation'].tolist()
    )

    return evaluation_df, norm_evaluation_df


def run_vectorized_llm_eval(df, k, judge):

    df['llm_evaluation'] = df.apply(
        lambda row: vectorized_llm_evaluation(
            row,
            k,
            judge)
            ,
        axis=1
    )

    llm_evaluation_df = pd.DataFrame(
        df['llm_evaluation'].tolist()
    )
    return llm_evaluation_df


def wandb_logging(df, k, model_run_details):
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")
    wandb.login(key=api_key)  # Automatically uses the API key
    wandb.init(
    # set the wandb project where this run will be logged
    project=project,
    run_name=f"{model_run_details['model_name']}_{model_run_details['distance']}_{model_run_details['suffix']}_top{k}",

    # track hyperparameters and run metadata
    config={"model_name": model_run_details['model_name'],
            "k": k,
            "distance": model_run_details['distance'],
            "quanitzed": model_run_details['quantized']
    }
)



def evaluation_pipeline(run_llm_judge: bool, file_path, log_to_wandb, save_eval_dir, result_dir, specific_k=None):
    print(f"Evaluating {file_path}")

    retrieval_df, k, model_run_details = load_retrieved(file_path=file_path)

    # give option to override k
    if specific_k is None:
        pass
    else:
        k = specific_k

    if log_to_wandb:
        wandb_logging(retrieval_df, k)

    # run eval
    results_df, norm_results_df = run_vectorized_traditional_eval(
        retrieval_df,
        k
    )

    # store results
    dfs_to_return = []
    df_labels = []

    # original urls
    dfs_to_return.append(results_df)
    df_labels.append("traditional_eval")

    #normalized urls
    dfs_to_return.append(norm_results_df)
    df_labels.append("norm_traditional_eval")

    if run_llm_judge:
        judge = llm_as_judge()
        llm_results_df = run_vectorized_llm_eval(retrieval_df, k, judge)
        dfs_to_return.append(llm_results_df)
        df_labels.append("llm_eval")

    for df, label in zip(dfs_to_return, df_labels):
        df['model_name'] = model_run_details['model_name']
        df['embedding_dim'] = model_run_details['embedding_dim']
        df['model_run_details'] =  [model_run_details] * len(df)

        summary_df = df.describe().reset_index()
        summary_df = summary_df.loc[:, summary_df.columns != 'query_id'] if 'query_id' in summary_df.columns else summary_df
        summary_table = wandb.Table(dataframe=summary_df)
        averages = summary_df[summary_df['index'] == 'mean']

        # Ensure the directory exits
        save_eval_results_dir = result_dir.replace('results/',save_eval_dir)
        if not os.path.exists(save_eval_results_dir):
            os.makedirs(save_eval_results_dir)
            print(f"Directory created: {save_eval_results_dir}")

        if log_to_wandb:
            wandb.log({label: summary_table})
            wandb.log({"Averages": averages})
            wandb.finish()
        # write files
        write_file_path = file_path.replace('_results.csv','').replace('.csv','').replace('non_ml_results/','').replace('results/',save_eval_dir)

        summary_df.to_csv(f"{write_file_path}_{label}_aggregate_metrics.csv")
        df.to_csv(f"{write_file_path}_{label}_detailed_metrics.csv")
        
    return write_file_path


def visualize_embeddings(model_run_details, queries, file_name):
    model_name = model_run_details['model_name']
    quantized_model = model_run_details['quantized']
    fe = FeatureExtractor(model_name=model_name, quantized=quantized_model)
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
    plt.savefig(f'figs/t_sne_embeddings_{file_name}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()



def main(use_llm_judge, file_path, log_to_wandb, save_eval_dir):
    evaluation_pipeline(run_llm_judge = use_llm_judge, file_path=file_path, log_to_wandb=log_to_wandb, save_eval_dir=save_eval_dir)


if __name__ == "__main__":
      # Create the argument parser
      parser = argparse.ArgumentParser(description="Run the evaluation pipeline with specified parameters.")

      parser.add_argument("-f", type=str, help="Path to the file to be processed")
      parser.add_argument("-log_to_wandb", type=str, default=False, help="Whether to log to Weights & Biases")

      parser.add_argument("-llm",type=lambda x: str(x).lower() in ['true', '1', 'yes'],default=False,help="Whether or noto use LLM judge (default: False")

      parser.add_argument("--save_eval_dir", type=str, default="evaluation_results/",help="directory to save eval results")

      # Parse the command-line arguments
      args = parser.parse_args()
      main(use_llm_judge=args.llm, file_path=args.f, log_to_wandb=args.log_to_wandb, save_eval_dir=args.save_eval_dir)



