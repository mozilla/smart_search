import pickle
import pandas as pd
import numpy as np
import gc
import yaml
import sys
import os
from pipeline_utils.utils import clean_text, serialize_f32, normalize_url, log_performance, preprocess_url
from pipeline_utils.feature_extractor_optimized import FeatureExtractor
import time
from pathlib import Path


with open("evaluation_pipeline/model_config.yml", "r") as f:
    config = yaml.safe_load(f)


@log_performance
def process_history(row_limit, preprocess,  history_file_path, features):
    browsing_history = pd.read_csv(history_file_path).head(row_limit)

    browsing_history['norm_url'] = browsing_history['url'].apply(normalize_url)

    browsing_history["combined_text"] = browsing_history.apply(
        lambda row: create_embedding_string(
            row,
            features,
            preprocess=preprocess),
        axis=1)

    feature_string = '_'.join(features)

    processed_history_file_path = f"../data/history_{preprocess*'preprocess_'}{row_limit}_{feature_string}.csv"

    browsing_history.to_csv(processed_history_file_path)

    print(f"Browsing history is {len(browsing_history)}")

    return browsing_history, processed_history_file_path

def create_embedding_string(row, features, preprocess):
    """
    Builds a combined embedding string from a set of features in a row.

    - If preprocess=True, special logic is applied for 'title', 'description', and 'url':
        - 'title' and 'description' are passed through clean_text().
        - 'url' is passed through preprocess_url() and appended as `source: domain`.
    - If preprocess=False, the raw feature values are simply converted to strings and joined.
    - Any other feature is just included as is (converted to string).
    
    Parameters
    ----------
    row : pd.Series or dict-like
        A single row of the dataframe.
    features : list of str
        Column names to include in the combined string.
    preprocess : bool, optional
        Whether to clean text fields and preprocess URL. Defaults to False.

    Returns
    -------
    str
        A single string combining all requested features.
    """
    processed_parts = []

    for col in features:
        value = row[col] if col in row else ""

        if pd.isna(value):
            value = ""

        if preprocess:
            # Special handling for url
            if col == "url":
                domain = preprocess_url(value)  # e.g. "example.com"
                processed_parts.append({domain})
            else:
                processed_parts.append(clean_text(value))
        else:
            processed_parts.append(str(value))

    # Join the parts with " "
    return " ".join(processed_parts)


def generate_embeddings(fe, texts):
    """
    Generates embeddings in a memory-efficient way using a generator.

    Args:
        fe: FeatureExtractor instance
        texts (list or str): List of text strings or a single string.

    Yields:
        np.ndarray: Embedding for each text.
    """
    for text in texts:
        embedding = fe.get_embedding(text)  # Get embedding for a single text
        yield embedding  # Yield instead of storing everything in memory

@log_performance
def create_embeddings(fe,  history, model_run_details):
    texts = history['combined_text'].values.tolist()

    models_parameters = config["models"][model_run_details['model_name']]
    print(f"Using prefix: {models_parameters['prefix_document']}")

    texts_to_embed = [models_parameters['prefix_document'] + text for text in texts]

    embeddings = list(generate_embeddings(fe, texts_to_embed))


    # Save only the essential information
    embeddings_dict = {model_run_details['model_name'] : embeddings}

    embeddings_path = f"../data/embeddings_dict_{model_run_details['embedding_file_name']}.pkl"

    with open(embeddings_path, "wb") as f:
        pickle.dump(
            embeddings_dict,
            f
        )
    print(f"Embeddings written to {embeddings_path}")

    return embeddings_path



def feature_pipeline(history_file_path, model_run_details, embeddings_exist):
    print(f" {'Using existing embeddings' if embeddings_exist else 'generating embeddings'}")

    directory = Path("../data")
    directory.mkdir(parents=True, exist_ok=True)  # Create if not exists

    history, history_path = process_history(
        model_run_details['row_limit'],
        model_run_details['preprocess'],
        history_file_path,
        model_run_details['features']
    )




    if embeddings_exist:
        print(f"Going to use pre-generated embeddings from {model_run_details['embedding_file_name']}")
        embeddings_path = f"../data/embeddings_dict_{model_run_details['embedding_file_name']}.pkl"

    else:
        embedding_start = time.time()

        fe = FeatureExtractor(
            model_run_details['model_name'],
            model_run_details['quantized'],
            model_run_details['pooling'],
        )

        embeddings_path = create_embeddings(
            fe,
            history,
            model_run_details
        )
        embedding_end = time.time()
        print(f"Embeddings to {embedding_end - embedding_start} to generate")
        print(f"Embeddings written to: {embeddings_path}")
    
        del fe
        gc.collect()

    return embeddings_path, history_path




def main(history_file_path, model_run_details):
    feature_pipeline(history_file_path, model_run_details)


if __name__ == "__main__":
    main()

