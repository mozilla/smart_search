# Evaluation Pipeline Overview
This folder contains two scripts that can be used to run the evaluation pipeline for the Semantic Search Your History Level 1 POC.


## Contents
- `retrieval.py`: Performs the retrieval aspect of the pipeline. Need to provide a csv file with browsing history, as well as a set of reference queries and URLs ('golden set')
- `evaluation.py`: Given a results file that contains the query, retrieved docs, relevant docs, calculates traditional information retrieval metrics, as well an option to run an LLM judge (recommended to run on GPU).

# How to run
- In your virtualenv, install requirements from requirements.txt in this subdirectory

## Retrieval
- cd to the root directory
- `python evaluation_pipeline/retrieval.py "output.csv" --golden_path "golden_query_set.csv"`
- Args:
    - history_file_path: Path to csv of browsing history
    - model_name: model to use for retrieval (must be in EMBEDDING_MODELS_DICT under src)
    - k: number of results per retrieval
    - threshold: distance for threshold (default to 100)
    - golden_path: path to golden data set csv (query, url)
    -row_limit: for testing different variants of history size
- Output:
    - CSV file with {model_name}_results.csv

## Evaluation
- cd to root directory
- Must have formatted retrieval results in results/ directory (can be from retrieval script, or manually curated from Firefox logs).
- `python evaluation_pipeline/evaluation.py -f "results/Xenova_all_MiniLM_L6_v2_results.csv"`
- Args:
 - -f path to results file csv
 - -llm whether or not to run LLM judge (recommended to run on GPU only)

 - Output:
    - CSV file with {model_name}_results.csv

# Retrieval Data Input Structure Requirements
- `history_file_path`: CSV file of browsing history exported from `moz_places`
    - url
    - last_visit_date
    - title
    - description

- `golden_query_file_path`: CSV file of (query, url) pairs for the 'true' URL. Used to evaluate retrieval.  If there are multiple relevant URLs, use one row per (query, doc) pair. (query 1, doc1), (query 1, doc2).
    - search_query
    - url


# Evaluation Data Input Structure Requirements
If running evaluation using a manual data set (like from firefox), it must be formatted like:
- query_id (a unique ID for the query, doesn't matter what it is)
for i in k:
- retrieval_i_id
- retrieval_i_title
- retrieval_i_url
- retrieval_i_combined_text
- retrieval_i_distance
- retrieved_ids (a list of retrieved ids in rank order)

- model_name
- query (string version)
- relevant_docs
- k


# Supported models
To add a model to the run, go to src.constants and update EMBEDDING_MODELS_DICT