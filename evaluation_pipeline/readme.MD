# Evaluation Pipeline Overview
This folder contains scripts needed to run the evaluation pipeline for browsing history. The helper scripts are located under:
- `pipeline_utils`  
    - `features.py`
    - `vector_db.py`
    - `retrieval.py`
    - `evaluation.py`
    # Helpers
    - `utils.py`
    - `feature_extractor_optimized.py`
    - `metrics.py`




## To run the pipline
0.  Create virtual env + install `pip install -r requirements.txt` from the evaluation_pipeline directory

1. Update file paths in  `run_config.yml`: Specify the run options in the `run config.yml`.  
- History (used for retrieval)
- Golden Path (queries + URLs for ground truth)

# Optional Settings:
- You can configure the run options, such as models to run, whether or not to re-generate embeddings (or to use ones that are existing)
- Whether or not to run LLM judge (not recommneded unless on GPU)

2.  `python evaluation_pipeline/run_full_pipeline.py`  

# Compare Results Across Models
`python run_model_comparison.py --eval_results_dir <eval_results_dir>`
- eval_results_dir:  Where the retrieval results are stored
- k: value of k used for retrieval
- llm: whether or not LLM judge results are included



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