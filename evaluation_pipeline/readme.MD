# Evaluation Pipeline Overview
This folder contains two scripts that can be used to run the evaluation pipeline for the Semantic Search Your History Level 1 POC.


## Contents
- `retrieval.py`: Performs the retrieval aspect of the pipeline. Need to provide a csv file with browsing history, as well as a set of reference queries and URLs ('golden set')
- `evaluation.py`: Given a results file that contains the query, retrieved docs, relevant docs, calculates traditional information retrieval metrics, as well an option to run an LLM judge (recommended to run on GPU).

# How to run
- In your virtualenv, install requirements

## Retrieval
- cd to the root directory
- `python retrieval.py


# Data Input Structure Requirements
- `history_file_path`: CSV file of browsing history exported from `moz_places`
    - url
    - last_visit_date
    - title
    - description

- `golden_query_file_path`: a csv file of (query, url) pairs for the 'true' URL. Used to evaluate retrieval.  If there are multiple relevant URLs, use one row per (query, doc) pair. (query 1, doc1), (query 1, doc2).
    - search_query
    - url

# Supported models
To add a model to the run, go to src.constants and update EMBEDDING_MODELS_DICT