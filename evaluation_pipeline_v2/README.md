# Profile Retrieval Evaluation Pipeline

This folder provides an evaluation pipeline for comparing profile retrieval performance across different embedding models.

Some of the scripts are adapted from the original implementation in [evaluation_pipeline](../evaluation_pipeline).

It automates the full process of:
1. Generating embeddings for all profiles in a Hugging Face dataset ([default dataset](https://huggingface.co/datasets/frankjc2022/semantic-history-search))
2. Running retrieval evaluations on multiple models
3. Computing standard ranking metrics
4. Generating an HTML report to visualize the results

---

## Folder Structure

```graphql
.
├── generate_result.py      # Main entry point
├── retrieval.py            # Embedding generation and retrieval logic
├── evaluation.py           # Evaluation metrics computation
├── build_report.py         # Generates HTML visualization report
└── results/                # Auto-generated outputs (per profile)
```

---

## Overview of Each Script

### generate_result.py
The main driver script, coordinates the entire evaluation workflow.

- Retrieves all profiles from the specified Hugging Face dataset.
- Runs retrieval evaluation for each profile using:
  - `Xenova/all-MiniLM-L6-v2`
  - `minishlab/potion-retrieval-32M` (static embedding)
- Saves results to:
    ```
    results/<profile_name>/evaluation_results/
    ```
- Generates a summary visualization:
    ```
    results/eval_report.html
    ```
> To test additional models, also update the MODELS list in retrieval.py.

---

### retrieval.py
Handles embedding generation and retrieval for each profile.

- Creates embeddings for profile data and saves them as `.pkl` files:
    ```
    results/<profile_name>/embeddings__{model_name_normalized}.pkl
    ```
    If the file already exists, it will load it instead of regenerating.
- Command-line arguments:
    ```
  python retrieval.py \
  --model_name Xenova/all-MiniLM-L6-v2 \
  --k 2 \
  --threshold 10.0 \
  --save_path ./results/<profile_name> \
  --profile "example_profile" \
  --profile_id "12345" \
  --hf_repo "org/dataset"
    ```

| Argument       | Type  | Default                   | Description                                    |
| -------------- | ----- | ------------------------- |------------------------------------------------|
| `--model_name` | str   | `Xenova/all-MiniLM-L6-v2` | Model to use for embedding                     |
| `--k`          | int   | 2                         | Top-K retrieval results                        |
| `--threshold`  | float | 10.0                      | Retrieval threshold (cosine distance) |
| `--save_path`  | str   | `None`                    | Directory to save embeddings/results           |
| `--profile`    | str   | `None`                    | Profile name to process                        |
| `--profile_id` | str   | `None`                    | Specific profile ID (takes precedence)         |
| `--hf_repo`    | str   | `None`                    | Hugging Face dataset repository                |


---

### evaluation.py
Computes all retrieval evaluation metrics.

Implements:
- Precision@k
- Recall@k
- NDCG@k
- Reciprocal Rank
- Average Precision

For detailed explanations of each metric and how they are calculated, see [metrics_overview.md](./metrics_overview.md).

> Note:
The previous version included an on-topic rate metric using an LLM judge (via API or local inference), but this is currently disabled due to cost/speed concerns.

Example usage:
```bash
python evaluation.py -f results/profile_name/result__{model_name_normalized}.csv --save_path ./results/profile_name
```

---

### build_report.py
Aggregates all profile evaluation results and produces an HTML visualization.
- Reads all per-profile CSV results.
- Generates a single report at:
    ```
    results/eval_report.html
    ```
- Makes it easy to explore and compare performance across models and profiles.

---

## Running the Full Pipeline
To run the full evaluation process end-to-end:
```
python generate_result.py
```
This will:
1. Retrieve all profiles from the target Hugging Face dataset
2. Run embedding and retrieval evaluation for each profile
3. Save all metrics
4. Generate a summarized interactive HTML report

---

## Notes & Implementation Details
- When computing similarity directly (without `sqlite-vec`) like in the [static embedding notebook](static_embedding_evaluation.ipynb), results may differ from the pipeline.
- The old version used a custom ONNX feature extractor, which is much slower.
- Direct model loading gives slightly lower scores but is significantly faster.
- Since this serves as a baseline for static embeddings, the trade-off is acceptable.

---

## Output Structure

```
results/
├── <profile_name>/
│   └── evaluation_results/
│       ├── <model_name>_traditional_eval_aggregate_metrics.csv
│       ├── <model_name>_traditional_eval_detailed_metrics.csv
│   └── ...
└── eval_report.html
```