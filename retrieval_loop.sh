#!/bin/bash

# List of model names to iterate through
model_names=("Xenova/all-MiniLM-L6-v2"
"nomic-ai/nomic-embed-text-v1.5"
"Xenova/all-mpnet-base-v2"
"Xenova/paraphrase-mpnet-base-v2"
"Xenova/all-MiniLM-L12-v2"
"nomic-ai/modernbert-embed-base"
"Xenova/LaBSE"
"Xenova/e5-base-v2")

# Loop through each model name and run the command
for model_name in "${model_names[@]}"
do
    echo "Running retrieval with model: $model_name"
    python evaluation_pipeline/retrieval.py "output.csv" \
        --golden_path "golden_query_set.csv" \
        --row_limit 10001 \
        --model_name "$model_name"
done
