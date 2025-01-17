#!/bin/bash

# Directory containing the files
directory="results"

# Loop through all CSV files in the directory
for file in "$directory"/*.csv
do
    echo "Processing file: $file"
    python evaluation_pipeline/evaluation.py -f "$file"
done
