import os
import glob
from retrieval import get_normalized_name

model_names = ["Xenova/all-MiniLM-L6-v2"]
temporal_reorder = False

input_filename = "data/software_engineer_mul/history.csv"
query_filename = "data/software_engineer_mul/query.csv"
save_path = "results/software_engineer_mul"

if temporal_reorder:
    save_path += "_temporal"

cwd = os.getcwd()

for model_name in model_names:

    cmd = f"""
    python retrieval.py {os.path.join(cwd, input_filename)} \
        --golden_path {os.path.join(cwd, query_filename)} \
        --row_limit 10000 \
        --model_name "{model_name}" \
        --save_path {save_path} \
        --k 200 
    """

    if temporal_reorder:
        cmd += " --temporal_reorder "

    os.system(cmd)

    result_filename = f"{save_path}/result__{get_normalized_name(model_name)}.csv"
    cmd = f"""python evaluation.py -f {os.path.join(cwd, result_filename)} --save_path {save_path}"""
    os.system(cmd)


result_filenames = glob.glob(f"{save_path}/result__*.csv")

for result_filename in result_filenames:
    cmd = f"""python evaluation.py -f {os.path.join(cwd, result_filename)} --save_path {save_path}"""
    os.system(cmd)


############
# - first time run, will generate embeddings_dict_10001.pkl and embeddings_sizes_10001.pkl, which create tables for each model in the EMBEDDING_DICT
# - if want to add new model, or create for another set of data, need to remove them and recreate