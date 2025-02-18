import sys
import os
from pipeline_utils.utils import build_model_run_details_dict
from pipeline_utils.features import feature_pipeline
from pipeline_utils.vector_db import vector_db_pipeline
from pipeline_utils.retrieval import retrieval_pipeline
from pipeline_utils.non_ml_comparison import non_ml_baseline_pipeline
from pipeline_utils.evaluation import evaluation_pipeline
import yaml
import gc
import multiprocessing
from pipeline_utils.utils import log_performance

@log_performance
def full_pipeline(config_path, model_name):

    print(f"Running model {model_name}:  PID: {os.getpid()}")

    # Load the config file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Inputs
    history_file_path = config["input"]["history_file_path"]
    golden_df_file_path = config["input"]["golden_df_file_path"]
    row_limit = config["input"]["history_row_limit"]
    input_dir = config["input"]["input_dir"]

    # Output dirs
    save_result_dir = config["output"]["save_result_dir"]
    save_eval_dir = config["output"]["save_eval_dir"]

    # Model Settings
    pooling = config["model"]["pooling"]
    quantized_model = config["model"]["quantized_model"]
    preprocess = config["model"]["preprocess_input"]
    embeddings_exist = config["model"]["embeddings_already_exist"]
    
    # Retrieval
    distance_measure = config["retrieval"]["distance_measure"]
    k = config["retrieval"]["k"]
    features = config["retrieval"]["features"]
    include_non_ml_baseline = config['retrieval']['include_non_ml_baseline']
    binary_quantization = config['retrieval']['binary_quantization']

    # build run metadata in dict format
    model_run_details = build_model_run_details_dict(
         history_file_path,
         model_name,
         features,
         quantized_model,
         distance_measure,
         pooling,
         preprocess,
         row_limit,
         k,
         binary_quantization)

    # features
    print("features")
    embeddings_path, browsing_history_file_path = feature_pipeline(
        history_file_path=history_file_path,
        model_run_details=model_run_details,
        embeddings_exist=embeddings_exist
    )

    # vector db
    print("vector db")
    db, queries, processed_golden_df_path  = vector_db_pipeline(
        browsing_history_path=browsing_history_file_path,
        golden_df_file_path=golden_df_file_path,
        model_run_details=model_run_details
    )
    
    # retrieval
    print("retrieval")
    retrieval_save_path = retrieval_pipeline(
        model_run_details=model_run_details,
        queries=queries,
        db=db,
        save_result_dir=save_result_dir,
        input_dir=input_dir
    )

    # non ML baseline (if specified)
    print("non ML")
    if include_non_ml_baseline:
        saved_files = non_ml_baseline_pipeline(
            history_file_path=history_file_path,
            save_dir=save_result_dir,
            query_file_path=golden_df_file_path,
            model_run_details=model_run_details
        )

        for non_ml_file in saved_files:
            evaluation_pipeline(
                run_llm_judge=False,
                file_path=non_ml_file,
                log_to_wandb=False,
                save_eval_dir=save_eval_dir,
                result_dir=save_result_dir
            )


    # evaluation
    print("evaluation")
    write_file_path = evaluation_pipeline(
        run_llm_judge=False,
        file_path=retrieval_save_path,
        log_to_wandb=False,
        save_eval_dir=save_eval_dir,
        result_dir=save_result_dir
    )

    del model_run_details, embeddings_path, browsing_history_file_path
    del db, queries, processed_golden_df_path, retrieval_save_path
    gc.collect()  # Force garbage collection

    return write_file_path


def run_model(config, model_name):
    """Wrapper function to run full_pipeline in a separate process"""
    try:
        full_pipeline(config, model_name)
    except Exception as e:
        print(f"Error running model {model_name}: {e}")

def main(config_path):
    """Loops through models and runs full_pipeline for each in a separate process"""

    # Load the config file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    processes = []

    for model_name in config["model_names"]:
        print(f"Starting model run: {model_name}")

        # Create a new process for each model
        p = multiprocessing.Process(target=run_model, args=(config_path, model_name))
        p.start()
        processes.append(p)

        # Wait for process to finish before starting the next one (sequential execution)
        p.join()

    print("All models completed.")


if __name__ == "__main__":
    main("evaluation_pipeline/run_config.yml")