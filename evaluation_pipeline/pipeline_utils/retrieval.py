import time
import pandas as pd
import os
import gc
from collections import defaultdict
import yaml
from pipeline_utils.feature_extractor_optimized import FeatureExtractor
from pipeline_utils.utils import log_performance,  serialize_f32, clean_text, convert_dict_to_df, build_model_run_details_dict

with open("evaluation_pipeline/model_config.yml", "r") as f:
    config = yaml.safe_load(f)


# retrieval
@log_performance
def query_and_result(fe, prefix, query, db, model_run_details):
    if model_run_details['preprocess']:
        query_to_embed = clean_text(query)
    else:
        query_to_embed = query

    query_to_embed =  prefix + query_to_embed

    distance_measure = model_run_details['distance_measure']

    query_embedding = fe.get_embedding(query_to_embed)

    embedding_dim = len(query_embedding)

    binary_quantization = model_run_details['binary_quantization']
    if binary_quantization:
        coarse_filter = model_run_details['binary_quantization_coarse_filter']
        print(coarse_filter)

        # this is not working yet 
        rows = db.execute(
            f'''
            with coarse_matches as (
            SELECT a.rowid,
            a.embedding,
            b.title,
            b.description,
            b.url,
            b.norm_url,
            b.combined_text

            FROM vec_items_{model_run_details['model_name_normalized']} a
    
            inner join search_data b
            on a.rowid = b.rowid
    
            WHERE
            embedding_coarse match vec_quantize_binary(?)
            and b.url not like '%google.com/search?%'
    
            ORDER BY DISTANCE
            LIMIT {coarse_filter}
        ) coarse_matches
    
        SELECT rowid,
            title,
            description,
            url,
            norm_url,
            combined_text,
            vec_distance_{distance_measure}(embedding, ?) AS {distance_measure}_distance
    
        FROM coarse_matches
    
        ORDER BY {distance_measure}_distance

        LIMIT {model_run_details['k']}

            ''',
         [serialize_f32(query_embedding), serialize_f32(query_embedding)],
         ).fetchall()

    else:

         # NO BINARY QUANTIZATIOn
         rows = db.execute(
         f"""
          SELECT
             a.rowid,
             b.title,
             b.description,
             b.url,
             b.norm_url,
             b.combined_text,
             vec_distance_{distance_measure}(embedding, ?) AS {distance_measure}_distance

           FROM vec_items_{model_run_details['model_name_normalized']} a
        
           inner join search_data b
           on a.rowid = b.rowid
        
           where b.url not like '%google.com/search?%'
           ORDER BY {distance_measure}_distance
           LIMIT {model_run_details['k']}
         """,
         [serialize_f32(query_embedding)],
         ).fetchall()


    results = []

    for row in rows:
        results.append({
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "url": row[3],
            "norm_url": row[4],
            "combined_text": row[5],
            "distance": row[6],
        })

    return results, embedding_dim

def get_relevant_doc_ids(db, url_type, input_dir):
    # join to search query to get doc ID for ground truth URL
    results = db.execute(
        f'''SELECT a.query,
          b.{url_type},
          b.rowid,
          b.combined_text

        FROM ground_truth a

        left join search_data b
        on a.{url_type} = b.{url_type}
    '''
    ).fetchall()

    # Each query_id maps to a dict of lists
    ground_truth_dict = defaultdict(lambda: {
        "doc_ids": [],
        f"{url_type}s": [],
        "combined_texts": []
    })

    for query, url, id_, combined_text in results:
        ground_truth_dict[query]["doc_ids"].append(id_)
        ground_truth_dict[query][f"{url_type}s"].append(url)
        ground_truth_dict[query]["combined_texts"].append(combined_text)


    ground_truth_df = pd.DataFrame.from_dict(ground_truth_dict).T
    ground_truth_df.to_csv(f'{input_dir}ground_truth_with_metadata_{url_type}.csv')

    return ground_truth_dict


@log_performance
def run_retrieval(fe, prefix, queries, db, model_run_details):

    print("running retreival")
    retreival_dict = {}

    for query in queries:
        results, embedding_dim = query_and_result(
            fe,
            prefix,
            query,
            db=db,
            model_run_details=model_run_details
        )

        retreival_dict[query] = results

    return retreival_dict, embedding_dim



def retrieval_pipeline(model_run_details, queries, db, save_result_dir, input_dir):
    process_start = time.time()


    fe = FeatureExtractor(
        model_name=model_run_details['model_name'],
        quantized=model_run_details['quantized'],
        pooling=model_run_details['pooling'],
    )

    # run retrieval
    models_parameters = config["models"][model_run_details['model_name']]
    prefix = models_parameters['prefix_query']
    print(f"Using prefix: {prefix}")

    retrieval_dict, embedding_dim = run_retrieval(
        fe,
        prefix,
        queries,
        db,
        model_run_details
    )

    ground_truth_dict = get_relevant_doc_ids(
        db,
        url_type='url',
        input_dir=input_dir,
    )

    norm_ground_truth_dict = get_relevant_doc_ids(
        db,
        url_type='norm_url',
         input_dir=input_dir,
    )

    # reshape to capture metadata
    retreival_df = convert_dict_to_df(
                            retrieval_dict,
                            ground_truth_dict,
                            norm_ground_truth_dict,
                            model_run_details
                            )

    model_run_details['embedding_dim'] = embedding_dim

    retreival_df['model_run_details'] = [model_run_details] * len(retreival_df)

    # save to csv
    retrieval_save_path = f"{save_result_dir}{model_run_details['retrieval_file_name']}_results.csv"

    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
        print(f"Directory created: {save_result_dir}")

    retreival_df.to_csv(
        retrieval_save_path,
        index=False
    )

    process_end = time.time()

    print(f"Process took {process_end - process_start} seconds")

    # memory cleanup
    del fe
    del db
    del retreival_df
    gc.collect()
    return retrieval_save_path


def main(model_run_details, queries, db, save_result_dir):
    retrieval_pipeline(model_run_details, queries, db, save_result_dir)


if __name__ == "__main__":
    main()
