import pickle
from pipeline_utils.utils import log_performance, normalize_url, serialize_f32
import sqlite3
import sqlite_vec
import pandas as pd

@log_performance
def create_db():
    # vector database (in memory)
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    sqlite_version, vec_version = db.execute(
        "select sqlite_version(), vec_version()"
    ).fetchone()
    print(f"sqlite_version={sqlite_version}, vec_version={vec_version}")
    return db




def load_history_in_db(db, browsing_history_path):
    browsing_history = pd.read_csv(browsing_history_path)

    db.execute('''CREATE TABLE IF NOT EXISTS search_data (
        url TEXT,
        title TEXT,
        description TEXT,
        combined_text TEXT,
        norm_url TEXT
        )
        ''')
    for idx, row in browsing_history.iterrows():
            db.execute("""
                INSERT INTO search_data (rowid, url, title, description, combined_text, norm_url)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (idx, row['url'], row['title'], row['description'],  row['combined_text'], row['norm_url'])
            )
    return db



@log_performance
def read_and_insert_embeddings_into_db(db, model_run_details):
    print(model_run_details)

    embedding_file_name = model_run_details['embedding_file_name']
    binary_quantization = model_run_details['binary_quantization']

    path = f"../data/embeddings_dict_{embedding_file_name}.pkl"
    print(f"loading embeddings from {path}")

    with open(path, "rb") as f:
        embeddings_dict = pickle.load(f)

    EMBEDDING_SIZE = embeddings_dict[model_run_details['model_name']][0].shape[0]

    print("creating table")



    if binary_quantization:

        items = []

        for idx, vec in enumerate(embeddings_dict[model_run_details['model_name']]):
            serialized_vec = serialize_f32(list(vec))
            items.append((idx, serialized_vec, serialized_vec))


        db.execute(f'''CREATE VIRTUAL TABLE vec_items_{model_run_details['model_name_normalized']} USING vec0(
                   embedding float[{EMBEDDING_SIZE}],
                   embedding_coarse bit[{EMBEDDING_SIZE}]
                   )
            '''
        )


        db.executemany(f'''INSERT INTO vec_items_{model_run_details['model_name_normalized']}
                    (rowid, embedding, embedding_coarse)
                    VALUES (?, ?, vec_quantize_binary(?))
                    ''',
                    items
        )


    else:
        db.execute(f'''CREATE VIRTUAL TABLE vec_items_{model_run_details['model_name_normalized']} USING vec0(
                   embedding float[{EMBEDDING_SIZE}]
                   )
            '''
        )

        items = []

        for idx, vec in enumerate(embeddings_dict[model_run_details['model_name']]):
            serialized_vec = serialize_f32(list(vec))
            items.append((idx, serialized_vec))


        # Perform batch insertion into the vec0 table
        db.executemany(f'''INSERT INTO vec_items_{model_run_details['model_name_normalized']}
                    (rowid, embedding)
                    VALUES (?, ?)
                    ''',
                    items
        )


    print("loaded succesfully")

    db.commit()

    return db


def load_ground_truth_into_db(db, golden_df_file_path):
    print("loading golden")
    golden_df = pd.read_csv(golden_df_file_path)

    if 'search_query' in golden_df.columns:
        query_field_name = 'search_query'
    else:
        query_field_name = 'query'

    golden_df['norm_url'] = golden_df['url'].apply(normalize_url)

    golden_df['query'] = golden_df[query_field_name]

    processed_golden_df_path = 'processed_golden_df.csv'

    golden_df.to_csv(processed_golden_df_path)

    queries = golden_df['query'].unique()

    golden_df[['query', 'url','norm_url']].to_sql("ground_truth", db, if_exists="replace", index=True)
    print("finished loading golden")

    return queries, processed_golden_df_path


def vector_db_pipeline(browsing_history_path, golden_df_file_path, model_run_details):
    db = create_db()

    load_history_in_db(
        db,
        browsing_history_path
    )

    read_and_insert_embeddings_into_db(
        db,
        model_run_details
    )

    queries, processed_golden_df_path = load_ground_truth_into_db(
        db,
        golden_df_file_path
    )

    return db, queries, processed_golden_df_path


def main(browsing_history_path, golden_df_file_path, model_run_details):
    vector_db_pipeline(browsing_history_path, golden_df_file_path, model_run_details)


if __name__ == "__main__":
    main()

