import streamlit as st
import numpy as np
import sqlite3
import sqlite_vec
from typing import List
import struct
import pickle
import pandas as pd
from constants import EMBEDDING_MODELS_DICT
from feature_extractor import FeatureExtractor
from kg import extract_tags_for_queries, infer_topics, fetch_entity_relations_with_keywords

@st.cache_data
def get_embeddings_data():
    path = "data/embeddings_dict.pkl"

    with open(path, "rb") as f:
        embeddings_dict = pickle.load(f)
    return embeddings_dict

@st.cache_data
def get_embeddings_size():
    path = "data/embeddings_sizes.pkl"

    with open(path, "rb") as f:
        embeddings_sizes = pickle.load(f)
    return embeddings_sizes

@st.cache_data
def get_history_data():
    return pd.read_csv("data/history_output_file.csv")


@st.cache_resource
def get_sql_connection():
    db_path = "data/ml_kg_10k_bkp.db"
    connection = sqlite3.connect(db_path, check_same_thread=False)
    places_db_path = "data/places.sqlite"
    connection.execute(f"ATTACH DATABASE '{places_db_path}' AS places_db;")
    return connection

st.set_page_config(layout="wide")
st.title("History search Demo App")
db = sqlite3.connect(":memory:")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)

embeddings_sizes = get_embeddings_size()
embeddings_dict = get_embeddings_data()
history = get_history_data()

conn = get_sql_connection()

query = st.text_input("Enter a query", "github")
num_results = st.slider("Number of results: ", 2, 6, 2)
col1, col2 = st.columns(2)
with col1:
    st.markdown("**:violet[Semantic search]**")
    model_name = st.selectbox("Choose a embedding model: ", embeddings_sizes.keys())
    EMBEDDING_SIZE = embeddings_sizes[model_name]

    items = []
    for idx, vec in enumerate(embeddings_dict[model_name]):
        items.append((idx, list(vec)))
    model_name_normalized = model_name.replace("/","_").replace("-","_").replace(".","_")

    db.execute(f"CREATE VIRTUAL TABLE vec_items_{model_name_normalized} USING vec0(embedding float[{EMBEDDING_SIZE}])")

    with db:
        for item in items:
            db.execute(
                f"INSERT INTO vec_items_{model_name_normalized}(rowid, embedding) VALUES (?, ?)",
                [item[0], serialize_f32(item[1])],
            )


    fe = FeatureExtractor(EMBEDDING_MODELS_DICT, model_name=model_name)
    query_embedding = fe.get_embeddings([query])[0]
    

    rows = db.execute(
        f"""
        SELECT
            rowid,
            vec_distance_cosine(embedding, ?) AS cosine_distance
        FROM vec_items_{model_name_normalized}
        ORDER BY cosine_distance
        LIMIT {num_results}
        """,
        [serialize_f32(query_embedding)],
    ).fetchall()

    row_indices = [row for row, score in rows]
    distance = [score for row, score in rows]

    selected_rows = history.iloc[row_indices].copy()
    selected_rows["distance"] = distance
    st.write("Selected rows are", selected_rows)

with col2:
    search_tags = extract_tags_for_queries([query])[0]
    search_topics = infer_topics([query], pbar=False)[0]
    st.write(f"search_tags = {search_tags}")
    st.write(f"search_topics = {search_topics}")
    results = fetch_entity_relations_with_keywords(conn, query, search_tags, search_topics).head(num_results)
    st.markdown("**:violet[KG approach]**")
    st.write("Selected rows are", results)    
