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
    return pd.read_csv("data/history.csv")

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
model_name = st.selectbox("Choose a model: ", embeddings_sizes.keys())
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

query = st.text_input("Enter a query", "github")
fe = FeatureExtractor(EMBEDDING_MODELS_DICT, model_name=model_name)
query_embedding = fe.get_embeddings([query])[0]
num_results = st.slider("Number of results: ", 3, 6, 3)

rows = db.execute(
    f"""
      SELECT
        rowid,
        distance
      FROM vec_items_{model_name_normalized}
      WHERE embedding MATCH ?
      ORDER BY distance
      LIMIT {num_results}
    """,
    [serialize_f32(query_embedding)],
).fetchall()

row_indices = [row for row, score in rows]
distance = [score for row, score in rows]

selected_rows = history.iloc[row_indices].copy()
selected_rows["distance"] = distance
st.write("Selected rows are", selected_rows)
