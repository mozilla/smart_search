import sqlite3
import pandas as pd
import spacy
from gliner import GLiNER
from tqdm import tqdm
from constants import TOPIC_LABELS
from entity_relation_scorer import get_ers_data
import logging

nlp = spacy.load("en_core_web_sm")
# set GENERATE_TOPIC True for the first run and then reset to False to get speedup
GENERATE_TOPIC = False
DATA_PATH = "data/"
KG_DB_PATH = f"{DATA_PATH}/ml_kg.db"
gliner_model = GLiNER.from_pretrained("urchade/gliner_largev2")
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_connection_to_places():
    return sqlite3.connect(f"{DATA_PATH}/places.sqlite")

def get_connection_to_kg():
    return sqlite3.connect(KG_DB_PATH)


def extract_additional_path_info(row):
    """break down the url path and extract useful path_info """
    url = row['url']
    host = row['host']
    path = url.replace(f"https://{host}", "").replace(f"http://{host}", "")
    path = path.strip("/")
    path = path.replace(".html", "").replace(".htm", "")
    path_info = path.split("/")
    return path_info

def extract_tags_batch(df):
    """
    extract tags using spacy pipeline on combined title + description columns
    based on analysis - useful tags - ADJ, PROPN, NOUN
    """
    texts = (df['title'].fillna('') + " " + df['description'].fillna('')).str.strip()
    docs = nlp.pipe(texts, disable=["ner"])
    useful_tags = ["ADJ", "PROPN", "NOUN"]
    # Extract tags for each document
    tags_list = []
    for doc in docs:
        tags = set()
        for token in doc:
            if token.pos_ in useful_tags and not token.is_stop:
                tags.add(token.text.strip().lower())
        tags_list.append(list(tags))
    return tags_list

def generate_topics(df):
    """
    Generates topics for each entry from title + description.
    Uses gliner zsl with TOPIC_LABELS
    - usage: for the first time set the GENERATE_TOPIC as True to generate.
             for subsequent runs set the GENERATE_TOPIC as False to get speedup
    """
    texts = (df['title'].fillna('') + " " + df['description'].fillna('')).str.strip().values.tolist()
    if GENERATE_TOPIC:
        topics = []
        for text in tqdm(texts):
            entities = gliner_model.predict_entities(text, TOPIC_LABELS, threshold=0.3)
            themes = list({entity["label"] for entity in entities})
            topics.append(themes)
        df['topics'] = topics
        df.to_parquet(f"{DATA_PATH}/input_data_df.parquet", index=False)
    else:
        df_bkp = pd.read_parquet(f"{DATA_PATH}/input_data_df.parquet")
        topics_lkp = df_bkp.set_index('url_hash')['topics'].to_dict()
        df['topics'] = df['url_hash'].map(topics_lkp)
    return df

def get_kg_input_data(firefox_conn, row_limit):
    """
    Get KG input data
    expects the places.sqlite copy in the DATA_PATH
    returns the top frecent entries (with various filters applied)
    uses row_limit to restrict the number of entries
    """
    firefox_cursor = firefox_conn.cursor()
    input_data = firefox_cursor.execute(f"""
        WITH TOP_FRECENT_PLACES AS
        (SELECT p.url, p.title, p.description, p.id AS place_id, p.frecency, p.origin_id, p.url_hash
        FROM moz_places p
        WHERE p.title NOTNULL
        AND url not like '%google.com/search?%'
        ORDER BY frecency DESC
        LIMIT {row_limit}
        ) 

        , TOP_PLACES_INFO AS
        (select * from TOP_FRECENT_PLACES
        UNION

        SELECT p.url, p.title, p.description, p.id AS place_id, p.frecency, p.origin_id, p.url_hash
        FROM moz_places p
        WHERE p.id in (select distinct(place_id) from moz_inputhistory)
        )
        , KEYWORDS_INFO AS
        (SELECT 
            ih.place_id, 
            json_group_array(
                json_object(
                    'keyword', ih.input,
                    'use_count', ih.use_count
                )
            ) AS keyword_data
        FROM 
            moz_inputhistory ih
        WHERE ih.input != ''
        GROUP BY 
            ih.place_id
        ORDER BY 
            ih.use_count DESC
        )

        , DOMAIN_INFO AS
        (SELECT 
            id AS origin_id, 
            host, 
            CAST(frecency AS REAL) / (SELECT SUM(frecency) * 1.0 FROM moz_origins WHERE frecency IS NOT NULL) AS domain_frecency
        FROM 
            moz_origins
        WHERE 
            frecency IS NOT NULL
        )

        SELECT p.*, kw.keyword_data, d.host, d.domain_frecency 
        FROM TOP_PLACES_INFO p
        LEFT JOIN KEYWORDS_INFO kw
        ON p.place_id = kw.place_id
        LEFT JOIN DOMAIN_INFO d
        ON p.origin_id = d.origin_id
        ORDER BY p.frecency DESC

    """).fetchall()
    firefox_conn.close()
    input_cols = ['url', 'title', 'description', 'place_id', 'frecency', 'origin_id', 'url_hash', 'keyword_data', 'host', 'domain_frecency']
    input_data_df = pd.DataFrame(input_data,
                                 columns=input_cols)
    # enrich with path_info and tags
    input_data_df['path_info'] = input_data_df.apply(extract_additional_path_info, axis=1)
    input_data_df['tags'] = extract_tags_batch(input_data_df)
    input_data_df = generate_topics(input_data_df)
    logger.info(input_data_df.head().T)
    ers_df = get_ers_data(input_data_df)
    logger.info(f"\n Number of entity-relations in KG = {len(ers_df)}")
    return ers_df
    
def load_kg_db(kg_conn, ers_df):
    kg_conn.execute("""
    CREATE TABLE IF NOT EXISTS ml_kg_info (
        entity TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        relation TEXT NOT NULL,
        url_hash INTEGER NOT NULL,
        score REAL NOT NULL
    );
    """)
    # load the entity relations score data into KG
    ers_df.to_sql("ml_kg_info", kg_conn, if_exists="replace", index=False)
    logger.info("ml_kg_info table should be loaded now")
    kg_conn.close()


def main():
    row_limit = 10000
    ers_df = get_kg_input_data(get_connection_to_places(), row_limit)
    load_kg_db(get_connection_to_kg(), ers_df)

if __name__ == '__main__':
    main()