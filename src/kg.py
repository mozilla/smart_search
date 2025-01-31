import spacy
import pandas as pd
from gliner import GLiNER
from tqdm import tqdm
from constants import TOPIC_LABELS

gliner_model = GLiNER.from_pretrained("urchade/gliner_largev2")
labels = TOPIC_LABELS[::]

nlp = spacy.load("en_core_web_sm")

def extract_tags_for_queries(queries):
    texts = queries[::]
    
    docs = nlp.pipe(texts, disable=["ner"])

    tags_list = []
    for doc in docs:
        tags = set()
        
        for token in doc:
            # print(token.pos_)
            if token.pos_ in ["ADJ", "PROPN", "NOUN"] and not token.is_stop:
                tags.add(token.text.strip().lower())
        
        tags_list.append(list(tags))  # Append the tags for this document
    
    return tags_list

def infer_topics(queries, pbar=True):
    topics = []
    if pbar:
        for query in tqdm(queries):
            entities = gliner_model.predict_entities(query, labels, threshold=0.3)
            themes = list({entity["label"] for entity in entities})
            topics.append(themes)
    else:
        for query in queries:
            entities = gliner_model.predict_entities(query, labels, threshold=0.3)
            themes = list({entity["label"] for entity in entities})
            topics.append(themes)
    return topics
    
def fetch_entity_relations_with_keywords(conn, search_keyword, search_tags, search_topics):
    # Convert the list of search keywords into a string suitable for SQL
    if not search_keyword:
        raise ValueError("search_keywords list cannot be empty.")

    keyword_placeholder = f"'{search_keyword}'"
    # print("keyword_placeholder = ", keyword_placeholder)
    
    tag_placeholder = ', '.join(f"'{tag}'" for tag in search_tags)
    # print("tag_placeholder = ", tag_placeholder)

    topic_placeholder = ', '.join(f"'{topic}'" for topic in search_topics)
    # print("topic_placeholder = ", topic_placeholder)

    # Define the query with the dynamic IN clause
    query = f"""
    WITH entity_relations_info AS (
        SELECT 
            m.entity,
            m.entity_type,
            m.relation,
            m.url_hash,
            m.score,
            p.url,
            p.title,
            p.frecency
        FROM 
            ml_kg_info m
        JOIN 
            places_db.moz_places p
        ON 
            m.url_hash = p.url_hash
        WHERE
            (m.entity IN ({keyword_placeholder}) AND
            m.entity_type = 'keyword') OR
            (m.entity IN ({tag_placeholder})  AND
             m.entity_type = 'tag') OR
            (m.entity IN ({topic_placeholder})  AND
             m.entity_type = 'topic')
        ORDER BY 
            m.score DESC
    )

    SELECT 
        url_hash, 
        url, 
        title,
        SUM(score) AS total_score
    FROM 
        entity_relations_info
    GROUP BY 
        url_hash, url, title
    ORDER BY 
        total_score DESC;
    
    """

    results = pd.read_sql_query(query, conn)
    return results
