import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

"""
purpose of this module is to help build entity, relation and score for KG
"""


def _generate_entity_rltn_score(src_entity, src_entity_type, relation, tgt_entity, score):
    return (src_entity, src_entity_type, relation, tgt_entity, score)

def _extract_keyword_entities_rltn_score(df, entity_name, entity_type, relation, tgt_entity_name, score_col=None):
    sel_df = df.loc[~df[entity_name].isna(), [entity_name, tgt_entity_name]].reset_index(drop=True)
    for ers_info, tgt_val in zip(sel_df[entity_name].apply(json.loads), sel_df[tgt_entity_name]):
        for ers in ers_info:
            for key, val in ers.items():
                # print(key, val, tgt_val)
                if key == entity_type:
                    src_entity = val
                if score_col and key == score_col:
                    score = 1+val
                else:
                    score = None
            yield _generate_entity_rltn_score(src_entity, entity_type, relation, tgt_val, score)
    
    
def _extract_domain_entities_rltn_score(df, entity_name, relation, tgt_entity_name, score_col=None):
    sel_df = df.loc[~df[entity_name].isna(), [entity_name, tgt_entity_name, score_col]].reset_index(drop=True)
    for _, row in sel_df.iterrows():
        yield _generate_entity_rltn_score(row[entity_name], entity_name, relation, row[tgt_entity_name], row[score_col])

def _extract_path_info_entities_rltn_score(df, entity_name, relation, tgt_entity_name, score_col=None):
    sel_df = df.loc[~df[entity_name].isna(), [entity_name, tgt_entity_name]].reset_index(drop=True)
    for _, row in sel_df.iterrows():
        for entity_val in row[entity_name]:
            if len(entity_val) > 2 and entity_val.isalpha():
                yield _generate_entity_rltn_score(entity_val, entity_name, relation, row[tgt_entity_name], score_col)

def _extract_tags_entities_rltn_score(df, entity_name, relation, tgt_entity_name, score_col=None):
    sel_df = df.loc[~df[entity_name].isna(), [entity_name, tgt_entity_name]].reset_index(drop=True)
    for _, row in sel_df.iterrows():
        for entity_val in row[entity_name]:
            if len(entity_val) > 2 and entity_val.isalnum():
                yield _generate_entity_rltn_score(entity_val, 'tag', relation, row[tgt_entity_name], score_col)

def _extract_topics_entities_rltn_score(df, entity_name, relation, tgt_entity_name, score_col=None):
    sel_df = df.loc[~df[entity_name].isna(), [entity_name, tgt_entity_name]].reset_index(drop=True)
    for _, row in sel_df.iterrows():
        for entity_val in row[entity_name]:
            if len(entity_val) > 1:
                yield _generate_entity_rltn_score(entity_val, 'topic', relation, row[tgt_entity_name], score_col)

def get_ers_data(input_data_df):
    keyword_ers = [ers for ers in (_extract_keyword_entities_rltn_score(input_data_df, 'keyword_data', 'keyword', 'refers_to', 'url_hash', 'use_count'))]
    logger.info(f" Number of keyword entities = {len(keyword_ers)}")
    logger.info(keyword_ers[:5])
    domain_ers = [ers for ers in _extract_domain_entities_rltn_score(input_data_df, 'host', 'contains', 'url_hash', 'domain_frecency')]
    logger.info(f"\n Number of domain entities = {len(domain_ers)}")
    logger.info(domain_ers[:5])
    path_info_ers = [ers for ers in _extract_path_info_entities_rltn_score(input_data_df, 'path_info', 'parses_to', 'url_hash', 1.0)]
    logger.info(f"\n Number of path info entities = {len(path_info_ers)}")
    logger.info(path_info_ers[:5])
    tags_ers = [ers for ers in _extract_tags_entities_rltn_score(input_data_df, 'tags', 'tagged_has', 'url_hash', 1.0)]
    logger.info(f"\n Number of tag entities = {len(tags_ers)}")
    logger.info(tags_ers[:5])
    topics_ers = [ers for ers in _extract_topics_entities_rltn_score(input_data_df, 'topics', 'belongs_to', 'url_hash', 1.0)]
    logger.info(f"\n Number of topic entities = {len(topics_ers)}")
    logger.info(topics_ers[:5])
    return pd.DataFrame(keyword_ers + domain_ers + path_info_ers + tags_ers + topics_ers,
            columns=['entity', 'entity_type', 'relation', 'url_hash', 'score'])