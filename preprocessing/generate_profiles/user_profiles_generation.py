# gliner, a NER model.
# input:
#   - a text: "Apple released the iPhone 12 in California."
#   - a list: ["company", "product", "location"]
# output:
# text with label with score
# demo: https://huggingface.co/spaces/urchade/gliner_multiv2.1

import os
import time
import hashlib
import random
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import commentjson

from gliner import GLiNER


def detect_themes(gliner_model, df, labels, use_gpu=False):

    texts = df["combined_text"].values.tolist()
    out = []

    if not use_gpu:
        for text in tqdm(texts, desc="GLiNER"):
            entities = gliner_model.predict_entities(text, labels)
            themes = list({ent["label"] for ent in entities})
            out.append(themes)
    else:
        gliner_model.model.to("cuda").eval()
        batch_size = 128
        for i in tqdm(range(0, len(texts), batch_size), desc="GLiNER (batched)"):
            batch = texts[i: i + batch_size]
            entities_list = gliner_model.batch_predict_entities(batch, labels)
            for j, entities in enumerate(entities_list):
                out.append(list({ent["label"] for ent in entities}))

    return out


def add_theme_to_data(
        input_pkl="./data/sample_data_by_lang.pkl",
        categories_path="./data/categories.json",
        output_pkl="./data/sample_data_by_lang_updated.pkl",
        model_name="urchade/gliner_largev2",
        use_gpu=False,
        rebuild=False
    ):

    input_pkl = Path(input_pkl)
    categories_path = Path(categories_path)
    output_pkl = Path(output_pkl)

    if not rebuild and output_pkl.exists():
        print(f"Found {output_pkl}, skipping.")
        return

    with open(input_pkl, "rb") as f:
        sample_data_by_lang = pickle.load(f)

    with open(categories_path, "r", encoding="utf-8") as f:
        categories = commentjson.load(f)

    gliner_model = GLiNER.from_pretrained(model_name)

    labels = categories[::]

    sample_data_by_lang_updated = {}
    for lang, sample_data in sample_data_by_lang.items():
        print(lang)
        themes_result = detect_themes(gliner_model, sample_data, labels, use_gpu)
        sample_data['themes'] = themes_result
        sample_data_by_lang_updated[lang] = sample_data

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(sample_data_by_lang_updated, f)



def apply_category_filter(categories, themes):
    for cat in categories:
        if cat in themes:
            return True
    return False


def generate_url_hash(url):
    hash_object = hashlib.md5(url.encode("utf-8"))
    url_hash = int(hash_object.hexdigest(), 16) % (10 ** 14)
    return url_hash


def generate_last_visit_date(index):
    base_time = int(time.time() * 1_000_000)  # Current time in microseconds
    return base_time + index * 2_000  # Increment by 2000 microseconds (2ms) per row


def generate_user_profile(user_profiles, sample_path):
    """
    For non-English only has 5000, English has 35000, so either allow replacement or set the num_samples lower
    """

    with open(sample_path, "rb") as f:
        sample_data_by_lang_updated = pickle.load(f)

    profile_name = random.choice(list(user_profiles.keys()))
    print(profile_name)
    profile = user_profiles[profile_name]
    categories = profile['categories']
    langs = random.choice(profile['lang'])
    desc = profile['description']
    print(categories)
    print(langs)
    num_samples = random.randint(1000, 5000)
    print(f"num_samples = {num_samples}")
    frecency = [random.randint(100,5000) for _ in range(num_samples)]
    res = []
    for lang in langs:
        df = sample_data_by_lang_updated.get(lang)
        res.append(df.loc[df['themes'].apply(list).apply(lambda themes: apply_category_filter(categories, themes))])
    res_df = pd.concat(res, axis=0).sample(num_samples).reset_index(drop=True)
    res_df['frecency'] = frecency
    res_df['url_hash'] = res_df['url'].apply(generate_url_hash)
    res_df["last_visit_date"] = [generate_last_visit_date(i) for i in range(len(res_df))]

    return profile_name, categories, langs, desc, res_df


def generate_user_profiles(
        json_path="./data/user_profiles.json",
        sample_path="./data/sample_data_by_lang_updated.pkl",
        output_path="./data/profiles_generated.pkl",
        n_profiles=25,
        random_state=42,
        rebuild=False,
    ):

    json_path = Path(json_path)
    output_path = Path(output_path)

    if not rebuild and output_path.exists():
        print(f"Found {output_path}, skipping.")
        return

    if random_state is not None:
        random.seed(random_state)

    with open(json_path, "r", encoding="utf-8") as f:
        user_profiles = commentjson.load(f)

    profiles_generated = []

    for i in range(n_profiles):
        profile_name, categories, langs, desc, res_df = generate_user_profile(user_profiles, sample_path)
        profiles_generated.append(
            {
                'profile_name': profile_name,
                'categories': categories,
                'langs': langs,
                'desc': desc,
                'df': res_df,
            }
        )

    with open(output_path, "wb") as f:
        pickle.dump(profiles_generated, f)


def generate_user_profile_queries(
    json_path="./data/user_profile_queries.json",
    output_path="./data/user_profile_queries.pkl",
    rebuild=False,
):

    json_path = Path(json_path)
    output_path = Path(output_path)

    if not rebuild and output_path.exists():
        print(f"Found {output_path}, skipping.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        user_profile_queries = commentjson.load(f)

    with open(output_path, "wb") as f:
        pickle.dump(user_profile_queries, f)


def choose_profile(idx, profiles_generated, profiles_dir):
    profile_name = profiles_generated[idx]['profile_name']
    print(f"profile chosen = {idx}; profile_name = {profile_name}")
    profile_file_name = profile_name.lower().replace(" ", "_")
    profile_history = profiles_generated[idx]['df'] \
        .drop(['topic', 'lang', 'domain', 'combined_text', 'themes'], axis=1) \
        .sort_values('frecency', ascending=False) \
        .reset_index(drop=True)

    profile_history.to_csv(f"{profiles_dir}{profile_file_name}.csv")
    return profile_name, profile_history


def user_profiles_generation(
        idx=2,
        profiles_dir="./data/profiles/",
        sample_data_by_lang_updated_path="./data/sample_data_by_lang_updated.pkl",
        sample_data_by_lang_updated_en_save_path="./data/en_search_web_history_data.csv",
        profiles_generated_path="./data/profiles_generated.pkl",
        user_profile_queries_path="./data/user_profile_queries.pkl",
    ):

    os.makedirs(f"{profiles_dir}", exist_ok=True)

    with open(sample_data_by_lang_updated_path, "rb") as f:
        sample_data_by_lang_updated = pickle.load(f)

    sample_data_by_lang_updated['en'].to_csv(sample_data_by_lang_updated_en_save_path, index=False)

    with open(profiles_generated_path, "rb") as f:
        profiles_generated = pickle.load(f)

    with open(user_profile_queries_path, "rb") as f:
        user_profile_queries = pickle.load(f)

    profile_name, profile_history = choose_profile(idx, profiles_generated, profiles_dir)
    profile_queries = user_profile_queries[profile_name]

    return profile_name, profile_history, profile_queries


if __name__ == "__main__":

    # 3+ hours
    add_theme_to_data(
        input_pkl="./data/sample_data_by_lang.pkl",
        categories_path="./data/categories.json",
        output_pkl="./data/sample_data_by_lang_updated.pkl",
        model_name="urchade/gliner_largev2",
        use_gpu=True,
        rebuild=False
    )

    generate_user_profile_queries(
        json_path="./data/user_profile_queries.json",
        output_path="./data/user_profile_queries.pkl",
        rebuild=False
    )

    generate_user_profiles(
        json_path="./data/user_profiles.json",
        sample_path="./data/sample_data_by_lang_updated.pkl",
        output_path="./data/profiles_generated.pkl",
        n_profiles=25,
        random_state=42,
        rebuild=False,
    )

    profile_name, profile_history, profile_queries = user_profiles_generation(
        idx=2,
        profiles_dir="./data/profiles/",
        sample_data_by_lang_updated_path="./data/sample_data_by_lang_updated.pkl",
        sample_data_by_lang_updated_en_save_path="./data/en_search_web_history_data.csv",
        profiles_generated_path="./data/profiles_generated.pkl",
        user_profile_queries_path="./data/user_profile_queries.pkl",
    )

    print(profile_name)
    print(profile_history.columns.tolist())
    print(profile_history.head(5))
    print(profile_queries)
