import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.llm_judge import llm_as_judge
from src.llm_api_judge import evaluation_prompt_batch_gpt
from src.llm_api_query import query_prompt_batch_gpt

import os
import time
import json
import hashlib
import random
import pickle
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re

import commentjson

from gliner import GLiNER

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")




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

    base_dir = Path(profiles_dir).resolve()
    save_dir = base_dir / f"{idx:03d}_{profile_file_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_to_path = save_dir / "history.csv"
    profile_history.to_csv(save_to_path, index=False)

    return profile_name, profile_history


def save_sameple_data_to_csv(
        sample_data_by_lang_updated_path="./data/sample_data_by_lang_updated.pkl",
        save_path="./data/",
        lang="en",
    ):


    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sample_data_by_lang_updated_save_path = f"{save_path}/{lang}_search_web_history_data.csv"

    if Path(sample_data_by_lang_updated_save_path).exists():
        print(f"Found {sample_data_by_lang_updated_save_path}, skipping.")
        return

    with open(sample_data_by_lang_updated_path, "rb") as f:
        sample_data_by_lang_updated = pickle.load(f)

    sample_data_by_lang_updated[lang].to_csv(sample_data_by_lang_updated_save_path, index=False)


def user_profiles_generation(
        idx=2,
        profiles_dir="./data/profiles/",
        profiles_generated_path="./data/profiles_generated.pkl",
        user_profile_queries_path="./data/user_profile_queries.pkl",
        use_api_llm=False,
    ):

    os.makedirs(f"{profiles_dir}/", exist_ok=True)

    with open(profiles_generated_path, "rb") as f:
        profiles_generated = pickle.load(f)

    with open(user_profile_queries_path, "rb") as f:
        user_profile_queries = pickle.load(f)

    profile_name, profile_history = choose_profile(idx, profiles_generated, profiles_dir)
    profile_queries = user_profile_queries[profile_name]

    # user_query_generation(profile_name, profile_history, user_profile_queries, idx, profiles_dir, use_api_llm)
    user_query_generation2(profile_name, profile_history, idx, profiles_dir)

    return profile_name, profile_history, profile_queries


def user_query_generation2(profile_name, profile_history, idx, profiles_dir):

    BATCH_SIZE = 100


    profile_file_name = profile_name.lower().replace(" ", "_")
    base_dir = Path(profiles_dir).resolve()
    save_dir = base_dir / f"{idx:03d}_{profile_file_name}"
    save_dir.mkdir(parents=True, exist_ok=True)


    random_histories = profile_history.sample(n=BATCH_SIZE, random_state=42).reset_index(drop=True)

    random_histories["combined_text"] = (
            random_histories["title"].fillna("") + " " + random_histories["description"].fillna("")
    ).str.strip()

    random_histories = list(random_histories[["url", "combined_text"]].itertuples(index=False, name=None))

    save_to_path = save_dir / "llm_query"
    save_to_path.mkdir(parents=True, exist_ok=True)

    res = []
    pending_rows = []
    pending_pairs = []
    pending_paths = []
    for i, history in enumerate(random_histories):

        save_to_file = f"{save_to_path}/q_{i:05d}.json"

        if Path(save_to_file).exists():
            with open(save_to_file, "r", encoding="utf-8") as f:
                response = json.load(f)
            url = response["url"]
            search_query = response["search_query"]
            combined_text = response["combined_text"]
            res.append({"url": url, "search_query": search_query, "combined_text": combined_text})
            continue

        history = {"url": history[0], "combined_text": history[1]}
        url = history["url"]
        combined_text = history["combined_text"]

        pending_rows.append((i, history))
        pending_pairs.append((url, combined_text))
        pending_paths.append(save_to_file)

        if len(pending_pairs) == BATCH_SIZE or (i + 1) == len(random_histories):
            outs = query_prompt_batch_gpt(pending_pairs, max_workers=BATCH_SIZE, openai_api_key=OPENAI_API_KEY)

            for (jj, hhistory), path, out in zip(pending_rows, pending_paths, outs):

                response = format_api_judge_response(out)
                if type(response) == str or "search_query" not in response or "url" not in response:
                    print(f"Can't parse response {response}, skipping.")
                    continue

                response["combined_text"] = hhistory["combined_text"]

                res.append({"url": response["url"], "search_query": response["search_query"], "combined_text": response["combined_text"]})

                with open(path, "w", encoding="utf-8") as f:
                    json.dump(response, f, indent=2)

            pending_rows.clear()
            pending_pairs.clear()
            pending_paths.clear()

    base_dir = Path(profiles_dir).resolve()
    save_dir = base_dir / f"{idx:03d}_{profile_file_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(res)
    df.to_csv(save_dir / "query.csv", index=False)



def user_query_generation(profile_name, profile_history, user_profile_queries, idx, profiles_dir, use_api_llm=False):
    BATCH_SIZE = 100

    if not use_api_llm:
        judge = llm_as_judge()

    profile_queries = [x for sub in user_profile_queries[profile_name]["queries"].values() for x in sub]

    profile_file_name = profile_name.lower().replace(" ", "_")
    base_dir = Path(profiles_dir).resolve()
    save_dir = base_dir / f"{idx:03d}_{profile_file_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    res = []

    for i, query in enumerate(profile_queries):
        best = ["", 0, 0, ""]

        save_to_path = save_dir / "llm_response" / f"q_{i:03d}"
        save_to_path.mkdir(parents=True, exist_ok=True)

        pending_rows = []   # [(j, row), ...]
        pending_pairs = []  # [(query, combined_text), ...]
        pending_paths = []

        for j, row in tqdm(profile_history.iterrows(), total=len(profile_history), desc=f"query {i+1:03d}/{len(profile_queries)} - Processing histories"):

            save_to_file = f"{save_to_path}/h_{j:05d}.json"

            if Path(save_to_file).exists():
                with open(save_to_file, "r", encoding="utf-8") as f:
                    response = json.load(f)
                url = response["history"]["url"]
                relevance_score = response["relevance_score"]
                binary_decision = response["binary_decision"]
                if relevance_score > best[1]:
                    best[0] = url
                    best[1] = relevance_score
                    best[2] = binary_decision
                    best[3] = f"q_{i:03d}/h_{j:05d}.json"
                continue

            combined_text = f"{row['title']} {row['description']}".strip()
            pending_rows.append((j, row))
            pending_pairs.append((query, combined_text))
            pending_paths.append(save_to_file)

            # if batch ready, run it and write files
            if len(pending_pairs) == BATCH_SIZE or (j+1) == len(profile_history):

                if not use_api_llm:
                    outs = judge.evaluation_prompt_batch(pending_pairs, batch_size=BATCH_SIZE)
                else:
                    outs = evaluation_prompt_batch_gpt(pending_pairs, max_workers=BATCH_SIZE, openai_api_key=OPENAI_API_KEY)

                for (jj, rrow), path, out in zip(pending_rows, pending_paths, outs):

                    # response = format_judge_response(out)
                    response = format_api_judge_response(out)
                    if type(response) == str or "relevance_score" not in response:
                        print(f"Can't parse response {response}, skipping.")
                        continue
                    response["query"] = query
                    response["combined_text_for_llm"] = f"{rrow['title']} {rrow['description']}".strip()
                    response["history"] = rrow.to_dict()
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(response, f, indent=2)

                    url = response["history"]["url"]
                    relevance_score = response["relevance_score"]
                    binary_decision = response["binary_decision"]
                    if relevance_score > best[1]:
                        best[0] = url
                        best[1] = relevance_score
                        best[2] = binary_decision
                        best[3] = f"q_{i:03d}/h_{jj:05d}.json"

                pending_rows.clear()
                pending_pairs.clear()
                pending_paths.clear()

        res.append({"search_query": query, "url": best[0], "relevance_score": best[1], "binary_decision": best[2], "combined_text_for_llm": best[3]})


    base_dir = Path(profiles_dir).resolve()
    save_dir = base_dir / f"{idx:03d}_{profile_file_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(res)
    df.to_csv(save_dir / "query_all.csv", index=False)

    df_pos = df[pd.to_numeric(df["binary_decision"], errors="coerce") == 1]
    df_pos.to_csv(save_dir / "query.csv", index=False)


def format_judge_response(answer):

    s0 = answer[0]['generated_text']
    s1 = s0.replace("\n", "")
    s2 = re.sub(r'```json|```', '', s1).strip()
    s3 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s2)

    try:
        formatted_answer = json.loads(s0)
    except Exception as e:
        try:
            print("cleaning json")
            formatted_answer = json.loads(s1)
        except Exception as e:
            try:
                print("trying another way")
                formatted_answer = json.loads(s2)
            except Exception as e:
                try:
                    print("escaping backslashes")
                    formatted_answer = json.loads(s3)
                except Exception as e:
                    formatted_answer = s0

    return formatted_answer

def format_api_judge_response(answer):

    s = answer['generated_text'].strip()
    return json.loads(s)


if __name__ == "__main__":

    # 3+ hours
    add_theme_to_data(
        input_pkl=os.path.join(DATA_DIR, "sample_data_by_lang.pkl"),
        categories_path=os.path.join(SCRIPT_DIR, "categories.json"),
        output_pkl=os.path.join(DATA_DIR, "sample_data_by_lang_updated.pkl"),
        model_name="urchade/gliner_largev2",
        use_gpu=True,
        rebuild=False
    )

    generate_user_profile_queries(
        json_path=os.path.join(SCRIPT_DIR, "user_profile_queries.json"),
        output_path=os.path.join(DATA_DIR, "user_profile_queries.pkl"),
        rebuild=False
    )

    generate_user_profiles(
        json_path=os.path.join(SCRIPT_DIR, "user_profiles.json"),
        sample_path=os.path.join(DATA_DIR, "sample_data_by_lang_updated.pkl"),
        output_path=os.path.join(DATA_DIR, "profiles_generated.pkl"),
        n_profiles=25,
        random_state=42,
        rebuild=False,
    )

    save_sameple_data_to_csv(
        sample_data_by_lang_updated_path=os.path.join(DATA_DIR, "sample_data_by_lang_updated.pkl"),
        save_path=DATA_DIR,
        lang="en",
    )

    OPENAI_API_KEY = ""

    for i in range(25):
        profile_name, profile_history, profile_queries = user_profiles_generation(
            idx=i,
            profiles_dir=os.path.join(SCRIPT_DIR, "./data/generated_profiles_2/"),
            profiles_generated_path=os.path.join(DATA_DIR, "profiles_generated.pkl"),
            user_profile_queries_path=os.path.join(DATA_DIR, "user_profile_queries.pkl"),
            use_api_llm=True
        )

        print(profile_name)
        print(profile_history.columns.tolist())
        print(profile_history.head(5))
        print(profile_queries)
