import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from src.constants import EMBEDDING_MODELS_DICT
from src.feature_extractor import FeatureExtractor

def build_embeddings_by_language(
    sample_pkl_path="./data/sample_data_by_lang.pkl",
    embeddings_pkl_path="./data/sample_embeddings_by_lang.pkl",
    model_name_en="Xenova/all-MiniLM-L6-v2",
    model_name_multi="Xenova/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size=512,
    rebuild=False
):

    sample_pkl_path = Path(sample_pkl_path)
    embeddings_pkl_path = Path(embeddings_pkl_path)
    embeddings_pkl_path.parent.mkdir(parents=True, exist_ok=True)

    # load from file
    if embeddings_pkl_path.exists() and not rebuild:
        with open(embeddings_pkl_path, "rb") as f:
            sample_embeddings_by_lang = pickle.load(f)
        print(f"Loaded embeddings PKL from {embeddings_pkl_path}")
        for lang, arr in sample_embeddings_by_lang.items():
            shape = np.array(arr).shape
            print(f"  {lang}: {shape}")
        return sample_embeddings_by_lang


    if not sample_pkl_path.exists():
        raise FileNotFoundError(f"{sample_pkl_path} not found. Run sample_by_language() first.")

    with open(sample_pkl_path, "rb") as f:
        sample_data_by_lang = pickle.load(f)


    print("Loading embedding models...")
    fe_en = FeatureExtractor(EMBEDDING_MODELS_DICT, model_name=model_name_en)
    fe_multi = FeatureExtractor(EMBEDDING_MODELS_DICT, model_name=model_name_multi)

    sample_embeddings_by_lang = {}
    langs = list(sample_data_by_lang.keys())

    for i in range(len(langs)):
        lang = langs[i]
        print(f"preparing for lang = {lang}")
        df_lang = sample_data_by_lang[lang].reset_index(drop=True)
        texts = df_lang["combined_text"].fillna("").values.tolist()
        extractor = fe_en if lang == "en" else fe_multi

        # batch embedding
        chunks = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"{i+1}/{len(langs)} {lang} batches", leave=False):
            batch = texts[i:i + batch_size]
            emb = extractor.get_embeddings(batch)
            emb = np.asarray(emb)
            chunks.append(emb)

        embeddings = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=np.float32)
        sample_embeddings_by_lang[lang] = embeddings
        print(f"  {lang}: {embeddings.shape}")


    with open(embeddings_pkl_path, "wb") as f:
        pickle.dump(sample_embeddings_by_lang, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved embeddings PKL to {embeddings_pkl_path}")

    return sample_embeddings_by_lang


if __name__ == "__main__":
    build_embeddings_by_language(
        sample_pkl_path="./data/sample_data_by_lang.pkl",
        embeddings_pkl_path="./data/sample_embeddings_by_lang.pkl",
        model_name_en="Xenova/all-MiniLM-L6-v2",
        model_name_multi="Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size=512,
        rebuild=False
    )