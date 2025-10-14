"""
Evaluate static embeddings
"""

import json
import shutil
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

MODELS = [
    # ("Xenova/all-MiniLM-L6-v2", []),
    ("sentence-transformers/static-retrieval-mrl-en-v1", [1024]),
    ("sentence-transformers/static-similarity-mrl-multilingual-v1", [1024]),
    ("minishlab/potion-retrieval-32M", [512]),
    ("minishlab/potion-multilingual-128M", [256]),
]
GOLDEN_QUERY_COLS = ["search_query", "url"]
PROFILE_COLS = [
    "url",
    "title",
    "description",
    "frecency",
    "last_visit_date",
]

PERSONAS = {
    "creative_hobbyist": {
        "golden_query_data": DATA_DIR / "creative_hobyist_profile_golden_queries.csv",
        "profile_data": DATA_DIR / "profiles/creative_hobbyist.csv",
    },
    # "chidam": {
    #     "golden_query_data": DATA_DIR / "chidam_golden_query.csv",
    #     "profile_data": DATA_DIR / "history_output_file.csv",
    # },
}


class Embedder(ABC):
    """Abstract base class models"""

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings into numerical vector representations."""
        pass


def get_model(model_name: str) -> Embedder:
    if model_name in {
        "Xenova/all-MiniLM-L6-v2",
        "sentence-transformers/static-retrieval-mrl-en-v1",
        "sentence-transformers/static-similarity-mrl-multilingual-v1",
    }:
        model_only = model_name.split("/")[1]
        return SentenceTransformer(model_only, device="cpu")
    elif model_name in {
        "minishlab/potion-retrieval-32M",
        "minishlab/potion-multilingual-128M",
    }:
        return StaticModel.from_pretrained(model_name)

    raise ValueError(f"Unknown model {model_name}")


def get_persona_data(persona_name: str):
    profile_data = PERSONAS[persona_name]["profile_data"]
    golden_query_data = PERSONAS[persona_name]["golden_query_data"]

    if persona_name not in PERSONAS.keys():
        raise ValueError(f"persona_name entered not in {list(PERSONAS.keys())}")

    golden_queries_df = pd.read_csv(golden_query_data)[GOLDEN_QUERY_COLS]
    profile_df = pd.read_csv(profile_data)

    return golden_queries_df, profile_df


def search(
    query: str,
    model: Embedder,
    history_embeddings: list[list[float]],
    history_df: pd.DataFrame,
    top_k: int = 2,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Returns the top_k predictions from the history ranked by similarity score.
    """
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, history_embeddings)[0]

    # Pull out anything that meets the threshold.
    valid_idx = np.where(similarity_scores >= threshold)[0]

    # Get the indexes ranked by similarity, descending.
    ranked_idx = valid_idx[np.argsort(-similarity_scores[valid_idx])[:top_k]]

    return history_df.iloc[ranked_idx]


def write_results(
    persona_name: str,
    model_name: str,
    results: list[dict],
) -> None:
    """
    Write evaluation results into 'correct' and 'incorrect' CSV files.
    """
    output_dir = DATA_DIR / Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    correct_rows = []
    incorrect_rows = []

    for result in results:
        predictions: pd.DataFrame = result["predictions"]
        row = {
            "query": result["query"],
            "actual": result["actual"],
            "predictions": predictions[["url", "title"]].to_dict(orient="records"),
        }
        if result["is_correct"]:
            correct_rows.append(row)
        else:
            incorrect_rows.append(row)

    base_name = f"{persona_name}_{model_name.split('/')[-1]}"
    correct_path = output_dir / f"{base_name}_correct.json"
    incorrect_path = output_dir / f"{base_name}_incorrect.json"

    with correct_path.open("w") as file:
        json.dump(correct_rows, file, indent=2, sort_keys=True)
    with incorrect_path.open("w") as file:
        json.dump(incorrect_rows, file, indent=2, sort_keys=True)

    print("  Correct results:", correct_path.relative_to(ROOT_DIR))
    print("  Incorrect results:", correct_path.relative_to(ROOT_DIR))


def main() -> None:
    for persona_name in PERSONAS:
        golden_queries_df, history_df = get_persona_data(persona_name)

        for model_name, dims in MODELS:
            print(f'Evaluating "{persona_name}" with "{model_name}"')
            model = get_model(model_name)
            history_titles: list[str] = history_df["title"].str.lower().values.tolist()  # type: ignore
            history_embeddings: list[list[float]] = model.encode(history_titles)

            results = []
            correct = 0
            for _idx, row in golden_queries_df.iterrows():
                query = row["search_query"].lower()
                golden_url = row["url"]
                predictions = search(
                    query,
                    model,
                    history_embeddings,
                    history_df,
                    top_k=3,
                    threshold=0.5,
                )
                is_correct = golden_url in predictions["url"].tolist()
                if is_correct:
                    correct += 1

                results.append(
                    {
                        "query": query,
                        "actual": golden_url,
                        "predictions": predictions,
                        "is_correct": is_correct,
                    }
                )

            print(f'  "{persona_name}" - "{model_name}"')
            print(f"  Results correct {correct} / {len(golden_queries_df)}")
            print(f"  Percent {correct / len(golden_queries_df) * 100.0:.2f}%")
            write_results(persona_name, model_name, results)


if __name__ == "__main__":
    main()
