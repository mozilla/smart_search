import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from urllib.parse import urlparse
import pickle
import zipfile
import requests
from pathlib import Path
import gzip

def download_mlsam_dataset(data_dir="./data"):

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    output_file = data_path / "mlsam-multilingual-summarization-dataset.zip"
    extract_dir = data_path / output_file.stem

    url = "https://www.kaggle.com/api/v1/datasets/download/thedevastator/mlsam-multilingual-summarization-dataset"

    if not output_file.exists():
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            block_size = 8192  # chunk size

            with open(output_file, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"Downloaded to {output_file}")
    else:
        print(f"Dataset file already exists at {output_file}, skipping download.")

    # Unzip
    if extract_dir.exists():
        print(f"Extracted folder already exists at {extract_dir}, skipping unzip.")
    else:
        with zipfile.ZipFile(output_file, "r") as zip_ref:
            members = zip_ref.infolist()
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    zip_ref.extract(member, extract_dir)
                    pbar.update(1)
        print(f"Extracted contents to {extract_dir}")


def download_msmarco_dataset(data_dir="./data"):

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    gz_file = data_path / "msmarco-docs.tsv.gz"
    tsv_file = data_path / "msmarco-docs.tsv"

    # source: https://github.com/microsoft/msmarco/blob/master/Datasets.md
    url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"

    if not gz_file.exists():
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            block_size = 8192

            with open(gz_file, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"Downloaded to {gz_file}")
    else:
        print(f"GZ file already exists at {gz_file}, skipping download.")

    # unzip
    if tsv_file.exists():
        print(f"Extracted file already exists at {tsv_file}, skipping extraction.")
    else:
        with gzip.open(gz_file, "rb") as f_in, open(tsv_file, "wb") as f_out, tqdm(unit="B", unit_scale=True, desc="Extracting") as pbar:
            for chunk in iter(lambda: f_in.read(8192), b""):
                f_out.write(chunk)
                pbar.update(len(chunk))

        print(f"Extracted contents to {tsv_file}")


def msmarco_to_csv(data_dir="./data", nrows=500_000, output_name="msmarco_english.csv"):

    data_path = Path(data_dir)
    tsv_file = data_path / "msmarco-docs.tsv"
    csv_file = data_path / output_name

    if not tsv_file.exists():
        raise FileNotFoundError(f"{tsv_file} not found. Run download_msmarco_dataset() first.")

    if csv_file.exists():
        print(f"CSV file already exists at {csv_file}, skipping conversion.")
        return

    print(f"Reading {tsv_file} (first {nrows:,} rows)...")

    df = pd.read_csv(
        tsv_file,
        sep="\t",
        header=None,
        names=["docid", "url", "title", "body"],
        nrows=nrows,
        quoting=3,           # no special handling of quotes
        on_bad_lines="skip"  # skip malformed rows
    )

    print(f"Saving {nrows:,} rows to {csv_file}...")
    df.to_csv(csv_file, index=False)

    print(f"Done! CSV saved to {csv_file}")
    return csv_file


def load_mlsam_combined(data_dir="./data"):
    """
    Follow the same process in curate_history_data.ipynb. Merge all csv files in MLSAM dataset folder into one,
    _train, _validataion, _test, all combine together.
    title must not N/A, title must not "unkown", description is summary's first 300 chars

    French, German, Spanish, Russian, and Turkish
    fr,     de,     es,      ru,          tu
    result headers: "url", "title", "description", "topic", "lang"
    """

    base = Path(data_dir) / "mlsam-multilingual-summarization-dataset"
    if not base.exists():
        raise FileNotFoundError(f"MLSAM folder not found at {base}. Run download_mlsam_dataset() first.")

    mlsam_frames = []
    files = [f for f in os.listdir(base) if f.lower().endswith(".csv")]
    print(f"Found {len(files)} MLSAM files in {base}")

    for i in range(len(files)):
        fname = files[i]
        fpath = base / fname
        df = pd.read_csv(fpath)

        df["lang"] = fname[:2]

        before = len(df)
        df = df.loc[df["topic"] != "unknown"]
        df = df.loc[~df["title"].isna()]
        after = len(df)
        print(f"{i+1:02d}/{len(files)} {fpath} -> {before} rows, kept {after}")

        df["description"] = df["summary"].fillna("").astype(str).str.slice(0, 300)

        mlsam_frames.append(df[["url", "title", "description", "topic", "lang"]])

    if not mlsam_frames:
        raise RuntimeError(f"No readable MLSAM CSVs found in {base}")

    mlsam_combined = pd.concat(mlsam_frames, axis=0, ignore_index=True)
    print("MLSAM combined rows:", len(mlsam_combined))
    return mlsam_combined


def load_msmarco_english(data_dir="./data", csv_name="msmarco_english.csv"):
    """
    language is en, title must not N/A, topic is empty string, description is body's first 300 chars
    result headers: "url", "title", "description", "topic", "lang"
    """

    path = Path(data_dir) / csv_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run msmarco_to_csv() first.")

    df = pd.read_csv(path)

    df["lang"] = "en"
    df["topic"] = ""
    df = df.loc[~df["title"].isna()].reset_index(drop=True)
    df["description"] = df["body"].fillna("").astype(str).str.slice(0, 300)

    english_df = df[["url", "title", "description", "topic", "lang"]]
    print("MS MARCO English rows:", len(english_df))
    return english_df


def merge_datasets(data_dir="./data", msmarco_csv_name="msmarco_english.csv", save_path=None):
    """
    merge MLSAM + MS MARCO (for English)
    """

    data_dir = Path(data_dir)
    if save_path is None:
        save_path = data_dir / "combined_merged.csv"
    else:
        save_path = Path(save_path)

    # If already saved, load and return
    if save_path.exists():
        combined_data = pd.read_csv(save_path, low_memory=False)
        print(f"Loaded merged CSV from {save_path}")
        print("Merged total rows:", len(combined_data))
        counts = combined_data["lang"].value_counts().sort_values(ascending=False)
        print("Language counts:")
        for lang, cnt in counts.items():
            print(f"  {lang}: {cnt:,}")
        return combined_data


    mlsam = load_mlsam_combined(data_dir=data_dir)
    msmarco_en = load_msmarco_english(data_dir=data_dir, csv_name=msmarco_csv_name)

    combined_data = pd.concat([mlsam, msmarco_en], axis=0, ignore_index=True)
    combined_data = combined_data.loc[combined_data['title'].apply(lambda title: len(title) > 5 and len(title) < 200)].reset_index(drop=True)
    combined_data['domain'] = combined_data['url'].apply(lambda x: urlparse(x).netloc.split(':')[0])
    combined_data['combined_text'] = combined_data['title'] + " " + combined_data['description'].fillna("")

    print("Merged total rows:", len(combined_data))
    counts = combined_data["lang"].value_counts().sort_values(ascending=False)
    print("Language counts:")
    for lang, cnt in counts.items():
        print(f"  {lang}: {cnt:,}")

    # save to file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    combined_data.to_csv(save_path, index=False)
    print(f"Saved merged CSV to {save_path}")

    return combined_data


def sample_by_language(combined_df: pd.DataFrame, sample_size=50_000, distribution=None, random_state=42, save_path="./data/sampled_df.csv", pkl_path="./data/sample_data_by_lang.pkl"):
    """
    """

    save_path = Path(save_path)
    pkl_path = Path(pkl_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and pkl_path.exists():
        df = pd.read_csv(save_path, low_memory=False)
        print(f"Loaded sampled CSV from {save_path}")
        print("Sampled total rows:", len(df))
        counts = df["lang"].value_counts().sort_values(ascending=False)
        print("Language counts:")
        for lang, cnt in counts.items():
            print(f"  {lang}: {cnt:,}")
        return df

    if distribution is None:
        distribution = {
            "en": 0.7 * sample_size, # 70% English
            "fr": 0.1 * sample_size, # 10% French
            "es": 0.1 * sample_size, # 10% Spanish
            "de": 0.1 * sample_size, # 10% German
            "ru": 0.1 * sample_size, # 10% Russian
        }

    sampled = []
    for lang, size in distribution.items():
        pool = combined_df[combined_df["lang"] == lang]
        n = min(int(size), len(pool))
        if n == 0:
            print(f"No rows available for language '{lang}', skipping.")
            continue
        sampled.append(pool.sample(n=n, random_state=random_state))

    if not sampled:
        raise RuntimeError("No samples could be drawn with the provided distribution.")

    result = pd.concat(sampled, axis=0, ignore_index=True)
    print("Sampled counts by lang:\n", result["lang"].value_counts())

    result.to_csv(save_path, index=False)
    print(f"Saved sampled CSV to {save_path}")

    # save pickle
    sample_data_by_lang = {}
    for lang in result["lang"].unique():
        sample_data_by_lang[lang] = result[result["lang"] == lang].reset_index(drop=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(sample_data_by_lang, f)
    print(f"Saved pickle to {pkl_path}")

    return result


if __name__ == "__main__":
    download_mlsam_dataset(data_dir="./data")
    download_msmarco_dataset(data_dir="./data")
    msmarco_to_csv(data_dir="./data", nrows=500_000, output_name="msmarco_english.csv")
    combined = merge_datasets(data_dir="./data", msmarco_csv_name="msmarco_english.csv")
    sampled_df = sample_by_language(combined_df=combined, sample_size=50_000, random_state=42, save_path="./data/sampled_df.csv", pkl_path="./data/sample_data_by_lang.pkl")

    # we need msmarco_dataset for English
    # why no tu (Turkish)? Because all of them are "unknown" topics