import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "jsons"
EMBEDDINGS_JSON = BASE_DIR / "embeddings.json"
EMBEDDINGS_JOBLIB = BASE_DIR / "embeddings.joblib"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.replace("\x00", " ").strip()


def _embed_batch(text_list):
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": EMBED_MODEL, "input": text_list},
            timeout=120,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Failed to connect to Ollama at {OLLAMA_HOST}. "
            "Start Ollama and ensure the embedding model is available "
            f"(example: `ollama pull {EMBED_MODEL}`)."
        ) from exc

    if r.status_code >= 400:
        error_body = r.text.strip()[:500]
        raise RuntimeError(
            f"Ollama embed API failed with status {r.status_code} "
            f"for batch size {len(text_list)}. Response: {error_body}"
        )

    try:
        payload = r.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Ollama embed API returned non-JSON response: {r.text[:500]}"
        ) from exc

    if "embeddings" not in payload:
        raise RuntimeError(
            f"Unexpected response from Ollama embed API: {payload}"
        )

    return payload["embeddings"]


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    all_embeddings = [None] * len(text_list)
    failed_indices = []

    for start in range(0, len(text_list), EMBED_BATCH_SIZE):
        batch = text_list[start : start + EMBED_BATCH_SIZE]
        try:
            batch_embeddings = _embed_batch(batch)
            for offset, embedding in enumerate(batch_embeddings):
                all_embeddings[start + offset] = embedding
        except RuntimeError as batch_err:
            if isinstance(batch_err.__cause__, requests.exceptions.ConnectionError):
                raise

            if len(batch) == 1:
                failed_indices.append(start)
                print(f"Skipping chunk index {start}: {batch_err}")
                continue

            # Fallback to per-item embedding when batch embedding fails.
            for offset, item in enumerate(batch):
                item_index = start + offset
                try:
                    item_embedding = _embed_batch([item])[0]
                    all_embeddings[item_index] = item_embedding
                except RuntimeError as item_err:
                    if isinstance(item_err.__cause__, requests.exceptions.ConnectionError):
                        raise

                    failed_indices.append(item_index)
                    print(f"Skipping chunk index {item_index}: {item_err}")

    return all_embeddings, failed_indices


def load_chunks():
    if not JSON_DIR.exists():
        raise FileNotFoundError(f"Missing jsons directory: {JSON_DIR}")

    records = []
    for json_file in sorted(JSON_DIR.iterdir()):
        if json_file.suffix.lower() != ".json":
            continue
        with json_file.open(encoding="utf-8") as f:
            content = json.load(f)
        print(f"Creating embeddings for {json_file.name}")
        for chunk in content.get("chunks", []):
            records.append(chunk)
    return records

def build_embeddings_dataframe():
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"No chunks found in {JSON_DIR}")

    input_texts = [normalize_text(chunk.get("text")) for chunk in chunks]
    embeddings, failed_indices = create_embedding(input_texts)
    if failed_indices:
        print(f"Skipped {len(failed_indices)} chunks due to embedding failures.")

    rows = []
    chunk_id = 0
    for index, chunk in enumerate(chunks):
        if embeddings[index] is None:
            continue
        row = dict(chunk)
        row["chunk_id"] = chunk_id
        row["embedding"] = embeddings[index]
        rows.append(row)
        chunk_id += 1

    df = pd.DataFrame.from_records(rows)
    for column in ("number", "title", "text", "embedding"):
        if column in df.columns:
            df[column] = df[column].astype(object)
    return df


def save_embeddings(df):
    records = df.to_dict(orient="records")
    with EMBEDDINGS_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f)

    try:
        joblib.dump(df, EMBEDDINGS_JOBLIB)
    except Exception as exc:
        print(f"Saved portable embeddings cache to {EMBEDDINGS_JSON.name}.")
        print(f"Legacy joblib cache could not be written: {exc}")
    else:
        print(f"Saved embeddings to {EMBEDDINGS_JSON.name} and {EMBEDDINGS_JOBLIB.name}")


def main():
    df = build_embeddings_dataframe()
    save_embeddings(df)


if __name__ == "__main__":
    main()
