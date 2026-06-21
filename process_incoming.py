import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

from preprocess_json import (
    EMBEDDINGS_JSON,
    EMBEDDINGS_JOBLIB,
    build_embeddings_dataframe,
    save_embeddings,
)

BASE_DIR = Path(__file__).resolve().parent

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2")


def get_installed_models():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=30)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Failed to connect to Ollama at {OLLAMA_HOST}. "
            "Start Ollama before running this script."
        ) from exc

    if r.status_code >= 400:
        raise RuntimeError(
            f"Ollama tags API failed with status {r.status_code}: {r.text[:500]}"
        )

    try:
        payload = r.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Ollama tags API returned non-JSON response: {r.text[:500]}"
        ) from exc

    models = payload.get("models", [])
    return [m.get("name") for m in models if m.get("name")]


def resolve_chat_model():
    installed_models = get_installed_models()
    if CHAT_MODEL in installed_models:
        return CHAT_MODEL

    preferred_chat_models = [
        "llama3.2",
        "llama3.1",
        "llama3",
        "qwen2.5",
        "mistral",
        "gemma2",
    ]

    for model_name in preferred_chat_models:
        for installed in installed_models:
            if installed.startswith(model_name):
                print(
                    f"Chat model '{CHAT_MODEL}' not found. Using installed model '{installed}' instead."
                )
                return installed

    non_embed_models = [m for m in installed_models if "embed" not in m.lower()]
    if non_embed_models:
        fallback = non_embed_models[0]
        print(
            f"Chat model '{CHAT_MODEL}' not found. Using installed model '{fallback}' instead."
        )
        return fallback

    raise RuntimeError(
        "No chat model found in Ollama. Pull one first "
        "(example: `ollama pull llama3.2`) and rerun."
    )


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
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
        raise RuntimeError(
            f"Ollama embed API failed with status {r.status_code}: {r.text[:500]}"
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

    embedding = payload["embeddings"]
    return embedding

def inference(prompt):
    chat_model = resolve_chat_model()
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                # "model": "deepseek-r1",
                "model": chat_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Failed to connect to Ollama at {OLLAMA_HOST}. "
            f"Start Ollama and ensure chat model `{chat_model}` is available."
        ) from exc

    if r.status_code >= 400:
        raise RuntimeError(
            f"Ollama generate API failed with status {r.status_code}: {r.text[:500]}"
        )

    try:
        response = r.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Ollama generate API returned non-JSON response: {r.text[:500]}"
        ) from exc

   # print(response)
    return response


@lru_cache(maxsize=1)
def load_embeddings_dataframe():
    if EMBEDDINGS_JSON.exists():
        with EMBEDDINGS_JSON.open(encoding="utf-8") as f:
            records = json.load(f)
        return pd.DataFrame.from_records(records)

    if EMBEDDINGS_JOBLIB.exists():
        try:
            cached = joblib.load(EMBEDDINGS_JOBLIB)
            if isinstance(cached, pd.DataFrame):
                return cached
            return pd.DataFrame(cached)
        except Exception as exc:
            print(
                "Existing embeddings.joblib could not be loaded, so the cache will be rebuilt."
            )
            print(f"Reason: {exc}")

    df = build_embeddings_dataframe()
    save_embeddings(df)
    return df


def get_top_matches(incoming_query, top_results=3, df=None):
    if df is None:
        df = load_embeddings_dataframe()
    if df.empty:
        raise RuntimeError("No embedded chunks available. Run preprocess_json.py first.")

    question_embedding = create_embedding([incoming_query])[0]
    similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()
    max_indx = similarities.argsort()[::-1][0:top_results]
    return df.loc[max_indx], similarities[max_indx]


def build_prompt(incoming_query, matches):
    return f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{matches[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''


def answer_query(incoming_query, top_results=3, persist=True, df=None):
    matches, scores = get_top_matches(incoming_query, top_results=top_results, df=df)
    prompt = build_prompt(incoming_query, matches)

    if persist:
        with open(BASE_DIR / "prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

    response = inference(prompt)["response"]

    if persist:
        with open(BASE_DIR / "response.txt", "w", encoding="utf-8") as f:
            f.write(response)

    return {
        "response": response,
        "prompt": prompt,
        "matches": matches,
        "scores": scores,
    }


def run_query(incoming_query):
    result = answer_query(incoming_query)
    print(result["response"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="Question to ask about the course content")
    args = parser.parse_args()

    incoming_query = args.query or input("Ask a Question: ")
    run_query(incoming_query)


if __name__ == "__main__":
    main()
