# Eureka - AI-Powered Teaching Assistant

## Step 1 - Collect your videos

Move all your video files to the videos folder

## Step 0 - Install dependencies

Install the Python packages used by the scripts:

`pip install -r requirements.txt`

If `embeddings.joblib` fails to load with a missing `pyarrow` error, reinstall the
requirements so the cache can be read back correctly.

## Step 2 - Convert to mp3

Convert all the video files to mp3 by ruunning video_to_mp3

## Step 3 - Convert mp3 to json

Convert all the mp3 files to json by ruunning mp3_to_json

## Step 4 - Convert the json files to Vectors

Use the file preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

Before running `preprocess_json.py`, make sure Ollama is running and the embedding model exists:

`ollama pull nomic-embed-text`

The scripts call Ollama on `http://localhost:11434` by default. You can override this using env vars:

- `OLLAMA_HOST` (example: `http://127.0.0.1:11434`)
- `EMBED_MODEL` (default `nomic-embed-text`)
- `EMBED_BATCH_SIZE` (default `32`; lower this if `/api/embed` returns HTTP 500)

If Ollama still returns NaN-related embedding errors for specific chunks, `preprocess_json.py` will skip only those failing chunks and continue.

## Step 5 - Prompt generation and feeding to LLM

Read the joblib file and load it into the memory. Then create a relevant prompt as per the user query and feed it to the LLM

Before running `process_incoming.py`, pull at least one chat model:

`ollama pull llama3.2`

Optional env var:
- `CHAT_MODEL` (default `llama3.2`)

## Frontend

Run the interactive UI with:

`streamlit run app.py`

The app includes visual themes, quick question templates, and a chat-style interface.
