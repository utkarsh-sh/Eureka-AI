<div align="center">

<br/>

# ⚡ Eureka-AI

### The RAG-Powered Teaching Assistant That Knows Your Course Inside Out

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai)
[![RAG](https://img.shields.io/badge/Architecture-RAG-6C3FE8?style=for-the-badge)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge)](https://github.com/utkarsh-sh/Eureka-AI/pulls)

<br/>

> **"Where is X taught in my course?"** — Stop scrubbing through hours of video.<br/>Eureka finds the exact video and timestamp in seconds.

<br/>

</div>

---

## 🤔 The Problem It Solves

You've built a 50-video course. Students and teachers constantly ask: *"Which lesson covers CSS flexbox?"* or *"At what timestamp are HTML forms explained?"*

Manually hunting through video content is a time-sink. Search bars don't understand semantic meaning. Eureka-AI **does**.

It transcribes your course videos, converts every spoken word into vector embeddings, and uses **Retrieval-Augmented Generation (RAG)** to retrieve the most relevant video chunks for any natural-language question — then synthesizes a helpful, human-readable answer pointing to the exact video and timestamp.

---

## 💡 Core Features

### 🎯 Semantic Video Search via RAG
Eureka doesn't keyword-match — it *understands*. It embeds both your query and your course transcript chunks into the same vector space, then retrieves the top semantic matches using cosine similarity. The LLM then synthesizes those results into a clear, conversational answer.

### 🧠 Fully Local & Private
Eureka runs 100% on your own machine using [Ollama](https://ollama.ai). No API keys. No cloud data. Your course content never leaves your hardware.

### ⚡ Intelligent Model Fallback
If your preferred chat model isn't installed, Eureka automatically detects what's available in Ollama and falls back gracefully — no crash, no confusion.

### 📌 Timestamp-Aware Retrieval
Every embedded chunk carries its source metadata: video title, lesson number, `start` time, and `end` time in seconds. Eureka tells students exactly where to jump in the video.

### 🗂️ Persistent Vector Cache
Embeddings are generated once and saved to `embeddings.json` + `embeddings.joblib`. Subsequent runs skip re-embedding entirely — responses feel near-instant.

### 🎨 Polished Streamlit UI
Four switchable visual themes (Aurora, Midnight, Sunset, Forest), quick-launch question templates, a full chat interface, and a matched-sources panel that surfaces the raw chunks powering each answer.

### 🛡️ Robust Error Handling
Batched embedding with per-chunk fallback, connection error messages with actionable remediation steps, NaN chunk skipping, and graceful degradation at every stage of the pipeline.

---

## 🛠 Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | [Streamlit](https://streamlit.io) | Chat UI, themes, quick templates, sources panel |
| **Embedding Model** | [nomic-embed-text](https://ollama.ai/library/nomic-embed-text) via Ollama | Converts text chunks → dense vectors |
| **Chat Model** | [Llama 3.2](https://ollama.ai/library/llama3.2) via Ollama | Generates natural-language answers from retrieved context |
| **Vector Similarity** | [scikit-learn](https://scikit-learn.org) `cosine_similarity` | Ranks chunks by semantic relevance |
| **Data Layer** | [pandas](https://pandas.pydata.org) + [joblib](https://joblib.readthedocs.io) + [pyarrow](https://arrow.apache.org/docs/python/) | Dataframe ops, vector cache serialization |
| **Numerical Core** | [NumPy](https://numpy.org) | Embedding array stacking and indexing |
| **LLM Runtime** | [Ollama](https://ollama.ai) | Local inference server (`/api/embed`, `/api/generate`) |
| **Video Processing** | FFmpeg / custom script | Extracts audio from video files |
| **Transcription** | Whisper / ASR pipeline | Converts MP3 audio to timestamped JSON chunks |
| **Language** | Python 3.9+ | Entire backend and pipeline |

---

## 🔬 How RAG Works in Eureka

```
┌──────────────────────────────────────────────────────────────┐
│                       INDEXING PHASE                         │
│  (Run once — results cached to embeddings.json)              │
│                                                              │
│  Video Files ──► MP3 ──► Timestamped JSON Chunks             │
│                                    │                         │
│                                    ▼                         │
│                         nomic-embed-text                     │
│                         (via Ollama API)                     │
│                                    │                         │
│                                    ▼                         │
│                      Vector Embeddings DataFrame             │
│                      [title | number | start | end           │
│                       text  | embedding]                     │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                       QUERY PHASE                            │
│  (Runs on every user question)                               │
│                                                              │
│  User Question                                               │
│       │                                                      │
│       ▼                                                      │
│  nomic-embed-text ──► Query Vector                           │
│       │                                                      │
│       ▼                                                      │
│  Cosine Similarity vs. all stored chunk embeddings           │
│       │                                                      │
│       ▼                                                      │
│  Top-3 Matching Chunks (title + timestamp + transcript text) │
│       │                                                      │
│       ▼                                                      │
│  Prompt = [System context] + [Chunks] + [User Question]      │
│       │                                                      │
│       ▼                                                      │
│  Llama 3.2 ──► Human-Readable Answer with Timestamps        │
└──────────────────────────────────────────────────────────────┘
```

The key insight: by grounding the LLM in *retrieved course content*, Eureka produces answers that are **accurate, sourced, and specific** — not hallucinated guesses from general training data.

---

## 🚀 Installation Guide

### Prerequisites

Make sure you have the following installed before you begin:

- Python 3.9 or higher
- [Ollama](https://ollama.ai/download) — the local LLM runtime
- FFmpeg (for video-to-audio conversion)

---

### Step 0 — Clone the Repository

```bash
git clone https://github.com/utkarsh-sh/Eureka-AI.git
cd Eureka-AI
```

---

### Step 1 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If `embeddings.joblib` fails to load with a `pyarrow` error, simply run `pip install -r requirements.txt` again to ensure all packages are correctly installed.

---

### Step 2 — Pull Required Ollama Models

Start the Ollama server, then pull both the embedding and chat models:

```bash
# Start Ollama (if not already running as a service)
ollama serve

# In a new terminal, pull the models
ollama pull nomic-embed-text   # Embedding model
ollama pull llama3.2           # Chat model
```

> You can substitute `llama3.2` with any compatible Ollama chat model (e.g. `mistral`, `qwen2.5`, `gemma2`). Eureka will auto-detect installed models.

---

### Step 3 — Add Your Course Videos

Create a `videos/` directory at the project root and drop in your course video files:

```
Eureka-AI/
├── videos/
│   ├── lesson-01-intro-to-html.mp4
│   ├── lesson-02-css-basics.mp4
│   └── ...
```

---

### Step 4 — Convert Videos to MP3

```bash
python video_to_mp3.py
```

This extracts audio from each video and saves `.mp3` files alongside the originals (or to a dedicated output directory).

---

### Step 5 — Transcribe Audio to JSON Chunks

Run the transcription pipeline to convert MP3s into timestamped JSON files saved under `jsons/`:

```bash
python mp3_to_json.py
```

Each output JSON holds a `chunks` array, where every chunk contains the spoken text, its start/end timestamps in seconds, the video title, and lesson number.

---

### Step 6 — Build the Vector Embeddings

```bash
python preprocess_json.py
```

This reads all files in `jsons/`, sends text chunks to `nomic-embed-text` via Ollama in batches, and saves the resulting embeddings to `embeddings.json` and `embeddings.joblib`.

> You only need to run this once. Re-running skips chunks that are already cached.

**Environment variable overrides:**

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `EMBED_BATCH_SIZE` | `32` | Reduce if `/api/embed` returns HTTP 500 |
| `CHAT_MODEL` | `llama3.2` | Chat model for answer generation |

---

### Step 7 — Launch the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start asking questions about your course!

---

## 📊 Usage Examples

### Via the Streamlit Chat UI

Once the app is running, you can type naturally or click a quick-launch template:

```
💬 You:    "Where in the course are CSS selectors explained?"

🎓 Eureka: "CSS selectors are covered in detail in Lesson 7:
            'Styling with CSS' starting at 4:22 (262 seconds).
            The instructor covers type selectors, class selectors,
            and ID selectors through to about 6:10 (370 seconds).
            There's also a quick recap in Lesson 12 at the 2:00 mark."
```

```
💬 You:    "Which video explains the HTML form tag?"

🎓 Eureka: "HTML forms are introduced in Lesson 14: 'Building
            Interactive Forms', starting from the very beginning
            (0:00). The `<form>`, `<input>`, and `<label>` tags
            are all covered within the first 8 minutes."
```

---

### Via the Command Line

You can also query Eureka without the UI:

```bash
python process_incoming.py "Where is flexbox taught?"
```

Or in Python:

```python
from process_incoming import answer_query, load_embeddings_dataframe

df = load_embeddings_dataframe()

result = answer_query("Which video covers semantic HTML tags?", df=df)

print(result["response"])
# → "Semantic tags like <header>, <main>, <footer>, and <article>
#    are introduced in Lesson 5: 'Writing Clean HTML'. The key
#    discussion starts around 3:15 (195 seconds)..."

# Inspect the raw matched chunks
print(result["matches"][["title", "number", "start", "end"]])
#    title                number  start  end
#    Writing Clean HTML   5       195    312
#    HTML Structure Deep  8       42     180
#    Forms and Semantics  14      0      90
```

---

### Customizing the Embedding or Chat Model

```bash
# Use a different embedding model
EMBED_MODEL=mxbai-embed-large python preprocess_json.py

# Use a different chat model at runtime
CHAT_MODEL=mistral streamlit run app.py

# Point to a remote Ollama instance
OLLAMA_HOST=http://192.168.1.100:11434 streamlit run app.py
```

---

## 📁 Project Structure

```
Eureka-AI/
│
├── app.py                  # Streamlit frontend — chat UI, themes, quick templates
├── process_incoming.py     # RAG query engine — embedding, retrieval, LLM inference
├── preprocess_json.py      # Indexing pipeline — batch embed + cache to disk
├── video_to_mp3.py         # Video → MP3 audio extraction
│
├── jsons/                  # Timestamped transcript JSON chunks (one per video)
├── embeddings.json         # Primary portable vector cache (JSON format)
├── embeddings.joblib       # Secondary vector cache (joblib/pickle format)
│
├── prompt.txt              # Last generated RAG prompt (debug artifact)
├── response.txt            # Last LLM response (debug artifact)
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Configuration Reference

Eureka is configured entirely via environment variables — no config files to edit:

```bash
export OLLAMA_HOST=http://localhost:11434   # Ollama server URL
export EMBED_MODEL=nomic-embed-text         # Embedding model
export CHAT_MODEL=llama3.2                  # Chat/generation model
export EMBED_BATCH_SIZE=32                  # Reduce on low-memory systems
```

---

## 🗺️ Roadmap

- [ ] YouTube playlist ingestion (auto-download + transcribe)
- [ ] Multi-course support with course-selector UI
- [ ] Configurable top-K retrieval depth
- [ ] Streaming LLM responses in the chat UI
- [ ] Direct deep-link generation to video timestamps (YouTube, Vimeo)
- [ ] Reranker pass for higher-precision retrieval
- [ ] Export Q&A sessions as course study guides

---

## 🤝 Contributing

Contributions are warmly welcome! Here's how to get started:

```bash
# Fork and clone
git clone https://github.com/<your-username>/Eureka-AI.git
cd Eureka-AI

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes, then push
git push origin feature/your-feature-name
```

Then open a Pull Request against `main`. Please include a clear description of what you've changed and why.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

Built with ❤️ by [utkarsh-sh](https://github.com/utkarsh-sh)

⭐ **Star this repo if Eureka saved you from scrubbing through videos!** ⭐

</div>
