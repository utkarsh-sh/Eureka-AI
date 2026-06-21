import streamlit as st

from process_incoming import (
    CHAT_MODEL,
    EMBED_MODEL,
    OLLAMA_HOST,
    answer_query,
    load_embeddings_dataframe,
)


THEMES = {
    "Aurora": {"accent": "#7c3aed", "accent_2": "#22c55e", "bg": "#0b1020"},
    "Midnight": {"accent": "#38bdf8", "accent_2": "#a855f7", "bg": "#0f172a"},
    "Sunset": {"accent": "#fb7185", "accent_2": "#f59e0b", "bg": "#1f1147"},
    "Forest": {"accent": "#34d399", "accent_2": "#22c55e", "bg": "#071a16"},
}

QUERY_PRESETS = [
    ("Where is form taught in HTML?", "Where is form taught in HTML course?"),
    ("Which video explains CSS selectors?", "Which video explains CSS selectors?"),
    ("Find the part about semantic tags.", "Find the part about semantic tags."),
    ("Where are tables explained?", "Where are tables explained?"),
]


def inject_styles(theme_name):
    theme = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(255,255,255,0.08), transparent 30%),
                radial-gradient(circle at bottom right, rgba(255,255,255,0.04), transparent 25%),
                linear-gradient(135deg, {theme["bg"]} 0%, #050816 100%);
            color: #f8fafc;
        }}
        .hero-card, .feature-card, .template-box, .info-card {{
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 24px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(14px);
        }}
        .hero-title {{
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1.0;
            margin-bottom: 0.4rem;
            letter-spacing: -0.02em;
        }}
        .hero-subtitle {{
            color: rgba(226, 232, 240, 0.82);
            font-size: 1rem;
            margin-bottom: 0.85rem;
        }}
        .hero-badge {{
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            color: #f8fafc;
            background: linear-gradient(135deg, {theme["accent"]}, {theme["accent_2"]});
            margin-bottom: 0.65rem;
        }}
        .pill {{
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            color: #f8fafc;
            border: 1px solid rgba(255,255,255,0.09);
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            font-size: 0.82rem;
        }}
        .template-title {{
            font-size: 1.03rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .template-desc {{
            color: rgba(226,232,240,0.78);
            font-size: 0.9rem;
            margin-bottom: 0.7rem;
        }}
        .stButton>button {{
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.14);
            background: linear-gradient(135deg, {theme["accent"]}, {theme["accent_2"]});
            color: white;
            font-weight: 700;
            width: 100%;
            transition: transform 180ms ease, box-shadow 180ms ease;
        }}
        .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
        }}
        div[data-testid="stChatMessage"] {{
            background: transparent;
        }}
        div[data-testid="stChatMessageContent"] {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 0.75rem 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_catalog():
    df = load_embeddings_dataframe()
    stats = {
        "chunks": int(len(df)),
        "videos": int(df["title"].nunique()) if "title" in df.columns else 0,
        "lessons": int(df["number"].nunique()) if "number" in df.columns else 0,
        "has_embeddings": int(df["embedding"].notna().sum()) if "embedding" in df.columns else 0,
    }
    return df, stats


def queue_question(question):
    st.session_state.pending_question = question


def process_pending_question(df):
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Finding the best match and generating an answer..."):
        try:
            result = answer_query(question, df=df)
        except RuntimeError as exc:
            st.session_state.messages.append({"role": "assistant", "content": str(exc)})
            st.session_state.last_matches = []
            return

    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    st.session_state.last_matches = result["matches"][
        ["title", "number", "start", "end", "text"]
    ].to_dict(orient="records")


APP_NAME = "Eureka - AI-Powered Teaching Assistant"

st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I’m Eureka. I can help you find the exact video and timestamp for any topic in the course.",
        }
    ]
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "last_matches" not in st.session_state:
    st.session_state.last_matches = []

theme_name = st.sidebar.radio("Visual theme", list(THEMES.keys()), index=0)
inject_styles(theme_name)

df, stats = load_catalog()

if st.session_state.pending_question:
    process_pending_question(df)

with st.sidebar:
    st.markdown(f"## {APP_NAME}")
    st.caption("Interactive course assistant with a polished chat UI.")
    st.markdown(f"<span class='pill'>Host: {OLLAMA_HOST}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill'>Embed: {EMBED_MODEL}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='pill'>Chat: {CHAT_MODEL}</span>", unsafe_allow_html=True)
    st.divider()
    st.metric("Videos", stats["videos"])
    st.metric("Lessons", stats["lessons"])
    st.metric("Chunks", stats["chunks"])
    st.metric("Loaded vectors", stats["has_embeddings"])

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">⚡ Fast • Friendly • Focused</div>
        <div class="hero-title">Eureka</div>
        <div class="hero-subtitle">
            Ask naturally about the course and jump straight to the video, topic, and timestamp you need.
        </div>
        <div class="pill">Cached course index</div>
        <div class="pill">Instant topic search</div>
        <div class="pill">Course-aware answers</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown("### Ask your question")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask about a topic, timestamp, or lesson...")
    if user_query:
        st.session_state.pending_question = user_query
        st.rerun()

with right:
    st.markdown("### Quick templates")
    st.caption("Tap a card to launch a ready-made question.")
    for label, question in QUERY_PRESETS:
        if st.button(label, key=f"preset-{label}"):
            queue_question(question)
            st.rerun()

    st.write("")
    st.markdown(
        """
        <div class="feature-card">
            <strong>Why it feels smoother now</strong><br/>
            The course index is loaded once and reused, so follow-up questions feel much faster.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
tab1, tab2 = st.tabs(["Featured templates", "Matched sources"])

with tab1:
    cols = st.columns(2, gap="large")
    template_data = [
        ("Minimal", "Bright, calm, and focused on readability."),
        ("Glass", "Soft glow cards for a premium look."),
        ("Neon", "High-contrast gradients with a modern vibe."),
        ("Ocean", "Cool tones for an educational dashboard."),
    ]
    for index, (name, description) in enumerate(template_data):
        with cols[index % 2]:
            st.markdown(
                f"""
                <div class="template-box">
                    <div class="template-title">{name}</div>
                    <div class="template-desc">{description}</div>
                    <div class="pill">Interactive</div>
                    <div class="pill">Responsive</div>
                    <div class="pill">Aesthetic</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab2:
    if st.session_state.last_matches:
        st.dataframe(st.session_state.last_matches, use_container_width=True, hide_index=True)
    else:
        st.info("Matched sources will appear here after you ask a question.")
