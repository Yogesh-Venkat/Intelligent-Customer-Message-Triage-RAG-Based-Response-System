# rag_chatbot_app.py
"""
Streamlit RAG app using ChromaDB (prepared by prepare_kb_chroma.py).
Enhanced with modern UI/UX design.
Usage:
    set GOOGLE_API_KEY in environment
    streamlit run rag_chatbot_app.py
"""

import os
import time
from typing import List, Tuple
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import pickle
import re
import spacy
import json
from typing import List, Any, Dict
import spacy
from spacy.util import is_package
from spacy.cli import download

MODEL = "en_core_web_sm"

if not is_package(MODEL):
    download(MODEL)

nlp = spacy.load(MODEL, disable=["parser", "ner"])




# ---------- CONFIG ----------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent
API_KEY = REPO_ROOT / "GOOGLE_API_KEY.env"
load_dotenv(API_KEY)
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "kb_chunks_01"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10
k = TOP_K
GEMINI_MODEL = "gemini-2.5-flash"
RESULTS_DIR = "results_eval"

# ---------- CUSTOM CSS ----------
def load_custom_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Card styling */
        .stCard {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        
        /* Header styling */
        h1 {
            color: #2d3748;
            font-weight: 700;
            text-align: center;
            padding: 20px 0;
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        /* Answer box styling */
        .answer-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            margin: 20px 0;
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 5px;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-label {
            font-size: 14px;
            color: #718096;
            margin-top: 5px;
        }
        
        /* Source card styling */
        .source-card {
            background: #f7fafc;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        
        .source-card:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Input field styling */
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 2px solid #e2e8f0;
            padding: 12px;
            font-size: 16px;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: #f7fafc;
        }
        
        /* Info box */
        .info-box {
            background: #ebf8ff;
            border-left: 4px solid #3182ce;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        
        /* Success box */
        .success-box {
            background: #f0fff4;
            border-left: 4px solid #38a169;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
    </style>
    """, unsafe_allow_html=True)


# ---------- EMBEDDING ----------
@st.cache_resource(show_spinner=True)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(model, texts: List[str]) -> np.ndarray:
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return emb.astype("float32")


# ---------- CHROMA ----------
@st.cache_resource(show_spinner=True)
def load_chroma_collection():
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(f"Chroma DB folder '{CHROMA_DB_DIR}' not found. Run prepare_kb_chroma.py first.")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(f"Could not load collection '{COLLECTION_NAME}'. Ensure you ran the prepare script.") from e
    return collection


def query_chroma(collection, embedder, query: str, top_k: int = TOP_K):
    q_vec = embed_texts(embedder, [query])[0].tolist()
    res = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    hits = []
    for doc, dist, meta in zip(docs, dists, metas):
        hits.append((doc, float(dist), meta))
    return hits

vectorizer = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("best_model.pkl", "rb"))



URL = r'https?://\S+|www\.\S+'
EMAIL = r'\S+@\S+'
PHONE = r'\+?\d[\d\-\s]{7,}\d'

def clean_text(text):
    text = text.lower()
    text = re.sub(URL, ' ', text)
    text = re.sub(EMAIL, ' ', text)
    text = re.sub(PHONE, ' ', text)
    text = re.sub(r'[^a-z0-9\s\?\!]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)
def preprocess(text):
    return lemmatize(clean_text(text))

# ---------- GEMINI ----------
@st.cache_resource(show_spinner=False)
def init_gemini():
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set the environment variable GOOGLE_API_KEY before running this app.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)

def build_prompt(
    query: str,
    contexts: List[str],
    metas: List[Dict] = None
) -> str:
    ctx_blocks = []

    for i, ctx in enumerate(contexts):
        label = f"Source {i+1}"
        if metas and i < len(metas):
            code = metas[i].get("heading_code", "")
            title = metas[i].get("heading_title", "") or metas[i].get("heading_line", "")
            if code or title:
                label += f" ({code} {title})"
        ctx_blocks.append(f"[{label}]\n{ctx}")

    ctx = "\n\n---\n\n".join(ctx_blocks)

    prompt = f"""
    You are a factual customer support assistant.

    Use ONLY the information in the provided context.
    You may COMBINE information from multiple sources.
    Paraphrasing is allowed.
    Do NOT use outside knowledge.

    Only say:
    "I'm not sure about that from the current knowledge base."
    IF the answer truly cannot be derived from the context.

    [CONTEXT]
    {ctx}

    [QUESTION]
    {query}

    Provide a clear, concise answer.

    If you use bullet points:
    - Put EACH bullet on a NEW LINE
    - Start each bullet with "* "
    - Do NOT place multiple bullets on the same line

    Do NOT mention the word "context" in your answer.
    """ 
    return prompt.strip()


def generate_response(model, user_query: str, contexts: List[str]) -> Tuple[str, float]:
    prompt = build_prompt(user_query, contexts)
    start = time.time()

    try:
        response = model.generate_content(prompt)
        text = response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        # ✅ Proper quota handling
        if "quota" in str(e).lower() or "exceeded" in str(e).lower():
            raise  # <-- LET THE REAL 429 PROPAGATE

        # Other unexpected errors
        raise

    latency = time.time() - start
    text = (text or "").strip()

    # Defensive cleanup: if the model didn't follow the strict format, attempt a safe fallback:
    # If model genuinely refused, keep the refusal
    if not text or "I'm not sure about that" in text:
        return text, latency

    # Otherwise trust the model output
    return text, latency



# ---------- STREAMLIT UI ----------
def main():
    st.set_page_config(
        page_title="AI Customer Support",
        layout="wide",
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Header
    st.markdown("""
        <h1>💻 AI-Powered Customer Support Assistant</h1>
        <div class='info-box'>
            <strong>💡 Smart Knowledge Base Search</strong><br>
            Ask questions about orders, products, troubleshooting, warranties, billing, and escalation procedures.
        </div>
    """, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Go to",
            ["💬 Chatbot", "📊 Evaluation Metrics"],
            index=0
        )
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>ChromaDB</div>
            <div class='metric-label'>Vector Database</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-value'>Gemini 2.5 Flash</div>
            <div class='metric-label'>AI Model</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 📊 Knowledge Base")
        st.info(f"**DB Folder:** `{CHROMA_DB_DIR}`\n\n**Collection:** `{COLLECTION_NAME}`")
        
        #if os.path.exists(CHUNKS_CSV):
            #if st.checkbox("📄 View Chunk Statistics"):
                #df = pd.read_csv(CHUNKS_CSV)
                #st.success(f"**Total Chunks:** {len(df)}")
                #st.dataframe(df.head(50), use_container_width=True)

    if page == "💬 Chatbot":


        # Load resources
        with st.spinner("🔄 Initializing AI models and knowledge base..."):
            try:
                embedder = load_embedder()
                collection = load_chroma_collection()
                gemini = init_gemini()
                st.success("✅ System ready!")
            except Exception as e:
                st.error(f"❌ Initialization error: {e}")
                return

        # Main query interface
       
        query = st.text_input(
            "🔍 What would you like to know?",
            placeholder="e.g., How do I track my order? What's your return policy?",
            key="query_input"
        )
    
    
        debug = st.checkbox("🔧 Show debug information", value=False)


        if st.button("🚀 Get Answer"):

            
            if not query.strip():
                st.warning("⚠️ Please enter a question first!")
                return
            with st.spinner("🔍 Searching knowledge base..."):
                p = preprocess(query)
                x = vectorizer.transform([p])
                pred = model.predict(x)[0]
                st.info("### Result: **Actionable**" if pred==1 else "### Result: **Non-Actionable**")
            if pred==0:
                return
            with st.spinner("🔍 Searching knowledge base..."):
                hits = query_chroma(collection, embedder, query, top_k=k)
                
                if not hits:
                    st.warning("⚠️ No relevant information found in the knowledge base.")
                    return

                contexts = [h[0] for h in hits]
                metas = [h[2] for h in hits]

            # Metrics row
            
            # Generate answer
            with st.spinner("🤖 AI is crafting your answer..."):
                answer_text, latency = generate_response(gemini, query, contexts)

            # Display answer
            st.markdown(f"""
            <div class='answer-box'>
                <h3 style='margin-top:0; color: white;'>💬 Answer</h3>
                <p style='font-size: 16px; line-height: 1.6;'>{answer_text}</p>
            </div>
            """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{len(contexts)}</div>
                    <div class='metric-label'>Sources Found</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{latency:.2f}s</div>
                    <div class='metric-label'>Response Time</div>
                </div>
                """, unsafe_allow_html=True)

            # Show sources
            st.markdown("### 📚 Information Sources")
            
            for i, (txt, dist, meta) in enumerate(hits, start=1):
                code = meta.get("heading_code", "N/A")
                title = meta.get("heading_title") or meta.get("heading_line", "Untitled")
                section = meta.get("section_header", "General")
                
                with st.expander(f"📄 Source {i}", expanded=(i==1)):
                    st.markdown(f"""
                    <div class='source-card'>
                        <strong>Section:</strong> {section}<br>
                        <strong>Relevance Score:</strong> {(1-dist)*100:.1f}%<br>
                        <strong>Distance:</strong> {dist:.4f}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("**Content:**")
                    st.text_area(f"source_{i}", txt, height=150, disabled=True, label_visibility="collapsed")

            # Debug information
            if debug:
                st.markdown("### 🔧 Debug Information")
                prov_rows = []
                for i, (txt, dist, meta) in enumerate(hits, start=1):
                    prov_rows.append({
                        "Rank": i,
                        "Code": meta.get("heading_code", ""),
                        "Title": meta.get("heading_title", "") or meta.get("heading_line", ""),
                        "Section": meta.get("section_header", ""),
                        "Distance": f"{dist:.4f}",
                        "Preview": txt[:200].replace("\n", " ") + ("..." if len(txt) > 200 else "")
                    })
                prov_df = pd.DataFrame(prov_rows)
                st.dataframe(prov_df, use_container_width=True)

        else:
            # Welcome message
            st.markdown("""
            <div class='success-box'>
                <h4 style='margin-top:0;'>👋 Welcome to your AI Assistant!</h4>
                <p>I'm here to help you with:</p>
                <ul>
                    <li>📦 Order tracking and status</li>
                    <li>🔄 Returns and refunds</li>
                    <li>🛠️ Product troubleshooting</li>
                    <li>📜 Warranty information</li>
                    <li>💳 Billing inquiries</li>
                    <li>🆘 Escalation procedures</li>
                </ul>
                <p><strong>Just type your question above and click "Get Answer"!</strong></p>
            </div>
            """, unsafe_allow_html=True)

    if page == "📊 Evaluation Metrics":
            with open("results/model_results.json") as f:
                results = json.load(f)
            df = pd.DataFrame(results).T
            st.subheader("📊 Classification Model Performance Comparison")
            st.dataframe(df)

            st.bar_chart(df["f1_score"], use_container_width=True)


            st.markdown("## 📊 RAG-Based Customer Service Chatbot Evaluation Metrics")
            summary_path = os.path.join(RESULTS_DIR, "summary_metrics.json")
            per_q_path = os.path.join(RESULTS_DIR, "per_question_metrics.csv")

            if not os.path.exists(summary_path):
                st.warning("⚠️ Evaluation results not found. Please run eval.py first.")
                st.stop()

            # -------- Load data --------
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            per_q_df = None
            if os.path.exists(per_q_path):
                per_q_df = pd.read_csv(per_q_path)

            # -------- High-level KPIs --------
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Response Accuracy",
                f"{summary.get('accuracy_overall_avg', 0)*100:.1f} %"
            )

            col2.metric(
                "Avg Context Relevance",
                f"{summary.get('avg_context_relevance', 0):.2f}"
            )

            col3.metric(
                "Avg Latency (sec)",
                f"{summary.get('avg_latency_sec', 0):.2f}"
            )

            col4.metric(
                "User Satisfaction",
                "N/A" if summary.get("avg_user_satisfaction") is None
                else f"{summary['avg_user_satisfaction']:.2f}"
            )

            st.markdown("---")

            # -------- Precision / Recall --------
            if "avg_precision" in summary or "avg_recall" in summary:
                st.markdown("### 🎯 Retrieval Quality")

                c1, c2 = st.columns(2)
                c1.metric(
                    "Average Precision",
                    f"{summary.get('avg_precision', 0):.2f}"
                )
                c2.metric(
                    "Average Recall",
                    f"{summary.get('avg_recall', 0):.2f}"
                )

            # -------- Judgement Distribution --------
            st.markdown("### 🧠 Answer Quality Breakdown")

            judgement_counts = summary.get("judgement_counts", {})
            if judgement_counts:
                chart_df = pd.DataFrame(
                    judgement_counts.items(),
                    columns=["Judgement", "Count"]
                )
                st.bar_chart(chart_df.set_index("Judgement"),  use_container_width=True)

            # -------- Per-question details --------
            if per_q_df is not None:
                st.markdown("### 📋 Per-Question Evaluation Details")

                st.dataframe(
                    per_q_df[[
                        "question",
                        "judgement",
                        "latency_sec",
                        "context_relevance_mean",
                        "expected_model_embed_cosine"
                    ]],
                    use_container_width=True
                )

                with st.expander("🔍 Full Evaluation Table"):
                    st.dataframe(per_q_df, use_container_width=True)

if __name__ == "__main__":
    main()

