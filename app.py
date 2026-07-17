import os
import time
import json
import boto3
import streamlit as st
from pathlib import Path
from datetime import datetime
from rag_core import load_oracle, run_rag as _run_rag_core, run_judge as _run_judge_core

ARCHIVE_FILE = "archives_oracle.json"

# ── Monitoring : log chaque interaction sur Backblaze B2 (compatible API S3),
#    consommé par le DAG monitoring_dag.py ──
def _get_secret(name, default=None):
    """Lit une variable soit depuis st.secrets (Streamlit Cloud), soit depuis
    les variables d'environnement (usage local avec .env)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)

S3_BUCKET = _get_secret("B2_BUCKET_NAME", "clinical-oracle-docs")
S3_LOGS_PREFIX = "logs/"
_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is None:
        try:
            endpoint_url = _get_secret("B2_ENDPOINT_URL")
            access_key = _get_secret("B2_ACCESS_KEY_ID")
            secret_key = _get_secret("B2_SECRET_ACCESS_KEY")
            if not (endpoint_url and access_key and secret_key):
                _s3_client = False  # credentials Backblaze B2 manquants -> logging désactivé silencieusement
            else:
                _s3_client = boto3.client(
                    "s3",
                    endpoint_url=endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                )
        except Exception:
            _s3_client = False
    return _s3_client

def log_interaction_to_s3(query, answer, judge_scores, sources_details):
    client = get_s3_client()
    if not client:
        return
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": query,
        "answer": answer,
        "n_sources": len(sources_details),
        **(judge_scores or {}),
    }
    key = f"{S3_LOGS_PREFIX}{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.json"
    try:
        client.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(record, ensure_ascii=False).encode("utf-8"))
    except Exception as e:
        print(f"[monitoring] Échec du log S3 (non bloquant) : {e}")

if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "k_val" not in st.session_state:
    st.session_state.k_val = 12
if "expert_overlay" not in st.session_state:
    st.session_state.expert_overlay = True
if "show_scores" not in st.session_state:
    st.session_state.show_scores = False
if "enable_judge" not in st.session_state:
    st.session_state.enable_judge = True

def save_to_archive(history):
    archive_data = []
    if os.path.exists(ARCHIVE_FILE):
        with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
            archive_data = json.load(f)
    archive_data.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "full_chat": history
    })
    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(archive_data, f, indent=4, ensure_ascii=False)

st.set_page_config(page_title="THE CLINICAL ORACLE", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;700&display=swap');
    .stApp { background-color: #000000 !important; color: #FFFFFF !important; font-family: 'JetBrains Mono', monospace; }
    [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 3px solid #0047AB !important; box-shadow: 5px 0 25px rgba(0,71,171,0.6); }
    .oracle-title { font-family: 'Orbitron', sans-serif; color: #0047AB; text-shadow: 0 0 15px #0047AB, 0 0 30px #0000FF; text-align: center; font-size: 3.5rem; font-weight: 900; letter-spacing: 8px; padding: 20px; }
    .nih-subtitle { color: #0047AB; text-align: center; font-family: 'Orbitron'; letter-spacing: 4px; font-size: 0.9rem; margin-top: -20px; margin-bottom: 30px; }
    div[data-baseweb="input"] { border: 2px solid #0047AB !important; background-color: #000000 !important; border-radius: 5px !important; }
    .chat-entry { border-left: 2px solid #0047AB; padding-left: 15px; margin-bottom: 25px; background: rgba(0,71,171,0.05); }
    .judge-box { border: 1px solid #0047AB; padding: 10px; margin-top: 10px; background: rgba(0,71,171,0.08); border-radius: 5px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
    .score-high { color: #2DC653; font-weight: bold; }
    .score-mid  { color: #FF9F1C; font-weight: bold; }
    .score-low  { color: #E63946; font-weight: bold; }
    .stMarkdown p, .stMarkdown li, .stMarkdown h3 { color: #FFFFFF !important; }
    .stProgress > div > div > div > div { background-color: #0047AB !important; box-shadow: 0 0 15px #0000FF; }
    .stButton>button { background: #000000 !important; color: #0047AB !important; border: 1px solid #0047AB !important; font-family: 'Orbitron', sans-serif; font-weight: bold; }
    .stButton>button:hover { border: 1px solid #FFFFFF !important; color: #FFFFFF !important; box-shadow: 0 0 15px #0047AB; }
    .stExpander { border: 1px solid #0047AB !important; background: rgba(0,71,171,0.05) !important; }
</style>
""", unsafe_allow_html=True)

vectorstore, llm, embeddings = st.cache_resource(show_spinner=False)(load_oracle)()

def run_rag(question: str, k: int) -> dict:
    return _run_rag_core(question, k, vectorstore, llm, embeddings)

def run_judge(question, context, answer):
    return _run_judge_core(question, context, answer, llm)

def score_class(v):
    return "score-high" if v >= 7 else ("score-mid" if v >= 4 else "score-low")

# BOOT
if not st.session_state.initialized:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("<br><br>", unsafe_allow_html=True)
        _, col, _ = st.columns([1, 2, 1])
        with col:
            if Path("logo.png").exists():
                st.image("logo.png", width=400)
            st.markdown("<div class='oracle-title'>THE CLINICAL ORACLE</div>", unsafe_allow_html=True)
            bar = st.progress(0)
            for i in range(101):
                time.sleep(0.006)
                bar.progress(i)
    st.session_state.initialized = True
    placeholder.empty()
    st.rerun()

# SIDEBAR
with st.sidebar:
    if Path("logo.png").exists():
        st.image("logo.png", use_column_width=True)  # ✅ FIXED: was use_container_width=True
    st.markdown("<h2 style='color:#0047AB;font-family:Orbitron;text-align:center;'>COMMAND CENTER</h2>", unsafe_allow_html=True)
    if st.button("🗑️ CLEAR CONVERSATION"):
        st.session_state.chat_history = []
        st.session_state.last_docs = []
        st.rerun()
    tabs = st.tabs(["SETTINGS", "ARCHIVES"])
    with tabs[0]:
        new_k = st.slider("Scan Depth (Chunks)", 4, 30, st.session_state.k_val)
        if new_k != st.session_state.k_val: st.session_state.k_val = new_k
        new_expert = st.toggle("Expert Data Overlay", value=st.session_state.expert_overlay)
        if new_expert != st.session_state.expert_overlay: st.session_state.expert_overlay = new_expert
        new_scores = st.toggle("Show Similarity Scores", value=st.session_state.show_scores)
        if new_scores != st.session_state.show_scores: st.session_state.show_scores = new_scores
        new_judge = st.toggle("⚖️ LLM-as-Judge", value=st.session_state.enable_judge)
        if new_judge != st.session_state.enable_judge: st.session_state.enable_judge = new_judge
        st.markdown(f"<small style='color:#0047AB'>k actuel : **{st.session_state.k_val}** chunks</small>", unsafe_allow_html=True)
    with tabs[1]:
        if os.path.exists(ARCHIVE_FILE):
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                history_files = json.load(f)
            for item in reversed(history_files[-5:]):
                if st.button(f"📄 {item['timestamp']}", key=item['timestamp']):
                    st.session_state.chat_history = item['full_chat']
                    st.rerun()
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#0047AB;font-family:Orbitron;font-size:0.7rem;'>MEDICAL AGENT v4.0 ELITE</div>", unsafe_allow_html=True)

# MAIN
st.markdown("<div class='oracle-title'>THE CLINICAL ORACLE</div>", unsafe_allow_html=True)
st.markdown("<div class='nih-subtitle'>NIH CLINICAL INTELLIGENCE SYSTEM</div>", unsafe_allow_html=True)

for entry in st.session_state.chat_history:
    st.markdown(f"**>> QUERY:** {entry['query']}")
    st.markdown(f"<div class='chat-entry'>{entry['response']}</div>", unsafe_allow_html=True)
    if entry.get("judge_scores"):
        j = entry["judge_scores"]
        st.markdown(f"""<div class='judge-box'>
        ⚖️ <b>LLM-AS-JUDGE</b> &nbsp;|&nbsp;
        Faithfulness: <span class='{score_class(j["faithfulness"])}'>{j["faithfulness"]}/10</span> &nbsp;|&nbsp;
        Relevance: <span class='{score_class(j["relevance"])}'>{j["relevance"]}/10</span> &nbsp;|&nbsp;
        Completeness: <span class='{score_class(j["completeness"])}'>{j["completeness"]}/10</span> &nbsp;|&nbsp;
        Citation: <span class='{score_class(j["citation"])}'>{j["citation"]}/10</span> &nbsp;|&nbsp;
        <b>Overall: <span class='{score_class(j["overall"])}'>{j["overall"]}/10</span></b><br>
        💬 {j.get("feedback", "")}
        </div>""", unsafe_allow_html=True)

with st.form(key='chat_form', clear_on_submit=True):
    query = st.text_input(">> INITIALIZE ORACLE QUERY :")
    submit_button = st.form_submit_button(label='SEND TO CORE')

if submit_button and query:
    with st.spinner("⚡ ORACLE ANALYZING..."):
        result = run_rag(query, k=st.session_state.k_val)
        judge_scores = None
        if st.session_state.enable_judge:
            judge_scores = run_judge(query, result.get("context_preview",""), result["answer"])
        st.session_state.chat_history.append({
            "query": query, "response": result["answer"],
            "query_used": result["query_used"], "sources": result["sources_details"],
            "judge_scores": judge_scores
        })
        st.session_state.last_docs = result["sources_details"]
        log_interaction_to_s3(query, result["answer"], judge_scores, result["sources_details"])
        st.rerun()

if st.session_state.chat_history:
    st.markdown("---")
    if st.session_state.expert_overlay and st.session_state.last_docs:
        st.markdown("### 📁 RAW DATA CHUNKS (LAST SCAN)")
        for i, src in enumerate(st.session_state.last_docs):
            score_text   = f" | SIM: {src['similarity']}" if st.session_state.show_scores else ""
            quality_icon = "✅" if src["quality"]=="HIGH" else ("⚠️" if src["quality"]=="MEDIUM" else "❌")
            with st.expander(f"{quality_icon} SOURCE {i+1} | {src['source']}{score_text}"):
                st.write(src["content"])
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 ARCHIVE FULL SESSION"):
            save_to_archive(st.session_state.chat_history)
            st.success("SESSION PERSISTED.")
    with c2:
        full_text = "\n\n".join([f"Q: {e['query']}\nRewritten: {e.get('query_used','')}\nA: {e['response']}" for e in st.session_state.chat_history])
        st.download_button("📄 DOWNLOAD FULL REPORT", full_text, file_name="full_report.txt")
