"""
DeepAgent BRD Generator — Streamlit UI
Integrates RAGAnything (multimodal RAG) with DeepAgent to generate
Business Requirements Documents from any uploaded files + user query.
"""

import streamlit as st
import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepAgent BRD Generator",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0c10;
    --surface: #12151c;
    --surface2: #1a1f2a;
    --accent: #4fffb0;
    --accent2: #7b61ff;
    --accent3: #ff6b6b;
    --text: #e8eaf0;
    --muted: #6b7280;
    --border: rgba(75, 255, 176, 0.15);
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'DM Mono', monospace;
    color: var(--text);
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0a0c10 0%, #12151c 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 2rem 0 1.5rem;
    margin-bottom: 2rem;
    text-align: center;
}

.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.main-header p {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}

.status-ready    { background: rgba(75,255,176,.12); color: var(--accent); border: 1px solid rgba(75,255,176,.3); }
.status-working  { background: rgba(123,97,255,.12); color: var(--accent2); border: 1px solid rgba(123,97,255,.3); }
.status-error    { background: rgba(255,107,107,.12); color: var(--accent3); border: 1px solid rgba(255,107,107,.3); }

/* Step counter */
.step-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
}

/* File chips */
.file-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.75rem;
    color: var(--text);
    margin: 2px;
}

/* BRD output */
.brd-output {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.8;
    white-space: pre-wrap;
    color: var(--text);
    max-height: 70vh;
    overflow-y: auto;
}

/* Overrides for Streamlit widgets */
.stFileUploader > div {
    border: 2px dashed var(--border) !important;
    background: var(--surface) !important;
    border-radius: 12px !important;
}

.stTextArea textarea,
.stTextInput input,
.stSelectbox select {
    background-color: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #00d488 100%) !important;
    color: #0a0c10 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(75,255,176,0.25) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

/* Download button */
.stDownloadButton > button {
    background: var(--surface2) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

hr { border-color: var(--border) !important; }

.stAlert { border-radius: 8px !important; }

/* Section titles */
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**🤖 DeepAgent Model**")
    model_choice = st.selectbox(
        "Model",
        [
            "google_genai:gemini-3-flash-preview",
            "google_genai:gemini-1.5-pro",
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "anthropic:claude-opus-4-6",
            "anthropic:claude-sonnet-4-6",
        ],
        label_visibility="collapsed",
    )

    st.markdown("**🔑 API Keys**")
    google_api_key = st.text_input("Google API Key", type="password",
        placeholder="AIza...",
        help="Required for Gemini models")
    openai_api_key = st.text_input("OpenAI API Key", type="password",
        placeholder="sk-...",
        help="Required for OpenAI models")
    anthropic_api_key = st.text_input("Anthropic API Key", type="password",
        placeholder="sk-ant-...",
        help="Required for Claude models")

    st.markdown("---")
    st.markdown("**🔍 RAG Settings**")

    rag_mode = st.selectbox(
        "Query Mode",
        ["hybrid", "local", "global", "naive"],
        help="hybrid = best results; local = entity-focused; global = summary-level; naive = simple search",
    )

    embedding_model = st.selectbox(
        "Embedding Model",
        ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
    )

    enable_image = st.checkbox("Process Images", value=True)
    enable_table = st.checkbox("Process Tables", value=True)
    enable_equation = st.checkbox("Process Equations", value=True)

    st.markdown("---")
    st.markdown("**📄 BRD Template**")
    brd_sections = st.multiselect(
        "Include Sections",
        [
            "Executive Summary",
            "Project Objectives",
            "Scope & Boundaries",
            "Stakeholders",
            "Functional Requirements",
            "Non-Functional Requirements",
            "Constraints & Assumptions",
            "Dependencies",
            "Risk Assessment",
            "Success Criteria",
            "Timeline & Milestones",
            "Glossary",
        ],
        default=[
            "Executive Summary",
            "Project Objectives",
            "Scope & Boundaries",
            "Functional Requirements",
            "Non-Functional Requirements",
            "Constraints & Assumptions",
            "Risk Assessment",
        ],
    )

    st.markdown("---")
    st.caption("DeepAgent BRD Generator v1.0")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📋 DeepAgent BRD Generator</h1>
  <p>Upload documents · Ask anything · Get a professional BRD</p>
</div>
""", unsafe_allow_html=True)


# ── Main layout: two columns ──────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # ── STEP 1: Upload ────────────────────────────────────────────────────────
    st.markdown('<div class="step-label">Step 01 — Upload Files</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop files here (PDF, DOCX, PPTX, XLSX, TXT, MD, PNG, JPG, …)",
        accept_multiple_files=True,
        type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls",
              "txt", "md", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        help="All formats supported by RAGAnything are accepted",
    )

    if uploaded_files:
        chips = "".join(
            f'<span class="file-chip">📄 {f.name} '
            f'<span style="color:var(--muted)">({f.size//1024}KB)</span></span>'
            for f in uploaded_files
        )
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 2: Query ─────────────────────────────────────────────────────────
    st.markdown('<div class="step-label">Step 02 — Describe Your BRD Needs</div>', unsafe_allow_html=True)

    user_query = st.text_area(
        "Query",
        placeholder=(
            "e.g. 'Generate a BRD for a customer portal based on these requirements docs. "
            "Focus on e-commerce checkout flow and payment integrations.'"
        ),
        height=120,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 3: Run ───────────────────────────────────────────────────────────
    st.markdown('<div class="step-label">Step 03 — Generate</div>', unsafe_allow_html=True)

    run_btn = st.button("🚀 Generate BRD", use_container_width=True, type="primary")


with col_right:
    st.markdown('<div class="step-label">BRD Output</div>', unsafe_allow_html=True)

    output_placeholder = st.empty()
    progress_placeholder = st.empty()
    download_placeholder = st.empty()

    # Default empty state
    if "brd_result" not in st.session_state:
        st.session_state.brd_result = None
        st.session_state.brd_status = "idle"

    if st.session_state.brd_result:
        output_placeholder.markdown(
            f'<div class="brd-output">{st.session_state.brd_result}</div>',
            unsafe_allow_html=True,
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_placeholder.download_button(
            label="⬇️ Download BRD (.md)",
            data=st.session_state.brd_result,
            file_name=f"BRD_{ts}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        output_placeholder.markdown(
            '<div class="brd-output" style="color:var(--muted);text-align:center;padding-top:6rem;">'
            '⬅️ Upload files, enter a query, and click Generate BRD'
            '</div>',
            unsafe_allow_html=True,
        )


# ── Processing logic ──────────────────────────────────────────────────────────
if run_btn:
    if not uploaded_files:
        st.error("Please upload at least one file before generating.")
        st.stop()
    if not user_query.strip():
        st.error("Please describe your BRD requirements in the query box.")
        st.stop()

    # Determine which API key to set
    if model_choice.startswith("google"):
        if not google_api_key:
            st.error("Google API Key required for Gemini models. Add it in the sidebar.")
            st.stop()
        os.environ["GOOGLE_API_KEY"] = google_api_key
    elif model_choice.startswith("openai"):
        if not openai_api_key:
            st.error("OpenAI API Key required. Add it in the sidebar.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif model_choice.startswith("anthropic"):
        if not anthropic_api_key:
            st.error("Anthropic API Key required. Add it in the sidebar.")
            st.stop()
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Also pass OpenAI key for embeddings (RAGAnything uses it by default)
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    with st.spinner(""):
        progress_placeholder.progress(0, text="Initializing pipeline…")
        output_placeholder.markdown(
            '<div class="brd-output" style="color:var(--muted)">'
            '⏳ Processing files and generating BRD…\n'
            'This may take 1–3 minutes depending on file size.'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            # Import heavy deps lazily so the UI loads fast
            from rag_pipeline import run_brd_pipeline

            brd_text = asyncio.run(
                run_brd_pipeline(
                    uploaded_files=uploaded_files,
                    user_query=user_query,
                    model=model_choice,
                    rag_mode=rag_mode,
                    brd_sections=brd_sections,
                    enable_image=enable_image,
                    enable_table=enable_table,
                    enable_equation=enable_equation,
                    openai_api_key=openai_api_key,
                    progress_cb=lambda pct, msg: progress_placeholder.progress(pct, text=msg),
                )
            )

            st.session_state.brd_result = brd_text
            st.session_state.brd_status = "done"
            progress_placeholder.progress(100, text="✅ BRD generated successfully!")
            st.rerun()

        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Pipeline error: {e}")
            with st.expander("Full traceback"):
                import traceback
                st.code(traceback.format_exc())