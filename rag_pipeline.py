"""
rag_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Integrates RAGAnything (multimodal RAG) with DeepAgent to produce a full BRD.

Flow:
  1. Save uploaded files to a temp working directory.
  2. Initialise RAGAnything with the configured LLM/embedding functions.
  3. Ingest every file via process_document_complete().
  4. Query the RAG knowledge base with the user's intent + BRD section list.
  5. Pass the enriched context to DeepAgent which writes the final BRD.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from deepagents import create_deep_agent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_func(api_key: str, base_url: Optional[str] = None, model: str = "gpt-4o-mini"):
    """Return a synchronous LLM callable for RAGAnything."""
    def llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    return llm


def _make_vision_func(api_key: str, base_url: Optional[str] = None,
                      vision_model: str = "gpt-4o", text_model: str = "gpt-4o-mini"):
    """Return a vision-capable LLM callable for RAGAnything image processing."""
    def vision(prompt, system_prompt=None, history_messages=None,
               image_data=None, messages=None, **kwargs):
        # Full messages list passed in (VLM enhanced query)
        if messages:
            return openai_complete_if_cache(
                vision_model, "",
                system_prompt=None, history_messages=[],
                messages=messages, api_key=api_key, base_url=base_url, **kwargs,
            )
        # Single image
        if image_data:
            msg_list = []
            if system_prompt:
                msg_list.append({"role": "system", "content": system_prompt})
            msg_list.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ],
            })
            return openai_complete_if_cache(
                vision_model, "",
                system_prompt=None, history_messages=[],
                messages=msg_list, api_key=api_key, base_url=base_url, **kwargs,
            )
        # Pure text fallback
        return openai_complete_if_cache(
            text_model, prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key, base_url=base_url, **kwargs,
        )
    return vision


def _make_embedding_func(api_key: str, base_url: Optional[str] = None,
                         model: str = "text-embedding-3-large"):
    """Return an EmbeddingFunc for RAGAnything."""
    return EmbeddingFunc(
        embedding_dim=3072 if "large" in model else 1536,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts, model=model, api_key=api_key, base_url=base_url,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BRD prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_brd_prompt(user_query: str, rag_context: str,
                      sections: List[str]) -> str:
    sections_list = "\n".join(f"  - {s}" for s in sections)
    return f"""
You are an expert Business Analyst. Based on the retrieved document context below
and the user's instructions, produce a complete, professional Business Requirements
Document (BRD).

USER INSTRUCTIONS:
{user_query}

DOCUMENT CONTEXT (retrieved via RAG):
{rag_context}

REQUIRED BRD SECTIONS (include ALL of these in order):
{sections_list}

FORMATTING RULES:
- Use Markdown headings (## for each section).
- Under each heading write detailed, specific content derived from the documents.
- Use bullet points for lists; tables for comparisons or structured data.
- Be concrete: name features, systems, stakeholders, and constraints explicitly.
- Avoid generic filler text; every sentence must add value.
- End with a ## Glossary of key terms found in the documents.

Now write the complete BRD:
""".strip()


BRD_SYSTEM_PROMPT = """
You are a senior Business Analyst specialised in writing comprehensive, 
actionable Business Requirements Documents (BRDs). You extract requirements 
from diverse source materials (PDFs, presentations, spreadsheets, images, tables)
and synthesise them into a single coherent document that product teams, developers,
and stakeholders can act upon immediately.

Your output must be:
- Written in clear, professional English.
- Structured with proper Markdown headings.
- Derived exclusively from the provided context; never invent requirements.
- Free of boilerplate — every line must add specific, verifiable value.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def run_brd_pipeline(
    uploaded_files,
    user_query: str,
    model: str = "google_genai:gemini-2.0-flash-preview",
    rag_mode: str = "hybrid",
    brd_sections: Optional[List[str]] = None,
    enable_image: bool = True,
    enable_table: bool = True,
    enable_equation: bool = True,
    openai_api_key: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> str:
    """
    Full pipeline:
      upload files → RAG ingestion → RAG query → DeepAgent BRD generation.

    Parameters
    ----------
    uploaded_files : list[UploadedFile]  (Streamlit UploadedFile objects)
    user_query     : str  — User's free-form BRD instructions
    model          : str  — DeepAgent model string (e.g. google_genai:gemini-…)
    rag_mode       : str  — LightRAG query mode
    brd_sections   : list[str]  — BRD sections to include
    openai_api_key : str  — OpenAI key for embeddings & RAGAnything LLM calls
    progress_cb    : optional callback(pct:int, msg:str)
    """

    def progress(pct: int, msg: str):
        if progress_cb:
            progress_cb(pct, msg)

    if brd_sections is None:
        brd_sections = [
            "Executive Summary", "Project Objectives", "Scope & Boundaries",
            "Functional Requirements", "Non-Functional Requirements",
            "Constraints & Assumptions", "Risk Assessment",
        ]

    # ── 1. Save uploaded files to temp dir ───────────────────────────────────
    progress(5, "Saving uploaded files…")
    work_dir = tempfile.mkdtemp(prefix="deep_brd_")
    upload_dir = os.path.join(work_dir, "uploads")
    rag_dir = os.path.join(work_dir, "rag_storage")
    output_dir = os.path.join(work_dir, "parsed_output")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(rag_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    saved_paths: List[str] = []
    for uf in uploaded_files:
        dest = os.path.join(upload_dir, uf.name)
        with open(dest, "wb") as f:
            f.write(uf.read())
        saved_paths.append(dest)

    try:
        # ── 2. Resolve OpenAI key (needed for RAGAnything embeddings) ─────────
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "An OpenAI API key is required for RAGAnything embeddings. "
                "Add it in the sidebar."
            )

        # ── 3. Initialise RAGAnything ─────────────────────────────────────────
        progress(10, "Initialising RAGAnything…")

        config = RAGAnythingConfig(
            working_dir=rag_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=enable_image,
            enable_table_processing=enable_table,
            enable_equation_processing=enable_equation,
        )

        llm_func = _make_llm_func(api_key)
        vision_func = _make_vision_func(api_key)
        embed_func = _make_embedding_func(api_key)

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=embed_func,
        )

        # ── 4. Ingest files ───────────────────────────────────────────────────
        total = len(saved_paths)
        for i, fpath in enumerate(saved_paths):
            fname = Path(fpath).name
            pct = 10 + int((i / total) * 55)   # 10 → 65 %
            progress(pct, f"Processing [{i+1}/{total}]: {fname}…")

            try:
                await rag.process_document_complete(
                    file_path=fpath,
                    output_dir=output_dir,
                    parse_method="auto",
                )
            except Exception as e:
                # Soft-fail: skip unsupported file but continue
                progress(pct, f"⚠️ Skipped {fname}: {e}")

        # ── 5. RAG query ──────────────────────────────────────────────────────
        progress(70, "Querying knowledge base…")

        rag_query = (
            f"Extract all requirements, objectives, constraints, stakeholders, "
            f"functional features, non-functional requirements, risks, and assumptions "
            f"relevant to: {user_query}. "
            f"Include data from tables, figures, and equations if relevant."
        )

        rag_context = await rag.aquery(rag_query, mode=rag_mode)

        # ── 6. Build BRD prompt ───────────────────────────────────────────────
        progress(80, "Building BRD prompt…")
        brd_prompt = _build_brd_prompt(user_query, rag_context, brd_sections)

        # ── 7. DeepAgent invocation ───────────────────────────────────────────
        progress(85, "DeepAgent generating BRD…")

        deep_agent = create_deep_agent(
            model=model,
            system_prompt=BRD_SYSTEM_PROMPT,
        )

        result = deep_agent.invoke(
            {"messages": [{"role": "user", "content": brd_prompt}]}
        )

        # Extract text from result
        brd_text = _extract_agent_text(result)

        progress(100, "✅ Done!")
        return brd_text

    finally:
        # Clean up temp files
        shutil.rmtree(work_dir, ignore_errors=True)


def _extract_agent_text(result) -> str:
    """
    Extract the final text from a DeepAgent invoke() result.
    The result is typically a dict with a 'messages' key containing
    AIMessage objects, or a plain string.
    """
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            # LangChain AIMessage
            if hasattr(last, "content"):
                return last.content
            # dict format
            if isinstance(last, dict):
                return last.get("content", str(last))
        # Fallback: look for 'output' key
        if "output" in result:
            return str(result["output"])

    return str(result)