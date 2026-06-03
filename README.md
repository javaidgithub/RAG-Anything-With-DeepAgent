# 📋 DeepAgent BRD Generator

A Streamlit application that combines **RAGAnything** (multimodal RAG) with
**DeepAgent** to automatically generate professional Business Requirements
Documents (BRDs) from any uploaded files.

## Architecture

```
Uploaded Files (PDF, DOCX, PPTX, XLSX, TXT, MD, PNG, JPG, …)
        │
        ▼
┌─────────────────────────────────────┐
│        RAGAnything Pipeline         │
│  ┌───────────────────────────────┐  │
│  │  MinerU Document Parser       │  │  ← Extracts text, tables,
│  │  (PDF / Office / Image / …)   │  │    equations, images
│  └──────────────┬────────────────┘  │
│                 ▼                   │
│  ┌───────────────────────────────┐  │
│  │  Multimodal Content Analysis  │  │  ← Vision LLM for images,
│  │  (Image / Table / Equation)   │  │    table interpreter, LaTeX
│  └──────────────┬────────────────┘  │
│                 ▼                   │
│  ┌───────────────────────────────┐  │
│  │  LightRAG Knowledge Graph     │  │  ← Entity extraction +
│  │  (Vector + Graph Index)       │  │    cross-modal relationships
│  └──────────────┬────────────────┘  │
└─────────────────┼───────────────────┘
                  │ hybrid / local / global query
                  ▼
         RAG Context (rich text)
                  │
                  ▼
┌─────────────────────────────────────┐
│           DeepAgent                 │
│  model: gemini / gpt-4o / claude    │
│  system: Senior Business Analyst    │
│  input:  RAG context + user query   │
│  output: Full BRD in Markdown       │
└─────────────────────────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
# Install LibreOffice for Office document support (macOS / Linux)
brew install --cask libreoffice   # macOS
sudo apt-get install libreoffice  # Ubuntu/Debian

# Install Python packages
pip install -r requirements.txt

# Verify MinerU
mineru --version
```

### 2. Configure API keys

```bash
cp env.example .env
# Edit .env and fill in your keys
```

Or enter them directly in the sidebar when the app is running.

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. **Upload files** — any combination of PDFs, Word docs, presentations,
   spreadsheets, images, markdown files, etc.
2. **Describe your needs** — e.g. *"Generate a BRD for a mobile banking app
   based on these requirements docs. Focus on authentication and payments."*
3. **Pick BRD sections** in the sidebar (defaults are pre-selected).
4. **Click Generate BRD** — the pipeline will:
   - Parse and index all files with RAGAnything
   - Query the knowledge base for relevant context
   - Feed context + query to DeepAgent
   - Render the complete BRD in the right panel
5. **Download** the Markdown BRD.

## File Format Support

| Format          | Extension(s)                  | Notes                           |
|-----------------|-------------------------------|---------------------------------|
| PDF             | .pdf                          | Text + images + tables          |
| Word            | .docx, .doc                   | Requires LibreOffice            |
| PowerPoint      | .pptx, .ppt                   | Requires LibreOffice            |
| Excel           | .xlsx, .xls                   | Requires LibreOffice            |
| Images          | .jpg, .png, .bmp, .tiff, .webp| Vision LLM analysis             |
| Text/Markdown   | .txt, .md                     | Direct text ingestion           |

## Supported Models

| Provider  | Models                                          |
|-----------|-------------------------------------------------|
| Google    | gemini-2.0-flash-preview, gemini-1.5-pro        |
| OpenAI    | gpt-4o, gpt-4o-mini                             |
| Anthropic | claude-opus-4-6, claude-sonnet-4-6              |

> **Note**: RAGAnything's internal LLM and embedding calls always use OpenAI
> (gpt-4o-mini + text-embedding-3-large). An OpenAI API key is therefore
> required regardless of which DeepAgent model you choose.

## Project Structure

```
deep-brd-agent/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAGAnything + DeepAgent integration
├── requirements.txt    # Python dependencies
├── env.example         # Environment variable template
└── README.md           # This file
```

## Configuration Options (Sidebar)

| Option           | Description                                     |
|------------------|-------------------------------------------------|
| DeepAgent Model  | LLM model for BRD generation                    |
| API Keys         | Provider keys (Google / OpenAI / Anthropic)     |
| Query Mode       | RAG retrieval strategy (hybrid recommended)     |
| Embedding Model  | OpenAI embedding model                          |
| Process Images   | Enable vision analysis for image content        |
| Process Tables   | Enable structured data interpretation           |
| Process Equations| Enable LaTeX equation parsing                   |
| BRD Sections     | Sections to include in the output document      |