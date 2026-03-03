# Learning Path Agentic System

AI agent that generates personalized learning plans for employees. Given a portal ID and a learning goal, it identifies skill gaps, retrieves relevant courses via hybrid search (BM25 + RAG), generates a prerequisite-aware plan, and validates it — all orchestrated as a LangGraph state graph with a Streamlit UI.

Built with LangGraph, OpenAI, SQLite, ChromaDB, and Streamlit.

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd learning-path-agentic-system
uv venv .venv
source .venv/bin/activate (Linux) / .venv\Scripts\Activate.ps1 (on windows)
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 3. Initialize database and load data

(resumable)
python scripts/ingest_data.py (resumable)

# 4. Run batch enrichment (llm)
python scripts/extract_skills.py (resumable)
python scripts/extract_prerequisites.py (resumable)
python scripts/embed_courses.py (resumable)

# 5. Launch the app
streamlit run app.py
```

Open http://localhost:8501, enter a portal ID and a learning goal, and click **Generate Learning Plan**.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An OpenAI API key

## Environment

```bash
cp .env.example .env
```

| Variable         | Required | Default                 | Description                                 |
| ---------------- | -------- | ----------------------- | ------------------------------------------- |
| `OPENAI_API_KEY` | Yes      | —                       | OpenAI API key for LLM calls and embeddings |
| `DB_PATH`        | No       | `data/learning_path.db` | Path to the SQLite database                 |
| `CHROMA_PATH`    | No       | `data/chroma`           | Path to ChromaDB persistent storage         |

## Setup Steps

### 1. Initialize Storage

Creates the SQLite database with 7 tables and indexes.

```bash
python scripts/init_db.py
```

### 2. Ingest Data

Loads course catalog, user profiles, and completion history from `excel_data/` into SQLite. Idempotent — safe to re-run.

```bash
python scripts/ingest_data.py
```

Source files:

- `excel_data/Course_Master_List.xlsx` — course catalog (~5600 courses)
- `excel_data/User_Master_List.xlsx` — employee profiles
- `excel_data/Completion_Data.xlsx` — course completion records

### 3. Batch Enrichment

Three scripts enrich the course catalog. They require `OPENAI_API_KEY` and make API calls. Run in this order:

```bash
# a) Extract skills from course summaries
python scripts/extract_skills.py

# b) Extract prerequisite relationships
python scripts/extract_prerequisites.py

# c) Generate embeddings for semantic search
python scripts/embed_courses.py
```

All three are idempotent (skip already-processed courses). Use `--force` to re-process:

```bash
python scripts/extract_skills.py --force
python scripts/extract_prerequisites.py --force
```

### 4. Launch the App

```bash
streamlit run app.py
```

The pipeline takes 30-60 seconds per request (multiple LLM calls).

## How It Works

```
User Input (portal_id, goal_text)
    │
    ▼
analyze_node ── extracts goal skills, loads user context, computes skill gaps
    │
    ▼ (if gaps found)
retrieve_node ── BM25 + semantic retrieval, Reciprocal Rank Fusion merge
    │
    ▼
generate_node ── LLM generates ordered learning plan
    │
    ▼
validate_node ── checks prerequisites, hours, skill coverage
    │
    ▼ (if violations found, up to 3 times)
repair_node ── LLM fixes constraint violations
    │
    ▼
finalize_node ── persists plan to DB, updates status
    │
    ▼
Output (LearningPlanResponse + ValidationResult)
```

Two-layer architecture:

- **Offline batch processors** — enrich the course catalog (skill extraction, prerequisite extraction, embedding indexing). Run once after data ingestion.
- **Online LangGraph agent** — per-request plan generation orchestrated as a state graph. Called each time a user submits a goal.

## Project Structure

```
learning-path-agentic-system/
├── app.py                          # Streamlit web UI
├── .streamlit/config.toml          # Streamlit server and theme config
├── src/                                # Source packages
│   ├── config.py                       # Environment settings
│   ├── db/                             # Database layer
│   │   ├── connection.py               # SQLite connection factory
│   │   ├── schema.py                   # Table + index DDL (7 tables)
│   │   ├── ingest.py                   # Excel loaders
│   │   ├── user_context.py             # User profile + completions loader
│   │   └── user_skills.py              # Skill inference from completions
│   ├── llm/                            # LLM integrations
│   │   ├── client.py                   # OpenAI client factory
│   │   ├── prompts.py                  # System prompts
│   │   ├── extract_skills.py           # Batch skill extraction
│   │   ├── extract_prerequisites.py    # Batch prerequisite extraction
│   │   └── extract_goal_skills.py      # Goal skill extraction
│   ├── models/
│   │   └── schemas.py                  # Pydantic models
│   ├── vectorstore/
│   │   ├── chroma.py                   # ChromaDB collection factory
│   │   └── embed_courses.py            # Batch embedding processor
│   ├── analysis/
│   │   └── skill_gap.py                # Skill gap computation
│   ├── retrieval/
│   │   ├── bm25_retrieval.py           # BM25-based retrieval
│   │   ├── semantic_retrieval.py       # Semantic retrieval (ChromaDB)
│   │   └── hybrid.py                   # RRF hybrid merger
│   ├── generation/
│   │   ├── prereq_loader.py            # Prerequisite chain loader
│   │   ├── plan_generator.py           # LLM plan generation
│   │   └── plan_persistence.py         # Plan DB persistence
│   ├── validation/
│   │   ├── checks.py                   # Deterministic validation
│   │   ├── repair.py                   # LLM repair
│   │   └── loop.py                     # Validate-repair loop
│   └── orchestrator/
│       ├── state.py                    # LangGraph state definitions
│       ├── errors.py                   # Error constants
│       ├── nodes.py                    # Graph node functions
│       └── graph.py                    # StateGraph + run_pipeline()
├── scripts/
│   ├── init_db.py                # Initialize SQLite schema
│   ├── ingest_data.py            # Load Excel data
│   ├── extract_skills.py         # LLM skill extraction
│   ├── extract_prerequisites.py  # LLM prerequisite extraction
│   └── embed_courses.py          # OpenAI embedding indexing
├── excel_data/                    # Source Excel files
├── data/                          # Runtime storage (gitignored)
├── pyproject.toml                 # Dependencies
└── .env                           # Environment variables (not committed)
```

## Troubleshooting

**"Embedding function conflict: openai vs default"** — Delete ChromaDB data and re-embed:

```bash
rm -rf data/chroma
python scripts/embed_courses.py
```

**"No module named 'orchestrator'" (or similar)** — Activate venv and install:

```bash
source .venv/bin/activate
uv sync
```

**Pipeline takes 30-60+ seconds** — Normal. Multiple sequential LLM calls (analysis, generation, validation, possibly repair).

**"No skill gaps found"** — The user already has skills matching the goal. Try a different portal ID or a more specific goal.
