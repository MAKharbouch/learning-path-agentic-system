# Learning Path Agentic System -- Technical Overview

## What It Does

An AI-powered system that generates **personalized learning plans** for employees. Given an employee ID and a free-text learning goal (e.g., "I want to learn cloud architecture"), it:

1. Looks up who the employee is and what they've already completed
2. Uses GPT-4o to figure out what skills the goal requires
3. Compares those against skills the employee already has (inferred from completions)
4. Searches the course catalog (5,665 courses) using both keyword and semantic search
5. Generates a sequenced, time-budgeted learning plan
6. Validates the plan against 5 constraint checks, auto-repairing up to 3 times
7. Presents the result in a Streamlit web UI

---

## Architecture: Two Layers

```
+------------------------------------------------------------------+
|                        OFFLINE LAYER                              |
|                     (run once after data load)                    |
|                                                                   |
|  Excel Files ──> SQLite (courses, users, completions)             |
|       │                                                           |
|       ├──> GPT-4o extracts skills per course ──> course_skills    |
|       ├──> GPT-4o extracts prerequisites ──> course_prerequisites |
|       └──> OpenAI embeddings ──> ChromaDB vector index            |
+------------------------------------------------------------------+
                              │
                              ▼
+------------------------------------------------------------------+
|                        ONLINE LAYER                               |
|              (per-request, ~30-60 seconds via LangGraph)          |
|                                                                   |
|  User Input (portal_id + goal_text)                               |
|       │                                                           |
|       ├──> Analyze: user context + goal skills + skill gaps       |
|       ├──> Retrieve: SQL matching + semantic search + RRF fusion  |
|       ├──> Generate: GPT-4o sequenced learning plan               |
|       ├──> Validate: 5 deterministic checks                       |
|       ├──> Repair (0-3x): GPT-4o fixes violations                |
|       └──> Finalize: persist status to DB                         |
|                                                                   |
|  Streamlit UI displays the result                                 |
+------------------------------------------------------------------+
```

---

## Data

### Source Files (`excel_data/`)

| File | Records | Content |
|------|---------|---------|
| `Course_Master_List.xlsx` | 5,665 | Course catalog (ID, name, summary, category) |
| `User_Master_List.xlsx` | 5,062 | Employee profiles (portal ID, grade, country, practice area, training goal hours) |
| `Completion_Data.xlsx` | 85,395 | Completion records (who completed what course, when) |

### SQLite Database (`data/learning_path.db`) -- 7 Tables

| Table | Rows | Purpose |
|-------|------|---------|
| `courses` | 5,665 | Course catalog with LLM-processing flags |
| `users` | 5,062 | Employee profiles |
| `course_completions` | 85,395 | Who completed what |
| `course_skills` | 4,203 | LLM-extracted skills per course (1-5 per course) |
| `course_prerequisites` | 271 | LLM-extracted prerequisites per course (0-5 per course) |
| `learning_path_runs` | varies | Generated plan headers (UUID, status: draft/validated/failed) |
| `learning_path_courses` | varies | Plan line items (ordered courses with duration + reasoning) |

### ChromaDB Vector Store (`data/chroma/`)

- Collection: `course_embeddings`
- Model: `text-embedding-3-small` (OpenAI)
- Documents: `"{course_name}. {summary_text}"` for each course with a summary
- Used for semantic similarity search during retrieval

---

## Offline Pipeline (Scripts)

Run once in order. All are idempotent (safe to re-run).

```
scripts/init_db.py              # Create tables + indexes (no LLM)
       │
scripts/ingest_data.py          # Excel → SQLite (no LLM)
       │
scripts/extract_skills.py       # GPT-4o extracts skills per course
       │
scripts/extract_prerequisites.py # GPT-4o extracts prerequisites per course
       │
scripts/embed_courses.py        # OpenAI embeds courses into ChromaDB
```

Each LLM script processes courses one at a time with per-row atomic commits. A `processed_by_llm` / `prereqs_extracted` flag on the `courses` table tracks progress, making the scripts **resumable** -- if interrupted, they pick up where they left off.

---

## Online Pipeline (LangGraph)

The pipeline is a compiled LangGraph `StateGraph` with 6 nodes and conditional routing.

### Graph Visualization

```
__start__
    │
    ▼
 analyze ──error/no gaps──> __end__
    │
    ▼
 retrieve ──error──> __end__
    │
    ▼
 generate ──error──> __end__
    │
    ▼
 validate ──valid──> finalize ──> __end__
    │                    ▲
    │ invalid            │
    ▼                    │
  repair ──error──> __end__
    │
    └──> validate (loop, max 3 repairs)
```

### Node Details

#### 1. Analyze Node (Phase 5: Skill Gap Analysis)

Three steps:

- **Load User Context** -- SQL queries to get profile, completed courses, and inferred skills. Skills are inferred by joining completions with `course_skills` (no separate LLM call, always current).
- **Extract Goal Skills** -- GPT-4o with structured output converts the free-text goal into a list of `RequiredSkill` objects (skill name, level, importance).
- **Compute Gaps** -- Pure comparison: "missing" gaps (user lacks the skill entirely) and "weak" gaps (user has the skill but at a lower level). Sorted by priority.

If there are zero gaps, the pipeline ends early (the user already knows everything).

#### 2. Retrieve Node (Phase 6: Hybrid Course Retrieval)

Two retrieval legs fused together:

- **SQL Retrieval** -- Queries `course_skills` for exact/substring skill name matches against gap skills. Ranks by match count then average confidence. Excludes already-completed courses.
- **Semantic Retrieval** -- Builds a query string from all gaps (`"cloud architecture advanced level. kubernetes intermediate level."`), runs a ChromaDB nearest-neighbor search. Excludes completed courses.
- **Reciprocal Rank Fusion (RRF)** -- Merges both ranked lists using `score = sum(1 / (60 + rank))`. Each candidate gets a source tag: "sql", "semantic", or "both". Capped at 30 candidates.

#### 3. Generate Node (Phase 7: Plan Generation)

- Loads prerequisites for all candidate courses from the DB.
- Builds a detailed prompt including: skill gaps, candidate courses (with summaries truncated to 300 chars, skills, prerequisites), and budget hours.
- GPT-4o with structured output produces a `LearningPlanResponse`: ordered list of `PlannedCourse` objects (course_id, order, phase, duration_hours, targeted_skills, reasoning) plus a total_hours and skill_coverage summary.
- **Hallucination guard**: any course ID not in the candidate list is filtered out.
- Plan is persisted to SQLite as a "draft".

#### 4. Validate Node (Phase 8: 5 Deterministic Checks)

All pure functions, no LLM calls:

| Check | Severity | What It Verifies |
|-------|----------|-----------------|
| Prerequisite ordering | error | Prerequisite courses appear before courses that need them |
| Hour budget | error/warning | Total hours within user's training goal; sum matches declared total |
| Skill coverage | warning | High-priority gaps (importance >= 0.5) covered by at least one course |
| Duplicates | error | No repeated course IDs |
| Course order | error | Sequential numbering starting from 1 |

Returns `ValidationResult(is_valid: bool, violations: list)`.

#### 5. Repair Node (Phase 8: LLM Repair Loop)

If validation fails:

- Violations are formatted as "ERRORS (must fix)" and "WARNINGS (should fix)".
- GPT-4o receives the current plan, violations, and original context, then produces a corrected plan.
- Hallucination guard runs again.
- Loops back to validate. Maximum 3 repair iterations.

#### 6. Finalize Node

Updates the plan's status in SQLite from "draft" to "validated" or "failed". Deliberately lenient -- errors are logged but never crash the pipeline.

### LLM Calls Per Request

| Node | Model | Purpose | Count |
|------|-------|---------|-------|
| Analyze | gpt-4o | Extract required skills from goal text | 1 |
| Generate | gpt-4o | Generate sequenced learning plan | 1 |
| Repair | gpt-4o | Fix validation violations | 0-3 |
| **Total** | | | **2-5** |

All LLM calls use Pydantic structured output (via LangChain `with_structured_output()`) for guaranteed schema compliance.

### Error Handling

- Every node wraps its logic in try/except. Errors are stored in state (`error`, `error_node`) rather than crashing.
- Conditional routing checks for errors at every transition and routes to `__end__` if found.
- LLM-calling nodes have a `RetryPolicy(max_attempts=3)` for transient API failures.
- The OpenAI client itself has `max_retries=3` with exponential backoff.

---

## Streamlit UI (`app.py`)

Single-page app with three sections:

1. **Input Form** -- Portal ID (number) + Learning Goal (text) + Submit button
2. **Pipeline Execution** -- Calls `run_pipeline()` with a spinner/status indicator
3. **Result Display**:
   - Three metrics: course count, total hours, validation status
   - Validation violations (if any) as color-coded errors/warnings
   - Skill coverage summary
   - Expandable course cards showing: order, phase, duration, targeted skills, reasoning

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Package Manager | uv |
| LLM | OpenAI GPT-4o (via LangChain) |
| Embeddings | OpenAI text-embedding-3-small |
| Orchestration | LangGraph (StateGraph) |
| Vector Store | ChromaDB (persistent) |
| Database | SQLite (WAL mode, foreign keys) |
| Data Validation | Pydantic v2 |
| Web UI | Streamlit |
| Data Loading | pandas + openpyxl |

---

## Project Structure

```
├── app.py                              # Streamlit entry point
├── pyproject.toml                      # Dependencies + draw-graph script
├── .env.example                        # Environment template
├── excel_data/                         # Source Excel files
├── data/                               # Runtime storage (SQLite + ChromaDB)
├── scripts/
│   ├── init_db.py                      # Create schema
│   ├── ingest_data.py                  # Load Excel → SQLite
│   ├── extract_skills.py              # Batch skill extraction (GPT-4o)
│   ├── extract_prerequisites.py       # Batch prerequisite extraction (GPT-4o)
│   └── embed_courses.py              # Batch embedding (text-embedding-3-small)
└── src/                                # Source packages
    ├── config.py                       # Environment settings
    ├── db/
    │   ├── connection.py               # SQLite connection factory
    │   ├── schema.py                   # 7 tables + 7 indexes
    │   ├── ingest.py                   # Excel-to-SQLite pipeline
    │   ├── user_context.py             # User profile + completions loader
    │   └── user_skills.py              # Skill inference from completions
    ├── llm/
    │   ├── client.py                   # OpenAI client factory
    │   ├── prompts.py                  # All system prompts
    │   ├── extract_skills.py           # Batch skill extraction logic
    │   ├── extract_prerequisites.py    # Batch prerequisite extraction logic
    │   └── extract_goal_skills.py      # Goal → required skills (online)
    ├── models/
    │   └── schemas.py                  # 17 Pydantic models
    ├── vectorstore/
    │   ├── chroma.py                   # ChromaDB collection factory
    │   └── embed_courses.py            # Batch embedding processor
    ├── analysis/
    │   └── skill_gap.py                # Gap computation engine
    ├── retrieval/
│   ├── bm25_retrieval.py           # BM25 skill-matching retrieval
│   ├── semantic_retrieval.py       # ChromaDB embedding retrieval
│   └── hybrid.py                   # Reciprocal Rank Fusion merger
    ├── generation/
    │   ├── prereq_loader.py            # Prerequisite chain loader
    │   ├── plan_generator.py           # LLM plan generation
    │   └── plan_persistence.py         # Plan DB persistence
    ├── validation/
    │   ├── checks.py                   # 5 deterministic validators
    │   ├── repair.py                   # LLM-driven plan repair
    │   └── loop.py                     # Validate-repair loop
    └── orchestrator/
        ├── state.py                    # LangGraph TypedDict state
        ├── errors.py                   # Constants (MAX_REPAIRS=3)
        ├── nodes.py                    # 5 node wrapper functions
        ├── graph.py                    # StateGraph builder + run_pipeline()
        └── draw_graph.py              # Graph PNG visualization
```

---

## Key Design Decisions

1. **Two-layer separation** -- Expensive batch enrichment (skills, prerequisites, embeddings) runs once offline. The online pipeline only makes 2-5 LLM calls per request, keeping response time to ~30-60 seconds.

2. **Hybrid retrieval with RRF** -- Neither keyword search nor semantic search alone is sufficient. SQL retrieval catches exact skill matches; semantic retrieval catches conceptually similar courses. Reciprocal Rank Fusion combines both without requiring score normalization.

3. **Structured output everywhere** -- All LLM calls return Pydantic models via structured output (JSON schema enforcement). This eliminates parsing failures and guarantees type safety.

4. **Deterministic validation + LLM repair** -- The validation checks are pure functions (no LLM, no randomness). Only the repair step uses an LLM. This gives predictable, auditable constraint enforcement with intelligent auto-correction.

5. **Hallucination guards** -- After both generation and repair, any course ID not in the candidate list is filtered out. The LLM cannot inject courses that weren't retrieved.

6. **Skill inference without extra tables** -- User skills are derived on-the-fly by joining completions with course_skills. No separate "user_skills" table to maintain or sync. Always reflects current data.

7. **Idempotent everything** -- Every script, every DB operation, every vector upsert is safe to re-run. Scripts are resumable via processing flags on the courses table.

8. **Error containment in LangGraph** -- Nodes catch exceptions and store them in state rather than crashing. Routing functions check for errors at every transition. The pipeline degrades gracefully.
