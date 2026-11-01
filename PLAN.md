# Intelligent Parking Space Reservation Chatbot — Implementation Plan

## Project Overview

An intelligent chatbot that answers parking facility questions, retrieves live pricing and availability from a database, collects reservation details conversationally, enforces PII guardrails, and evaluates its own retrieval quality.

Built with: Python 3.11+, LangChain, LangGraph, Groq API, HuggingFace embeddings, ChromaDB, SQLite, Microsoft Presidio.

---

## Architecture

### Information Flow

```
User input
    │
    ▼
┌─────────────────────────────────────────────┐
│              LangGraph Workflow              │
│                                             │
│  ┌──────────┐     ┌────────────────────┐    │
│  │ Guardrail│────▶│   Intent Router    │    │
│  │  (PII +  │     │  (classify input)  │    │
│  │ injection│     └────────┬───────────┘    │
│  │  check)  │              │                │
│  └──────────┘    ┌─────────┼─────────┐      │
│                  ▼         ▼         ▼      │
│            ┌──────────┐ ┌──────┐ ┌──────┐  │
│            │ RAG Node │ │ SQL  │ │Reserv│  │
│            │(ChromaDB)│ │ Node │ │ Node │  │
│            └──────────┘ └──────┘ └──────┘  │
│                  │         │         │      │
│                  └─────────┴─────────┘      │
│                            │                │
│                     ┌──────▼──────┐         │
│                     │  LLM Node   │         │
│                     │  (Groq)     │         │
│                     └─────────────┘         │
└─────────────────────────────────────────────┘
    │
    ▼
Response to user
```

### Component Map

| Module | Responsibility |
|---|---|
| `src/config/` | Settings, env vars, logging setup |
| `src/ingestion/` | Load, clean, chunk, embed docs → ChromaDB |
| `src/rag/` | Retriever, prompt builder, RAG chain |
| `src/database/` | SQLite schema, seed data, query functions |
| `src/chatbot/` | LangGraph graph, state, nodes, intent routing |
| `src/guardrails/` | Presidio PII detection, injection checks, custom validators |
| `src/evaluation/` | Eval dataset, metrics (P@K, R@K, MRR, latency), report |
| `data/documents/` | Raw `.txt`/`.md` parking knowledge files |
| `data/evaluation/` | Eval dataset JSON |
| `tests/` | Pytest test files per module |
| `storage/` | Persistent ChromaDB + SQLite (gitignored) |

### Key Architectural Decisions

1. **LangGraph state machine** — `ConversationState` TypedDict carries message history, current intent, reservation form fields, validation errors, and routing flags. New nodes for Stages 2–4 slot in without restructuring.

2. **Hybrid routing** — Intent classification is a conditional edge inside LangGraph. Static questions → RAG node (ChromaDB). Dynamic questions (price, hours, availability) → SQL node (SQLite). Reservation intent → Reservation collection node. The LLM never invents prices or hours.

3. **ChromaDB with document hashing** — SHA-256 hash of each source file is compared against stored metadata before re-ingesting, avoiding redundant embedding work.

4. **PII guardrail position** — Applied at input (before routing) and at output (before returning to user). Reservation data never enters ChromaDB — it goes only to SQLite.

5. **Groq LLM abstraction** — Thin `LLMProvider` class; model name is configurable via `.env`. Future model swaps require only a config change.

---

## Project Structure

```
parking-chatbot/
│
├── data/
│   ├── documents/          # Raw parking knowledge files (.txt / .md)
│   └── evaluation/         # Eval dataset JSON
│
├── storage/                # Gitignored — generated at runtime
│   ├── chroma/             # Persistent ChromaDB
│   └── parking.db          # SQLite database
│
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py     # Env vars, paths, logging config
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py       # Document loading and cleaning
│   │   ├── chunker.py      # Text splitting
│   │   └── embedder.py     # Embedding + ChromaDB persistence
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py    # ChromaDB retriever setup
│   │   ├── prompt.py       # Prompt templates
│   │   └── chain.py        # RAG chain (retriever + LLM)
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.py       # SQLite table creation + seed data
│   │   └── queries.py      # Parameterised query functions
│   │
│   ├── chatbot/
│   │   ├── __init__.py
│   │   ├── state.py        # ConversationState TypedDict
│   │   ├── nodes.py        # LangGraph node functions
│   │   ├── router.py       # Intent classification + conditional edges
│   │   └── graph.py        # Graph assembly and compilation
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── pii.py          # Presidio analyser + anonymiser
│   │   ├── validators.py   # Vehicle reg, date/time, name validators
│   │   └── injection.py    # Prompt injection detection
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── dataset.py      # Eval dataset loading
│       ├── metrics.py      # P@K, R@K, MRR, latency
│       └── report.py       # Report generation
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_rag.py
│   ├── test_database.py
│   ├── test_chatbot.py
│   ├── test_guardrails.py
│   └── test_evaluation.py
│
├── reports/                # Generated evaluation reports
│
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md               # Full setup and usage guide
├── PLAN.md                 # This file
└── main.py                 # Entry point (CLI chatbot + ingestion + eval)
```

---

## Stage 1 Milestones

| # | Milestone | Deliverables | Status |
|---|---|---|---|
| **M1** | Project scaffold & config | Directory structure, `requirements.txt`, `.env.example`, `settings.py`, logging, `.gitignore` | **Done** |
| **M2** | Knowledge documents & ingestion | Sample parking docs, loader, chunker, embedder, ChromaDB persistence, hash-based re-ingestion guard | **Done** |
| **M3** | SQLite database | Schema + seed data (prices, hours, availability), parameterised query functions | **Done** |
| **M4** | RAG pipeline | Retriever, prompt template, Groq LLM chain, grounded responses, fallback when context is insufficient | **Done** |
| **M5** | LangGraph chatbot | `ConversationState`, nodes, conditional routing, reservation collector, interactive CLI | **Done** |
| **M6** | Guardrails | Presidio PII detection, custom vehicle reg validator, injection detection, output scrubbing | **Done** |
| **M7** | Evaluation | Eval dataset, P@K / R@K / MRR / latency metrics, report generator | **Done** |
| **M8** | Tests & README | Full pytest suite for all modules, README with setup/usage/examples | **Done** |

Each milestone ends with a testable checkpoint. The next milestone does not begin until the current one is verified.

---

## Dependencies

```
# LLM / LangChain ecosystem
langchain==0.3.*
langchain-community==0.3.*
langchain-groq==0.2.*
langgraph==0.2.*

# Embeddings
sentence-transformers==3.*
langchain-huggingface==0.1.*

# Vector store
chromadb==0.5.*

# PII detection
presidio-analyzer==2.2.*
presidio-anonymizer==2.2.*
spacy==3.7.*
# Post-install: python -m spacy download en_core_web_lg

# Env / config
python-dotenv==1.0.*

# Evaluation (optional)
ragas==0.1.*

# Testing
pytest==8.*
pytest-mock==3.*

# Utilities
pydantic==2.*
```

### External Requirements

- **Groq API key** — free tier sufficient (console.groq.com)
- No paid OpenAI key required

### One-time Post-Install Step

```bash
python -m spacy download en_core_web_lg
```

---

## Scope Boundaries

| In scope (Stage 1) | Out of scope (Stages 2–4) |
|---|---|
| RAG knowledge retrieval | Admin approval workflow |
| SQLite dynamic data | Reservation confirmation emails |
| Reservation data collection + validation | Advanced UI / web frontend |
| PII detection and guardrails | Deployment / containerisation |
| Retrieval evaluation metrics | Multi-facility support |

---

## Post-Stage-1 Update: Database-driven occupancy

Implemented after Stage 1 completion. Key changes:

- Added `vehicle_capacity` and `reservations` tables to SQLite schema
- Availability computed exclusively from active reservations (not hardcoded values)
- `vehicle_type` added to `RESERVATION_FIELDS`; booking flow shows live DB menu
- `save_reservation()` uses `BEGIN IMMEDIATE` transaction for atomic capacity check + insert
- Duplicate/overlap detection, capacity enforcement, cancellation all verified
- Test count: 166 → 203 tests, all passing
