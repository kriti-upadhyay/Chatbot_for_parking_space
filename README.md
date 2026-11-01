# Intelligent Parking Space Reservation Chatbot

A production-quality RAG chatbot that answers questions about a parking facility, retrieves live pricing and availability from a database, collects and validates reservation details through natural conversation, routes bookings through an authenticated administrator approval workflow, notifies users of outcomes, and writes confirmed reservations to a permanent approval log via an MCP server — all orchestrated with LangGraph.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Environment Setup](#5-environment-setup)
6. [Installation](#6-installation)
7. [Environment Variable Configuration](#7-environment-variable-configuration)
8. [How to Ingest Documents](#8-how-to-ingest-documents)
9. [How to Initialize the Database](#9-how-to-initialize-the-database)
10. [How to Run the User Chatbot](#10-how-to-run-the-user-chatbot)
11. [How to Run the Admin Dashboard](#11-how-to-run-the-admin-dashboard)
12. [How to Run Tests](#12-how-to-run-tests)
13. [How to Run Evaluation](#13-how-to-run-evaluation)
14. [Example Conversations](#14-example-conversations)
15. [Verifying the MCP Approval Log](#15-verifying-the-mcp-approval-log)
16. [Stage Roadmap](#16-stage-roadmap)

---

## 1. Project Overview

CityPark Chatbot is a three-component, multi-turn conversational AI system covering the full reservation lifecycle:

**User Agent (Agent 1)**
- Answers static parking questions using RAG (ChromaDB + Groq LLM).
- Retrieves live prices, hours, and availability from SQLite — the LLM never invents these.
- Collects reservation details interactively (6 fields including vehicle type from a live DB menu).
- Persists every booking atomically with `BEGIN IMMEDIATE` to prevent overbooking.
- Polls reservation status on each subsequent turn and notifies the user the moment an admin decides — exactly once.

**Admin Agent (Agent 2)**
- Password-authenticated Streamlit dashboard.
- Read-only LLM assistant can list, filter, and explain reservations — it cannot approve or reject.
- **Approve** and **Reject** buttons are the only way to write a decision to the database.
- Approval re-checks capacity inside a transaction; duplicate decisions are blocked.
- After each approval, the MCP client is called automatically to export the record.

**MCP Approval Logger (Stage 3)**
- FastMCP server exposes one tool: `write_approval_record`.
- Called via the MCP JSON-RPC protocol (in-process `Client`) — not a direct function call.
- Writes confirmed reservations to `storage/approved_reservations.txt` in the exact required format.
- Authenticated with a time-based HMAC token derived from `MCP_SHARED_SECRET` (independent of `ADMIN_PASSWORD`).
- Fully idempotent: repeated calls for the same reservation produce exactly one line.
- Crash-safe: a `EXPORTING` sentinel in the database tracks partial writes; recovery scans the file, handles partial blocks, and preserves all other valid records.
- Failed exports leave the reservation `confirmed` in the database; the **Retry Export** button allows safe re-attempts.

**Guardrails (all components)**
- Scored prompt-injection detection (7 patterns with accumulated score).
- PII detection and output scrubbing via Microsoft Presidio.
- Custom UK vehicle registration validator.
- Admin password verified with `hmac.compare_digest` (constant-time, no logs).

---

## 2. Architecture

### Full System Flow

```
User                    User Agent (LangGraph)           Database          Admin / MCP
 │                            │                             │                   │
 │── "I want to reserve" ────▶│                             │                   │
 │                       status_check_node                  │                   │
 │                       guardrail_node                     │                   │
 │                       reservation_node ─────── save ────▶│ status=pending    │
 │◀── "Saved! Ref #1" ────────│                             │                   │
 │                                                          │                   │
 │                                                          │◀─ list pending ───│
 │                                                          │─ approve #1 ─────▶│ status=confirmed
 │                                                          │                   │── MCP call ──▶ write_approval_record(1)
 │                                                          │                   │               write to .txt file
 │                                                          │ exported_at=now   │◀──────────────
 │── "Any update?" ──────────▶│                             │                   │
 │                       status_check_node ──── poll ──────▶│                   │
 │◀── "Approved by admin" ────│                             │                   │
```

### User Agent Node Graph

```
START
  └─► status_check_node   (notify if admin decided → END)
        └─► guardrail_node (injection + PII)
              └─► intent_router_node
                    ├─► rag_node       (general_faq  → ChromaDB + Groq)
                    ├─► sql_node       (dynamic_info → SQLite live queries)
                    └─► reservation_node (reservation/input → SQLite save)
                          each → END
```

### Hybrid RAG + SQL Strategy

| Information Type | Storage | Retrieval |
|---|---|---|
| Facility description, rules, FAQs, guides | ChromaDB (vector store) | Similarity search |
| Pricing, working hours | SQLite (`parking_prices`, `working_hours`) | Parameterised SQL |
| Space availability | SQLite (`vehicle_capacity` + `reservations`) | Computed: total − active |
| Reservations (user-submitted) | SQLite (`reservations`) | Direct write; never in ChromaDB |
| Admin decisions | SQLite (`reservations.status`, `admin_notes`, `decided_by`) | Direct write via buttons only |
| Export state | SQLite (`reservations.exported_at`) | `NULL` / `EXPORTING` / ISO timestamp |

### Key Design Decisions

| Decision | Rationale |
|---|---|
| LLM cannot approve/reject (architectural) | Approve/reject tools are never given to the LLM; buttons are the only write path |
| MCP tool called via `Client` (genuine protocol) | Uses full JSON-RPC handshake (`initialize` → `tools/list` → `tools/call`); not a direct function call |
| `MCP_SHARED_SECRET` separate from `ADMIN_PASSWORD` | Independent rotation; compromise of one does not compromise the other |
| HMAC time-based token (1-min window) | Stateless, replay-resistant; only a holder of `MCP_SHARED_SECRET` can generate valid tokens |
| `EXPORTING` sentinel in DB before file write | Concurrent callers serialise on the DB claim; crash recovery detects mid-write state |
| Filter-rewrite preserves later valid records | Truncating to EOF would delete records appended after the partial block by a different reservation |
| `os.replace()` atomic rename | Readers never see a partially-written file during recovery |
| Idempotency: scan before every recovery write | Sentinel alone is not proof of success; the following data line is verified for completeness |
| `BEGIN IMMEDIATE` in approve + save | Write lock covers capacity-check + UPDATE/INSERT; prevents concurrent overbooking |
| `hmac.compare_digest` for all password checks | Constant-time comparison prevents timing-based enumeration |
| `pending_reservation_id` in user state | One-shot status polling without websockets; cleared after delivery |

---

## 3. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.11+ (tested on 3.14) | |
| LLM framework | LangChain 0.3.x | Non-deprecated LCEL APIs |
| Workflow orchestration | LangGraph 0.2.x | Two separate StateGraphs (user + admin) |
| LLM provider | Groq API (`qwen/qwen3.8-27b`) | Free tier; model configurable |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | 384-dim, CPU-only |
| Vector database | ChromaDB 1.x | Persistent local storage |
| Relational database | SQLite (stdlib) | All dynamic data + reservations + decisions + export state |
| PII detection | Microsoft Presidio 2.2 + spaCy 3.8 | Custom UK vehicle reg recognizer |
| MCP server | FastMCP 4.x | `write_approval_record` tool; in-process `Client` for genuine MCP protocol |
| File locking | filelock 3.x | Cross-platform exclusive lock; thread + process safe |
| Admin authentication | `hmac.compare_digest` + env var | Constant-time comparison |
| MCP authentication | Time-based HMAC + `MCP_SHARED_SECRET` | Independent of admin password |
| User UI | Streamlit 1.35+ (`streamlit_app.py`) | Chat interface, port 8501 |
| Admin UI | Streamlit 1.35+ (`admin_app.py`) | Approval dashboard + LLM assistant + Retry Export, port 8502 |
| Testing | pytest 8.x + pytest-mock | All external calls mocked; 314 tests |
| Configuration | python-dotenv + pydantic-settings | |

---

## 4. Project Structure

```
parking-chatbot/
│
├── data/
│   ├── documents/               # 6 parking knowledge .md files
│   └── evaluation/
│       └── eval_dataset.json    # 15 labelled evaluation questions
│
├── storage/                     # Generated at runtime — gitignored
│   ├── chroma/                  # Persistent ChromaDB
│   ├── parking.db               # SQLite database (5 tables)
│   ├── doc_hashes.json          # Change-detection hashes for ingestion
│   └── approved_reservations.txt  # MCP approval log (Stage 3)
│
├── src/
│   ├── config/
│   │   └── settings.py          # Env vars, paths, logging (incl. MCP_SHARED_SECRET)
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── prompt.py
│   │   └── chain.py
│   ├── database/
│   │   ├── schema.py            # DDL + migration (incl. exported_at column)
│   │   └── queries.py           # All queries incl. approve/reject + get_pending_exports
│   ├── chatbot/                 # User Agent (Agent 1)
│   │   ├── state.py
│   │   ├── router.py
│   │   ├── nodes.py             # incl. status_check_node
│   │   └── graph.py
│   ├── admin/                   # Admin Agent (Agent 2)
│   │   ├── state.py
│   │   ├── auth.py              # verify_admin_password()
│   │   ├── tools.py             # 3 read-only LangChain tools
│   │   ├── nodes.py
│   │   └── graph.py
│   ├── mcp_server/              # MCP Approval Logger (Stage 3)
│   │   ├── server.py            # FastMCP server — write_approval_record tool
│   │   ├── file_writer.py       # Scan, append, filter-rewrite, fsync
│   │   └── client.py            # generate_mcp_token() + call_write_approval_record()
│   ├── guardrails/
│   │   ├── validators.py
│   │   ├── injection.py
│   │   └── pii.py
│   └── evaluation/
│       ├── dataset.py
│       ├── metrics.py
│       └── report.py
│
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_config.py           #  7 tests
│   ├── test_ingestion.py        # 18 tests
│   ├── test_database.py         # 68 tests
│   ├── test_rag.py              # 18 tests
│   ├── test_chatbot.py          # 35 tests
│   ├── test_guardrails.py       # 43 tests
│   ├── test_evaluation.py       # 33 tests
│   ├── test_admin_db.py         # 30 tests
│   ├── test_admin_agent.py      # 21 tests
│   ├── test_user_notifications.py # 16 tests
│   └── test_mcp_server.py       # 44 tests  ← Stage 3
│
├── reports/
├── .env.example
├── .env                         # gitignored
├── .gitignore
├── .venv/                       # gitignored
├── pytest.ini
├── requirements.txt
├── PLAN.md
├── streamlit_app.py             # User chatbot — port 8501
├── admin_app.py                 # Admin dashboard — port 8502
├── README.md
└── main.py
```

### SQLite Database Tables

| Table | Purpose |
|---|---|
| `parking_prices` | Tiered pricing per space type |
| `working_hours` | Staffed hours by day type |
| `parking_availability` | Per-floor/zone total capacity (display only) |
| `vehicle_capacity` | Total bookable spaces per vehicle type (authoritative) |
| `reservations` | Every booking; occupancy, decisions, and export state stored here |

**Key `reservations` columns added over time:**

| Column | Stage | Meaning |
|---|---|---|
| `status` | 1 | `pending` / `confirmed` / `rejected` / `cancelled` / `expired` |
| `admin_notes`, `decided_at`, `decided_by` | 2 | Admin decision metadata |
| `exported_at` | 3 | `NULL` = not exported; `EXPORTING` = in progress / crashed; ISO timestamp = done |

---

## 5. Environment Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or higher |
| Groq API key | Free account at [console.groq.com](https://console.groq.com) |
| Git | Any recent version |
| C++ compiler | **Not required** (all wheels are pre-built) |

### Virtual Environment

A pre-built `.venv` is included. To activate:

```bash
# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Git Bash) / macOS / Linux
source .venv/Scripts/activate
```

Or create fresh:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 6. Installation

```bash
git clone <repository-url>
cd parking-chatbot
source .venv/Scripts/activate

# If installing fresh:
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **Note on Python 3.14:** `requirements.txt` pins ChromaDB ≥ 1.0 and spaCy ≥ 3.8, both of which ship pre-built wheels. No C++ compiler is needed.

---

## 7. Environment Variable Configuration

```bash
cp .env.example .env
```

Open `.env` and set:

```env
# Required — get from https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Groq model (check your account for available models)
GROQ_MODEL_NAME=qwen/qwen3.8-27b

# Required for the admin dashboard
ADMIN_PASSWORD=choose_a_strong_password

# Required for the MCP approval logger (Stage 3)
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
MCP_SHARED_SECRET=your_random_hex_secret_here

# Optional
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHROMA_COLLECTION_NAME=parking_knowledge
RETRIEVAL_K=4
LOG_LEVEL=INFO
# APPROVAL_LOG_FILE=storage/approved_reservations.txt
```

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `GROQ_MODEL_NAME` | `qwen/qwen3.8-27b` | Groq model identifier |
| `ADMIN_PASSWORD` | *(required for admin UI)* | Dashboard login password. **Never commit the real value.** |
| `MCP_SHARED_SECRET` | *(required for export)* | Signs MCP auth tokens. **Must be different from `ADMIN_PASSWORD`.** Generate with `secrets.token_hex(32)`. |
| `APPROVAL_LOG_FILE` | `storage/approved_reservations.txt` | Output path for the approval log. Overridable but the MCP tool ignores any caller-supplied path. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHROMA_COLLECTION_NAME` | `parking_knowledge` | ChromaDB collection name |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per query |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |

> **Model availability:** To list models on your Groq account:
> ```bash
> python -c "
> from groq import Groq; import os; from dotenv import load_dotenv
> load_dotenv()
> [print(m.id) for m in Groq(api_key=os.getenv('GROQ_API_KEY')).models.list().data]
> "
> ```

---

## 8. How to Ingest Documents

```bash
python main.py ingest
```

Expected:
```
INFO | Added 49 chunks to ChromaDB collection 'parking_knowledge'
INFO | Ingestion complete: {'ingested': 6, 'skipped': 0, ...}
```

---

## 9. How to Initialize the Database

```bash
python main.py initdb
```

Safe to run multiple times. Applies all column migrations (`admin_notes`, `decided_at`, `decided_by`, `exported_at`) via `ALTER TABLE ADD COLUMN` — no data is lost on existing databases.

**Initial vehicle capacities:**

| Vehicle Type | Total Spaces |
|---|---|
| Standard | 720 |
| Disabled (Blue Badge) | 50 |
| EV Charging | 40 |
| Motorcycle | 20 |
| Oversized | 20 |
| **Total** | **850** |

---

## 10. How to Run the User Chatbot

**Important:** Run `ingest` and `initdb` before the first launch.

### Browser UI (recommended, port 8501)

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501**.

### Command-line Interface

```bash
python main.py chat
```

---

## 11. How to Run the Admin Dashboard

```bash
streamlit run admin_app.py --server.port 8502
```

Open **http://localhost:8502**. Requires `ADMIN_PASSWORD` and `MCP_SHARED_SECRET` set in `.env`.

### Dashboard features

| Feature | Details |
|---|---|
| Password gate | Login form; verified with `hmac.compare_digest` |
| Reservation table | Filterable by status |
| Approve button | Approves + triggers MCP export automatically |
| Reject button | Requires a written reason |
| **Retry Export** | Shown for confirmed reservations where export failed; re-calls the MCP tool |
| Duplicate prevention | Already-decided reservations show a warning |
| LLM assistant (sidebar) | Read-only; cannot approve, reject, or export |
| Logout | Clears session immediately |

### Security notes

- `ADMIN_PASSWORD` and `MCP_SHARED_SECRET` are independent secrets — rotate either without affecting the other.
- The MCP tool validates a time-based HMAC token (1-min window) before any file or DB operation.
- The LLM assistant is never given the `write_approval_record` tool; approval requires a human button click.
- All password and token comparisons use `hmac.compare_digest` to prevent timing attacks.

---

## 12. How to Run Tests

All **314 tests** run fully offline — no API keys, no model downloads, no network calls.

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run Stage 3 tests only
python -m pytest tests/test_mcp_server.py -v

# Run with coverage
pip install pytest-cov
python -m pytest --cov=src --cov-report=term-missing
```

Expected:
```
====================== 314 passed, 1 warning in X.Xs ======================
```

### Test modules

| File | Tests | What it covers |
|---|---|---|
| `test_config.py` | 7 | Settings, env overrides, paths, logging |
| `test_ingestion.py` | 18 | Document loading, chunking, hashing, ChromaDB upsert |
| `test_database.py` | 68 | Schema, seed data, pricing/hours, `save_reservation`, duplicates, capacity, cancellation |
| `test_rag.py` | 18 | Prompt templates, retriever, chain behavior, error handling |
| `test_chatbot.py` | 35 | State, routing, nodes, vehicle type menu, DB-save, capacity/duplicate/error handling |
| `test_guardrails.py` | 43 | Validators, injection detection, PII detection |
| `test_evaluation.py` | 33 | Dataset loading, P@K, R@K, MRR, latency, report generation |
| `test_admin_db.py` | 30 | Migration, approve, reject, duplicate decisions, capacity recheck |
| `test_admin_agent.py` | 21 | Auth, read-only tools, LLM-cannot-approve constraint, admin graph |
| `test_user_notifications.py` | 16 | `status_check_node`, one-time delivery, lifecycle, graph integration |
| `test_mcp_server.py` | 44 | Field sanitisation, file scanning, filter-rewrite, idempotency, crash recovery, partial blocks with later records, concurrent calls, auth, unauthorised invocation, MCP protocol |

---

## 13. How to Run Evaluation

```bash
python main.py evaluate
```

Requires documents to be ingested first. Generates a Markdown report in `reports/`.

---

## 14. Example Conversations

### General FAQ

```
You: Where is CityPark located?

Bot: CityPark is located at 14 Central Plaza, City Centre, CC1 2AB.
```

### Live Availability

```
You: Are there any parking spaces available?

Bot: Current parking availability at CityPark:
       • Standard: 720 of 720 available (0% occupied)
       • EV Charging: 40 of 40 available (0% occupied)
       ...
     Availability is calculated from confirmed reservations in real time.
```

### Reservation → Admin Approval → MCP Export → User Notification

**User submits reservation:**
```
You: I'd like to make a reservation

Bot: What is your first name?
You: Alice
Bot: What is your surname?
You: Smith
Bot: Please select your vehicle type:
       1. Disabled (Blue Badge) — 50 spaces available
       2. EV Charging — 40 spaces available
       3. Motorcycle — 20 spaces available
       4. Oversized — 20 spaces available
       5. Standard — 720 spaces available
You: 5
Bot: Vehicle registration?
You: AB12 CDE
Bot: Start date and time? (YYYY-MM-DD HH:MM)
You: 2024-07-20 09:00
Bot: End date and time?
You: 2024-07-20 17:00

Bot: Your reservation has been saved! (Reference: #1)
     Pending administrator approval.
```

**Admin approves (http://localhost:8502):**
```
[Clicks ✅ Approve on reservation #1]
→ "Reservation #1 approved."
→ MCP export runs automatically.
→ storage/approved_reservations.txt updated.
```

**User notified on next message:**
```
You: What time do you open?

Bot: ✅ Good news! Your reservation (#1) has been approved by admin.
     You can proceed to CityPark on your reserved date.
     [continues to answer the hours question]
```

### Approval Log File

```
Name | Car Number | Reservation Period | Approval Time
# reservation:1
Alice Smith | AB12CDE | 2024-07-20 09:00 – 2024-07-20 17:00 | 2024-07-20 11:05:30
# reservation:2
Bob Jones | XY34ZAB | 2024-07-21 10:00 – 2024-07-21 18:00 | 2024-07-21 09:12:44
```

The `# reservation:N` sentinel lines are idempotency keys — they prevent duplicate lines if the process crashes or the button is clicked twice. Human readers see only the four-field data lines in the required format.

---

## 15. Verifying the MCP Approval Log

### Step 1 — Confirm the migration ran

```bash
python -c "
import sqlite3
conn = sqlite3.connect('storage/parking.db')
cols = [r[1] for r in conn.execute('PRAGMA table_info(reservations)')]
conn.close()
print('exported_at present:', 'exported_at' in cols)
"
```

Expected: `exported_at present: True`

### Step 2 — Run Stage 3 tests (offline, no keys needed)

```bash
python -m pytest tests/test_mcp_server.py -v
```

Expected: **44 passed**

### Step 3 — Quick end-to-end script

```bash
python -c "
from src.database.schema import initialise_database
from src.database.queries import save_reservation, approve_reservation
from src.mcp_server.client import call_write_approval_record
from src.config import settings

initialise_database()

res = save_reservation(
    first_name='Test', surname='User',
    vehicle_type='standard', vehicle_reg='TS01 TST',
    start_datetime='2090-08-01 09:00', end_datetime='2090-08-01 17:00',
)
approve_reservation(res['reservation_id'], admin_id='admin')

result = call_write_approval_record(res['reservation_id'])
print('MCP result:', result)

log = settings.approval_log_file
if log.exists():
    print()
    print(log.read_text(encoding='utf-8'))
"
```

**Expected output:**
```
MCP result: {'success': True, 'error': None, 'already_exported': False, 'record': '...'}

Name | Car Number | Reservation Period | Approval Time
# reservation:1
Test User | TS01TST | 2090-08-01 09:00 – 2090-08-01 17:00 | 2026-...
```

Running the same script a second time returns `already_exported: True` and the file still contains exactly one data line.

### Step 4 — Through the admin UI

1. Make a reservation via the user chatbot (port 8501).
2. Open the admin dashboard (port 8502) and click **✅ Approve**.
3. Check the output: `cat storage/approved_reservations.txt`
4. To test **Retry Export**: temporarily remove `MCP_SHARED_SECRET` from `.env`, approve a reservation (export fails silently), restore the secret, then click **🔄 Retry Export** on that row.

---

## 16. Stage Roadmap

| Stage | Status | Description |
|---|---|---|
| **Stage 1** | ✅ Complete | RAG system, SQLite dynamic data + reservations, chatbot, guardrails, evaluation, Streamlit user UI |
| **Stage 2** | ✅ Complete | Admin approval agent, authenticated dashboard, user status notifications, two-agent LangGraph integration |
| **Stage 3** | ✅ Complete | MCP approval logger — idempotent crash-safe file export with HMAC auth and filter-rewrite recovery |
| **Stage 4** | Planned | Deployment; containerisation; multi-facility support |

---

## Quick Start (TL;DR)

```bash
# 1. Activate the virtual environment
source .venv/Scripts/activate          # Windows Git Bash
# or: .venv\Scripts\activate           # Windows CMD

# 2. Configure
cp .env.example .env
# Edit .env:
#   GROQ_API_KEY=...
#   ADMIN_PASSWORD=...
#   MCP_SHARED_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")

# 3. Set up data
python main.py ingest       # embed knowledge docs into ChromaDB
python main.py initdb       # create/migrate SQLite (adds exported_at column safely)

# 4. Run both interfaces
streamlit run streamlit_app.py                        # user chatbot  — port 8501
streamlit run admin_app.py --server.port 8502         # admin panel   — port 8502

# 5. Test (no API key needed — all 314 tests run offline)
python -m pytest

# 6. Evaluate (requires ChromaDB to be populated)
python main.py evaluate

# 7. Check the approval log after approving a reservation
cat storage/approved_reservations.txt
```
