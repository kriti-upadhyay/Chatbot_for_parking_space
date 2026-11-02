# Intelligent Parking Space Reservation Chatbot

A conversational parking reservation system built with LangChain and LangGraph. It answers facility questions using RAG, retrieves live pricing and availability from SQLite, collects reservation details, routes bookings through administrator approval, notifies users of decisions, and exports approved reservations through an MCP server.

## 1. Project Overview

### User Portal
- Answers facility FAQs using ChromaDB and Groq.
- Retrieves live prices, hours, and availability from SQLite.
- Collects and validates reservation details through natural conversation.
- Extracts reservation details supplied together in a message and asks only for missing or invalid fields.
- Saves reservations as pending administrator approval.
- Checks reservation status and notifies users when an administrator approves or rejects a booking.

### Admin Portal
- Accessed from the **Admin** button at the top of the user portal.
- Requires a password configured through `ADMIN_PASSWORD`.
- Provides a reservation dashboard with approve and reject actions.
- Rechecks capacity when approving a reservation.
- Automatically attempts MCP export after approval.
- Supports retrying failed exports.
- Does not include an admin chatbot; approval and rejection are performed through dashboard controls.

### MCP Approval Logger
- Exposes the `write_approval_record` tool through FastMCP.
- Uses the MCP client protocol to call the tool.
- Writes approved reservations to `storage/approved_reservations.txt`.
- Uses `MCP_SHARED_SECRET` for authentication, independently of the admin password.
- Supports idempotent exports and recovery from interrupted writes.

## 2. Architecture

```text
User
 │
 ▼
Single Streamlit App
 ├── User Portal
 │    └── LangGraph user workflow
 │         ├── Status check
 │         ├── Guardrails
 │         ├── FAQ / RAG
 │         ├── Live SQL lookup
 │         └── Reservation collection and save
 │
 └── Admin button
      └── Password verification
           └── Admin dashboard
                ├── View reservations
                ├── Approve / reject
                └── Retry failed MCP exports

SQLite ── approved reservation ──► MCP client ──► MCP server
                                                   │
                                                   ▼
                                  storage/approved_reservations.txt
```

### Hybrid RAG + SQL

| Information | Source | Retrieval |
|---|---|---|
| Facility description, rules, FAQs | ChromaDB | Similarity search |
| Prices and opening hours | SQLite | Parameterized SQL |
| Availability | SQLite | Computed from capacity and active reservations |
| Reservations and admin decisions | SQLite | Direct database operations |
| Approval export state | SQLite | Export status and timestamp |

The LLM does not invent live prices or availability, and it cannot approve or reject reservations.

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM framework | LangChain |
| Workflow orchestration | LangGraph |
| LLM provider | Groq |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Relational database | SQLite |
| Admin and user interface | Streamlit |
| MCP server | FastMCP |
| Guardrails | Microsoft Presidio and custom validators |
| Testing | pytest |

## 4. Project Structure

```text
parking-chatbot/
├── data/
│   ├── documents/                 # Parking knowledge documents
│   └── evaluation/
├── storage/                       # Generated at runtime; gitignored
│   ├── chroma/
│   ├── parking.db
│   └── approved_reservations.txt
├── src/
│   ├── config/
│   ├── ingestion/
│   ├── rag/
│   ├── database/
│   ├── chatbot/                   # User workflow
│   ├── workflow/                  # Reservation approval workflow
│   ├── admin/                     # Admin authentication and actions
│   ├── mcp_server/
│   ├── guardrails/
│   └── evaluation/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
├── streamlit_app.py               # User portal and admin entry point
├── admin_app.py                   # Legacy/separate admin entry point, if retained
├── main.py
└── README.md
```

The exact files may vary as the project evolves. The user-facing application is launched through `streamlit_app.py`.

## 5. Environment Setup

### Prerequisites
- Python 3.11 or higher
- Git
- Groq API key

### Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 6. Environment Variables

Copy the example file:

```bash
cp .env.example .env
```

Configure the required values in `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL_NAME=your_available_groq_model

ADMIN_PASSWORD=choose_a_strong_password
MCP_SHARED_SECRET=your_random_hex_secret

EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHROMA_COLLECTION_NAME=parking_knowledge
RETRIEVAL_K=4
LOG_LEVEL=INFO
```

Generate an MCP secret with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Keep `.env`, API keys, passwords, and database files out of version control. `ADMIN_PASSWORD` and `MCP_SHARED_SECRET` must be different.

## 7. Ingest Documents

Run this before using RAG for the first time:

```bash
python main.py ingest
```

## 8. Initialize the Database

```bash
python main.py initdb
```

The initialization command creates or migrates the SQLite database. It is designed to be safe to run again on an existing database.

## 9. Run the Application

Start the single Streamlit application:

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

### Access the Admin Portal
1. Click **Admin** at the top of the user portal.
2. Enter the configured admin password.
3. Use the dashboard to review and approve or reject reservations.
4. Log out or return to the user portal when finished.

You do not need to launch `admin_app.py` separately for the integrated admin portal.

## 10. Reservation Approval and Export

1. The user submits a reservation through the chatbot.
2. The reservation is saved with pending status.
3. The administrator approves or rejects it in the dashboard.
4. On approval, the application attempts to export the reservation through the MCP server.
5. The user portal checks the reservation status and displays the decision.
6. If an export fails, the administrator can use **Retry Export**.

The approval log uses this format:

```text
Name | Car Number | Reservation Period | Approval Time
```

The export process is designed to avoid duplicate records and recover from interrupted writes.

## 11. Run Tests

Run the full test suite:

```bash
python -m pytest
```

Run tests for a specific component, for example:

```bash
python -m pytest tests/test_mcp_server.py -v
```

Test counts can change as the project evolves; use the current test output as the source of truth.

## 12. Run Evaluation

After ingesting the documents:

```bash
python main.py evaluate
```

Evaluation reports are generated in `reports/`.

## 13. Stage Roadmap

| Stage | Description |
|---|---|
| Stage 1 | RAG chatbot, SQLite data access, reservations, guardrails, and user interface |
| Stage 2 | Admin approval workflow and user status notifications |
| Stage 3 | Authenticated, idempotent MCP approval logging |
| Stage 4 | LangGraph workflow integration and application improvements |
| Future | Deployment, containerization, and multi-facility support |

## Quick Start

```bash
# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Configure .env with required credentials
# Initialize application data
python main.py ingest
python main.py initdb

# Launch the user portal and integrated admin entry point
streamlit run streamlit_app.py

# Run tests
python -m pytest
```
