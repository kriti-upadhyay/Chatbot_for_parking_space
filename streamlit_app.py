"""
Streamlit UI for the CityPark Parking Chatbot.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st

# ── Page config — must be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="CityPark Chatbot",
    page_icon="🅿️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Imports (after set_page_config) ───────────────────────────────────────────
from src.chatbot.state import empty_state
from src.config import get_logger, settings

logger = get_logger(__name__)


# ── Helper: system status ─────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _system_status() -> dict:
    """Check ChromaDB and SQLite readiness.  Cached for 30 s."""
    status = {"chunks": 0, "chromadb_ok": False, "sqlite_ok": False}
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        col = client.get_collection(settings.chroma_collection_name)
        status["chunks"] = col.count()
        status["chromadb_ok"] = status["chunks"] > 0
    except Exception:
        pass
    try:
        import sqlite3
        if settings.db_path.exists():
            conn = sqlite3.connect(str(settings.db_path))
            n = conn.execute("SELECT COUNT(*) FROM parking_prices").fetchone()[0]
            conn.close()
            status["sqlite_ok"] = n > 0
    except Exception:
        pass
    return status


# ── Session state ─────────────────────────────────────────────────────────────

def _init():
    if "ready" in st.session_state:
        return
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Welcome to **CityPark**! I can help you with:\n\n"
                "- 📍 Location and driving directions\n"
                "- 💷 Parking prices and current availability\n"
                "- 🕐 Opening and staffing hours\n"
                "- ♿ Facilities and accessibility\n"
                "- 📋 Making a reservation\n\n"
                "What would you like to know?"
            ),
        }
    ]
    st.session_state.conv_state = empty_state()
    st.session_state.graph = None
    st.session_state.graph_error = None
    st.session_state.ready = True


@st.cache_resource(show_spinner=False)
def _load_graph():
    """Build the LangGraph graph once per Streamlit server process."""
    from src.chatbot.graph import build_graph
    return build_graph()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.title("🅿️ CityPark")
        st.caption("Intelligent Parking Assistant")
        st.divider()

        # System status
        st.subheader("System Status")
        status = _system_status()

        col1, col2 = st.columns(2)
        with col1:
            if status["chromadb_ok"]:
                st.success(f"✅ Knowledge\n{status['chunks']} chunks")
            else:
                st.error("❌ Knowledge\nRun: ingest")
        with col2:
            if status["sqlite_ok"]:
                st.success("✅ Database\nReady")
            else:
                st.error("❌ Database\nRun: initdb")

        st.divider()

        # Quick examples
        st.subheader("Try asking…")
        examples = [
            "Where is CityPark located?",
            "How much does it cost to park?",
            "Are there EV charging points?",
            "What are the opening hours?",
            "How many spaces are available?",
            "I want to make a reservation",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
                st.session_state._quick_input = ex

        st.divider()

        # Clear conversation
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]  # keep welcome
            st.session_state.conv_state = empty_state()
            st.rerun()

        st.caption("Stage 2 · RAG + LangGraph · Groq API")
        st.divider()
        st.markdown(
            "**Administrator?** Open the "
            "[Admin Dashboard](http://localhost:8502) "
            "to review pending reservations."
        )


# ── Main chat area ────────────────────────────────────────────────────────────

def _render_chat():
    st.title("CityPark Parking Chatbot")
    st.caption("Ask me anything about CityPark — prices, availability, directions, or reservations.")

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🅿️" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Handle quick-input from sidebar buttons
    quick = st.session_state.pop("_quick_input", None)

    # Chat input
    user_input = st.chat_input("Type your message…") or quick

    if not user_input:
        return

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Load graph (cached — only built once)
    with st.spinner("Thinking…"):
        try:
            graph = _load_graph()
        except Exception as exc:
            st.error(
                f"⚠️ Could not start the chatbot: {exc}\n\n"
                "Make sure your `.env` file has a valid `GROQ_API_KEY` and "
                "that you have run `python main.py ingest` and `python main.py initdb`."
            )
            return

        # Run one turn through the graph
        try:
            new_state = graph.invoke(
                {**st.session_state.conv_state, "user_input": user_input}
            )
            st.session_state.conv_state = new_state
            response = new_state.get("response", "")
        except Exception as exc:
            logger.exception("Graph invocation failed")
            response = (
                "I'm sorry, something went wrong. "
                "Please try again or contact CityPark on 0800 123 4567."
            )

    # Show bot response
    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar="🅿️"):
            st.markdown(response)

    st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

_init()
_render_sidebar()
_render_chat()
