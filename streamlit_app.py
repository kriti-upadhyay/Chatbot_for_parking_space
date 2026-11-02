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
    # Admin panel state (not set here — defaults to False/absent)
    # st.session_state.show_admin_panel: bool
    # st.session_state.admin_authenticated: bool
    # st.session_state.admin_id: str


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


# ── Top bar — Admin button always visible at the top of the main area ────────

def _render_top_bar() -> None:
    """
    Render a one-line bar at the very top of the main page containing the
    Admin Panel button (or a Back-to-chat button when admin view is active).
    Stays at the top regardless of chat history length.
    """
    _, right = st.columns([5, 1])
    with right:
        if st.session_state.get("show_admin_panel"):
            if st.button("← Chat", use_container_width=True,
                         help="Return to the user chatbot"):
                for key in ("admin_authenticated", "admin_id", "show_admin_panel"):
                    st.session_state.pop(key, None)
                st.rerun()
        else:
            if st.button("🔐 Admin", use_container_width=True,
                         help="Open the administrator dashboard"):
                st.session_state.show_admin_panel = True
                st.rerun()


# ── Inline admin panel ───────────────────────────────────────────────────────

def _render_admin_login() -> bool:
    """
    Show the admin login form inside the user portal.

    Returns True if the admin is already authenticated (or just logged in).
    Reads the password from ADMIN_PASSWORD env var via verify_admin_password().
    The password is never displayed or stored beyond the form submit.
    """
    from src.admin.auth import is_admin_configured, verify_admin_password

    if st.session_state.get("admin_authenticated"):
        return True

    st.subheader("🔐 Administrator Login")

    if not is_admin_configured():
        st.error(
            "⚠️ `ADMIN_PASSWORD` is not configured.  "
            "Set it in your `.env` file and restart the app."
        )
        return False

    with st.form("admin_login_inline"):
        pwd = st.text_input("Admin password", type="password",
                            placeholder="Enter admin password")
        col_a, col_b = st.columns(2)
        with col_a:
            submitted = st.form_submit_button("Log in", use_container_width=True,
                                              type="primary")
        with col_b:
            cancel = st.form_submit_button("← Back to chat", use_container_width=True)

    if cancel:
        st.session_state.show_admin_panel = False
        st.rerun()

    if submitted:
        if verify_admin_password(pwd):
            st.session_state.admin_authenticated = True
            st.session_state.admin_id = "admin"
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


def _render_admin_panel() -> None:
    """
    Render the full admin dashboard inside the user portal.

    Reuses render_reservations() from the shared dashboard module so the
    approval/export logic is never duplicated.
    """
    from src.admin.dashboard import render_reservations

    admin_id = st.session_state.get("admin_id", "admin")

    col_title, col_logout = st.columns([4, 1])
    with col_title:
        st.title("🔐 CityPark Admin Dashboard")
        st.caption(f"Logged in as: **{admin_id}**")
    with col_logout:
        st.write("")  # vertical alignment
        if st.button("🚪 Log out / Back to chat", use_container_width=True):
            for key in ("admin_authenticated", "admin_id", "show_admin_panel"):
                st.session_state.pop(key, None)
            st.rerun()

    render_reservations(admin_id)


# ── Reservation status auto-poll ─────────────────────────────────────────────

@st.fragment(run_every=3)
def _poll_reservation_status():
    """
    Poll the database every 3 seconds for an admin decision on the user's
    pending reservation.  When a decision arrives, inject a notification
    message into the chat history and clear the tracked reservation ID so
    the poll stops and the notification is not repeated.

    Uses the existing get_reservation_status() query — the database is
    always the source of truth.  Does nothing when there is no pending
    reservation or the status is still 'pending'.
    """
    rid = st.session_state.conv_state.get("pending_reservation_id")
    if rid is None:
        return

    # Guard against re-delivering the same notification on page reload.
    notified = st.session_state.setdefault("notified_rids", set())
    if rid in notified:
        st.session_state.conv_state = {
            **st.session_state.conv_state,
            "pending_reservation_id": None,
        }
        return

    try:
        from src.database.queries import get_reservation_status
        status_info = get_reservation_status(rid)
    except Exception:
        return  # DB unavailable — retry next cycle

    if status_info is None or status_info["status"] == "pending":
        return  # still waiting

    status     = status_info["status"]
    notes      = status_info.get("admin_notes") or ""
    decided_by = status_info.get("decided_by") or "the administrator"

    if status == "confirmed":
        notification = (
            f"✅ **Good news!** Your reservation (#{rid}) has been "
            f"**approved** by {decided_by}.\n"
            + (f"Note from administrator: {notes}\n" if notes else "")
            + "You can proceed to CityPark on your reserved date."
        )
    elif status == "rejected":
        notification = (
            f"❌ We're sorry — your reservation (#{rid}) has been "
            f"**declined** by {decided_by}.\n"
            + (f"Reason: {notes}\n" if notes else "")
            + "Please contact CityPark on 0800 123 4567 if you have questions."
        )
    else:
        notification = (
            f"ℹ️ Your reservation (#{rid}) is no longer active "
            f"(status: {status}). "
            "Please contact CityPark on 0800 123 4567 for assistance."
        )

    # Push notification into chat history and clear the pending ID.
    st.session_state.messages.append({"role": "assistant", "content": notification})
    st.session_state.conv_state = {
        **st.session_state.conv_state,
        "pending_reservation_id": None,
    }
    notified.add(rid)
    logger.info("Auto-poll delivered status notification for reservation %d: %s", rid, status)
    st.rerun()


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
            prev_rid = st.session_state.conv_state.get("pending_reservation_id")
            new_state = graph.invoke(
                {**st.session_state.conv_state, "user_input": user_input}
            )
            st.session_state.conv_state = new_state
            response = new_state.get("response", "")

            # Stage 4: if a reservation was just saved this turn, start its workflow
            new_rid = new_state.get("pending_reservation_id")
            if new_rid and new_rid != prev_rid:
                try:
                    from src.workflow.graph import start_reservation_workflow
                    start_reservation_workflow(
                        new_rid,
                        new_state.get("reservation", {}),
                    )
                    logger.info("Workflow started for reservation %d.", new_rid)
                except Exception as wf_exc:
                    logger.exception(
                        "Failed to start workflow for reservation %d", new_rid
                    )
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
_render_top_bar()   # Admin button always at the top of the main area

if st.session_state.get("show_admin_panel"):
    # ── Admin panel view ─────────────────────────────────────────────────────
    if _render_admin_login():
        _render_admin_panel()
else:
    # ── Normal user chat view ────────────────────────────────────────────────
    _poll_reservation_status()   # auto-refresh every 3 s when a reservation is pending
    _render_chat()
