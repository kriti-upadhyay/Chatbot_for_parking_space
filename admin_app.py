"""
CityPark Administrator Dashboard (Stage 2).

Password-gated Streamlit page for reviewing and deciding on pending
parking reservations.

Run with:
    streamlit run admin_app.py

The LLM agent (read-only) is available in the sidebar chat.
Approve / Reject buttons are the only way to write a decision to the database.
"""

import streamlit as st

st.set_page_config(
    page_title="CityPark Admin",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import get_logger, settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------

def _auth_gate() -> bool:
    """
    Render a password prompt on first visit.
    Returns True if the admin is authenticated for this session.
    """
    if st.session_state.get("admin_authenticated"):
        return True

    st.title("🔐 CityPark Administrator Login")

    from src.admin.auth import is_admin_configured
    if not is_admin_configured():
        st.error(
            "⚠️ `ADMIN_PASSWORD` is not set in your `.env` file.  "
            "Set it before using the admin dashboard."
        )
        return False

    with st.form("login_form"):
        # password_input hides the characters; value never stored beyond this call
        pwd = st.text_input("Admin password", type="password",
                            placeholder="Enter admin password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        from src.admin.auth import verify_admin_password
        if verify_admin_password(pwd):
            st.session_state.admin_authenticated = True
            st.session_state.admin_id = "admin"   # single-admin deployment
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


# ---------------------------------------------------------------------------
# Session initialisation
# ---------------------------------------------------------------------------

def _init():
    if "admin_init" in st.session_state:
        return
    st.session_state.admin_messages = []
    st.session_state.admin_graph_state = None
    st.session_state.admin_init = True
    st.session_state.reject_reason = {}   # {reservation_id: reason_text}


@st.cache_resource(show_spinner=False)
def _load_admin_graph():
    from src.admin.graph import build_admin_graph
    return build_admin_graph()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_dt(dt_str: str | None) -> str:
    return dt_str or "—"


def _status_badge(status: str) -> str:
    colours = {
        "pending":   "🟡",
        "confirmed": "🟢",
        "rejected":  "🔴",
        "cancelled": "⚫",
        "expired":   "⚪",
    }
    return f"{colours.get(status, '⬜')} {status.title()}"


# ---------------------------------------------------------------------------
# Reservation table
# ---------------------------------------------------------------------------

def _render_reservations():
    st.subheader("📋 Reservations")

    from src.database.queries import get_all_reservations

    # Status filter
    status_filter = st.selectbox(
        "Filter by status",
        options=["All", "Pending", "Confirmed", "Rejected", "Cancelled"],
        index=0,
    )
    filter_val = None if status_filter == "All" else status_filter.lower()
    rows = get_all_reservations(status=filter_val)

    if not rows:
        st.info("No reservations found.")
        return

    st.caption(f"{len(rows)} reservation(s) shown")

    for r in rows:
        rid = r["id"]
        status = r["status"]

        with st.expander(
            f"#{rid} · {r['first_name']} {r['surname']} · "
            f"{r['vehicle_reg']} ({r['vehicle_type'].replace('_',' ').title()}) · "
            f"{_status_badge(status)}",
            expanded=(status == "pending"),
        ):
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**Name:** {r['first_name']} {r['surname']}")
                st.markdown(f"**Vehicle type:** {r['vehicle_type'].replace('_',' ').title()}")
                st.markdown(f"**Registration:** {r['vehicle_reg']}")
                st.markdown(f"**Start:** {_format_dt(r.get('start_datetime'))}")
                st.markdown(f"**End:** {_format_dt(r.get('end_datetime'))}")
            with col_r:
                st.markdown(f"**Status:** {_status_badge(status)}")
                st.markdown(f"**Submitted:** {_format_dt(r.get('created_at'))}")
                if r.get("decided_by"):
                    st.markdown(f"**Decided by:** {r['decided_by']}")
                    st.markdown(f"**Decision time:** {_format_dt(r.get('decided_at'))}")
                if r.get("admin_notes"):
                    st.markdown(f"**Notes:** {r['admin_notes']}")

            # Approve / Reject buttons — only for pending reservations
            if status == "pending":
                st.divider()
                bcol1, bcol2 = st.columns(2)

                with bcol1:
                    if st.button("✅ Approve", key=f"approve_{rid}",
                                 use_container_width=True, type="primary"):
                        from src.database.queries import approve_reservation
                        from src.mcp_server.client import call_write_approval_record
                        result = approve_reservation(
                            rid,
                            admin_id=st.session_state.get("admin_id", "admin"),
                            notes="",
                        )
                        if result["success"]:
                            st.success(f"Reservation #{rid} approved.")
                            logger.info("Admin approved reservation %d", rid)
                            # Export record via MCP tool
                            export = call_write_approval_record(rid)
                            if not export.get("success"):
                                err = export.get("error", "unknown")
                                st.warning(
                                    f"Reservation #{rid} approved, but the export "
                                    f"log could not be written ({err}). "
                                    "Use the Retry Export button below to try again."
                                )
                        else:
                            err = result.get("error", "unknown")
                            if err == "already_decided":
                                st.warning("This reservation was already decided.")
                            elif err == "capacity_exceeded":
                                st.error("Cannot approve — capacity is now exceeded.")
                            else:
                                st.error(f"Approval failed: {err}")
                        st.rerun()

                with bcol2:
                    reason_key = f"reason_{rid}"
                    reason = st.text_input(
                        "Rejection reason (required)",
                        key=reason_key,
                        placeholder="e.g. No EV spaces available on that date",
                    )
                    if st.button("❌ Reject", key=f"reject_{rid}",
                                 use_container_width=True):
                        if not reason.strip():
                            st.error("Please enter a rejection reason before rejecting.")
                        else:
                            from src.database.queries import reject_reservation
                            result = reject_reservation(
                                rid,
                                admin_id=st.session_state.get("admin_id", "admin"),
                                reason=reason.strip(),
                            )
                            if result["success"]:
                                st.success(f"Reservation #{rid} rejected.")
                                logger.info("Admin rejected reservation %d", rid)
                            else:
                                err = result.get("error", "unknown")
                                if err == "already_decided":
                                    st.warning("This reservation was already decided.")
                                elif err == "reason_required":
                                    st.error("A rejection reason is required.")
                                else:
                                    st.error(f"Rejection failed: {err}")
                            st.rerun()

            # Retry Export button — shown for confirmed reservations with failed/missing export
            if status == "confirmed":
                exported_at = r.get("exported_at")
                if exported_at is None or exported_at == "EXPORTING":
                    st.divider()
                    st.caption("⚠️ Export to approval log has not completed.")
                    if st.button("🔄 Retry Export", key=f"retry_export_{rid}",
                                 use_container_width=True):
                        from src.mcp_server.client import call_write_approval_record
                        export = call_write_approval_record(rid)
                        if export.get("success"):
                            st.success(f"Reservation #{rid} exported successfully.")
                        elif export.get("already_exported"):
                            st.info("Already exported.")
                        else:
                            st.error(f"Export failed: {export.get('error', 'unknown')}")
                        st.rerun()


# ---------------------------------------------------------------------------
# Admin LLM chat sidebar
# ---------------------------------------------------------------------------

def _render_admin_chat():
    st.sidebar.title("🤖 Admin Assistant")
    st.sidebar.caption(
        "Ask me about reservations.  I can list, filter, and explain — "
        "but only the buttons above can approve or reject."
    )

    # Message history
    for msg in st.session_state.admin_messages:
        role = msg["role"]
        with st.sidebar.chat_message(role, avatar="🔐" if role == "assistant" else "👤"):
            st.sidebar.markdown(msg["content"])

    user_input = st.sidebar.chat_input("Ask about reservations…")
    if not user_input:
        return

    st.session_state.admin_messages.append({"role": "user", "content": user_input})
    with st.sidebar.chat_message("user", avatar="👤"):
        st.sidebar.markdown(user_input)

    with st.sidebar.spinner("Thinking…"):
        try:
            from langchain_core.messages import HumanMessage
            from src.admin.state import AdminState

            graph = _load_admin_graph()
            admin_id = st.session_state.get("admin_id", "admin")

            # Build or continue conversation state
            if st.session_state.admin_graph_state is None:
                from src.admin.state import empty_admin_state
                state = empty_admin_state(admin_id=admin_id)
            else:
                state = st.session_state.admin_graph_state

            state = {**state, "messages": state.get("messages", []) + [
                HumanMessage(content=user_input)
            ]}
            result = graph.invoke(state)
            st.session_state.admin_graph_state = result
            response = result.get("last_result", "")
        except Exception as exc:
            logger.exception("Admin agent error")
            response = f"Error: {exc}"

    if response:
        st.session_state.admin_messages.append(
            {"role": "assistant", "content": response}
        )
        with st.sidebar.chat_message("assistant", avatar="🔐"):
            st.sidebar.markdown(response)

    st.rerun()


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

def _render_logout():
    st.sidebar.divider()
    if st.sidebar.button("🚪 Log out", use_container_width=True):
        for key in ("admin_authenticated", "admin_id", "admin_messages",
                    "admin_graph_state", "admin_init"):
            st.session_state.pop(key, None)
        st.rerun()
    st.sidebar.caption("Stage 2 · CityPark Admin · LangGraph")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_init()

if not _auth_gate():
    st.stop()

st.title("🔐 CityPark Administrator Dashboard")
st.caption(
    f"Logged in as: **{st.session_state.get('admin_id', 'admin')}** · "
    "Use the buttons below to approve or reject reservations."
)

_render_reservations()
_render_admin_chat()
_render_logout()
