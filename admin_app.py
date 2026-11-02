"""
CityPark Administrator Dashboard.

Simple, password-gated dashboard for reviewing and deciding on pending
parking reservations.  No LLM chatbot — just the reservation list with
Approve / Reject buttons and the Retry Export action.

Run with:
    streamlit run admin_app.py --server.port 8502
"""

import streamlit as st

st.set_page_config(
    page_title="CityPark Admin",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.admin.auth import is_admin_configured, verify_admin_password
from src.admin.dashboard import render_reservations
from src.config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------

def _auth_gate() -> bool:
    """Show login form; return True once authenticated."""
    if st.session_state.get("admin_authenticated"):
        return True

    st.title("🔐 CityPark Administrator Login")

    if not is_admin_configured():
        st.error(
            "⚠️ `ADMIN_PASSWORD` is not set in your `.env` file.  "
            "Set it before using the admin dashboard."
        )
        return False

    with st.form("login_form"):
        pwd = st.text_input("Admin password", type="password",
                            placeholder="Enter admin password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if verify_admin_password(pwd):
            st.session_state.admin_authenticated = True
            st.session_state.admin_id = "admin"
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


# ---------------------------------------------------------------------------
# Sidebar — logout only
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("🔐 CityPark Admin")
        st.caption(f"Logged in as: **{st.session_state.get('admin_id', 'admin')}**")
        st.divider()
        if st.button("🚪 Log out", use_container_width=True):
            for key in ("admin_authenticated", "admin_id"):
                st.session_state.pop(key, None)
            st.rerun()
        st.caption("CityPark · Admin Dashboard")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if not _auth_gate():
    st.stop()

st.title("🔐 CityPark Administrator Dashboard")
st.caption("Approve or reject pending reservations below.")

_render_sidebar()
render_reservations(st.session_state.get("admin_id", "admin"))
