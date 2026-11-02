"""
Shared admin dashboard rendering.

Contains the reservation table with approve/reject/retry-export logic,
extracted so it can be embedded in both admin_app.py and streamlit_app.py
without duplicating the approval code.

Approval follows the Stage 4 workflow-first pattern:
  • If a workflow checkpoint exists and is paused → resume it (handles approve + MCP export).
  • If a workflow checkpoint is already completed  → warn, do nothing.
  • If no checkpoint (pre-Stage-4 reservation)     → direct DB call + MCP export fallback.
"""

from __future__ import annotations

import streamlit as st

from src.config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(dt_str: str | None) -> str:
    return dt_str or "—"


def _badge(status: str) -> str:
    icons = {
        "pending":   "🟡",
        "confirmed": "🟢",
        "rejected":  "🔴",
        "cancelled": "⚫",
        "expired":   "⚪",
    }
    return f"{icons.get(status, '⬜')} {status.title()}"


# ---------------------------------------------------------------------------
# Reservation table
# ---------------------------------------------------------------------------

def render_reservations(admin_id: str) -> None:
    """
    Render the full reservation table with approve / reject / retry-export
    buttons.  *admin_id* is the authenticated administrator identifier stored
    in session state.
    """
    st.subheader("📋 Reservations")

    from src.database.queries import get_all_reservations

    status_filter = st.selectbox(
        "Filter by status",
        options=["All", "Pending", "Confirmed", "Rejected", "Cancelled"],
        index=0,
        key="admin_status_filter",
    )
    filter_val = None if status_filter == "All" else status_filter.lower()
    rows = get_all_reservations(status=filter_val)

    if not rows:
        st.info("No reservations found.")
        return

    st.caption(f"{len(rows)} reservation(s) shown")

    for r in rows:
        rid    = r["id"]
        status = r["status"]

        with st.expander(
            f"#{rid} · {r['first_name']} {r['surname']} · "
            f"{r['vehicle_reg']} ({r['vehicle_type'].replace('_', ' ').title()}) · "
            f"{_badge(status)}",
            expanded=(status == "pending"),
        ):
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**Name:** {r['first_name']} {r['surname']}")
                st.markdown(f"**Vehicle type:** {r['vehicle_type'].replace('_',' ').title()}")
                st.markdown(f"**Registration:** {r['vehicle_reg']}")
                st.markdown(f"**Start:** {_fmt(r.get('start_datetime'))}")
                st.markdown(f"**End:** {_fmt(r.get('end_datetime'))}")
            with col_r:
                st.markdown(f"**Status:** {_badge(status)}")
                st.markdown(f"**Submitted:** {_fmt(r.get('created_at'))}")
                if r.get("decided_by"):
                    st.markdown(f"**Decided by:** {r['decided_by']}")
                    st.markdown(f"**Decision time:** {_fmt(r.get('decided_at'))}")
                if r.get("admin_notes"):
                    st.markdown(f"**Notes:** {r['admin_notes']}")

            # ── Approve / Reject — pending only ──────────────────────────────
            if status == "pending":
                st.divider()
                bcol1, bcol2 = st.columns(2)

                with bcol1:
                    if st.button("✅ Approve", key=f"approve_{rid}",
                                 use_container_width=True, type="primary"):
                        _do_approve(rid, admin_id)
                        st.rerun()

                with bcol2:
                    reason = st.text_input(
                        "Rejection reason (required)",
                        key=f"reason_{rid}",
                        placeholder="e.g. No EV spaces available on that date",
                    )
                    if st.button("❌ Reject", key=f"reject_{rid}",
                                 use_container_width=True):
                        if not reason.strip():
                            st.error("Please enter a rejection reason before rejecting.")
                        else:
                            _do_reject(rid, admin_id, reason.strip())
                            st.rerun()

            # ── Retry Export — confirmed with incomplete export ───────────────
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
# Approve / reject helpers (Stage 4 workflow-first, direct fallback)
# ---------------------------------------------------------------------------

def _do_approve(rid: int, admin_id: str) -> None:
    from src.workflow.graph import get_workflow_graph_state, resume_reservation_workflow

    wf_snap = get_workflow_graph_state(rid)

    if wf_snap is None:
        # Pre-Stage-4 reservation — direct DB call + MCP export.
        from src.database.queries import approve_reservation
        from src.mcp_server.client import call_write_approval_record
        result = approve_reservation(rid, admin_id=admin_id, notes="")
        if result["success"]:
            st.success(f"Reservation #{rid} approved.")
            logger.info("Admin approved reservation %d", rid)
            export = call_write_approval_record(rid)
            if not export.get("success"):
                st.warning(
                    f"Reservation #{rid} approved, but the export log could not be "
                    f"written ({export.get('error', 'unknown')}). "
                    "Use Retry Export to try again."
                )
        else:
            _show_approve_error(result.get("error", "unknown"))

    elif wf_snap.next == ():
        st.warning("This reservation was already decided (workflow complete).")

    else:
        wf = resume_reservation_workflow(rid, "confirmed", admin_id, admin_notes="")
        fs = wf.get("final_state", {})
        if fs.get("error"):
            _show_approve_error(fs["error"])
        else:
            st.success(f"Reservation #{rid} approved.")
            logger.info("Workflow approved reservation %d", rid)
            export = fs.get("export_result") or {}
            if not export.get("success") and not export.get("already_exported"):
                st.warning(
                    f"Reservation #{rid} approved, but the export log could not be "
                    "written. Use Retry Export to try again."
                )


def _do_reject(rid: int, admin_id: str, reason: str) -> None:
    from src.workflow.graph import get_workflow_graph_state, resume_reservation_workflow

    wf_snap = get_workflow_graph_state(rid)

    if wf_snap is None:
        from src.database.queries import reject_reservation
        result = reject_reservation(rid, admin_id=admin_id, reason=reason)
        if result["success"]:
            st.success(f"Reservation #{rid} rejected.")
            logger.info("Admin rejected reservation %d", rid)
        else:
            _show_reject_error(result.get("error", "unknown"))

    elif wf_snap.next == ():
        st.warning("This reservation was already decided (workflow complete).")

    else:
        wf = resume_reservation_workflow(rid, "rejected", admin_id, reason)
        fs = wf.get("final_state", {})
        if fs.get("error"):
            _show_reject_error(fs["error"])
        else:
            st.success(f"Reservation #{rid} rejected.")
            logger.info("Workflow rejected reservation %d", rid)


def _show_approve_error(err: str) -> None:
    if err == "already_decided":
        st.warning("This reservation was already decided.")
    elif err == "capacity_exceeded":
        st.error("Cannot approve — capacity is now exceeded.")
    else:
        st.error(f"Approval failed: {err}")


def _show_reject_error(err: str) -> None:
    if err == "already_decided":
        st.warning("This reservation was already decided.")
    elif err == "reason_required":
        st.error("A rejection reason is required.")
    else:
        st.error(f"Rejection failed: {err}")
