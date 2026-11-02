"""
LangGraph node functions for the reservation lifecycle workflow.

Nodes:
  await_admin_node — calls interrupt(); graph pauses here waiting for admin
  approve_node     — calls approve_reservation(); routes to export or END on error
  reject_node      — calls reject_reservation()
  mcp_export_node  — calls call_write_approval_record(); non-fatal on failure
"""

from __future__ import annotations

from langgraph.types import interrupt

from src.config import get_logger
from src.workflow.state import ReservationWorkflowState

logger = get_logger(__name__)


def await_admin_node(state: ReservationWorkflowState) -> dict:
    """
    Pause the workflow until an administrator makes a decision.

    Publishes the reservation details as the interrupt payload so any
    listener (admin UI, future notification) can see what is waiting.

    The resume payload must be a dict with keys:
        decision    : 'confirmed' | 'rejected'
        admin_id    : str
        admin_notes : str  (approval notes or rejection reason)
    """
    payload = interrupt({
        "reservation_id": state["reservation_id"],
        "reservation":    state["reservation"],
    })
    # payload is whatever was passed to Command(resume=...)
    return {
        "decision":    payload.get("decision"),
        "admin_id":    payload.get("admin_id"),
        "admin_notes": payload.get("admin_notes", ""),
    }


def approve_node(state: ReservationWorkflowState) -> dict:
    """
    Call approve_reservation() from the existing queries module.

    Sets state["error"] on capacity_exceeded, already_decided, or db_error.
    On success, state["error"] is None and the graph proceeds to mcp_export_node.
    """
    from src.database.queries import approve_reservation

    result = approve_reservation(
        state["reservation_id"],
        admin_id=state["admin_id"] or "workflow",
        notes=state["admin_notes"],
    )
    if result["success"]:
        logger.info(
            "Workflow: reservation %d approved.", state["reservation_id"]
        )
        return {"error": None}
    else:
        err = result.get("error", "db_error")
        logger.warning(
            "Workflow: approve_reservation failed for %d: %s",
            state["reservation_id"], err,
        )
        return {"error": err}


def reject_node(state: ReservationWorkflowState) -> dict:
    """
    Call reject_reservation() from the existing queries module.

    Requires a non-empty admin_notes (used as the rejection reason).
    Sets state["error"] on db_error or missing reason.
    """
    from src.database.queries import reject_reservation

    reason = state["admin_notes"].strip()
    if not reason:
        logger.warning(
            "Workflow: reject called for %d without a reason.",
            state["reservation_id"],
        )
        return {"error": "reason_required"}

    result = reject_reservation(
        state["reservation_id"],
        admin_id=state["admin_id"] or "workflow",
        reason=reason,
    )
    if result["success"]:
        logger.info(
            "Workflow: reservation %d rejected.", state["reservation_id"]
        )
        return {"error": None}
    else:
        err = result.get("error", "db_error")
        logger.warning(
            "Workflow: reject_reservation failed for %d: %s",
            state["reservation_id"], err,
        )
        return {"error": err}


def mcp_export_node(state: ReservationWorkflowState) -> dict:
    """
    Write the approved reservation to the approval log via the MCP tool.

    Non-fatal: if the export fails, the workflow still completes and
    state["export_result"]["success"] == False.  The Retry Export button
    in admin_app.py handles subsequent attempts independently.

    call_write_approval_record() uses asyncio.run() internally.  This is
    safe here because the workflow graph is invoked synchronously (graph.invoke,
    not graph.ainvoke), so this node runs in a synchronous context.
    """
    from src.mcp_server.client import call_write_approval_record

    try:
        result = call_write_approval_record(state["reservation_id"])
    except Exception as exc:
        logger.exception(
            "Workflow: MCP export raised for reservation %d", state["reservation_id"]
        )
        result = {"success": False, "error": "client_error", "message": str(exc)}

    if not result.get("success"):
        logger.warning(
            "Workflow: export failed for reservation %d: %s",
            state["reservation_id"], result.get("error"),
        )
    else:
        logger.info(
            "Workflow: export complete for reservation %d.", state["reservation_id"]
        )

    return {"export_result": result}
