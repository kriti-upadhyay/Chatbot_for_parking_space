"""
Reservation lifecycle workflow graph (Stage 4).

One workflow instance runs per reservation, identified by:
    thread_id = f"reservation_{reservation_id}"

The workflow pauses at await_admin_node (via interrupt()) and resumes when
the admin clicks Approve or Reject in admin_app.py.  State is persisted in
the existing storage/parking.db file via SqliteSaver so the workflow
survives application restarts.

Public API
----------
build_workflow_graph(checkpointer) — compile the StateGraph
start_reservation_workflow(rid, reservation) — start and immediately pause
get_workflow_graph_state(rid) — StateSnapshot | None (None = no checkpoint)
resume_reservation_workflow(rid, decision, admin_id, admin_notes) — admin decides
WORKFLOW_THREAD_ID(rid) — canonical thread_id for a reservation

Backward compatibility
----------------------
get_workflow_graph_state returns None for pre-Stage-4 reservations.  Callers
fall back to the existing direct approve_reservation() / call_write_approval_record()
path for those rows.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.config import get_logger, settings
from src.workflow.nodes import (
    approve_node,
    await_admin_node,
    mcp_export_node,
    reject_node,
)
from src.workflow.state import ReservationWorkflowState

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Thread-id convention
# ---------------------------------------------------------------------------

def WORKFLOW_THREAD_ID(reservation_id: int) -> str:
    """Canonical LangGraph thread_id for a reservation workflow."""
    return f"reservation_{reservation_id}"


def _cfg(reservation_id: int) -> dict:
    return {"configurable": {"thread_id": WORKFLOW_THREAD_ID(reservation_id)}}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _route_after_await(state: ReservationWorkflowState) -> str:
    if state.get("decision") == "confirmed":
        return "approve_node"
    return "reject_node"


def _route_after_approve(state: ReservationWorkflowState) -> str:
    """Go to export if approve succeeded; skip export on any error."""
    if state.get("error"):
        return END
    return "mcp_export_node"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_workflow_graph(checkpointer):
    """
    Compile the reservation lifecycle StateGraph.

    Args:
        checkpointer: any LangGraph-compatible checkpointer instance
                      (SqliteSaver or MemorySaver for tests).

    Returns:
        Compiled CompiledGraph.
    """
    builder = StateGraph(ReservationWorkflowState)

    builder.add_node("await_admin_node", await_admin_node)
    builder.add_node("approve_node",     approve_node)
    builder.add_node("reject_node",      reject_node)
    builder.add_node("mcp_export_node",  mcp_export_node)

    builder.add_edge(START, "await_admin_node")

    builder.add_conditional_edges(
        "await_admin_node",
        _route_after_await,
        {"approve_node": "approve_node", "reject_node": "reject_node"},
    )

    builder.add_conditional_edges(
        "approve_node",
        _route_after_approve,
        {"mcp_export_node": "mcp_export_node", END: END},
    )

    builder.add_edge("reject_node",      END)
    builder.add_edge("mcp_export_node",  END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.debug("Reservation workflow graph compiled.")
    return compiled


# ---------------------------------------------------------------------------
# SqliteSaver factory
# ---------------------------------------------------------------------------

@contextmanager
def _open_checkpointer() -> Generator:
    """
    Open a SqliteSaver context manager backed by the project SQLite database.

    Used as a context manager for each workflow operation so that the
    connection is properly closed afterwards.

    Raises:
        RuntimeError: if langgraph-checkpoint-sqlite is not installed.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "langgraph-checkpoint-sqlite is not installed in this Python environment. "
            "Run: pip install 'langgraph-checkpoint-sqlite==3.1.3'"
        ) from exc
    with SqliteSaver.from_conn_string(str(settings.db_path)) as saver:
        yield saver


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def start_reservation_workflow(
    reservation_id: int,
    reservation: dict,
) -> None:
    """
    Start the workflow for a newly saved reservation and immediately pause.

    Invokes the graph once; await_admin_node calls interrupt() on the first
    invocation so the graph suspends immediately and persists to disk.

    Args:
        reservation_id: the DB id returned by save_reservation()
        reservation:    the collected field values (snapshot for the workflow)
    """
    initial: ReservationWorkflowState = {
        "reservation_id": reservation_id,
        "reservation":    reservation,
        "decision":       None,
        "admin_id":       None,
        "admin_notes":    "",
        "export_result":  None,
        "error":          None,
    }
    with _open_checkpointer() as saver:
        graph = build_workflow_graph(saver)
        graph.invoke(initial, config=_cfg(reservation_id))

    logger.info(
        "Workflow started and paused for reservation %d.", reservation_id
    )


def get_workflow_graph_state(reservation_id: int):
    """
    Return the current StateSnapshot for a reservation workflow, or None.

    Returns None if:
    - No checkpoint exists for this reservation_id (pre-Stage-4 reservation), or
    - langgraph-checkpoint-sqlite is not installed in this environment.

    Use ``snap.next == ('await_admin_node',)`` to confirm the workflow is
    still waiting for an admin decision.
    """
    try:
        with _open_checkpointer() as saver:
            graph = build_workflow_graph(saver)
            snap = graph.get_state(_cfg(reservation_id))
    except RuntimeError:
        logger.warning(
            "SqliteSaver unavailable — workflow state not accessible for reservation %d. "
            "Install langgraph-checkpoint-sqlite==3.1.3.",
            reservation_id,
        )
        return None

    if snap.metadata is None:
        return None
    return snap


def resume_reservation_workflow(
    reservation_id: int,
    decision: str,
    admin_id: str,
    admin_notes: str,
) -> dict:
    """
    Resume a paused workflow with an administrator's decision.

    Args:
        reservation_id: the reservation being decided
        decision:       'confirmed' or 'rejected'
        admin_id:       authenticated admin identifier
        admin_notes:    approval notes or mandatory rejection reason

    Returns:
        dict with keys:
          workflow_found (bool)  — False if no checkpoint exists (backward compat)
          final_state (dict)     — final workflow state (only when workflow_found)
    """
    try:
        with _open_checkpointer() as saver:
            graph = build_workflow_graph(saver)
            cfg = _cfg(reservation_id)

            snap = graph.get_state(cfg)
            if snap.metadata is None:
                logger.debug(
                    "No workflow checkpoint for reservation %d; fallback required.",
                    reservation_id,
                )
                return {"workflow_found": False}

            resume_payload = {
                "decision":    decision,
                "admin_id":    admin_id,
                "admin_notes": admin_notes,
            }
            final = graph.invoke(Command(resume=resume_payload), config=cfg)
    except RuntimeError:
        logger.warning(
            "SqliteSaver unavailable — cannot resume workflow for reservation %d. "
            "Using direct fallback.",
            reservation_id,
        )
        return {"workflow_found": False}

    logger.info(
        "Workflow resumed for reservation %d: decision=%s", reservation_id, decision
    )
    return {"workflow_found": True, "final_state": final}
