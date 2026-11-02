"""
State for the reservation lifecycle workflow graph (Stage 4).

One instance of this state exists per reservation, identified by
thread_id = f"reservation_{reservation_id}" in the checkpointer.

All fields are JSON-serialisable so SqliteSaver can persist them.
"""

from typing import TypedDict


class ReservationWorkflowState(TypedDict):
    # Set at workflow start and never changed.
    reservation_id: int
    reservation: dict          # snapshot of reservation fields at submission time

    # Set when await_admin_node resumes.
    decision: str | None       # None while pending; 'confirmed' or 'rejected'
    admin_id: str | None
    admin_notes: str           # approval notes or rejection reason

    # Set by mcp_export_node (approve path only).
    export_result: dict | None

    # Set if approve_node or reject_node encounters a DB error.
    error: str | None
