"""
LangChain read-only tools for the admin agent.

IMPORTANT: Approve and reject operations are intentionally NOT exposed as
LangChain tools.  They are performed only through explicit authenticated
button actions in the UI (admin_app.py).  This guarantees that the LLM
can never independently approve or reject a reservation.

Tools provided:
  list_pending     — list reservations awaiting a decision
  get_details      — retrieve full details of one reservation
  list_all         — list reservations with an optional status filter
"""

from langchain_core.tools import tool

from src.database.queries import (
    get_all_reservations,
    get_pending_reservations,
    get_reservation_by_id,
)


@tool
def list_pending(dummy: str = "") -> str:
    """
    List all parking reservations that are currently awaiting an
    administrator decision (status = pending).

    Returns a formatted text summary.  Pass an empty string as the argument.
    """
    rows = get_pending_reservations()
    if not rows:
        return "There are no pending reservations at this time."

    lines = [f"**{len(rows)} pending reservation(s):**\n"]
    for r in rows:
        lines.append(
            f"  • #{r['id']} | {r['first_name']} {r['surname']} | "
            f"{r['vehicle_reg']} ({r['vehicle_type']}) | "
            f"{r['start_datetime']} → {r['end_datetime']} | "
            f"submitted {r['created_at']}"
        )
    return "\n".join(lines)


@tool
def get_details(reservation_id: int) -> str:
    """
    Retrieve the full details of a single reservation by its numeric ID.

    Args:
        reservation_id: The integer ID of the reservation (e.g. 3).
    """
    row = get_reservation_by_id(reservation_id)
    if row is None:
        return f"No reservation found with ID #{reservation_id}."

    lines = [f"**Reservation #{row['id']}**\n"]
    field_labels = {
        "first_name":     "First name",
        "surname":        "Surname",
        "vehicle_type":   "Vehicle type",
        "vehicle_reg":    "Registration",
        "start_datetime": "Parking start",
        "end_datetime":   "Parking end",
        "status":         "Status",
        "created_at":     "Submitted at",
        "admin_notes":    "Admin notes",
        "decided_at":     "Decision time",
        "decided_by":     "Decided by",
    }
    for key, label in field_labels.items():
        value = row.get(key)
        if value is not None:
            lines.append(f"  • {label}: {value}")
    return "\n".join(lines)


@tool
def list_all(status: str = "") -> str:
    """
    List reservations with an optional status filter.

    Args:
        status: One of pending, confirmed, rejected, cancelled, expired.
                Leave empty to list all reservations.
    """
    rows = get_all_reservations(status=status or None)
    if not rows:
        qualifier = f" with status '{status}'" if status else ""
        return f"No reservations found{qualifier}."

    label = f"status='{status}'" if status else "all statuses"
    lines = [f"**{len(rows)} reservation(s) — {label}:**\n"]
    for r in rows:
        lines.append(
            f"  • #{r['id']} | {r['first_name']} {r['surname']} | "
            f"{r['vehicle_reg']} | {r['status']} | "
            f"{r['start_datetime']} → {r['end_datetime']}"
        )
    return "\n".join(lines)


# The set of tools the admin LLM agent may call.
ADMIN_TOOLS = [list_pending, get_details, list_all]
