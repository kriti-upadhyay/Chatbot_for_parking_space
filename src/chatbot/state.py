"""
Conversation state shared across all LangGraph nodes.

The `messages` field uses LangGraph's `add_messages` reducer so that each
node can append messages without replacing the full history.

All other fields are overwritten on each graph invocation — they hold the
*current turn's* context, not accumulated history.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class ConversationState(TypedDict):
    # ── Persistent across turns (managed by add_messages reducer) ──────────
    messages: Annotated[list, add_messages]  # full message history

    # ── Per-turn inputs ─────────────────────────────────────────────────────
    user_input: str          # raw text entered this turn
    intent: str | None       # classified routing intent for this turn
    response: str            # final text response shown to the user

    # ── Reservation form state (persists across multi-turn collection) ──────
    reservation: dict        # collected field values; keys = RESERVATION_FIELDS
    reservation_active: bool # True while we are collecting reservation details
    next_field_to_ask: str | None  # next field the bot should prompt for
    validation_errors: list[str]   # errors from the previous field attempt

    # ── Stage 2: pending reservation tracking ───────────────────────────────
    # Set to the DB id after a successful save; cleared once the admin
    # decision has been delivered to the user (exactly once).
    pending_reservation_id: int | None


# Ordered list of reservation fields the bot collects.
RESERVATION_FIELDS: list[str] = [
    "first_name",
    "surname",
    "vehicle_type",   # shown as a dynamic DB-driven menu
    "vehicle_reg",
    "start_datetime",
    "end_datetime",
]

# Human-readable prompts for each field.
# vehicle_type uses a dynamic prompt (generated in nodes.py from live DB data).
FIELD_PROMPTS: dict[str, str] = {
    "first_name":     "What is your first name?",
    "surname":        "What is your surname?",
    "vehicle_type":   "(see available types below)",  # replaced at runtime
    "vehicle_reg":    "What is your vehicle registration number? (e.g. AB12 CDE)",
    "start_datetime": "What date and time would you like to start parking?\n"
                      "(Please use the format: YYYY-MM-DD HH:MM  e.g. 2024-06-15 09:00)",
    "end_datetime":   "What date and time would you like to finish parking?\n"
                      "(Please use the format: YYYY-MM-DD HH:MM  e.g. 2024-06-15 17:00)",
}

# Display labels used in the confirmation summary.
FIELD_LABELS: dict[str, str] = {
    "first_name":     "First name",
    "surname":        "Surname",
    "vehicle_type":   "Vehicle type",
    "vehicle_reg":    "Vehicle registration",
    "start_datetime": "Parking start",
    "end_datetime":   "Parking end",
}


def empty_state() -> ConversationState:
    """Return a fresh, fully-initialised ConversationState."""
    return ConversationState(
        messages=[],
        user_input="",
        intent=None,
        response="",
        reservation={},
        reservation_active=False,
        next_field_to_ask=None,
        validation_errors=[],
        pending_reservation_id=None,
    )
