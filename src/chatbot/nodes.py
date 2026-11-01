"""
LangGraph node functions.

Each node takes the full ConversationState, does work, and returns a dict
of fields to update.  LangGraph merges the returned dict into the state
(using the `add_messages` reducer for the `messages` field).

Nodes defined here:
  guardrail_node      — injection detection + PII check (Presidio)
  intent_router_node  — classify intent and set state["intent"]
  rag_node            — answer with RAG chain (output PII-scrubbed)
  sql_node            — answer from SQLite dynamic data
  reservation_node    — multi-turn reservation form collection
                        (saves confirmed bookings to the database)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage

from src.chatbot.router import classify_intent
from src.chatbot.state import (
    FIELD_LABELS,
    FIELD_PROMPTS,
    RESERVATION_FIELDS,
    ConversationState,
)
from src.config import get_logger
from src.guardrails.injection import get_injection_detector
from src.guardrails.pii import get_pii_detector
from src.guardrails.validators import validate_reservation_field

if TYPE_CHECKING:
    from src.rag.chain import RagChain

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Guardrail node
# ---------------------------------------------------------------------------

def guardrail_node(state: ConversationState) -> dict:
    """
    Input safety gate — runs on every user message before routing.

    1. Injection detection (scored pattern matching).
    2. PII check: logs a warning if personal data appears outside a
       reservation context so it is never routed to the vector store.
    """
    user_input = state.get("user_input", "").strip()
    reservation_active = state.get("reservation_active", False)

    injection_result = get_injection_detector().check(user_input)
    if injection_result.is_injection:
        blocked_msg = (
            "I'm sorry, I can't process that request. "
            "Please ask me about CityPark parking services."
        )
        return {
            "response": blocked_msg,
            "intent": "blocked",
            "messages": [
                HumanMessage(content=user_input),
                AIMessage(content=blocked_msg),
            ],
        }

    if not reservation_active:
        pii_result = get_pii_detector().check(user_input)
        if pii_result.has_pii:
            logger.warning(
                "PII detected in non-reservation input (entities: %s). "
                "Input will NOT be stored in vector database.",
                pii_result.entity_types,
            )

    return {"messages": [HumanMessage(content=user_input)]}


# ---------------------------------------------------------------------------
# Intent router node
# ---------------------------------------------------------------------------

def intent_router_node(state: ConversationState) -> dict:
    """Classify the user's intent and update state["intent"]."""
    user_input = state.get("user_input", "")
    reservation_active = state.get("reservation_active", False)
    intent = classify_intent(user_input, reservation_active=reservation_active)
    return {"intent": intent}


# ---------------------------------------------------------------------------
# RAG node
# ---------------------------------------------------------------------------

def make_rag_node(rag_chain: RagChain):
    """Factory that returns a rag_node bound to *rag_chain*."""
    def rag_node(state: ConversationState) -> dict:
        question = state.get("user_input", "")
        result = rag_chain.query(question)
        answer = get_pii_detector().scrub_output(result.answer)
        return {
            "response": answer,
            "messages": [AIMessage(content=answer)],
        }
    return rag_node


# ---------------------------------------------------------------------------
# SQL node — dynamic info (prices, hours, availability)
# ---------------------------------------------------------------------------

_PRICE_KW = frozenset({"price", "cost", "how much", "rate", "charge", "fee", "tariff", "pay"})
_HOURS_KW = frozenset({"hour", "open", "close", "when", "time", "schedule", "staff"})
_AVAIL_KW = frozenset({"available", "availability", "space", "free", "full", "capacity", "how many"})


def _detect_dynamic_topic(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in _PRICE_KW):
        return "prices"
    if any(kw in t for kw in _HOURS_KW):
        return "hours"
    if any(kw in t for kw in _AVAIL_KW):
        return "availability"
    return "prices"


def _format_prices(space_type: str | None = None) -> str:
    from src.database.queries import get_prices

    rows = get_prices(space_type=space_type or "standard")
    label = (space_type or "Standard").replace("_", " ").title()
    lines = [f"**{label} parking rates at CityPark:**"]
    if not rows:
        return "I'm sorry, I couldn't retrieve pricing information right now."
    for row in rows:
        lines.append(f"  • {row['description']}: £{row['price_gbp']:.2f}")
    if not space_type or space_type == "standard":
        lines.append(
            "\nDiscounts are available for Blue Badge holders, motorcycles, and "
            "EV charging bays. Ask me for specific rates."
        )
    return "\n".join(lines)


def _format_hours() -> str:
    from src.database.queries import get_working_hours

    rows = get_working_hours()
    lines = [
        "**CityPark opening hours:**",
        "The facility is open **24 hours a day, 7 days a week**, including bank holidays.",
        "",
        "Security staff are on site during these hours:",
    ]
    day_labels = {
        "weekday": "Monday – Friday", "saturday": "Saturday",
        "sunday": "Sunday", "bank_holiday": "Bank holidays",
    }
    for row in rows:
        label = day_labels.get(row["day_type"], row["day_type"].title())
        lines.append(f"  • {label}: {row['staffed_start']} – {row['staffed_end']}")
    lines.append("\nOutside staffed hours the facility remains monitored by CCTV.")
    return "\n".join(lines)


def _format_availability() -> str:
    """Format live availability computed from actual reservations."""
    from src.database.queries import get_available_vehicle_types

    rows = get_available_vehicle_types()
    if not rows:
        return "I'm sorry, I couldn't retrieve availability information right now."

    total_available = sum(r["available_spaces"] for r in rows)
    lines = ["**Current parking availability at CityPark:**"]
    for row in rows:
        avail = row["available_spaces"]
        total = row["total_spaces"]
        booked = row["booked_count"]
        pct = round(100.0 * booked / total, 1) if total else 0.0
        status = "FULL" if avail == 0 else f"{avail} of {total} available ({pct}% occupied)"
        lines.append(f"  • {row['display_name']}: {status}")

    lines.append(f"\n**Total available spaces: {total_available}**")
    lines.append("\nAvailability is calculated from confirmed reservations in real time.")
    return "\n".join(lines)


def sql_node(state: ConversationState) -> dict:
    """Answer dynamic questions directly from SQLite — no LLM involved."""
    user_input = state.get("user_input", "")
    topic = _detect_dynamic_topic(user_input)

    try:
        if topic == "prices":
            answer = _format_prices()
        elif topic == "hours":
            answer = _format_hours()
        else:
            answer = _format_availability()
    except Exception:
        logger.exception("SQL node failed to retrieve dynamic info.")
        answer = (
            "I'm sorry, I couldn't retrieve that information right now. "
            "Please try again or contact customer services on 0800 123 4567."
        )

    return {
        "response": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Reservation node helpers
# ---------------------------------------------------------------------------

def _build_vehicle_type_menu(db_path=None) -> tuple[str, list[dict]]:
    """
    Fetch available vehicle types from the DB and format a numbered menu.

    Returns (prompt_text, available_types_list).
    available_types_list may be empty if the facility is full.
    """
    from src.database.queries import get_available_vehicle_types

    types = get_available_vehicle_types(db_path=db_path)
    available = [t for t in types if t["available_spaces"] > 0]

    if not available:
        return (
            "I'm sorry, the car park is currently full for all vehicle types. "
            "Please try again later or call customer services on 0800 123 4567.",
            [],
        )

    lines = ["Please select your vehicle type (enter the number):"]
    for i, t in enumerate(available, 1):
        lines.append(
            f"  {i}. {t['display_name']} — {t['available_spaces']} space"
            f"{'s' if t['available_spaces'] != 1 else ''} available"
        )
    return "\n".join(lines), available


def _parse_vehicle_type(user_input: str, available: list[dict]) -> str | None:
    """
    Parse the user's vehicle type selection.

    Accepts a number (1–N) or the type name / display name (case-insensitive).
    Returns the canonical vehicle_type string, or None if unrecognised.
    """
    text = user_input.strip()

    # Try numeric selection
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(available):
            return available[idx]["vehicle_type"]
        return None

    # Try matching by vehicle_type or display_name
    text_lower = text.lower()
    for t in available:
        if (text_lower == t["vehicle_type"].lower()
                or text_lower == t["display_name"].lower()
                or text_lower in t["display_name"].lower()):
            return t["vehicle_type"]
    return None


def _validate_field(field: str, value: str, reservation: dict) -> str | None:
    """Dispatch to guardrails validators. Returns error string or None."""
    result = validate_reservation_field(field, value, reservation)
    return result.error_message if not result.is_valid else None


def _next_missing_field(reservation: dict) -> str | None:
    """Return the first RESERVATION_FIELDS entry not yet stored."""
    for field in RESERVATION_FIELDS:
        if not reservation.get(field):
            return field
    return None


def _build_confirmation(reservation: dict, reservation_id: int) -> str:
    lines = [
        f"Your reservation has been saved! (Reference: #{reservation_id})",
        "",
        "**Booking summary:**",
        "",
    ]
    for field in RESERVATION_FIELDS:
        label = FIELD_LABELS[field]
        value = reservation.get(field, "—")
        lines.append(f"  • {label}: {value}")

    lines += [
        "",
        "Your reservation is currently **pending administrator approval**.",
        "You will be notified once it is confirmed.",
        "",
        "To modify or cancel, contact CityPark customer services on 0800 123 4567 "
        f"and quote reference #{reservation_id}.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reservation node — multi-turn form collection + DB persistence
# ---------------------------------------------------------------------------

def reservation_node(state: ConversationState) -> dict:
    """
    Multi-turn reservation form collector.

    Turn 1  (intent == reservation, not yet active):
        Greet, ask for first_name.

    Turns 2–N (intent == reservation_input):
        Validate the previous answer, store it, ask for the next field.
        The vehicle_type field uses a live DB-driven numbered menu.

    Final turn (all 6 fields collected):
        Attempt to save the reservation to the database inside a transaction.
        On success: show confirmation with reservation ID.
        On failure: explain the specific error, keep the form open if fixable.
    """
    user_input  = state.get("user_input", "").strip()
    reservation = dict(state.get("reservation", {}))
    intent      = state.get("intent", "")
    next_field  = state.get("next_field_to_ask")

    # ── Initialise flow ──────────────────────────────────────────────────────
    if intent == "reservation" and not state.get("reservation_active"):
        first_field = RESERVATION_FIELDS[0]
        greeting = (
            "I'd be happy to help you reserve a parking space at CityPark!\n\n"
            + FIELD_PROMPTS[first_field]
        )
        return {
            "response": greeting,
            "reservation": {},
            "reservation_active": True,
            "next_field_to_ask": first_field,
            "validation_errors": [],
            "messages": [AIMessage(content=greeting)],
        }

    # ── Process a field answer ───────────────────────────────────────────────
    if next_field:

        # ── vehicle_type: special DB-driven handling ─────────────────────────
        if next_field == "vehicle_type":
            menu_text, available = _build_vehicle_type_menu()

            if not available:
                # Facility full — abort booking
                return {
                    "response": menu_text,
                    "reservation": {},
                    "reservation_active": False,
                    "next_field_to_ask": None,
                    "validation_errors": ["no_capacity"],
                    "messages": [AIMessage(content=menu_text)],
                }

            selected = _parse_vehicle_type(user_input, available)
            if selected is None:
                # Invalid selection — re-show menu
                error_msg = (
                    f"Sorry, '{user_input}' is not a valid selection.\n\n"
                    + menu_text
                )
                return {
                    "response": error_msg,
                    "reservation": reservation,
                    "reservation_active": True,
                    "next_field_to_ask": "vehicle_type",
                    "validation_errors": ["invalid_vehicle_type"],
                    "messages": [AIMessage(content=error_msg)],
                }

            # Confirm the selected type still has space (re-check after menu render)
            chosen = next((t for t in available if t["vehicle_type"] == selected), None)
            if chosen and chosen["available_spaces"] <= 0:
                retry_menu, retry_available = _build_vehicle_type_menu()
                if not retry_available:
                    reply = (
                        "The vehicle type you selected just became fully booked. "
                        + retry_menu
                    )
                else:
                    reply = (
                        f"Sorry, {chosen['display_name']} spaces just became fully booked.\n\n"
                        + retry_menu
                    )
                return {
                    "response": reply,
                    "reservation": reservation,
                    "reservation_active": bool(retry_available),
                    "next_field_to_ask": "vehicle_type" if retry_available else None,
                    "validation_errors": ["capacity_exceeded"],
                    "messages": [AIMessage(content=reply)],
                }

            reservation["vehicle_type"] = selected
            upcoming = _next_missing_field(reservation)
            reply = FIELD_PROMPTS[upcoming] if upcoming else ""

            if upcoming:
                return {
                    "response": reply,
                    "reservation": reservation,
                    "reservation_active": True,
                    "next_field_to_ask": upcoming,
                    "validation_errors": [],
                    "messages": [AIMessage(content=reply)],
                }

        # ── All other fields: standard validate-and-store ────────────────────
        else:
            error = _validate_field(next_field, user_input, reservation)
            if error:
                reply = f"{error}\n\n{FIELD_PROMPTS[next_field]}"
                return {
                    "response": reply,
                    "reservation": reservation,
                    "reservation_active": True,
                    "next_field_to_ask": next_field,
                    "validation_errors": [error],
                    "messages": [AIMessage(content=reply)],
                }

            # Normalise and store
            normalized = user_input.strip()
            if next_field == "vehicle_reg":
                normalized = normalized.upper()
            reservation[next_field] = normalized

        # ── Check if form is complete after storing the field ────────────────
        upcoming = _next_missing_field(reservation)

        if upcoming is not None:
            # Still fields remaining — ask for the next one
            if upcoming == "vehicle_type":
                menu_text, available = _build_vehicle_type_menu()
                if not available:
                    return {
                        "response": menu_text,
                        "reservation": {},
                        "reservation_active": False,
                        "next_field_to_ask": None,
                        "validation_errors": ["no_capacity"],
                        "messages": [AIMessage(content=menu_text)],
                    }
                return {
                    "response": menu_text,
                    "reservation": reservation,
                    "reservation_active": True,
                    "next_field_to_ask": "vehicle_type",
                    "validation_errors": [],
                    "messages": [AIMessage(content=menu_text)],
                }

            reply = FIELD_PROMPTS[upcoming]
            return {
                "response": reply,
                "reservation": reservation,
                "reservation_active": True,
                "next_field_to_ask": upcoming,
                "validation_errors": [],
                "messages": [AIMessage(content=reply)],
            }

        # ── All fields collected — attempt database save ─────────────────────
        from src.database.queries import save_reservation

        save_result = save_reservation(
            first_name=reservation["first_name"],
            surname=reservation["surname"],
            vehicle_type=reservation["vehicle_type"],
            vehicle_reg=reservation["vehicle_reg"],
            start_datetime=reservation["start_datetime"],
            end_datetime=reservation["end_datetime"],
        )

        if save_result["success"]:
            confirmation = _build_confirmation(reservation, save_result["reservation_id"])
            return {
                "response": confirmation,
                "reservation": reservation,
                "reservation_active": False,
                "next_field_to_ask": None,
                "validation_errors": [],
                "messages": [AIMessage(content=confirmation)],
            }

        # Handle save failures
        error_code = save_result.get("error", "db_error")

        if error_code == "capacity_exceeded":
            menu_text, available = _build_vehicle_type_menu()
            reply = (
                f"Sorry, the {reservation.get('vehicle_type', 'selected')} spaces just "
                f"became fully booked while we were completing your form.\n\n"
                + (menu_text if available else
                   "Unfortunately the car park is now completely full.")
            )
            reservation.pop("vehicle_type", None)
            return {
                "response": reply,
                "reservation": reservation,
                "reservation_active": bool(available),
                "next_field_to_ask": "vehicle_type" if available else None,
                "validation_errors": ["capacity_exceeded"],
                "messages": [AIMessage(content=reply)],
            }

        if error_code == "duplicate_reservation":
            reply = (
                f"A reservation for vehicle {reservation.get('vehicle_reg')} "
                f"already exists that overlaps with the times you requested. "
                "Please choose different dates or contact customer services on 0800 123 4567."
            )
            return {
                "response": reply,
                "reservation": reservation,
                "reservation_active": False,
                "next_field_to_ask": None,
                "validation_errors": ["duplicate_reservation"],
                "messages": [AIMessage(content=reply)],
            }

        # Generic DB / unexpected error
        reply = (
            "I'm sorry, your reservation could not be saved due to a system error. "
            "Please try again or contact customer services on 0800 123 4567."
        )
        return {
            "response": reply,
            "reservation": reservation,
            "reservation_active": False,
            "next_field_to_ask": None,
            "validation_errors": [error_code],
            "messages": [AIMessage(content=reply)],
        }

    # ── Fallback (should not happen in normal flow) ──────────────────────────
    reply = "I'm collecting your reservation details. " + FIELD_PROMPTS[RESERVATION_FIELDS[0]]
    return {
        "response": reply,
        "reservation": {},
        "reservation_active": True,
        "next_field_to_ask": RESERVATION_FIELDS[0],
        "validation_errors": [],
        "messages": [AIMessage(content=reply)],
    }
