"""
Intent classification and conditional-edge routing.

Classification is done with a keyword-based approach: fast, transparent,
and requires no extra LLM call for common cases.  The mapping is:

  reservation_input  — state flag `reservation_active` is set
  reservation        — user explicitly wants to make a booking
  dynamic_info       — prices, opening hours, or availability question
  general_faq        — everything else (routed to RAG)
"""

from src.config import get_logger
from src.chatbot.state import ConversationState

logger = get_logger(__name__)

# Intent string constants — used as keys in conditional-edge dicts.
INTENT_RESERVATION_INPUT = "reservation_input"
INTENT_RESERVATION       = "reservation"
INTENT_DYNAMIC_INFO      = "dynamic_info"
INTENT_GENERAL_FAQ       = "general_faq"

_RESERVATION_KEYWORDS: frozenset[str] = frozenset({
    "book", "booking", "reserve", "reservation", "make a booking",
    "make a reservation", "i want to park", "i'd like to park",
    "i would like to park",
})

_DYNAMIC_PRICE_KEYWORDS: frozenset[str] = frozenset({
    "price", "prices", "cost", "costs", "how much", "rate", "rates",
    "charge", "charges", "fee", "fees", "tariff", "pay",
})

_DYNAMIC_HOURS_KEYWORDS: frozenset[str] = frozenset({
    "hour", "hours", "open", "opens", "close", "closes", "closing",
    "opening", "when", "time", "times", "schedule", "24 hour",
    "staffed", "staff hours",
})

_DYNAMIC_AVAIL_KEYWORDS: frozenset[str] = frozenset({
    "available", "availability", "space", "spaces", "free",
    "full", "capacity", "how many", "any spaces", "is there room",
})

_DYNAMIC_KEYWORDS: frozenset[str] = (
    _DYNAMIC_PRICE_KEYWORDS | _DYNAMIC_HOURS_KEYWORDS | _DYNAMIC_AVAIL_KEYWORDS
)


def classify_intent(user_input: str, reservation_active: bool = False) -> str:
    """
    Return one of the INTENT_* constants for *user_input*.

    Args:
        user_input:         Raw text from the user.
        reservation_active: If True, we are mid-collection — always returns
                            INTENT_RESERVATION_INPUT regardless of content.
    """
    if reservation_active:
        return INTENT_RESERVATION_INPUT

    text = user_input.lower().strip()

    if any(kw in text for kw in _RESERVATION_KEYWORDS):
        logger.debug("Intent: reservation (keyword match)")
        return INTENT_RESERVATION

    if any(kw in text for kw in _DYNAMIC_KEYWORDS):
        logger.debug("Intent: dynamic_info (keyword match)")
        return INTENT_DYNAMIC_INFO

    logger.debug("Intent: general_faq (default)")
    return INTENT_GENERAL_FAQ


def route_intent(state: ConversationState) -> str:
    """
    Conditional-edge function for the LangGraph router.

    Returns the node name to route to based on `state["intent"]`.
    """
    intent = state.get("intent") or INTENT_GENERAL_FAQ

    routing = {
        INTENT_RESERVATION_INPUT: "reservation_node",
        INTENT_RESERVATION:       "reservation_node",
        INTENT_DYNAMIC_INFO:      "sql_node",
        INTENT_GENERAL_FAQ:       "rag_node",
    }

    destination = routing.get(intent, "rag_node")
    logger.debug("Routing intent '%s' → %s", intent, destination)
    return destination
