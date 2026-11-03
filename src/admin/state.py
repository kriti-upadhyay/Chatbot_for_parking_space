"""
State type for the admin LangGraph agent.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AdminState(TypedDict):
    # Full message history for the admin conversation turn.
    messages: Annotated[list, add_messages]

    # The authenticated admin's identifier (set once at session start).
    admin_id: str

    # Plain-text result of the last tool call, forwarded to the UI.
    last_result: str


def empty_admin_state(admin_id: str = "") -> AdminState:
    return AdminState(messages=[], admin_id=admin_id, last_result="")
