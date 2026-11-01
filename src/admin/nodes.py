"""
LangGraph node functions for the admin agent.

The admin agent is a read-only tool-calling LLM.  It can list and explain
reservations but cannot approve or reject them — those actions are only
available through the explicit UI buttons (admin_app.py).

Nodes:
  admin_agent_node — single ReAct-style agent node that calls read-only tools
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.admin.state import AdminState
from src.admin.tools import ADMIN_TOOLS
from src.config import get_logger, settings

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an assistant for the CityPark parking administrator.
You help the administrator review and understand pending reservation requests.

You have access to these read-only tools:
- list_pending: show reservations awaiting a decision
- get_details: show full details of a specific reservation by ID
- list_all: show all reservations, optionally filtered by status

IMPORTANT CONSTRAINTS:
1. You CANNOT approve or reject reservations.  Those actions are performed
   exclusively through the dashboard buttons by an authenticated administrator.
2. Do not suggest, simulate, or pretend to perform approval or rejection.
3. Answer only questions related to CityPark reservations and parking administration.
4. If asked to approve or reject, explain that the buttons in the dashboard must be used.
"""


def build_admin_llm() -> ChatGroq:
    """Return a ChatGroq LLM bound to the admin read-only tools."""
    llm = ChatGroq(
        model=settings.groq_model_name,
        api_key=settings.groq_api_key,  # type: ignore[arg-type]
        temperature=0,
    )
    return llm.bind_tools(ADMIN_TOOLS)


def admin_agent_node(state: AdminState) -> dict:
    """
    Single ReAct-style agent node.

    Processes the latest user message, calls tools as needed (up to a
    small loop limit), and returns the final AI response.
    """
    llm_with_tools = build_admin_llm()

    # Prepend system message + current conversation
    messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(state["messages"])

    # ReAct loop — call tools until the LLM produces a plain text response
    # or the iteration limit is reached.
    max_iterations = 5
    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            # No tool call — final answer
            break

        # Execute each tool call and append results
        from langchain_core.messages import ToolMessage
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            logger.debug("Admin agent calling tool: %s(%s)", tool_name, tool_args)

            # Find and invoke the matching tool
            tool_fn = next((t for t in ADMIN_TOOLS if t.name == tool_name), None)
            if tool_fn is None:
                result = f"Unknown tool: {tool_name}"
            else:
                try:
                    result = tool_fn.invoke(tool_args)
                except Exception as exc:
                    result = f"Tool error: {exc}"
                    logger.exception("Admin tool %s failed", tool_name)

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    # Extract the final text response
    final = messages[-1]
    content = getattr(final, "content", "") or ""

    # Only return the messages appended this turn (after the original history)
    new_messages = messages[1 + len(state["messages"]):]

    return {
        "messages": new_messages,
        "last_result": content,
    }
