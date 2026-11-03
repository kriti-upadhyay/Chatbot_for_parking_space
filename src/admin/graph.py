"""
Admin LangGraph graph.

Single-node graph — the admin agent handles the full ReAct loop internally
so only one node is needed.  The graph is kept simple deliberately; the
approve/reject actions bypass this graph entirely and call the DB functions
directly from the UI.
"""

from langgraph.graph import END, START, StateGraph

from src.admin.nodes import admin_agent_node
from src.admin.state import AdminState
from src.config import get_logger

logger = get_logger(__name__)


def build_admin_graph():
    """Build and compile the admin agent graph."""
    builder = StateGraph(AdminState)
    builder.add_node("admin_agent_node", admin_agent_node)
    builder.add_edge(START, "admin_agent_node")
    builder.add_edge("admin_agent_node", END)
    compiled = builder.compile()
    logger.info("Admin agent graph compiled.")
    return compiled
