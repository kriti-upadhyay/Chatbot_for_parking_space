"""
LangGraph graph assembly.

Wires all nodes into a compiled StateGraph and exposes ``build_graph()``.

Graph topology:
  START
    └─► guardrail_node
          └─► intent_router_node  (if not blocked)
                ├─► rag_node       (general_faq)
                ├─► sql_node       (dynamic_info)
                └─► reservation_node  (reservation | reservation_input)
                      each → END

A blocked input (injection detected) goes: guardrail_node → END directly,
skipping the router entirely.
"""

from langgraph.graph import END, START, StateGraph

from src.chatbot.nodes import (
    guardrail_node,
    intent_router_node,
    make_rag_node,
    reservation_node,
    sql_node,
)
from src.chatbot.router import route_intent
from src.chatbot.state import ConversationState
from src.config import get_logger

logger = get_logger(__name__)


def _after_guardrail(state: ConversationState) -> str:
    """
    Conditional edge: if the guardrail blocked the input, go to END;
    otherwise proceed to the intent router.
    """
    if state.get("intent") == "blocked":
        return END
    return "intent_router_node"


def build_graph(rag_chain=None):
    """
    Build and compile the conversation graph.

    Args:
        rag_chain: A ``RagChain`` instance.  If None, one is built from
                   settings (requires GROQ_API_KEY and a populated ChromaDB).
                   Pass a mock during testing.

    Returns:
        A compiled LangGraph ``CompiledGraph``.
    """
    if rag_chain is None:
        from src.rag.chain import build_rag_chain
        rag_chain = build_rag_chain()
        logger.info("RagChain built inside build_graph()")

    rag_node = make_rag_node(rag_chain)

    # ── Build graph ──────────────────────────────────────────────────────────
    builder = StateGraph(ConversationState)

    # Register nodes
    builder.add_node("guardrail_node",     guardrail_node)
    builder.add_node("intent_router_node", intent_router_node)
    builder.add_node("rag_node",           rag_node)
    builder.add_node("sql_node",           sql_node)
    builder.add_node("reservation_node",   reservation_node)

    # Entry point
    builder.add_edge(START, "guardrail_node")

    # After guardrail: either go to router or END (if blocked)
    builder.add_conditional_edges(
        "guardrail_node",
        _after_guardrail,
        {
            "intent_router_node": "intent_router_node",
            END: END,
        },
    )

    # After router: dispatch to the appropriate handler node
    builder.add_conditional_edges(
        "intent_router_node",
        route_intent,
        {
            "rag_node":         "rag_node",
            "sql_node":         "sql_node",
            "reservation_node": "reservation_node",
        },
    )

    # All handler nodes lead to END
    builder.add_edge("rag_node",         END)
    builder.add_edge("sql_node",         END)
    builder.add_edge("reservation_node", END)

    compiled = builder.compile()
    logger.info("LangGraph conversation graph compiled.")
    return compiled
