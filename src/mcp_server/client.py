"""
MCP client helper for the admin dashboard.

Provides:
  generate_mcp_token()          — HMAC-SHA256 signed with MCP_SHARED_SECRET
  call_write_approval_record()  — synchronous wrapper around the async MCP call

call_write_approval_record() uses FastMCP's Client with the in-process
server instance, which runs the full MCP JSON-RPC protocol (initialize /
tools/list / tools/call) over an in-memory transport.  This is genuine MCP
protocol communication, not a direct Python function call.

Only call generate_mcp_token() and call_write_approval_record() after the
admin session has been authenticated (admin_app.py checks
st.session_state.admin_authenticated before calling either function).
"""

from __future__ import annotations

import asyncio
import hmac
import time

from src.config import get_logger, settings

logger = get_logger(__name__)


def generate_mcp_token() -> str:
    """
    Generate a short-lived HMAC-SHA256 token signed with MCP_SHARED_SECRET.

    The token encodes the current UTC minute; it is valid for the current
    minute and the previous minute (~1-2 min window) as validated by the
    server.

    Raises:
        RuntimeError: if MCP_SHARED_SECRET is not configured.
    """
    secret = settings.mcp_shared_secret
    if not secret:
        raise RuntimeError(
            "MCP_SHARED_SECRET is not configured. "
            "Add it to your .env file before using the admin export feature."
        )
    minute = int(time.time()) // 60
    return hmac.new(
        secret.encode("utf-8"),
        str(minute).encode("utf-8"),
        "sha256",
    ).hexdigest()


async def _call_tool_async(reservation_id: int) -> dict:
    """Call the MCP tool via the in-process FastMCP Client."""
    from fastmcp import Client
    from src.mcp_server.server import mcp

    token = generate_mcp_token()
    async with Client(mcp) as client:
        result = await client.call_tool(
            "write_approval_record",
            {"reservation_id": reservation_id, "auth_token": token},
        )
    return result.data if result.data is not None else {}


def call_write_approval_record(reservation_id: int) -> dict:
    """
    Synchronous wrapper around the async MCP tool call.

    Safe to call from Streamlit (which is synchronous).  Internally uses
    asyncio.run(), which creates a fresh event loop for each call.
    """
    try:
        return asyncio.run(_call_tool_async(reservation_id))
    except Exception as exc:
        logger.exception(
            "MCP client call failed for reservation %d", reservation_id
        )
        return {"success": False, "error": "client_error", "message": str(exc)}
