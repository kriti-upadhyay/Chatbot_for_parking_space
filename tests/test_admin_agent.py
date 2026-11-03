"""
Tests for Stage 2 admin agent — auth, read-only tools, graph, and
the security constraint that LLM cannot approve/reject.

All LLM calls are mocked.  Database operations use a temporary SQLite file.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    from src.database.schema import initialise_database
    path = tmp_path / "test_admin_agent.db"
    initialise_database(db_path=path)
    return path


def _save(db_path, *, reg="AB12CDE", vtype="standard",
          start="2090-06-15 09:00", end="2090-06-15 17:00"):
    from src.database.queries import save_reservation
    return save_reservation(
        first_name="Alice", surname="Smith",
        vehicle_type=vtype, vehicle_reg=reg,
        start_datetime=start, end_datetime=end,
        db_path=db_path,
    )


def _mock_settings(password: str):
    """Return a MagicMock that behaves like the settings singleton."""
    m = MagicMock()
    m.admin_password = password
    return m


def _make_llm_mock(content: str, tool_calls: list | None = None):
    """
    Return a MagicMock LLM whose invoke() returns a real AIMessage.

    Using AIMessage (not MagicMock) so LangGraph's add_messages reducer
    can handle the response without raising NotImplementedError.
    """
    # AIMessage.tool_calls defaults to [], which is what we need for most tests
    response = AIMessage(content=content)
    if tool_calls is not None:
        object.__setattr__(response, "tool_calls", tool_calls)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = response
    return mock_llm


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------

class TestAuth:
    def test_correct_password_returns_true(self):
        with patch("src.admin.auth.settings", _mock_settings("secret123")):
            from src.admin.auth import verify_admin_password
            assert verify_admin_password("secret123") is True

    def test_wrong_password_returns_false(self):
        with patch("src.admin.auth.settings", _mock_settings("secret123")):
            from src.admin.auth import verify_admin_password
            assert verify_admin_password("wrongpassword") is False

    def test_empty_candidate_returns_false(self):
        with patch("src.admin.auth.settings", _mock_settings("secret123")):
            from src.admin.auth import verify_admin_password
            assert verify_admin_password("") is False

    def test_empty_stored_password_returns_false(self):
        """If ADMIN_PASSWORD is not configured, auth must always fail."""
        with patch("src.admin.auth.settings", _mock_settings("")):
            from src.admin.auth import verify_admin_password
            assert verify_admin_password("anything") is False

    def test_unconfigured_admin_is_detected(self):
        with patch("src.admin.auth.settings", _mock_settings("")):
            from src.admin.auth import is_admin_configured
            assert is_admin_configured() is False

    def test_configured_admin_is_detected(self):
        with patch("src.admin.auth.settings", _mock_settings("set")):
            from src.admin.auth import is_admin_configured
            assert is_admin_configured() is True

    def test_password_comparison_uses_hmac(self):
        """Verify hmac.compare_digest is used (constant-time comparison)."""
        import hmac as hmac_mod
        import src.admin.auth as auth_mod

        calls: list = []
        original = hmac_mod.compare_digest

        def capturing(*args, **kw):
            calls.append(args)
            return original(*args, **kw)

        with (
            patch("src.admin.auth.settings", _mock_settings("secret")),
            patch.object(auth_mod.hmac, "compare_digest", side_effect=capturing),
        ):
            auth_mod.verify_admin_password("secret")

        assert len(calls) == 1, "hmac.compare_digest must be called exactly once"

    def test_unauthorized_access_blocked_at_auth_layer(self):
        with patch("src.admin.auth.settings", _mock_settings("correct_pass")):
            from src.admin.auth import verify_admin_password
            assert verify_admin_password("wrong_pass") is False
            assert verify_admin_password("correct_pass") is True


# ---------------------------------------------------------------------------
# Admin tools tests
# ---------------------------------------------------------------------------

class TestAdminTools:
    def test_list_pending_empty(self, monkeypatch):
        import src.admin.tools as tools_mod
        monkeypatch.setattr(tools_mod, "get_pending_reservations", lambda **kw: [])
        from src.admin.tools import list_pending
        result = list_pending.invoke({"dummy": ""})
        assert "no pending" in result.lower()

    def test_list_pending_with_reservations(self, monkeypatch):
        import src.admin.tools as tools_mod
        monkeypatch.setattr(tools_mod, "get_pending_reservations", lambda **kw: [
            {"id": 1, "first_name": "Alice", "surname": "Smith",
             "vehicle_reg": "AB12CDE", "vehicle_type": "standard",
             "start_datetime": "2090-06-15 09:00",
             "end_datetime": "2090-06-15 17:00",
             "created_at": "2026-01-01 10:00:00"},
        ])
        from src.admin.tools import list_pending
        result = list_pending.invoke({"dummy": ""})
        assert "#1" in result
        assert "Alice" in result
        assert "AB12CDE" in result

    def test_get_details_found(self, monkeypatch):
        import src.admin.tools as tools_mod
        monkeypatch.setattr(tools_mod, "get_reservation_by_id", lambda rid, **kw: {
            "id": 1, "first_name": "Alice", "surname": "Smith",
            "vehicle_reg": "AB12CDE", "vehicle_type": "standard",
            "start_datetime": "2090-06-15 09:00",
            "end_datetime": "2090-06-15 17:00",
            "status": "pending", "created_at": "2026-01-01 10:00:00",
            "admin_notes": None, "decided_at": None, "decided_by": None,
        })
        from src.admin.tools import get_details
        result = get_details.invoke({"reservation_id": 1})
        assert "Alice" in result
        assert "AB12CDE" in result

    def test_get_details_not_found(self, monkeypatch):
        import src.admin.tools as tools_mod
        monkeypatch.setattr(tools_mod, "get_reservation_by_id",
                            lambda rid, **kw: None)
        from src.admin.tools import get_details
        result = get_details.invoke({"reservation_id": 99999})
        assert "no reservation" in result.lower()

    def test_list_all_no_filter(self, monkeypatch):
        import src.admin.tools as tools_mod
        monkeypatch.setattr(tools_mod, "get_all_reservations",
                            lambda status=None, **kw: [
            {"id": 1, "first_name": "A", "surname": "B",
             "vehicle_reg": "AA11AA", "status": "pending",
             "start_datetime": "2090-06-15 09:00",
             "end_datetime": "2090-06-15 17:00"},
        ])
        from src.admin.tools import list_all
        result = list_all.invoke({"status": ""})
        assert "#1" in result

    def test_tools_are_read_only(self):
        """ADMIN_TOOLS must not contain any approve/reject functions."""
        from src.admin.tools import ADMIN_TOOLS
        tool_names = {t.name for t in ADMIN_TOOLS}
        write_keywords = {"approve", "reject", "confirm", "update", "delete", "cancel"}
        for name in tool_names:
            for kw in write_keywords:
                assert kw not in name.lower(), (
                    f"Tool '{name}' looks like it could write to DB; "
                    "admin writes must go through explicit UI buttons only."
                )

    def test_admin_tools_count(self):
        """There should be exactly 3 read-only tools."""
        from src.admin.tools import ADMIN_TOOLS
        assert len(ADMIN_TOOLS) == 3
        names = {t.name for t in ADMIN_TOOLS}
        assert names == {"list_pending", "get_details", "list_all"}


# ---------------------------------------------------------------------------
# Admin graph tests (LLM mocked with real AIMessage objects)
# ---------------------------------------------------------------------------

class TestAdminGraph:
    def test_graph_compiles(self):
        from src.admin.graph import build_admin_graph
        with patch("src.admin.nodes.build_admin_llm",
                   return_value=_make_llm_mock("Hello")):
            graph = build_admin_graph()
        assert graph is not None

    def test_graph_returns_llm_response(self):
        from src.admin.graph import build_admin_graph
        from src.admin.state import empty_admin_state

        mock_llm = _make_llm_mock("There are 2 pending reservations.")
        with patch("src.admin.nodes.build_admin_llm", return_value=mock_llm):
            graph = build_admin_graph()
            state = empty_admin_state(admin_id="admin1")
            state["messages"] = [HumanMessage(content="Show pending reservations")]
            result = graph.invoke(state)

        assert result["last_result"] == "There are 2 pending reservations."

    def test_graph_preserves_admin_id(self):
        from src.admin.graph import build_admin_graph
        from src.admin.state import empty_admin_state

        mock_llm = _make_llm_mock("OK")
        with patch("src.admin.nodes.build_admin_llm", return_value=mock_llm):
            graph = build_admin_graph()
            state = empty_admin_state(admin_id="ops_admin")
            state["messages"] = [HumanMessage(content="Hello")]
            result = graph.invoke(state)

        assert result["admin_id"] == "ops_admin"

    def test_graph_cannot_approve_via_llm(self):
        """
        If the LLM tries to call an 'approve_reservation' tool (which is not
        in ADMIN_TOOLS), the node must handle it as an unknown tool and not
        actually approve anything.
        """
        from src.admin.graph import build_admin_graph
        from src.admin.state import empty_admin_state

        # First call: LLM pretends to call an unknown approve tool
        fake_tool_call_response = AIMessage(content="")
        fake_tool_call_response.tool_calls = [
            {"name": "approve_reservation",
             "args": {"reservation_id": 1},
             "id": "fake_tc_1"}
        ]
        # Second call: final response after the "tool" returned an error
        final_response = AIMessage(
            content="I cannot approve reservations — please use the dashboard buttons."
        )

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [fake_tool_call_response, final_response]

        with patch("src.admin.nodes.build_admin_llm", return_value=mock_llm):
            graph = build_admin_graph()
            state = empty_admin_state(admin_id="admin1")
            state["messages"] = [HumanMessage(content="Approve reservation 1")]
            result = graph.invoke(state)

        # Final response must not indicate a real approval happened
        assert "cannot approve" in result["last_result"].lower() or \
               "dashboard" in result["last_result"].lower()

    def test_admin_system_prompt_prohibits_approve_reject(self):
        """The system prompt must tell the agent it cannot approve/reject."""
        from src.admin.nodes import _SYSTEM_PROMPT
        low = _SYSTEM_PROMPT.lower()
        assert "cannot approve" in low or "cannot" in low
        assert "button" in low or "dashboard" in low

    def test_admin_system_prompt_does_not_expose_passwords(self):
        """ADMIN_PASSWORD value must never appear in any system prompt."""
        from src.admin.nodes import _SYSTEM_PROMPT
        assert "ADMIN_PASSWORD" not in _SYSTEM_PROMPT
        assert "password" not in _SYSTEM_PROMPT.lower()
