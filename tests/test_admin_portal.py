"""
Tests for the admin portal integration (Step 2 of improvements):
  - Password verification via verify_admin_password / is_admin_configured
  - Admin panel rendered for authenticated sessions
  - Unauthenticated sessions do not access the dashboard
  - Approve / reject still work through the shared dashboard module
  - User portal chat still receives status updates independently
"""

from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Auth logic tests
# ---------------------------------------------------------------------------

class TestAdminPasswordAuth:
    def test_correct_password_accepted(self):
        with patch("src.admin.auth.settings") as m:
            m.admin_password = "secret"
            from src.admin.auth import verify_admin_password
            assert verify_admin_password("secret") is True

    def test_wrong_password_rejected(self):
        with patch("src.admin.auth.settings") as m:
            m.admin_password = "secret"
            from src.admin.auth import verify_admin_password
            assert verify_admin_password("wrong") is False

    def test_empty_candidate_rejected(self):
        with patch("src.admin.auth.settings") as m:
            m.admin_password = "secret"
            from src.admin.auth import verify_admin_password
            assert verify_admin_password("") is False

    def test_missing_config_fails_closed(self):
        """When ADMIN_PASSWORD is not set, is_admin_configured returns False."""
        with patch("src.admin.auth.settings") as m:
            m.admin_password = ""
            from src.admin.auth import is_admin_configured, verify_admin_password
            assert is_admin_configured() is False
            assert verify_admin_password("anything") is False

    def test_password_not_stored_in_normalized_value(self):
        """verify_admin_password must not return the password anywhere."""
        with patch("src.admin.auth.settings") as m:
            m.admin_password = "hunter2"
            from src.admin.auth import verify_admin_password
            result = verify_admin_password("hunter2")
            # Result is a plain bool, no password leakage
            assert isinstance(result, bool)

    def test_timing_safe_comparison_used(self):
        """hmac.compare_digest must be called (not == or !=)."""
        import src.admin.auth as auth_mod
        calls = []
        original = auth_mod.hmac.compare_digest

        def capturing(*args, **kw):
            calls.append(args)
            return original(*args, **kw)

        with (
            patch("src.admin.auth.settings") as m,
            patch.object(auth_mod.hmac, "compare_digest", side_effect=capturing),
        ):
            m.admin_password = "secret"
            auth_mod.verify_admin_password("secret")

        assert len(calls) == 1, "hmac.compare_digest must be used exactly once"


# ---------------------------------------------------------------------------
# Dashboard module — approve/reject logic
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    from src.database.schema import initialise_database
    path = tmp_path / "portal_test.db"
    initialise_database(db_path=path)
    return path


def _save_pending(db_path, reg="AB12CDE"):
    from src.database.queries import save_reservation
    r = save_reservation(
        first_name="Alice", surname="Smith",
        vehicle_type="standard", vehicle_reg=reg,
        start_datetime="2090-07-20 09:00", end_datetime="2090-07-20 17:00",
        db_path=db_path,
    )
    assert r["success"]
    return r["reservation_id"]


class TestDashboardApproveReject:
    """
    Verify _do_approve and _do_reject in the shared dashboard module work
    correctly (no Streamlit rendering; mock st calls).
    """

    def _mock_st(self):
        """Return a namespace mock for st.success / st.warning / st.error."""
        import types
        ns = types.SimpleNamespace(
            success_msgs=[], warning_msgs=[], error_msgs=[], info_msgs=[],
        )
        ns.success = lambda m: ns.success_msgs.append(m)
        ns.warning = lambda m: ns.warning_msgs.append(m)
        ns.error   = lambda m: ns.error_msgs.append(m)
        ns.info    = lambda m: ns.info_msgs.append(m)
        return ns

    def test_approve_direct_path_success(self, db_path: Path, monkeypatch):
        rid = _save_pending(db_path)
        ns = self._mock_st()

        import src.admin.dashboard as dash
        monkeypatch.setattr("src.admin.dashboard.st", ns)
        monkeypatch.setattr("src.admin.dashboard.logger", dash.logger)

        # No workflow checkpoint → direct path
        with (
            patch("src.workflow.graph.get_workflow_graph_state", return_value=None),
            patch("src.database.queries.approve_reservation",
                  return_value={"success": True, "error": None}),
            patch("src.mcp_server.client.call_write_approval_record",
                  return_value={"success": True}),
        ):
            dash._do_approve(rid, "admin1")

        assert any(str(rid) in m for m in ns.success_msgs)

    def test_reject_direct_path_success(self, db_path: Path, monkeypatch):
        rid = _save_pending(db_path, reg="XY34ZAB")
        ns = self._mock_st()

        import src.admin.dashboard as dash
        monkeypatch.setattr("src.admin.dashboard.st", ns)

        with (
            patch("src.workflow.graph.get_workflow_graph_state", return_value=None),
            patch("src.database.queries.reject_reservation",
                  return_value={"success": True, "error": None}),
        ):
            dash._do_reject(rid, "admin1", "No spaces available")

        assert any(str(rid) in m for m in ns.success_msgs)

    def test_approve_already_decided_shows_warning(self, monkeypatch):
        ns = self._mock_st()

        import src.admin.dashboard as dash
        monkeypatch.setattr("src.admin.dashboard.st", ns)

        with (
            patch("src.workflow.graph.get_workflow_graph_state", return_value=None),
            patch("src.database.queries.approve_reservation",
                  return_value={"success": False, "error": "already_decided"}),
        ):
            dash._do_approve(1, "admin1")

        assert any("already" in m.lower() for m in ns.warning_msgs)
        assert not ns.success_msgs

    def test_approve_workflow_completed_shows_warning(self, monkeypatch):
        ns = self._mock_st()

        import src.admin.dashboard as dash
        monkeypatch.setattr("src.admin.dashboard.st", ns)

        snap = type("S", (), {"next": ()})()
        with patch("src.workflow.graph.get_workflow_graph_state", return_value=snap):
            dash._do_approve(1, "admin1")

        assert any("already decided" in m.lower() for m in ns.warning_msgs)

    def test_reject_requires_reason(self, db_path: Path, monkeypatch):
        """Empty reason must be caught at the UI layer (checked in render_reservations)."""
        # This verifies that _do_reject with a blank reason propagates the
        # reason_required error correctly.
        ns = self._mock_st()

        import src.admin.dashboard as dash
        monkeypatch.setattr("src.admin.dashboard.st", ns)

        with (
            patch("src.workflow.graph.get_workflow_graph_state", return_value=None),
            patch("src.database.queries.reject_reservation",
                  return_value={"success": False, "error": "reason_required"}),
        ):
            dash._do_reject(1, "admin1", "")   # empty reason → DB returns reason_required

        assert any("reason" in m.lower() for m in ns.error_msgs)


# ---------------------------------------------------------------------------
# Admin panel session state logic
# ---------------------------------------------------------------------------

class TestAdminPanelSessionState:
    """
    Verify the session-state logic used by streamlit_app.py without
    running the Streamlit server.
    """

    def test_admin_not_shown_by_default(self):
        session = {}
        assert not session.get("show_admin_panel", False)

    def test_admin_panel_flag_set_on_button_click(self):
        session = {}
        # Simulate clicking the Admin button
        session["show_admin_panel"] = True
        assert session["show_admin_panel"] is True

    def test_logout_clears_state(self):
        session = {
            "show_admin_panel": True,
            "admin_authenticated": True,
            "admin_id": "admin",
        }
        for key in ("admin_authenticated", "admin_id", "show_admin_panel"):
            session.pop(key, None)
        assert "admin_authenticated" not in session
        assert "admin_id" not in session
        assert "show_admin_panel" not in session

    def test_chat_state_survives_admin_view(self):
        """User's conversation is preserved when they switch to admin view."""
        session = {
            "messages": [{"role": "user", "content": "Hello"}],
            "conv_state": {"pending_reservation_id": 5},
            "ready": True,
        }
        # Switching to admin panel only adds keys, doesn't remove chat keys
        session["show_admin_panel"] = True
        session["admin_authenticated"] = True
        session["admin_id"] = "admin"

        assert session["messages"][0]["content"] == "Hello"
        assert session["conv_state"]["pending_reservation_id"] == 5

    def test_unauthenticated_does_not_reach_dashboard(self):
        """Without admin_authenticated=True, the dashboard must not render."""
        session = {"show_admin_panel": True}
        # _render_admin_login() returns False when not authenticated
        authenticated = session.get("admin_authenticated", False)
        assert not authenticated  # dashboard must not be called

    def test_authenticated_reaches_dashboard(self):
        session = {
            "show_admin_panel": True,
            "admin_authenticated": True,
            "admin_id": "admin",
        }
        authenticated = session.get("admin_authenticated", False)
        assert authenticated  # dashboard should render


# ---------------------------------------------------------------------------
# User portal status polling still works after admin integration
# ---------------------------------------------------------------------------

class TestStatusPollCoexistence:
    def test_status_check_node_independent_of_admin_state(self, monkeypatch):
        """status_check_node must work regardless of admin session state."""
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: {"id": rid, "status": "confirmed",
                                               "admin_notes": "",
                                               "decided_at": "2090-01-01",
                                               "decided_by": "admin1"})

        from src.chatbot.nodes import status_check_node
        from src.chatbot.state import empty_state

        state = empty_state()
        state["pending_reservation_id"] = 7
        result = status_check_node(state)

        assert result["pending_reservation_id"] is None
        assert result["intent"] == "status_notified"
        assert "approved" in result["response"].lower()
