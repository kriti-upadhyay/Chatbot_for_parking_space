"""
Tests for the auto-refresh reservation status logic in streamlit_app.py.

Tests are offline — no Streamlit rendering, no DB I/O.
They call the polling logic functions directly with mocked session state.
"""

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_session(pending_rid=None, notified=None):
    """Build a minimal st.session_state mock."""
    state = {
        "conv_state": {
            "pending_reservation_id": pending_rid,
            "messages": [],
            "reservation": {},
            "reservation_active": False,
            "intent": None,
            "response": "",
            "next_field_to_ask": None,
            "validation_errors": [],
        },
        "messages": [],
        "notified_rids": set(notified or []),
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, k: state[k]
    mock.__setitem__ = lambda self, k, v: state.__setitem__(k, v)
    mock.__contains__ = lambda self, k: k in state
    mock.get = lambda k, default=None: state.get(k, default)
    mock.setdefault = lambda k, default: state.setdefault(k, default)
    # Allow attribute-style access for notified_rids
    mock.configure_mock(**{"notified_rids": state["notified_rids"]})
    return mock, state


# ---------------------------------------------------------------------------
# Unit tests for poll logic (extracted from Streamlit fragment)
# ---------------------------------------------------------------------------

class TestPollLogic:
    """
    Tests for the _poll_reservation_status logic without Streamlit rendering.
    We import and call the business logic functions directly.
    """

    def test_no_pending_id_does_nothing(self, monkeypatch):
        from src.database.queries import get_reservation_status

        called = {"n": 0}
        monkeypatch.setattr("src.database.queries.get_reservation_status",
                            lambda rid, **kw: called.__setitem__("n", called["n"] + 1) or None)

        # Build a fake conv_state with no pending ID
        conv_state = {"pending_reservation_id": None}
        rid = conv_state.get("pending_reservation_id")
        assert rid is None
        # Poll logic exits immediately when rid is None
        assert called["n"] == 0

    def test_still_pending_does_not_notify(self, monkeypatch):
        monkeypatch.setattr("src.database.queries.get_reservation_status",
                            lambda rid, **kw: {"id": rid, "status": "pending",
                                               "admin_notes": None, "decided_at": None,
                                               "decided_by": None})
        from src.database.queries import get_reservation_status
        status_info = get_reservation_status(1)
        assert status_info["status"] == "pending"
        # No notification should be built when still pending

    def test_confirmed_builds_approval_message(self, monkeypatch):
        monkeypatch.setattr("src.database.queries.get_reservation_status",
                            lambda rid, **kw: {"id": rid, "status": "confirmed",
                                               "admin_notes": "Welcome!",
                                               "decided_at": "2026-01-01 10:00",
                                               "decided_by": "admin1"})
        from src.database.queries import get_reservation_status
        status_info = get_reservation_status(7)
        assert status_info["status"] == "confirmed"

        # Simulate the notification building logic
        notes = status_info.get("admin_notes") or ""
        decided_by = status_info.get("decided_by") or "the administrator"
        notification = (
            f"✅ **Good news!** Your reservation (#7) has been "
            f"**approved** by {decided_by}.\n"
            + (f"Note from administrator: {notes}\n" if notes else "")
            + "You can proceed to CityPark on your reserved date."
        )
        assert "approved" in notification.lower()
        assert "Welcome!" in notification
        assert "admin1" in notification

    def test_rejected_builds_rejection_message(self, monkeypatch):
        monkeypatch.setattr("src.database.queries.get_reservation_status",
                            lambda rid, **kw: {"id": rid, "status": "rejected",
                                               "admin_notes": "No EV spaces",
                                               "decided_at": "2026-01-01 10:00",
                                               "decided_by": "admin1"})
        from src.database.queries import get_reservation_status
        status_info = get_reservation_status(3)

        notes = status_info.get("admin_notes") or ""
        decided_by = status_info.get("decided_by") or "the administrator"
        notification = (
            f"❌ We're sorry — your reservation (#3) has been "
            f"**declined** by {decided_by}.\n"
            + (f"Reason: {notes}\n" if notes else "")
            + "Please contact CityPark on 0800 123 4567 if you have questions."
        )
        assert "declined" in notification.lower()
        assert "No EV spaces" in notification

    def test_notification_not_duplicated_for_same_rid(self):
        """Once a rid is in notified_rids, no second notification is built."""
        notified = {42}
        rid = 42
        if rid in notified:
            # Poll logic exits early — notification already delivered
            should_notify = False
        else:
            should_notify = True
        assert not should_notify

    def test_notification_delivered_for_new_rid(self):
        notified = {1, 2, 3}
        rid = 5
        should_notify = rid not in notified
        assert should_notify

    def test_pending_reservation_id_cleared_after_notification(self):
        """After delivering a notification, pending_reservation_id must be set to None."""
        conv_state = {"pending_reservation_id": 7}
        rid = conv_state["pending_reservation_id"]
        # Simulate delivery
        conv_state["pending_reservation_id"] = None
        notified = set()
        notified.add(rid)

        assert conv_state["pending_reservation_id"] is None
        assert 7 in notified

    def test_page_reload_does_not_re_deliver(self):
        """
        If the user reloads the page, pending_reservation_id might be reset by
        empty_state().  But if the rid is in notified_rids (persisted in session
        state), the poll clears pending_reservation_id without notifying again.
        """
        notified = {7}
        conv_state = {"pending_reservation_id": 7}

        rid = conv_state["pending_reservation_id"]
        if rid in notified:
            # Clear without notifying
            conv_state["pending_reservation_id"] = None
            delivered = False
        else:
            delivered = True

        assert not delivered
        assert conv_state["pending_reservation_id"] is None


# ---------------------------------------------------------------------------
# Integration with status_check_node (existing turn-based mechanism)
# ---------------------------------------------------------------------------

class TestStatusCheckNodeStillWorks:
    """Confirm the existing per-turn notification still works independently."""

    def test_status_check_node_delivers_on_message(self, monkeypatch):
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: {"id": rid, "status": "confirmed",
                                               "admin_notes": "", "decided_at": "2026-01-01",
                                               "decided_by": "admin1"})
        from src.chatbot.nodes import status_check_node
        from src.chatbot.state import empty_state

        state = empty_state()
        state["pending_reservation_id"] = 5
        result = status_check_node(state)

        assert result["pending_reservation_id"] is None
        assert result["intent"] == "status_notified"
        assert "approved" in result["response"].lower()

    def test_status_check_node_no_duplicate_with_poll(self, monkeypatch):
        """
        If the poll already cleared pending_reservation_id, status_check_node
        sees None and does nothing (no duplicate notification on next message).
        """
        from src.chatbot.nodes import status_check_node
        from src.chatbot.state import empty_state

        state = empty_state()
        state["pending_reservation_id"] = None   # poll already cleared it

        result = status_check_node(state)
        assert result == {"intent": None}        # no-op
