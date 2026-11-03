"""
Tests for Stage 2 user-side status notification.

Covers: status_check_node behaviour, one-time delivery guarantee,
pending_reservation_id lifecycle, and graph-level integration.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.chatbot.state import ConversationState, empty_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> ConversationState:
    s = empty_state()
    s.update(overrides)
    return s


def _mock_status(status: str, notes: str = "", decided_by: str = "admin1"):
    return {"id": 1, "status": status,
            "admin_notes": notes, "decided_at": "2090-01-01 10:00:00",
            "decided_by": decided_by}


# ---------------------------------------------------------------------------
# status_check_node unit tests
# ---------------------------------------------------------------------------

class TestStatusCheckNode:
    def _no_notification(self, result: dict) -> bool:
        """True if the result represents a no-op (no notification delivered)."""
        return "response" not in result and "pending_reservation_id" not in result

    def test_no_pending_id_returns_no_notification(self):
        from src.chatbot.nodes import status_check_node
        result = status_check_node(_state(pending_reservation_id=None))
        assert self._no_notification(result)

    def test_still_pending_returns_no_notification(self, monkeypatch):
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: _mock_status("pending"))
        from src.chatbot.nodes import status_check_node
        result = status_check_node(_state(pending_reservation_id=1))
        assert self._no_notification(result)

    def test_confirmed_delivers_notification(self, monkeypatch):
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: _mock_status("confirmed", "Priority customer"))
        from src.chatbot.nodes import status_check_node
        result = status_check_node(_state(pending_reservation_id=1))
        assert "approved" in result["response"].lower()
        assert "Priority customer" in result["response"]
        assert result["pending_reservation_id"] is None  # cleared

    def test_rejected_delivers_notification(self, monkeypatch):
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: _mock_status("rejected", "No EV spaces left"))
        from src.chatbot.nodes import status_check_node
        result = status_check_node(_state(pending_reservation_id=1))
        assert "declined" in result["response"].lower()
        assert "No EV spaces left" in result["response"]
        assert result["pending_reservation_id"] is None

    def test_notification_delivered_exactly_once(self, monkeypatch):
        """
        After the notification is delivered, pending_reservation_id is None.
        A second call must return no notification.
        """
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: _mock_status("confirmed"))
        from src.chatbot.nodes import status_check_node

        state = _state(pending_reservation_id=1)
        first = status_check_node(state)
        assert first["pending_reservation_id"] is None

        # Simulate the state update: pending_reservation_id is now None
        state["pending_reservation_id"] = None
        second = status_check_node(state)
        assert self._no_notification(second)

    def test_cancelled_delivers_notification(self, monkeypatch):
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: _mock_status("cancelled"))
        from src.chatbot.nodes import status_check_node
        result = status_check_node(_state(pending_reservation_id=1))
        assert "no longer active" in result["response"].lower()
        assert result["pending_reservation_id"] is None

    def test_nonexistent_reservation_returns_no_notification(self, monkeypatch):
        """If the reservation ID is not found in DB, no notification is delivered."""
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: None)
        from src.chatbot.nodes import status_check_node
        result = status_check_node(_state(pending_reservation_id=99999))
        assert self._no_notification(result)

    def test_notification_includes_reservation_id(self, monkeypatch):
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: _mock_status("confirmed"))
        from src.chatbot.nodes import status_check_node
        result = status_check_node(_state(pending_reservation_id=42))
        assert "#42" in result["response"]

    def test_notification_includes_decided_by(self, monkeypatch):
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: _mock_status("confirmed",
                                                           decided_by="ops_admin"))
        from src.chatbot.nodes import status_check_node
        result = status_check_node(_state(pending_reservation_id=1))
        assert "ops_admin" in result["response"]


# ---------------------------------------------------------------------------
# State field tests
# ---------------------------------------------------------------------------

class TestPendingReservationIdField:
    def test_empty_state_has_pending_id_none(self):
        s = empty_state()
        assert s["pending_reservation_id"] is None

    def test_reservation_node_sets_pending_id_on_success(self, monkeypatch):
        """After a successful DB save, reservation_node must set pending_reservation_id."""
        import src.database.queries as q
        monkeypatch.setattr(q, "save_reservation",
                            lambda **kw: {"success": True, "reservation_id": 7,
                                          "error": None, "available_spaces": 719})
        from src.chatbot.nodes import reservation_node
        from src.chatbot.router import INTENT_RESERVATION_INPUT

        reservation = {
            "first_name": "Alice", "surname": "Smith",
            "vehicle_type": "standard", "vehicle_reg": "AB12CDE",
            "start_datetime": "2090-06-15 09:00",
        }
        result = reservation_node(_state(
            user_input="2090-06-15 17:00",
            intent=INTENT_RESERVATION_INPUT,
            reservation_active=True,
            reservation=reservation,
            next_field_to_ask="end_datetime",
        ))
        assert result["pending_reservation_id"] == 7

    def test_reservation_node_does_not_set_pending_id_on_failure(self, monkeypatch):
        """If DB save fails, pending_reservation_id must not be set."""
        import src.database.queries as q
        monkeypatch.setattr(q, "save_reservation",
                            lambda **kw: {"success": False, "error": "db_error",
                                          "reservation_id": None})
        from src.chatbot.nodes import reservation_node
        from src.chatbot.router import INTENT_RESERVATION_INPUT

        reservation = {
            "first_name": "Alice", "surname": "Smith",
            "vehicle_type": "standard", "vehicle_reg": "AB12CDE",
            "start_datetime": "2090-06-15 09:00",
        }
        result = reservation_node(_state(
            user_input="2090-06-15 17:00",
            intent=INTENT_RESERVATION_INPUT,
            reservation_active=True,
            reservation=reservation,
            next_field_to_ask="end_datetime",
        ))
        assert result.get("pending_reservation_id") is None or \
               result.get("pending_reservation_id") == 0


# ---------------------------------------------------------------------------
# Graph-level integration tests
# ---------------------------------------------------------------------------

class TestGraphNotifications:
    def test_graph_delivers_approval_notification(self, monkeypatch):
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: _mock_status("confirmed", "Welcome!"))
        from src.chatbot.graph import build_graph

        mock_chain = MagicMock()
        graph = build_graph(rag_chain=mock_chain)
        state = empty_state()
        state["pending_reservation_id"] = 5

        result = graph.invoke({**state, "user_input": "Hello"})
        assert "approved" in result["response"].lower()
        assert result["pending_reservation_id"] is None
        # RAG/SQL chain must NOT have been called
        assert mock_chain.query.call_count == 0

    def test_graph_delivers_rejection_notification(self, monkeypatch):
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: _mock_status("rejected", "Fully booked"))
        from src.chatbot.graph import build_graph

        mock_chain = MagicMock()
        graph = build_graph(rag_chain=mock_chain)
        state = empty_state()
        state["pending_reservation_id"] = 3

        result = graph.invoke({**state, "user_input": "Any update?"})
        assert "declined" in result["response"].lower()
        assert result["pending_reservation_id"] is None

    def test_graph_continues_normally_when_still_pending(self, monkeypatch):
        import src.database.queries as q
        monkeypatch.setattr(q, "get_reservation_status",
                            lambda rid, **kw: _mock_status("pending"))
        from src.chatbot.graph import build_graph

        mock_chain = MagicMock()
        mock_chain.query.return_value = MagicMock(
            answer="CityPark is at 14 Central Plaza.",
            source_doc_ids=["location_directions"],
            had_relevant_context=True,
        )
        graph = build_graph(rag_chain=mock_chain)
        state = empty_state()
        state["pending_reservation_id"] = 2

        # Still pending → notification NOT delivered → normal FAQ routing
        result = graph.invoke({**state, "user_input": "Where is the car park?"})
        assert result["pending_reservation_id"] == 2  # still set
        assert "14 Central Plaza" in result["response"]

    def test_notification_not_delivered_twice(self, monkeypatch):
        """Once delivered, the graph must not deliver the same notification again."""
        import src.database.queries as q
        call_count = {"n": 0}

        def mock_status(rid, **kw):
            call_count["n"] += 1
            return _mock_status("confirmed")

        monkeypatch.setattr(q, "get_reservation_status", mock_status)
        from src.chatbot.graph import build_graph

        mock_chain = MagicMock()
        mock_chain.query.return_value = MagicMock(
            answer="CityPark answer.", source_doc_ids=[], had_relevant_context=True,
        )
        graph = build_graph(rag_chain=mock_chain)

        # Turn 1: pending_reservation_id is set → delivers notification
        state = empty_state()
        state["pending_reservation_id"] = 1
        state1 = graph.invoke({**state, "user_input": "Any news?"})
        assert "approved" in state1["response"].lower()

        # Turn 2: pending_reservation_id is now None → no second notification
        state2 = graph.invoke({**state1, "user_input": "Where is the car park?"})
        assert "approved" not in state2["response"].lower()
        assert "CityPark" in state2["response"]
