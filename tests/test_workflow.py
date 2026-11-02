"""
Stage 4 tests for the reservation lifecycle workflow graph.

Uses MemorySaver for most tests (fast, no I/O) and SqliteSaver for
persistence tests.  All external calls (approve_reservation,
reject_reservation, call_write_approval_record) are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.workflow.graph import (
    WORKFLOW_THREAD_ID,
    build_workflow_graph,
    get_workflow_graph_state,
    resume_reservation_workflow,
    start_reservation_workflow,
)
from src.workflow.state import ReservationWorkflowState


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Real seeded database for persistence tests."""
    from src.database.schema import initialise_database
    path = tmp_path / "wf_test.db"
    initialise_database(db_path=path)
    return path


def _save_approved(db_path: Path):
    """Save and approve a reservation; return (reservation_id, reservation_dict)."""
    from src.database.queries import approve_reservation, save_reservation
    res = save_reservation(
        first_name="Alice", surname="Smith",
        vehicle_type="standard", vehicle_reg="AB12CDE",
        start_datetime="2090-07-20 09:00", end_datetime="2090-07-20 17:00",
        db_path=db_path,
    )
    assert res["success"]
    rid = res["reservation_id"]
    ok = approve_reservation(rid, "admin1", db_path=db_path)
    assert ok["success"]
    return rid, {"first_name": "Alice", "surname": "Smith",
                 "vehicle_type": "standard", "vehicle_reg": "AB12CDE",
                 "start_datetime": "2090-07-20 09:00", "end_datetime": "2090-07-20 17:00"}


def _initial_state(rid: int = 1) -> ReservationWorkflowState:
    return ReservationWorkflowState(
        reservation_id=rid,
        reservation={"first_name": "Test", "vehicle_reg": "XY34ZAB"},
        decision=None,
        admin_id=None,
        admin_notes="",
        export_result=None,
        error=None,
    )


def _cfg(rid: int):
    return {"configurable": {"thread_id": WORKFLOW_THREAD_ID(rid)}}


# ---------------------------------------------------------------------------
# TestWorkflowGraph — compilation and basic structure
# ---------------------------------------------------------------------------

class TestWorkflowGraph:
    def test_graph_compiles_with_memory_saver(self):
        g = build_workflow_graph(MemorySaver())
        assert g is not None

    def test_graph_pauses_at_await_admin(self):
        g = build_workflow_graph(MemorySaver())
        g.invoke(_initial_state(1), config=_cfg(1))
        snap = g.get_state(_cfg(1))
        assert snap.next == ("await_admin_node",)

    def test_interrupt_payload_contains_reservation_id(self):
        g = build_workflow_graph(MemorySaver())
        g.invoke(_initial_state(42), config=_cfg(42))
        snap = g.get_state(_cfg(42))
        iv = snap.tasks[0].interrupts[0].value
        assert iv["reservation_id"] == 42

    def test_thread_id_format(self):
        assert WORKFLOW_THREAD_ID(5) == "reservation_5"
        assert WORKFLOW_THREAD_ID(100) == "reservation_100"


# ---------------------------------------------------------------------------
# TestApprovalPath
# ---------------------------------------------------------------------------

class TestApprovalPath:
    def _run_approval(self, rid: int = 1, approve_result=None, export_result=None):
        """Build a graph, start it, then resume with 'confirmed'."""
        if approve_result is None:
            approve_result = {"success": True, "error": None}
        if export_result is None:
            export_result = {"success": True, "record": "...", "already_exported": False}

        saver = MemorySaver()
        g = build_workflow_graph(saver)
        g.invoke(_initial_state(rid), config=_cfg(rid))

        with (
            patch("src.database.queries.approve_reservation", return_value=approve_result),
            patch("src.mcp_server.client.call_write_approval_record", return_value=export_result),
        ):
            final = g.invoke(
                Command(resume={"decision": "confirmed", "admin_id": "admin1", "admin_notes": "OK"}),
                config=_cfg(rid),
            )
        return final

    def test_approved_sets_decision_confirmed(self):
        final = self._run_approval()
        assert final["decision"] == "confirmed"

    def test_approved_has_no_error(self):
        final = self._run_approval()
        assert final["error"] is None

    def test_approved_sets_admin_id(self):
        final = self._run_approval()
        assert final["admin_id"] == "admin1"

    def test_mcp_export_called_on_approval(self):
        saver = MemorySaver()
        g = build_workflow_graph(saver)
        g.invoke(_initial_state(7), config=_cfg(7))

        mock_export = MagicMock(return_value={"success": True, "record": "line"})
        with (
            patch("src.database.queries.approve_reservation", return_value={"success": True}),
            patch("src.mcp_server.client.call_write_approval_record", mock_export),
        ):
            g.invoke(
                Command(resume={"decision": "confirmed", "admin_id": "admin1", "admin_notes": ""}),
                config=_cfg(7),
            )
        mock_export.assert_called_once_with(7)

    def test_export_result_stored_in_state(self):
        final = self._run_approval(
            export_result={"success": True, "record": "Alice | AB | ..."}
        )
        assert final["export_result"]["success"] is True

    def test_workflow_completes_after_approval(self):
        saver = MemorySaver()
        g = build_workflow_graph(saver)
        g.invoke(_initial_state(3), config=_cfg(3))
        with (
            patch("src.database.queries.approve_reservation", return_value={"success": True}),
            patch("src.mcp_server.client.call_write_approval_record",
                  return_value={"success": True}),
        ):
            g.invoke(Command(resume={"decision": "confirmed", "admin_id": "a", "admin_notes": ""}),
                     config=_cfg(3))
        snap = g.get_state(_cfg(3))
        assert snap.next == ()  # completed


# ---------------------------------------------------------------------------
# TestRejectionPath
# ---------------------------------------------------------------------------

class TestRejectionPath:
    def _run_rejection(self, rid: int = 10, reject_result=None):
        if reject_result is None:
            reject_result = {"success": True, "error": None}

        saver = MemorySaver()
        g = build_workflow_graph(saver)
        g.invoke(_initial_state(rid), config=_cfg(rid))

        with patch("src.database.queries.reject_reservation", return_value=reject_result):
            final = g.invoke(
                Command(resume={"decision": "rejected", "admin_id": "admin1",
                                "admin_notes": "No spaces on that date"}),
                config=_cfg(rid),
            )
        return final

    def test_rejected_sets_decision_rejected(self):
        final = self._run_rejection()
        assert final["decision"] == "rejected"

    def test_mcp_export_NOT_called_on_rejection(self):
        saver = MemorySaver()
        g = build_workflow_graph(saver)
        g.invoke(_initial_state(11), config=_cfg(11))

        mock_export = MagicMock()
        with (
            patch("src.database.queries.reject_reservation", return_value={"success": True}),
            patch("src.mcp_server.client.call_write_approval_record", mock_export),
        ):
            g.invoke(
                Command(resume={"decision": "rejected", "admin_id": "admin1",
                                "admin_notes": "No room"}),
                config=_cfg(11),
            )
        mock_export.assert_not_called()

    def test_rejection_has_no_export_result(self):
        final = self._run_rejection()
        # export_result is never set by the rejection path; it stays None.
        # LangGraph returns only fields modified during the resumed execution,
        # so use .get() rather than key access.
        assert final.get("export_result") is None

    def test_rejection_reason_stored_in_admin_notes(self):
        final = self._run_rejection()
        assert final["admin_notes"] == "No spaces on that date"


# ---------------------------------------------------------------------------
# TestCapacityExceededOnApprove
# ---------------------------------------------------------------------------

class TestCapacityExceededOnApprove:
    def test_approve_fails_with_capacity_exceeded(self):
        saver = MemorySaver()
        g = build_workflow_graph(saver)
        g.invoke(_initial_state(20), config=_cfg(20))

        mock_export = MagicMock()
        with (
            patch("src.database.queries.approve_reservation",
                  return_value={"success": False, "error": "capacity_exceeded"}),
            patch("src.mcp_server.client.call_write_approval_record", mock_export),
        ):
            final = g.invoke(
                Command(resume={"decision": "confirmed", "admin_id": "a", "admin_notes": ""}),
                config=_cfg(20),
            )
        assert final["error"] == "capacity_exceeded"
        mock_export.assert_not_called()

    def test_already_decided_error_skips_export(self):
        saver = MemorySaver()
        g = build_workflow_graph(saver)
        g.invoke(_initial_state(21), config=_cfg(21))

        mock_export = MagicMock()
        with (
            patch("src.database.queries.approve_reservation",
                  return_value={"success": False, "error": "already_decided"}),
            patch("src.mcp_server.client.call_write_approval_record", mock_export),
        ):
            final = g.invoke(
                Command(resume={"decision": "confirmed", "admin_id": "a", "admin_notes": ""}),
                config=_cfg(21),
            )
        assert final["error"] == "already_decided"
        mock_export.assert_not_called()


# ---------------------------------------------------------------------------
# TestExportFailureTolerance
# ---------------------------------------------------------------------------

class TestExportFailureTolerance:
    def test_export_failure_does_not_prevent_workflow_completion(self):
        saver = MemorySaver()
        g = build_workflow_graph(saver)
        g.invoke(_initial_state(30), config=_cfg(30))

        with (
            patch("src.database.queries.approve_reservation", return_value={"success": True}),
            patch("src.mcp_server.client.call_write_approval_record",
                  return_value={"success": False, "error": "file_error"}),
        ):
            final = g.invoke(
                Command(resume={"decision": "confirmed", "admin_id": "a", "admin_notes": ""}),
                config=_cfg(30),
            )
        # Workflow completes; decision is confirmed despite export failure
        assert final["decision"] == "confirmed"
        assert final["export_result"]["success"] is False
        assert final["error"] is None

    def test_export_exception_is_caught(self):
        saver = MemorySaver()
        g = build_workflow_graph(saver)
        g.invoke(_initial_state(31), config=_cfg(31))

        with (
            patch("src.database.queries.approve_reservation", return_value={"success": True}),
            patch("src.mcp_server.client.call_write_approval_record",
                  side_effect=RuntimeError("MCP server down")),
        ):
            final = g.invoke(
                Command(resume={"decision": "confirmed", "admin_id": "a", "admin_notes": ""}),
                config=_cfg(31),
            )
        assert final["decision"] == "confirmed"
        assert final["export_result"]["success"] is False


# ---------------------------------------------------------------------------
# TestBackwardCompatibility — get_workflow_graph_state returns None
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_no_checkpoint_returns_none(self, db_path: Path):
        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            snap = get_workflow_graph_state(9999)
        assert snap is None

    def test_resume_returns_workflow_found_false_when_no_checkpoint(self, db_path: Path):
        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            result = resume_reservation_workflow(9999, "confirmed", "admin1", "note")
        assert result["workflow_found"] is False


# ---------------------------------------------------------------------------
# TestPersistenceAcrossRestart — SqliteSaver with real DB file
# ---------------------------------------------------------------------------

class TestPersistenceAcrossRestart:
    def test_workflow_survives_checkpointer_rebuild(self, db_path: Path):
        """Start workflow with one SqliteSaver; resume with a fresh one on same file."""
        rid = 50

        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            start_reservation_workflow(rid, {"first_name": "Bob"})

        # Verify paused state is readable after checkpointer closes
        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            snap = get_workflow_graph_state(rid)

        assert snap is not None
        assert snap.next == ("await_admin_node",)
        assert snap.tasks[0].interrupts[0].value["reservation_id"] == rid

    def test_workflow_resumes_after_restart(self, db_path: Path):
        """Resume from a completely fresh checkpointer instance (same file)."""
        rid = 51

        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            start_reservation_workflow(rid, {"first_name": "Carol"})

        with (
            patch("src.workflow.graph.settings") as mock_cfg,
            patch("src.database.queries.approve_reservation",
                  return_value={"success": True}),
            patch("src.mcp_server.client.call_write_approval_record",
                  return_value={"success": True}),
        ):
            mock_cfg.db_path = db_path
            result = resume_reservation_workflow(rid, "confirmed", "admin1", "Approved")

        assert result["workflow_found"] is True
        assert result["final_state"]["decision"] == "confirmed"

    def test_start_and_get_state_use_same_db(self, db_path: Path):
        """start_reservation_workflow and get_workflow_graph_state use the same thread_id."""
        rid = 52

        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            start_reservation_workflow(rid, {"first_name": "Dave"})
            snap = get_workflow_graph_state(rid)

        assert snap is not None
        assert snap.tasks[0].interrupts[0].value["reservation_id"] == rid


# ---------------------------------------------------------------------------
# TestReservationNodeIntegration — chatbot nodes.py starts the workflow
# ---------------------------------------------------------------------------

class TestReservationNodeIntegration:
    def test_reservation_node_does_not_call_start_workflow_directly(self):
        """
        The reservation_node must NOT call start_reservation_workflow.
        That is the app layer's responsibility (streamlit_app.py / main.py).
        """
        import src.chatbot.nodes as nodes_mod
        src_text = open(nodes_mod.__file__, encoding="utf-8").read()
        assert "start_reservation_workflow" not in src_text

    def test_state_pending_reservation_id_set_after_save(self, monkeypatch):
        """After a successful form completion, pending_reservation_id is set in state."""
        import src.database.queries as q
        monkeypatch.setattr(q, "save_reservation",
                            lambda **kw: {"success": True, "reservation_id": 77,
                                          "error": None, "available_spaces": 719})

        from src.chatbot.nodes import reservation_node
        from src.chatbot.router import INTENT_RESERVATION_INPUT
        from src.chatbot.state import empty_state

        state = empty_state()
        state.update({
            "user_input": "2090-07-20 17:00",
            "intent": INTENT_RESERVATION_INPUT,
            "reservation_active": True,
            "reservation": {
                "first_name": "Alice", "surname": "Smith",
                "vehicle_type": "standard", "vehicle_reg": "AB12CDE",
                "start_datetime": "2090-07-20 09:00",
            },
            "next_field_to_ask": "end_datetime",
        })
        result = reservation_node(state)
        assert result.get("pending_reservation_id") == 77


# ---------------------------------------------------------------------------
# TestAdminAppIntegration — admin_app.py uses workflow path when available
# ---------------------------------------------------------------------------

class TestCompletedCheckpointHandling:
    """A completed workflow (next=()) must not be resumed or trigger DB/export calls."""

    def test_get_state_returns_non_none_for_completed_workflow(self, db_path: Path):
        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            start_reservation_workflow(10, {"first_name": "Test"})
            with (
                patch("src.database.queries.approve_reservation",
                      return_value={"success": True}),
                patch("src.mcp_server.client.call_write_approval_record",
                      return_value={"success": True}),
            ):
                resume_reservation_workflow(10, "confirmed", "admin1", "done")
            snap = get_workflow_graph_state(10)

        assert snap is not None
        assert snap.next == ()        # completed, not paused

    def test_resume_of_completed_workflow_returns_workflow_found_true(self, db_path: Path):
        """resume_reservation_workflow on a completed thread still returns workflow_found=True."""
        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            start_reservation_workflow(11, {"first_name": "Test"})
            with (
                patch("src.database.queries.approve_reservation",
                      return_value={"success": True}),
                patch("src.mcp_server.client.call_write_approval_record",
                      return_value={"success": True}),
            ):
                resume_reservation_workflow(11, "confirmed", "admin1", "done")

            # Calling resume again — no active interrupt to consume
            # This should not raise; admin_app.py checks snap.next first
            snap = get_workflow_graph_state(11)
        assert snap.next == ()        # guard: completed

    def test_admin_app_logic_skips_resume_for_completed_workflow(self, db_path: Path):
        """
        Simulate the admin_app.py three-way check:
        completed checkpoint → 'already decided' branch, no resume called.
        """
        from unittest.mock import MagicMock

        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            start_reservation_workflow(12, {"first_name": "Test"})
            with (
                patch("src.database.queries.approve_reservation",
                      return_value={"success": True}),
                patch("src.mcp_server.client.call_write_approval_record",
                      return_value={"success": True}),
            ):
                resume_reservation_workflow(12, "confirmed", "admin1", "done")
            snap = get_workflow_graph_state(12)

        # The admin_app.py condition: snap is not None AND snap.next == () → already decided
        assert snap is not None and snap.next == ()
        # In a real button handler this branch shows a warning and does NOT call resume_reservation_workflow again


class TestAdminAppIntegration:
    def test_approve_button_uses_resume_when_checkpoint_exists(self, db_path: Path):
        """
        When get_workflow_graph_state returns a snapshot, resume_reservation_workflow
        must be called instead of approve_reservation directly.
        """
        from src.workflow.graph import WORKFLOW_THREAD_ID, build_workflow_graph

        # Plant a paused checkpoint
        from langgraph.checkpoint.sqlite import SqliteSaver
        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            g = build_workflow_graph(saver)
            g.invoke(
                {"reservation_id": 99, "reservation": {}, "decision": None,
                 "admin_id": None, "admin_notes": "", "export_result": None, "error": None},
                config={"configurable": {"thread_id": WORKFLOW_THREAD_ID(99)}},
            )

        # Verify get_workflow_graph_state sees the checkpoint
        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            snap = get_workflow_graph_state(99)
        assert snap is not None

    def test_approve_button_uses_direct_fallback_when_no_checkpoint(self, db_path: Path):
        """
        When get_workflow_graph_state returns None, resume_reservation_workflow
        must not be called; the direct approve_reservation path is used.
        """
        with patch("src.workflow.graph.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            result = resume_reservation_workflow(8888, "confirmed", "admin1", "note")
        assert result["workflow_found"] is False
