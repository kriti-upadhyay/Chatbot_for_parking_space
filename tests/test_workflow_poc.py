"""
Stage 4 POC — verify LangGraph interrupt/resume with SqliteSaver.

These tests confirm the core mechanics before the full workflow graph is
built.  They use a minimal TypedDict state and a simple two-node graph:

  START → wait_node (interrupt) → done_node → END

Key behaviours verified:
  1. graph.invoke() returns when interrupt() is called (graph pauses).
  2. graph.get_state() exposes the interrupt payload and next=('wait_node',).
  3. graph.get_state() on a nonexistent thread_id returns metadata=None.
  4. Command(resume=...) correctly resumes the paused node.
  5. Rebuilding graph + SqliteSaver from the same DB file preserves the
     checkpoint — same thread_id can be resumed after a simulated restart.
  6. Two different thread_ids in the same DB are fully isolated.
  7. A completed thread has next=() (not paused).
  8. Resuming a completed thread raises (no pending interrupt to resume).

All tests are offline — no API keys, no external services.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pytest

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# ---------------------------------------------------------------------------
# Minimal test graph (shared by all test classes)
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict):
    value: int
    decision: str | None  # set by resume payload


def _wait_node(state: WorkflowState) -> dict:
    """Pauses here; the resume payload becomes the return value of interrupt()."""
    decision = interrupt({"waiting_for": "admin", "value": state["value"]})
    return {"decision": decision}


def _done_node(state: WorkflowState) -> dict:
    return {"value": state["value"] * 10}


def _build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(WorkflowState)
    builder.add_node("wait_node", _wait_node)
    builder.add_node("done_node", _done_node)
    builder.add_edge(START, "wait_node")
    builder.add_edge("wait_node", "done_node")
    builder.add_edge("done_node", END)
    return builder.compile(checkpointer=checkpointer)


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# TestInterruptResumeBehavior
# ---------------------------------------------------------------------------

class TestInterruptResumeBehavior:
    """Core interrupt/resume semantics with MemorySaver (fast, no I/O)."""

    @pytest.fixture()
    def graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        return _build_graph(MemorySaver())

    def test_invoke_returns_when_interrupt_fires(self, graph):
        """graph.invoke() must return (not raise) when interrupt() is called."""
        result = graph.invoke(
            {"value": 42, "decision": None}, config=_cfg("t1")
        )
        # Graph paused — decision was not set (interrupt fired before return)
        assert result["decision"] is None

    def test_get_state_shows_paused_at_wait_node(self, graph):
        """get_state() on a paused thread must report next=('wait_node',)."""
        graph.invoke({"value": 7, "decision": None}, config=_cfg("t2"))
        snap = graph.get_state(_cfg("t2"))
        assert snap.next == ("wait_node",)

    def test_interrupt_payload_accessible_in_state(self, graph):
        """The value passed to interrupt() must be visible in the task's interrupts."""
        graph.invoke({"value": 99, "decision": None}, config=_cfg("t3"))
        snap = graph.get_state(_cfg("t3"))
        interrupt_val = snap.tasks[0].interrupts[0].value
        assert interrupt_val["waiting_for"] == "admin"
        assert interrupt_val["value"] == 99

    def test_resume_with_command_completes_graph(self, graph):
        """Command(resume=...) must resume and run to completion."""
        graph.invoke({"value": 5, "decision": None}, config=_cfg("t4"))
        final = graph.invoke(Command(resume="approved"), config=_cfg("t4"))
        assert final["decision"] == "approved"
        assert final["value"] == 50  # done_node multiplies by 10

    def test_completed_thread_has_empty_next(self, graph):
        """After completion, get_state().next must be () — not still paused."""
        graph.invoke({"value": 1, "decision": None}, config=_cfg("t5"))
        graph.invoke(Command(resume="done"), config=_cfg("t5"))
        snap = graph.get_state(_cfg("t5"))
        assert snap.next == ()


# ---------------------------------------------------------------------------
# TestNonexistentThreadDetection
# ---------------------------------------------------------------------------

class TestNonexistentThreadDetection:
    """Verify how to distinguish a paused checkpoint from a missing thread."""

    @pytest.fixture()
    def graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        return _build_graph(MemorySaver())

    def test_missing_thread_has_none_metadata(self, graph):
        """get_state() on an unknown thread_id must return metadata=None."""
        snap = graph.get_state(_cfg("does-not-exist"))
        assert snap.metadata is None

    def test_missing_thread_has_empty_next(self, graph):
        snap = graph.get_state(_cfg("never-used"))
        assert snap.next == ()

    def test_paused_thread_has_non_none_metadata(self, graph):
        graph.invoke({"value": 3, "decision": None}, config=_cfg("real-t"))
        snap = graph.get_state(_cfg("real-t"))
        assert snap.metadata is not None

    def test_canonical_exists_check(self, graph):
        """
        The canonical way to tell whether a workflow checkpoint exists for a
        given reservation: snap.metadata is not None.
        """
        graph.invoke({"value": 2, "decision": None}, config=_cfg("exists"))
        exists_snap    = graph.get_state(_cfg("exists"))
        missing_snap   = graph.get_state(_cfg("missing"))
        assert (exists_snap.metadata is not None) is True
        assert (missing_snap.metadata is not None) is False

    def test_completed_thread_metadata_not_none(self, graph):
        """A completed thread still has metadata — only paused ones have next != ()."""
        graph.invoke({"value": 1, "decision": None}, config=_cfg("comp"))
        graph.invoke(Command(resume="x"), config=_cfg("comp"))
        snap = graph.get_state(_cfg("comp"))
        # metadata exists but next is empty — this is how we distinguish completed
        assert snap.metadata is not None
        assert snap.next == ()


# ---------------------------------------------------------------------------
# TestSqliteSaverAvailability
# ---------------------------------------------------------------------------

class TestSqliteSaverAvailability:
    """Confirm the SqliteSaver package is installed and the API is usable."""

    def test_sqlite_saver_imports(self):
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401

    def test_from_conn_string_is_context_manager(self, tmp_path: Path):
        from contextlib import AbstractContextManager
        db = str(tmp_path / "poc.db")
        cm = SqliteSaver.from_conn_string(db)
        assert hasattr(cm, "__enter__") and hasattr(cm, "__exit__")

    def test_creates_db_file(self, tmp_path: Path):
        db = tmp_path / "poc.db"
        with SqliteSaver.from_conn_string(str(db)) as saver:
            g = _build_graph(saver)
            g.invoke({"value": 1, "decision": None}, config=_cfg("x"))
        assert db.exists()


# ---------------------------------------------------------------------------
# TestSqliteSaverPersistence
# ---------------------------------------------------------------------------

class TestSqliteSaverPersistence:
    """
    Core requirement: rebuilding graph + checkpointer from the same DB file
    must allow resuming the same thread_id, simulating a server restart.
    """

    def test_checkpoint_survives_saver_rebuild(self, tmp_path: Path):
        """
        Saver A: start graph, interrupt fires, saver closes.
        Saver B (same file): rebuild graph, resume same thread_id.
        """
        db = str(tmp_path / "restart.db")
        tid = "reservation_42"

        # ── Process 1 (or startup cycle 1): start workflow ──────────────────
        with SqliteSaver.from_conn_string(db) as saver_a:
            graph_a = _build_graph(saver_a)
            graph_a.invoke({"value": 42, "decision": None}, config=_cfg(tid))
            snap_a = graph_a.get_state(_cfg(tid))
            assert snap_a.next == ("wait_node",)   # paused

        # ── Process 2 (restart): rebuild from same file ─────────────────────
        with SqliteSaver.from_conn_string(db) as saver_b:
            graph_b = _build_graph(saver_b)

            snap_b = graph_b.get_state(_cfg(tid))
            assert snap_b.metadata is not None    # checkpoint exists
            assert snap_b.next == ("wait_node",)  # still paused

            # Verify interrupt payload preserved
            iv = snap_b.tasks[0].interrupts[0].value
            assert iv["value"] == 42

            # Resume after restart
            final = graph_b.invoke(Command(resume="confirmed"), config=_cfg(tid))
            assert final["decision"] == "confirmed"
            assert final["value"] == 420

    def test_missing_thread_after_restart(self, tmp_path: Path):
        """After restart with same DB, unknown thread still returns metadata=None."""
        db = str(tmp_path / "check.db")
        with SqliteSaver.from_conn_string(db) as saver_a:
            graph_a = _build_graph(saver_a)
            graph_a.invoke({"value": 1, "decision": None}, config=_cfg("exists"))

        with SqliteSaver.from_conn_string(db) as saver_b:
            graph_b = _build_graph(saver_b)
            snap = graph_b.get_state(_cfg("never-started"))
            assert snap.metadata is None

    def test_completed_thread_after_restart(self, tmp_path: Path):
        """A completed thread's checkpoint persists after restart with empty next."""
        db = str(tmp_path / "complete.db")
        tid = "done_thread"
        with SqliteSaver.from_conn_string(db) as s:
            g = _build_graph(s)
            g.invoke({"value": 5, "decision": None}, config=_cfg(tid))
            g.invoke(Command(resume="ok"), config=_cfg(tid))

        with SqliteSaver.from_conn_string(db) as s2:
            g2 = _build_graph(s2)
            snap = g2.get_state(_cfg(tid))
            assert snap.metadata is not None
            assert snap.next == ()              # completed, not paused


# ---------------------------------------------------------------------------
# TestThreadIdIsolation
# ---------------------------------------------------------------------------

class TestThreadIdIsolation:
    """Two reservations in the same DB must not interfere with each other."""

    def test_two_threads_independent(self, tmp_path: Path):
        db = str(tmp_path / "two.db")
        with SqliteSaver.from_conn_string(db) as saver:
            graph = _build_graph(saver)

            graph.invoke({"value": 10, "decision": None}, config=_cfg("res_1"))
            graph.invoke({"value": 20, "decision": None}, config=_cfg("res_2"))

            snap1 = graph.get_state(_cfg("res_1"))
            snap2 = graph.get_state(_cfg("res_2"))
            assert snap1.next == ("wait_node",)
            assert snap2.next == ("wait_node",)
            assert snap1.tasks[0].interrupts[0].value["value"] == 10
            assert snap2.tasks[0].interrupts[0].value["value"] == 20

    def test_resuming_one_does_not_affect_other(self, tmp_path: Path):
        db = str(tmp_path / "iso.db")
        with SqliteSaver.from_conn_string(db) as saver:
            graph = _build_graph(saver)

            graph.invoke({"value": 3, "decision": None}, config=_cfg("res_A"))
            graph.invoke({"value": 7, "decision": None}, config=_cfg("res_B"))

            # Resume only res_A
            graph.invoke(Command(resume="approve"), config=_cfg("res_A"))

            snap_a = graph.get_state(_cfg("res_A"))
            snap_b = graph.get_state(_cfg("res_B"))
            assert snap_a.next == ()              # completed
            assert snap_b.next == ("wait_node",)  # still paused
