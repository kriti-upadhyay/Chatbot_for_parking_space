"""
Stage 3 tests for the MCP approval logger.

Covers:
  - Field sanitisation
  - File-writer primitives (scan, append, filter-rewrite)
  - MCP tool via Client (genuine protocol)
  - Idempotency, crash recovery, partial-block with later records
  - Concurrent calls and first-fails-second-retries
  - Authentication (MCP_SHARED_SECRET, not ADMIN_PASSWORD)
  - Unauthorised invocation
"""

from __future__ import annotations

import asyncio
import hmac
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    from src.database.schema import initialise_database
    path = tmp_path / "test_mcp.db"
    initialise_database(db_path=path)
    return path


@pytest.fixture()
def log_file(tmp_path: Path) -> Path:
    return tmp_path / "approved.txt"


def _save_and_approve(db_path: Path, *, rid_override=None) -> int:
    """Save + approve a reservation; return its DB id."""
    from src.database.queries import approve_reservation, save_reservation
    res = save_reservation(
        first_name="Alice", surname="Smith",
        vehicle_type="standard", vehicle_reg="AB12CDE",
        start_datetime="2090-07-20 09:00", end_datetime="2090-07-20 17:00",
        db_path=db_path,
    )
    assert res["success"]
    rid = res["reservation_id"]
    ok = approve_reservation(rid, admin_id="admin1", db_path=db_path)
    assert ok["success"]
    return rid


def _mock_settings(db_path: Path, log_file: Path, mcp_secret: str = "test_secret"):
    """Return a context manager that patches settings for MCP tests."""
    import contextlib
    @contextlib.contextmanager
    def _ctx():
        import src.mcp_server.server as srv
        import src.mcp_server.client as cli
        import src.config.settings as cfg_mod
        mock = type("S", (), {
            "mcp_shared_secret": mcp_secret,
            "admin_password": "admin_pass",
            "db_path": db_path,
            "approval_log_file": log_file,
        })()
        with (
            patch.object(srv, "settings", mock),
            patch.object(cli, "settings", mock),
            patch("src.mcp_server.file_writer.get_logger", lambda n: __import__("logging").getLogger(n)),
        ):
            yield mock
    return _ctx()


def _make_token(secret: str = "test_secret") -> str:
    minute = int(time.time()) // 60
    return hmac.new(secret.encode(), str(minute).encode(), "sha256").hexdigest()


async def _call(reservation_id: int, db_path: Path, log_file: Path,
                secret: str = "test_secret") -> dict:
    """Call the MCP tool through the genuine FastMCP Client."""
    from fastmcp import Client
    from src.mcp_server.server import mcp
    token = _make_token(secret)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "write_approval_record",
            {"reservation_id": reservation_id, "auth_token": token},
        )
    return result.data


def _sync_call(reservation_id, db_path, log_file, secret="test_secret") -> dict:
    return asyncio.run(_call(reservation_id, db_path, log_file, secret))


# ---------------------------------------------------------------------------
# TestFieldSanitisation
# ---------------------------------------------------------------------------

class TestFieldSanitisation:
    def test_pipe_stripped_from_name(self):
        from src.mcp_server.file_writer import sanitise_field
        assert "|" not in sanitise_field("Alice|Smith")

    def test_newline_stripped(self):
        from src.mcp_server.file_writer import sanitise_field
        assert "\n" not in sanitise_field("line1\nline2")
        assert "\r" not in sanitise_field("val\rval")

    def test_long_value_truncated(self):
        from src.mcp_server.file_writer import sanitise_field
        assert len(sanitise_field("A" * 300)) == 200

    def test_control_chars_stripped(self):
        from src.mcp_server.file_writer import sanitise_field
        result = sanitise_field("AB\x00\x01\x1fCD")
        assert "\x00" not in result and "\x01" not in result

    def test_empty_field_raises_in_format_block(self):
        from src.mcp_server.file_writer import format_block
        with pytest.raises(ValueError, match="empty"):
            format_block(1, {
                "first_name": "|", "surname": "|",  # become empty after strip
                "vehicle_reg": "AB12CDE",
                "start_datetime": "2090-07-20 09:00",
                "end_datetime": "2090-07-20 17:00",
                "decided_at": "2090-07-20 11:00:00",
            })

    def test_format_block_correct_format(self):
        from src.mcp_server.file_writer import format_block, SENTINEL_PREFIX
        sentinel, data = format_block(42, {
            "first_name": "Alice", "surname": "Smith",
            "vehicle_reg": "AB12CDE",
            "start_datetime": "2090-07-20 09:00",
            "end_datetime": "2090-07-20 17:00",
            "decided_at": "2090-07-20 11:05:30",
        })
        assert sentinel == f"{SENTINEL_PREFIX}42\n"
        assert data.count("|") == 3
        assert data.endswith("\n")
        fields = data.strip().split("|")
        assert len(fields) == 4
        assert all(f.strip() for f in fields)


# ---------------------------------------------------------------------------
# TestScanReservationBlock
# ---------------------------------------------------------------------------

class TestScanReservationBlock:
    def _write(self, path: Path, content: str):
        path.write_text(content, encoding="utf-8")

    def test_complete_block_found(self, log_file: Path):
        from src.mcp_server.file_writer import scan_reservation_block, SENTINEL_PREFIX
        self._write(log_file,
            f"Name | Car Number | Reservation Period | Approval Time\n"
            f"{SENTINEL_PREFIX}42\n"
            "Alice Smith | AB12CDE | 2090-07-20 09:00 – 17:00 | 2090-07-20 11:00:00\n"
        )
        result, offset = scan_reservation_block(log_file, 42)
        assert result == "complete"
        assert offset >= 0

    def test_sentinel_at_eof_is_partial(self, log_file: Path):
        from src.mcp_server.file_writer import scan_reservation_block, SENTINEL_PREFIX
        self._write(log_file, f"{SENTINEL_PREFIX}42\n")
        result, _ = scan_reservation_block(log_file, 42)
        assert result == "partial"

    def test_sentinel_with_truncated_data_is_partial(self, log_file: Path):
        from src.mcp_server.file_writer import scan_reservation_block, SENTINEL_PREFIX
        self._write(log_file, f"{SENTINEL_PREFIX}42\nAlice Smith | AB12CDE | 2090")  # no newline
        result, _ = scan_reservation_block(log_file, 42)
        assert result == "partial"

    def test_sentinel_with_wrong_field_count_is_partial(self, log_file: Path):
        from src.mcp_server.file_writer import scan_reservation_block, SENTINEL_PREFIX
        self._write(log_file, f"{SENTINEL_PREFIX}42\nfield1 | field2 | field3\n")  # 3 fields
        result, _ = scan_reservation_block(log_file, 42)
        assert result == "partial"

    def test_no_sentinel_is_not_found(self, log_file: Path):
        from src.mcp_server.file_writer import scan_reservation_block
        self._write(log_file, "Name | Car Number | Reservation Period | Approval Time\n")
        result, offset = scan_reservation_block(log_file, 42)
        assert result == "not_found"
        assert offset == -1

    def test_nonexistent_file_is_not_found(self, tmp_path: Path):
        from src.mcp_server.file_writer import scan_reservation_block
        result, _ = scan_reservation_block(tmp_path / "missing.txt", 42)
        assert result == "not_found"

    def test_finds_correct_reservation_among_several(self, log_file: Path):
        from src.mcp_server.file_writer import scan_reservation_block, SENTINEL_PREFIX
        self._write(log_file,
            f"{SENTINEL_PREFIX}1\nAlice | A1 | 2090-01 – 2090-01 | 2090-01\n"
            f"{SENTINEL_PREFIX}42\nBob | B2 | 2090-02 – 2090-02 | 2090-02\n"
            f"{SENTINEL_PREFIX}99\nCarol | C3 | 2090-03 – 2090-03 | 2090-03\n"
        )
        assert scan_reservation_block(log_file, 42)[0] == "complete"
        assert scan_reservation_block(log_file, 1)[0] == "complete"
        assert scan_reservation_block(log_file, 50)[0] == "not_found"


# ---------------------------------------------------------------------------
# TestFileWriter
# ---------------------------------------------------------------------------

class TestFileWriter:
    def test_new_file_gets_header(self, log_file: Path):
        from src.mcp_server.file_writer import HEADER, append_block
        append_block(log_file, "# reservation:1\n",
                     "Alice | AB | 2090 – 2090 | 2090\n")
        content = log_file.read_text(encoding="utf-8")
        assert content.startswith(HEADER)

    def test_existing_file_no_second_header(self, log_file: Path):
        from src.mcp_server.file_writer import HEADER, append_block
        append_block(log_file, "# reservation:1\n", "A | B | C – D | E\n")
        append_block(log_file, "# reservation:2\n", "F | G | H – I | J\n")
        content = log_file.read_text(encoding="utf-8")
        assert content.count(HEADER) == 1

    def test_correct_pipe_delimited_format(self, log_file: Path):
        from src.mcp_server.file_writer import format_block, append_block
        sentinel, data = format_block(5, {
            "first_name": "Alice", "surname": "Smith",
            "vehicle_reg": "AB12CDE",
            "start_datetime": "2090-07-20 09:00",
            "end_datetime": "2090-07-20 17:00",
            "decided_at": "2090-07-20 11:05:30",
        })
        append_block(log_file, sentinel, data)
        lines = log_file.read_text(encoding="utf-8").splitlines()
        data_line = [l for l in lines if "|" in l and not l.startswith("Name")][0]
        assert data_line.count("|") == 3
        fields = [f.strip() for f in data_line.split("|")]
        assert fields[0] == "Alice Smith"
        assert fields[1] == "AB12CDE"
        assert "2090-07-20 09:00" in fields[2]
        assert "2090-07-20 17:00" in fields[2]
        assert fields[3] == "2090-07-20 11:05:30"


# ---------------------------------------------------------------------------
# TestFilterRewrite
# ---------------------------------------------------------------------------

class TestFilterRewrite:
    def _write(self, path: Path, content: str):
        path.write_text(content, encoding="utf-8")

    def test_partial_at_eof_replaced(self, log_file: Path):
        from src.mcp_server.file_writer import (
            SENTINEL_PREFIX, filter_rewrite_and_append, scan_reservation_block
        )
        self._write(log_file, f"Name | Car Number | Reservation Period | Approval Time\n{SENTINEL_PREFIX}42\n")
        filter_rewrite_and_append(
            log_file, 42, f"{SENTINEL_PREFIX}42\n",
            "Alice Smith | AB12CDE | 2090-07-20 09:00 – 17:00 | 2090-07-20 11:00:00\n"
        )
        assert scan_reservation_block(log_file, 42)[0] == "complete"
        content = log_file.read_text(encoding="utf-8")
        assert content.count("# reservation:42") == 1

    def test_partial_block_for_42_preserves_record_for_7(self, log_file: Path):
        """
        File: header + partial block for 42 + complete block for 7.
        Recovery for 42 must preserve 7's block exactly.
        """
        from src.mcp_server.file_writer import (
            SENTINEL_PREFIX, HEADER, filter_rewrite_and_append,
            scan_reservation_block,
        )
        self._write(log_file,
            HEADER
            + f"{SENTINEL_PREFIX}42\nAlice Smith | AB12CDE | 2090"   # truncated, no \n
            + f"\n{SENTINEL_PREFIX}7\n"
            + "Bob Jones | XY34ZAB | 2090-07-21 10:00 – 17:00 | 2090-07-21 11:00:00\n"
        )
        filter_rewrite_and_append(
            log_file, 42,
            f"{SENTINEL_PREFIX}42\n",
            "Alice Smith | AB12CDE | 2090-07-20 09:00 – 17:00 | 2090-07-20 11:00:00\n",
        )
        # 7 preserved
        assert scan_reservation_block(log_file, 7)[0] == "complete"
        # 42 complete
        assert scan_reservation_block(log_file, 42)[0] == "complete"
        # Exactly two data lines
        data_lines = [
            l for l in log_file.read_text(encoding="utf-8").splitlines()
            if l.count("|") == 3 and not l.startswith("Name")
        ]
        assert len(data_lines) == 2

    def test_partial_in_middle_preserved_later_records(self, log_file: Path):
        from src.mcp_server.file_writer import (
            SENTINEL_PREFIX, HEADER, filter_rewrite_and_append,
            scan_reservation_block,
        )
        # IDs 1, 42 (partial), 99 written in sequence
        self._write(log_file,
            HEADER
            + f"{SENTINEL_PREFIX}1\nAlice | A1 | 2090-01 – 2090-01 | 2090-01\n"
            + f"{SENTINEL_PREFIX}42\n"   # partial
            + f"{SENTINEL_PREFIX}99\nCarol | C3 | 2090-03 – 2090-03 | 2090-03\n"
        )
        filter_rewrite_and_append(
            log_file, 42,
            f"{SENTINEL_PREFIX}42\n",
            "Bob | B2 | 2090-02 – 2090-02 | 2090-02\n",
        )
        assert scan_reservation_block(log_file, 1)[0]  == "complete"
        assert scan_reservation_block(log_file, 42)[0] == "complete"
        assert scan_reservation_block(log_file, 99)[0] == "complete"


# ---------------------------------------------------------------------------
# TestMCPProtocol
# ---------------------------------------------------------------------------

class TestMCPProtocol:
    def test_client_initialises_and_tools_list_works(self, db_path, log_file):
        async def run():
            from fastmcp import Client
            from src.mcp_server.server import mcp
            with _mock_settings(db_path, log_file):
                async with Client(mcp) as client:
                    tools = await client.list_tools()
                    return [t.name for t in tools]
        names = asyncio.run(run())
        assert "write_approval_record" in names

    def test_tool_call_returns_call_tool_result(self, db_path, log_file):
        """Result must be a genuine CallToolResult, not a raw dict."""
        from fastmcp.client.client import CallToolResult
        async def run():
            from fastmcp import Client
            from src.mcp_server.server import mcp
            with _mock_settings(db_path, log_file):
                async with Client(mcp) as client:
                    return await client.call_tool(
                        "write_approval_record",
                        {"reservation_id": 9999, "auth_token": "bad"}
                    )
        result = asyncio.run(run())
        assert isinstance(result, CallToolResult)

    def test_tool_schema_has_required_params(self, db_path, log_file):
        async def run():
            from fastmcp import Client
            from src.mcp_server.server import mcp
            with _mock_settings(db_path, log_file):
                async with Client(mcp) as client:
                    tools = await client.list_tools()
                    return {t.name: t for t in tools}
        tools = asyncio.run(run())
        tool = tools["write_approval_record"]
        param_names = set(tool.input_schema.get("properties", {}).keys())
        assert "reservation_id" in param_names
        assert "auth_token" in param_names
        # output path must NOT be a parameter
        assert "file_path" not in param_names
        assert "output_path" not in param_names


# ---------------------------------------------------------------------------
# TestAuthentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_valid_token_accepted(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file, "my_secret"):
            result = _sync_call(rid, db_path, log_file, "my_secret")
        assert result["success"] is True

    def test_token_from_admin_password_rejected(self, db_path, log_file):
        """Token must be derived from MCP_SHARED_SECRET, not ADMIN_PASSWORD."""
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file, mcp_secret="mcp_secret"):
            token = hmac.new(
                b"admin_pass",  # wrong key
                str(int(time.time()) // 60).encode(), "sha256"
            ).hexdigest()
            async def run():
                from fastmcp import Client
                from src.mcp_server.server import mcp
                async with Client(mcp) as client:
                    return (await client.call_tool(
                        "write_approval_record",
                        {"reservation_id": rid, "auth_token": token}
                    )).data
            result = asyncio.run(run())
        assert result["error"] == "unauthorized"

    def test_expired_token_rejected(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file, "my_secret"):
            # Token from 5 minutes ago
            old_minute = int(time.time()) // 60 - 5
            old_token = hmac.new(b"my_secret", str(old_minute).encode(), "sha256").hexdigest()
            async def run():
                from fastmcp import Client
                from src.mcp_server.server import mcp
                async with Client(mcp) as client:
                    return (await client.call_tool(
                        "write_approval_record",
                        {"reservation_id": rid, "auth_token": old_token}
                    )).data
            result = asyncio.run(run())
        assert result["error"] == "unauthorized"

    def test_empty_token_rejected(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file):
            async def run():
                from fastmcp import Client
                from src.mcp_server.server import mcp
                async with Client(mcp) as client:
                    return (await client.call_tool(
                        "write_approval_record",
                        {"reservation_id": rid, "auth_token": ""}
                    )).data
            result = asyncio.run(run())
        assert result["error"] == "unauthorized"

    def test_unconfigured_secret_rejects_all(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file, mcp_secret=""):
            result = _sync_call(rid, db_path, log_file, "anything")
        assert result["error"] == "unauthorized"

    def test_confirmed_reservation_invalid_token_no_file_write(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file):
            async def run():
                from fastmcp import Client
                from src.mcp_server.server import mcp
                async with Client(mcp) as client:
                    return (await client.call_tool(
                        "write_approval_record",
                        {"reservation_id": rid, "auth_token": "invalid"}
                    )).data
            result = asyncio.run(run())
        assert result["error"] == "unauthorized"
        assert not log_file.exists()

    def test_confirmed_reservation_invalid_token_exported_at_unchanged(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file):
            async def run():
                from fastmcp import Client
                from src.mcp_server.server import mcp
                async with Client(mcp) as client:
                    await client.call_tool(
                        "write_approval_record",
                        {"reservation_id": rid, "auth_token": "bad"}
                    )
            asyncio.run(run())
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT exported_at FROM reservations WHERE id=?", (rid,)).fetchone()
        conn.close()
        assert row[0] is None   # exported_at unchanged


# ---------------------------------------------------------------------------
# TestUnauthorisedInvocation
# ---------------------------------------------------------------------------

class TestUnauthorisedInvocation:
    def _call_with_valid_token(self, rid, db_path, log_file):
        return _sync_call(rid, db_path, log_file)

    def test_pending_reservation_not_confirmed(self, db_path, log_file):
        from src.database.queries import save_reservation
        res = save_reservation(
            first_name="Alice", surname="Smith", vehicle_type="standard",
            vehicle_reg="AB12CDE", start_datetime="2090-07-20 09:00",
            end_datetime="2090-07-20 17:00", db_path=db_path,
        )
        rid = res["reservation_id"]
        with _mock_settings(db_path, log_file):
            result = self._call_with_valid_token(rid, db_path, log_file)
        assert result["error"] == "not_confirmed"
        assert result.get("current_status") == "pending"
        assert not log_file.exists()

    def test_rejected_reservation_not_confirmed(self, db_path, log_file):
        from src.database.queries import reject_reservation, save_reservation
        res = save_reservation(
            first_name="A", surname="B", vehicle_type="standard",
            vehicle_reg="XY34ZAB", start_datetime="2090-08-01 09:00",
            end_datetime="2090-08-01 17:00", db_path=db_path,
        )
        rid = res["reservation_id"]
        reject_reservation(rid, "admin1", "test", db_path=db_path)
        with _mock_settings(db_path, log_file):
            result = self._call_with_valid_token(rid, db_path, log_file)
        assert result["error"] == "not_confirmed"

    def test_nonexistent_id_not_found(self, db_path, log_file):
        with _mock_settings(db_path, log_file):
            result = self._call_with_valid_token(99999, db_path, log_file)
        assert result["error"] == "not_found"
        assert not log_file.exists()


# ---------------------------------------------------------------------------
# TestIdempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_second_call_returns_already_exported(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file):
            r1 = _sync_call(rid, db_path, log_file)
            r2 = _sync_call(rid, db_path, log_file)
        assert r1["success"] is True
        assert r1.get("already_exported") is False
        assert r2["success"] is True
        assert r2.get("already_exported") is True

    def test_file_has_exactly_one_record_after_two_calls(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file):
            _sync_call(rid, db_path, log_file)
            _sync_call(rid, db_path, log_file)
        data_lines = [
            l for l in log_file.read_text(encoding="utf-8").splitlines()
            if l.count("|") == 3 and not l.startswith("Name")
        ]
        assert len(data_lines) == 1

    def test_third_call_still_already_exported(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file):
            _sync_call(rid, db_path, log_file)
            _sync_call(rid, db_path, log_file)
            r3 = _sync_call(rid, db_path, log_file)
        assert r3["already_exported"] is True


# ---------------------------------------------------------------------------
# TestCrashRecovery
# ---------------------------------------------------------------------------

class TestCrashRecovery:
    def _set_exporting(self, db_path, rid):
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE reservations SET exported_at='EXPORTING' WHERE id=?", (rid,))
        conn.commit()
        conn.close()

    def test_exporting_no_file_retries_and_writes(self, db_path, log_file):
        """Crash after EXPORTING committed but before file write."""
        rid = _save_and_approve(db_path)
        self._set_exporting(db_path, rid)
        with _mock_settings(db_path, log_file):
            result = _sync_call(rid, db_path, log_file)
        assert result["success"] is True
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT exported_at FROM reservations WHERE id=?", (rid,)).fetchone()
        conn.close()
        assert row[0] not in (None, "EXPORTING")

    def test_exporting_complete_file_completes_db(self, db_path, log_file):
        """Crash after file write but before DB update."""
        from src.mcp_server.file_writer import (
            SENTINEL_PREFIX, HEADER, append_block, format_block
        )
        rid = _save_and_approve(db_path)
        self._set_exporting(db_path, rid)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM reservations WHERE id=?", (rid,)).fetchone())
        conn.close()
        row["decided_at"] = row["decided_at"] or "2090-07-20 11:05:30"
        sentinel, data = format_block(rid, row)
        append_block(log_file, sentinel, data)

        with _mock_settings(db_path, log_file):
            result = _sync_call(rid, db_path, log_file)

        assert result["success"] is True
        # Only one data line in file
        data_lines = [
            l for l in log_file.read_text(encoding="utf-8").splitlines()
            if l.count("|") == 3 and not l.startswith("Name")
        ]
        assert len(data_lines) == 1

    def test_exporting_partial_file_rewrites(self, db_path, log_file):
        """Crash mid-write: sentinel on disk, data line truncated."""
        from src.mcp_server.file_writer import SENTINEL_PREFIX, HEADER
        rid = _save_and_approve(db_path)
        self._set_exporting(db_path, rid)
        # Write partial block (sentinel + truncated data, no newline)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            HEADER + f"{SENTINEL_PREFIX}{rid}\nAlice Smith | AB12CDE | 2090",
            encoding="utf-8",
        )
        with _mock_settings(db_path, log_file):
            result = _sync_call(rid, db_path, log_file)
        assert result["success"] is True
        data_lines = [
            l for l in log_file.read_text(encoding="utf-8").splitlines()
            if l.count("|") == 3 and not l.startswith("Name")
        ]
        assert len(data_lines) == 1

    def test_crash_recovery_preserves_other_records(self, db_path, log_file):
        """
        Partial block for reservation 42 followed by a complete block for 7.
        Recovery for 42 must preserve 7.
        """
        from src.database.queries import approve_reservation, save_reservation
        from src.mcp_server.file_writer import SENTINEL_PREFIX, HEADER

        r7 = save_reservation(
            first_name="Bob", surname="Jones", vehicle_type="motorcycle",
            vehicle_reg="XY34ZAB",
            start_datetime="2090-07-21 10:00", end_datetime="2090-07-21 18:00",
            db_path=db_path,
        )
        rid7 = r7["reservation_id"]
        approve_reservation(rid7, "admin1", db_path=db_path)

        rid42 = _save_and_approve(db_path)
        self._set_exporting(db_path, rid42)

        # Plant: partial block for rid42, complete for rid7
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            HEADER
            + f"{SENTINEL_PREFIX}{rid42}\nAlice Smith | AB12CDE | 2090"   # partial
            + f"\n{SENTINEL_PREFIX}{rid7}\n"
            + "Bob Jones | XY34ZAB | 2090-07-21 10:00 – 18:00 | 2090-07-21 11:00:00\n",
            encoding="utf-8",
        )
        with _mock_settings(db_path, log_file):
            result = _sync_call(rid42, db_path, log_file)

        assert result["success"] is True
        content = log_file.read_text(encoding="utf-8")
        data_lines = [l for l in content.splitlines() if l.count("|") == 3 and not l.startswith("Name")]
        assert len(data_lines) == 2
        assert "Bob Jones" in content
        assert "Alice Smith" in content


# ---------------------------------------------------------------------------
# TestConcurrentCalls
# ---------------------------------------------------------------------------

class TestConcurrentCalls:
    def test_two_concurrent_calls_produce_one_record(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        results = []

        def worker():
            with _mock_settings(db_path, log_file):
                results.append(_sync_call(rid, db_path, log_file))

        t1, t2 = threading.Thread(target=worker), threading.Thread(target=worker)
        t1.start(); t2.start()
        t1.join(); t2.join()

        successes = [r for r in results if r.get("success")]
        assert len(successes) == 2  # both see success (one exports, one already_exported)
        already = sum(1 for r in results if r.get("already_exported"))
        assert already <= 1  # at most one sees already_exported

        data_lines = [
            l for l in log_file.read_text(encoding="utf-8").splitlines()
            if l.count("|") == 3 and not l.startswith("Name")
        ]
        assert len(data_lines) == 1

    def test_first_fails_second_retries(self, db_path, log_file):
        """
        First call: file write raises IOError → EXPORTING stays in DB.
        Second call (retry): recovers, writes, sets exported_at.
        """
        rid = _save_and_approve(db_path)
        write_call_count = {"n": 0}
        original_append = __import__("src.mcp_server.file_writer", fromlist=["append_block"]).append_block

        def failing_then_ok(file_path, sentinel, data):
            write_call_count["n"] += 1
            if write_call_count["n"] == 1:
                raise IOError("Simulated disk full")
            original_append(file_path, sentinel, data)

        with _mock_settings(db_path, log_file):
            with patch("src.mcp_server.server.append_block", side_effect=failing_then_ok):
                r1 = _sync_call(rid, db_path, log_file)

            # Verify EXPORTING is still in DB (not reset to NULL)
            conn = sqlite3.connect(str(db_path))
            exported = conn.execute(
                "SELECT exported_at FROM reservations WHERE id=?", (rid,)
            ).fetchone()[0]
            conn.close()
            assert exported == "EXPORTING"

            # Second call (retry) succeeds
            r2 = _sync_call(rid, db_path, log_file)

        assert r1["error"] == "file_error"
        assert r2["success"] is True

        # DB now has a real timestamp
        conn = sqlite3.connect(str(db_path))
        exported = conn.execute(
            "SELECT exported_at FROM reservations WHERE id=?", (rid,)
        ).fetchone()[0]
        conn.close()
        assert exported not in (None, "EXPORTING")

        data_lines = [
            l for l in log_file.read_text(encoding="utf-8").splitlines()
            if l.count("|") == 3 and not l.startswith("Name")
        ]
        assert len(data_lines) == 1


# ---------------------------------------------------------------------------
# TestSuccessPath
# ---------------------------------------------------------------------------

class TestSuccessPath:
    def test_approval_log_has_correct_record(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file):
            result = _sync_call(rid, db_path, log_file)
        assert result["success"] is True
        assert result.get("record")
        content = log_file.read_text(encoding="utf-8")
        assert "Alice Smith" in content
        assert "AB12CDE" in content
        assert "2090-07-20 09:00" in content
        assert "2090-07-20 17:00" in content

    def test_exported_at_set_after_success(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file):
            _sync_call(rid, db_path, log_file)
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT exported_at FROM reservations WHERE id=?", (rid,)).fetchone()
        conn.close()
        assert row[0] not in (None, "EXPORTING")

    def test_exported_at_not_set_when_file_write_fails(self, db_path, log_file):
        rid = _save_and_approve(db_path)
        with _mock_settings(db_path, log_file):
            with patch("src.mcp_server.server.append_block", side_effect=IOError("fail")):
                result = _sync_call(rid, db_path, log_file)
        assert result["error"] == "file_error"
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT exported_at FROM reservations WHERE id=?", (rid,)).fetchone()
        conn.close()
        assert row[0] == "EXPORTING"   # not NULL and not a real timestamp
