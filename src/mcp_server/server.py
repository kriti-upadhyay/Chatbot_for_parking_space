"""
CityPark Approval Logger — FastMCP server.

Exposes one tool:
    write_approval_record(reservation_id, auth_token)

The LLM is never given access to this tool.  It is called exclusively by
the authenticated admin dashboard (admin_app.py) via the MCP client in
client.py, after the UI Approve button is clicked.

Security model
--------------
- auth_token:  time-based HMAC-SHA256 signed with MCP_SHARED_SECRET
  (a secret independent of ADMIN_PASSWORD).  Valid for the current
  minute and the previous minute (~1-2 min window).
- The tool validates the token before any DB or file operation.
- Checking `status = 'confirmed'` is NOT authentication; it is a
  precondition check that happens after auth.

Reliability model (protocol, inside the file lock)
---------------------------------------------------
1.  Acquire exclusive file lock (filelock.FileLock).
2.  BEGIN IMMEDIATE; read exported_at:
    • real timestamp  → return already_exported (idempotent).
    • 'EXPORTING'     → enter recovery path.
    • NULL            → claim EXPORTING; proceed to write.
3.  Scan file (recovery path):
    • 'complete' → skip write; fall through to DB update.
    • 'partial'  → filter-rewrite file (preserves all other valid records).
    • 'not_found'→ proceed to append.
4.  Write block; fsync.
5.  Release file lock.
6.  BEGIN IMMEDIATE; UPDATE exported_at = now() WHERE exported_at='EXPORTING'.
"""

from __future__ import annotations

import hmac
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import FastMCP

from src.config import get_logger, settings
from src.mcp_server.file_writer import (
    append_block,
    filter_rewrite_and_append,
    format_block,
    scan_reservation_block,
)

logger = get_logger(__name__)

mcp = FastMCP("CityPark Approval Logger")

_EXPORTING = "EXPORTING"


# ---------------------------------------------------------------------------
# Auth token helpers
# ---------------------------------------------------------------------------

def _validate_mcp_token(token: str) -> bool:
    """
    Validate a time-based HMAC-SHA256 token derived from MCP_SHARED_SECRET.

    Accepts tokens from the current minute and the previous minute to
    absorb minor clock skew between the caller and the server.
    """
    secret = settings.mcp_shared_secret
    if not secret or not token:
        return False
    minute = int(time.time()) // 60
    for t in (minute, minute - 1):
        expected = hmac.new(
            secret.encode("utf-8"),
            str(t).encode("utf-8"),
            "sha256",
        ).hexdigest()
        if hmac.compare_digest(
            token.encode("utf-8"), expected.encode("utf-8")
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def write_approval_record(
    reservation_id: int,
    auth_token: str,
) -> dict:
    """
    Write an approved reservation record to the approval log file.

    The output file path is read from settings; the caller cannot override it.

    Returns a dict with keys: success, error, record, already_exported.
    """
    # ── 1. Authenticate ───────────────────────────────────────────────────────
    if not _validate_mcp_token(auth_token):
        logger.warning(
            "MCP tool called with invalid auth_token for reservation %d",
            reservation_id,
        )
        return {"success": False, "error": "unauthorized"}

    log_file: Path = settings.approval_log_file
    lock_path = log_file.with_suffix(log_file.suffix + ".lock")
    db_path = settings.db_path

    from filelock import FileLock, Timeout

    with FileLock(str(lock_path), timeout=10):
        # ── 2. Read DB state (inside file lock — no stale reads) ───────────
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        action = None
        reservation = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM reservations WHERE id = ?",
                (reservation_id,),
            ).fetchone()

            if row is None:
                conn.rollback()
                return {"success": False, "error": "not_found"}

            if row["status"] != "confirmed":
                conn.rollback()
                return {
                    "success": False,
                    "error": "not_confirmed",
                    "current_status": row["status"],
                }

            exported_at = row["exported_at"]

            if exported_at and exported_at != _EXPORTING:
                # Already successfully exported — idempotent return.
                conn.rollback()
                return {"success": True, "already_exported": True, "error": None}

            reservation = dict(row)

            if exported_at == _EXPORTING:
                action = "recover"
                conn.rollback()
            else:
                # NULL — claim the export slot atomically.
                conn.execute(
                    "UPDATE reservations SET exported_at = ? WHERE id = ?",
                    (_EXPORTING, reservation_id),
                )
                conn.commit()
                action = "write"

        except sqlite3.Error:
            conn.rollback()
            logger.exception("DB read failed for reservation %d", reservation_id)
            return {"success": False, "error": "db_error"}
        finally:
            conn.close()

        # ── 3. Determine write action ──────────────────────────────────────
        sentinel, data = None, None
        try:
            sentinel, data = format_block(reservation_id, reservation)
        except ValueError as exc:
            logger.error("format_block failed: %s", exc)
            return {"success": False, "error": "invalid_field", "message": str(exc)}

        if action == "recover":
            scan_result, _ = scan_reservation_block(log_file, reservation_id)

            if scan_result == "complete":
                # File write already happened; skip to DB update.
                logger.info(
                    "Recovery: reservation %d already written to file.", reservation_id
                )

            elif scan_result == "partial":
                try:
                    filter_rewrite_and_append(log_file, reservation_id, sentinel, data)
                    logger.info(
                        "Recovery: rewrote file, added reservation %d.", reservation_id
                    )
                except Exception as exc:
                    logger.exception(
                        "filter_rewrite_and_append failed for reservation %d", reservation_id
                    )
                    return {
                        "success": False,
                        "error": "file_error",
                        "message": str(exc),
                    }

            else:  # not_found
                try:
                    append_block(log_file, sentinel, data)
                    logger.info(
                        "Recovery: appended reservation %d to file.", reservation_id
                    )
                except Exception as exc:
                    logger.exception(
                        "append_block failed for reservation %d", reservation_id
                    )
                    return {
                        "success": False,
                        "error": "file_error",
                        "message": str(exc),
                    }

        else:  # action == "write"
            try:
                append_block(log_file, sentinel, data)
                logger.info("Appended reservation %d to approval log.", reservation_id)
            except Exception as exc:
                logger.exception(
                    "append_block failed for reservation %d", reservation_id
                )
                # exported_at stays 'EXPORTING' for retry
                return {
                    "success": False,
                    "error": "file_error",
                    "message": str(exc),
                }

    # ── File lock released ────────────────────────────────────────────────────

    # ── 6. Update exported_at ─────────────────────────────────────────────────
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE reservations SET exported_at = ? WHERE id = ? AND exported_at = ?",
            (now_str, reservation_id, _EXPORTING),
        )
        conn.commit()
    except sqlite3.Error:
        logger.exception(
            "Failed to set exported_at for reservation %d", reservation_id
        )
        return {"success": False, "error": "db_error"}
    finally:
        conn.close()

    record = data.rstrip("\n") if data else ""
    logger.info("Reservation %d export complete.", reservation_id)
    return {
        "success": True,
        "error": None,
        "record": record,
        "already_exported": False,
    }
