"""
Atomic file-writing for approved reservation records.

Output format per approved reservation:

    # reservation:<id>               ← sentinel (not in required visible format)
    Name | Car Number | Reservation Period | Approval Time

A complete block is the sentinel line immediately followed by a valid
4-field pipe-delimited data line ending with a newline.  Any block where
the data line is missing, truncated, or malformed is a partial block.

The header line ("Name | Car Number | Reservation Period | Approval Time")
is written once when the file is new or empty.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

from src.config import get_logger

logger = get_logger(__name__)

SENTINEL_PREFIX: str = "# reservation:"
HEADER: str = "Name | Car Number | Reservation Period | Approval Time\n"
_MAX_FIELD: int = 200
# Period separator rendered as en-dash (visible format)
_PERIOD_SEP: str = " – "


# ---------------------------------------------------------------------------
# Field sanitisation
# ---------------------------------------------------------------------------

def sanitise_field(value: str) -> str:
    """Strip pipe characters, newlines, control chars; truncate to 200 chars."""
    value = re.sub(r"[\|\n\r\x00-\x1f\x7f]", "", value).strip()
    return value[:_MAX_FIELD]


def format_block(reservation_id: int, reservation: dict) -> Tuple[str, str]:
    """
    Build the sentinel and data line for a reservation.

    Returns:
        (sentinel_line, data_line) — both end with \\n.

    Raises:
        ValueError: if any field is empty after sanitisation.
    """
    name = sanitise_field(
        f"{reservation['first_name']} {reservation['surname']}"
    )
    reg = sanitise_field(reservation["vehicle_reg"])
    start = sanitise_field(reservation["start_datetime"])
    end = sanitise_field(reservation["end_datetime"])
    decided = sanitise_field(reservation.get("decided_at") or "")

    period = f"{start}{_PERIOD_SEP}{end}"

    for fname, fval in [
        ("name", name), ("reg", reg), ("start", start),
        ("end", end), ("approval_time", decided),
    ]:
        if not fval:
            raise ValueError(
                f"Field '{fname}' is empty after sanitisation "
                f"for reservation {reservation_id}"
            )

    sentinel = f"{SENTINEL_PREFIX}{reservation_id}\n"
    data = f"{name} | {reg} | {period} | {decided}\n"
    return sentinel, data


# ---------------------------------------------------------------------------
# Block scanning
# ---------------------------------------------------------------------------

def _is_valid_data_line(line: str) -> bool:
    """Return True if *line* is a non-empty, 4-field pipe-delimited record."""
    # Accept both LF and CRLF line endings (Windows vs Unix)
    stripped = line.rstrip("\r\n")
    if not stripped:
        return False
    # Must have ended with a newline (not truncated mid-line at EOF)
    if not line.endswith("\n"):
        return False
    parts = stripped.split("|")
    return len(parts) == 4 and all(p.strip() for p in parts)


def scan_reservation_block(
    file_path: Path,
    reservation_id: int,
) -> Tuple[str, int]:
    """
    Scan *file_path* for the block belonging to *reservation_id*.

    Returns:
        ('complete', byte_offset)  — sentinel + valid data line found
        ('partial',  byte_offset)  — sentinel found but data line missing/invalid
        ('not_found', -1)          — sentinel not present
    """
    if not file_path.exists():
        return "not_found", -1

    sentinel_bare = f"{SENTINEL_PREFIX}{reservation_id}"

    with open(file_path, "rb") as f:
        content_bytes = f.read()

    content = content_bytes.decode("utf-8", errors="replace")
    lines = content.splitlines(keepends=True)

    byte_pos = 0
    for i, line in enumerate(lines):
        # Strip trailing CR/LF before comparing sentinel
        if line.rstrip("\r\n") == sentinel_bare:
            offset = byte_pos
            if i + 1 < len(lines) and _is_valid_data_line(lines[i + 1]):
                return "complete", offset
            else:
                return "partial", offset
        byte_pos += len(line.encode("utf-8"))

    return "not_found", -1


# ---------------------------------------------------------------------------
# Block collection (for filter-rewrite)
# ---------------------------------------------------------------------------

def _collect_complete_blocks(
    file_path: Path,
    exclude_id: int | None = None,
) -> list[str]:
    """
    Parse *file_path* and return all complete blocks as raw strings.

    Partial blocks and orphan lines are discarded.  The optional
    *exclude_id* is silently dropped even if complete (used during
    recovery to remove a reservation's old partial/complete block before
    rewriting it).
    """
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip("\r\n").startswith(SENTINEL_PREFIX):
            try:
                rid = int(line.rstrip("\r\n")[len(SENTINEL_PREFIX):].strip())
            except ValueError:
                i += 1
                continue
            if (
                i + 1 < len(lines)
                and _is_valid_data_line(lines[i + 1])
            ):
                if rid != exclude_id:
                    blocks.append(line + lines[i + 1])
                i += 2
            else:
                i += 1  # partial — discard
        else:
            i += 1  # header or orphan — skip
    return blocks


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def append_block(file_path: Path, sentinel: str, data: str) -> None:
    """
    Append a complete block to *file_path*.

    Writes the header if the file is new or empty.
    Uses a single write() call for both sentinel and data, then fsyncs.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = (not file_path.exists()) or file_path.stat().st_size == 0
    mode = "a"

    with open(file_path, mode, encoding="utf-8") as f:
        if is_new:
            f.write(HEADER)
        f.write(sentinel + data)
        f.flush()
        os.fsync(f.fileno())

    logger.debug("Appended block to %s", file_path)


def filter_rewrite_and_append(
    file_path: Path,
    reservation_id: int,
    sentinel: str,
    data: str,
) -> None:
    """
    Rewrite *file_path* preserving all complete blocks except any existing
    block for *reservation_id*, then append the new complete block.

    Uses a temp file + atomic os.replace() so readers never see a partial
    file.  fsyncs the temp file before the rename.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Collect all complete blocks excluding the one we're (re)writing
    complete_blocks = _collect_complete_blocks(file_path, exclude_id=reservation_id)

    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(HEADER)
        for block in complete_blocks:
            f.write(block)
        f.write(sentinel + data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, file_path)
    logger.debug(
        "Filter-rewrote %s (kept %d blocks, added reservation %d)",
        file_path, len(complete_blocks), reservation_id,
    )
