"""
Parameterised query functions for dynamic parking data.

All functions accept an optional ``db_path`` argument so they can be
pointed at a test or alternate database without touching global settings.

Return values are plain dicts (not sqlite3.Row objects) so they serialise
cleanly to JSON and are easy to assert against in tests.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.config import get_logger, settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def _connection(db_path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Yield a short-lived, read-only-friendly SQLite connection."""
    path = db_path or settings.db_path
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Pricing queries
# ---------------------------------------------------------------------------

def get_prices(
    space_type: str | None = None,
    db_path: Path | None = None,
) -> list[dict]:
    """
    Return pricing tiers, optionally filtered by *space_type*.

    Args:
        space_type: One of standard | disabled | ev_charging | motorcycle |
                    oversized.  If None, all types are returned.

    Returns:
        List of dicts with keys: id, space_type, tier_name, max_duration_h,
        price_gbp, description.  Ordered by space_type, then price ascending.
    """
    with _connection(db_path) as conn:
        if space_type:
            rows = conn.execute(
                """
                SELECT * FROM parking_prices
                WHERE space_type = ?
                ORDER BY price_gbp ASC
                """,
                (space_type.lower(),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM parking_prices
                ORDER BY space_type, price_gbp ASC
                """
            ).fetchall()

    return _rows_to_dicts(rows)


def get_price_for_duration(
    duration_hours: float,
    space_type: str = "standard",
    db_path: Path | None = None,
) -> dict | None:
    """
    Return the applicable price tier for a given parking duration and space type.

    Selects the cheapest tier whose ``max_duration_h`` is >= *duration_hours*,
    or the daily-maximum tier (NULL max_duration_h) if none matches.

    Args:
        duration_hours: Intended parking duration in hours.
        space_type:     Space type (default: standard).

    Returns:
        Single pricing dict, or None if no price data exists for the type.
    """
    with _connection(db_path) as conn:
        # First try to find a tier that covers the duration
        row = conn.execute(
            """
            SELECT * FROM parking_prices
            WHERE space_type = ?
              AND max_duration_h IS NOT NULL
              AND max_duration_h >= ?
            ORDER BY max_duration_h ASC
            LIMIT 1
            """,
            (space_type.lower(), duration_hours),
        ).fetchone()

        if row is None:
            # Fall back to the daily maximum tier
            row = conn.execute(
                """
                SELECT * FROM parking_prices
                WHERE space_type = ?
                  AND max_duration_h IS NULL
                LIMIT 1
                """,
                (space_type.lower(),),
            ).fetchone()

    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Working hours queries
# ---------------------------------------------------------------------------

def get_working_hours(
    day_type: str | None = None,
    db_path: Path | None = None,
) -> list[dict]:
    """
    Return working hours records, optionally filtered by *day_type*.

    Args:
        day_type: One of weekday | saturday | sunday | bank_holiday.
                  If None, all records are returned.

    Returns:
        List of dicts with keys: id, day_type, is_24_hours, open_time,
        close_time, staffed_start, staffed_end, notes.
    """
    with _connection(db_path) as conn:
        if day_type:
            rows = conn.execute(
                "SELECT * FROM working_hours WHERE day_type = ?",
                (day_type.lower(),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM working_hours ORDER BY id"
            ).fetchall()

    return _rows_to_dicts(rows)


def get_staffed_hours_summary(db_path: Path | None = None) -> list[dict]:
    """
    Return a concise summary of staffed hours across all day types.

    Returns:
        List of dicts: day_type, staffed_start, staffed_end, is_24_hours.
    """
    with _connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT day_type, is_24_hours, staffed_start, staffed_end
            FROM working_hours
            ORDER BY id
            """
        ).fetchall()

    return _rows_to_dicts(rows)


# ---------------------------------------------------------------------------
# Availability queries
# ---------------------------------------------------------------------------

def get_availability(
    space_type: str | None = None,
    floor_number: int | None = None,
    db_path: Path | None = None,
) -> list[dict]:
    """
    Return parking availability records.

    Args:
        space_type:   Filter by space type (optional).
        floor_number: Filter by floor number (optional).

    Returns:
        List of dicts with keys: id, floor_number, zone, space_type,
        total_spaces, available_spaces, last_updated.
        Ordered by floor_number, zone.
    """
    conditions = []
    params: list = []

    if space_type:
        conditions.append("space_type = ?")
        params.append(space_type.lower())
    if floor_number is not None:
        conditions.append("floor_number = ?")
        params.append(floor_number)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with _connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM parking_availability {where} ORDER BY floor_number, zone",
            params,
        ).fetchall()

    return _rows_to_dicts(rows)


def get_availability_summary(db_path: Path | None = None) -> list[dict]:
    """
    Return total and available space counts grouped by space type.

    Returns:
        List of dicts: space_type, total_spaces, available_spaces,
        occupancy_pct.
    """
    with _connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                space_type,
                SUM(total_spaces)     AS total_spaces,
                SUM(available_spaces) AS available_spaces,
                ROUND(
                    100.0 * (SUM(total_spaces) - SUM(available_spaces))
                    / NULLIF(SUM(total_spaces), 0),
                    1
                ) AS occupancy_pct
            FROM parking_availability
            GROUP BY space_type
            ORDER BY space_type
            """
        ).fetchall()

    return _rows_to_dicts(rows)


def get_total_available_spaces(db_path: Path | None = None) -> int:
    """Return the total number of available spaces across all types."""
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT SUM(available_spaces) AS total FROM parking_availability"
        ).fetchone()

    return int(row["total"] or 0)


# ---------------------------------------------------------------------------
# Vehicle capacity queries (live — computed from reservations)
# ---------------------------------------------------------------------------

def get_available_vehicle_types(db_path: Path | None = None) -> list[dict]:
    """
    Return each vehicle type with its total and live available space count.

    Availability is computed exclusively from active reservations
    (status IN ('pending', 'confirmed')).  No hardcoded values are used.

    Returns:
        List of dicts: vehicle_type, display_name, total_spaces,
        booked_count, available_spaces.  Ordered by display_name.
        Only returns types where total_spaces > 0.
    """
    with _connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                vc.vehicle_type,
                vc.display_name,
                vc.total_spaces,
                COALESCE(
                    (SELECT COUNT(*) FROM reservations r
                     WHERE r.vehicle_type = vc.vehicle_type
                       AND r.status IN ('pending','confirmed')),
                    0
                ) AS booked_count,
                vc.total_spaces - COALESCE(
                    (SELECT COUNT(*) FROM reservations r
                     WHERE r.vehicle_type = vc.vehicle_type
                       AND r.status IN ('pending','confirmed')),
                    0
                ) AS available_spaces
            FROM vehicle_capacity vc
            ORDER BY vc.display_name
            """
        ).fetchall()
    return _rows_to_dicts(rows)


# ---------------------------------------------------------------------------
# Reservation CRUD
# ---------------------------------------------------------------------------

def save_reservation(
    first_name: str,
    surname: str,
    vehicle_type: str,
    vehicle_reg: str,
    start_datetime: str,
    end_datetime: str,
    db_path: Path | None = None,
) -> dict:
    """
    Persist a new reservation atomically.

    Uses BEGIN IMMEDIATE to acquire a write lock for the entire
    check-then-insert sequence, preventing concurrent overbooking.

    Returns:
        dict with keys:
          success (bool)
          reservation_id (int | None)
          error (str | None) — one of:
            'invalid_vehicle_type', 'capacity_exceeded',
            'duplicate_reservation', 'db_error'
          available_spaces (int) — remaining spaces after this booking
                                   (only set on success or capacity_exceeded)
    """
    path = db_path or settings.db_path
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        # BEGIN IMMEDIATE: no other writer can start until this txn ends.
        conn.execute("BEGIN IMMEDIATE")

        # 1. Verify vehicle type exists.
        vc_row = conn.execute(
            "SELECT total_spaces FROM vehicle_capacity WHERE vehicle_type = ?",
            (vehicle_type,),
        ).fetchone()
        if vc_row is None:
            conn.rollback()
            return {"success": False, "error": "invalid_vehicle_type",
                    "reservation_id": None}

        total = vc_row["total_spaces"]

        # 2. Count active reservations for this type.
        booked = conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE vehicle_type = ?"
            " AND status IN ('pending','confirmed')",
            (vehicle_type,),
        ).fetchone()[0]
        available = total - booked

        if available <= 0:
            conn.rollback()
            return {"success": False, "error": "capacity_exceeded",
                    "reservation_id": None, "available_spaces": 0}

        # 3. Check for overlapping reservation on the same vehicle registration.
        #    Overlap: existing.start < new.end  AND  existing.end > new.start
        overlap = conn.execute(
            """
            SELECT id FROM reservations
            WHERE vehicle_reg = ?
              AND status IN ('pending','confirmed')
              AND start_datetime < ?
              AND end_datetime   > ?
            LIMIT 1
            """,
            (vehicle_reg.upper(), end_datetime, start_datetime),
        ).fetchone()
        if overlap:
            conn.rollback()
            return {"success": False, "error": "duplicate_reservation",
                    "reservation_id": None}

        # 4. Insert — all checks passed.
        cursor = conn.execute(
            """
            INSERT INTO reservations
                (vehicle_type, vehicle_reg, first_name, surname,
                 start_datetime, end_datetime, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """,
            (vehicle_type, vehicle_reg.upper(),
             first_name, surname, start_datetime, end_datetime),
        )
        reservation_id = cursor.lastrowid
        conn.commit()

        logger.info(
            "Reservation saved: id=%d type=%s reg=%s %s→%s",
            reservation_id, vehicle_type, vehicle_reg, start_datetime, end_datetime,
        )
        return {
            "success": True,
            "reservation_id": reservation_id,
            "error": None,
            "available_spaces": available - 1,
        }

    except sqlite3.Error:
        conn.rollback()
        logger.exception("save_reservation DB error")
        return {"success": False, "error": "db_error", "reservation_id": None}
    finally:
        conn.close()


def cancel_reservation(
    reservation_id: int,
    db_path: Path | None = None,
) -> bool:
    """
    Cancel a reservation by setting its status to 'cancelled'.

    Returns True if a row was updated, False otherwise.
    """
    with _connection(db_path) as conn:
        cursor = conn.execute(
            "UPDATE reservations SET status='cancelled'"
            " WHERE id=? AND status IN ('pending','confirmed')",
            (reservation_id,),
        )
        conn.commit()  # type: ignore[attr-defined]
        updated = cursor.rowcount > 0

    if updated:
        logger.info("Reservation %d cancelled.", reservation_id)
    return updated


def get_reservation_by_id(
    reservation_id: int,
    db_path: Path | None = None,
) -> dict | None:
    """Return a single reservation dict, or None if not found."""
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
        ).fetchone()
    return dict(row) if row else None


def get_pending_exports(db_path: Path | None = None) -> list[dict]:
    """
    Return confirmed reservations whose export has not yet completed.

    Includes both NULL (never attempted) and 'EXPORTING' (in progress or
    crashed before DB commit).  Used by admin_app.py to show the Retry
    Export button.
    """
    with _connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM reservations
            WHERE status = 'confirmed'
              AND (exported_at IS NULL OR exported_at = 'EXPORTING')
            ORDER BY created_at DESC
            """
        ).fetchall()
    return _rows_to_dicts(rows)


def get_reservations_for_reg(
    vehicle_reg: str,
    db_path: Path | None = None,
) -> list[dict]:
    """Return all reservations (any status) for a given vehicle registration."""
    with _connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM reservations WHERE vehicle_reg = ? ORDER BY start_datetime",
            (vehicle_reg.upper(),),
        ).fetchall()
    return _rows_to_dicts(rows)


# ---------------------------------------------------------------------------
# Admin query functions (Stage 2)
# ---------------------------------------------------------------------------

def get_pending_reservations(db_path: Path | None = None) -> list[dict]:
    """
    Return all reservations with status='pending', newest first.

    Used by the admin dashboard to list items awaiting a decision.
    """
    with _connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM reservations WHERE status = 'pending'"
            " ORDER BY created_at DESC"
        ).fetchall()
    return _rows_to_dicts(rows)


def get_all_reservations(
    status: str | None = None,
    db_path: Path | None = None,
) -> list[dict]:
    """
    Return reservations, optionally filtered by *status*.

    Args:
        status: One of pending | confirmed | cancelled | rejected | expired.
                If None, all reservations are returned.

    Returns:
        List of reservation dicts ordered by created_at descending.
    """
    with _connection(db_path) as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM reservations WHERE status = ? ORDER BY created_at DESC",
                (status.lower(),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM reservations ORDER BY created_at DESC"
            ).fetchall()
    return _rows_to_dicts(rows)


def get_reservation_status(
    reservation_id: int,
    db_path: Path | None = None,
) -> dict | None:
    """
    Return status fields for a single reservation, or None if not found.

    Returns only the fields needed for user-side polling:
    id, status, admin_notes, decided_at, decided_by.
    """
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, status, admin_notes, decided_at, decided_by"
            " FROM reservations WHERE id = ?",
            (reservation_id,),
        ).fetchone()
    return dict(row) if row else None


def approve_reservation(
    reservation_id: int,
    admin_id: str,
    notes: str = "",
    db_path: Path | None = None,
) -> dict:
    """
    Approve a pending reservation atomically.

    Steps (inside BEGIN IMMEDIATE):
      1. Verify the reservation exists and is still 'pending'.
      2. Re-check that the vehicle type still has capacity
         (guards against edge cases where capacity changed after save).
      3. Update status → 'confirmed'; record admin_id, notes, timestamp.

    Returns:
        dict with keys: success (bool), error (str | None).
        error values: 'not_found', 'already_decided', 'capacity_exceeded',
                      'db_error'.
    """
    path = db_path or settings.db_path
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")

        # 1. Fetch the reservation.
        row = conn.execute(
            "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
        ).fetchone()

        if row is None:
            conn.rollback()
            return {"success": False, "error": "not_found"}

        if row["status"] != "pending":
            conn.rollback()
            return {"success": False, "error": "already_decided",
                    "current_status": row["status"]}

        vehicle_type = row["vehicle_type"]

        # 2. Re-verify capacity.
        vc = conn.execute(
            "SELECT total_spaces FROM vehicle_capacity WHERE vehicle_type = ?",
            (vehicle_type,),
        ).fetchone()

        if vc is None:
            conn.rollback()
            return {"success": False, "error": "invalid_vehicle_type"}

        booked = conn.execute(
            "SELECT COUNT(*) FROM reservations"
            " WHERE vehicle_type = ? AND status IN ('pending','confirmed')",
            (vehicle_type,),
        ).fetchone()[0]

        # The reservation being approved is already in the booked count as
        # 'pending'.  If booked > total_spaces something went wrong externally.
        if booked > vc["total_spaces"]:
            conn.rollback()
            return {"success": False, "error": "capacity_exceeded"}

        # 3. Confirm.
        conn.execute(
            """
            UPDATE reservations
            SET status     = 'confirmed',
                admin_notes = ?,
                decided_at  = strftime('%Y-%m-%d %H:%M:%S','now'),
                decided_by  = ?
            WHERE id = ?
            """,
            (notes, admin_id, reservation_id),
        )
        conn.commit()

        logger.info(
            "Reservation %d approved by %s.", reservation_id, admin_id
        )
        return {"success": True, "error": None}

    except sqlite3.Error:
        conn.rollback()
        logger.exception("approve_reservation DB error for id=%d", reservation_id)
        return {"success": False, "error": "db_error"}
    finally:
        conn.close()


def reject_reservation(
    reservation_id: int,
    admin_id: str,
    reason: str,
    db_path: Path | None = None,
) -> dict:
    """
    Reject a pending reservation atomically.

    Steps (inside BEGIN IMMEDIATE):
      1. Verify the reservation exists and is still 'pending'.
      2. Update status → 'rejected'; record admin_id, reason, timestamp.

    Returns:
        dict with keys: success (bool), error (str | None).
        error values: 'not_found', 'already_decided', 'db_error'.
    """
    if not reason or not reason.strip():
        return {"success": False, "error": "reason_required"}

    path = db_path or settings.db_path
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute(
            "SELECT id, status FROM reservations WHERE id = ?",
            (reservation_id,),
        ).fetchone()

        if row is None:
            conn.rollback()
            return {"success": False, "error": "not_found"}

        if row["status"] != "pending":
            conn.rollback()
            return {"success": False, "error": "already_decided",
                    "current_status": row["status"]}

        conn.execute(
            """
            UPDATE reservations
            SET status      = 'rejected',
                admin_notes = ?,
                decided_at  = strftime('%Y-%m-%d %H:%M:%S','now'),
                decided_by  = ?
            WHERE id = ?
            """,
            (reason.strip(), admin_id, reservation_id),
        )
        conn.commit()

        logger.info(
            "Reservation %d rejected by %s. Reason: %s",
            reservation_id, admin_id, reason,
        )
        return {"success": True, "error": None}

    except sqlite3.Error:
        conn.rollback()
        logger.exception("reject_reservation DB error for id=%d", reservation_id)
        return {"success": False, "error": "db_error"}
    finally:
        conn.close()
