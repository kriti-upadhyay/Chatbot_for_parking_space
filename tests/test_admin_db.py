"""
Tests for Stage 2 admin database query functions.

Covers: migration, pending list, approve, reject, duplicate-decision
prevention, capacity recheck, and status polling.

All tests use an isolated temporary SQLite file.
"""

import sqlite3
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    from src.database.schema import initialise_database
    path = tmp_path / "test_admin.db"
    initialise_database(db_path=path)
    return path


def _save(db_path, *, reg="AB12CDE", vtype="standard",
          start="2090-06-15 09:00", end="2090-06-15 17:00",
          first="Alice", surname="Smith"):
    from src.database.queries import save_reservation
    return save_reservation(
        first_name=first, surname=surname,
        vehicle_type=vtype, vehicle_reg=reg,
        start_datetime=start, end_datetime=end,
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------

class TestMigration:
    def test_new_database_has_admin_columns(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        cols = {row[1] for row in conn.execute("PRAGMA table_info(reservations)")}
        conn.close()
        assert "admin_notes" in cols
        assert "decided_at"  in cols
        assert "decided_by"  in cols

    def test_migration_is_idempotent(self, db_path: Path):
        """Running initialise_database twice must not fail or duplicate columns."""
        from src.database.schema import initialise_database
        initialise_database(db_path=db_path)   # second call
        conn = sqlite3.connect(str(db_path))
        count = sum(
            1 for row in conn.execute("PRAGMA table_info(reservations)")
            if row[1] in ("admin_notes", "decided_at", "decided_by")
        )
        conn.close()
        assert count == 3

    def test_migration_preserves_existing_reservations(self, db_path: Path):
        """Existing Stage 1 reservations must survive the migration unchanged."""
        result = _save(db_path)
        assert result["success"] is True
        rid = result["reservation_id"]

        # Simulate an old database that already has the reservation but
        # not the new columns by creating a fresh DB, adding a reservation,
        # removing the new columns (not possible in SQLite, so we verify
        # that the row is still there after a second init call).
        from src.database.schema import initialise_database
        initialise_database(db_path=db_path)

        from src.database.queries import get_reservation_by_id
        res = get_reservation_by_id(rid, db_path=db_path)
        assert res is not None
        assert res["first_name"] == "Alice"
        assert res["status"] == "pending"
        # New columns must be NULL (not filled by migration for old rows)
        assert res["admin_notes"] is None
        assert res["decided_at"]  is None
        assert res["decided_by"]  is None


# ---------------------------------------------------------------------------
# get_pending_reservations tests
# ---------------------------------------------------------------------------

class TestGetPendingReservations:
    def test_empty_when_no_reservations(self, db_path: Path):
        from src.database.queries import get_pending_reservations
        assert get_pending_reservations(db_path=db_path) == []

    def test_returns_pending_reservations(self, db_path: Path):
        from src.database.queries import get_pending_reservations
        _save(db_path, reg="AA11AAA")
        _save(db_path, reg="BB22BBB",
              start="2090-06-16 09:00", end="2090-06-16 17:00")
        rows = get_pending_reservations(db_path=db_path)
        assert len(rows) == 2
        assert all(r["status"] == "pending" for r in rows)

    def test_does_not_include_confirmed_or_rejected(self, db_path: Path):
        from src.database.queries import (
            approve_reservation, get_pending_reservations,
            reject_reservation,
        )
        r1 = _save(db_path, reg="AA11AAA")
        r2 = _save(db_path, reg="BB22BBB",
                   start="2090-06-16 09:00", end="2090-06-16 17:00")
        r3 = _save(db_path, reg="CC33CCC",
                   start="2090-06-17 09:00", end="2090-06-17 17:00")

        approve_reservation(r1["reservation_id"], "admin1", db_path=db_path)
        reject_reservation(r2["reservation_id"], "admin1", "No space", db_path=db_path)

        pending = get_pending_reservations(db_path=db_path)
        assert len(pending) == 1
        assert pending[0]["id"] == r3["reservation_id"]

    def test_row_has_expected_keys(self, db_path: Path):
        from src.database.queries import get_pending_reservations
        _save(db_path)
        row = get_pending_reservations(db_path=db_path)[0]
        for key in ("id", "vehicle_type", "vehicle_reg", "first_name",
                    "surname", "start_datetime", "end_datetime",
                    "status", "created_at"):
            assert key in row


# ---------------------------------------------------------------------------
# get_all_reservations tests
# ---------------------------------------------------------------------------

class TestGetAllReservations:
    def test_no_filter_returns_all(self, db_path: Path):
        from src.database.queries import approve_reservation, get_all_reservations
        r1 = _save(db_path, reg="AA11AAA")
        r2 = _save(db_path, reg="BB22BBB",
                   start="2090-06-16 09:00", end="2090-06-16 17:00")
        approve_reservation(r1["reservation_id"], "admin1", db_path=db_path)
        rows = get_all_reservations(db_path=db_path)
        assert len(rows) == 2

    def test_filter_by_confirmed(self, db_path: Path):
        from src.database.queries import approve_reservation, get_all_reservations
        r1 = _save(db_path, reg="AA11AAA")
        _save(db_path, reg="BB22BBB",
              start="2090-06-16 09:00", end="2090-06-16 17:00")
        approve_reservation(r1["reservation_id"], "admin1", db_path=db_path)
        rows = get_all_reservations(status="confirmed", db_path=db_path)
        assert len(rows) == 1
        assert rows[0]["status"] == "confirmed"

    def test_unknown_status_returns_empty(self, db_path: Path):
        from src.database.queries import get_all_reservations
        _save(db_path)
        rows = get_all_reservations(status="spaceship", db_path=db_path)
        assert rows == []


# ---------------------------------------------------------------------------
# get_reservation_status tests
# ---------------------------------------------------------------------------

class TestGetReservationStatus:
    def test_returns_status_fields(self, db_path: Path):
        from src.database.queries import get_reservation_status
        r = _save(db_path)
        status = get_reservation_status(r["reservation_id"], db_path=db_path)
        assert status is not None
        assert status["status"] == "pending"
        assert "admin_notes" in status
        assert "decided_at" in status
        assert "decided_by" in status

    def test_returns_none_for_missing_id(self, db_path: Path):
        from src.database.queries import get_reservation_status
        assert get_reservation_status(99999, db_path=db_path) is None

    def test_shows_decided_fields_after_approval(self, db_path: Path):
        from src.database.queries import approve_reservation, get_reservation_status
        r = _save(db_path)
        approve_reservation(r["reservation_id"], "admin_a", notes="All good", db_path=db_path)
        status = get_reservation_status(r["reservation_id"], db_path=db_path)
        assert status["status"] == "confirmed"
        assert status["admin_notes"] == "All good"
        assert status["decided_by"] == "admin_a"
        assert status["decided_at"] is not None


# ---------------------------------------------------------------------------
# approve_reservation tests
# ---------------------------------------------------------------------------

class TestApproveReservation:
    def test_successful_approval(self, db_path: Path):
        from src.database.queries import approve_reservation
        r = _save(db_path)
        result = approve_reservation(r["reservation_id"], "admin1", "Looks good",
                                     db_path=db_path)
        assert result["success"] is True
        assert result["error"] is None

    def test_approval_updates_status_to_confirmed(self, db_path: Path):
        from src.database.queries import approve_reservation, get_reservation_by_id
        r = _save(db_path)
        approve_reservation(r["reservation_id"], "admin1", db_path=db_path)
        res = get_reservation_by_id(r["reservation_id"], db_path=db_path)
        assert res["status"] == "confirmed"

    def test_approval_records_admin_fields(self, db_path: Path):
        from src.database.queries import approve_reservation, get_reservation_by_id
        r = _save(db_path)
        approve_reservation(r["reservation_id"], "admin_ops", notes="Priority customer",
                            db_path=db_path)
        res = get_reservation_by_id(r["reservation_id"], db_path=db_path)
        assert res["decided_by"] == "admin_ops"
        assert res["admin_notes"] == "Priority customer"
        assert res["decided_at"] is not None

    def test_approval_without_notes_succeeds(self, db_path: Path):
        from src.database.queries import approve_reservation
        r = _save(db_path)
        result = approve_reservation(r["reservation_id"], "admin1", db_path=db_path)
        assert result["success"] is True

    def test_approve_nonexistent_returns_not_found(self, db_path: Path):
        from src.database.queries import approve_reservation
        result = approve_reservation(99999, "admin1", db_path=db_path)
        assert result["success"] is False
        assert result["error"] == "not_found"

    def test_approved_reservation_still_counts_toward_occupancy(self, db_path: Path):
        """Confirming a reservation must not create a second occupancy count."""
        from src.database.queries import approve_reservation, get_available_vehicle_types
        before = next(t for t in get_available_vehicle_types(db_path=db_path)
                      if t["vehicle_type"] == "standard")["available_spaces"]
        r = _save(db_path)
        after_save = next(t for t in get_available_vehicle_types(db_path=db_path)
                          if t["vehicle_type"] == "standard")["available_spaces"]
        approve_reservation(r["reservation_id"], "admin1", db_path=db_path)
        after_approve = next(t for t in get_available_vehicle_types(db_path=db_path)
                             if t["vehicle_type"] == "standard")["available_spaces"]
        assert after_save == before - 1        # save consumed one space
        assert after_approve == after_save     # approve did NOT consume another


# ---------------------------------------------------------------------------
# Duplicate decision prevention
# ---------------------------------------------------------------------------

class TestDuplicateDecision:
    def test_cannot_approve_already_confirmed(self, db_path: Path):
        from src.database.queries import approve_reservation
        r = _save(db_path)
        approve_reservation(r["reservation_id"], "admin1", db_path=db_path)
        result = approve_reservation(r["reservation_id"], "admin2", db_path=db_path)
        assert result["success"] is False
        assert result["error"] == "already_decided"
        assert result["current_status"] == "confirmed"

    def test_cannot_approve_already_rejected(self, db_path: Path):
        from src.database.queries import approve_reservation, reject_reservation
        r = _save(db_path)
        reject_reservation(r["reservation_id"], "admin1", "No space", db_path=db_path)
        result = approve_reservation(r["reservation_id"], "admin1", db_path=db_path)
        assert result["success"] is False
        assert result["error"] == "already_decided"

    def test_cannot_reject_already_confirmed(self, db_path: Path):
        from src.database.queries import approve_reservation, reject_reservation
        r = _save(db_path)
        approve_reservation(r["reservation_id"], "admin1", db_path=db_path)
        result = reject_reservation(r["reservation_id"], "admin1", "Oops",
                                    db_path=db_path)
        assert result["success"] is False
        assert result["error"] == "already_decided"

    def test_concurrent_approvals_only_one_succeeds(self, db_path: Path):
        """Simulate two attempts to approve the same reservation back-to-back."""
        from src.database.queries import approve_reservation
        r = _save(db_path)
        res1 = approve_reservation(r["reservation_id"], "admin1", db_path=db_path)
        res2 = approve_reservation(r["reservation_id"], "admin2", db_path=db_path)
        successes = [res1["success"], res2["success"]]
        assert successes.count(True) == 1
        assert successes.count(False) == 1


# ---------------------------------------------------------------------------
# reject_reservation tests
# ---------------------------------------------------------------------------

class TestRejectReservation:
    def test_successful_rejection(self, db_path: Path):
        from src.database.queries import reject_reservation
        r = _save(db_path)
        result = reject_reservation(r["reservation_id"], "admin1",
                                    "Outside operating hours", db_path=db_path)
        assert result["success"] is True

    def test_rejection_updates_status_to_rejected(self, db_path: Path):
        from src.database.queries import get_reservation_by_id, reject_reservation
        r = _save(db_path)
        reject_reservation(r["reservation_id"], "admin1", "No capacity",
                           db_path=db_path)
        res = get_reservation_by_id(r["reservation_id"], db_path=db_path)
        assert res["status"] == "rejected"
        assert res["admin_notes"] == "No capacity"

    def test_rejection_releases_capacity(self, db_path: Path):
        """Rejected reservation must free its held space."""
        from src.database.queries import get_available_vehicle_types, reject_reservation
        before = next(t for t in get_available_vehicle_types(db_path=db_path)
                      if t["vehicle_type"] == "standard")["available_spaces"]
        r = _save(db_path)
        after_save = next(t for t in get_available_vehicle_types(db_path=db_path)
                          if t["vehicle_type"] == "standard")["available_spaces"]
        reject_reservation(r["reservation_id"], "admin1", "Full", db_path=db_path)
        after_reject = next(t for t in get_available_vehicle_types(db_path=db_path)
                            if t["vehicle_type"] == "standard")["available_spaces"]
        assert after_save == before - 1
        assert after_reject == before    # space returned

    def test_reject_requires_reason(self, db_path: Path):
        from src.database.queries import reject_reservation
        r = _save(db_path)
        result = reject_reservation(r["reservation_id"], "admin1", "",
                                    db_path=db_path)
        assert result["success"] is False
        assert result["error"] == "reason_required"

    def test_reject_nonexistent_returns_not_found(self, db_path: Path):
        from src.database.queries import reject_reservation
        result = reject_reservation(99999, "admin1", "Reason", db_path=db_path)
        assert result["success"] is False
        assert result["error"] == "not_found"


# ---------------------------------------------------------------------------
# Capacity recheck in approve
# ---------------------------------------------------------------------------

class TestCapacityRecheckOnApproval:
    def test_approve_still_works_when_capacity_not_exceeded(self, db_path: Path):
        """Normal case: pending reservation holds space, approval is allowed."""
        from src.database.queries import approve_reservation
        r = _save(db_path)
        result = approve_reservation(r["reservation_id"], "admin1", db_path=db_path)
        assert result["success"] is True

    def test_approve_fails_when_capacity_externally_reduced(self, db_path: Path):
        """
        If the vehicle_capacity total is reduced BELOW the current booked count
        after reservations were saved, the capacity recheck blocks approval.

        We save 2 motorcycle reservations (non-overlapping times so both pass
        the duplicate check), then reduce capacity to 1.  With booked=2 and
        total=1, approving either reservation must fail.
        """
        import sqlite3 as _sq
        from src.database.queries import approve_reservation

        r1 = _save(db_path, vtype="motorcycle", reg="MC0001",
                   start="2090-07-01 09:00", end="2090-07-01 12:00")
        r2 = _save(db_path, vtype="motorcycle", reg="MC0002",
                   start="2090-07-01 13:00", end="2090-07-01 17:00")
        assert r1["success"] and r2["success"]

        # Reduce motorcycle capacity to 1 (still satisfies CHECK > 0)
        # Now booked (2) > total_spaces (1) → capacity recheck must block
        conn = _sq.connect(str(db_path))
        conn.execute(
            "UPDATE vehicle_capacity SET total_spaces = 1"
            " WHERE vehicle_type = 'motorcycle'"
        )
        conn.commit()
        conn.close()

        result = approve_reservation(r1["reservation_id"], "admin1", db_path=db_path)
        assert result["success"] is False
        assert result["error"] == "capacity_exceeded"
