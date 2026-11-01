"""
Tests for src/database — schema creation, seed data, and query functions.

All tests use a fresh temporary SQLite file so the real database is never
touched and tests are fully isolated from each other.
"""

from pathlib import Path

import pytest


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Create and return a seeded temporary database path."""
    from src.database.schema import initialise_database

    path = tmp_path / "test_parking.db"
    initialise_database(db_path=path)
    return path


# ---------------------------------------------------------------------------
# Schema / initialisation tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_tables_are_created(self, db_path: Path):
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()

        assert "parking_prices" in tables
        assert "working_hours" in tables
        assert "parking_availability" in tables
        assert "vehicle_capacity" in tables, "vehicle_capacity table must exist"
        assert "reservations" in tables, "reservations table must exist"

    def test_seed_data_is_present(self, db_path: Path):
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        price_count = conn.execute("SELECT COUNT(*) FROM parking_prices").fetchone()[0]
        hours_count = conn.execute("SELECT COUNT(*) FROM working_hours").fetchone()[0]
        avail_count = conn.execute("SELECT COUNT(*) FROM parking_availability").fetchone()[0]
        conn.close()

        assert price_count > 0, "parking_prices should have seed rows"
        assert hours_count == 4, "working_hours should have 4 day-type rows"
        assert avail_count > 0, "parking_availability should have seed rows"

    def test_initialise_is_idempotent(self, db_path: Path):
        """Calling initialise_database twice should not duplicate rows."""
        import sqlite3
        from src.database.schema import initialise_database

        initialise_database(db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        count_after = conn.execute("SELECT COUNT(*) FROM parking_prices").fetchone()[0]
        conn.close()

        from src.database.schema import _SEED_PRICES
        assert count_after == len(_SEED_PRICES)


# ---------------------------------------------------------------------------
# Pricing query tests
# ---------------------------------------------------------------------------

class TestPricingQueries:
    def test_get_all_prices_returns_records(self, db_path: Path):
        from src.database.queries import get_prices

        rows = get_prices(db_path=db_path)
        assert len(rows) > 0

    def test_get_prices_filtered_by_type(self, db_path: Path):
        from src.database.queries import get_prices

        rows = get_prices(space_type="motorcycle", db_path=db_path)
        assert len(rows) > 0
        assert all(r["space_type"] == "motorcycle" for r in rows)

    def test_get_prices_unknown_type_returns_empty(self, db_path: Path):
        from src.database.queries import get_prices

        rows = get_prices(space_type="spaceship", db_path=db_path)
        assert rows == []

    def test_price_for_short_duration(self, db_path: Path):
        from src.database.queries import get_price_for_duration

        result = get_price_for_duration(0.5, space_type="standard", db_path=db_path)
        assert result is not None
        assert result["space_type"] == "standard"
        assert result["max_duration_h"] == 1.0
        assert result["price_gbp"] == 2.50

    def test_price_for_long_duration_hits_daily_max(self, db_path: Path):
        from src.database.queries import get_price_for_duration

        result = get_price_for_duration(10, space_type="standard", db_path=db_path)
        assert result is not None
        assert result["max_duration_h"] is None   # NULL = daily max tier
        assert result["price_gbp"] == 14.00

    def test_disabled_rate_is_cheaper_than_standard(self, db_path: Path):
        from src.database.queries import get_price_for_duration

        standard = get_price_for_duration(1.5, space_type="standard", db_path=db_path)
        disabled = get_price_for_duration(1.5, space_type="disabled", db_path=db_path)
        assert standard is not None and disabled is not None
        assert disabled["price_gbp"] < standard["price_gbp"]

    def test_price_row_has_expected_keys(self, db_path: Path):
        from src.database.queries import get_prices

        row = get_prices(space_type="standard", db_path=db_path)[0]
        for key in ("space_type", "tier_name", "max_duration_h", "price_gbp", "description"):
            assert key in row


# ---------------------------------------------------------------------------
# Working hours query tests
# ---------------------------------------------------------------------------

class TestWorkingHoursQueries:
    def test_get_all_hours_returns_four_records(self, db_path: Path):
        from src.database.queries import get_working_hours

        rows = get_working_hours(db_path=db_path)
        assert len(rows) == 4

    def test_get_hours_filtered_by_day_type(self, db_path: Path):
        from src.database.queries import get_working_hours

        rows = get_working_hours(day_type="weekday", db_path=db_path)
        assert len(rows) == 1
        assert rows[0]["day_type"] == "weekday"

    def test_facility_is_24_hours(self, db_path: Path):
        from src.database.queries import get_working_hours

        rows = get_working_hours(db_path=db_path)
        assert all(r["is_24_hours"] == 1 for r in rows)

    def test_staffed_hours_summary_has_correct_keys(self, db_path: Path):
        from src.database.queries import get_staffed_hours_summary

        rows = get_staffed_hours_summary(db_path=db_path)
        assert len(rows) == 4
        for row in rows:
            assert "day_type" in row
            assert "staffed_start" in row
            assert "staffed_end" in row

    def test_unknown_day_type_returns_empty(self, db_path: Path):
        from src.database.queries import get_working_hours

        rows = get_working_hours(day_type="martian", db_path=db_path)
        assert rows == []


# ---------------------------------------------------------------------------
# Availability query tests
# ---------------------------------------------------------------------------

class TestAvailabilityQueries:
    def test_get_all_availability_returns_records(self, db_path: Path):
        from src.database.queries import get_availability

        rows = get_availability(db_path=db_path)
        assert len(rows) > 0

    def test_filter_by_space_type(self, db_path: Path):
        from src.database.queries import get_availability

        rows = get_availability(space_type="ev_charging", db_path=db_path)
        assert len(rows) > 0
        assert all(r["space_type"] == "ev_charging" for r in rows)

    def test_filter_by_floor(self, db_path: Path):
        from src.database.queries import get_availability

        rows = get_availability(floor_number=0, db_path=db_path)
        assert len(rows) > 0
        assert all(r["floor_number"] == 0 for r in rows)

    def test_filter_by_type_and_floor(self, db_path: Path):
        from src.database.queries import get_availability

        rows = get_availability(space_type="standard", floor_number=1, db_path=db_path)
        assert len(rows) > 0
        assert all(r["space_type"] == "standard" and r["floor_number"] == 1 for r in rows)

    def test_availability_summary_groups_by_type(self, db_path: Path):
        from src.database.queries import get_availability_summary

        rows = get_availability_summary(db_path=db_path)
        types = {r["space_type"] for r in rows}
        assert "standard" in types
        assert "ev_charging" in types
        assert "motorcycle" in types

    def test_summary_available_never_exceeds_total(self, db_path: Path):
        from src.database.queries import get_availability_summary

        for row in get_availability_summary(db_path=db_path):
            assert row["available_spaces"] <= row["total_spaces"]

    def test_total_available_spaces_is_positive(self, db_path: Path):
        from src.database.queries import get_total_available_spaces

        total = get_total_available_spaces(db_path=db_path)
        assert total > 0

    def test_availability_row_has_expected_keys(self, db_path: Path):
        from src.database.queries import get_availability

        row = get_availability(db_path=db_path)[0]
        for key in ("floor_number", "zone", "space_type", "total_spaces", "available_spaces"):
            assert key in row

    def test_initial_available_spaces_equals_total(self, db_path: Path):
        """At init, available_spaces must equal total_spaces — no hardcoded partial occupancy."""
        from src.database.queries import get_availability

        for row in get_availability(db_path=db_path):
            assert row["available_spaces"] == row["total_spaces"], (
                f"Row floor={row['floor_number']} zone={row['zone']} "
                f"type={row['space_type']}: available={row['available_spaces']} "
                f"!= total={row['total_spaces']}"
            )


# ---------------------------------------------------------------------------
# Vehicle capacity tests
# ---------------------------------------------------------------------------

class TestVehicleCapacity:
    def test_vehicle_capacity_seeded(self, db_path: Path):
        from src.database.queries import get_available_vehicle_types
        rows = get_available_vehicle_types(db_path=db_path)
        assert len(rows) == 5
        types = {r["vehicle_type"] for r in rows}
        assert types == {"standard", "disabled", "ev_charging", "motorcycle", "oversized"}

    def test_initial_occupancy_is_zero(self, db_path: Path):
        """With no reservations, booked_count must be 0 for every type."""
        from src.database.queries import get_available_vehicle_types
        for row in get_available_vehicle_types(db_path=db_path):
            assert row["booked_count"] == 0, (
                f"{row['vehicle_type']} has non-zero initial occupancy"
            )

    def test_initial_available_equals_total(self, db_path: Path):
        """available_spaces must equal total_spaces when no reservations exist."""
        from src.database.queries import get_available_vehicle_types
        for row in get_available_vehicle_types(db_path=db_path):
            assert row["available_spaces"] == row["total_spaces"]

    def test_standard_has_720_spaces(self, db_path: Path):
        from src.database.queries import get_available_vehicle_types
        rows = {r["vehicle_type"]: r for r in get_available_vehicle_types(db_path=db_path)}
        assert rows["standard"]["total_spaces"] == 720

    def test_available_vehicle_types_row_has_required_keys(self, db_path: Path):
        from src.database.queries import get_available_vehicle_types
        row = get_available_vehicle_types(db_path=db_path)[0]
        for key in ("vehicle_type", "display_name", "total_spaces",
                    "booked_count", "available_spaces"):
            assert key in row


# ---------------------------------------------------------------------------
# Reservation CRUD tests
# ---------------------------------------------------------------------------

def _reservation_kwargs(db_path, **overrides):
    base = dict(
        first_name="Alice", surname="Smith",
        vehicle_type="standard", vehicle_reg="AB12CDE",
        start_datetime="2090-06-15 09:00",
        end_datetime="2090-06-15 17:00",
        db_path=db_path,
    )
    base.update(overrides)
    return base


class TestSaveReservation:
    def test_successful_save_returns_reservation_id(self, db_path: Path):
        from src.database.queries import save_reservation
        result = save_reservation(**_reservation_kwargs(db_path))
        assert result["success"] is True
        assert isinstance(result["reservation_id"], int)
        assert result["reservation_id"] > 0

    def test_booking_increments_booked_count_by_exactly_one(self, db_path: Path):
        from src.database.queries import get_available_vehicle_types, save_reservation
        before = next(r for r in get_available_vehicle_types(db_path=db_path)
                      if r["vehicle_type"] == "standard")
        save_reservation(**_reservation_kwargs(db_path))
        after = next(r for r in get_available_vehicle_types(db_path=db_path)
                     if r["vehicle_type"] == "standard")
        assert after["booked_count"] == before["booked_count"] + 1
        assert after["available_spaces"] == before["available_spaces"] - 1

    def test_same_booking_twice_is_rejected_as_duplicate(self, db_path: Path):
        from src.database.queries import save_reservation
        save_reservation(**_reservation_kwargs(db_path))
        result = save_reservation(**_reservation_kwargs(db_path))
        assert result["success"] is False
        assert result["error"] == "duplicate_reservation"

    def test_overlapping_booking_same_reg_rejected(self, db_path: Path):
        from src.database.queries import save_reservation
        save_reservation(**_reservation_kwargs(db_path,
            start_datetime="2090-06-15 09:00", end_datetime="2090-06-15 17:00"))
        # Overlapping window (starts inside first booking)
        result = save_reservation(**_reservation_kwargs(db_path,
            start_datetime="2090-06-15 12:00", end_datetime="2090-06-15 20:00"))
        assert result["success"] is False
        assert result["error"] == "duplicate_reservation"

    def test_non_overlapping_same_reg_allowed(self, db_path: Path):
        from src.database.queries import save_reservation
        save_reservation(**_reservation_kwargs(db_path,
            start_datetime="2090-06-15 09:00", end_datetime="2090-06-15 12:00"))
        result = save_reservation(**_reservation_kwargs(db_path,
            start_datetime="2090-06-15 12:00", end_datetime="2090-06-15 17:00"))
        assert result["success"] is True

    def test_invalid_vehicle_type_rejected(self, db_path: Path):
        from src.database.queries import save_reservation
        result = save_reservation(**_reservation_kwargs(db_path, vehicle_type="hovercraft"))
        assert result["success"] is False
        assert result["error"] == "invalid_vehicle_type"

    def test_capacity_exceeded_rejected(self, db_path: Path):
        """Fill all spaces for a type then try one more."""
        import sqlite3
        from src.database.queries import get_available_vehicle_types, save_reservation

        # Get motorcycle capacity (smallest = 20)
        cap = next(r for r in get_available_vehicle_types(db_path=db_path)
                   if r["vehicle_type"] == "motorcycle")["total_spaces"]

        # Fill every space with unique regs
        for i in range(cap):
            res = save_reservation(
                first_name="User", surname=str(i),
                vehicle_type="motorcycle",
                vehicle_reg=f"MC{i:04d}",
                start_datetime="2090-07-01 09:00",
                end_datetime="2090-07-01 17:00",
                db_path=db_path,
            )
            assert res["success"] is True, f"Booking {i} should succeed"

        # One more must fail
        overflow = save_reservation(
            first_name="Over", surname="Flow",
            vehicle_type="motorcycle", vehicle_reg="OVERFLOW",
            start_datetime="2090-07-01 09:00",
            end_datetime="2090-07-01 17:00",
            db_path=db_path,
        )
        assert overflow["success"] is False
        assert overflow["error"] == "capacity_exceeded"
        assert overflow.get("available_spaces") == 0

    def test_booking_does_not_double_count_on_retry(self, db_path: Path):
        """Calling save_reservation twice with same data must not count twice."""
        from src.database.queries import get_available_vehicle_types, save_reservation

        save_reservation(**_reservation_kwargs(db_path))
        save_reservation(**_reservation_kwargs(db_path))  # duplicate — rejected

        after = next(r for r in get_available_vehicle_types(db_path=db_path)
                     if r["vehicle_type"] == "standard")
        assert after["booked_count"] == 1  # still only 1, not 2


class TestCancelReservation:
    def test_cancel_returns_true_for_active_reservation(self, db_path: Path):
        from src.database.queries import cancel_reservation, save_reservation
        result = save_reservation(**_reservation_kwargs(db_path))
        assert cancel_reservation(result["reservation_id"], db_path=db_path) is True

    def test_cancellation_releases_capacity(self, db_path: Path):
        from src.database.queries import (
            cancel_reservation, get_available_vehicle_types, save_reservation,
        )
        before = next(r for r in get_available_vehicle_types(db_path=db_path)
                      if r["vehicle_type"] == "standard")["available_spaces"]
        result = save_reservation(**_reservation_kwargs(db_path))
        cancel_reservation(result["reservation_id"], db_path=db_path)
        after = next(r for r in get_available_vehicle_types(db_path=db_path)
                     if r["vehicle_type"] == "standard")["available_spaces"]
        assert after == before  # capacity restored

    def test_cancel_nonexistent_reservation_returns_false(self, db_path: Path):
        from src.database.queries import cancel_reservation
        assert cancel_reservation(99999, db_path=db_path) is False

    def test_cancel_already_cancelled_returns_false(self, db_path: Path):
        from src.database.queries import cancel_reservation, save_reservation
        result = save_reservation(**_reservation_kwargs(db_path))
        cancel_reservation(result["reservation_id"], db_path=db_path)
        # Second cancel on same ID should fail
        assert cancel_reservation(result["reservation_id"], db_path=db_path) is False


class TestGetReservation:
    def test_get_by_id_returns_correct_reservation(self, db_path: Path):
        from src.database.queries import get_reservation_by_id, save_reservation
        result = save_reservation(**_reservation_kwargs(db_path))
        reservation = get_reservation_by_id(result["reservation_id"], db_path=db_path)
        assert reservation is not None
        assert reservation["first_name"] == "Alice"
        assert reservation["vehicle_type"] == "standard"
        assert reservation["status"] == "pending"

    def test_get_nonexistent_returns_none(self, db_path: Path):
        from src.database.queries import get_reservation_by_id
        assert get_reservation_by_id(99999, db_path=db_path) is None

    def test_get_reservations_for_reg(self, db_path: Path):
        from src.database.queries import get_reservations_for_reg, save_reservation
        save_reservation(**_reservation_kwargs(db_path,
            start_datetime="2090-06-15 09:00", end_datetime="2090-06-15 12:00"))
        save_reservation(**_reservation_kwargs(db_path,
            start_datetime="2090-06-15 13:00", end_datetime="2090-06-15 17:00"))
        rows = get_reservations_for_reg("AB12CDE", db_path=db_path)
        assert len(rows) == 2
