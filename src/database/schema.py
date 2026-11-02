"""
SQLite schema creation and seed data.

Call ``initialise_database()`` once to create tables and populate seed data.
Safe to call multiple times — existing rows are preserved via INSERT OR IGNORE.

Tables:
  parking_prices      — tiered pricing by space type
  working_hours       — staffed hours by day type
  parking_availability — per-floor/zone total capacity (display only)
  vehicle_capacity    — total bookable spaces per vehicle type (authoritative)
  reservations        — every booking with status; occupancy derived from here
"""

import sqlite3
from pathlib import Path

from src.config import get_logger, settings

logger = get_logger(__name__)

# Active reservation statuses that count toward occupancy.
OCCUPANCY_STATUSES = ("pending", "confirmed")

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_PARKING_PRICES = """
CREATE TABLE IF NOT EXISTS parking_prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    space_type      TEXT    NOT NULL,
    tier_name       TEXT    NOT NULL,
    max_duration_h  REAL,
    price_gbp       REAL    NOT NULL,
    description     TEXT    NOT NULL,
    UNIQUE(space_type, tier_name)
);
"""

_CREATE_WORKING_HOURS = """
CREATE TABLE IF NOT EXISTS working_hours (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    day_type       TEXT    NOT NULL UNIQUE,
    is_24_hours    INTEGER NOT NULL DEFAULT 0,
    open_time      TEXT    NOT NULL,
    close_time     TEXT    NOT NULL,
    staffed_start  TEXT,
    staffed_end    TEXT,
    notes          TEXT
);
"""

_CREATE_PARKING_AVAILABILITY = """
CREATE TABLE IF NOT EXISTS parking_availability (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    floor_number     INTEGER NOT NULL,
    zone             TEXT    NOT NULL,
    space_type       TEXT    NOT NULL,
    total_spaces     INTEGER NOT NULL,
    available_spaces INTEGER NOT NULL,
    last_updated     TEXT    NOT NULL,
    UNIQUE(floor_number, zone, space_type)
);
"""

_CREATE_VEHICLE_CAPACITY = """
CREATE TABLE IF NOT EXISTS vehicle_capacity (
    vehicle_type    TEXT    PRIMARY KEY,
    total_spaces    INTEGER NOT NULL CHECK(total_spaces > 0),
    display_name    TEXT    NOT NULL
);
"""

_CREATE_RESERVATIONS = """
CREATE TABLE IF NOT EXISTS reservations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_type    TEXT    NOT NULL REFERENCES vehicle_capacity(vehicle_type),
    vehicle_reg     TEXT    NOT NULL,
    first_name      TEXT    NOT NULL,
    surname         TEXT    NOT NULL,
    start_datetime  TEXT    NOT NULL,
    end_datetime    TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending'
                            CHECK(status IN ('pending','confirmed','cancelled','rejected','expired')),
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now')),
    admin_notes     TEXT,
    decided_at      TEXT,
    decided_by      TEXT,
    exported_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_res_type_status
    ON reservations(vehicle_type, status);
CREATE INDEX IF NOT EXISTS idx_res_reg
    ON reservations(vehicle_reg);
"""

# Columns added after initial release — used by _migrate_reservations for
# existing databases.  Add new columns here; never remove old ones.
_MIGRATION_COLUMNS = [
    ("admin_notes", "TEXT"),
    ("decided_at",  "TEXT"),
    ("decided_by",  "TEXT"),
    ("exported_at", "TEXT"),   # Stage 3: 'EXPORTING' | ISO timestamp | NULL
]

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED_PRICES = [
    ("standard",    "up_to_1h",   1.0,  2.50, "Up to 1 hour"),
    ("standard",    "1h_to_2h",   2.0,  4.50, "1 to 2 hours"),
    ("standard",    "2h_to_3h",   3.0,  6.50, "2 to 3 hours"),
    ("standard",    "3h_to_4h",   4.0,  8.50, "3 to 4 hours"),
    ("standard",    "daily_max",  None, 14.00, "Over 4 hours (daily maximum)"),
    ("disabled",    "up_to_1h",   1.0,  1.50, "Blue Badge — up to 1 hour"),
    ("disabled",    "1h_to_2h",   2.0,  2.70, "Blue Badge — 1 to 2 hours"),
    ("disabled",    "2h_to_3h",   3.0,  3.90, "Blue Badge — 2 to 3 hours"),
    ("disabled",    "3h_to_4h",   4.0,  5.10, "Blue Badge — 3 to 4 hours"),
    ("disabled",    "daily_max",  None, 8.40, "Blue Badge — over 4 hours (daily maximum)"),
    ("ev_charging", "up_to_1h",   1.0,  2.50, "EV — up to 1 hour"),
    ("ev_charging", "1h_to_2h",   2.0,  4.50, "EV — 1 to 2 hours"),
    ("ev_charging", "2h_to_3h",   3.0,  6.50, "EV — 2 to 3 hours"),
    ("ev_charging", "3h_to_4h",   4.0,  8.50, "EV — 3 to 4 hours"),
    ("ev_charging", "daily_max",  None, 14.00, "EV — over 4 hours (daily maximum)"),
    ("motorcycle",  "up_to_1h",   1.0,  1.00, "Motorcycle — up to 1 hour"),
    ("motorcycle",  "1h_to_4h",   4.0,  2.50, "Motorcycle — 1 to 4 hours"),
    ("motorcycle",  "daily_max",  None, 5.00, "Motorcycle — over 4 hours (daily maximum)"),
    ("oversized",   "up_to_1h",   1.0,  4.00, "Oversized — up to 1 hour"),
    ("oversized",   "1h_to_2h",   2.0,  7.00, "Oversized — 1 to 2 hours"),
    ("oversized",   "2h_to_3h",   3.0, 10.00, "Oversized — 2 to 3 hours"),
    ("oversized",   "daily_max",  None, 18.00, "Oversized — over 3 hours (daily maximum)"),
]

_SEED_HOURS = [
    ("weekday",      1, "00:00", "23:59", "07:00", "23:00",
     "Monday–Friday.  Facility open 24 hours; security staff 07:00–23:00."),
    ("saturday",     1, "00:00", "23:59", "07:00", "23:00",
     "Saturday.  Facility open 24 hours; security staff 07:00–23:00."),
    ("sunday",       1, "00:00", "23:59", "08:00", "22:00",
     "Sunday.  Facility open 24 hours; security staff 08:00–22:00."),
    ("bank_holiday", 1, "00:00", "23:59", "08:00", "20:00",
     "Bank holidays.  Facility open 24 hours; reduced security staffing 08:00–20:00."),
]

# authoritative capacity per bookable vehicle type (total = 850 spaces)
_SEED_VEHICLE_CAPACITY = [
    ("standard",    720, "Standard"),
    ("disabled",     50, "Disabled (Blue Badge)"),
    ("ev_charging",  40, "EV Charging"),
    ("motorcycle",   20, "Motorcycle"),
    ("oversized",    20, "Oversized"),
]

# Per-floor/zone detail — available_spaces always equals total_spaces at init;
# occupancy is computed exclusively from the reservations table.
_SEED_AVAILABILITY = [
    # (floor, zone, space_type, total, last_updated)  — available = total
    (0, "A", "motorcycle",   20,  "2024-01-15 09:00:00"),
    (0, "B", "standard",    120,  "2024-01-15 09:00:00"),
    (0, "C", "standard",    100,  "2024-01-15 09:00:00"),
    (0, "D", "ev_charging",  16,  "2024-01-15 09:00:00"),
    (0, "E", "disabled",     25,  "2024-01-15 09:00:00"),
    (1, "A", "standard",    130,  "2024-01-15 09:00:00"),
    (1, "B", "standard",    130,  "2024-01-15 09:00:00"),
    (1, "C", "disabled",     25,  "2024-01-15 09:00:00"),
    (2, "A", "standard",    130,  "2024-01-15 09:00:00"),
    (2, "B", "ev_charging",  24,  "2024-01-15 09:00:00"),
    (2, "C", "standard",    120,  "2024-01-15 09:00:00"),
    (3, "A", "standard",    130,  "2024-01-15 09:00:00"),
    (3, "B", "standard",    130,  "2024-01-15 09:00:00"),
    (4, "A", "standard",    130,  "2024-01-15 09:00:00"),
    (4, "B", "standard",    130,  "2024-01-15 09:00:00"),
    (5, "A", "standard",    130,  "2024-01-15 09:00:00"),
    (5, "B", "standard",    130,  "2024-01-15 09:00:00"),
    (6, "A", "standard",     80,  "2024-01-15 09:00:00"),
    (6, "C", "oversized",    20,  "2024-01-15 09:00:00"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialise_database(db_path: Path | None = None) -> None:
    """
    Create all tables and populate seed data.  Safe to call multiple times.

    On every call, resets parking_availability.available_spaces = total_spaces
    so the display table never shows stale hardcoded values from older versions.
    Occupancy is always computed from the reservations table.
    """
    if db_path is None:
        db_path = settings.db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Initialising database at %s", db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        _create_tables(conn)
        _seed_data(conn)
        _reset_availability_to_total(conn)
        _migrate_reservations(conn)
        conn.commit()
        logger.info("Database initialised successfully.")
    except Exception:
        conn.rollback()
        logger.exception("Database initialisation failed.")
        raise
    finally:
        conn.close()


def _create_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(_CREATE_PARKING_PRICES)
    cursor.execute(_CREATE_WORKING_HOURS)
    cursor.execute(_CREATE_PARKING_AVAILABILITY)
    cursor.execute(_CREATE_VEHICLE_CAPACITY)
    # reservations DDL contains two statements; execute them separately
    for stmt in _CREATE_RESERVATIONS.strip().split(";\n"):
        stmt = stmt.strip()
        if stmt:
            cursor.execute(stmt)


def _seed_data(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    cursor.executemany(
        "INSERT OR IGNORE INTO parking_prices"
        " (space_type,tier_name,max_duration_h,price_gbp,description) VALUES(?,?,?,?,?)",
        _SEED_PRICES,
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO working_hours"
        " (day_type,is_24_hours,open_time,close_time,staffed_start,staffed_end,notes) VALUES(?,?,?,?,?,?,?)",
        _SEED_HOURS,
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO vehicle_capacity (vehicle_type,total_spaces,display_name) VALUES(?,?,?)",
        _SEED_VEHICLE_CAPACITY,
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO parking_availability"
        " (floor_number,zone,space_type,total_spaces,available_spaces,last_updated)"
        " VALUES(?,?,?,?,?,?)",
        # available_spaces = total_spaces (5th column duplicates 4th)
        [(f, z, st, tot, tot, ts) for f, z, st, tot, ts in _SEED_AVAILABILITY],
    )


def _reset_availability_to_total(conn: sqlite3.Connection) -> None:
    """Ensure available_spaces always reflects total_spaces (never hardcoded partials)."""
    conn.execute(
        "UPDATE parking_availability SET available_spaces = total_spaces"
        " WHERE available_spaces != total_spaces"
    )


def _migrate_reservations(conn: sqlite3.Connection) -> None:
    """
    Safe, idempotent migration for existing databases.

    Adds Stage 2 admin-decision columns to the reservations table if they
    are not already present.  Existing rows keep NULL for the new columns,
    which is correct — they were decided before Stage 2 was deployed.
    """
    existing_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(reservations)").fetchall()
    }
    for col_name, col_type in _MIGRATION_COLUMNS:
        if col_name not in existing_cols:
            conn.execute(
                f"ALTER TABLE reservations ADD COLUMN {col_name} {col_type}"
            )
            logger.info("Migration: added column '%s' to reservations.", col_name)
