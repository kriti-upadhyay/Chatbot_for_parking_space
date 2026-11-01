"""
Input validators for reservation form fields.

Each validator returns a ``ValidationResult``.  These are the single
source of truth for field validation — imported by the reservation node
and tested independently.
"""

import re
from dataclasses import dataclass
from datetime import datetime

from src.config import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: str | None = None
    normalized_value: str | None = None


# ---------------------------------------------------------------------------
# Individual field validators
# ---------------------------------------------------------------------------

def validate_name(value: str) -> ValidationResult:
    """Validate a first or last name."""
    v = value.strip()
    if not v:
        return ValidationResult(False, "Name cannot be empty.")
    if len(v) > 50:
        return ValidationResult(False, "Name must be 50 characters or fewer.")
    if not re.match(r"^[A-Za-z\s'\-]+$", v):
        return ValidationResult(
            False,
            "Please enter a valid name (letters, spaces, hyphens, and apostrophes only).",
        )
    return ValidationResult(True, normalized_value=v)


# ---------------------------------------------------------------------------
# UK vehicle registration
# ---------------------------------------------------------------------------

# Patterns ordered from most-specific to least-specific.
_UK_REG_PATTERNS = [
    # Post-2001: AA00 AAA (7 chars without space)
    re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$"),
    # 1983–2001 prefix: A000 AAA
    re.compile(r"^[A-Z][0-9]{1,3}[A-Z]{3}$"),
    # 1963–1983 suffix: AAA 000A
    re.compile(r"^[A-Z]{3}[0-9]{1,3}[A-Z]$"),
    # Diplomatic / custom: shorter patterns
    re.compile(r"^[A-Z0-9]{2,7}$"),
]


def validate_vehicle_reg(value: str) -> ValidationResult:
    """
    Validate a UK vehicle registration number.

    Normalises to uppercase with spaces removed before matching.
    Returns the registration in normalised form (uppercase, no spaces).
    """
    v = value.strip().upper().replace(" ", "")
    if not v:
        return ValidationResult(False, "Vehicle registration cannot be empty.")
    if len(v) > 8:
        return ValidationResult(
            False,
            "That doesn't look like a valid UK registration number. "
            "Please enter it in the format AB12 CDE.",
        )
    for pattern in _UK_REG_PATTERNS:
        if pattern.match(v):
            return ValidationResult(True, normalized_value=v)
    return ValidationResult(
        False,
        "Please enter a valid UK vehicle registration number (e.g. AB12 CDE).",
    )


# ---------------------------------------------------------------------------
# Date / time validator
# ---------------------------------------------------------------------------

_DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M",
]


def validate_datetime(
    value: str,
    field: str = "start_datetime",
    reservation: dict | None = None,
) -> ValidationResult:
    """
    Validate an ISO-like datetime string.

    Rules:
    - Must parse as a recognisable datetime format.
    - Must be in the future.
    - If field is ``end_datetime``, must be after ``start_datetime``.
    """
    v = value.strip()
    dt: datetime | None = None

    for fmt in _DATETIME_FORMATS:
        try:
            dt = datetime.strptime(v, fmt)
            break
        except ValueError:
            continue

    if dt is None:
        return ValidationResult(
            False,
            "Please enter the date and time in the format YYYY-MM-DD HH:MM "
            "(e.g. 2024-06-15 09:00).",
        )

    if dt < datetime.now():
        return ValidationResult(False, "The date and time must be in the future.")

    if field == "end_datetime" and reservation:
        start_raw = reservation.get("start_datetime")
        if start_raw:
            for fmt in _DATETIME_FORMATS:
                try:
                    start_dt = datetime.strptime(start_raw, fmt)
                    if dt <= start_dt:
                        return ValidationResult(
                            False, "The end time must be after the start time."
                        )
                    break
                except ValueError:
                    continue

    return ValidationResult(True, normalized_value=dt.strftime("%Y-%m-%d %H:%M"))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def validate_vehicle_type(
    value: str,
    valid_types: list[str] | None = None,
) -> ValidationResult:
    """
    Validate a vehicle type string against an optional allowlist.

    The reservation node handles vehicle_type via a numbered menu and
    ``_parse_vehicle_type()``; this validator is used for direct string
    submission (e.g. tests, API consumers).

    Args:
        value:       User-supplied vehicle type string.
        valid_types: Canonical type names that are currently available.
                     If None, any non-empty string is accepted.
    """
    v = value.strip().lower()
    if not v:
        return ValidationResult(False, "Vehicle type cannot be empty.")

    if valid_types is not None:
        canonical_map = {t.lower(): t for t in valid_types}
        if v not in canonical_map:
            options = ", ".join(valid_types)
            return ValidationResult(
                False,
                f"'{value}' is not a valid vehicle type. Choose from: {options}.",
            )
        return ValidationResult(True, normalized_value=canonical_map[v])

    return ValidationResult(True, normalized_value=v)


def validate_reservation_field(
    field: str,
    value: str,
    reservation: dict | None = None,
) -> ValidationResult:
    """Route a field value to the correct validator."""
    if field == "first_name":
        return validate_name(value)
    if field == "surname":
        return validate_name(value)
    if field == "vehicle_type":
        # vehicle_type is handled via menu in nodes.py; basic non-empty check here
        return validate_vehicle_type(value)
    if field == "vehicle_reg":
        return validate_vehicle_reg(value)
    if field in ("start_datetime", "end_datetime"):
        return validate_datetime(value, field=field, reservation=reservation or {})

    logger.warning("validate_reservation_field called with unknown field: %s", field)
    return ValidationResult(True, normalized_value=value.strip())
