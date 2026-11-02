"""
Reservation field extractor.

Attempts to parse all six reservation fields from a single free-text message.
Returns only the fields that were successfully extracted AND pass validation.
Empty fields are omitted so the caller knows they still need to be collected.

This is intentionally a best-effort parser: it never invents or defaults
values.  If a field cannot be confidently extracted, it is simply left out
so the reservation node will ask for it explicitly.

Extracted fields follow the same canonical formats used by validate_datetime
and validate_vehicle_reg, so they can be passed directly to the existing
validators and DB functions.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from src.config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Vehicle type aliases
# ---------------------------------------------------------------------------

_VEHICLE_TYPE_ALIASES: dict[str, str] = {
    "standard":        "standard",
    "normal":          "standard",
    "regular":         "standard",
    # "car" intentionally omitted — too ambiguous in general conversation
    "saloon":          "standard",
    "hatchback":       "standard",
    "disabled":        "disabled",
    "blue badge":      "disabled",
    "accessible":      "disabled",
    "disability":      "disabled",
    "wheelchair":      "disabled",
    "ev":              "ev_charging",
    "electric":        "ev_charging",
    "ev charging":     "ev_charging",
    "electric vehicle":"ev_charging",
    "motorcycle":      "motorcycle",
    "motorbike":       "motorcycle",
    "moped":           "motorcycle",
    "bike":            "motorcycle",
    "oversized":       "oversized",
    "van":             "oversized",
    "lorry":           "oversized",
    "large":           "oversized",
    "minibus":         "oversized",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_vehicle_type(text: str) -> str | None:
    """Return canonical vehicle_type if found in *text*, else None."""
    t = text.lower()
    # Longest match first to prefer "ev charging" over "ev"
    for alias in sorted(_VEHICLE_TYPE_ALIASES, key=len, reverse=True):
        if alias in t:
            return _VEHICLE_TYPE_ALIASES[alias]
    return None


def _extract_vehicle_reg(text: str) -> str | None:
    """
    Extract a UK vehicle registration from *text*.

    Matches post-2001 (AB12 CDE), prefix-era (A123 ABC) and generic
    alphanumeric plates.  Returns normalised uppercase without spaces.
    """
    patterns = [
        r"\b([A-Za-z]{2}[0-9]{2}\s?[A-Za-z]{3})\b",   # post-2001
        r"\b([A-Za-z][0-9]{1,3}\s?[A-Za-z]{3})\b",     # prefix era
        r"\b([A-Za-z]{3}\s?[0-9]{1,3}[A-Za-z])\b",     # suffix era
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper().replace(" ", "")
    return None


def _extract_name(text: str) -> tuple[str | None, str | None]:
    """
    Heuristically extract first name and surname from *text*.

    Looks for patterns like:
      "I'm John Smith"  "My name is Jane Doe"  "I am Alice Brown"
    Returns (first_name, surname) or (None, None).
    """
    name_patterns = [
        r"(?:i'm|i am|my name is|name[:\s]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)",
        r"(?:i'm|i am)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)",
    ]
    for pat in name_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None, None


def _parse_natural_datetime(text: str, reference: datetime | None = None) -> datetime | None:
    """
    Parse a natural-language date/time from *text* using dateutil.

    Returns a datetime or None if nothing recognisable is found.
    """
    try:
        from dateutil import parser as du_parser
        from dateutil.parser import ParserError
    except ImportError:
        return None

    # Narrow down the text to the date-like portion to avoid false positives
    # on plain integers (e.g. vehicle registrations like "AB12 CDE").
    date_patterns = [
        # "3 November 2026", "3rd Nov 2026", "November 3 2026"
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
        r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{2,4}\b",
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b",
        # ISO and slash formats
        r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    ]

    date_str: str | None = None
    for pat in date_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            date_str = m.group(0)
            break

    if date_str is None:
        return None

    # Extract time component separately.
    # Require am/pm or HH:MM format to avoid matching bare date digits.
    time_pat = r"\b(\d{1,2}(?::\d{2})?)\s*(?:am|pm)\b"
    time_m = re.search(time_pat, text, re.IGNORECASE)
    if time_m:
        time_str = time_m.group(0)
        combined = f"{date_str} {time_str}"
    else:
        # Fall back to HH:MM format (no am/pm)
        hhmm_pat = r"\b([01]?\d|2[0-3]):[0-5]\d\b"
        hhmm_m = re.search(hhmm_pat, text)
        if hhmm_m:
            combined = f"{date_str} {hhmm_m.group(0)}"
        else:
            combined = date_str

    # For ISO-format dates (YYYY-MM-DD ...) use strptime directly so that
    # dateutil's dayfirst heuristic cannot swap month and day.
    # e.g. "2026-11-03 10:00" must stay as November 3, not March 11.
    if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", combined):
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(combined, fmt)
            except ValueError:
                continue

    # Natural-language dates — use dateutil with dayfirst for "3 Nov 2090"
    try:
        dt = du_parser.parse(combined, dayfirst=True)
        return dt
    except (ParserError, ValueError, OverflowError):
        return None


def _extract_duration_hours(text: str) -> float | None:
    """
    Extract a duration in hours from phrases like "for 3 hours", "2.5 hours", "30 minutes".
    Returns duration in hours as a float, or None.
    """
    # "for X hours" / "X hour"
    m = re.search(r"for\s+(\d+(?:\.\d+)?)\s*hours?", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # "X minutes"
    m = re.search(r"(\d+(?:\.\d+)?)\s*minutes?", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 60.0
    # "X.X hours" without "for"
    m = re.search(r"(\d+(?:\.\d+)?)\s*hours?", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Labeled datetime extraction
# ---------------------------------------------------------------------------

# Labels recognised for start and end datetimes (longer labels take priority).
_START_LABELS = ["start date", "start time", "parking start", "from date", "from time"]
_END_LABELS   = ["end date",   "end time",   "parking end",   "to date",   "until date",
                 "until time", "finish date", "finish time"]


def _parse_iso_or_strptime(value: str) -> datetime | None:
    """
    Parse a datetime string using strict strptime formats only (no dateutil).

    Used for values extracted from explicit labels where the user has already
    provided the date in a machine-readable form.
    """
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y/%m/%d %H:%M",
                "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M",
                "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _extract_labeled_datetimes(
    text: str,
) -> tuple[datetime | None, datetime | None]:
    """
    Extract start and end datetimes from explicitly labeled fields.

    Handles structured messages such as:
        "start date: 2026-11-03 10:00  end date: 2026-11-04 10:00"
        "start time: 2026-11-03 10:00, end time: 2026-11-04 10:00"

    For each label, grabs the first ISO/numeric datetime that follows it,
    stopping at the next label keyword so values don't bleed across fields.
    Returns (start_dt, end_dt); either may be None.
    """
    # Build a combined stop-pattern of all label words so we don't overshoot
    all_label_words = "|".join(
        re.escape(lbl) for lbl in sorted(
            _START_LABELS + _END_LABELS, key=len, reverse=True
        )
    )
    # Regex: label word(s) → optional colon/dash → capture value up to next label
    label_value_pat = re.compile(
        rf"({all_label_words})\s*[:\-]?\s*([\d/\-T: ]+?)(?=\s*(?:{all_label_words})|$)",
        re.IGNORECASE,
    )

    start_dt: datetime | None = None
    end_dt:   datetime | None = None

    for m in label_value_pat.finditer(text):
        label = m.group(1).lower().strip()
        raw   = m.group(2).strip()
        dt    = _parse_iso_or_strptime(raw)
        if dt is None:
            continue
        if any(label.startswith(s) or s in label for s in ("start", "from", "parking start")):
            if start_dt is None:
                start_dt = dt
        elif any(label.startswith(s) or s in label
                 for s in ("end", "to", "until", "finish", "parking end")):
            if end_dt is None:
                end_dt = dt

    return start_dt, end_dt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_reservation_fields(text: str) -> dict[str, str]:
    """
    Extract and validate reservation fields from a free-text message.

    Only returns fields that were:
    1. Successfully extracted from the text, AND
    2. Pass the existing validators (validate_reservation_field).

    Invalid extractions are silently dropped so the caller will ask
    the user to provide them explicitly.

    Returns a dict with zero or more of:
        first_name, surname, vehicle_type, vehicle_reg,
        start_datetime, end_datetime

    The values are in the same canonical format expected by the reservation
    form (e.g. vehicle_reg without spaces, datetime as "YYYY-MM-DD HH:MM").
    """
    from src.guardrails.validators import validate_reservation_field

    result: dict[str, str] = {}

    # ── Names ─────────────────────────────────────────────────────────────
    first, last = _extract_name(text)
    if first:
        vr = validate_reservation_field("first_name", first)
        if vr.is_valid and vr.normalized_value:
            result["first_name"] = vr.normalized_value
    if last:
        vr = validate_reservation_field("surname", last)
        if vr.is_valid and vr.normalized_value:
            result["surname"] = vr.normalized_value

    # ── Vehicle type ───────────────────────────────────────────────────────
    vtype = _extract_vehicle_type(text)
    if vtype:
        result["vehicle_type"] = vtype

    # ── Vehicle registration ───────────────────────────────────────────────
    vreg = _extract_vehicle_reg(text)
    if vreg:
        vr = validate_reservation_field("vehicle_reg", vreg)
        if vr.is_valid and vr.normalized_value:
            result["vehicle_reg"] = vr.normalized_value

    # ── Datetimes ──────────────────────────────────────────────────────────
    now = datetime.now()

    # Labeled fields take priority ("start date: …", "end date: …") because
    # they are unambiguous and bypass heuristic date-matching.
    labeled_start, labeled_end = _extract_labeled_datetimes(text)

    start_dt = labeled_start or _parse_natural_datetime(text)

    if start_dt and start_dt > now:
        result["start_datetime"] = start_dt.strftime("%Y-%m-%d %H:%M")

        # Prefer an explicit labeled end datetime; fall back to duration.
        end_dt: datetime | None = None
        if labeled_end and labeled_end > start_dt:
            end_dt = labeled_end
        else:
            duration = _extract_duration_hours(text)
            if duration and duration > 0:
                end_dt = start_dt + timedelta(hours=duration)

        if end_dt and end_dt > start_dt:
            result["end_datetime"] = end_dt.strftime("%Y-%m-%d %H:%M")

    if result:
        logger.debug("Extracted reservation fields: %s", list(result.keys()))

    return result
