"""
Tests for src/chatbot/extractor.py — reservation field extraction from free text.

All tests are offline (no DB, no LLM).
"""

from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract(text):
    from src.chatbot.extractor import extract_reservation_fields
    return extract_reservation_fields(text)


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------

class TestNameExtraction:
    def test_im_first_last(self):
        r = _extract("I'm Jaya Kumari. I want to book a spot.")
        assert r.get("first_name") == "Jaya"
        assert r.get("surname") == "Kumari"

    def test_my_name_is(self):
        r = _extract("My name is Alice Smith and I'd like to reserve.")
        assert r.get("first_name") == "Alice"
        assert r.get("surname") == "Smith"

    def test_i_am(self):
        r = _extract("I am John Brown. Please book standard for me.")
        assert r.get("first_name") == "John"
        assert r.get("surname") == "Brown"

    def test_no_name_returns_empty_name_fields(self):
        r = _extract("Book me a standard spot on 3 November 2090 at 5pm for 2 hours.")
        assert "first_name" not in r
        assert "surname" not in r

    def test_invalid_name_not_included(self):
        r = _extract("I'm 123 456. Book a spot.")
        assert "first_name" not in r


# ---------------------------------------------------------------------------
# Vehicle type extraction
# ---------------------------------------------------------------------------

class TestVehicleTypeExtraction:
    def test_standard_keyword(self):
        assert _extract("I need a standard parking spot.")["vehicle_type"] == "standard"

    def test_ev_charging_keyword(self):
        assert _extract("I have an electric vehicle.")["vehicle_type"] == "ev_charging"

    def test_motorcycle_keyword(self):
        assert _extract("I'm coming on my motorcycle.")["vehicle_type"] == "motorcycle"

    def test_disabled_blue_badge(self):
        assert _extract("I have a blue badge and need accessible parking.")["vehicle_type"] == "disabled"

    def test_oversized_van(self):
        assert _extract("I'll be arriving in a van.")["vehicle_type"] == "oversized"

    def test_no_vehicle_type_not_extracted(self):
        r = _extract("I'm Alice Smith, reg AB12CDE, 3rd Nov 2090 10am for 2 hours.")
        assert "vehicle_type" not in r


# ---------------------------------------------------------------------------
# Vehicle registration extraction
# ---------------------------------------------------------------------------

class TestVehicleRegExtraction:
    def test_post_2001_with_space(self):
        r = _extract("My registration is HF12 CHE.")
        assert r.get("vehicle_reg") == "HF12CHE"

    def test_post_2001_without_space(self):
        r = _extract("Reg: AB12CDE.")
        assert r.get("vehicle_reg") == "AB12CDE"

    def test_lowercase_normalised(self):
        r = _extract("registration hf12 che")
        assert r.get("vehicle_reg") == "HF12CHE"

    def test_invalid_string_not_extracted(self):
        r = _extract("No plate mentioned here.")
        assert "vehicle_reg" not in r


# ---------------------------------------------------------------------------
# Date / time extraction
# ---------------------------------------------------------------------------

class TestDateTimeExtraction:
    def test_full_natural_date_with_time(self):
        r = _extract("Book for 3 November 2090 at 5pm.")
        assert "start_datetime" in r
        assert "2090-11-03" in r["start_datetime"]
        assert "17:00" in r["start_datetime"]

    def test_ordinal_date(self):
        r = _extract("I need parking on 3rd Nov 2090 from 9 AM.")
        assert "start_datetime" in r
        assert "2090-11-03" in r["start_datetime"]

    def test_no_date_not_extracted(self):
        r = _extract("I'm Alice Smith and I drive a standard car, reg AB12CDE.")
        assert "start_datetime" not in r

    def test_past_date_not_extracted(self):
        r = _extract("I was at CityPark on 1 January 2000 at 9am.")
        assert "start_datetime" not in r


# ---------------------------------------------------------------------------
# Duration / end time calculation
# ---------------------------------------------------------------------------

class TestDurationExtraction:
    def test_for_3_hours(self):
        r = _extract("3 November 2090 at 5pm for 3 hours.")
        assert "end_datetime" in r
        assert r["end_datetime"] == "2090-11-03 20:00"

    def test_for_2_5_hours(self):
        r = _extract("3 November 2090 at 4pm for 2.5 hours.")
        assert "end_datetime" in r
        assert r["end_datetime"] == "2090-11-03 18:30"

    def test_for_30_minutes(self):
        r = _extract("3 November 2090 at 10am for 30 minutes.")
        assert "end_datetime" in r
        assert r["end_datetime"] == "2090-11-03 10:30"

    def test_no_duration_no_end_time(self):
        r = _extract("3 November 2090 at 5pm.")
        assert "end_datetime" not in r


# ---------------------------------------------------------------------------
# Complete one-message extraction
# ---------------------------------------------------------------------------

class TestCompleteExtraction:
    def test_full_message_from_requirements(self):
        msg = (
            "I'm Jaya Kumari. I want to book a standard parking spot on "
            "3 November 2090 from 5 PM for 3 hours. "
            "My vehicle registration is HF12 CHE."
        )
        r = _extract(msg)
        assert r.get("first_name") == "Jaya"
        assert r.get("surname") == "Kumari"
        assert r.get("vehicle_type") == "standard"
        assert r.get("vehicle_reg") == "HF12CHE"
        assert "start_datetime" in r
        assert "17:00" in r["start_datetime"]
        assert "end_datetime" in r
        assert "20:00" in r["end_datetime"]

    def test_partial_extraction_returns_only_found_fields(self):
        """If only name and reg provided, only those fields are returned."""
        r = _extract("I'm Bob Jones and my car is XY34ZAB.")
        assert "first_name" in r
        assert "surname" in r
        assert "vehicle_reg" in r
        assert "vehicle_type" not in r
        assert "start_datetime" not in r

    def test_empty_message_returns_empty_dict(self):
        r = _extract("")
        assert r == {}

    def test_irrelevant_message_returns_empty_dict(self):
        r = _extract("What are the parking rates?")
        assert r == {}


# ---------------------------------------------------------------------------
# Reservation node integration — bot does not re-ask for captured fields
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Regression tests for structured / labeled messages
# ---------------------------------------------------------------------------

class TestLabeledDatetimeExtraction:
    """Explicit 'start date:' / 'end date:' labels must be parsed correctly
    without dateutil dayfirst ambiguity and without re-asking for dates."""

    def test_regression_full_labeled_message(self):
        """
        Regression: the exact failing message from the user report must
        produce all 6 fields without any missing or mis-parsed dates.
        """
        r = _extract(
            "I want to make a reservation: Name: jaya Kumari "
            "Vehicle type: Standard Registeration number: AB23 HCR "
            "start date: 2026-11-03 10:00 end date: 2026-11-04 10:00"
        )
        assert r.get("first_name") == "jaya"
        assert r.get("surname") == "Kumari"
        assert r.get("vehicle_type") == "standard"
        assert r.get("vehicle_reg") == "AB23HCR"
        assert r.get("start_datetime") == "2026-11-03 10:00", (
            f"start_datetime mis-parsed as {r.get('start_datetime')!r}; "
            "expected '2026-11-03 10:00' (November 3, not March 11)"
        )
        assert r.get("end_datetime") == "2026-11-04 10:00"

    def test_iso_date_not_swapped_by_dayfirst(self):
        """ISO 'YYYY-MM-DD' dates must NOT be re-interpreted with dayfirst."""
        r = _extract(
            "I'm Test User. Standard parking. Reg TS01TST. "
            "start date: 2026-11-03 10:00 end date: 2026-11-03 18:00"
        )
        assert r.get("start_datetime") == "2026-11-03 10:00"
        assert r.get("end_datetime") == "2026-11-03 18:00"

    def test_end_date_label_only_no_duration(self):
        """Explicit 'end date:' field is extracted even without a duration phrase."""
        r = _extract("start date: 2090-06-15 09:00 end date: 2090-06-15 17:00")
        assert r.get("start_datetime") == "2090-06-15 09:00"
        assert r.get("end_datetime") == "2090-06-15 17:00"

    def test_end_time_label_variant(self):
        r = _extract("start time: 2090-07-01 08:00 end time: 2090-07-01 20:00")
        assert r.get("start_datetime") == "2090-07-01 08:00"
        assert r.get("end_datetime") == "2090-07-01 20:00"

    def test_partial_labeled_only_start(self):
        """Only start date given via label — end_datetime should be absent."""
        r = _extract("start date: 2090-08-01 09:00")
        assert r.get("start_datetime") == "2090-08-01 09:00"
        assert "end_datetime" not in r

    def test_invalid_labeled_date_rejected(self):
        """A syntactically invalid date after a label must not crash and not appear."""
        r = _extract("start date: not-a-date end date: also-invalid")
        assert "start_datetime" not in r
        assert "end_datetime" not in r

    def test_past_labeled_date_rejected(self):
        """A past date in a labeled field must be rejected (validator rejects it)."""
        r = _extract("start date: 2000-01-01 09:00 end date: 2000-01-01 17:00")
        assert "start_datetime" not in r
        assert "end_datetime" not in r

    def test_end_before_start_rejected(self):
        """An end date earlier than the start must not appear in the result."""
        r = _extract("start date: 2090-06-15 17:00 end date: 2090-06-15 09:00")
        assert r.get("start_datetime") == "2090-06-15 17:00"
        assert "end_datetime" not in r   # end < start → dropped

    def test_duration_fallback_still_works_with_labeled_start(self):
        """If only start is labeled and a duration is given, end is calculated."""
        r = _extract("start date: 2090-06-15 09:00 for 3 hours")
        assert r.get("start_datetime") == "2090-06-15 09:00"
        assert r.get("end_datetime") == "2090-06-15 12:00"


class TestReservationNodeWithExtraction:
    """End-to-end: the reservation_node must not re-ask for already-captured fields."""

    def _state(self, **overrides):
        from src.chatbot.state import empty_state
        s = empty_state()
        s.update(overrides)
        return s

    def test_partial_message_skips_to_first_missing_field(self, monkeypatch):
        """
        Message provides name + reg but no type/dates.
        Bot should skip to asking for vehicle_type, not re-ask name.
        """
        import src.database.queries as q
        monkeypatch.setattr(q, "get_available_vehicle_types", lambda **kw: [
            {"vehicle_type": "standard", "display_name": "Standard",
             "total_spaces": 720, "booked_count": 0, "available_spaces": 720},
        ])
        from src.chatbot.nodes import reservation_node
        from src.chatbot.router import INTENT_RESERVATION

        result = reservation_node(self._state(
            user_input="I'm Alice Smith. Reg is AB12CDE.",
            intent=INTENT_RESERVATION,
            reservation_active=False,
        ))

        assert result["reservation"].get("first_name") == "Alice"
        assert result["reservation"].get("surname") == "Smith"
        assert result["reservation"].get("vehicle_reg") == "AB12CDE"
        # Bot must ask for vehicle_type (next missing), NOT first_name again
        assert result["next_field_to_ask"] == "vehicle_type"
        assert "first name" not in result["response"].lower()

    def test_complete_message_goes_straight_to_save(self, monkeypatch):
        """All 6 fields in one message → save_reservation called, no form questions."""
        import src.database.queries as q
        monkeypatch.setattr(q, "get_available_vehicle_types", lambda **kw: [
            {"vehicle_type": "standard", "display_name": "Standard",
             "total_spaces": 720, "booked_count": 0, "available_spaces": 720},
        ])
        monkeypatch.setattr(q, "save_reservation",
                            lambda **kw: {"success": True, "reservation_id": 99,
                                          "error": None, "available_spaces": 719})

        from src.chatbot.nodes import reservation_node
        from src.chatbot.router import INTENT_RESERVATION

        msg = (
            "I'm Jaya Kumari. I want a standard spot on "
            "3 November 2090 from 5 PM for 3 hours. Reg HF12 CHE."
        )
        result = reservation_node(self._state(
            user_input=msg,
            intent=INTENT_RESERVATION,
            reservation_active=False,
        ))

        assert result["reservation_active"] is False
        assert result["pending_reservation_id"] == 99
        assert "Jaya Kumari" in result["response"] or "#99" in result["response"]

    def test_gradual_flow_still_works(self):
        """Sending just 'I want to make a reservation' still asks for first_name."""
        from src.chatbot.nodes import reservation_node
        from src.chatbot.router import INTENT_RESERVATION

        result = reservation_node(self._state(
            user_input="I want to make a reservation",
            intent=INTENT_RESERVATION,
            reservation_active=False,
        ))
        assert result["next_field_to_ask"] == "first_name"
        assert result["reservation_active"] is True

    def test_bot_does_not_reask_captured_fields(self, monkeypatch):
        """
        If the user provided name + vehicle type + reg + start/end in one message,
        ensure none of those fields appear as questions in the response.
        """
        import src.database.queries as q
        monkeypatch.setattr(q, "get_available_vehicle_types", lambda **kw: [
            {"vehicle_type": "standard", "display_name": "Standard",
             "total_spaces": 720, "booked_count": 0, "available_spaces": 720},
        ])

        from src.chatbot.nodes import reservation_node
        from src.chatbot.router import INTENT_RESERVATION

        # Provides name, vehicle type and reg, but no dates
        result = reservation_node(self._state(
            user_input="I'm Bob Jones. Standard parking, my reg is XY34ZAB.",
            intent=INTENT_RESERVATION,
            reservation_active=False,
        ))

        resp = result["response"].lower()
        assert "first name" not in resp      # not re-asked
        assert "surname" not in resp         # not re-asked
        assert "registration" not in resp    # not re-asked
