"""Test that reply_templates.py preserves API values exactly.

This test verifies that the templates in reply_templates.py pass values
from the API response unchanged into the spoken text, as required by the
acceptance criteria: "Figures pass from the validated response into the
template unchanged and are verbalised digit-faithfully."
"""
import pytest
# Import the actual functions (not to be confused with test functions)
from agent.reply_templates import (
    test_rate_reply as rate_reply,
    doctor_availability_reply, booking_reply,
    doctors_by_department_reply, missing_slot_prompt,
    booking_confirmation_prompt, booking_correction_prompt,
)
from agent.bn_normalize import verbalize


class TestReplyTemplateValuePreservation:
    """Test that reply templates preserve exact API values."""

    def test_rate_reply_preserves_exact_rate(self):
        """The rate from API should be inserted exactly as received."""
        slots = {"test_name": "Uric Acid"}
        result = {
            "found": True,
            "test_name": "Uric Acid",
            "test_name_bn": "ইউরিক এসিড",
            "rate_inr": 650,
            "sample_type": "Blood",
            "report_time_hours": 24,
        }
        
        reply = rate_reply(slots, result)
        
        # The exact rate value (650) should be in the reply
        assert "650" in reply
        # No rounding or modification should occur
        assert "650.0" not in reply  # Should not add decimal
        assert "649" not in reply  # Should not round down
        assert "651" not in reply  # Should not round up

    def test_test_rate_reply_with_different_rates(self):
        """Various rate values should be preserved exactly."""
        test_cases = [
            350, 650, 850, 1200, 2500, 5000, 9999
        ]
        
        for rate in test_cases:
            slots = {"test_name": "Test"}
            result = {
                "found": True,
                "test_name": "Test",
                "test_name_bn": "টেস্ট",
                "rate_inr": rate,
                "sample_type": "Blood",
                "report_time_hours": 24,
            }
            
            reply = rate_reply(slots, result)
            assert str(rate) in reply, f"Rate {rate} should be preserved in reply"

    def test_test_rate_reply_preserves_report_time(self):
        """Report time hours should be preserved exactly."""
        slots = {"test_name": "CBC"}
        result = {
            "found": True,
            "test_name": "CBC",
            "test_name_bn": "সিবিসি",
            "rate_inr": 850,
            "sample_type": "Blood",
            "report_time_hours": 4,
        }
        
        reply = rate_reply(slots, result)
        assert "4" in reply
        assert "4.0" not in reply  # Should not add decimal

    def test_booking_reply_preserves_confirmation_id(self):
        """Confirmation ID should be preserved exactly."""
        slots = {
            "doctor_name": "Dr. Sen",
            "date": "2026-08-24",
            "time_slot": "09:30",
            "patient_name": "Rahul",
            "phone": "9876543210",
        }
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply = booking_reply(slots, result)
        # The exact confirmation ID should be preserved
        assert "KCD-20260824-0031" in reply
        # No modification should occur
        assert "KCD-20260824-0032" not in reply  # Should not change
        assert "KCD-20260824-31" not in reply  # Should not truncate

    def test_booking_reply_preserves_date(self):
        """Date should be preserved exactly in ISO format."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply = booking_reply(slots, result)
        # The exact date should be preserved
        assert "2026-08-24" in reply
        # No modification should occur
        assert "2026-08-25" not in reply  # Should not change
        assert "08-24" not in reply or "2026-08-24" in reply  # Should not truncate year

    def test_booking_reply_preserves_time_slot(self):
        """Time slot should be preserved exactly."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply = booking_reply(slots, result)
        # The exact time slot should be preserved
        assert "09:30" in reply
        # No modification should occur
        assert "09:00" not in reply  # Should not round
        assert "10:00" not in reply  # Should not round

    def test_doctor_availability_reply_preserves_date(self):
        """Date should be preserved exactly."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "found": True,
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "available": True,
            "chamber_hours": "18:00-20:00",
            "next_available_date": None,
        }
        
        reply = doctor_availability_reply(slots, result)
        # The exact date should be preserved
        assert "2026-08-24" in reply

    def test_doctor_availability_reply_preserves_chamber_hours(self):
        """Chamber hours should be preserved exactly."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "found": True,
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "available": True,
            "chamber_hours": "18:00-20:00",
            "next_available_date": None,
        }
        
        reply = doctor_availability_reply(slots, result)
        # The exact chamber hours should be preserved
        assert "18:00-20:00" in reply
        # No modification should occur
        assert "18:00-19:00" not in reply  # Should not change
        assert "6:00-8:00" not in reply  # Should not change format

    def test_booking_reply_preserves_alternative_slots(self):
        """Alternative time slots should be preserved exactly."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": False,
            "reason": "slot_taken",
            "alternative_slots": ["09:00", "09:15", "10:00"],
        }
        
        reply = booking_reply(slots, result)
        # The exact alternative slots should be preserved
        assert "09:00" in reply
        assert "09:15" in reply
        assert "10:00" in reply


class TestTemplateNoValueModification:
    """Test that templates don't perform any arithmetic or string manipulation."""

    def test_no_string_formatting_modification(self):
        """String formatting should not modify values."""
        slots = {"test_name": "Test"}
        result = {
            "found": True,
            "test_name": "Test",
            "test_name_bn": "টেস্ট",
            "rate_inr": 1234,
            "sample_type": "Blood",
            "report_time_hours": 48,
        }
        
        reply = rate_reply(slots, result)
        
        # The value should appear exactly as is
        assert "1234" in reply
        # Should not be reformatted (e.g., with commas)
        assert "1,234" not in reply
        # Should not be padded
        assert "01234" not in reply

    def test_no_arithmetic_operations(self):
        """Templates should not perform arithmetic on values."""
        slots = {"test_name": "Test"}
        result = {
            "found": True,
            "test_name": "Test",
            "test_name_bn": "টেস্ট",
            "rate_inr": 999,
            "sample_type": "Blood",
            "report_time_hours": 1,
        }
        
        reply = rate_reply(slots, result)
        
        # Should not add values
        assert "1000" not in reply
        # Should not multiply values
        assert "1998" not in reply
        # Original value should be present
        assert "999" in reply


class TestNoSpokenPunctuationArtifact:
    """Answers sound like a person, not a database row (Answer Quality and
    Grounding). Acceptance criterion: "No field label, colon or bracket is
    ever spoken and every structured value renders as a natural clause in
    the reply language. An automated check fails a build containing a
    spoken punctuation artefact." This class IS that automated check.

    Every reply-producing function is called with representative fixture
    data across every language it supports, and the result is then run
    through agent.bn_normalize.verbalize() -- the same step agent/tts.py's
    synthesize() applies before anything reaches the caller's ear -- before
    asserting no ':', '：', '[', ']', '{' or '}' survives. Checking the
    POST-verbalize text, not the raw template string, is deliberate: a raw
    24h time like "18:00" or a date like "2026-08-24" legitimately contains
    punctuation in the template layer, but verbalize()'s _RE_TIME/_RE_DATE
    patterns rewrite those into spoken words before synthesis, so they were
    never actually spoken and must not fail this check. Only a literal
    label-colon or bracket that survives verbalize() unchanged is a real
    violation -- which is exactly what "Sample: X" / "Time: X" /
    "Confirmation number: X" used to be before this story's fix.
    """

    _BANNED = (":", "：", "[", "]", "{", "}")

    def _assert_clean(self, text: str, language: str):
        spoken = verbalize(text, language=language)
        for ch in self._BANNED:
            assert ch not in spoken, (
                f"spoken punctuation artefact {ch!r} survived verbalize() "
                f"for language={language!r}: {spoken!r}"
            )

    # -- missing_slot_prompt: every (intent, missing) pair this function
    # actually maps, in each supported language --
    _SLOT_PROMPT_KEYS = [
        ("test_rate", "test_name"),
        ("doctor_availability", "doctor_name"),
        ("doctors_by_department", "department"),
        ("doctors_by_department", "date"),
        ("book_appointment", "doctor_name"),
        ("book_appointment", "date"),
        ("book_appointment", "time_slot"),
        ("book_appointment", "patient_name"),
        ("book_appointment", "phone"),
    ]

    @pytest.mark.parametrize("intent,missing", _SLOT_PROMPT_KEYS)
    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_missing_slot_prompt_is_clean(self, intent, missing, language):
        self._assert_clean(missing_slot_prompt(intent, missing, language=language), language)

    # -- test_rate_reply: found, found-with-sample-and-hours, not-found
    # with suggestions, not-found without suggestions --
    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_rate_reply_found_is_clean(self, language):
        slots = {"test_name": "Uric Acid"}
        result = {
            "found": True, "test_name": "Uric Acid", "test_name_bn": "ইউরিক এসিড",
            "rate_inr": 650, "sample_type": "Blood", "report_time_hours": 24,
        }
        self._assert_clean(rate_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_rate_reply_not_found_with_suggestions_is_clean(self, language):
        slots = {"test_name": "Yuric Acid"}
        result = {"found": False, "did_you_mean": ["Uric Acid", "Urea"]}
        self._assert_clean(rate_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_rate_reply_not_found_without_suggestions_is_clean(self, language):
        slots = {"test_name": "Nonexistent Test"}
        result = {"found": False, "did_you_mean": []}
        self._assert_clean(rate_reply(slots, result, language=language), language)

    # -- doctor_availability_reply: not-found, available, unavailable with
    # next date, unavailable with no schedule --
    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_doctor_availability_not_found_is_clean(self, language):
        slots = {"doctor_name": "Nobody"}
        result = {"found": False}
        self._assert_clean(doctor_availability_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_doctor_availability_available_is_clean(self, language):
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "found": True, "doctor_name": "Dr. A. Sen", "doctor_name_bn": "সেন",
            "date": "2026-08-24", "available": True, "chamber_hours": "18:00-20:00",
            "next_available_date": None,
        }
        self._assert_clean(doctor_availability_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_doctor_availability_next_date_is_clean(self, language):
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "found": True, "doctor_name": "Dr. A. Sen", "doctor_name_bn": "সেন",
            "available": False, "next_available_date": "2026-08-26",
        }
        self._assert_clean(doctor_availability_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_doctor_availability_no_schedule_is_clean(self, language):
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "found": True, "doctor_name": "Dr. A. Sen", "doctor_name_bn": "সেন",
            "available": False, "next_available_date": None,
        }
        self._assert_clean(doctor_availability_reply(slots, result, language=language), language)

    # -- booking_reply: success, slot_taken with alts, slot_taken without
    # alts, doctor_not_found, generic failure --
    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_booking_reply_success_is_clean(self, language):
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True, "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen", "doctor_name_bn": "সেন",
            "date": "2026-08-24", "time_slot": "09:30",
        }
        self._assert_clean(booking_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_booking_reply_slot_taken_with_alts_is_clean(self, language):
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": False, "reason": "slot_taken",
            "alternative_slots": ["09:00", "09:15", "10:00"],
        }
        self._assert_clean(booking_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_booking_reply_slot_taken_without_alts_is_clean(self, language):
        slots = {"doctor_name": "Dr. Sen"}
        result = {"success": False, "reason": "slot_taken", "alternative_slots": []}
        self._assert_clean(booking_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_booking_reply_doctor_not_found_is_clean(self, language):
        slots = {"doctor_name": "Nobody"}
        result = {"success": False, "reason": "doctor_not_found"}
        self._assert_clean(booking_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_booking_reply_generic_failure_is_clean(self, language):
        slots = {"doctor_name": "Dr. Sen"}
        result = {"success": False, "reason": "unknown_error"}
        self._assert_clean(booking_reply(slots, result, language=language), language)

    # -- doctors_by_department_reply: not-found, no doctors (filtered by
    # date and not), one doctor, two doctors, three-or-more doctors --
    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    def test_doctors_by_department_not_found_is_clean(self, language):
        slots = {"department": "Nowhere"}
        result = {"found": False}
        self._assert_clean(doctors_by_department_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    @pytest.mark.parametrize("has_date", [True, False])
    def test_doctors_by_department_no_doctors_is_clean(self, language, has_date):
        slots = {"department": "Cardiology"}
        result = {"found": True, "department": "Cardiology", "doctors": []}
        if has_date:
            result["date"] = "2026-08-24"
        self._assert_clean(doctors_by_department_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish"])
    @pytest.mark.parametrize("doctor_count", [1, 2, 3])
    def test_doctors_by_department_listing_is_clean(self, language, doctor_count):
        slots = {"department": "Cardiology"}
        doctors = [
            {"name": f"Dr. {i} Sen", "doctor_name_bn": f"সেন{i}"} for i in range(doctor_count)
        ]
        result = {"found": True, "department": "Cardiology", "doctors": doctors}
        self._assert_clean(doctors_by_department_reply(slots, result, language=language), language)

    # -- booking_confirmation_prompt / booking_correction_prompt: all four
    # languages, since these two already support "banglish" --
    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish", "banglish"])
    def test_booking_confirmation_prompt_is_clean(self, language):
        slots = {
            "doctor_name": "Dr. A. Sen", "date": "2026-08-24", "time_slot": "09:30",
            "patient_name": "Rahul", "phone": "9876543210",
        }
        self._assert_clean(booking_confirmation_prompt(slots, language=language), language)

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish", "banglish"])
    def test_booking_correction_prompt_is_clean(self, language):
        self._assert_clean(booking_correction_prompt(language=language), language)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
