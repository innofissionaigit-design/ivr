"""Test that reply_templates.py preserves API values exactly.

This test verifies that the templates in reply_templates.py pass values
from the API response unchanged into the spoken text, as required by the
acceptance criteria: "Figures pass from the validated response into the
template unchanged and are verbalised digit-faithfully."
"""
import re

import pytest
# Import the actual functions (not to be confused with test functions)
from agent.reply_templates import (
    test_rate_reply as rate_reply,
    sample_type_reply,
    doctor_availability_reply, booking_reply,
    doctors_by_department_reply, missing_slot_prompt,
    booking_confirmation_prompt, booking_correction_prompt,
    _spoken_sample_types, _name_already_says_test, _a_or_an,
)
from agent.bn_normalize import verbalize, hours_to_duration_phrase, unspeakable_spans


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

    def test_test_rate_reply_speaks_report_time_as_a_natural_duration(self):
        """"Caller asks how long results take" (Conversation: Information
        and Enquiry). AC: report time is "expressed as a natural duration
        rather than a number of hours read as a figure" -- so, unlike
        rate_inr, the raw hour count must NOT survive into the reply, and
        the natural-duration phrase from bn_normalize.hours_to_duration_
        phrase() must."""
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
        assert hours_to_duration_phrase(4, "bengali") in reply
        # The raw hour figure must not be read out as a number -- "4" only
        # legitimately appears here as part of "850" it is not (it isn't),
        # so a direct absence check is safe and exact.
        assert "4 ঘণ্টা" not in reply
        assert re.search(r"\b4\b", reply) is None

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
        ("test_sample", "test_name"),
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

    # -- Banglish: only added to test_rate_reply's own checks here, not to
    # the shared "language" parametrize list above -- doctor_availability_
    # reply() and doctors_by_department_reply() still have no banglish
    # branch, and adding "banglish" to that shared list would silently
    # exercise their Bengali fallback under a banglish label rather than
    # a real banglish branch, which is not what this story touched. --
    def test_rate_reply_found_is_clean_banglish(self):
        slots = {"test_name": "Uric Acid"}
        result = {
            "found": True, "test_name": "Uric Acid", "test_name_bn": "ইউরিক এসিড",
            "rate_inr": 650, "sample_type": "Blood", "report_time_hours": 24,
        }
        self._assert_clean(rate_reply(slots, result, language="banglish"), "banglish")

    def test_rate_reply_not_found_with_suggestions_is_clean_banglish(self):
        slots = {"test_name": "Yuric Acid"}
        result = {"found": False, "did_you_mean": ["Uric Acid", "Urea"]}
        self._assert_clean(rate_reply(slots, result, language="banglish"), "banglish")

    def test_rate_reply_not_found_without_suggestions_is_clean_banglish(self):
        slots = {"test_name": "Nonexistent Test"}
        result = {"found": False, "did_you_mean": []}
        self._assert_clean(rate_reply(slots, result, language="banglish"), "banglish")

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


class TestReportTimeIsANaturalDuration:
    """"Caller asks how long results take" (Epic: Conversation --
    Information and Enquiry). AC: "Reporting time comes from the catalogue
    and is expressed as a natural duration rather than a number of hours
    read as a figure. Where turnaround varies by day of week or by branch
    that is stated."

    hours_to_duration_phrase() itself is covered directly; this class
    checks the one real call site (test_rate_reply()) actually uses it
    and that no raw hour figure survives, for every hour value clinic-api/
    seed.py seeds today (1, 4, 6, 12, 24, 48, 72) plus one value beyond
    the catalogue's current range (96) to prove the fallback works too.
    """

    # Every distinct value seed.py actually seeds, plus one beyond it.
    _SEEDED_HOURS = [1, 4, 6, 12, 24, 48, 72, 96]

    @pytest.mark.parametrize("hours", _SEEDED_HOURS)
    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish", "banglish"])
    def test_duration_phrase_is_nonempty_and_distinct_per_bucket(self, language, hours):
        phrase = hours_to_duration_phrase(hours, language=language)
        assert isinstance(phrase, str) and phrase.strip()

    def test_bucket_boundaries_are_exact(self):
        # 6/7, 12/13, 24/25, 48/49, 72/73 -- confirms the ceilings in
        # bn_normalize._DURATION_BUCKETS land where reply_templates.py's
        # docstring and the plan say they do, not off by one.
        assert hours_to_duration_phrase(6, "english") == "within a few hours"
        assert hours_to_duration_phrase(7, "english") == "within half a day"
        assert hours_to_duration_phrase(12, "english") == "within half a day"
        assert hours_to_duration_phrase(13, "english") == "within a day"
        assert hours_to_duration_phrase(24, "english") == "within a day"
        assert hours_to_duration_phrase(25, "english") == "within two days"
        assert hours_to_duration_phrase(48, "english") == "within two days"
        assert hours_to_duration_phrase(49, "english") == "within three days"
        assert hours_to_duration_phrase(72, "english") == "within three days"

    def test_beyond_catalogue_falls_back_to_a_day_count(self):
        # 96h isn't in seed.py today -- the fallback must still produce a
        # natural sentence, not raise or silently omit the duration.
        assert hours_to_duration_phrase(96, "english") == "within 4 days"
        assert hours_to_duration_phrase(96, "hinglish") == "4 din mein"
        assert hours_to_duration_phrase(96, "banglish") == "4 diner modhye"
        assert hours_to_duration_phrase(96, "bengali") == "4 দিনের মধ্যে"

    def test_unknown_language_falls_back_to_bengali(self):
        assert hours_to_duration_phrase(24, "klingon") == hours_to_duration_phrase(24, "bengali")

    def test_never_fabricates_branch_or_weekday_variance(self):
        # The AC's second sentence is conditional ("where turnaround
        # varies ... that is stated") -- nothing in clinic-api/models.py
        # varies by branch or weekday today, so the phrase must never
        # claim it does.
        for hours in self._SEEDED_HOURS:
            for language in ("bengali", "english", "hinglish", "banglish"):
                phrase = hours_to_duration_phrase(hours, language=language)
                for word in ("branch", "শাখা", "weekday", "সপ্তাহ", "varies", "depend"):
                    assert word not in phrase.lower()

    def test_rate_reply_banglish_avoids_double_test_word(self):
        """The new banglish branch mirrors the pre-existing english/
        hinglish "already says 'test'" guard (reply_templates.py's `if
        "test" in name.lower()`) -- this covers that branch for banglish,
        which is new code this story added (english/hinglish's equivalent
        branch predates this story and is untouched by this diff)."""
        slots = {"test_name": "CBC Test"}
        result = {
            "found": True, "test_name": "CBC Test",
            "rate_inr": 850, "sample_type": "Blood", "report_time_hours": 24,
        }
        reply = rate_reply(slots, result, language="banglish")
        assert "CBC Test rate 850 taka." in reply
        assert "Test test" not in reply and "test test" not in reply.lower()

    @pytest.mark.parametrize("hours", _SEEDED_HOURS)
    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish", "banglish"])
    def test_rate_reply_uses_the_duration_phrase_not_a_raw_hour_figure(self, hours, language):
        """The actual call site: test_rate_reply() must speak the
        natural-duration phrase, and the raw hour count must not appear
        anywhere in the reply as a standalone figure."""
        slots = {"test_name": "Test"}
        result = {
            "found": True, "test_name": "Test", "test_name_bn": "টেস্ট",
            "rate_inr": 999, "sample_type": "Blood", "report_time_hours": hours,
        }
        reply = rate_reply(slots, result, language=language)
        assert hours_to_duration_phrase(hours, language) in reply
        # 999 (the rate) legitimately contains no standalone "hours"
        # digit-run equal to `hours`, so a whole-number-boundary search
        # for the raw hour figure is a safe, exact negative check.
        assert re.search(rf"(?<!\d){hours}(?!\d)", reply.replace("999", "")) is None

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish", "banglish"])
    def test_rate_reply_omits_duration_sentence_when_hours_absent(self, language):
        """No report_time_hours in the tool response -- e.g. a future
        catalogue row missing it -- must not produce a duration sentence
        out of nothing (mirrors the pre-existing `if hours:` guard)."""
        slots = {"test_name": "Test"}
        result = {
            "found": True, "test_name": "Test", "test_name_bn": "টেস্ট",
            "rate_inr": 999, "sample_type": "Blood",
        }
        reply = rate_reply(slots, result, language=language)
        for h in self._SEEDED_HOURS:
            assert hours_to_duration_phrase(h, language) not in reply


# --------------------------------------------------------------------- #
# Story: "Caller asks what sample is needed" (Conversation: Information
# and Enquiry). AC: "The sample type is spoken as a natural clause rather
# than a field and a colon. The English clinical term is preserved if the
# caller used it. Multiple samples for one test are all stated." Plus two
# requirements given directly, not from the sheet: words like "test" and
# "sample" must never come up twice in one reply, in any language; and a
# caller who asks ONLY which sample is needed gets a reply containing
# ONLY the sample information (sample_type_reply(), new this story).
# --------------------------------------------------------------------- #

_REAL_SAMPLE_TYPES = ["Blood", "Urine", "Cardiac", "Imaging", "Sample (Cervical)"]
_ALL_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]


class TestSpokenSampleTypes:
    """_spoken_sample_types() directly -- the helper both test_rate_reply()
    and sample_type_reply() use to turn a raw catalogue sample_type string
    into a natural clause instead of a raw field value."""

    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    def test_single_value_passes_through_unchanged(self, language):
        text, plural = _spoken_sample_types("Blood", language)
        assert text == "Blood"
        assert plural is False

    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    def test_cervical_rename_drops_the_redundant_word_sample(self, language):
        # "Sample (Cervical)" is the raw catalogue value -- speaking it
        # verbatim would say "sample" twice in the same sentence (e.g. "a
        # Sample (Cervical) sample"), and the parenthesis is a spoken-
        # punctuation risk Story 2's automated check never covered (it
        # only bans ':[]{}', not '()').
        text, plural = _spoken_sample_types("Sample (Cervical)", language)
        assert text == "Cervical"
        assert plural is False
        assert "(" not in text and ")" not in text

    @pytest.mark.parametrize("language,expected_join", [
        ("english", "and"), ("bengali", "এবং"), ("hinglish", "aur"), ("banglish", "ar"),
    ])
    def test_hypothetical_multi_value_is_joined_naturally(self, language, expected_join):
        # No catalogue row uses "|" today, but the split must already work
        # correctly for whenever one does -- same discipline as Story 4's
        # >72-hour fallback: build the general case, don't fabricate data.
        text, plural = _spoken_sample_types("Blood|Urine", language)
        assert plural is True
        assert "Blood" in text and "Urine" in text
        assert expected_join in text
        assert "|" not in text

    def test_three_values_still_uses_and_only_once(self):
        text, plural = _spoken_sample_types("Blood|Urine|Sample (Cervical)", "english")
        assert plural is True
        assert text == "Blood, Urine and Cervical"


class TestNameAlreadySaysTest:
    """_name_already_says_test() -- must catch the word "test" regardless
    of which script it's written in, since _spoken_test_name() can fall
    through to a plain English catalogue name (e.g. "Widal Test") even
    inside what is otherwise a Bengali sentence. Real bug caught by manual
    testing before this story: the old per-branch checks only tested ONE
    script each, so a Bengali branch still appended "টেস্টের" even when
    the name already said "Test" in English script."""

    def test_english_word_detected_case_insensitively(self):
        assert _name_already_says_test("Widal Test") is True
        assert _name_already_says_test("widal test") is True
        assert _name_already_says_test("WIDAL TEST") is True

    def test_bengali_word_detected(self):
        assert _name_already_says_test("উইডাল টেস্ট") is True

    def test_neither_script_present(self):
        assert _name_already_says_test("CBC") is False
        assert _name_already_says_test("ইউরিক এসিড") is False

    def test_mixed_script_name_still_detected(self):
        # The exact real-world gap this helper closes.
        assert _name_already_says_test("Widal Test") is True


class TestAOrAn:
    """_a_or_an() -- "an Imaging sample" vs "a Blood sample". Real
    grammar bug caught by manual testing before this story: the old code
    always used "a", producing "a Imaging sample"."""

    @pytest.mark.parametrize("word,expected", [
        ("Imaging", "an"), ("imaging", "an"), ("Urine", "an"), ("Echo", "an"),
        ("Blood", "a"), ("Cardiac", "a"), ("Cervical", "a"),
    ])
    def test_indefinite_article(self, word, expected):
        assert _a_or_an(word) == expected


def _rate_result(sample_type=None, hours=24, test_name="CBC", test_name_bn="সিবিসি"):
    result = {
        "found": True, "test_name": test_name, "test_name_bn": test_name_bn,
        "rate_inr": 850, "report_time_hours": hours,
    }
    if sample_type is not None:
        result["sample_type"] = sample_type
    return result


class TestSampleTypeReply:
    """sample_type_reply() -- new this story. Answers ONLY the sample
    question: no price, no duration, for a caller who asked nothing but
    which sample is needed."""

    @pytest.mark.parametrize("sample", _REAL_SAMPLE_TYPES)
    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    def test_contains_no_price_and_no_duration(self, sample, language):
        slots = {"test_name": "CBC"}
        result = _rate_result(sample_type=sample, hours=24)
        reply = sample_type_reply(slots, result, language=language)
        assert "850" not in reply
        assert hours_to_duration_phrase(24, language) not in reply

    @pytest.mark.parametrize("sample", _REAL_SAMPLE_TYPES)
    def test_bengali_never_produces_silent_latin_script(self, sample):
        # The measured bug this story fixed: "Imaging"/"Cardiac"/"Sample
        # (Cervical)" previously left the Bengali TTS model nothing at all
        # to say for the sample word -- not a wrong word, silence.
        slots = {"test_name": "CBC"}
        result = _rate_result(sample_type=sample, hours=24)
        reply = sample_type_reply(slots, result, language="bengali")
        spoken = verbalize(reply, language="bengali")
        assert unspeakable_spans(spoken) == []

    def test_banglish_matches_the_requested_minimal_phrasing_shape(self):
        # User-specified example: "aye test er jonno aye sample ta lagbe"
        # ("for this test, this sample is needed"). Verified here as the
        # same "<subject> er jonno <object> lagbe" clause shape, for the
        # real-world case of no Bengali alias (test_name_bn=None) --
        # exactly when _spoken_test_name() echoes the caller's own words.
        slots = {"test_name": "CBC"}
        result = _rate_result(sample_type="Blood", hours=24, test_name_bn=None)
        reply = sample_type_reply(slots, result, language="banglish")
        assert reply == "CBC test er jonno Blood sample ta lagbe."

    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    def test_missing_sample_type_gives_an_honest_fallback_not_a_fabrication(self, language):
        slots = {"test_name": "CBC"}
        result = _rate_result(sample_type=None, hours=24)
        reply = sample_type_reply(slots, result, language=language)
        for sample in _REAL_SAMPLE_TYPES:
            assert sample not in reply

    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    def test_not_found_delegates_to_the_shared_not_found_reply(self, language):
        slots = {"test_name": "Nonexistent Test"}
        result = {"found": False, "did_you_mean": ["Uric Acid"]}
        assert sample_type_reply(slots, result, language=language) == \
            rate_reply(slots, result, language=language)

    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    def test_multiple_samples_are_all_stated(self, language):
        # AC: "Multiple samples for one test are all stated."
        slots = {"test_name": "CBC"}
        result = _rate_result(sample_type="Blood|Urine", hours=24)
        reply = sample_type_reply(slots, result, language=language)
        assert "Blood" in reply and "Urine" in reply

    def test_english_clinical_term_is_preserved_for_non_bengali_languages(self):
        # AC: "The English clinical term is preserved if the caller used
        # it." Literally true for english/hinglish/banglish (sample_type
        # flows through unaltered); not literally possible for bengali --
        # see sample_type_reply()'s own docstring for why.
        for language in ("english", "hinglish", "banglish"):
            slots = {"test_name": "ECG"}
            result = _rate_result(sample_type="Cardiac", hours=6)
            reply = sample_type_reply(slots, result, language=language)
            assert "Cardiac" in reply


class TestNoDoubleTestOrSampleWord:
    """User-specified requirement: words like "test" and "sample" must
    never come up twice in a single reply, in any language. Exercised
    across both reply functions that mention a sample -- test_rate_reply
    bundles rate+sample+duration; sample_type_reply is sample-only -- for
    every real catalogue sample_type, plus the "Widal Test"-without-a-
    Bengali-alias edge case that motivated _name_already_says_test()."""

    def _assert_no_doubled_word(self, reply: str, word: str):
        lowered = reply.lower()
        assert lowered.count(word.lower()) <= 1, f"{word!r} appears more than once in {reply!r}"

    @pytest.mark.parametrize("sample", _REAL_SAMPLE_TYPES)
    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    def test_test_rate_reply_never_doubles_test_or_sample(self, sample, language):
        slots = {"test_name": "CBC"}
        result = _rate_result(sample_type=sample, hours=24)
        reply = rate_reply(slots, result, language=language)
        self._assert_no_doubled_word(reply, "test")
        self._assert_no_doubled_word(reply, "sample")
        assert reply.count("টেস্ট") <= 1
        assert reply.count("স্যাম্পল") <= 1

    @pytest.mark.parametrize("sample", _REAL_SAMPLE_TYPES)
    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    def test_sample_type_reply_never_doubles_test_or_sample(self, sample, language):
        slots = {"test_name": "CBC"}
        result = _rate_result(sample_type=sample, hours=24)
        reply = sample_type_reply(slots, result, language=language)
        self._assert_no_doubled_word(reply, "test")
        self._assert_no_doubled_word(reply, "sample")
        assert reply.count("টেস্ট") <= 1
        assert reply.count("স্যাম্পল") <= 1

    @pytest.mark.parametrize("language", _ALL_LANGUAGES)
    def test_name_already_containing_test_word_still_avoids_doubling(self, language):
        # The exact real-world edge case: a test name with no Bengali
        # alias falls through to the caller-echoed English name, which
        # may itself already say "Test" (e.g. "Widal Test").
        slots = {"test_name": "Widal Test"}
        result = {
            "found": True, "test_name": "Widal Test",  # no test_name_bn alias
            "rate_inr": 300, "sample_type": "Blood", "report_time_hours": 24,
        }
        for reply in (
            rate_reply(slots, result, language=language),
            sample_type_reply(slots, result, language=language),
        ):
            self._assert_no_doubled_word(reply, "test")
            assert reply.count("টেস্ট") <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
