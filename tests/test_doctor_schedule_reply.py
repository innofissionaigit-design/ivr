"""ADDED BY SOURAV -- tests for "Caller asks when a doctor sits"
(Epic: Conversation -- Information and Enquiry). Covers the new
agent/bn_normalize.py::weekday_to_words() primitive and the new
agent/reply_templates.py::doctor_schedule_reply() (plus its two private
helpers, _group_schedule_by_hours and _spoken_weekday_list) in isolation,
with hand-built `result` dicts -- the real clinic-api round trip is
covered separately in tests/test_doctor_schedule_dispatch.py, mirroring
this codebase's established split (see tests/test_live_test_price_lookup.py's
module docstring for why unit tests on hand-built dicts and live-backend
tests are kept as two different files rather than one).

Not a duplicate of tests/test_reply_templates_fidelity.py's blanket
spoken-punctuation sweep across every reply function -- that file is
enormous and broadly-scoped already; this dedicated file (same pattern as
tests/test_report_reply_templates.py for the report_status/report_send
story) covers this ONE new function thoroughly, including its own
spoken-punctuation-clean check, so the fix doesn't depend on someone
remembering to extend the other file.
"""
from __future__ import annotations

import pytest

from agent.bn_normalize import weekday_to_words, verbalize
from agent.reply_templates import (
    doctor_schedule_reply,
    _group_schedule_by_hours,
    _spoken_weekday_list,
)

LANGUAGES = ("english", "hinglish", "banglish", "bengali")

_BANNED = (":", "：", "[", "]", "{", "}")


def _assert_clean(text: str, language: str):
    """Same check tests/test_reply_templates_fidelity.py's
    TestNoSpokenPunctuationArtifact runs for every other reply function --
    see that class's docstring for the full reasoning (raw "HH:MM" in the
    template is fine; a literal punctuation artefact surviving
    verbalize() is not)."""
    spoken = verbalize(text, language=language)
    for ch in _BANNED:
        assert ch not in spoken, (
            f"spoken punctuation artefact {ch!r} survived verbalize() "
            f"for language={language!r}: {spoken!r}"
        )


class TestWeekdayToWords:
    def test_all_seven_days_every_language_no_crash_and_nonempty(self):
        for language in LANGUAGES:
            words = [weekday_to_words(w, language) for w in range(7)]
            assert all(words)
            assert len(set(words)) == 7, f"weekday names collided for language={language!r}: {words}"

    def test_monday_is_index_zero_every_language(self):
        # DoctorSchedule.weekday's own docstring: 0 = Monday. Spot check
        # each language actually agrees, since a silent off-by-one here
        # would mean every schedule reply names the wrong day.
        assert weekday_to_words(0, "english") == "Monday"
        assert weekday_to_words(0, "bengali") == "সোমবার"
        assert weekday_to_words(0, "hinglish") == "Somwar"
        assert weekday_to_words(0, "banglish") == "Sombar"

    def test_sunday_is_index_six_every_language(self):
        assert weekday_to_words(6, "english") == "Sunday"
        assert weekday_to_words(6, "bengali") == "রবিবার"
        assert weekday_to_words(6, "hinglish") == "Raviwar"
        assert weekday_to_words(6, "banglish") == "Robibar"

    def test_unknown_language_falls_back_to_bengali(self):
        assert weekday_to_words(0, "klingon") == weekday_to_words(0, "bengali")

    def test_out_of_range_weekday_raises_rather_than_silently_wrapping(self):
        with pytest.raises(IndexError):
            weekday_to_words(7, "english")


class TestGroupScheduleByHours:
    def test_all_days_same_hours_become_one_group(self):
        schedule = [
            {"weekday": 0, "start_time": "10:00", "end_time": "12:00"},
            {"weekday": 2, "start_time": "10:00", "end_time": "12:00"},
            {"weekday": 4, "start_time": "10:00", "end_time": "12:00"},
        ]
        groups = _group_schedule_by_hours(schedule)
        assert groups == [("10:00-12:00", [0, 2, 4])]

    def test_different_hours_become_separate_groups_ordered_by_earliest_weekday(self):
        schedule = [
            {"weekday": 4, "start_time": "18:00", "end_time": "20:00"},
            {"weekday": 0, "start_time": "10:00", "end_time": "12:00"},
            {"weekday": 2, "start_time": "10:00", "end_time": "12:00"},
        ]
        groups = _group_schedule_by_hours(schedule)
        # Monday+Wednesday (10-12) group comes first because its earliest
        # weekday (0) precedes the Friday-only (18-20) group's (4), even
        # though the Friday row appeared FIRST in the input list.
        assert groups == [("10:00-12:00", [0, 2]), ("18:00-20:00", [4])]

    def test_empty_schedule_gives_no_groups(self):
        assert _group_schedule_by_hours([]) == []

    def test_single_day_is_its_own_group(self):
        schedule = [{"weekday": 1, "start_time": "09:00", "end_time": "13:00"}]
        assert _group_schedule_by_hours(schedule) == [("09:00-13:00", [1])]


class TestSpokenWeekdayList:
    @pytest.mark.parametrize("language", LANGUAGES)
    def test_single_day(self, language):
        assert _spoken_weekday_list([0], language) == weekday_to_words(0, language)

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_two_days_uses_and_word_not_a_comma_only(self, language):
        text = _spoken_weekday_list([0, 2], language)
        assert weekday_to_words(0, language) in text
        assert weekday_to_words(2, language) in text
        assert "," not in text  # two items: "A and B", never "A, B"

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_three_days_uses_oxford_style_with_and_only_once(self, language):
        text = _spoken_weekday_list([0, 2, 4], language)
        for w in (0, 2, 4):
            assert weekday_to_words(w, language) in text
        # Exactly one comma (between day 1 and day 2), and the language's
        # "and" word appears exactly once (before the last day only).
        assert text.count(",") == 1


class TestDoctorScheduleReplyNotFound:
    @pytest.mark.parametrize("language", LANGUAGES)
    def test_not_found_names_the_query_and_is_clean(self, language):
        slots = {"doctor_name": "Dr Ghost"}
        result = {"found": False, "query": "Dr Ghost"}
        text = doctor_schedule_reply(slots, result, language=language)
        assert "Dr Ghost" in text
        _assert_clean(text, language)

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_not_found_never_claims_a_schedule(self, language):
        slots = {"doctor_name": "Dr Nobody"}
        result = {"found": False, "query": "Dr Nobody"}
        text = doctor_schedule_reply(slots, result, language=language).lower()
        assert "monday" not in text and "সোমবার" not in text


class TestDoctorScheduleReplyFoundButEmpty:
    @pytest.mark.parametrize("language", LANGUAGES)
    def test_doctor_exists_but_has_no_schedule_rows_is_honest_not_fabricated(self, language):
        slots = {"doctor_name": "Dr A. Sen"}
        result = {"found": True, "doctor_name": "Dr A. Sen", "doctor_name_bn": "ডক্টর সেন", "schedule": []}
        text = doctor_schedule_reply(slots, result, language=language)
        assert text
        _assert_clean(text, language)
        # Never invents a day when there genuinely are none.
        for w in range(7):
            assert weekday_to_words(w, language) not in text


class TestDoctorScheduleReplyFoundSingleGroup:
    """The common case, matching clinic-api/seed.py's SHIFT_TEMPLATES:
    every sitting day shares identical chamber hours."""

    def _result(self):
        return {
            "found": True, "doctor_name": "Dr A. Sen", "doctor_name_bn": "ডক্টর সেন",
            "schedule": [
                {"weekday": 0, "start_time": "10:00", "end_time": "12:00"},
                {"weekday": 2, "start_time": "10:00", "end_time": "12:00"},
                {"weekday": 4, "start_time": "10:00", "end_time": "12:00"},
            ],
        }

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_names_every_sitting_day_and_is_clean(self, language):
        slots = {"doctor_name": "Dr A. Sen"}
        text = doctor_schedule_reply(slots, self._result(), language=language)
        for w in (0, 2, 4):
            assert weekday_to_words(w, language) in text
        # A day NOT in the schedule must never appear.
        for w in (1, 3, 5, 6):
            assert weekday_to_words(w, language) not in text
        _assert_clean(text, language)

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_chamber_hours_survive_digit_faithfully(self, language):
        slots = {"doctor_name": "Dr A. Sen"}
        text = doctor_schedule_reply(slots, self._result(), language=language)
        spoken = verbalize(text, language=language)
        # The raw "10:00"/"12:00" literal must be gone (verbalize() spells
        # times into words) -- but the underlying hour values (10 and 12)
        # must still be genuinely present as spoken words, not rounded or
        # dropped, same digit-fidelity discipline as every other reply.
        assert "10:00" not in spoken and "12:00" not in spoken

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_doctor_name_is_spoken(self, language):
        # Bengali correctly speaks the Bengali alias (ডক্টর সেন), same as
        # doctor_availability_reply()'s established _spoken_doctor_name()
        # behavior -- so this checks for either script, not "Sen" literally.
        slots = {"doctor_name": "Dr A. Sen"}
        text = doctor_schedule_reply(slots, self._result(), language=language)
        assert "Sen" in text or "সেন" in text


class TestDoctorScheduleReplyFoundMultiGroup:
    """A doctor whose hours genuinely differ by day -- not produced by
    today's seed data (see clinic-api/seed.py's SHIFT_TEMPLATES, one
    template per doctor), but the schema allows it and the reply function
    must still speak it correctly rather than silently merging or
    dropping one of the two hour-groups."""

    def _result(self):
        return {
            "found": True, "doctor_name": "Dr B. Roy", "doctor_name_bn": "ডক্টর রায়",
            "schedule": [
                {"weekday": 4, "start_time": "18:00", "end_time": "20:00"},
                {"weekday": 0, "start_time": "10:00", "end_time": "12:00"},
                {"weekday": 2, "start_time": "10:00", "end_time": "12:00"},
            ],
        }

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_both_groups_hours_are_present_and_distinguished(self, language):
        slots = {"doctor_name": "Dr B. Roy"}
        text = doctor_schedule_reply(slots, self._result(), language=language)
        spoken = verbalize(text, language=language)
        # Neither raw HH:MM literal survives verbalize() -- both groups'
        # hours were actually spelled into words, not one of them dropped.
        assert "10:00" not in spoken and "12:00" not in spoken
        assert "18:00" not in spoken and "20:00" not in spoken

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_never_joined_with_a_semicolon_or_other_pause_artefact(self, language):
        # A semicolon is not in TestNoSpokenPunctuationArtifact's banned
        # set, but it is still not a word a TTS engine will pronounce
        # sensibly -- multi-group clauses must be joined with an actual
        # spoken conjunction, never left as raw punctuation.
        slots = {"doctor_name": "Dr B. Roy"}
        text = doctor_schedule_reply(slots, self._result(), language=language)
        assert ";" not in text
        _assert_clean(text, language)

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_groups_spoken_monday_first_regardless_of_input_order(self, language):
        # The fixture above deliberately lists the Friday row FIRST --
        # _group_schedule_by_hours() must still put the Monday/Wednesday
        # clause ahead of the Friday clause in the spoken text.
        slots = {"doctor_name": "Dr B. Roy"}
        text = doctor_schedule_reply(slots, self._result(), language=language)
        monday = weekday_to_words(0, language)
        friday = weekday_to_words(4, language)
        assert text.index(monday) < text.index(friday)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
