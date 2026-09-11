"""ADDED BY SOURAV -- "Caller asks opening hours, address or directions"
(Epic: Conversation -- Information and Enquiry). Reply-level coverage for
clinic_info_reply(), mirroring tests/test_health_package_reply.py's own
structure: found/not-found across all 4 languages, the same automated
spoken-punctuation-cleanliness check test_reply_templates_fidelity.py
already applies to every other reply function, plus this function's own
info_topic-narrowing behaviour (hours/address/directions/None) and its
"today" resolution contract with main.py (slots["today_weekday"] is
main.py's own resolved datetime.date.today().weekday() -- this function
must never guess a day when that value is missing or out of range).
"""
from __future__ import annotations

import pytest

from agent.bn_normalize import verbalize
from agent.reply_templates import clinic_info_reply

_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]


def _hours(**overrides) -> dict:
    week = {
        day: {"closed": False, "open": "08:00", "close": "20:00"}
        for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday")
    }
    week["sunday"] = {"closed": True, "open": None, "close": None}
    week.update(overrides)
    return week


def _found_result(**overrides) -> dict:
    base = {
        "found": True,
        "clinic_name": "Kolkata Care Polyclinic",
        "phone": "03322223333",
        "address": "42 Lake View Road, Kolkata, West Bengal 700029",
        "directions": "Opposite Lake View Metro Station, next to City Pharmacy.",
        "hours": _hours(),
    }
    base.update(overrides)
    return base


# Monday == weekday index 0, matches _CLINIC_WEEKDAY_KEYS' own Monday-first
# ordering (see clinic-api/main.py's _CLINIC_WEEKDAYS comment).
_MONDAY = 0
_SUNDAY = 6


class TestHoursTopic:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_open_day_states_open_and_close_time(self, language):
        slots = {"info_topic": "hours", "today_weekday": _MONDAY}
        reply = clinic_info_reply(slots, _found_result(), language=language)
        assert "08:00" in reply
        assert "20:00" in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_closed_day_is_stated_honestly_not_fabricated(self, language):
        # Sunday is seeded closed (sunday_open/close are None) -- this must
        # never be papered over with a guessed or carried-over time.
        slots = {"info_topic": "hours", "today_weekday": _SUNDAY}
        reply = clinic_info_reply(slots, _found_result(), language=language)
        assert "08:00" not in reply
        assert "20:00" not in reply
        assert "None" not in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_missing_today_weekday_is_an_honest_fallback_not_a_guess(self, language):
        slots = {"info_topic": "hours", "today_weekday": None}
        reply = clinic_info_reply(slots, _found_result(), language=language)
        assert "08:00" not in reply
        assert "20:00" not in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_out_of_range_today_weekday_is_also_an_honest_fallback(self, language):
        # Defensive: main.py's own weekday() call can never produce this,
        # but this function must not blow up (or silently mis-index) if
        # it ever receives a bad value some other caller passes in.
        slots = {"info_topic": "hours", "today_weekday": 9}
        reply = clinic_info_reply(slots, _found_result(), language=language)
        assert "08:00" not in reply
        assert "20:00" not in reply


class TestAddressTopic:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_reply_includes_the_full_address(self, language):
        slots = {"info_topic": "address", "today_weekday": _MONDAY}
        reply = clinic_info_reply(slots, _found_result(), language=language)
        assert "42 Lake View Road, Kolkata, West Bengal 700029" in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_address_topic_never_mentions_hours_or_directions(self, language):
        slots = {"info_topic": "address", "today_weekday": _MONDAY}
        reply = clinic_info_reply(slots, _found_result(), language=language)
        assert "08:00" not in reply
        assert "City Pharmacy" not in reply


class TestDirectionsTopic:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_reply_includes_the_full_directions_text(self, language):
        slots = {"info_topic": "directions", "today_weekday": _MONDAY}
        reply = clinic_info_reply(slots, _found_result(), language=language)
        assert "Opposite Lake View Metro Station, next to City Pharmacy." in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_directions_topic_never_mentions_hours_or_address(self, language):
        slots = {"info_topic": "directions", "today_weekday": _MONDAY}
        reply = clinic_info_reply(slots, _found_result(), language=language)
        assert "08:00" not in reply
        assert "Lake View Road" not in reply


class TestNoTopicSpeaksAllThreeTogether:
    """A caller who asks generally ("tell me about your clinic") or asks
    more than one of hours/address/directions in the same breath gets all
    three -- guessing which one to drop would silently lose information."""

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_hours_address_and_directions_are_all_present(self, language):
        slots = {"info_topic": None, "today_weekday": _MONDAY}
        reply = clinic_info_reply(slots, _found_result(), language=language)
        assert "08:00" in reply
        assert "42 Lake View Road, Kolkata, West Bengal 700029" in reply
        assert "Opposite Lake View Metro Station, next to City Pharmacy." in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_closed_sunday_still_reports_all_three_honestly(self, language):
        slots = {"info_topic": None, "today_weekday": _SUNDAY}
        reply = clinic_info_reply(slots, _found_result(), language=language)
        assert "08:00" not in reply  # closed today -- no fabricated hours
        assert "42 Lake View Road, Kolkata, West Bengal 700029" in reply
        assert "Opposite Lake View Metro Station, next to City Pharmacy." in reply


class TestNotFoundDelegatesHonestly:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_not_found_across_every_topic(self, language):
        for topic in ("hours", "address", "directions", None):
            slots = {"info_topic": topic, "today_weekday": _MONDAY}
            reply = clinic_info_reply(slots, {"found": False}, language=language)
            assert reply  # a real, honest sentence -- never a crash or blank string


class TestSpokenPunctuationCleanliness:
    """Same automated check test_reply_templates_fidelity.py applies to
    every other reply function -- a spoken colon or bracket reads to a
    caller as a database field, not a sentence."""

    _BANNED = (":", "：", "[", "]", "{", "}")

    def _assert_clean(self, text: str, language: str):
        spoken = verbalize(text, language=language)
        for ch in self._BANNED:
            assert ch not in spoken, (
                f"spoken punctuation artefact {ch!r} survived verbalize() "
                f"for language={language!r}: {spoken!r}"
            )

    @pytest.mark.parametrize("language", _LANGUAGES)
    @pytest.mark.parametrize("topic", ["hours", "address", "directions", None])
    def test_found_reply_is_clean(self, language, topic):
        slots = {"info_topic": topic, "today_weekday": _MONDAY}
        self._assert_clean(clinic_info_reply(slots, _found_result(), language=language), language)

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_not_found_reply_is_clean(self, language):
        slots = {"info_topic": None, "today_weekday": _MONDAY}
        self._assert_clean(clinic_info_reply(slots, {"found": False}, language=language), language)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
