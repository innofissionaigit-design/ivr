"""ADDED BY SOURAV -- fixes a real production bug, reported directly from
a live call transcript:

    [User] How long does it take to get the urine test report?
    [AI]   Urine test rate is 200 taka.
    [User] How long will it take to get the urine test report?
    [AI]   Urine test rate is 200 taka.

A caller asking about REPORT TURNAROUND TIME was being misclassified as
"test_rate" and answered with the test's PRICE instead. See
duration_reply()'s own docstring in agent/reply_templates.py for the
full root-cause writeup: rate_reply() was narrowed by an earlier
story ("Caller asks the price of a test") to speak ONLY the price, but
agent/llm.py's intent prompt was never updated to match, and
bn_normalize.hours_to_duration_phrase() (built by an even earlier story,
"Caller asks how long results take") was left with no caller-visible path
to reach it. This file covers the new reply function this fix adds --
duration_reply(), reusing hours_to_duration_phrase() exactly as it
already exists and is already unit-tested elsewhere (tests/test_bn_
normalize*.py), never rebuilding it.

Structured to mirror tests/test_test_sample_intent.py's sibling reply-
level coverage and test_reply_templates_fidelity.py's spoken-punctuation-
cleanliness discipline (the same automated colon/bracket check, applied
here rather than added to that already-large shared file, since this is a
self-contained new function with its own dedicated docstring/bug writeup).
"""
from __future__ import annotations

import pytest

from agent.bn_normalize import hours_to_duration_phrase, verbalize
# Aliased -- these names start with "test_", and pytest would otherwise
# try to collect the imported functions themselves as test items (see
# test_reply_templates_fidelity.py's own import of test_rate_reply for
# the same precedent).
from agent.reply_templates import (
    test_duration_reply as duration_reply,
    test_rate_reply as rate_reply,
    missing_slot_prompt,
)

_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]


def _result(hours=24, **overrides):
    base = {
        "found": True, "test_name": "Urine Routine Examination",
        "test_name_bn": "ইউরিন রুটিন পরীক্ষা",
        "rate_inr": "200", "sample_type": "Urine", "report_time_hours": hours,
    }
    base.update(overrides)
    return base


class TestTheReportedBugIsFixed:
    """Reproduces the exact transcript that was reported, at the
    reply-function level: given the SAME clinic-api response test_rate_
    reply() receives, duration_reply() must speak the DURATION, never
    the price, and rate_reply() must keep speaking ONLY the price
    (unchanged, regression guard)."""

    def test_duration_reply_never_mentions_the_price(self):
        slots = {"test_name": "Urine Routine Examination"}
        result = _result()
        reply = duration_reply(slots, result, language="english")
        assert "200" not in reply
        assert "rupee" not in reply.lower()

    def test_duration_reply_speaks_a_natural_duration_not_a_bare_hour_count(self):
        slots = {"test_name": "Urine Routine Examination"}
        result = _result(hours=24)
        reply = duration_reply(slots, result, language="english")
        assert reply == f"The Urine Routine Examination test report will be ready {hours_to_duration_phrase(24, 'english')}."
        assert "24" not in reply

    def test_rate_reply_is_unaffected_still_speaks_only_the_price(self):
        # Regression guard: this fix must not touch rate_reply()'s
        # own, deliberately narrowed scope.
        slots = {"test_name": "Urine Routine Examination"}
        result = _result()
        reply = rate_reply(slots, result, language="english")
        assert "200" in reply
        assert hours_to_duration_phrase(24, "english") not in reply


class TestFoundAcrossAllFourLanguages:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_reply_contains_the_duration_phrase(self, language):
        slots = {"test_name": "Urine Routine Examination"}
        result = _result(hours=24)
        reply = duration_reply(slots, result, language=language)
        assert hours_to_duration_phrase(24, language) in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    @pytest.mark.parametrize("hours", [3, 10, 24, 48, 72, 96])
    def test_every_duration_bucket_is_reachable(self, language, hours):
        # Digit-fidelity discipline applied to the bucket boundaries
        # themselves: the catalogue value still drives which phrase comes
        # out, nothing is guessed independently of report_time_hours.
        slots = {"test_name": "Urine Routine Examination"}
        result = _result(hours=hours)
        reply = duration_reply(slots, result, language=language)
        assert hours_to_duration_phrase(hours, language) in reply


class TestNotFoundDelegatesToTheSharedHelper:
    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish", "banglish"])
    def test_not_found_with_suggestions(self, language):
        slots = {"test_name": "Yuric Assid"}
        result = {"found": False, "did_you_mean": ["Uric Acid"]}
        reply = duration_reply(slots, result, language=language)
        assert "Uric Acid" in reply

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish", "banglish"])
    def test_not_found_without_suggestions(self, language):
        slots = {"test_name": "Nonexistent Test"}
        result = {"found": False, "did_you_mean": []}
        reply = duration_reply(slots, result, language=language)
        assert "Nonexistent Test" in reply


class TestMissingReportTimeIsHonest:
    """Never fabricates a duration that clinic-api never returned --
    mirrors sample_type_reply()'s own missing-sample honest fallback."""

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_missing_hours_gives_an_honest_fallback_not_a_guess(self, language):
        slots = {"test_name": "Urine Routine Examination"}
        result = _result(hours=None)
        reply = duration_reply(slots, result, language=language)
        assert "None" not in reply
        for bucket_phrase in (
            hours_to_duration_phrase(6, language),
            hours_to_duration_phrase(24, language),
        ):
            assert bucket_phrase not in reply


class TestNeverDoublesTheWordTest:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_name_already_containing_test_is_not_doubled(self, language):
        slots = {"test_name": "Widal Test"}
        result = _result(hours=24, test_name="Widal Test", test_name_bn="ওয়াইডাল টেস্ট")
        reply = duration_reply(slots, result, language=language)
        assert reply.lower().count("test") <= 1


class TestSpokenPunctuationCleanliness:
    """Same automated check test_reply_templates_fidelity.py applies to
    every other reply function -- a spoken colon/bracket reads to a caller
    as a database field, not a sentence."""

    _BANNED = (":", "：", "[", "]", "{", "}")

    def _assert_clean(self, text: str, language: str):
        spoken = verbalize(text, language=language)
        for ch in self._BANNED:
            assert ch not in spoken, (
                f"spoken punctuation artefact {ch!r} survived verbalize() "
                f"for language={language!r}: {spoken!r}"
            )

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_found_reply_is_clean(self, language):
        slots = {"test_name": "Urine Routine Examination"}
        result = _result(hours=24)
        self._assert_clean(duration_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_not_found_reply_is_clean(self, language):
        slots = {"test_name": "Yuric Assid"}
        result = {"found": False, "did_you_mean": ["Uric Acid"]}
        self._assert_clean(duration_reply(slots, result, language=language), language)

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_missing_hours_reply_is_clean(self, language):
        slots = {"test_name": "Urine Routine Examination"}
        result = _result(hours=None)
        self._assert_clean(duration_reply(slots, result, language=language), language)


class TestMissingSlotPromptRegisteredForAllThreeLanguages:
    def test_english(self):
        assert missing_slot_prompt("test_duration", "test_name", language="english") != \
            "Sorry, could you please clarify?"

    def test_hinglish(self):
        assert missing_slot_prompt("test_duration", "test_name", language="hinglish") != \
            "Sorry, thoda clear kar sakte ho?"

    def test_bengali(self):
        assert missing_slot_prompt("test_duration", "test_name", language="bengali") != \
            "দুঃখিত, একটু স্পষ্ট করে বলবেন?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
