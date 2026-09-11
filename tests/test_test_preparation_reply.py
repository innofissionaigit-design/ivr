"""ADDED BY SOURAV -- reply-template-level tests for "Caller asks how to
prepare for a test" (Epic: Conversation -- Information and Enquiry).

Mirrors tests/test_test_duration_reply.py's own structure exactly (see
that file's module docstring for the precedent this follows): a self-
contained new function gets its own dedicated test file rather than being
folded into the already-large tests/test_reply_templates_fidelity.py,
including that same file's spoken-punctuation-cleanliness discipline.

Three outcomes this locks in, matching test_preparation_reply()'s own
docstring in agent/reply_templates.py exactly:

  1. found=False -> shared not-found/did-you-mean reply (same helper every
     other test_* reply function already uses).
  2. found=True, advisory_available=False -> the honest "we don't have
     this yet" fallback -- NEVER a guessed "no special preparation
     needed". This is the caller-facing consequence of clinic-api/
     models.py's LabTest advisory columns being nullable with no default.
  3. found=True, advisory_available=True -> the business's own pre-
     written advisory_script_* spoken VERBATIM for the caller's language,
     with only the literal "{test_name}" placeholder substituted.
"""
from __future__ import annotations

import pytest

from agent.bn_normalize import verbalize
from agent.reply_templates import (
    test_preparation_reply as prep_reply,
    missing_slot_prompt,
)

_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]

_SCRIPT_FIELD = {
    "english": "advisory_script_en",
    "hinglish": "advisory_script_hinglish",
    "banglish": "advisory_script_banglish",
    "bengali": "advisory_script_bn",
}


def _advisory_result(**overrides):
    base = {
        "found": True, "test_name": "Blood Sugar Fasting",
        "test_name_bn": "ব্লাড সুগার ফাস্টিং",
        "advisory_available": True,
        "fasting_required": True,
        "fasting_hours": "8-12 hours",
        "water_allowance": "Only plain water permitted during fasting period",
        "medication_hold": "Hold morning anti-diabetic medication until after blood collection",
        "timing_rule": "Morning sample collection preferred",
        "advisory_script_en": "For {test_name}, you need to fast for 8 to 12 hours.",
        "advisory_script_hinglish": "{test_name} ke liye aapko 8 se 12 ghante khali pet rehna hoga.",
        "advisory_script_banglish": "{test_name}-er jonno apnake 8 theke 12 ghanta khali pete thakte hobe.",
        "advisory_script_bn": "{test_name}-এর জন্য আপনাকে ৮ থেকে ১২ ঘণ্টা খালি পেটে থাকতে হবে।",
    }
    base.update(overrides)
    return base


def _no_advisory_result(**overrides):
    base = {
        "found": True, "test_name": "ESR", "test_name_bn": "ইএসআর",
        "advisory_available": False,
    }
    base.update(overrides)
    return base


class TestAdvisoryAvailableSpeaksTheScriptVerbatim:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_reply_is_exactly_the_scripted_sentence_with_name_substituted(self, language):
        slots = {"test_name": "Blood Sugar Fasting"}
        result = _advisory_result()
        reply = prep_reply(slots, result, language=language)
        expected_name = result["test_name_bn"] if language == "bengali" else slots["test_name"]
        expected = result[_SCRIPT_FIELD[language]].replace("{test_name}", expected_name)
        assert reply == expected

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_placeholder_never_survives_into_the_spoken_reply(self, language):
        slots = {"test_name": "Blood Sugar Fasting"}
        result = _advisory_result()
        reply = prep_reply(slots, result, language=language)
        assert "{test_name}" not in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_never_recomposes_from_the_structured_fields(self, language):
        # The script is spoken exactly as the business supplied it -- this
        # function must never build its own sentence out of
        # fasting_hours/water_allowance/medication_hold/timing_rule.
        slots = {"test_name": "Blood Sugar Fasting"}
        result = _advisory_result(
            fasting_hours="99-100 hours",  # a value that would stick out if recomposed
            timing_rule="Some other timing rule entirely",
        )
        reply = prep_reply(slots, result, language=language)
        assert "99-100" not in reply
        assert "Some other timing rule entirely" not in reply

    def test_bengali_prefers_the_bengali_alias_for_substitution(self):
        slots = {"test_name": "Blood Sugar Fasting"}
        result = _advisory_result()
        reply = prep_reply(slots, result, language="bengali")
        assert "ব্লাড সুগার ফাস্টিং" in reply

    @pytest.mark.parametrize("language", ["english", "hinglish", "banglish"])
    def test_non_bengali_languages_never_leak_the_bengali_alias(self, language):
        slots = {"test_name": "Blood Sugar Fasting"}
        result = _advisory_result()
        reply = prep_reply(slots, result, language=language)
        assert "ব্লাড সুগার ফাস্টিং" not in reply


class TestAdvisoryUnavailableIsHonestNeverGuessed:
    """The whole reason clinic-api/models.py's LabTest advisory columns
    are nullable with no default: a test the business never reviewed must
    never be told "no special preparation needed" -- that is a specific,
    possibly-wrong medical claim, not a safe guess."""

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_found_but_unavailable_never_claims_no_preparation_needed(self, language):
        slots = {"test_name": "ESR"}
        result = _no_advisory_result()
        reply = prep_reply(slots, result, language=language)
        for phrase in ("no special", "no fasting", "not required", "lagbe nah", "lagta nahi", "দরকার নেই"):
            assert phrase.lower() not in reply.lower()

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_points_the_caller_at_a_human_instead(self, language):
        slots = {"test_name": "ESR"}
        result = _no_advisory_result()
        reply = prep_reply(slots, result, language=language)
        for word in ("counter", "doctor", "কাউন্টার", "ডাক্তার"):
            if word.lower() in reply.lower():
                break
        else:
            pytest.fail(f"expected a human-pointer word in {reply!r}")

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_no_structured_or_script_fields_leak_through(self, language):
        # Defensive: even if a caller-side bug somehow left structured
        # fields on the result alongside advisory_available=False, this
        # reply must not go hunting for them.
        slots = {"test_name": "ESR"}
        result = _no_advisory_result(fasting_required=False, fasting_hours="0 hours")
        reply = prep_reply(slots, result, language=language)
        assert "0 hours" not in reply

    def test_malformed_row_missing_this_languages_script_gets_the_same_honest_fallback(self):
        # Defensive-only branch: advisory_available=True but the specific
        # language's script column came back empty/None. Never speaks a
        # blank reply or raises -- falls back to the same honest wording
        # as advisory_available=False.
        slots = {"test_name": "Blood Sugar Fasting"}
        result = _advisory_result(advisory_script_en=None)
        reply = prep_reply(slots, result, language="english")
        assert reply  # non-empty
        assert "{test_name}" not in reply


class TestNotFoundDelegatesToTheSharedHelper:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_not_found_with_suggestions(self, language):
        slots = {"test_name": "Zzznonexistent"}
        result = {"found": False, "did_you_mean": ["Blood Sugar Fasting"]}
        reply = prep_reply(slots, result, language=language)
        assert "Blood Sugar Fasting" in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_not_found_without_suggestions(self, language):
        slots = {"test_name": "Zzznonexistent"}
        result = {"found": False, "did_you_mean": []}
        reply = prep_reply(slots, result, language=language)
        assert "Zzznonexistent" in reply


class TestSpokenPunctuationCleanliness:
    """Same automated check test_reply_templates_fidelity.py applies to
    every other reply function -- a spoken colon/bracket reads to a
    caller as a database field, not a sentence. Also specifically proves
    the "{test_name}" placeholder's own braces never survive, since this
    is the one reply function in this codebase that starts from a raw
    template string containing braces before substitution."""

    _BANNED = (":", "：", "[", "]", "{", "}")

    def _assert_clean(self, text: str, language: str):
        spoken = verbalize(text, language=language)
        for ch in self._BANNED:
            assert ch not in spoken, (
                f"spoken punctuation artefact {ch!r} survived verbalize() "
                f"for language={language!r}: {spoken!r}"
            )

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_advisory_available_reply_is_clean(self, language):
        slots = {"test_name": "Blood Sugar Fasting"}
        self._assert_clean(prep_reply(slots, _advisory_result(), language=language), language)

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_advisory_unavailable_reply_is_clean(self, language):
        slots = {"test_name": "ESR"}
        self._assert_clean(prep_reply(slots, _no_advisory_result(), language=language), language)

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_not_found_reply_is_clean(self, language):
        slots = {"test_name": "Zzznonexistent"}
        result = {"found": False, "did_you_mean": ["Blood Sugar Fasting"]}
        self._assert_clean(prep_reply(slots, result, language=language), language)


class TestMissingSlotPromptRegisteredForAllThreeLanguages:
    def test_english(self):
        assert missing_slot_prompt("test_preparation", "test_name", language="english") != \
            "Sorry, could you please clarify?"

    def test_hinglish(self):
        assert missing_slot_prompt("test_preparation", "test_name", language="hinglish") != \
            "Sorry, thoda clear kar sakte ho?"

    def test_bengali(self):
        assert missing_slot_prompt("test_preparation", "test_name", language="bengali") != \
            "দুঃখিত, একটু স্পষ্ট করে বলবেন?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
