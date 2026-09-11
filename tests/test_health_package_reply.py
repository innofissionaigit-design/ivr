"""ADDED BY SOURAV -- "Caller asks about a health package" (Epic:
Conversation -- Information and Enquiry). Reply-level coverage for
health_package_reply() (one named package) and health_packages_list_reply()
(no package named -- "what packages do you have"), mirroring
tests/test_test_duration_reply.py's own structure: found/not-found across
all 4 languages, the same automated spoken-punctuation-cleanliness check
test_reply_templates_fidelity.py already applies to every other reply
function, and digit-fidelity for price_inr (reusing test_rate_reply()'s
own _digit_faithful_rate() fix rather than re-solving it).
"""
from __future__ import annotations

import pytest

from agent.bn_normalize import verbalize
from agent.reply_templates import health_package_reply, health_packages_list_reply

_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]


def _found_result(**overrides) -> dict:
    base = {
        "found": True,
        "package_name": "Diabetes Screening Package",
        "package_name_bn": "diabetes checkup",
        "description": "Basic diabetes-focused screening package.",
        "price_inr": "1299.0",
        "tests": ["Blood Sugar Fasting", "Blood Sugar PP", "HbA1c"],
        "tests_bn": ["blood sugar fasting", "blood sugar pp", "hba1c"],
    }
    base.update(overrides)
    return base


class TestFoundAcrossAllFourLanguages:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_reply_mentions_the_price_digit_faithfully(self, language):
        result = _found_result(price_inr="1299.0")
        reply = health_package_reply({}, result, language=language)
        assert "1299" in reply
        # UPDATED-style digit fidelity: a whole-rupee amount's trailing
        # ".0" must never survive into the spoken sentence (same bug
        # class _digit_faithful_rate() was built for on rate_inr).
        assert "1299.0" not in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_every_included_test_is_named(self, language):
        result = _found_result()
        reply = health_package_reply({}, result, language=language)
        if language == "bengali":
            for name in result["tests_bn"]:
                assert name in reply
        else:
            for name in result["tests"]:
                assert name in reply

    def test_english_reply_includes_the_description(self):
        result = _found_result()
        reply = health_package_reply({}, result, language="english")
        assert "diabetes-focused screening" in reply

    @pytest.mark.parametrize("language", ["bengali", "hinglish", "banglish"])
    def test_non_english_replies_never_inject_the_raw_english_description(self, language):
        # FLAGGED, real gap (see health_package_reply()'s own docstring):
        # description is English-only in the database. This test locks in
        # the deliberate choice not to speak it outside English, rather
        # than silently letting untranslated prose leak into a Bengali/
        # Hinglish/Banglish sentence.
        result = _found_result()
        reply = health_package_reply({}, result, language=language)
        assert "diabetes-focused screening" not in reply

    def test_bengali_reply_uses_the_bengali_alias_for_the_package_name(self):
        result = _found_result()
        reply = health_package_reply({}, result, language="bengali")
        assert "diabetes checkup" in reply
        assert "Diabetes Screening Package" not in reply

    @pytest.mark.parametrize("language", ["english", "hinglish", "banglish"])
    def test_non_bengali_replies_never_use_the_bengali_alias(self, language):
        result = _found_result()
        reply = health_package_reply({}, result, language=language)
        assert "diabetes checkup" not in reply
        assert "Diabetes Screening Package" in reply


class TestNotFoundDelegatesHonestly:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_not_found_with_suggestions(self, language):
        slots = {"package_name": "Diabets Package"}
        result = {"found": False, "query": "Diabets Package", "did_you_mean": ["Diabetes Screening Package"]}
        reply = health_package_reply(slots, result, language=language)
        assert "Diabetes Screening Package" in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_not_found_without_suggestions(self, language):
        slots = {"package_name": "Nonexistent Package"}
        result = {"found": False, "query": "Nonexistent Package", "did_you_mean": []}
        reply = health_package_reply(slots, result, language=language)
        assert "Nonexistent Package" in reply


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
    def test_found_reply_is_clean(self, language):
        self._assert_clean(health_package_reply({}, _found_result(), language=language), language)

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_not_found_reply_is_clean(self, language):
        result = {"found": False, "query": "Yuric Assid", "did_you_mean": ["Diabetes Screening Package"]}
        self._assert_clean(health_package_reply({"package_name": "Yuric Assid"}, result, language=language), language)

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_list_reply_is_clean(self, language):
        result = {"packages": [_found_result(), _found_result(package_name="Basic Health Checkup",
                                                                package_name_bn="basic checkup",
                                                                price_inr="999.0")]}
        self._assert_clean(health_packages_list_reply(result, language=language), language)


class TestHealthPackagesListReply:
    """"What health packages do you have?" -- clinic-api/models.py's own
    HealthPackage docstring gives this as the FIRST example query, backed
    by package_name being left null (see agent/llm.py's own comment on
    VALID_INTENTS)."""

    def _two_packages(self):
        return {
            "packages": [
                _found_result(),
                _found_result(
                    package_name="Basic Health Checkup", package_name_bn="basic checkup",
                    price_inr="999.0", tests=["Complete Blood Count (CBC)"], tests_bn=["cbc"],
                ),
            ]
        }

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_every_package_name_is_present(self, language):
        result = self._two_packages()
        reply = health_packages_list_reply(result, language=language)
        if language == "bengali":
            assert "diabetes checkup" in reply
            assert "basic checkup" in reply
        else:
            assert "Diabetes Screening Package" in reply
            assert "Basic Health Checkup" in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_every_price_is_digit_faithful(self, language):
        result = self._two_packages()
        reply = health_packages_list_reply(result, language=language)
        assert "1299" in reply
        assert "999" in reply
        assert "1299.0" not in reply
        assert "999.0" not in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_empty_catalogue_is_an_honest_fallback_not_a_crash(self, language):
        reply = health_packages_list_reply({"packages": []}, language=language)
        assert reply  # a real sentence, not an empty string or a KeyError


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
