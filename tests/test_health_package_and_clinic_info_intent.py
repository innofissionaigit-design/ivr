"""ADDED BY SOURAV -- "Caller asks about a health package" combined with
"Caller asks opening hours, address or directions" (Epic: Conversation --
Information and Enquiry). Covers the two non-reply-template places these
new intents touch, mirroring tests/test_test_duration_intent.py's own
structure (its module docstring documents the same 3-place pattern every
new intent in this codebase touches -- llm.py, semantic_cache.py, main.py
dispatch -- this file covers the first two; dispatch coverage lives in
tests/test_health_package_and_clinic_info_dispatch.py instead).

Two real design choices these tests exist specifically to lock in (see
agent/llm.py's own comment on VALID_INTENTS and agent/semantic_cache.py's
own comment on _ENTITY_SLOTS for the full reasoning):

  1. "package_name" is OPTIONAL for "health_package" -- unlike every other
     single-entity intent (test_rate/doctor_schedule/doctors_by_department,
     all of which re-prompt when their one slot is missing), a caller
     asking "what packages do you have" with no package named is a
     complete, valid extraction, not an incomplete one.
  2. "health_package" is therefore deliberately NOT registered in
     _REQUIRED_ENTITY_FOR_INTENT (unlike test_rate/doctor_schedule/
     doctors_by_department, which all are) -- a missing package_name must
     stay L2-cache-eligible, since it is a complete answer, not a
     dangling extraction.
"""
from __future__ import annotations

from agent.llm import VALID_INTENTS, _validate
from agent.semantic_cache import _ENTITY_SLOTS, _REQUIRED_ENTITY_FOR_INTENT, SemanticCache


def _slots(**overrides) -> dict:
    base = {
        "test_name": None, "doctor_name": None, "department": None,
        "date": None, "time_slot": None, "patient_name": None, "phone": None,
        "package_name": None, "info_topic": None,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------- #
# agent/llm.py -- "health_package"
# --------------------------------------------------------------------- #

class TestHealthPackageIsAValidIntent:
    def test_registered_in_valid_intents(self):
        assert "health_package" in VALID_INTENTS

    def test_distinct_from_test_rate(self):
        # A package ("Diabetes Screening Package") is a different
        # catalogue row from a single lab test ("Blood Sugar Fasting") --
        # this intent must never collapse into test_rate's own slot.
        assert "health_package" != "test_rate"

    def test_validate_accepts_a_named_package_payload(self):
        data = {
            "intent": "health_package",
            "slots": _slots(package_name="Diabetes Screening Package"),
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_accepts_no_package_named_at_all(self):
        # The whole point of the design choice: "what packages do you
        # have" is a COMPLETE, valid extraction with package_name=None,
        # not a validation failure.
        data = {
            "intent": "health_package",
            "slots": _slots(package_name=None),
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is True, errors


# --------------------------------------------------------------------- #
# agent/llm.py -- "clinic_info"
# --------------------------------------------------------------------- #

class TestClinicInfoIsAValidIntent:
    def test_registered_in_valid_intents(self):
        assert "clinic_info" in VALID_INTENTS

    def test_validate_accepts_each_real_info_topic(self):
        for topic in ("hours", "address", "directions"):
            data = {
                "intent": "clinic_info",
                "slots": _slots(info_topic=topic),
                "direct_reply_bn": None,
            }
            ok, errors = _validate(data)
            assert ok is True, (topic, errors)

    def test_validate_accepts_no_topic_at_all(self):
        # A caller asking generally ("tell me about your clinic") or
        # asking more than one of hours/address/directions at once --
        # info_topic=None is a complete, valid extraction too.
        data = {
            "intent": "clinic_info",
            "slots": _slots(info_topic=None),
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is True, errors


class TestValidateRejectsUnrelatedInvalidIntent:
    def test_an_intent_not_in_the_enum_is_still_rejected(self):
        # Regression guard: adding two new intents must not have loosened
        # _validate() into accepting arbitrary strings.
        data = {
            "intent": "health_advice",
            "slots": _slots(),
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is False
        assert any("invalid intent" in e for e in errors)


# --------------------------------------------------------------------- #
# agent/semantic_cache.py
# --------------------------------------------------------------------- #

class TestPackageNameEntitySlot:
    def test_package_name_is_registered_as_an_entity_slot(self):
        assert "package_name" in _ENTITY_SLOTS

    def test_health_package_is_deliberately_not_a_required_entity(self):
        # See this file's own module docstring / agent/semantic_cache.py's
        # own comment on _ENTITY_SLOTS for why: unlike test_rate/
        # doctor_schedule/doctors_by_department, a missing package_name is
        # a complete answer ("list everything"), not an incomplete one.
        assert "health_package" not in _REQUIRED_ENTITY_FOR_INTENT

    def test_clinic_info_has_no_required_entity_either(self):
        assert "clinic_info" not in _REQUIRED_ENTITY_FOR_INTENT


class TestHealthPackageL2Eligibility:
    def test_extraction_with_a_named_package_is_l2_eligible(self):
        value = {"intent": "health_package", "slots": {"package_name": "Diabetes Screening Package"}}
        assert SemanticCache._is_l2_eligible(value) is True

    def test_extraction_with_no_package_named_is_ALSO_l2_eligible(self):
        # The key behavioural difference from test_rate/doctor_schedule/
        # doctors_by_department: those intents' own missing-entity
        # extractions are explicitly EXCLUDED from L2 (see
        # _REQUIRED_ENTITY_FOR_INTENT's own docstring on the cache-
        # poisoning bug this guards against). health_package's own
        # "list everything" answer carries no such risk -- it is the same
        # complete answer for every caller, regardless of phrasing.
        value = {"intent": "health_package", "slots": {"package_name": None}}
        assert SemanticCache._is_l2_eligible(value) is True

    def test_clinic_info_extraction_is_l2_eligible_with_or_without_a_topic(self):
        for topic in (None, "hours", "address", "directions"):
            value = {"intent": "clinic_info", "slots": {"info_topic": topic}}
            assert SemanticCache._is_l2_eligible(value) is True


class TestEntityGuardForPackageName:
    """Mirrors doctors_by_department's own entity-guard coverage
    (department is the closest existing precedent for a catalogue-name
    entity slot) -- a fuzzy cache hit must not be reused across two
    DIFFERENT named packages just because the sentence frame rhymes."""

    def test_a_hit_naming_the_same_package_the_new_utterance_mentions_passes(self):
        value = {"intent": "health_package", "slots": {"package_name": "Diabetes Screening Package"}}
        assert SemanticCache._entity_guard(value, "diabetes screening package koto") is True

    def test_a_hit_naming_a_different_package_than_the_new_utterance_is_rejected(self):
        value = {"intent": "health_package", "slots": {"package_name": "Diabetes Screening Package"}}
        assert SemanticCache._entity_guard(value, "full body health package koto") is False

    def test_a_hit_with_no_package_named_at_all_passes_trivially(self):
        # No entity to get wrong -- same as smalltalk/unclear/an unfilled
        # slot elsewhere (see _entity_guard's own docstring).
        value = {"intent": "health_package", "slots": {"package_name": None}}
        assert SemanticCache._entity_guard(value, "what packages do you have") is True
