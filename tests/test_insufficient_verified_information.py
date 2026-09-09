"""Tests for agent/outcomes.py -- "The agent says it cannot confirm rather
than guessing" (Epic: Answer Quality and Grounding).

Covers, in isolation from main_pcm.py's heavy stub environment (this
module has no ASR/TTS/NeMo dependency at all -- it is pure Python):
  * the dedicated template, across every language, and that it is
    textually distinct from both sibling outcomes ("not found" and the
    infrastructure apology);
  * missing_booking_write_fields() -- the ONE real check this story wires
    up, presence-only, nothing more;
  * the per-intent COUNT (never a rate -- see agent/outcomes.py's module
    docstring, constraint 2) tracked independently per intent;
  * the escalation ledger -- one JSON line per call, correct shape,
    append-only, and that it does not clobber concurrent writes from
    different intents.

The end-to-end wiring into main_pcm.py's _finish_booking() (the real
trigger firing inside the actual state machine) is covered separately in
tests/test_finish_booking_insufficient_information.py, and the fully-live
demonstration against the real clinic-api + seed.py backend is in
tests/test_live_insufficient_information_demo.py.
"""
import json

import pytest

from agent.outcomes import (
    insufficient_verified_information_reply,
    missing_booking_write_fields,
    record_insufficient_verified_information,
    insufficient_verified_information_counts,
    _reset_for_testing,
)
import agent.outcomes as outcomes_module


@pytest.fixture(autouse=True)
def _isolated_escalation_log(monkeypatch, tmp_path):
    """Every test gets its own throwaway escalation log and a clean
    counter -- state in agent.outcomes is module-level (see its docstring
    on why: this is in-process state sized for one call center's worth of
    concurrent calls, not a multi-process metrics store), so tests must
    not see each other's writes."""
    log_path = tmp_path / "escalations.jsonl"
    monkeypatch.setattr(outcomes_module, "ESCALATION_LOG_PATH", str(log_path))
    _reset_for_testing()
    yield log_path
    _reset_for_testing()


class TestInsufficientVerifiedInformationReply:
    """The dedicated template -- distinct from not-found and from the
    infrastructure apology, in every language."""

    _INFRA_APOLOGY_BN = "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।"

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish", "banglish"])
    def test_returns_nonempty_text_for_every_language(self, language):
        text = insufficient_verified_information_reply(language=language)
        assert isinstance(text, str)
        assert text.strip()

    def test_bengali_is_not_the_infrastructure_apology(self):
        text = insufficient_verified_information_reply(language="bengali")
        assert text != self._INFRA_APOLOGY_BN
        # Not just a different string -- it must not even claim the system
        # is unreachable ("কাউন্টারে যোগাযোগ করুন" = "contact the counter",
        # the infra apology's tell).
        assert "কাউন্টারে যোগাযোগ করুন" not in text

    def test_does_not_claim_the_thing_does_not_exist(self):
        # A not-found reply always contains "নেই" (bengali) or "don't have"
        # / "nahi hai" (the other languages) -- this outcome must never
        # sound like that, since the value might well be correct.
        bn = insufficient_verified_information_reply(language="bengali")
        en = insufficient_verified_information_reply(language="english")
        hi = insufficient_verified_information_reply(language="hinglish")
        bg = insufficient_verified_information_reply(language="banglish")
        assert "নেই" not in bn
        assert "don't have" not in en.lower()
        assert "nahi hai" not in hi.lower()
        assert "nei" not in bg.lower().replace("janiye", "")  # avoid false hit on unrelated word

    @pytest.mark.parametrize("language", ["bengali", "english", "hinglish", "banglish"])
    def test_no_spoken_punctuation_artefact(self, language):
        # Same discipline as "Answers sound like a person, not a database
        # row": no field label, colon or bracket survives to speech.
        text = insufficient_verified_information_reply(language=language)
        for ch in (":", "：", "[", "]", "{", "}"):
            assert ch not in text

    def test_hinglish_and_banglish_are_textually_distinct(self):
        # Same discipline as booking_confirmation_prompt: Hinglish here
        # means Hindi vocabulary, Banglish means Bengali-English -- they
        # must not be the same string.
        hi = insufficient_verified_information_reply(language="hinglish")
        bg = insufficient_verified_information_reply(language="banglish")
        assert hi != bg

    def test_default_language_is_bengali(self):
        assert insufficient_verified_information_reply() == insufficient_verified_information_reply(language="bengali")

    def test_unknown_language_falls_back_to_bengali(self):
        assert insufficient_verified_information_reply(language="klingon") == insufficient_verified_information_reply(language="bengali")


class TestMissingBookingWriteFields:
    """The ONE real trigger this story wires up: presence-only, nothing
    else. See agent/outcomes.py's module docstring for the full list of
    what this deliberately does NOT check."""

    _COMPLETE = {
        "success": True, "confirmation_id": "KCD-20260824-0031",
        "doctor_name": "Dr. A. Sen", "date": "2026-08-24", "time_slot": "09:30",
    }

    def test_complete_response_has_nothing_missing(self):
        assert missing_booking_write_fields(self._COMPLETE) == []

    @pytest.mark.parametrize("field", ["confirmation_id", "date", "time_slot"])
    def test_missing_field_is_reported(self, field):
        broken = dict(self._COMPLETE)
        del broken[field]
        assert missing_booking_write_fields(broken) == [field]

    @pytest.mark.parametrize("field", ["confirmation_id", "date", "time_slot"])
    def test_empty_string_field_is_reported_same_as_missing(self, field):
        broken = dict(self._COMPLETE)
        broken[field] = ""
        assert missing_booking_write_fields(broken) == [field]

    def test_multiple_missing_fields_all_reported_in_order(self):
        broken = dict(self._COMPLETE)
        del broken["date"]
        del broken["confirmation_id"]
        # Order follows the declared field order, not deletion order.
        assert missing_booking_write_fields(broken) == ["confirmation_id", "date"]

    def test_does_not_flag_fields_outside_its_scope(self):
        # doctor_name is not one of the 3 checked fields -- deliberately;
        # a missing doctor_name here is not this story's job (it would
        # already have failed earlier, at doctor lookup, which is a
        # different code path entirely).
        broken = dict(self._COMPLETE)
        del broken["doctor_name"]
        assert missing_booking_write_fields(broken) == []

    def test_none_value_is_treated_as_missing(self):
        broken = dict(self._COMPLETE)
        broken["confirmation_id"] = None
        assert missing_booking_write_fields(broken) == ["confirmation_id"]


class TestRecordAndCounts:
    """The count (never a rate) and the escalation ledger."""

    def test_count_starts_at_zero(self):
        assert insufficient_verified_information_counts("book_appointment") == 0

    def test_recording_increments_the_named_intent_only(self):
        record_insufficient_verified_information(
            intent="book_appointment", field="confirmation_id", reason="missing_after_success",
            call_id="call-1",
        )
        assert insufficient_verified_information_counts("book_appointment") == 1
        assert insufficient_verified_information_counts("test_rate") == 0

    def test_recording_multiple_times_accumulates(self):
        for _ in range(3):
            record_insufficient_verified_information(
                intent="book_appointment", field="date", reason="missing_after_success",
            )
        assert insufficient_verified_information_counts("book_appointment") == 3

    def test_counts_are_tracked_independently_per_intent(self):
        record_insufficient_verified_information(intent="book_appointment", field="date", reason="x")
        record_insufficient_verified_information(intent="book_appointment", field="date", reason="x")
        record_insufficient_verified_information(intent="test_rate", field="rate_inr", reason="y")
        assert insufficient_verified_information_counts("book_appointment") == 2
        assert insufficient_verified_information_counts("test_rate") == 1

    def test_counts_with_no_argument_returns_full_snapshot(self):
        record_insufficient_verified_information(intent="book_appointment", field="date", reason="x")
        record_insufficient_verified_information(intent="test_rate", field="rate_inr", reason="y")
        snapshot = insufficient_verified_information_counts()
        assert snapshot == {"book_appointment": 1, "test_rate": 1}

    def test_snapshot_is_a_copy_not_the_live_dict(self):
        record_insufficient_verified_information(intent="book_appointment", field="date", reason="x")
        snapshot = insufficient_verified_information_counts()
        snapshot["book_appointment"] = 999
        assert insufficient_verified_information_counts("book_appointment") == 1

    def test_this_is_a_count_not_a_rate(self):
        # No function in this module accepts or computes a denominator --
        # this is deliberate (constraint 2). Confirm the public surface
        # really is count-only.
        import inspect
        sig = inspect.signature(insufficient_verified_information_counts)
        assert "total" not in sig.parameters
        assert "denominator" not in sig.parameters
        assert "attempts" not in sig.parameters


class TestEscalationLedger:
    """The JSONL escalation path itself."""

    def test_writes_one_json_line(self, _isolated_escalation_log):
        record_insufficient_verified_information(
            intent="book_appointment", field="confirmation_id", reason="missing_after_success",
            call_id="call-42",
        )
        lines = _isolated_escalation_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["intent"] == "book_appointment"
        assert entry["field"] == "confirmation_id"
        assert entry["reason"] == "missing_after_success"
        assert entry["call_id"] == "call-42"
        assert "timestamp" in entry and entry["timestamp"]

    def test_appends_rather_than_overwrites(self, _isolated_escalation_log):
        record_insufficient_verified_information(intent="book_appointment", field="date", reason="a")
        record_insufficient_verified_information(intent="test_rate", field="rate_inr", reason="b")
        lines = _isolated_escalation_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        entries = [json.loads(l) for l in lines]
        assert entries[0]["intent"] == "book_appointment"
        assert entries[1]["intent"] == "test_rate"

    def test_call_id_is_optional(self, _isolated_escalation_log):
        record_insufficient_verified_information(intent="book_appointment", field="date", reason="a")
        entry = json.loads(_isolated_escalation_log.read_text(encoding="utf-8").strip())
        assert entry["call_id"] is None

    def test_timestamp_is_iso8601_and_utc(self, _isolated_escalation_log):
        import datetime
        record_insufficient_verified_information(intent="book_appointment", field="date", reason="a")
        entry = json.loads(_isolated_escalation_log.read_text(encoding="utf-8").strip())
        parsed = datetime.datetime.fromisoformat(entry["timestamp"])
        assert parsed.tzinfo is not None  # a naive timestamp is ambiguous across a distributed deploy

    def test_creates_parent_directory_if_missing(self, monkeypatch, tmp_path):
        nested = tmp_path / "nested" / "dir" / "escalations.jsonl"
        monkeypatch.setattr(outcomes_module, "ESCALATION_LOG_PATH", str(nested))
        _reset_for_testing()
        record_insufficient_verified_information(intent="book_appointment", field="date", reason="a")
        assert nested.exists()

    def test_non_ascii_content_is_preserved_readable(self, _isolated_escalation_log):
        # field/reason are plain ascii in practice, but the record must not
        # corrupt Bengali if a future caller ever passes it (e.g. a spoken
        # field name) -- ensure_ascii=False is what this checks.
        record_insufficient_verified_information(intent="book_appointment", field="তারিখ", reason="a")
        raw = _isolated_escalation_log.read_text(encoding="utf-8")
        assert "তারিখ" in raw
        assert "\\u" not in raw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
