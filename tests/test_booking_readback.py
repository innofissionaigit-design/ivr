"""Tests for story "Every critical value is read back before it is used"
(Answer Quality and Grounding, owner: Saurav).

Acceptance criteria under test (verbatim from the sprint sheet):
  1. Phone numbers, dates, times and names are confirmed aloud before any
     write.
  2. A rejection opens a correction path rather than repeating the prompt.
  3. Readback is mandatory regardless of confidence for values that
     affect a write.

Layout:
  - TestBookingConfirmationPromptFidelity / TestBookingCorrectionPrompt:
    pure template tests (agent/reply_templates.py), same style as the
    existing test_reply_templates_fidelity.py.
  - TestParseCorrectionField: pure parser tests (agent/slot_parse.py).
  - TestConfirmBookingState / TestConfirmCorrectionState /
    TestNoWriteBeforeConfirmation / TestSingleShotBookingAlsoConfirms:
    state-machine regression tests against main_pcm._continue_pending and
    main_pcm._dispatch_turn, with the tool client and _speak stubbed --
    deterministic, no network, no GPU (see tests/conftest.py for why
    main_pcm.py itself imports cleanly without the real ASR/VAD/LLM
    stack).

No pytest-asyncio dependency: the project's requirements.txt pins only
`pytest`, so async entry points are driven with plain asyncio.run() inside
ordinary sync test functions rather than adding a new test-only package.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from agent.reply_templates import (
    booking_confirmation_prompt, booking_correction_prompt, _spoken_doctor_name,
)
from agent.slot_parse import parse_correction_field, is_affirmative, is_negative

import main_pcm


def run(coro):
    """Drive a single coroutine to completion without pytest-asyncio."""
    return asyncio.run(coro)


# --------------------------------------------------------------------- #
# Pure template tests
# --------------------------------------------------------------------- #

SLOTS = {
    # "Dr. A. Sen" -- matches clinic-api/seed.py's actual convention
    # (every seeded doctor's canonical name already carries "Dr."), not a
    # simplified test fixture. Chosen deliberately after a real bug was
    # caught here: an earlier version of _spoken_doctor_name() prepended
    # a SECOND "Dr." for the english/hinglish/banglish branches, producing
    # "Dr. Dr. A. Sen" -- a plain "Sen" fixture would never have caught
    # that.
    "doctor_name": "Dr. A. Sen",
    "date": "2026-09-15",
    "time_slot": "18:30",
    "patient_name": "Rina Das",
    "phone": "9831234567",
}

ALL_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]


class TestBookingConfirmationPromptFidelity:
    """Mirrors test_reply_templates_fidelity.py's discipline: every value
    must appear byte-identical in the prompt, in every supported
    language -- no rounding, truncation or reformatting of the digits."""

    @pytest.mark.parametrize("language", ALL_LANGUAGES)
    def test_contains_every_critical_value_verbatim(self, language):
        prompt = booking_confirmation_prompt(SLOTS, language=language)
        for field, value in SLOTS.items():
            if field == "doctor_name":
                continue  # covered separately: honorific may legitimately change
            assert value in prompt, f"{value!r} missing from {language} readback: {prompt!r}"

    @pytest.mark.parametrize("language", ALL_LANGUAGES)
    def test_doctor_name_never_doubles_its_honorific(self, language):
        # Regression test for the exact bug caught during review: seed
        # data doctor names already start with "Dr." (clinic-api/seed.py),
        # so _spoken_doctor_name() must never prepend a second one.
        prompt = booking_confirmation_prompt(SLOTS, language=language)
        assert "Dr. Dr." not in prompt
        assert "ডাঃ Dr." not in prompt

    def test_phone_digits_are_not_reformatted(self):
        # No dashes, spaces or grouping inserted -- verbalize() downstream
        # (agent/bn_normalize.py, called from agent/tts.py) is what turns
        # this into digit-by-digit speech; the template must hand it the
        # raw string unchanged.
        prompt = booking_confirmation_prompt(SLOTS)
        assert "9831234567" in prompt
        assert "983-123-4567" not in prompt
        assert "98312 34567" not in prompt

    def test_date_stays_iso_for_downstream_verbalization(self):
        # bn_normalize.verbalize()'s _RE_DATE only matches YYYY-MM-DD --
        # pre-formatting the date here would silently break that regex
        # and the date would come out as three unrelated numbers instead
        # of a spoken date.
        prompt = booking_confirmation_prompt(SLOTS)
        assert "2026-09-15" in prompt

    def test_missing_fields_do_not_crash_and_render_empty(self):
        # Defensive: a caller of this function with a partially-empty
        # dict (should never happen once _next_missing() has returned
        # None, but the function must not raise if it ever does).
        prompt = booking_confirmation_prompt({})
        assert isinstance(prompt, str)

    def test_asks_a_yes_no_question(self):
        # Loose sanity check that this actually reads as a confirmation
        # ask, not a statement -- every language branch ends on a question.
        for language in ALL_LANGUAGES:
            prompt = booking_confirmation_prompt(SLOTS, language=language)
            assert "?" in prompt

    def test_banglish_is_not_hinglish_text(self):
        # The two must not collapse into the same wording -- "hinglish"
        # here is Hindi vocabulary ("hai", "karne", "kya"); "banglish" is
        # transliterated Bengali ("kore nin", "thik ache"). A caller who
        # code-switches Bengali+English should not be handed a Hindi
        # sentence just because both are "regional language + English".
        hinglish = booking_confirmation_prompt(SLOTS, language="hinglish")
        banglish = booking_confirmation_prompt(SLOTS, language="banglish")
        assert hinglish != banglish
        # Loose vocabulary check in each direction.
        assert "hai" not in banglish.lower()
        assert "ache" not in hinglish.lower()


class TestSpokenDoctorNameLanguageAwareness:
    """_spoken_doctor_name() directly, across all four languages -- this
    is the helper booking_confirmation_prompt() (and the pre-existing
    booking_reply()/doctor_availability_reply()) all rely on to get the
    doctor's name right."""

    def test_bengali_prefers_alias_with_honorific(self):
        assert _spoken_doctor_name(
            {"doctor_name": "Dr. A. Sen"}, {"doctor_name_bn": "সেন"}, language="bengali"
        ) == "ডাঃ সেন"

    def test_bengali_without_alias_falls_back_to_raw_name_unchanged(self):
        # This is the exact, byte-for-byte behaviour the ORIGINAL
        # function had for its only real-world call pattern (Bengali,
        # no language argument) -- must not regress.
        assert _spoken_doctor_name({"doctor_name": "Dr. A. Sen"}, {}, language="bengali") \
            == "Dr. A. Sen"

    @pytest.mark.parametrize("language", ["english", "hinglish", "banglish"])
    def test_non_bengali_uses_dr_honorific_not_the_bengali_prefix(self, language):
        name = _spoken_doctor_name({"doctor_name": "Amit Sen"}, {}, language=language)
        assert name == "Dr. Amit Sen"
        assert "ডাঃ" not in name

    @pytest.mark.parametrize("language", ["english", "hinglish", "banglish"])
    def test_non_bengali_does_not_double_an_existing_dr(self, language):
        for name in ("Dr. A. Sen", "dr. a. sen", "DR. A. SEN"):
            result = _spoken_doctor_name({"doctor_name": name}, {}, language=language)
            assert result.lower().count("dr.") == 1, f"{name!r} -> {result!r}"

    @pytest.mark.parametrize("language,expected", [
        ("bengali", "ডাক্তার"), ("english", "the doctor"),
        ("hinglish", "doctor"), ("banglish", "doctor"),
    ])
    def test_fallback_when_no_name_at_all(self, language, expected):
        assert _spoken_doctor_name({}, {}, language=language) == expected

    def test_non_bengali_ignores_bengali_alias_even_if_present(self):
        # A pure-Bengali-script alias ("সেন") would be exactly as
        # unreadable in an English sentence as "ডাঃ" is -- the English
        # branch must use the Latin name, not the alias, even when both
        # are available.
        name = _spoken_doctor_name(
            {"doctor_name": "Dr. A. Sen"}, {"doctor_name_bn": "সেন"}, language="english"
        )
        assert name == "Dr. A. Sen"
        assert "সেন" not in name


class TestAffirmativeNegativeAcrossLanguages:
    """is_affirmative()/is_negative() gate the entire confirm_booking
    state (see main_pcm.py) -- if these only understood Bengali, an
    English/Hinglish/Banglish caller's "yes" or "no" would silently fall
    through to the unparseable-reply retry path instead of being acted on
    immediately, regardless of which language the AGENT happened to speak
    the prompt in."""

    @pytest.mark.parametrize("text", [
        "হ্যাঁ", "ঠিক আছে",  # bengali (must still work -- no regression)
        "yes", "Yes", "YES", "yeah", "correct", "ok", "sure",  # english, case-insensitive
        "haan", "Haan", "thik ache", "sahi hai",  # hinglish/banglish
    ])
    def test_recognized_as_affirmative(self, text):
        assert is_affirmative(text) is True

    @pytest.mark.parametrize("text", [
        "না", "থাক",  # bengali (must still work -- no regression)
        "no", "No", "NO", "nope", "wrong", "incorrect",  # english, case-insensitive
        "nahi", "Nahi", "na", "galat",  # hinglish/banglish
    ])
    def test_recognized_as_negative(self, text):
        assert is_negative(text) is True

    @pytest.mark.parametrize("text", ["ফোন নম্বর", "Rina Das", "18:30", "", "maybe", "hmm"])
    def test_unrelated_text_is_neither(self, text):
        assert is_affirmative(text) is False
        assert is_negative(text) is False


class TestBookingCorrectionPrompt:
    def test_is_not_the_same_text_as_the_confirmation_prompt(self):
        # AC: "opens a correction path rather than repeating the prompt."
        # Directly encodes that these must never be textually identical.
        confirmation = booking_confirmation_prompt(SLOTS)
        correction = booking_correction_prompt()
        assert confirmation != correction

    @pytest.mark.parametrize("language", ALL_LANGUAGES)
    def test_offers_all_five_fields_as_options(self, language):
        prompt = booking_correction_prompt(language=language).lower()
        # Loose per-language keyword check -- the point is the caller is
        # given a specific menu, not a vague "please repeat".
        expected = {
            "bengali": ["ডাক্তার", "তারিখ", "সময়", "নাম", "ফোন"],
            "english": ["doctor", "date", "time", "name", "phone"],
            "hinglish": ["doctor", "date", "time", "naam", "phone"],
            "banglish": ["doctor", "date", "time", "naam", "phone"],
        }[language]
        for word in expected:
            assert word.lower() in prompt


# --------------------------------------------------------------------- #
# Pure parser tests
# --------------------------------------------------------------------- #

class TestParseCorrectionField:
    @pytest.mark.parametrize("text,expected", [
        ("ডাক্তার", "doctor_name"),
        ("ডাক্তারটা ভুল বলেছি", "doctor_name"),
        ("তারিখ", "date"),
        ("দিনটা ভুল", "date"),
        ("সময়টা ঠিক না", "time_slot"),
        ("ফোন নম্বর", "phone"),
        ("নম্বরটা ভুল", "phone"),
        ("নাম", "patient_name"),
        ("রোগীর নাম ভুল", "patient_name"),
        ("doctor", "doctor_name"),
        ("phone number", "phone"),
        ("name", "patient_name"),
    ])
    def test_recognized_fields(self, text, expected):
        assert parse_correction_field(text) == expected

    def test_doctor_name_wins_over_bare_naam_substring(self):
        # "ডাক্তারের নাম" (the doctor's name) contains "নাম" but must
        # resolve to doctor_name, not patient_name -- this is exactly
        # why doctor_name is checked before patient_name.
        assert parse_correction_field("ডাক্তারের নাম ভুল বলেছিলাম") == "doctor_name"

    @pytest.mark.parametrize("text", ["", "হ্যাঁ", "আচ্ছা ঠিক আছে", "xyz"])
    def test_unrecognized_returns_none(self, text):
        assert parse_correction_field(text) is None


# --------------------------------------------------------------------- #
# State-machine regression tests (main_pcm._continue_pending / _dispatch_turn)
# --------------------------------------------------------------------- #

class FakeToolsClient:
    """Stand-in for agent.tools_client.ClinicToolsClient. Records every
    call so tests can assert book_appointment is (or is not) invoked --
    the whole point of this story is that it must never fire before an
    explicit affirmative."""

    def __init__(self):
        self.book_appointment_calls = []

    async def book_appointment(self, doctor_name, date, time_slot, patient_name, phone):
        self.book_appointment_calls.append(
            {"doctor_name": doctor_name, "date": date, "time_slot": time_slot,
             "patient_name": patient_name, "phone": phone}
        )
        return {"success": True, "confirmation_id": "KCD-20260915-0099",
                "doctor_name": doctor_name, "date": date, "time_slot": time_slot}


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-call-1",
        pending=pending,
        dispatch_lock=asyncio.Lock(),
        send_json=_AsyncNoOp(),
    )


def full_pending(awaiting="confirm_booking", **slot_overrides):
    slots = dict(SLOTS)
    slots.update(slot_overrides)
    return {"awaiting": awaiting, "slots": slots, "candidates": None,
            "offered_date": None, "retries": 0}


@pytest.fixture(autouse=True)
def stub_speak_and_tools(monkeypatch):
    """Deterministic stand-ins for the two things _continue_pending talks
    to besides the parsers it exercises directly: the TTS-facing _speak()
    call and the tool client's write call."""
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    fake_tools = FakeToolsClient()
    monkeypatch.setattr(main_pcm, "_tools", fake_tools)
    return types.SimpleNamespace(spoken=spoken, tools=fake_tools)


class TestConfirmBookingState:
    """awaiting == "confirm_booking": the state entered once all 5 fields
    are known, before book_appointment() is ever called."""

    def test_affirmative_finishes_the_booking(self, stub_speak_and_tools):
        session = make_session(pending=full_pending())
        handled = run(main_pcm._continue_pending(session, "হ্যাঁ"))
        assert handled is True
        assert len(stub_speak_and_tools.tools.book_appointment_calls) == 1
        call = stub_speak_and_tools.tools.book_appointment_calls[0]
        assert call == {
            "doctor_name": SLOTS["doctor_name"], "date": SLOTS["date"],
            "time_slot": SLOTS["time_slot"], "patient_name": SLOTS["patient_name"],
            "phone": SLOTS["phone"],
        }
        assert session.pending is None  # cleared by _finish_booking

    @pytest.mark.parametrize("reply", ["yes", "Yes", "haan", "thik ache", "sure", "confirmed"])
    def test_affirmative_in_english_hinglish_or_banglish_also_books(
        self, stub_speak_and_tools, reply
    ):
        # The prompt itself may have been spoken in Bengali (main_pcm.py
        # never threads a detected language through today -- see the
        # session summary), but the CALLER's reply doesn't have to be:
        # a bilingual Kolkata caller answering a Bengali question with an
        # English "yes" is routine, not an edge case.
        session = make_session(pending=full_pending())
        handled = run(main_pcm._continue_pending(session, reply))
        assert handled is True
        assert len(stub_speak_and_tools.tools.book_appointment_calls) == 1

    @pytest.mark.parametrize("reply", ["no", "No", "nahi", "wrong", "incorrect"])
    def test_negative_in_english_or_hinglish_also_opens_correction(
        self, stub_speak_and_tools, reply
    ):
        session = make_session(pending=full_pending())
        handled = run(main_pcm._continue_pending(session, reply))
        assert handled is True
        assert stub_speak_and_tools.tools.book_appointment_calls == []
        assert session.pending["awaiting"] == "confirm_correction"

    def test_high_confidence_reply_still_does_not_skip_confirmation(
        self, stub_speak_and_tools
    ):
        # AC: "Readback is mandatory regardless of confidence." Simulated
        # here by an unambiguous-looking but non-affirmative reply -- it
        # must NOT be treated as an implicit yes.
        session = make_session(pending=full_pending())
        run(main_pcm._continue_pending(session, "দশটা"))  # sounds like a time, not yes/no
        assert stub_speak_and_tools.tools.book_appointment_calls == []

    def test_rejection_does_not_book_and_opens_correction_path(
        self, stub_speak_and_tools
    ):
        session = make_session(pending=full_pending())
        handled = run(main_pcm._continue_pending(session, "না"))
        assert handled is True
        assert stub_speak_and_tools.tools.book_appointment_calls == []
        assert session.pending["awaiting"] == "confirm_correction"
        # The spoken text must be the correction prompt, NOT a repeat of
        # the confirmation prompt -- this is the AC's literal wording.
        assert stub_speak_and_tools.spoken[-1] != booking_confirmation_prompt(SLOTS)

    def test_unparseable_reply_retries_then_gives_up(self, stub_speak_and_tools):
        session = make_session(pending=full_pending())
        for _ in range(2):
            handled = run(main_pcm._continue_pending(session, "কি বললেন?"))
            assert handled is True
            assert session.pending is not None
        # Third unparseable reply exceeds the retry budget (matches the
        # >2 cap every other awaiting-state in this module uses).
        handled = run(main_pcm._continue_pending(session, "কি বললেন?"))
        assert handled is False
        assert session.pending is None
        assert stub_speak_and_tools.tools.book_appointment_calls == []


class TestConfirmCorrectionState:
    """awaiting == "confirm_correction": which field is the caller
    fixing, entered only after a rejection above."""

    def test_naming_phone_reenters_phone_collection_with_others_kept(
        self, stub_speak_and_tools
    ):
        session = make_session(pending=full_pending(awaiting="confirm_correction"))
        handled = run(main_pcm._continue_pending(session, "ফোন নম্বরটা ভুল"))
        assert handled is True
        assert session.pending["awaiting"] == "phone"
        # The other 4 already-confirmed values are NOT thrown away --
        # only the disputed field is cleared.
        assert "phone" not in session.pending["slots"]
        for field in ("doctor_name", "date", "time_slot", "patient_name"):
            assert session.pending["slots"][field] == SLOTS[field]
        assert stub_speak_and_tools.tools.book_appointment_calls == []

    def test_full_correction_round_trip_reaches_confirm_booking_again(
        self, stub_speak_and_tools
    ):
        # Regression test for the whole loop: reject -> name a field ->
        # supply a new value -> back to confirm_booking -> affirm ->
        # exactly one booking, with the corrected value.
        session = make_session(pending=full_pending())
        run(main_pcm._continue_pending(session, "না"))  # reject
        assert session.pending["awaiting"] == "confirm_correction"

        run(main_pcm._continue_pending(session, "ফোন নম্বর"))  # name the field
        assert session.pending["awaiting"] == "phone"

        run(main_pcm._continue_pending(session, "৯৮৭৬৫৪৩২১০"))  # corrected phone (Bengali digits)
        assert session.pending["awaiting"] == "confirm_booking"
        assert session.pending["slots"]["phone"] == "9876543210"
        assert stub_speak_and_tools.tools.book_appointment_calls == []  # still not booked

        run(main_pcm._continue_pending(session, "হ্যাঁ"))  # affirm
        assert len(stub_speak_and_tools.tools.book_appointment_calls) == 1
        assert stub_speak_and_tools.tools.book_appointment_calls[0]["phone"] == "9876543210"

    def test_unrecognized_field_name_reprompts_without_losing_state(
        self, stub_speak_and_tools
    ):
        session = make_session(pending=full_pending(awaiting="confirm_correction"))
        handled = run(main_pcm._continue_pending(session, "xyz"))
        assert handled is True
        assert session.pending["awaiting"] == "confirm_correction"
        assert session.pending["slots"] == SLOTS


class TestNoWriteBeforeConfirmation:
    """Sanity check spanning the whole module: book_appointment must be
    reachable from exactly one place."""

    def test_finish_booking_is_the_only_caller_of_book_appointment(self):
        import ast
        import inspect

        source = inspect.getsource(main_pcm)
        tree = ast.parse(source)
        callers = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                for inner in ast.walk(node):
                    if (isinstance(inner, ast.Attribute) and inner.attr == "book_appointment"):
                        callers.append(node.name)
        assert callers == ["_finish_booking"], (
            f"book_appointment() must only ever be called from _finish_booking, "
            f"found it referenced inside: {callers}"
        )


class TestSingleShotBookingAlsoConfirms:
    """The other place all 5 fields can become known at once: a caller
    who gives everything in a single utterance, handled inline in
    _dispatch_turn's book_appointment branch rather than via
    _continue_pending. Exercised through _dispatch_turn directly with ASR
    and intent extraction stubbed -- still no network, no GPU, no model
    load (see tests/conftest.py)."""

    def test_all_fields_in_one_utterance_still_waits_for_confirmation(
        self, monkeypatch, stub_speak_and_tools, tmp_path
    ):
        class FakeASRResult:
            text = "ignored -- _resolve_intent is stubbed directly below"

        class FakeASR:
            async def transcribe_utterance(self, wav_path):
                return FakeASRResult()

        async def fake_resolve_intent(session, text):
            return {"intent": "book_appointment", "slots": dict(SLOTS)}

        monkeypatch.setattr(main_pcm, "_asr", FakeASR())
        monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)

        session = make_session(pending=None)
        wav_path = tmp_path / "utt.wav"
        wav_path.write_bytes(b"")  # _dispatch_turn just os.remove()s this

        run(main_pcm._dispatch_turn(session, str(wav_path)))

        assert stub_speak_and_tools.tools.book_appointment_calls == []
        assert session.pending is not None
        assert session.pending["awaiting"] == "confirm_booking"
        assert session.pending["slots"] == SLOTS
