"""No field label, colon or bracket is ever spoken.

story title: Answers sound like a person, not a database row
user story: As a patient, I want to hear a sentence, so that the agent sounds
    like someone at the counter.
acceptance criteria: No field label, colon or bracket is ever spoken and every
    structured value renders as a natural clause in the reply language. An
    automated check fails a build containing a spoken punctuation artefact.

Ported from dev_sourav's TestNoSpokenPunctuationArtifact, Bengali only.

THE CHECK RUNS THROUGH verbalize(), NOT ON THE RAW TEMPLATE, and that is the
whole design. A raw template contains colons that are perfectly fine -- a
chamber hour is "18:00-20:00" and a time slot is "18:30" -- because
bn_normalize.verbalize() turns those into Bengali words before synthesis. The
defect was never the character; it was the character SURVIVING to the
synthesiser. So the assertion is made on the string TTS actually receives,
which is the only place the question has an answer.

Confusing those two is the most likely way someone breaks this later: banning
":" from templates would fail on values that are already handled, and banning
it from nothing would miss the labels. The gate sits at exactly one point.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.bn_normalize import verbalize  # noqa: E402
from agent.reply_templates import (  # noqa: E402
    ANSWER_CHANGED_BN, DEFERRED_PART_BN, NEAR_MATCH_UNCLEAR_BN,
    RESUMING_PART_BN, UNANSWERED_PART_BN, UNSPEAKABLE_ESCALATION,
    unanswered_part_prompt,
    near_match_prompt, booking_confirm_prompt,
    booking_correction_prompt, booking_reply, date_range_confirm_prompt,
    doctor_availability_reply, doctors_by_department_reply, heard_confirm_prompt,
    missing_slot_prompt, test_rate_reply as rate_reply, with_change_notice,
)

# Characters that mean something on a page and nothing in a sentence. A caller
# hears them as a stumble, a mispronunciation, or -- with the Bengali
# tokenizer -- as nothing at all, which is worse.
FORBIDDEN = (":", "：", "[", "]", "{", "}", "<", ">", "|")

FOUND_TEST = {"found": True, "test_name": "Uric Acid", "test_name_bn": "ইউরিক অ্যাসিড",
              "rate_inr": 250, "sample_type": "Blood", "report_time_hours": 12}
FOUND_DOCTOR = {"found": True, "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
                "date": "2026-09-14", "available": True,
                "chamber_hours": "18:00-20:00", "next_available_date": None}
BOOKING = {"doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন", "date": "2026-09-14",
           "time_slot": "18:30", "patient_name": "রিয়া দাস", "phone": "9876543210"}


def _replies():
    """Every reply function, every branch, with realistic values.

    Named so a failure says which sentence broke rather than which index.
    """
    yield "rate/found", rate_reply({"test_name": "ইউরিক অ্যাসিড"}, FOUND_TEST)
    yield "rate/not-found", rate_reply({"test_name": "কিছু"},
                                       {"found": False, "query": "কিছু"})
    yield "rate/suggestions-1", rate_reply(
        {"test_name": "কিছু"},
        {"found": False, "query": "কিছু", "did_you_mean_bn": ["লিপিড প্রোফাইল"]})
    yield "rate/suggestions-3", rate_reply(
        {"test_name": "কিছু"},
        {"found": False, "query": "কিছু",
         "did_you_mean_bn": ["লিপিড প্রোফাইল", "এলএফটি", "সিবিসি"]})

    yield "doctor/available", doctor_availability_reply({"doctor_name": "সেন"}, FOUND_DOCTOR)
    yield "doctor/not-that-day", doctor_availability_reply(
        {"doctor_name": "সেন"},
        dict(FOUND_DOCTOR, available=False, chamber_hours=None,
             next_available_date="2026-09-17"))
    yield "doctor/no-fixed-day", doctor_availability_reply(
        {"doctor_name": "সেন"},
        dict(FOUND_DOCTOR, available=False, chamber_hours=None,
             next_available_date=None))
    yield "doctor/not-found", doctor_availability_reply(
        {"doctor_name": "ঘোষ"}, {"found": False, "query": "ঘোষ"})

    dept = {"found": True, "department": "Cardiology", "department_bn": "কার্ডিওলজি",
            "date": "2026-09-14"}
    one = {"name": "Dr. A Sen", "doctor_name_bn": "সেন"}
    two = {"name": "Dr. B Roy", "doctor_name_bn": "রায়"}
    three = {"name": "Dr. C Bose", "doctor_name_bn": "বসু"}
    yield "dept/1", doctors_by_department_reply({}, dict(dept, doctors=[one]))
    yield "dept/2", doctors_by_department_reply({}, dict(dept, doctors=[one, two]))
    yield "dept/3", doctors_by_department_reply({}, dict(dept, doctors=[one, two, three]))
    yield "dept/none-that-day", doctors_by_department_reply({}, dict(dept, doctors=[]))
    yield "dept/none-unfiltered", doctors_by_department_reply(
        {}, {"found": True, "department": "Cardiology", "department_bn": "কার্ডিওলজি",
             "date": None, "doctors": []})
    yield "dept/not-found", doctors_by_department_reply(
        {"department": "নিউরো"}, {"found": False, "query": "নিউরো"})

    yield "booking/success", booking_reply(
        {"doctor_name": "সেন"},
        {"success": True, "confirmation_id": "KCD-20260914-4A2F",
         "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
         "date": "2026-09-14", "time_slot": "18:30"})
    yield "booking/slot-taken-alts", booking_reply(
        {}, {"success": False, "reason": "slot_taken",
             "alternative_slots": ["17:30", "18:15", "19:00"]})
    yield "booking/slot-taken-none", booking_reply(
        {}, {"success": False, "reason": "slot_taken", "alternative_slots": []})
    yield "booking/doctor-not-found", booking_reply(
        {"doctor_name": "ঘোষ"}, {"success": False, "reason": "doctor_not_found"})
    yield "booking/other", booking_reply({}, {"success": False, "reason": "missing_field"})

    yield "booking/readback", booking_confirm_prompt(BOOKING)
    yield "booking/correction", booking_correction_prompt()
    yield "date/range", date_range_confirm_prompt("2026-09-14", "2026-09-20")
    yield "asr/echo", heard_confirm_prompt("কাল ডাক্তার সেন আছেন")
    yield "escalation", UNSPEAKABLE_ESCALATION

    # story title: The same question gets the same answer within one call
    # user story: As a caller who asks twice, I want the same answer, so that
    #   I know which one to believe.
    # acceptance criteria: Repeating a question in one call produces an
    #   identical factual answer unless the underlying data changed, in which
    #   case the change is stated. A test asserts consistency across three
    #   repeats with an unchanged backend.
    #
    # The JOINED sentence, not the preamble alone. What the caller hears when
    # a figure moves is one utterance, and a punctuation artefact at the seam
    # would belong to neither half on its own.
    yield "changed/preamble", ANSWER_CHANGED_BN
    yield "changed/rate", with_change_notice(
        rate_reply({"test_name": "ইউরিক অ্যাসিড"}, FOUND_TEST))
    yield "changed/doctor", with_change_notice(
        doctor_availability_reply({"doctor_name": "সেন"}, FOUND_DOCTOR))

    # story title: Near matches are offered rather than guessed or refused
    # user story: As a caller naming something loosely, I want the close
    #   matches offered, so that I am not told my test does not exist when it
    #   does.
    # acceptance criteria: When several catalogue rows fall within the match
    #   band the agent offers up to three by name and asks which. Candidates
    #   are generated across every supported language and romanised spelling.
    #   The did-you-mean path covers the ambiguous case and not only total
    #   failure.
    #
    # One, two and the capped three, plus the re-ask. The three-candidate
    # case is the one worth having: _spoken_list switches from "নাকি" to a
    # comma list there, and a comma list is exactly the database-row reading
    # E12-S3 bans.
    _sugar = [{"name": "Blood Sugar Fasting", "name_bn": "সুগার ফাস্টিং"},
              {"name": "Blood Sugar PP", "name_bn": "সুগার পিপি"}]
    yield "near/1", near_match_prompt(_sugar[:1])
    yield "near/2", near_match_prompt(_sugar)
    yield "near/3", near_match_prompt(
        _sugar + [{"name": "HbA1c", "name_bn": "এইচবিএ১সি"}])
    yield "near/unclear", NEAR_MATCH_UNCLEAR_BN

    # story title: A multi-part question is answered in full
    # user story: As a caller who asked two things, I want both answered, so
    #   that I do not have to ask again.
    # acceptance criteria: Every answerable part of a turn is answered in the
    #   order asked, and any part that cannot be answered is explicitly
    #   addressed rather than dropped. Completeness is scored on a labelled
    #   multi-part set.
    #
    # The subject-carrying form quotes the CALLER'S words, so it is the one
    # that can pick up whatever the ASR produced -- including a stray colon
    # or bracket out of a code-switched utterance.
    yield "multipart/unanswered", UNANSWERED_PART_BN
    yield "multipart/unanswered-named", unanswered_part_prompt("ইউরিক অ্যাসিড")
    yield "multipart/deferred", DEFERRED_PART_BN
    yield "multipart/resuming", RESUMING_PART_BN
    yield "near/no-spoken-name", near_match_prompt(
        [{"name": "Some Test", "name_bn": None}])

    for intent in ("test_rate", "doctor_availability", "doctors_by_department",
                   "book_appointment"):
        for field in ("test_name", "doctor_name", "department", "date",
                      "time_slot", "patient_name", "phone"):
            yield f"prompt/{intent}/{field}", missing_slot_prompt(intent, field)


CASES = list(_replies())


@pytest.mark.parametrize("name,reply", CASES, ids=[c[0] for c in CASES])
def test_no_punctuation_artefact_reaches_the_synthesiser(name, reply):
    spoken = verbalize(reply)
    offenders = [ch for ch in FORBIDDEN if ch in spoken]
    assert not offenders, (
        f"{name}: {offenders} survives verbalize() and would be spoken.\n"
        f"  template: {reply}\n"
        f"  spoken:   {spoken}"
    )


def test_the_gate_covers_every_public_reply_function():
    """A new reply function that nobody adds a case for is the way this gate
    quietly stops covering things. Fails if reply_templates grows one."""
    import agent.reply_templates as rt

    public = {n for n in dir(rt)
              if not n.startswith("_") and callable(getattr(rt, n))
              and getattr(rt, n).__module__ == rt.__name__}
    exercised = {
        "test_rate_reply", "doctor_availability_reply", "doctors_by_department_reply",
        "booking_reply", "booking_confirm_prompt", "booking_correction_prompt",
        "date_range_confirm_prompt", "heard_confirm_prompt", "missing_slot_prompt",
        "with_change_notice", "near_match_prompt", "unanswered_part_prompt",
    }
    assert public <= exercised, (
        f"reply function(s) {sorted(public - exercised)} have no case in this gate"
    )


def test_a_raw_value_colon_is_not_the_defect():
    """Guards the distinction the docstring makes. A time still contains a
    colon in the TEMPLATE and must keep doing so -- verbalize() is what turns
    it into words. Someone "fixing" this by banning colons from templates
    would break the times and fix nothing."""
    raw = booking_confirm_prompt(BOOKING)
    assert "18:30" in raw, "the raw template no longer carries the time"
    assert ":" not in verbalize(raw), "but it must not survive to the synthesiser"


def test_the_artefact_would_actually_be_caught():
    """The gate, pointed at a sentence that has the defect. If this stops
    failing, the check has stopped checking."""
    assert ":" in verbalize("স্যাম্পল: রক্ত।"), "a label colon must survive verbalize"
