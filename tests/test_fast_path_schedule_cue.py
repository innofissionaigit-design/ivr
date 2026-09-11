"""ADDED BY SOURAV -- "Caller asks when a doctor sits" story.

agent/fast_path.py had NO test file at all before this story (confirmed:
no tests/*.py referenced FastPath/fast_path prior to this change) -- a
real, pre-existing gap in a module that decides real caller routing
before the LLM ever sees an utterance, flagged in this story's test
report rather than silently left as-is or "fixed" by writing a full
test suite for the whole module, which is a separate undertaking beyond
this story's scope. This file covers ONLY the specific behavior this
story added/changed: the new _SCHEDULE_CUES abstain-check and its effect
on the pre-existing _AVAIL_CUES path.

REAL BUG THIS FIXES, found while building this story (not by pre-existing
test failure, since none existed): "ডাক্তার সেন কবে বসেন" ("when/which
days does Dr Sen sit") -- this story's OWN canonical phrasing -- used to
be intercepted by fast_path's pre-existing _AVAIL_CUES set (which
included the word "কবে") and confidently served as "doctor_availability"
with date=None, which main.py's dispatch then silently defaults to
TODAY. That means a caller asking the general schedule question would
have heard "is Dr Sen in today" instead of ever reaching the new
"doctor_schedule" intent this story built -- fast_path would have
silently short-circuited it before agent/llm.py's SYSTEM_PROMPT (which
DOES know the difference) ever ran. Fixed by moving "কবে"/"সময়সূচি"/
"শিডিউল" out of _AVAIL_CUES into their own _SCHEDULE_CUES set that
always abstains (fast_path has no doctor_schedule fast-path of its own --
a deliberate, flagged scope decision, see this story's test report).
"""
from __future__ import annotations

from agent.fast_path import Catalogue, FastPath

_DOCTOR_PAYLOAD = {
    "tests": [
        {"name": "Uric Acid", "aliases_bn": ["ইউরিক অ্যাসিড"]},
    ],
    "doctors": [
        {"name": "Dr. A. Sen", "aliases_bn": ["সেন"], "surname": "Sen"},
    ],
}


def _fast_path() -> FastPath:
    return FastPath(Catalogue(_DOCTOR_PAYLOAD))


class TestScheduleCueAbstains:
    def test_bare_kobe_boshen_abstains_rather_than_defaulting_to_today(self):
        fp = _fast_path()
        result = fp.resolve("সেন কবে বসেন")
        assert result is None, (
            "must abstain to the LLM, which knows this means doctor_schedule -- "
            "must NOT be served as doctor_availability defaulting to today"
        )

    def test_doctor_prefixed_kobe_boshen_still_abstains(self):
        fp = _fast_path()
        assert fp.resolve("ডাক্তার সেন কবে বসেন") is None

    def test_shoyshuchi_schedule_word_abstains(self):
        fp = _fast_path()
        assert fp.resolve("সেনের সময়সূচি কী") is None

    def test_english_loanword_shiditul_abstains(self):
        fp = _fast_path()
        assert fp.resolve("সেনের শিডিউল কী") is None

    def test_schedule_cue_abstains_even_with_a_digit_present(self):
        # A digit alone (e.g. a stray number in noisy ASR output) must not
        # override the schedule-cue abstain -- when "কবে" is present at
        # all, this always defers, rather than trying to guess whether the
        # digit makes it a resolvable specific-date question instead.
        fp = _fast_path()
        assert fp.resolve("সেন কবে বসেন 2026") is None


class TestPreExistingAvailabilityBehaviorUnchanged:
    """Regression guard: this story's fix must not touch the OTHER
    _AVAIL_CUES words' existing, deliberately-designed behavior (see
    main.py's own comment on why a bare presence question defaults to
    today)."""

    def test_explicit_today_still_fast_paths_to_doctor_availability(self):
        fp = _fast_path()
        result = fp.resolve("সেন আজ আছেন")
        assert result is not None
        assert result.intent == "doctor_availability"

    def test_bare_presence_question_without_kobe_still_fast_paths(self):
        # "আছেন" alone (no "কবে") is unaffected by this story's change --
        # still confidently resolves, still defaults to today, exactly as
        # before.
        fp = _fast_path()
        result = fp.resolve("সেন আছেন")
        assert result is not None
        assert result.intent == "doctor_availability"
        assert result.slots["date"] is None  # main.py defaults this to today itself

    def test_kokhon_cue_is_unaffected_kept_in_avail_cues(self):
        # "কখন" (what time) was deliberately NOT moved -- see
        # _SCHEDULE_CUES's own comment for why only "কবে" (which day, in
        # general) was reclassified, not "কখন" (what time, today).
        fp = _fast_path()
        result = fp.resolve("সেন কখন আছেন")
        assert result is not None
        assert result.intent == "doctor_availability"


class TestOtherIntentsUnaffected:
    """Regression guard: the schedule-cue check must not accidentally
    swallow unrelated intents that happen to share no words with it."""

    def test_test_rate_still_fast_paths_normally(self):
        fp = _fast_path()
        result = fp.resolve("ইউরিক অ্যাসিড টেস্টের রেট কত")
        assert result is not None
        assert result.intent == "test_rate"

    def test_booking_still_abstains_as_before(self):
        fp = _fast_path()
        assert fp.resolve("সেনের সাথে বুক করতে চাই") is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
