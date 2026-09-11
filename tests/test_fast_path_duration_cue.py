"""ADDED BY SOURAV -- fixes a real production bug, reported directly from
a live call transcript:

    [User] How long does it take to get the urine test report?
    [AI]   Urine test rate is 200 taka.
    [User] How long will it take to get the urine test report?
    [AI]   Urine test rate is 200 taka.

Mirrors tests/test_fast_path_schedule_cue.py's structure -- that story
found the exact same CLASS of bug (fast_path.py confidently serving the
wrong intent before the LLM's newer, correct distinction ever runs) for
"doctor_schedule" vs "doctor_availability"; this covers the sibling fix
for "test_rate" vs "test_duration".

REAL BUG THIS FIXES, found while verifying agent/llm.py's new
"test_duration" intent would actually be reachable on a real call (not
just correct in the LLM's own prompt): agent/fast_path.py's pre-existing
_RATE_CUES set included three verbs -- "কত পড়বে", "কত লাগবে", "কত নেবে" --
that are genuinely ambiguous in colloquial Bengali between "how much will
it COST" and "how much/long will it TAKE". A bare "ইউরিন টেস্টের রিপোর্ট
পেতে কত লাগবে" ("how long to get the urine test report", no explicit time
word) was being fast-pathed straight to "test_rate", bypassing agent/
llm.py's SYSTEM_PROMPT entirely -- reproducing the exact reported bug at
this layer too, regardless of how correctly the LLM's own prompt was
fixed. Utterances with an explicit time word ("কতদিন লাগবে", "কত সময়
লাগবে") were never affected -- no substring match, since a word sits
between "কত" and "লাগবে"/"সময়".

Fixed by splitting _RATE_CUES into the unambiguous money-word cues (kept
as-is) and a new _AMBIGUOUS_RATE_CUES set, plus a new _DURATION_SIGNAL_
CUES set ("রিপোর্ট"/"ফলাফল"/"রেজাল্ট"). resolve() now abstains whenever
the ONLY rate signal present is one of the ambiguous verbs (no unambiguous
money word also present) AND a duration-signal word is also present --
deferring to the LLM, which now has the "test_duration" intent to route
it to. An unambiguous money word alongside a duration word still resolves
as test_rate (e.g. "রিপোর্ট এর জন্য কত টাকা লাগবে" is genuinely a price
question).
"""
from __future__ import annotations

from agent.fast_path import Catalogue, FastPath

_PAYLOAD = {
    "tests": [
        {"name": "Urine Routine Examination", "aliases_bn": ["ইউরিন"]},
    ],
    "doctors": [
        {"name": "Dr. A. Sen", "aliases_bn": ["সেন"], "surname": "Sen"},
    ],
}


def _fast_path() -> FastPath:
    return FastPath(Catalogue(_PAYLOAD))


class TestAmbiguousRateCueAbstainsWhenDurationSignalIsPresent:
    def test_bare_report_petey_koto_lagbe_abstains(self):
        fp = _fast_path()
        result = fp.resolve("ইউরিন টেস্টের রিপোর্ট পেতে কত লাগবে")
        assert result is None, (
            "must abstain to the LLM, which now knows this is test_duration -- "
            "must NOT be served as test_rate answering the price instead"
        )

    def test_result_word_variant_also_abstains(self):
        fp = _fast_path()
        assert fp.resolve("ইউরিন টেস্টের ফলাফল পেতে কত পড়বে") is None

    def test_bare_kotto_nebe_with_report_word_abstains(self):
        fp = _fast_path()
        assert fp.resolve("ইউরিন রিপোর্ট পেতে কত নেবে") is None


class TestExplicitTimeWordPhrasingsAlreadyWorkedAndStillDo:
    """Regression guard: these never matched _RATE_CUES at all (a word
    sits between "কত" and "লাগবে"/"সময়"), so this story's fix must not
    change their behaviour."""

    def test_kotodin_lagbe_still_abstains(self):
        fp = _fast_path()
        assert fp.resolve("ইউরিন রিপোর্ট পেতে কতদিন লাগবে") is None

    def test_koto_shomoy_lagbe_still_abstains(self):
        fp = _fast_path()
        assert fp.resolve("ইউরিন রিপোর্ট পেতে কত সময় লাগবে") is None


class TestUnambiguousMoneyWordStillResolvesToTestRate:
    """The fix must not make genuine price questions abstain, even ones
    that also happen to mention "রিপোর্ট"."""

    def test_plain_rate_question_still_fast_paths(self):
        fp = _fast_path()
        result = fp.resolve("ইউরিন টেস্টের রেট কত")
        assert result is not None
        assert result.intent == "test_rate"

    def test_koto_taka_lagbe_still_fast_paths(self):
        fp = _fast_path()
        result = fp.resolve("ইউরিন টেস্টের জন্য কত টাকা লাগবে")
        assert result is not None
        assert result.intent == "test_rate"

    def test_unambiguous_money_word_with_report_word_still_fast_paths(self):
        # Genuinely a price question despite mentioning "রিপোর্ট" -- an
        # unambiguous money word ("কত টাকা") is present, so this must NOT
        # be swallowed by the duration-signal abstain.
        fp = _fast_path()
        result = fp.resolve("ইউরিন রিপোর্ট এর জন্য কত টাকা লাগবে")
        assert result is not None
        assert result.intent == "test_rate"


class TestOtherIntentsAndCuesUnaffected:
    """Regression guard: unrelated fast_path behaviour (booking, doctor
    availability, doctor schedule abstain) must be untouched by this fix."""

    def test_doctor_availability_still_fast_paths(self):
        fp = _fast_path()
        result = fp.resolve("সেন আজ আছেন")
        assert result is not None
        assert result.intent == "doctor_availability"

    def test_doctor_schedule_cue_still_abstains(self):
        fp = _fast_path()
        assert fp.resolve("সেন কবে বসেন") is None

    def test_booking_still_abstains(self):
        fp = _fast_path()
        assert fp.resolve("সেনের সাথে বুক করতে চাই") is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
