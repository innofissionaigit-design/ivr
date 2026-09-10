"""ADDED BY SOURAV -- dispatch-level tests for the "Lab Report Status &
Secure Delivery" combined story, covering the parts agent/report_flow.py's
own unit tests (tests/test_report_flow.py) cannot: the actual I/O glue in
main_pcm.py (_finish_report_flow, _handle_report_lookup, the
"report_status"/"report_send" _dispatch_turn branches, and the four new
_continue_pending states -- "report_phone", "which_report",
"confirm_delivery", "otp_code").

Pattern mirrors tests/test_test_sample_intent.py and
tests/test_booking_readback.py exactly: no pytest-asyncio, ASR/LLM
extraction stubbed via monkeypatch, driven with asyncio.run() inside
ordinary sync test functions. tests/conftest.py stubs the torch/nemo/
omegaconf imports main_pcm.py transitively needs so this runs without the
real GPU stack.

WHY main_pcm.py AND NOT main.py: both are produced from the same source
(main.py is hand-edited; main_pcm.py is regenerated from it by
tools/make_pcm_variant.py, which refuses to write unless the
_resolve_intent...</code> block is byte-identical between the two -- see
that script's own docstring). Testing one dispatch file exercises the
identical logic in both by construction; this suite additionally asserts
that byte-identity directly (TestTransportParity below), which is the
actual mechanical guarantee RULE 18 ("channel/language must not change
the backend decision") rests on for this story.
"""
from __future__ import annotations

import asyncio
import types

import pytest

import main_pcm
from agent.tools_client import ToolCallError


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- #
# Shared fakes -- same shapes as test_test_sample_intent.py /
# test_booking_readback.py's own fakes, extended with the 3 new
# tools_client methods this story adds.
# --------------------------------------------------------------------- #

class FakeToolsClient:
    def __init__(self):
        self.get_report_status_calls = []
        self.request_report_delivery_calls = []
        self.verify_report_otp_calls = []
        self.get_report_status_response = None
        self.request_report_delivery_response = None
        self.verify_report_otp_response = None

    async def get_report_status(self, phone, test_name=None):
        self.get_report_status_calls.append((phone, test_name))
        return self.get_report_status_response

    async def request_report_delivery(self, phone, report_number):
        self.request_report_delivery_calls.append((phone, report_number))
        return self.request_report_delivery_response

    async def verify_report_otp(self, phone, report_number, otp_code):
        self.verify_report_otp_calls.append((phone, report_number, otp_code))
        return self.verify_report_otp_response


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


class FakeASRResult:
    text = "ignored -- _resolve_intent is stubbed directly"


class FakeASR:
    async def transcribe_utterance(self, wav_path):
        return FakeASRResult()


@pytest.fixture
def env(monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    tools = FakeToolsClient()
    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    monkeypatch.setattr(main_pcm, "_asr", FakeASR())
    monkeypatch.setattr(main_pcm, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools)


def dispatch_with_intent(monkeypatch, intent, slots, tmp_path):
    async def fake_resolve_intent(session, text):
        return {"intent": intent, "slots": slots}

    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
    session = make_session(pending=None)
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return session


def continue_pending(session, text):
    handled = run(main_pcm._continue_pending(session, text))
    assert handled is True
    return session


READY_ENABLED = {
    "patient_found": True, "found": True, "status": "READY",
    "delivery_enabled": True, "report_number": "RPT-A", "test_name": "CBC",
}


# --------------------------------------------------------------------- #
# report_status / report_send entry -- phone gate (RULE 15)
# --------------------------------------------------------------------- #

class TestPhoneGate:
    def test_report_status_without_phone_asks_for_it_and_does_not_call_the_tool(
        self, monkeypatch, env, tmp_path
    ):
        session = dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": "CBC", "phone": None}, tmp_path,
        )
        assert env.tools.get_report_status_calls == []
        assert session.pending["awaiting"] == "report_phone"
        assert session.pending["flow"] == "report_status"
        assert session.pending["test_name"] == "CBC"
        assert len(env.spoken) == 1

    def test_report_send_without_phone_asks_for_it(self, monkeypatch, env, tmp_path):
        session = dispatch_with_intent(
            monkeypatch, "report_send", {"test_name": "CBC", "phone": None}, tmp_path,
        )
        assert env.tools.get_report_status_calls == []
        assert session.pending["awaiting"] == "report_phone"
        assert session.pending["flow"] == "report_send"

    def test_report_status_with_phone_in_the_same_utterance_calls_the_tool_immediately(
        self, monkeypatch, env, tmp_path
    ):
        env.tools.get_report_status_response = {
            "patient_found": True, "found": False, "reason": "NOT_FOUND",
        }
        dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": "CBC", "phone": "9000000001"}, tmp_path,
        )
        assert env.tools.get_report_status_calls == [("9000000001", "CBC")]

    def test_report_phone_pending_state_does_not_collide_with_booking_phone_correction(
        self, monkeypatch, env, tmp_path
    ):
        # Regression guard for the exact bug this story introduced and
        # then fixed: the report flow's phone-collection state MUST be
        # named "report_phone", not the bare "phone" the booking
        # correction flow already uses for the exact same field name --
        # see agent/report_flow.py's AWAITING_REPORT_PHONE comment. This
        # is asserted properly end-to-end in test_booking_readback.py
        # (TestConfirmCorrectionState); this is a narrower same-suite
        # sanity check that the two states are simply different strings.
        from agent.report_flow import AWAITING_REPORT_PHONE
        assert AWAITING_REPORT_PHONE == "report_phone"
        assert AWAITING_REPORT_PHONE != "phone"

    def test_report_phone_state_rejects_unparseable_phone_and_retries(
        self, monkeypatch, env, tmp_path
    ):
        session = make_session(pending={
            "awaiting": "report_phone", "flow": "report_status", "test_name": "CBC", "retries": 0,
        })
        continue_pending(session, "আমি জানি না")
        assert env.tools.get_report_status_calls == []
        assert session.pending["awaiting"] == "report_phone"
        assert session.pending["retries"] == 1

    def test_report_phone_state_gives_up_after_repeated_unparseable_replies(
        self, monkeypatch, env, tmp_path
    ):
        session = make_session(pending={
            "awaiting": "report_phone", "flow": "report_status", "test_name": "CBC", "retries": 2,
        })
        handled = run(main_pcm._continue_pending(session, "আমি জানি না"))
        assert handled is False
        assert session.pending is None

    def test_report_phone_state_says_no_and_drops_the_flow(self, monkeypatch, env, tmp_path):
        session = make_session(pending={
            "awaiting": "report_phone", "flow": "report_status", "test_name": "CBC", "retries": 0,
        })
        continue_pending(session, "না")
        assert session.pending is None
        assert env.tools.get_report_status_calls == []


# --------------------------------------------------------------------- #
# RULE 1: never invent a report / patient not found
# --------------------------------------------------------------------- #

class TestPatientOrReportNotFound:
    def test_unknown_phone_speaks_patient_not_found(self, monkeypatch, env, tmp_path):
        env.tools.get_report_status_response = {"patient_found": False}
        session = dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": None, "phone": "9999999999"}, tmp_path,
        )
        assert session.pending is None
        assert len(env.spoken) == 1

    def test_known_patient_no_matching_report_speaks_not_found(self, monkeypatch, env, tmp_path):
        env.tools.get_report_status_response = {
            "patient_found": True, "found": False, "reason": "NOT_FOUND",
        }
        session = dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": "MRI", "phone": "9000000008"}, tmp_path,
        )
        assert session.pending is None


# --------------------------------------------------------------------- #
# RULE 13: multiple reports -> which_report disambiguation
# --------------------------------------------------------------------- #

class TestWhichReportDisambiguation:
    CANDIDATES = [
        {"report_number": "RPT-10004", "test_name": "CBC", "status": "READY",
         "delivery_enabled": True},
        {"report_number": "RPT-10010", "test_name": "TSH", "status": "READY",
         "delivery_enabled": True},
    ]

    def test_ambiguous_result_parks_in_which_report_state(self, monkeypatch, env, tmp_path):
        env.tools.get_report_status_response = {
            "patient_found": True, "found": False, "reason": "AMBIGUOUS",
            "candidates": self.CANDIDATES,
        }
        session = dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": None, "phone": "9000000004"}, tmp_path,
        )
        assert session.pending["awaiting"] == "which_report"
        assert session.pending["candidates"] == self.CANDIDATES
        assert session.pending["phone"] == "9000000004"

    def test_naming_one_test_reconstructs_it_locally_without_a_second_lookup(
        self, monkeypatch, env, tmp_path
    ):
        session = make_session(pending={
            "awaiting": "which_report", "flow": "report_status",
            "candidates": self.CANDIDATES, "phone": "9000000004", "retries": 0,
        })
        continue_pending(session, "TSH")
        # No re-query -- the candidate list already carried everything
        # interpret_report_status_result needs (considered tradeoff, see
        # main.py's "which_report" comment).
        assert env.tools.get_report_status_calls == []
        assert session.pending["awaiting"] == "confirm_delivery"
        assert session.pending["report_number"] == "RPT-10010"

    def test_unmatched_answer_reprompts(self, monkeypatch, env, tmp_path):
        session = make_session(pending={
            "awaiting": "which_report", "flow": "report_status",
            "candidates": self.CANDIDATES, "phone": "9000000004", "retries": 0,
        })
        continue_pending(session, "এক্স-রে")
        assert session.pending["awaiting"] == "which_report"
        assert session.pending["retries"] == 1

    def test_report_send_flow_goes_straight_to_delivery_after_disambiguation(
        self, monkeypatch, env, tmp_path
    ):
        env.tools.request_report_delivery_response = {
            "success": True, "reason": "OTP_REQUIRED", "masked_phone": "xxxxx04",
        }
        session = make_session(pending={
            "awaiting": "which_report", "flow": "report_send",
            "candidates": self.CANDIDATES, "phone": "9000000004", "retries": 0,
        })
        continue_pending(session, "CBC")
        assert env.tools.request_report_delivery_calls == [("9000000004", "RPT-10004")]
        assert session.pending["awaiting"] == "otp_code"


# --------------------------------------------------------------------- #
# RULE 2/3/16: NOT_READY / PROCESSING / CANCELLED / delivery-disabled
# --------------------------------------------------------------------- #

class TestBlockedReports:
    @pytest.mark.parametrize("status", ["NOT_READY", "PROCESSING", "CANCELLED"])
    def test_report_status_just_reports_the_true_status(self, monkeypatch, env, tmp_path, status):
        env.tools.get_report_status_response = {
            "patient_found": True, "found": True, "status": status,
            "delivery_enabled": True, "report_number": "RPT-X",
        }
        session = dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": "TSH", "phone": "9000000003"}, tmp_path,
        )
        assert session.pending is None
        assert env.tools.request_report_delivery_calls == []

    @pytest.mark.parametrize("status", ["NOT_READY", "PROCESSING", "CANCELLED"])
    def test_report_send_never_starts_otp_for_an_ineligible_report(
        self, monkeypatch, env, tmp_path, status
    ):
        env.tools.get_report_status_response = {
            "patient_found": True, "found": True, "status": status,
            "delivery_enabled": True, "report_number": "RPT-X",
        }
        session = dispatch_with_intent(
            monkeypatch, "report_send", {"test_name": "TSH", "phone": "9000000003"}, tmp_path,
        )
        assert session.pending is None
        assert env.tools.request_report_delivery_calls == []

    def test_ready_but_delivery_disabled_never_offers_or_starts_otp(
        self, monkeypatch, env, tmp_path
    ):
        env.tools.get_report_status_response = {
            "patient_found": True, "found": True, "status": "READY",
            "delivery_enabled": False, "report_number": "RPT-I",
        }
        session = dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": "KFT", "phone": "9000000011"}, tmp_path,
        )
        assert session.pending is None
        assert env.tools.request_report_delivery_calls == []


# --------------------------------------------------------------------- #
# RULE 4: offer -> explicit yes -> OTP requested
# --------------------------------------------------------------------- #

class TestOfferAndConfirmDelivery:
    def test_ready_and_enabled_offers_and_waits(self, monkeypatch, env, tmp_path):
        env.tools.get_report_status_response = dict(READY_ENABLED)
        session = dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": "CBC", "phone": "9000000001"}, tmp_path,
        )
        assert session.pending["awaiting"] == "confirm_delivery"
        assert session.pending["report_number"] == "RPT-A"
        assert session.pending["phone"] == "9000000001"
        assert env.tools.request_report_delivery_calls == []  # not yet -- no "yes" given

    def test_yes_triggers_delivery_request_and_moves_to_otp_code(self, monkeypatch, env, tmp_path):
        env.tools.request_report_delivery_response = {
            "success": True, "reason": "OTP_REQUIRED", "masked_phone": "xxxxx01",
        }
        session = make_session(pending={
            "awaiting": "confirm_delivery", "report_number": "RPT-A",
            "phone": "9000000001", "retries": 0,
        })
        continue_pending(session, "হ্যাঁ")
        assert env.tools.request_report_delivery_calls == [("9000000001", "RPT-A")]
        assert session.pending["awaiting"] == "otp_code"
        assert session.pending["report_number"] == "RPT-A"

    def test_no_declines_and_drops_the_flow(self, monkeypatch, env, tmp_path):
        session = make_session(pending={
            "awaiting": "confirm_delivery", "report_number": "RPT-A",
            "phone": "9000000001", "retries": 0,
        })
        continue_pending(session, "না")
        assert session.pending is None
        assert env.tools.request_report_delivery_calls == []

    def test_unparseable_reply_reprompts_and_eventually_gives_up(self, monkeypatch, env, tmp_path):
        session = make_session(pending={
            "awaiting": "confirm_delivery", "report_number": "RPT-A",
            "phone": "9000000001", "retries": 2,
        })
        handled = run(main_pcm._continue_pending(session, "আচ্ছা ঠিক আছে হয়তো"))
        assert handled is False
        assert session.pending is None

    def test_report_state_changed_between_offer_and_yes_is_reported_truthfully(
        self, monkeypatch, env, tmp_path
    ):
        # Defense-in-depth: clinic-api re-checked and the report is no
        # longer eligible (e.g. cancelled in the interim) -- the caller
        # must hear the true state, never a false "sending now".
        env.tools.request_report_delivery_response = {"success": False, "reason": "CANCELLED"}
        session = make_session(pending={
            "awaiting": "confirm_delivery", "report_number": "RPT-A",
            "phone": "9000000001", "retries": 0,
        })
        continue_pending(session, "হ্যাঁ")
        assert session.pending is None


# --------------------------------------------------------------------- #
# RULE 4-9: OTP verification, and the ATTACK cases around it
# --------------------------------------------------------------------- #

class TestOtpVerification:
    def _pending(self, retries=0):
        return {"awaiting": "otp_code", "report_number": "RPT-A", "phone": "9000000001",
                "retries": retries}

    def test_correct_otp_delivers(self, monkeypatch, env, tmp_path):
        env.tools.verify_report_otp_response = {
            "success": True, "reason": "DELIVERY_SENT", "masked_phone": "xxxxx01",
            "signed_link_expires_minutes": 15,
        }
        session = make_session(pending=self._pending())
        continue_pending(session, "৪৮২৯১৩")
        assert env.tools.verify_report_otp_calls == [("9000000001", "RPT-A", "482913")]
        assert session.pending is None

    def test_wrong_otp_stays_in_otp_code_state(self, monkeypatch, env, tmp_path):
        env.tools.verify_report_otp_response = {"success": False, "reason": "OTP_INVALID"}
        session = make_session(pending=self._pending())
        continue_pending(session, "111111")
        assert session.pending["awaiting"] == "otp_code"
        assert session.pending["report_number"] == "RPT-A"

    def test_expired_otp_ends_flow(self, monkeypatch, env, tmp_path):
        env.tools.verify_report_otp_response = {"success": False, "reason": "OTP_EXPIRED"}
        session = make_session(pending=self._pending())
        continue_pending(session, "615204")
        assert session.pending is None

    def test_max_attempts_ends_flow_even_with_local_retries_left(self, monkeypatch, env, tmp_path):
        # RULE 8: clinic-api's server-side attempt cap is authoritative --
        # it must end the flow regardless of this transport's own local
        # retries counter (here still at 0, i.e. "plenty of local retries
        # left" by the transport's own bookkeeping).
        env.tools.verify_report_otp_response = {"success": False, "reason": "OTP_MAX_ATTEMPTS"}
        session = make_session(pending=self._pending(retries=0))
        continue_pending(session, "903217")
        assert session.pending is None

    def test_already_used_otp_ends_flow(self, monkeypatch, env, tmp_path):
        env.tools.verify_report_otp_response = {"success": False, "reason": "OTP_ALREADY_USED"}
        session = make_session(pending=self._pending())
        continue_pending(session, "731846")
        assert session.pending is None

    def test_delivery_failed_after_correct_otp_is_reported_truthfully(
        self, monkeypatch, env, tmp_path
    ):
        # RULE 17: never claim delivery succeeded when it did not.
        env.tools.verify_report_otp_response = {"success": False, "reason": "DELIVERY_FAILED"}
        session = make_session(pending=self._pending())
        continue_pending(session, "482913")
        assert session.pending is None

    def test_caller_says_no_mid_otp_declines_without_calling_verify(
        self, monkeypatch, env, tmp_path
    ):
        session = make_session(pending=self._pending())
        continue_pending(session, "না")
        assert env.tools.verify_report_otp_calls == []
        assert session.pending is None

    # ---- ATTACK 8: caller asks to be TOLD the otp instead of speaking it back ----

    def test_asking_to_be_told_the_otp_is_refused_not_treated_as_a_normal_retry(
        self, monkeypatch, env, tmp_path
    ):
        session = make_session(pending=self._pending())
        continue_pending(session, "what is the otp")
        assert env.tools.verify_report_otp_calls == []
        # Refused, but NOT counted against the ordinary unparseable-reply
        # retry budget -- distinct code path, see main.py's otp_code state.
        assert session.pending["retries"] == 0
        assert session.pending["awaiting"] == "otp_code"

    def test_an_utterance_naming_otp_but_containing_a_real_code_is_treated_as_an_attempt(
        self, monkeypatch, env, tmp_path
    ):
        # "the otp is 482913" contains the word "otp" but IS a genuine
        # attempt -- must not be misclassified as a disclosure request
        # (see agent/slot_parse.py's looks_like_otp_disclosure_request
        # ordering guarantee).
        env.tools.verify_report_otp_response = {
            "success": True, "reason": "DELIVERY_SENT", "masked_phone": "xxxxx01",
            "signed_link_expires_minutes": 15,
        }
        session = make_session(pending=self._pending())
        continue_pending(session, "the otp is 482913")
        assert env.tools.verify_report_otp_calls == [("9000000001", "RPT-A", "482913")]

    def test_gibberish_that_is_neither_a_code_nor_a_disclosure_request_reprompts(
        self, monkeypatch, env, tmp_path
    ):
        session = make_session(pending=self._pending())
        continue_pending(session, "আকাশ থেকে পড়লাম")
        assert env.tools.verify_report_otp_calls == []
        assert session.pending["retries"] == 1
        assert session.pending["awaiting"] == "otp_code"

    def test_gibberish_gives_up_after_repeated_failures(self, monkeypatch, env, tmp_path):
        session = make_session(pending=self._pending(retries=2))
        handled = run(main_pcm._continue_pending(session, "??"))
        assert handled is False
        assert session.pending is None


# --------------------------------------------------------------------- #
# Tool-call failure -- every new branch must fail the same infrastructure
# way as every pre-existing intent, not hang or crash.
# --------------------------------------------------------------------- #

class TestToolFailureFallback:
    def test_get_report_status_failure(self, monkeypatch, env, tmp_path):
        class FailingTools(FakeToolsClient):
            async def get_report_status(self, phone, test_name=None):
                raise ToolCallError("clinic-api unreachable")

        monkeypatch.setattr(main_pcm, "_tools", FailingTools())
        session = dispatch_with_intent(
            monkeypatch, "report_status", {"test_name": "CBC", "phone": "9000000001"}, tmp_path,
        )
        assert session.pending is None
        assert len(env.spoken) == 1
        assert "কাউন্টারে" in env.spoken[0]

    def test_request_report_delivery_failure_from_confirm_delivery_state(
        self, monkeypatch, env, tmp_path
    ):
        class FailingTools(FakeToolsClient):
            async def request_report_delivery(self, phone, report_number):
                raise ToolCallError("clinic-api unreachable")

        monkeypatch.setattr(main_pcm, "_tools", FailingTools())
        session = make_session(pending={
            "awaiting": "confirm_delivery", "report_number": "RPT-A",
            "phone": "9000000001", "retries": 0,
        })
        continue_pending(session, "হ্যাঁ")
        assert session.pending is None
        assert "কাউন্টারে" in env.spoken[0]

    def test_verify_report_otp_failure(self, monkeypatch, env, tmp_path):
        class FailingTools(FakeToolsClient):
            async def verify_report_otp(self, phone, report_number, otp_code):
                raise ToolCallError("clinic-api unreachable")

        monkeypatch.setattr(main_pcm, "_tools", FailingTools())
        session = make_session(pending={
            "awaiting": "otp_code", "report_number": "RPT-A",
            "phone": "9000000001", "retries": 0,
        })
        continue_pending(session, "482913")
        assert session.pending is None
        assert "কাউন্টারে" in env.spoken[0]


# --------------------------------------------------------------------- #
# RULE 18: transport parity between main.py and main_pcm.py
# --------------------------------------------------------------------- #

class TestTransportParity:
    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        # The actual mechanical guarantee tools/make_pcm_variant.py enforces
        # on every regeneration (it refuses to write otherwise) -- checked
        # again here so a future hand-edit to either file that breaks this
        # is caught by the ordinary test suite, not only by remembering to
        # re-run the generator.
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb

    def test_main_dot_py_has_the_report_status_and_report_send_branches(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "report_status"' in src
        assert 'intent == "report_send"' in src
        assert 'intent == "test_sample"' in src  # parity fix, see main.py's comment


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
