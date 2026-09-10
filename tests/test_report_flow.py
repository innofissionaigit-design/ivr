"""ADDED BY SOURAV -- unit tests for agent/report_flow.py, the pure
decision-logic module backing the "Lab Report Status & Secure Delivery"
combined story (previously two separate stories: "is my report ready" /
"send my report"). See that module's docstring for why every function in
it is a plain (dict/str in) -> (text, pending) out function with no I/O --
that is exactly what makes this file possible: every RULE this story's
plan lists as a "backend decision" (RULE 1-9, 13, 15, 16) can be asserted
directly here, with no CallSession, no WebSocket, no asyncio, and no real
clinic-api call anywhere in sight. main.py/main_pcm.py dispatch tests
(tests/test_report_status_send_dispatch.py) then only need to prove they
plumb these results through correctly, not re-verify the business rules a
second time.
"""
from __future__ import annotations

from agent.report_flow import (
    AWAITING_REPORT_PHONE,
    AWAITING_WHICH_REPORT,
    AWAITING_CONFIRM_DELIVERY,
    AWAITING_OTP_CODE,
    interpret_report_status_result,
    interpret_delivery_request_result,
    interpret_otp_verify_result,
    match_candidate_report,
)


# --------------------------------------------------------------------- #
# interpret_report_status_result
# --------------------------------------------------------------------- #

class TestPatientNotFound:
    def test_report_status_flow(self):
        text, pending = interpret_report_status_result({"patient_found": False}, "report_status")
        assert pending is None
        assert text

    def test_report_send_flow_same_reply(self):
        # RULE 1's identity equivalent: never guess who the caller is,
        # regardless of which intent triggered the lookup.
        a, _ = interpret_report_status_result({"patient_found": False}, "report_status")
        b, _ = interpret_report_status_result({"patient_found": False}, "report_send")
        assert a == b


class TestReportNotFound:
    def test_ends_the_flow(self):
        result = {"patient_found": True, "found": False, "reason": "NOT_FOUND"}
        text, pending = interpret_report_status_result(result, "report_status")
        assert pending is None
        assert text


class TestAmbiguousMultipleReports:
    """RULE 13: caller has more than one report matching -- ask which,
    using safe identifying information (test names), never silently
    picking one."""

    def test_offers_which_report_pending_with_candidates(self):
        candidates = [
            {"report_number": "RPT-10004", "test_name": "CBC", "status": "READY"},
            {"report_number": "RPT-10010", "test_name": "TSH", "status": "READY"},
        ]
        result = {"patient_found": True, "found": False, "reason": "AMBIGUOUS", "candidates": candidates}
        text, pending = interpret_report_status_result(result, "report_status")
        assert pending["awaiting"] == AWAITING_WHICH_REPORT
        assert pending["flow"] == "report_status"
        assert pending["candidates"] == candidates
        assert pending["retries"] == 0
        assert text

    def test_report_send_flow_also_asks_which_before_sending_anything(self):
        # RULE 13 applies identically regardless of which intent got here
        # -- a caller who opens with "send my report" but has two reports
        # must still be asked which, not have one guessed for delivery.
        candidates = [{"report_number": "RPT-A", "test_name": "CBC", "status": "READY"}]
        result = {"patient_found": True, "found": False, "reason": "AMBIGUOUS", "candidates": candidates}
        _, pending = interpret_report_status_result(result, "report_send")
        assert pending["awaiting"] == AWAITING_WHICH_REPORT
        assert pending["flow"] == "report_send"


class TestNotReadyStatuses:
    """RULE 2/3: NOT_READY / PROCESSING / CANCELLED must never lead to an
    OTP or delivery flow, only a truthful status reply."""

    def _result(self, status):
        return {
            "patient_found": True, "found": True, "status": status,
            "delivery_enabled": True, "report_number": "RPT-1",
        }

    def test_not_ready_report_status_just_answers(self):
        text, pending = interpret_report_status_result(self._result("NOT_READY"), "report_status")
        assert pending is None
        assert text

    def test_not_ready_report_send_is_blocked_not_started(self):
        text, pending = interpret_report_status_result(self._result("NOT_READY"), "report_send")
        assert pending is None
        assert text

    def test_processing_report_send_is_blocked(self):
        text, pending = interpret_report_status_result(self._result("PROCESSING"), "report_send")
        assert pending is None

    def test_cancelled_report_send_is_blocked(self):
        text, pending = interpret_report_status_result(self._result("CANCELLED"), "report_send")
        assert pending is None


class TestDeliveryDisabled:
    """RULE 16: a READY report with delivery_enabled=False must never
    start an OTP flow, on either intent -- and the internal flag itself
    must never leak into the spoken reply (asserted in
    test_report_reply_templates.py; here we only assert the FLOW
    decision)."""

    def _result(self):
        return {
            "patient_found": True, "found": True, "status": "READY",
            "delivery_enabled": False, "report_number": "RPT-I",
        }

    def test_report_status_reports_status_with_no_offer(self):
        text, pending = interpret_report_status_result(self._result(), "report_status")
        assert pending is None
        assert text

    def test_report_send_is_blocked(self):
        text, pending = interpret_report_status_result(self._result(), "report_send")
        assert pending is None
        assert text


class TestReadyAndDeliveryEnabled:
    """RULE 4: OTP/delivery only ever starts after an explicit yes for
    report_status (the "shall I send it?" offer); report_send skips
    straight to the sentinel that tells the transport file to call
    request_report_delivery() immediately, since the caller already asked."""

    def _result(self):
        return {
            "patient_found": True, "found": True, "status": "READY",
            "delivery_enabled": True, "report_number": "RPT-A",
        }

    def test_report_status_offers_and_waits_for_yes(self):
        text, pending = interpret_report_status_result(self._result(), "report_status")
        assert pending["awaiting"] == AWAITING_CONFIRM_DELIVERY
        assert pending["report_number"] == "RPT-A"
        assert pending["retries"] == 0
        assert text

    def test_report_send_returns_request_delivery_now_sentinel(self):
        text, pending = interpret_report_status_result(self._result(), "report_send")
        assert text is None
        assert pending == {"awaiting": "__request_delivery_now__", "report_number": "RPT-A"}


# --------------------------------------------------------------------- #
# interpret_delivery_request_result
# --------------------------------------------------------------------- #

class TestDeliveryRequestResult:
    def test_success_moves_to_otp_code_pending_with_report_number_carried(self):
        text, pending = interpret_delivery_request_result(
            {"success": True, "reason": "OTP_REQUIRED", "masked_phone": "xxxxx01"},
            report_number="RPT-A",
        )
        assert pending == {"awaiting": AWAITING_OTP_CODE, "report_number": "RPT-A", "retries": 0}
        assert text

    def test_masked_phone_is_spoken_not_the_full_number(self):
        # RULE 10 -- exercised at the reply-template layer, but the
        # interpret function must at minimum be handed the masked value
        # and not blow up if a full one were ever mistakenly passed.
        text, _ = interpret_delivery_request_result(
            {"success": True, "reason": "OTP_REQUIRED", "masked_phone": "xxxxx01"},
            report_number="RPT-A",
        )
        assert "xxxxx01" in text or "01" in text

    def test_patient_not_found_ends_flow(self):
        text, pending = interpret_delivery_request_result(
            {"success": False, "reason": "PATIENT_NOT_FOUND"}, report_number="RPT-A",
        )
        assert pending is None
        assert text

    def test_report_not_found_ends_flow(self):
        text, pending = interpret_delivery_request_result(
            {"success": False, "reason": "NOT_FOUND"}, report_number="RPT-A",
        )
        assert pending is None

    def test_report_state_changed_between_status_check_and_this_request(self):
        # RULE 3/16 defense in depth -- clinic-api re-checks eligibility
        # server-side even though the transport layer already checked it
        # moments ago; if that ever disagrees (report cancelled in the
        # interim, say), the caller must be told the true current state,
        # never a stale "sending now".
        for reason in ("NOT_READY", "PROCESSING", "CANCELLED", "DELIVERY_DISABLED"):
            text, pending = interpret_delivery_request_result(
                {"success": False, "reason": reason}, report_number="RPT-A",
            )
            assert pending is None, reason
            assert text, reason


# --------------------------------------------------------------------- #
# interpret_otp_verify_result
# --------------------------------------------------------------------- #

class TestOtpVerifyResult:
    def test_delivery_sent_ends_flow(self):
        text, pending = interpret_otp_verify_result(
            {"success": True, "reason": "DELIVERY_SENT", "masked_phone": "xxxxx01",
             "signed_link_expires_minutes": 15},
            report_number="RPT-A",
        )
        assert pending is None
        assert text

    def test_otp_invalid_stays_in_same_otp_code_state(self):
        # RULE 8: a wrong guess re-prompts for another try, bounded by the
        # TRANSPORT file's own retry counter -- this function always resets
        # its own local "retries" to 0 because the authoritative limit is
        # clinic-api's server-side attempt_count/max_attempts (RULE 8),
        # checked again on the NEXT verify call regardless of what this
        # function returns.
        text, pending = interpret_otp_verify_result(
            {"success": False, "reason": "OTP_INVALID"}, report_number="RPT-A",
        )
        assert pending == {"awaiting": AWAITING_OTP_CODE, "report_number": "RPT-A", "retries": 0}
        assert text

    def test_every_other_reason_ends_the_flow(self):
        for reason in ("OTP_EXPIRED", "OTP_ALREADY_USED", "OTP_MAX_ATTEMPTS",
                       "OTP_NOT_REQUESTED", "DELIVERY_FAILED", "NOT_READY",
                       "PROCESSING", "CANCELLED", "DELIVERY_DISABLED",
                       "PATIENT_NOT_FOUND", "NOT_FOUND"):
            text, pending = interpret_otp_verify_result(
                {"success": False, "reason": reason}, report_number="RPT-A",
            )
            assert pending is None, reason
            assert text, reason

    def test_never_reads_or_echoes_an_otp_code_field(self):
        # RULE 9, structural check: even if a caller (or a bug upstream)
        # somehow smuggled an "otp_code" key into the result dict, this
        # function must never surface it in the spoken reply.
        result = {"success": False, "reason": "OTP_INVALID", "otp_code": "999999"}
        text, _ = interpret_otp_verify_result(result, report_number="RPT-A")
        assert "999999" not in text


# --------------------------------------------------------------------- #
# match_candidate_report (RULE 13's disambiguation matcher)
# --------------------------------------------------------------------- #

class TestMatchCandidateReport:
    CANDIDATES = [
        {"report_number": "RPT-10004", "test_name": "CBC"},
        {"report_number": "RPT-10005", "test_name": "Vitamin D (25-OH)"},
        {"report_number": "RPT-10010", "test_name": "TSH"},
    ]

    def test_exact_name_matches(self):
        assert match_candidate_report("CBC", self.CANDIDATES) == "RPT-10004"

    def test_substring_matches(self):
        assert match_candidate_report("vitamin d", self.CANDIDATES) == "RPT-10005"

    def test_case_insensitive(self):
        assert match_candidate_report("tsh", self.CANDIDATES) == "RPT-10010"

    def test_empty_utterance_returns_none(self):
        assert match_candidate_report("", self.CANDIDATES) is None
        assert match_candidate_report("   ", self.CANDIDATES) is None

    def test_no_candidates_returns_none(self):
        assert match_candidate_report("CBC", []) is None

    def test_unrelated_text_returns_none(self):
        assert match_candidate_report("আমি জানি না", self.CANDIDATES) is None

    def test_does_not_confuse_same_patient_different_report_numbers(self):
        # RULE 14's disambiguation-side guarantee: matching is purely by
        # test_name text against THIS candidate list -- there is no path
        # here that could return a report_number belonging to a different
        # patient, since candidates are already scoped to one identity by
        # the time this function ever sees them.
        assert match_candidate_report("cbc", self.CANDIDATES) == "RPT-10004"
        assert match_candidate_report("cbc", self.CANDIDATES) != "RPT-10010"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
