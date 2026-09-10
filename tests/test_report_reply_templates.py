"""ADDED BY SOURAV -- targeted tests for the report_status/report_send
reply functions added to agent/reply_templates.py. Not a full duplicate of
tests/test_reply_templates_fidelity.py's fidelity sweep (spoken-punctuation
checks etc. -- those already run over every function in the module,
including these, automatically); this file is about the RULES specific to
this story that a generic fidelity sweep would not catch: RULE 9 (never
reveal the OTP), RULE 10 (masked phone only), RULE 16 (never name the
internal delivery_enabled flag), and multilingual coverage of the new
branches.
"""
from __future__ import annotations

import inspect

from agent.reply_templates import (
    patient_not_found_reply,
    report_not_found_reply,
    report_ambiguous_reply,
    report_status_reply,
    delivery_blocked_reply,
    delivery_declined_reply,
    otp_requested_reply,
    otp_disclosure_refusal_reply,
    otp_verify_reply,
)

LANGUAGES = ("english", "hinglish", "banglish", "bengali")


class TestNoLanguageBranchCrashes:
    """Every new function must handle all four languages without raising
    and without silently falling through to an empty string."""

    def test_patient_not_found(self):
        for lang in LANGUAGES:
            assert patient_not_found_reply(lang)

    def test_report_not_found(self):
        for lang in LANGUAGES:
            assert report_not_found_reply(lang)

    def test_report_ambiguous(self):
        result = {"candidates": [{"test_name": "CBC"}, {"test_name": "TSH"}]}
        for lang in LANGUAGES:
            text = report_ambiguous_reply(result, lang)
            assert text
            assert "CBC" in text and "TSH" in text

    def test_report_status_ready_and_enabled(self):
        result = {"test_name": "CBC", "status": "READY", "delivery_enabled": True}
        for lang in LANGUAGES:
            assert report_status_reply(result, lang)

    def test_report_status_ready_not_enabled(self):
        result = {"test_name": "KFT", "status": "READY", "delivery_enabled": False}
        for lang in LANGUAGES:
            assert report_status_reply(result, lang)

    def test_report_status_not_ready_processing_cancelled(self):
        for status in ("NOT_READY", "PROCESSING", "CANCELLED"):
            result = {"test_name": "X", "status": status}
            for lang in LANGUAGES:
                assert report_status_reply(result, lang)

    def test_delivery_blocked_every_reason(self):
        for reason in ("NOT_READY", "PROCESSING", "CANCELLED", "DELIVERY_DISABLED"):
            for lang in LANGUAGES:
                assert delivery_blocked_reply(reason, lang)

    def test_delivery_declined(self):
        for lang in LANGUAGES:
            assert delivery_declined_reply(lang)

    def test_otp_requested(self):
        result = {"masked_phone": "0001"}
        for lang in LANGUAGES:
            assert otp_requested_reply(result, lang)

    def test_otp_disclosure_refusal(self):
        for lang in LANGUAGES:
            assert otp_disclosure_refusal_reply(lang)

    def test_otp_verify_every_reason(self):
        reasons = [
            {"success": True, "reason": "DELIVERY_SENT", "masked_phone": "0001",
             "signed_link_expires_minutes": 15},
            {"success": False, "reason": "OTP_INVALID"},
            {"success": False, "reason": "OTP_EXPIRED"},
            {"success": False, "reason": "OTP_ALREADY_USED"},
            {"success": False, "reason": "OTP_MAX_ATTEMPTS"},
            {"success": False, "reason": "OTP_NOT_REQUESTED"},
            {"success": False, "reason": "DELIVERY_FAILED"},
            {"success": False, "reason": "NOT_READY"},
            {"success": False, "reason": "PROCESSING"},
            {"success": False, "reason": "CANCELLED"},
            {"success": False, "reason": "DELIVERY_DISABLED"},
            {"success": False, "reason": "PATIENT_NOT_FOUND"},
            {"success": False, "reason": "NOT_FOUND"},
        ]
        for result in reasons:
            for lang in LANGUAGES:
                text = otp_verify_reply(result, lang)
                assert text, result


class TestRule9NeverRevealsOtp:
    """Structural check: no function in this module's report_* / otp_*
    block ever reads an "otp_code" key out of any dict it is handed, even
    if one is present (a caller-facing reply must never be able to leak
    it, whether by design or by an upstream bug smuggling the field in)."""

    FUNCTIONS_TAKING_A_RESULT_DICT = [
        report_ambiguous_reply, report_status_reply, otp_requested_reply,
        otp_verify_reply,
    ]

    def test_source_never_reads_otp_code(self):
        for fn in self.FUNCTIONS_TAKING_A_RESULT_DICT:
            src = inspect.getsource(fn)
            assert "otp_code" not in src, fn.__name__

    def test_otp_verify_reply_ignores_a_smuggled_otp_code_field(self):
        result = {"success": False, "reason": "OTP_INVALID", "otp_code": "999999"}
        text = otp_verify_reply(result)
        assert "999999" not in text

    def test_otp_requested_reply_ignores_a_smuggled_otp_code_field(self):
        result = {"masked_phone": "0001", "otp_code": "999999"}
        text = otp_requested_reply(result)
        assert "999999" not in text


class TestRule10MaskedPhoneOnly:
    def test_otp_requested_reply_speaks_only_the_masked_value(self):
        result = {"masked_phone": "0001"}
        text = otp_requested_reply(result)
        assert "0001" in text
        # No branch of this function has any other phone-shaped field to
        # pull from -- the full number was never even passed in.
        assert "9000000001" not in text


class TestRule16NeverNamesTheInternalFlag:
    """The literal string "delivery_enabled" (or any English mention of
    a flag/setting) must never appear in a spoken reply -- Patient I
    hears a true status and, for report_send, a blocked-delivery message,
    never an explanation that a database column disabled it."""

    def test_report_status_reply_ready_not_enabled_never_names_the_flag(self):
        result = {"test_name": "KFT", "status": "READY", "delivery_enabled": False}
        for lang in LANGUAGES:
            text = report_status_reply(result, lang).lower()
            assert "delivery_enabled" not in text
            assert "flag" not in text
            assert "disabled" not in text  # english word for the setting itself

    def test_delivery_blocked_reply_disabled_never_names_the_flag(self):
        for lang in LANGUAGES:
            text = delivery_blocked_reply("DELIVERY_DISABLED", lang).lower()
            assert "delivery_enabled" not in text
            assert "flag" not in text


class TestRule2NoClinicalValueEverAppears:
    """report_status_reply's own docstring claims this holds structurally
    because `result` never carries a clinical field -- verified here by
    feeding it a `result` that (if the function ever DID read such a
    field) would leak a fabricated clinical value into the reply."""

    def test_a_stray_clinical_looking_field_is_never_spoken(self):
        result = {
            "test_name": "CBC", "status": "READY", "delivery_enabled": True,
            "hemoglobin": "9.1 g/dL", "diagnosis": "anemia",
        }
        for lang in LANGUAGES:
            text = report_status_reply(result, lang)
            assert "9.1" not in text
            assert "anemia" not in text.lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
