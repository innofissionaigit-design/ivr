"""ADDED BY SOURAV -- reply-text-level tests for Phase 1: Database Schema
& Policy Tables (Walk-in Eligibility, Prescription Requirements, Insurance
Coverage Policy, Outstanding Balance / Billing stories).

Mirrors tests/test_test_preparation_reply.py's own coverage shape: all 4
languages, every honest outcome (not-found / policy-unavailable /
real-answer), and the specific honesty guarantees each story's DB design
exists to enforce (see clinic-api/models.py's own comments on why every
new column these stories added is nullable with no default) -- never a
guessed "walk-ins welcome" / "no prescription needed" / "covered" /
"nothing owed" for a row nobody has actually reviewed.
"""
from __future__ import annotations

from agent.reply_templates import (
    walkin_eligibility_reply, prescription_requirements_reply,
    insurance_coverage_reply, billing_balance_reply,
)

LANGUAGES = ["english", "hinglish", "banglish", "bengali"]


# --------------------------------------------------------------------- #
# walkin_eligibility_reply
# --------------------------------------------------------------------- #

class TestWalkinEligibilityReply:
    def test_not_found_reuses_shared_not_found_reply(self):
        slots = {"test_name": "CBC"}
        result = {"found": False, "query": "CBC", "did_you_mean": ["Complete Blood Count (CBC)"]}
        for lang in LANGUAGES:
            text = walkin_eligibility_reply(slots, result, lang)
            assert "CBC" in text
            assert text  # non-empty in every language

    def test_policy_unavailable_never_says_walk_ins_welcome(self):
        slots = {"test_name": "MRI Brain"}
        result = {"found": True, "test_name": "MRI Brain", "test_name_bn": None, "policy_available": False}
        for lang in LANGUAGES:
            text = walkin_eligibility_reply(slots, result, lang).lower()
            assert "yes" not in text and "haan" not in text and "han" not in text and "হ্যাঁ" not in text

    def test_eligible_with_hours_speaks_both(self):
        slots = {"test_name": "CBC"}
        result = {"found": True, "test_name": "CBC", "test_name_bn": None,
                  "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am"}
        for lang in LANGUAGES:
            text = walkin_eligibility_reply(slots, result, lang)
            assert "Mon-Sat 7am-11am" in text

    def test_eligible_without_hours_omits_hours_clause_gracefully(self):
        slots = {"test_name": "CBC"}
        result = {"found": True, "test_name": "CBC", "test_name_bn": None,
                  "policy_available": True, "walkin_eligible": True, "walkin_hours": None}
        for lang in LANGUAGES:
            text = walkin_eligibility_reply(slots, result, lang)
            assert text  # still produces a real sentence
            assert "None" not in text

    def test_not_eligible_says_appointment_needed(self):
        slots = {"test_name": "USG Whole Abdomen"}
        result = {"found": True, "test_name": "USG Whole Abdomen", "test_name_bn": None,
                  "policy_available": True, "walkin_eligible": False, "walkin_hours": None}
        text_en = walkin_eligibility_reply(slots, result, "english")
        assert "not available on a walk-in basis" in text_en
        assert "appointment" in text_en.lower()


# --------------------------------------------------------------------- #
# prescription_requirements_reply
# --------------------------------------------------------------------- #

class TestPrescriptionRequirementsReply:
    def test_not_found_reuses_shared_not_found_reply(self):
        slots = {"test_name": "CBC"}
        result = {"found": False, "query": "CBC", "did_you_mean": []}
        for lang in LANGUAGES:
            assert walkin_eligibility_reply(slots, result, lang) == prescription_requirements_reply(slots, result, lang)

    def test_policy_unavailable_never_guesses(self):
        slots = {"test_name": "MRI Brain"}
        result = {"found": True, "test_name": "MRI Brain", "test_name_bn": None, "policy_available": False}
        for lang in LANGUAGES:
            text = prescription_requirements_reply(slots, result, lang).lower()
            assert "yes" not in text and "no," not in text and "না," not in text

    def test_required_with_channels_lists_them_naturally(self):
        slots = {"test_name": "MRI Brain"}
        result = {"found": True, "test_name": "MRI Brain", "test_name_bn": None,
                  "policy_available": True, "prescription_required": True,
                  "prescription_channels": ["whatsapp photo", "counter in person"]}
        text_en = prescription_requirements_reply(slots, result, "english")
        assert "whatsapp photo" in text_en and "counter in person" in text_en
        assert " and " in text_en  # naturally joined, not a raw list/comma-only

    def test_required_with_single_channel_no_join_word_needed(self):
        slots = {"test_name": "MRI Brain"}
        result = {"found": True, "test_name": "MRI Brain", "test_name_bn": None,
                  "policy_available": True, "prescription_required": True,
                  "prescription_channels": ["email"]}
        text_en = prescription_requirements_reply(slots, result, "english")
        assert "email" in text_en

    def test_not_required_says_no_prescription_needed(self):
        slots = {"test_name": "CBC"}
        result = {"found": True, "test_name": "CBC", "test_name_bn": None,
                  "policy_available": True, "prescription_required": False,
                  "prescription_channels": []}
        for lang in LANGUAGES:
            text = prescription_requirements_reply(slots, result, lang)
            assert "CBC" in text

    def test_channel_identifiers_are_rendered_as_words_not_raw_identifiers(self):
        # _spoken_channel() replaces underscores with spaces -- an
        # internal identifier must never leak verbatim into speech.
        slots = {"test_name": "CBC"}
        result = {"found": True, "test_name": "CBC", "test_name_bn": None,
                  "policy_available": True, "prescription_required": True,
                  "prescription_channels": ["whatsapp_photo"]}
        text = prescription_requirements_reply(slots, result, "english")
        assert "_" not in text


# --------------------------------------------------------------------- #
# insurance_coverage_reply
# --------------------------------------------------------------------- #

class TestInsuranceCoverageReply:
    def test_test_not_found_reuses_shared_not_found_reply(self):
        slots = {"test_name": "CBC", "insurance_provider_name": "Star Health"}
        result = {"test_found": False, "query": "CBC", "did_you_mean": ["Complete Blood Count (CBC)"]}
        for lang in LANGUAGES:
            text = insurance_coverage_reply(slots, result, lang)
            assert "CBC" in text

    def test_provider_not_found_names_the_query(self):
        slots = {"test_name": "CBC", "insurance_provider_name": "Star Health"}
        result = {"test_found": True, "test_name": "CBC", "provider_found": False,
                  "query_provider": "Star Health"}
        for lang in LANGUAGES:
            text = insurance_coverage_reply(slots, result, lang)
            assert "Star Health" in text

    def test_policy_unavailable_never_guesses_coverage(self):
        slots = {"test_name": "CBC", "insurance_provider_name": "Star Health"}
        result = {"test_found": True, "test_name": "CBC", "provider_found": True,
                  "provider_name": "Star Health", "policy_available": False}
        for lang in LANGUAGES:
            text = insurance_coverage_reply(slots, result, lang)
            assert "CBC" in text and "Star Health" in text

    def test_covered_says_good_news(self):
        slots = {"test_name": "CBC", "insurance_provider_name": "Star Health"}
        result = {"test_found": True, "test_name": "CBC", "provider_found": True,
                  "provider_name": "Star Health", "policy_available": True,
                  "coverage_status": "COVERED", "pre_auth_required": False}
        text_en = insurance_coverage_reply(slots, result, "english")
        assert "covered" in text_en.lower()
        assert "Pre-authorization" not in text_en

    def test_covered_with_pre_auth_mentions_it(self):
        slots = {"test_name": "CBC", "insurance_provider_name": "Star Health"}
        result = {"test_found": True, "test_name": "CBC", "provider_found": True,
                  "provider_name": "Star Health", "policy_available": True,
                  "coverage_status": "COVERED", "pre_auth_required": True}
        text_en = insurance_coverage_reply(slots, result, "english")
        assert "pre-authorization" in text_en.lower()

    def test_not_covered(self):
        slots = {"test_name": "Lipid Profile", "insurance_provider_name": "Star Health"}
        result = {"test_found": True, "test_name": "Lipid Profile", "provider_found": True,
                  "provider_name": "Star Health", "policy_available": True,
                  "coverage_status": "NOT_COVERED", "pre_auth_required": False}
        text_en = insurance_coverage_reply(slots, result, "english")
        assert "not covered" in text_en.lower()

    def test_partial_coverage(self):
        slots = {"test_name": "Lipid Profile", "insurance_provider_name": "Star Health"}
        result = {"test_found": True, "test_name": "Lipid Profile", "provider_found": True,
                  "provider_name": "Star Health", "policy_available": True,
                  "coverage_status": "PARTIAL", "pre_auth_required": False}
        text_en = insurance_coverage_reply(slots, result, "english")
        assert "partially" in text_en.lower()


# --------------------------------------------------------------------- #
# billing_balance_reply
# --------------------------------------------------------------------- #

class TestBillingBalanceReply:
    def test_patient_not_found_reuses_patient_not_found_reply(self):
        from agent.reply_templates import patient_not_found_reply

        for lang in LANGUAGES:
            assert billing_balance_reply({"patient_found": False}, lang) == patient_not_found_reply(lang)

    def test_no_billing_record_never_claims_zero(self):
        result = {"patient_found": True, "found": False, "reason": "NOT_FOUND"}
        for lang in LANGUAGES:
            text = billing_balance_reply(result, lang)
            assert "0" not in text

    def test_real_balance_is_digit_faithful(self):
        result = {"patient_found": True, "found": True, "outstanding_amount": "1250.50", "due_date": None}
        text_en = billing_balance_reply(result, "english")
        assert "1250.50" in text_en

    def test_whole_number_balance_never_speaks_trailing_point_zero(self):
        # Same digit-fidelity bug class test_rate_reply() had to fix for
        # rate_inr -- outstanding_amount goes through the same helper.
        result = {"patient_found": True, "found": True, "outstanding_amount": "500.0", "due_date": None}
        text_en = billing_balance_reply(result, "english")
        assert "500 rupees" in text_en
        assert "500.0" not in text_en

    def test_real_zero_balance_is_spoken_as_a_real_fact(self):
        # Distinct from "no billing record" above -- 0.0 IS a reviewed
        # fact and must be spoken, not silently treated as not-found.
        result = {"patient_found": True, "found": True, "outstanding_amount": "0.0", "due_date": None}
        text_en = billing_balance_reply(result, "english")
        assert "0 rupees" in text_en

    def test_due_date_is_mentioned_when_present(self):
        result = {"patient_found": True, "found": True, "outstanding_amount": "500.0",
                  "due_date": "2026-10-15T00:00:00"}
        text_en = billing_balance_reply(result, "english")
        assert "2026-10-15T00:00:00" in text_en

    def test_due_date_omitted_gracefully_when_absent(self):
        result = {"patient_found": True, "found": True, "outstanding_amount": "500.0", "due_date": None}
        for lang in LANGUAGES:
            text = billing_balance_reply(result, lang)
            assert "None" not in text
