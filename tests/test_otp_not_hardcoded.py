"""ADDED BY SOURAV -- "otp will not be hardcoded" (the user's own
explicit instruction, replacing clinic-api/main.py's old FRESH_OTP_CODE
constant and clinic-api/seed.py's fixed literal ReportOTP codes).

Covers the two new production code paths this introduced:

  1. models.generate_otp_code() -- the single shared random-OTP
     generator every ReportOTP row (seeded or freshly minted live) now
     goes through instead of a fixed literal.

  2. clinic-api/otp_messaging_config.py's send_otp_via_provider() -- the
     one file a deploying company edits to connect this system's OTP
     delivery to their own SMS/WhatsApp/e-mail provider. Covers both
     halves of its contract: it actually POSTs {phone, otp_code} when a
     URL is configured, and it is a safe, non-raising no-op (returning
     False) whenever no URL is set or the provider call fails -- a
     company's provider being down must never block the OTP flow itself.

tests/test_clinic_api_reports.py already exercises generate_otp_code()
indirectly (every OTP-related assertion there now reads a row's real
code back from the database instead of assuming a literal) -- this file
is the direct, focused coverage for the two building blocks themselves.
"""
from __future__ import annotations

import os
import re
import sys
import types

import httpx
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CLINIC_API_DIR = os.path.join(TESTS_DIR, "..", "clinic-api")

_SIX_DIGITS = re.compile(r"^\d{6}$")


@pytest.fixture
def clinic_api_modules():
    """Duplicated import-and-reset pattern (not shared across files), per
    this repo's own one-fixture-per-file convention (see
    tests/test_live_test_price_lookup.py's own fixture docstring)."""
    sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("models", "otp_messaging_config"):
        sys.modules.pop(mod, None)
    import models as clinic_models
    import otp_messaging_config as otp_config

    yield types.SimpleNamespace(models=clinic_models, otp_config=otp_config)

    sys.path.remove(CLINIC_API_DIR)


# --------------------------------------------------------------------- #
# models.generate_otp_code()
# --------------------------------------------------------------------- #

class TestGenerateOtpCode:
    def test_is_always_exactly_six_digits(self, clinic_api_modules):
        for _ in range(200):
            code = clinic_api_modules.models.generate_otp_code()
            assert _SIX_DIGITS.match(code), repr(code)

    def test_is_not_a_fixed_value_across_calls(self, clinic_api_modules):
        # The whole point of this story: a real random value, never a
        # constant like the old FRESH_OTP_CODE = "135790". With 1,000,000
        # possible codes, 100 draws landing on fewer than 2 distinct
        # values would mean this is still effectively hardcoded.
        codes = {clinic_api_modules.models.generate_otp_code() for _ in range(100)}
        assert len(codes) > 1

    def test_zero_pads_small_values_instead_of_truncating_to_fewer_digits(
        self, clinic_api_modules, monkeypatch
    ):
        # Deterministic check that leading zeros survive (a caller
        # speaking "zero four two nine one three" back is a real Bengali/
        # Hinglish/Banglish utterance shape -- agent/slot_parse.py's
        # parse_otp() already handles it; this just proves the generator
        # can actually produce that shape in the first place).
        monkeypatch.setattr(clinic_api_modules.models.secrets, "randbelow", lambda n: 42)
        assert clinic_api_modules.models.generate_otp_code() == "000042"


# --------------------------------------------------------------------- #
# otp_messaging_config.send_otp_via_provider()
# --------------------------------------------------------------------- #

class TestSendOtpViaProviderWithNoUrlConfigured:
    def test_returns_false_without_raising(self, clinic_api_modules, monkeypatch):
        monkeypatch.setattr(clinic_api_modules.otp_config, "OTP_MESSAGING_WEBHOOK_URL", "")
        result = clinic_api_modules.otp_config.send_otp_via_provider("9000000001", "123456")
        assert result is False

    def test_never_logs_the_actual_otp_code(self, clinic_api_modules, monkeypatch, caplog):
        monkeypatch.setattr(clinic_api_modules.otp_config, "OTP_MESSAGING_WEBHOOK_URL", "")
        with caplog.at_level("INFO", logger="clinic_api.otp_messaging"):
            clinic_api_modules.otp_config.send_otp_via_provider("9000000001", "739284")
        assert "739284" not in caplog.text


class TestSendOtpViaProviderWithUrlConfigured:
    def test_posts_phone_and_code_to_the_configured_url(self, clinic_api_modules, monkeypatch):
        captured = {}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return httpx.Response(200, request=httpx.Request("POST", url))

        monkeypatch.setattr(
            clinic_api_modules.otp_config, "OTP_MESSAGING_WEBHOOK_URL",
            "https://example-provider.test/send-otp",
        )
        monkeypatch.setattr(clinic_api_modules.otp_config.httpx, "post", fake_post)

        result = clinic_api_modules.otp_config.send_otp_via_provider("9000000001", "482913")

        assert result is True
        assert captured["url"] == "https://example-provider.test/send-otp"
        assert captured["json"] == {"phone": "9000000001", "otp_code": "482913"}

    def test_a_provider_failure_returns_false_without_raising(self, clinic_api_modules, monkeypatch):
        # A company's provider being briefly down (network error, 500,
        # timeout) must never propagate as an exception -- see this
        # file's own module docstring ("WHY A PROVIDER FAILURE NEVER
        # BLOCKS THE OTP FLOW").
        def fake_post(url, json, timeout):
            raise httpx.ConnectTimeout("boom", request=httpx.Request("POST", url))

        monkeypatch.setattr(
            clinic_api_modules.otp_config, "OTP_MESSAGING_WEBHOOK_URL",
            "https://example-provider.test/send-otp",
        )
        monkeypatch.setattr(clinic_api_modules.otp_config.httpx, "post", fake_post)

        result = clinic_api_modules.otp_config.send_otp_via_provider("9000000001", "482913")
        assert result is False

    def test_a_non_2xx_response_returns_false_without_raising(self, clinic_api_modules, monkeypatch):
        def fake_post(url, json, timeout):
            return httpx.Response(500, request=httpx.Request("POST", url))

        monkeypatch.setattr(
            clinic_api_modules.otp_config, "OTP_MESSAGING_WEBHOOK_URL",
            "https://example-provider.test/send-otp",
        )
        monkeypatch.setattr(clinic_api_modules.otp_config.httpx, "post", fake_post)

        result = clinic_api_modules.otp_config.send_otp_via_provider("9000000001", "482913")
        assert result is False

    def test_never_logs_the_actual_otp_code_even_on_failure(self, clinic_api_modules, monkeypatch, caplog):
        def fake_post(url, json, timeout):
            raise httpx.ConnectTimeout("boom", request=httpx.Request("POST", url))

        monkeypatch.setattr(
            clinic_api_modules.otp_config, "OTP_MESSAGING_WEBHOOK_URL",
            "https://example-provider.test/send-otp",
        )
        monkeypatch.setattr(clinic_api_modules.otp_config.httpx, "post", fake_post)

        with caplog.at_level("ERROR", logger="clinic_api.otp_messaging"):
            clinic_api_modules.otp_config.send_otp_via_provider("9000000001", "955114")
        assert "955114" not in caplog.text


# --------------------------------------------------------------------- #
# clinic-api/main.py -- request_report_delivery() actually calls
# send_otp_via_provider() every time, and never lets its outcome affect
# the endpoint's own success response.
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real_clinic_api(tmp_path_factory):
    """Duplicated fixture (not shared across files), per this repo's own
    established one-fixture-per-file convention (see
    tests/test_clinic_api_reports.py's own identical fixture)."""
    db_path = tmp_path_factory.mktemp("clinic_otp_messaging_live") / "clinic.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("db", "models", "seed", "main", "otp_messaging_config"):
        sys.modules.pop(mod, None)
    import db as clinic_db
    import models as clinic_models
    clinic_models.Base.metadata.create_all(clinic_db.engine)
    from seed import seed
    seed()

    import main as clinic_main
    from fastapi.testclient import TestClient
    client = TestClient(clinic_main.app)

    yield types.SimpleNamespace(client=client, db=clinic_db, models=clinic_models, main=clinic_main)

    sys.path.remove(CLINIC_API_DIR)


class TestRequestDeliveryCallsSendOtpViaProvider:
    def test_a_successful_delivery_request_calls_it_with_the_real_code(
        self, real_clinic_api, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            real_clinic_api.main, "send_otp_via_provider",
            lambda phone, otp_code: calls.append((phone, otp_code)) or True,
        )

        r = real_clinic_api.client.post(
            "/api/v1/reports/delivery/request",
            json={"phone": "9000000009", "report_number": "RPT-10011"},
        ).json()
        assert r["success"] is True

        with real_clinic_api.db.SessionLocal() as db:
            report = db.query(real_clinic_api.models.LabReport).filter_by(
                report_number="RPT-10011").first()
            otp_row = (
                db.query(real_clinic_api.models.ReportOTP)
                .filter_by(report_id=report.id, used=False)
                .order_by(real_clinic_api.models.ReportOTP.created_at.desc())
                .first()
            )

        assert calls == [("9000000009", otp_row.otp_code)]

    def test_a_provider_that_raises_never_breaks_the_delivery_request_itself(
        self, real_clinic_api, monkeypatch
    ):
        # "Fail safe, not fail open", but pointed the other way: the OTP
        # row is already committed by the time this is called, so a
        # messaging-provider failure must never turn a real success into
        # a reported failure. send_otp_via_provider()'s own real
        # implementation already promises never to raise (see
        # TestSendOtpViaProviderWithUrlConfigured above) -- this
        # deliberately-broken stub proves request_report_delivery() does
        # not rest ENTIRELY on that promise holding forever: its own
        # try/except around the call site is real defense in depth, not
        # dead code.
        monkeypatch.setattr(
            real_clinic_api.main, "send_otp_via_provider",
            lambda phone, otp_code: (_ for _ in ()).throw(RuntimeError("provider exploded")),
        )

        r = real_clinic_api.client.post(
            "/api/v1/reports/delivery/request",
            json={"phone": "9000000010", "report_number": "RPT-10012"},
        ).json()
        assert r["success"] is True
        assert r["reason"] == "OTP_REQUIRED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
