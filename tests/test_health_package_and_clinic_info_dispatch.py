"""ADDED BY SOURAV -- dispatch-level and real-backend tests for "Caller
asks about a health package" combined with "Caller asks opening hours,
address or directions" (Epic: Conversation -- Information and Enquiry).

Mirrors tests/test_doctor_schedule_dispatch.py's own two-half structure
(see that file's module docstring for the full rationale):

  1. TestDispatchWithFakeTools -- drives main_pcm._dispatch_turn() with a
     stubbed ClinicToolsClient, isolating main.py/main_pcm.py's own
     dispatch-branch logic: which tool method gets called for each intent
     and slot combination, and which reply function renders the result.

  2. TestAgainstRealClinicApi -- boots the REAL clinic-api FastAPI app on
     a real temporary SQLite database seeded with the REAL seed.py data,
     driving the same dispatch end to end against genuine seeded rows.

TestTransportParity closes the loop the same way that file's does: proves
main.py and main_pcm.py's shared "reasoning half" stayed byte-identical
and that both files actually contain the new branches.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import sys
import types

import pytest

import main_pcm

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CLINIC_API_DIR = os.path.join(TESTS_DIR, "..", "clinic-api")


def run(coro):
    return asyncio.run(coro)


class FakeToolsClient:
    def __init__(self):
        self.search_health_package_calls = []
        self.search_health_package_response = None
        self.get_health_packages_calls = 0
        self.get_health_packages_response = None
        self.get_clinic_info_calls = 0
        self.get_clinic_info_response = None

    async def search_health_package(self, package_name):
        self.search_health_package_calls.append(package_name)
        return self.search_health_package_response

    async def get_health_packages(self):
        self.get_health_packages_calls += 1
        return self.get_health_packages_response

    async def get_clinic_info(self):
        self.get_clinic_info_calls += 1
        return self.get_clinic_info_response


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session():
    return types.SimpleNamespace(
        call_id="test-health-package-clinic-info-call",
        pending=None,
        dispatch_lock=asyncio.Lock(),
        send_json=_AsyncNoOp(),
    )


class FakeASRResult:
    # Real Bengali text -- see test_doctor_schedule_dispatch.py's own
    # FakeASRResult comment for why this must not be the old English
    # placeholder now that dispatch calls detect_language() on it.
    text = "হেলথ প্যাকেজ সম্পর্কে জানতে চাই"


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
    session = make_session()
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return session


class TestHealthPackageDispatch:
    def test_named_package_calls_search_not_list(self, monkeypatch, env, tmp_path):
        env.tools.search_health_package_response = {
            "found": True, "package_name": "Diabetes Screening Package",
            "package_name_bn": "diabetes checkup", "description": "desc",
            "price_inr": "1299.0", "tests": ["Blood Sugar Fasting"], "tests_bn": ["blood sugar fasting"],
        }
        session = dispatch_with_intent(
            monkeypatch, "health_package", {"package_name": "Diabetes Screening Package"}, tmp_path,
        )
        assert env.tools.search_health_package_calls == ["Diabetes Screening Package"]
        assert env.tools.get_health_packages_calls == 0
        assert len(env.spoken) == 1
        assert "1299" in env.spoken[0]
        assert session.pending is None

    def test_no_package_named_calls_list_not_search(self, monkeypatch, env, tmp_path):
        env.tools.get_health_packages_response = {
            "packages": [
                {"found": True, "package_name": "Basic Health Checkup", "package_name_bn": "basic checkup",
                 "description": "desc", "price_inr": "999.0",
                 "tests": ["Complete Blood Count (CBC)"], "tests_bn": ["cbc"]},
            ]
        }
        session = dispatch_with_intent(
            monkeypatch, "health_package", {"package_name": None}, tmp_path,
        )
        assert env.tools.search_health_package_calls == []
        assert env.tools.get_health_packages_calls == 1
        assert len(env.spoken) == 1
        # FakeASRResult's utterance is Bengali (see its own docstring, and
        # test_doctor_schedule_dispatch.py's identical precedent) -- the
        # dispatched language is therefore "bengali", so the reply speaks
        # the package's Bengali alias, not its English catalogue name
        # (see agent/reply_templates.py::_spoken_package_name()).
        assert "basic checkup" in env.spoken[0]
        assert session.pending is None

    def test_package_not_found_speaks_the_not_found_reply(self, monkeypatch, env, tmp_path):
        env.tools.search_health_package_response = {
            "found": False, "query": "Ghost Package", "did_you_mean": [],
        }
        session = dispatch_with_intent(
            monkeypatch, "health_package", {"package_name": "Ghost Package"}, tmp_path,
        )
        assert len(env.spoken) == 1
        assert "Ghost Package" in env.spoken[0]
        assert session.pending is None

    def test_search_tool_failure_gives_the_shared_infrastructure_apology(self, monkeypatch, env, tmp_path):
        from agent.tools_client import ToolCallError

        async def failing_search(package_name):
            raise ToolCallError("boom")

        env.tools.search_health_package = failing_search
        session = dispatch_with_intent(
            monkeypatch, "health_package", {"package_name": "Diabetes Screening Package"}, tmp_path,
        )
        assert len(env.spoken) == 1
        assert session.pending is None

    def test_list_tool_failure_gives_the_shared_infrastructure_apology(self, monkeypatch, env, tmp_path):
        from agent.tools_client import ToolCallError

        async def failing_list():
            raise ToolCallError("boom")

        env.tools.get_health_packages = failing_list
        session = dispatch_with_intent(
            monkeypatch, "health_package", {"package_name": None}, tmp_path,
        )
        assert len(env.spoken) == 1
        assert session.pending is None


class TestClinicInfoDispatch:
    def _result(self, **overrides):
        base = {
            "found": True,
            "clinic_name": "Kolkata Care Polyclinic",
            "phone": "03322223333",
            "address": "42 Lake View Road, Kolkata, West Bengal 700029",
            "directions": "Opposite Lake View Metro Station, next to City Pharmacy.",
            "hours": {
                day: {"closed": False, "open": "08:00", "close": "20:00"}
                for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday")
            },
        }
        base["hours"]["sunday"] = {"closed": True, "open": None, "close": None}
        base.update(overrides)
        return base

    @pytest.mark.parametrize("topic", ["hours", "address", "directions", None])
    def test_always_calls_get_clinic_info_regardless_of_topic(self, monkeypatch, env, tmp_path, topic):
        env.tools.get_clinic_info_response = self._result()
        session = dispatch_with_intent(
            monkeypatch, "clinic_info", {"info_topic": topic}, tmp_path,
        )
        assert env.tools.get_clinic_info_calls == 1
        assert len(env.spoken) == 1
        assert session.pending is None

    def test_today_weekday_is_resolved_in_dispatch_not_left_to_the_reply_template(
        self, monkeypatch, env, tmp_path
    ):
        # main.py resolves "today" itself (same precedent as
        # doctor_availability's own date_iso default) -- this test proves
        # the reply actually reflects TODAY's real hours, not a fixed or
        # missing day, by cross-checking against a real datetime.date call
        # made directly in the test (same pattern as
        # tests/test_live_insufficient_information_demo.py).
        env.tools.get_clinic_info_response = self._result()
        session = dispatch_with_intent(
            monkeypatch, "clinic_info", {"info_topic": "hours"}, tmp_path,
        )
        today_key = (
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        )[datetime.date.today().weekday()]
        if today_key == "sunday":
            assert "08:00" not in env.spoken[0]
        else:
            assert "08:00" in env.spoken[0]
        assert session.pending is None

    def test_address_topic_speaks_the_address(self, monkeypatch, env, tmp_path):
        env.tools.get_clinic_info_response = self._result()
        session = dispatch_with_intent(
            monkeypatch, "clinic_info", {"info_topic": "address"}, tmp_path,
        )
        assert "42 Lake View Road, Kolkata, West Bengal 700029" in env.spoken[0]
        assert session.pending is None

    def test_directions_topic_speaks_the_directions(self, monkeypatch, env, tmp_path):
        env.tools.get_clinic_info_response = self._result()
        session = dispatch_with_intent(
            monkeypatch, "clinic_info", {"info_topic": "directions"}, tmp_path,
        )
        assert "Opposite Lake View Metro Station, next to City Pharmacy." in env.spoken[0]
        assert session.pending is None

    def test_no_topic_speaks_address_and_directions_together(self, monkeypatch, env, tmp_path):
        env.tools.get_clinic_info_response = self._result()
        session = dispatch_with_intent(
            monkeypatch, "clinic_info", {"info_topic": None}, tmp_path,
        )
        assert "42 Lake View Road, Kolkata, West Bengal 700029" in env.spoken[0]
        assert "Opposite Lake View Metro Station, next to City Pharmacy." in env.spoken[0]
        assert session.pending is None

    def test_not_found_speaks_the_not_found_reply(self, monkeypatch, env, tmp_path):
        env.tools.get_clinic_info_response = {"found": False}
        session = dispatch_with_intent(
            monkeypatch, "clinic_info", {"info_topic": None}, tmp_path,
        )
        assert len(env.spoken) == 1
        assert session.pending is None

    def test_tool_failure_gives_the_shared_infrastructure_apology(self, monkeypatch, env, tmp_path):
        from agent.tools_client import ToolCallError

        async def failing_get_clinic_info():
            raise ToolCallError("boom")

        env.tools.get_clinic_info = failing_get_clinic_info
        session = dispatch_with_intent(
            monkeypatch, "clinic_info", {"info_topic": None}, tmp_path,
        )
        assert len(env.spoken) == 1
        assert session.pending is None


@pytest.fixture(scope="module")
def real_clinic_api(tmp_path_factory):
    """Boots the REAL clinic-api FastAPI app against a throwaway SQLite
    file and seeds it with the REAL seed.py data -- duplicated
    self-contained fixture, matching this codebase's own established
    per-file pattern (see test_doctor_schedule_dispatch.py's own fixture
    docstring)."""
    db_path = tmp_path_factory.mktemp("clinic_health_package_info_live") / "clinic.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("db", "models", "seed", "main"):
        sys.modules.pop(mod, None)
    import db as clinic_db
    import models as clinic_models
    clinic_models.Base.metadata.create_all(clinic_db.engine)
    from seed import seed
    seed()

    import main as clinic_main
    from fastapi.testclient import TestClient
    client = TestClient(clinic_main.app)

    yield types.SimpleNamespace(client=client, db=clinic_db, models=clinic_models)

    sys.path.remove(CLINIC_API_DIR)


class RealClinicToolsClient:
    """Stands in for agent.tools_client.ClinicToolsClient -- calls the
    REAL clinic-api app in-process via the FastAPI TestClient, exactly
    like test_doctor_schedule_dispatch.py's own RealClinicToolsClient."""

    def __init__(self, client):
        self._client = client
        self.raw_responses = []

    async def search_health_package(self, package_name: str) -> dict:
        r = self._client.get("/api/v1/health-packages/search", params={"name": package_name})
        result = r.json()
        self.raw_responses.append(dict(result))
        return result

    async def get_health_packages(self) -> dict:
        r = self._client.get("/api/v1/health-packages")
        result = r.json()
        self.raw_responses.append(dict(result))
        return result

    async def get_clinic_info(self) -> dict:
        r = self._client.get("/api/v1/clinic/info")
        result = r.json()
        self.raw_responses.append(dict(result))
        return result


def _dispatch(monkeypatch, tools, intent, slots, tmp_path):
    async def fake_resolve_intent(session, text):
        return {"intent": intent, "slots": slots}

    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(main_pcm, "_asr", FakeASR())
    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    monkeypatch.setattr(main_pcm, "_tools", tools)

    session = make_session()
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return spoken


class TestAgainstRealClinicApi:
    def test_a_real_seeded_package_speaks_its_real_price_and_tests(
        self, monkeypatch, real_clinic_api, tmp_path
    ):
        db = real_clinic_api.db.SessionLocal()
        try:
            pkg = db.query(real_clinic_api.models.HealthPackage).first()
            assert pkg is not None, "seed.py must still seed at least one health package -- fixture assumption"
        finally:
            db.close()

        tools = RealClinicToolsClient(real_clinic_api.client)
        spoken = _dispatch(
            monkeypatch, tools, "health_package", {"package_name": pkg.name}, tmp_path,
        )

        assert len(spoken) == 1
        assert tools.raw_responses[0]["found"] is True

    def test_no_package_named_lists_every_real_seeded_package(
        self, monkeypatch, real_clinic_api, tmp_path
    ):
        # FakeASRResult's utterance is Bengali (see its own docstring),
        # so the reply speaks each package's Bengali alias (first
        # "|"-separated entry of HealthPackage.aliases), not its English
        # catalogue name -- same _first_alias_bn() convention every other
        # catalogue reply in this codebase follows.
        db = real_clinic_api.db.SessionLocal()
        try:
            packages = db.query(real_clinic_api.models.HealthPackage).all()
            assert packages, "seed.py must still seed at least one health package -- fixture assumption"
            first_aliases = []
            for p in packages:
                alias = next((a.strip() for a in (p.aliases or "").split("|") if a.strip()), None)
                assert alias, f"{p.name!r} must be seeded with at least one alias -- fixture assumption"
                first_aliases.append(alias)
        finally:
            db.close()

        tools = RealClinicToolsClient(real_clinic_api.client)
        spoken = _dispatch(
            monkeypatch, tools, "health_package", {"package_name": None}, tmp_path,
        )

        assert len(spoken) == 1
        for alias in first_aliases:
            assert alias in spoken[0]

    def test_real_misspelled_or_unknown_package_gets_the_not_found_path(
        self, monkeypatch, real_clinic_api, tmp_path
    ):
        tools = RealClinicToolsClient(real_clinic_api.client)
        spoken = _dispatch(
            monkeypatch, tools, "health_package", {"package_name": "Zzznonexistent Package"}, tmp_path,
        )
        assert tools.raw_responses[0]["found"] is False
        assert "Zzznonexistent Package" in spoken[0]

    def test_real_clinic_info_speaks_the_real_seeded_address(
        self, monkeypatch, real_clinic_api, tmp_path
    ):
        db = real_clinic_api.db.SessionLocal()
        try:
            info = db.query(real_clinic_api.models.ClinicInfo).first()
            assert info is not None, "seed.py must still seed a ClinicInfo row -- fixture assumption"
            expected_address = info.address
        finally:
            db.close()

        tools = RealClinicToolsClient(real_clinic_api.client)
        spoken = _dispatch(
            monkeypatch, tools, "clinic_info", {"info_topic": "address"}, tmp_path,
        )

        assert tools.raw_responses[0]["found"] is True
        assert expected_address in spoken[0]

    def test_real_endpoint_response_shapes_match_what_the_reply_functions_expect(
        self, real_clinic_api,
    ):
        # Direct endpoint checks (no dispatch) -- a clinic-api response
        # shape drift would be caught here even before it ever reached a
        # live dispatch turn, same precedent as
        # test_doctor_schedule_dispatch.py's own equivalent test.
        r = real_clinic_api.client.get("/api/v1/clinic/info")
        body = r.json()
        assert body["found"] is True
        for key in ("clinic_name", "phone", "address", "directions", "hours"):
            assert key in body
        for day, entry in body["hours"].items():
            assert set(entry.keys()) == {"closed", "open", "close"}

        r2 = real_clinic_api.client.get("/api/v1/health-packages")
        body2 = r2.json()
        assert "packages" in body2
        for pkg in body2["packages"]:
            for key in ("found", "package_name", "package_name_bn", "description",
                        "price_inr", "tests", "tests_bn"):
                assert key in pkg


class TestTransportParity:
    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb

    def test_main_dot_py_has_both_new_branches(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "health_package"' in src
        assert "search_health_package" in src
        assert "get_health_packages" in src
        assert "health_package_reply" in src
        assert "health_packages_list_reply" in src
        assert 'intent == "clinic_info"' in src
        assert "get_clinic_info" in src
        assert "clinic_info_reply" in src

    def test_main_pcm_dot_py_has_both_new_branches(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "health_package"' in src
        assert 'intent == "clinic_info"' in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
