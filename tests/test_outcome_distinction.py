"""A thing not existing is never confused with a system being down.

story title: A thing not existing is never confused with a system being down
user story: As a caller, I want to know whether my test does not exist or the
    system cannot be reached, so that I know whether to call back.
acceptance criteria: The two produce different spoken sentences and different
    metrics, and the distinction survives every refactor. This behaviour exists
    today and gains a permanent regression case.

WHY HALF OF THIS IS A STATIC CHECK
----------------------------------
"Survives every refactor" is the phrase that decides the shape. Behavioural
tests prove the two paths differ TODAY; they say nothing about a refactor that
adds an eighth failure handler and reaches for a template in it, because the
new handler has no test. The AST checks below hold over whatever main.py
becomes: every ToolCallError handler speaks the one system-down constant, and
none of them may call a reply_templates function at all.

The distinction carries opposite instructions to the caller -- "stop asking"
versus "call back" -- which is why it is worth defending at this cost.
"""
from __future__ import annotations

import ast
import asyncio
import json
import pathlib
import sys

import httpx
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from agent import tool_outcome  # noqa: E402
from agent.reply_templates import (  # noqa: E402
    booking_reply, doctor_availability_reply, doctors_by_department_reply,
    test_rate_reply as rate_reply,
)
from agent.tools_client import ClinicToolsClient, ToolCallError  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"


def _tree() -> ast.Module:
    return ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))


def _speak_calls_in(node) -> list[ast.Call]:
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "_speak"]


def _tool_error_handlers(tree: ast.Module) -> list[ast.ExceptHandler]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        names = ([node.type] if isinstance(node.type, ast.Name)
                 else list(getattr(node.type, "elts", [])))
        if any(isinstance(n, ast.Name) and n.id == "ToolCallError" for n in names):
            out.append(node)
    return out


# ------------------------------------------------- survives every refactor

def test_every_failure_handler_speaks_the_one_unreachable_sentence():
    handlers = _tool_error_handlers(_tree())
    assert len(handlers) >= 7, f"expected the known failure handlers, found {len(handlers)}"

    for handler in handlers:
        calls = _speak_calls_in(handler)
        assert calls, f"ToolCallError handler at line {handler.lineno} says nothing"
        for call in calls:
            arg = call.args[1] if len(call.args) > 1 else None
            assert isinstance(arg, ast.Name) and arg.id == "SYSTEM_UNREACHABLE_BN", (
                f"handler at line {handler.lineno} does not speak the shared "
                f"system-unreachable constant"
            )
            reasons = {k.arg: k.value for k in call.keywords}
            fallback = reasons.get("fallback_reason")
            assert isinstance(fallback, ast.Constant) and fallback.value == "tool_failure", (
                f"handler at line {handler.lineno} has the wrong fallback clip -- "
                f"the caller would hear the generic busy line, not 'cannot check'"
            )


def test_no_failure_handler_may_speak_a_template():
    """The strong form of the rule. A not-found sentence NAMES a thing that
    does not exist; a failure handler does not know that and must never claim
    it. Banning the whole template layer from these blocks is easier to keep
    than a list of which sentences are safe."""
    tree = _tree()
    template_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "agent.reply_templates":
            template_names.update(a.asname or a.name for a in node.names)

    for handler in _tool_error_handlers(tree):
        for call in _speak_calls_in(handler):
            for arg in call.args[1:2]:
                offending = {n.func.id for n in ast.walk(arg)
                             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id in template_names}
                assert not offending, (
                    f"handler at line {handler.lineno} speaks template(s) {offending}"
                )


def test_a_turn_can_never_die_silently():
    """_dispatch_turn is fired by create_task with nothing attached. Without an
    outer handler an uncaught exception is an unretrieved task exception and
    the caller hears NOTHING -- a third outcome, and the one this module's own
    docstring promises never happens. Delete the wrapper and this fails."""
    tree = _tree()
    wrapper = next((n for n in ast.walk(tree)
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == "_dispatch_turn"), None)
    assert wrapper is not None
    handlers = [n for n in ast.walk(wrapper) if isinstance(n, ast.ExceptHandler)]
    assert any(isinstance(h.type, ast.Name) and h.type.id == "Exception" for h in handlers), (
        "_dispatch_turn has no catch-all -- a crashed turn would be silence"
    )
    assert any(n.id == "SYSTEM_UNREACHABLE_BN"
               for h in handlers for n in ast.walk(h) if isinstance(n, ast.Name)), (
        "a crashed turn must answer as unreachable, never as not-found"
    )


def test_the_two_sentence_families_are_disjoint():
    """The property a caller actually depends on: no wording appears on both
    sides. One means stop asking, the other means call back."""
    not_found = {
        rate_reply({"test_name": "x"}, {"found": False, "query": "x"}),
        doctor_availability_reply({"doctor_name": "x"}, {"found": False, "query": "x"}),
        doctors_by_department_reply({"department": "x"}, {"found": False, "query": "x"}),
        booking_reply({"doctor_name": "x"}, {"success": False, "reason": "doctor_not_found"}),
    }
    assert main.SYSTEM_UNREACHABLE_BN not in not_found
    for sentence in not_found:
        assert sentence != main.SYSTEM_UNREACHABLE_BN
        # And no not-found sentence may borrow the "call back later" framing,
        # which is the instruction that belongs only to the other side.
        assert "কাউন্টারে যোগাযোগ করুন" not in sentence


# --------------------------------------------------- different metrics

class _Response:
    """A stand-in for httpx.Response that exposes `.text`, not `.json()`.

    It used to implement .json(), and updating it was the right response to
    tools_client switching to _parse_exact(): the double had stopped matching
    the interface production actually calls, so five tests failed for exactly
    the reason a test double should fail. Carrying `.text` means the real
    json.loads(..., parse_float=str) path now runs inside these tests rather
    than being stubbed past -- including the JSONDecodeError that the
    malformed-body case depends on.
    """

    def __init__(self, payload=None, *, body=None, status=200):
        self.text = body if body is not None else json.dumps(payload)
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


class _Transport:
    def __init__(self, outcome):
        self._outcome = outcome

    async def get(self, url, params=None):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    post = get


def _client(outcome):
    c = ClinicToolsClient("http://x")
    c._client = _Transport(outcome)
    return c


TEST_FOUND = {"found": True, "test_name": "Uric Acid", "test_name_bn": "ইউরিক অ্যাসিড",
              "rate_inr": 250, "sample_type": "Blood", "report_time_hours": 12}


def test_a_missing_thing_counts_as_not_found_and_returns_normally():
    client = _client(_Response({"found": False, "query": "কিছু"}))
    payload = asyncio.run(client.get_test_rate("কিছু"))
    assert payload["found"] is False, "not-found is an ANSWER, never an exception"
    counts = client.snapshot()["test_rate"]
    assert (counts["not_found"], counts["unreachable"]) == (1, 0)


def test_an_unreachable_clinic_counts_separately_and_raises():
    client = _client(httpx.ConnectError("refused"))
    with pytest.raises(ToolCallError):
        asyncio.run(client.get_test_rate("ইউরিক অ্যাসিড"))
    counts = client.snapshot()["test_rate"]
    assert (counts["not_found"], counts["unreachable"]) == (0, 1)


def test_a_200_with_an_html_body_is_unreachable_not_a_crash():
    """REGRESSION. json.JSONDecodeError subclasses ValueError, NOT
    httpx.HTTPError, so a proxy answering 200 with an error page escaped every
    handler and killed the turn -- on this deployment the most likely real
    shape of "the system is down", and it produced neither sentence."""
    client = _client(_Response(body="<html>502 Bad Gateway</html>"))
    with pytest.raises(ToolCallError):
        asyncio.run(client.get_test_rate("ইউরিক অ্যাসিড"))
    assert client.snapshot()["test_rate"]["unreachable"] == 1


def test_a_found_answer_counts_as_answered():
    client = _client(_Response(TEST_FOUND))
    asyncio.run(client.get_test_rate("ইউরিক অ্যাসিড"))
    counts = client.snapshot()["test_rate"]
    assert (counts["answered"], counts["not_found"], counts["unreachable"]) == (1, 0, 0)


@pytest.mark.parametrize("reason,expected", [
    ("doctor_not_found", tool_outcome.NOT_FOUND),
    ("slot_taken", tool_outcome.ANSWERED),
    ("missing_field", tool_outcome.ANSWERED),
    ("doctor_not_available_that_day", tool_outcome.ANSWERED),
])
def test_only_one_booking_failure_means_a_thing_does_not_exist(reason, expected):
    """The rule that would rot first if it were not pinned. A taken slot is the
    clinic ANSWERING -- the doctor exists, the appointment just cannot be made
    as asked. Counting those as not-found would leave the metric still moving
    up and down while no longer meaning what its name says."""
    assert tool_outcome.classify("book_appointment", {"success": False, "reason": reason}) == expected


def test_an_empty_doctor_list_is_an_answer_not_an_absence():
    """found=true with no doctors means the department exists and nobody sits
    that day. That is a fact about the schedule, not a missing department."""
    assert tool_outcome.classify(
        "doctors_by_department",
        {"found": True, "department": "Cardiology", "doctors": []}) == tool_outcome.ANSWERED


def test_the_rate_is_unknown_before_anything_is_asked():
    """None, not 0.0. A rate over zero calls is not zero, and an alert on
    "rate == 0" firing at every process start is how a metric earns itself a
    permanent exclusion rule."""
    assert ClinicToolsClient("http://x").snapshot()["test_rate"]["not_found_rate"] is None


def test_the_rate_is_derived_once_and_not_by_the_reader():
    client = _client(_Response({"found": False, "query": "x"}))
    for _ in range(3):
        asyncio.run(client.get_test_rate("x"))
    client._client = _Transport(_Response(TEST_FOUND))
    asyncio.run(client.get_test_rate("y"))
    assert client.snapshot()["test_rate"]["not_found_rate"] == 0.75


def test_unreachable_is_left_out_of_the_rate():
    """The rate is a property of ANSWERS. Folding outages into the denominator
    would make an outage look like the catalogue improving."""
    client = _client(_Response({"found": False, "query": "x"}))
    asyncio.run(client.get_test_rate("x"))
    client._client = _Transport(httpx.ConnectError("refused"))
    for _ in range(9):
        with pytest.raises(ToolCallError):
            asyncio.run(client.get_test_rate("x"))
    assert client.snapshot()["test_rate"]["not_found_rate"] == 1.0
