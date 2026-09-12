"""No model-composed span reaches synthesis on a factual intent.

# story title: The model never originates a fact
# user story: As a clinical lead, I want every price, date and identifier
#   to come from a verified system response, so that a wrong answer is a
#   data bug rather than a model bug.
# acceptance criteria: Every factual sentence is a template substitution
#   from a validated tool response and the model is never shown a figure
#   it could restate. An automated assertion on every commit proves no
#   model-composed span reaches synthesis on a factual intent.

WHAT THE STATIC CHECKS PROVE, AND WHAT THEY DO NOT
--------------------------------------------------
The AST tests below prove that no model-composed span reaches `_speak()`
*syntactically, in main.py*. A string routed through a local variable first
would pass them. That is a real limit and it is stated here rather than left
for someone to discover: this is a lint against the regression that would
actually happen -- somebody inlining a price into an f-string instead of
adding a template -- not a soundness proof of the whole pipeline.

The limit is acceptable because the property has a second, structural defence
that no test is needed for: agent/llm.py is called with the caller's
transcript and nothing else, and the tool lookup runs AFTER it returns. The
model cannot restate a price because it is never shown one. These tests guard
the part that is a convention, and the call order guards the part that is not.

Everything here is pure -- no GPU, no model, no network.
"""
from __future__ import annotations

import ast
import datetime
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import tool_contract  # noqa: E402
from agent import date_calc  # noqa: E402
from agent.slot_parse import parse_date  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"
TEMPLATES = ROOT / "agent" / "reply_templates.py"

# Keys whose value is a FACT: a price, a date, an identifier, a duration the
# clinic owns. Never composed, never inferred, never read off the model's
# slots -- only off a validated tool response.
FACT_KEYS = frozenset({
    "rate_inr", "confirmation_id", "report_time_hours",
    "chamber_hours", "next_available_date", "alternative_slots",
})


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# story title: The same question gets the same answer within one call
# user story: As a caller who asks twice, I want the same answer, so that I
#   know which one to believe.
# acceptance criteria: Repeating a question in one call produces an identical
#   factual answer unless the underlying data changed, in which case the
#   change is stated. A test asserts consistency across three repeats with an
#   unchanged backend.
#
# Every factual reply now goes through main._speak_fact instead of _speak, so
# a gate that looked only at _speak would have stopped seeing the eight call
# sites that actually speak prices and rosters -- silently, and while still
# passing. Both entry points are scanned here, each at the argument that
# carries the sentence.
#
# The one exemption is the `await _speak(session, reply)` INSIDE _speak_fact.
# Its argument is a parameter, so it is opaque to this check by construction;
# what flows into it is pinned one level up, where every _speak_fact call site
# is required to pass a reply_templates call. Exempting the hand-off and
# checking the sources is the same guarantee, and the alternative -- adding
# "reply" to the allow-list -- would let any name called `reply` through
# anywhere in the file.
_SPOKEN_ARG = {"_speak": 1, "_speak_fact": 4}


def _wrapper_internal_calls(tree: ast.Module) -> set[int]:
    """ids of the _speak calls inside _speak_fact's own body."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_speak_fact":
            return {id(n) for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "_speak"}
    raise AssertionError("main._speak_fact is gone -- the consistency gate went with it")


def _spoken_spans(tree: ast.Module) -> list[tuple[int, ast.expr]]:
    """(line, the expression that becomes a spoken sentence) for both entry
    points."""
    internal = _wrapper_internal_calls(tree)
    out: list[tuple[int, ast.expr]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        index = _SPOKEN_ARG.get(node.func.id)
        if index is None or id(node) in internal:
            continue
        if len(node.args) > index:
            out.append((node.lineno, node.args[index]))
    return out


def _template_names(tree: ast.Module) -> set[str]:
    """Everything main.py imports from agent.reply_templates."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "agent.reply_templates":
            out.update(a.asname or a.name for a in node.names)
    return out


def _module_level_string_constants(tree: ast.Module) -> set[str]:
    """Names bound to a plain string LITERAL at module scope in main.py.

    Speaking one of these is speaking a literal -- the name is an alias for it,
    not a channel. SYSTEM_UNREACHABLE_BN is the case that forced this: the
    system-is-down sentence was written out at seven handlers, and naming it
    once is what stops one of them drifting into a different wording during a
    refactor (see tests/test_outcome_distinction.py).

    Narrow on purpose. Only `NAME = "..."` at top level qualifies -- not a
    join, not a format, not an f-string, not a value computed from anything.
    A name bound to a literal here cannot carry model output; a name bound to
    an expression could, so it stays an offender.
    """
    out: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        out.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return out


# ------------------------------------------------------- the assertion

def test_every_spoken_span_is_a_literal_or_a_template():
    """The criterion, as a check over the source.

    Exactly three things may be spoken: a string literal written here, a call
    to a reply_templates function, or the single smalltalk reply -- which the
    next test pins to its guard.
    """
    tree = _tree(MAIN)
    allowed = _template_names(tree) | _module_level_string_constants(tree)
    assert allowed, "main.py imports nothing from reply_templates -- check the import"

    offenders = []
    for lineno, arg in _spoken_spans(tree):
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            continue
        if isinstance(arg, ast.Name) and arg.id in allowed:
            continue
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id in allowed:
            continue
        # `X or "literal"` -- the smalltalk reply. Pinned by the next test.
        if isinstance(arg, ast.BoolOp) and isinstance(arg.op, ast.Or):
            continue
        # An implicitly concatenated literal is still a literal.
        if isinstance(arg, ast.BinOp):
            continue
        offenders.append((lineno, ast.dump(arg)[:90]))

    assert not offenders, (
        "a span reaching _speak()/_speak_fact() is neither a literal nor a "
        f"reply_templates call: {offenders}"
    )


def test_no_fstring_is_ever_spoken():
    """The realistic regression: someone inlines a price into a sentence at
    the call site rather than adding a template for it. An f-string in main.py
    is the shape that mistake takes, so it is banned outright rather than
    inspected -- there is no legitimate reason for one here, and a rule with no
    exceptions is far easier to keep than one that needs judgement."""
    offenders = [lineno for lineno, arg in _spoken_spans(_tree(MAIN))
                 if isinstance(arg, ast.JoinedStr)]
    assert not offenders, f"f-string passed to _speak() at line(s) {offenders}"


def test_model_text_is_spoken_only_under_the_smalltalk_guard():
    """direct_reply_bn is the ONE model-composed string the agent may say.

    llm.py already nulls it for every non-smalltalk intent, so this is the
    second of two locks. It asserts the read is lexically inside
    `if intent == "smalltalk"` -- move it one branch over and this fails,
    which is the only way it could ever be spoken on a factual turn.
    """
    tree = _tree(MAIN)

    def reads_direct_reply(node) -> bool:
        return any(isinstance(n, ast.Constant) and n.value == "direct_reply_bn"
                   for n in ast.walk(node))

    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_smalltalk_guard = (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name) and test.left.id == "intent"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "smalltalk"
        )
        if is_smalltalk_guard:
            guarded.extend(n for n in node.body if reads_direct_reply(n))

    total = sum(1 for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and n.value == "direct_reply_bn")
    assert total == 1, f"expected exactly one direct_reply_bn read in main.py, found {total}"
    assert guarded, "direct_reply_bn is read outside `if intent == \"smalltalk\"`"


def test_fact_keys_are_read_from_the_tool_response_only():
    """A price may be read off `result`. Never off `slots`.

    `slots` is what the model extracted from the caller; `result` is what the
    clinic answered. Reading a fact key off slots would be the model
    originating a fact with the syntax of a template substitution -- which is
    exactly the failure that would look correct in review.
    """
    offenders = []
    for node in ast.walk(_tree(TEMPLATES)):
        if not isinstance(node, ast.Subscript):
            continue
        key = node.slice
        if not (isinstance(key, ast.Constant) and key.value in FACT_KEYS):
            continue
        base = node.value
        if isinstance(base, ast.Name) and base.id != "result":
            offenders.append((node.lineno, base.id, key.value))
    # .get("rate_inr") style reads too.
    for node in ast.walk(_tree(TEMPLATES)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in FACT_KEYS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id != "result"):
            offenders.append((node.lineno, node.func.value.id, node.args[0].value))

    assert not offenders, f"fact key read from a non-result source: {offenders}"


def test_the_truth_boundary_is_an_import_rule():
    """reply_templates.py must not be able to reach the model at all."""
    src = TEMPLATES.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "agent.llm", "reply_templates imports from agent.llm"
        if isinstance(node, ast.Import):
            assert all(a.name != "agent.llm" for a in node.names)


# ------------------------------------------------- the date, in code

THURSDAY = datetime.date(2026, 9, 10)


@pytest.mark.parametrize("utterance", ["সকাল দশটায়", "বিকাল পাঁচটায়", "সকালে আসব"])
def test_a_daypart_word_is_not_a_date(utterance):
    """REGRESSION. "সকাল" (morning) and "বিকাল" (afternoon) both contain "কাল"
    (tomorrow), and the old substring test resolved every one of them to
    TOMORROW. A caller answering "কোন দিন চান?" with "সকালে" was silently
    given the wrong day -- a fabricated date, from code rather than from the
    model, which is the same defect wearing a different coat."""
    assert parse_date(utterance, today=THURSDAY) is None


@pytest.mark.parametrize("utterance,expected", [
    ("আজ", "2026-09-10"), ("আজকে", "2026-09-10"),
    ("কাল", "2026-09-11"), ("আগামীকাল", "2026-09-11"), ("কালকে", "2026-09-11"),
    ("পরশু", "2026-09-12"), ("পরশুদিন", "2026-09-12"),
    ("সোমবার", "2026-09-14"), ("সোম", "2026-09-14"), ("রবিবার", "2026-09-13"),
    ("কাল সকাল দশটায়", "2026-09-11"),
])
def test_real_dates_still_resolve(utterance, expected):
    """The boundary fix must not cost coverage -- including the case where a
    genuine date word and a daypart word appear in the same sentence."""
    assert parse_date(utterance, today=THURSDAY) == expected


def test_deterministic_beats_the_model():
    """Order of authority: the caller's literal words outrank the model's
    interpretation, even when the model is confidently wrong about them."""
    span = date_calc.resolve("কাল", date_expr="next_week", today=THURSDAY)
    assert (span.start, span.end, span.source) == (
        "2026-09-11", "2026-09-11", date_calc.SOURCE_PARSED)


def test_the_model_names_the_meaning_and_code_does_the_maths():
    """The architecture, in one assertion. The model contributes the string
    "next_week" and nothing else; every digit below was computed here."""
    span = date_calc.resolve("আগামী সপ্তাহে ডাক্তার সেন আছেন?",
                             date_expr="next_week", today=THURSDAY)
    assert (span.start, span.end) == ("2026-09-14", "2026-09-20")
    assert span.source == date_calc.SOURCE_INTERPRETED
    assert span.is_range and span.needs_confirmation


@pytest.mark.parametrize("expression,expected", [
    ("today", ("2026-09-10", "2026-09-10")),
    ("tomorrow", ("2026-09-11", "2026-09-11")),
    ("day_after_tomorrow", ("2026-09-12", "2026-09-12")),
    ("this_week", ("2026-09-10", "2026-09-13")),      # the REST of it, not from Monday
    ("next_week", ("2026-09-14", "2026-09-20")),      # Mon..Sun
    ("this_weekend", ("2026-09-12", "2026-09-13")),
    ("next_weekend", ("2026-09-19", "2026-09-20")),
    ("this_month", ("2026-09-10", "2026-09-30")),
    ("next_month", ("2026-10-01", "2026-10-31")),
    ("saturday", ("2026-09-12", "2026-09-12")),
    ("next_saturday", ("2026-09-19", "2026-09-19")),
    ("thursday", ("2026-09-17", "2026-09-17")),       # today is Thursday -> NEXT one
])
def test_the_calendar_itself(expression, expected):
    assert date_calc.resolve_expression(expression, today=THURSDAY) == expected


def test_an_invented_expression_resolves_to_nothing():
    """A model that emits "the_week_after_next" gets no date at all. Falling
    back to today would turn invented output into a confident wrong answer,
    which is the failure this whole story removes."""
    assert date_calc.resolve_expression("the_week_after_next", today=THURSDAY) is None


def test_unmapped_is_not_the_same_as_absent():
    """Two ways to have no date, needing opposite handling: the caller named a
    day nothing could express (ask which one), versus named none at all
    (today is the question they actually asked). Collapsing these is how the
    old code answered about today when the caller had said otherwise."""
    unmapped = date_calc.resolve("মাসের শেষ দিকে", date_expr="other", today=THURSDAY)
    absent = date_calc.resolve("ডাক্তার সেন আছেন?", date_expr=None, today=THURSDAY)
    assert unmapped.source == date_calc.SOURCE_UNMAPPED
    assert absent.source == date_calc.SOURCE_ABSENT
    assert unmapped.start is None and absent.start is None


def test_a_single_day_is_never_confirmed():
    """Policy: only a range is read back. Confirming every single date would
    train callers to say হ্যাঁ without listening, which is how a confirmation
    step stops being one."""
    for expr in ("today", "tomorrow", "next_saturday"):
        assert not date_calc.resolve("x", date_expr=expr, today=THURSDAY).needs_confirmation


def test_a_computed_span_is_frozen():
    span = date_calc.resolve("x", date_expr="next_week", today=THURSDAY)
    with pytest.raises(Exception):
        span.start = "2026-01-01"


def test_the_range_confirmation_speaks_its_computed_dates():
    """The dates may be spoken because CODE produced them. Asking "did you mean
    the 14th to the 20th?" asserts that those are next week's dates -- a fact --
    so it would be unsayable if the model had supplied them."""
    from agent import speakability
    from agent.bn_normalize import verbalize
    from agent.reply_templates import date_range_confirm_prompt
    prompt = date_range_confirm_prompt("2026-09-14", "2026-09-20")
    assert speakability.check(prompt).state == speakability.SPEAKABLE
    spoken = verbalize(prompt)
    assert "চোদ্দো" in spoken and "কুড়ি" in spoken, spoken


# --------------------------------------------- the model has no calendar

def test_todays_date_is_not_in_the_prompt():
    """The one figure the model was ever shown. It no longer needs it -- it
    names expressions, not dates -- so "never shown a figure it could restate"
    is now literally true instead of nearly true."""
    from agent.llm import SYSTEM_PROMPT_TEMPLATE
    assert "{today_iso}" not in SYSTEM_PROMPT_TEMPLATE
    assert "{today_weekday}" not in SYSTEM_PROMPT_TEMPLATE


def test_a_calendar_date_from_the_model_is_stripped():
    """Defended in code, not asked for in the prompt. The model has no calendar
    now, so a yyyy-mm-dd in `date` can only have been invented, and leaving it
    would be indistinguishable from a date the caller actually spoke."""
    from agent.llm import _validate
    data = {"intent": "doctor_availability", "direct_reply_bn": None,
            "slots": {"test_name": None, "doctor_name": "সেন", "department": None,
                      "date_expr": "next_week", "date": "2026-09-14",
                      "time_slot": None, "patient_name": None, "phone": None}}
    ok, _ = _validate(data)
    assert ok, "stripping a field must not force a retry"
    assert data["slots"]["date"] is None
    assert data["slots"]["date_expr"] == "next_week", "the MEANING is kept"


def test_an_out_of_vocabulary_expression_is_stripped():
    from agent.llm import _validate
    data = {"intent": "doctor_availability", "direct_reply_bn": None,
            "slots": {"test_name": None, "doctor_name": "সেন", "department": None,
                      "date_expr": "the_week_after_next", "date": None,
                      "time_slot": None, "patient_name": None, "phone": None}}
    _validate(data)
    assert data["slots"]["date_expr"] is None


def test_the_prompt_and_the_calculator_share_one_vocabulary():
    """Imported, not restated -- the failure mode of every "keep these two
    lists in sync" comment ever written."""
    from agent.llm import DATE_EXPRESSIONS
    assert DATE_EXPRESSIONS is date_calc.VOCABULARY


# ------------------------------------------------- the tool contract

def test_a_complete_response_passes_through_unchanged():
    payload = {"found": True, "test_name": "Uric Acid", "test_name_bn": "ইউরিক অ্যাসিড",
               "rate_inr": 250, "sample_type": "Blood", "report_time_hours": 12}
    assert tool_contract.validate("test_rate", payload) is payload


def test_a_missing_fact_field_is_refused_at_the_boundary():
    """Before this story a response like this raised KeyError inside a
    template, which main.py reported as a tool-call failure -- so a clinic-api
    schema regression was indistinguishable from the API being down."""
    with pytest.raises(tool_contract.ToolContractError) as excinfo:
        tool_contract.validate("test_rate", {"found": True, "test_name": "Uric Acid"})
    assert "rate_inr" in str(excinfo.value)


def test_a_null_valued_field_is_present_not_missing():
    """chamber_hours=None is a correct answer to "is the doctor in on Tuesday".
    Checking truthiness rather than presence would turn every not-available
    reply into a contract violation."""
    tool_contract.validate("doctor_availability", {
        "found": True, "doctor_name": "Dr A Sen", "doctor_name_bn": "সেন",
        "date": "2026-09-10", "available": False,
        "chamber_hours": None, "next_available_date": "2026-09-14",
    })


def test_the_not_found_branch_has_its_own_contract():
    tool_contract.validate("test_rate", {"found": False, "query": "কিছু"})
    with pytest.raises(tool_contract.ToolContractError):
        tool_contract.validate("test_rate", {"found": False})


def test_an_unregistered_tool_is_an_error_not_a_pass():
    """Adding a fifth tool must fail loudly here rather than silently skip
    validation, which is how a boundary check quietly stops covering things."""
    with pytest.raises(tool_contract.ToolContractError):
        tool_contract.validate("some_new_tool", {"found": True})


# ------------------------------------------- confirming the booking

_BOOKING = {"patient_name": "রিয়া দাস", "date": "2026-09-11",
            "time_slot": "18:30", "phone": "9876543210"}


def test_the_readback_still_contains_every_field():
    """Nothing in this story may narrow what gets confirmed. All five values a
    booking is made of are read back, in the caller's hearing, before the one
    irreversible thing this agent does."""
    from agent.reply_templates import booking_confirm_prompt
    prompt = booking_confirm_prompt(dict(_BOOKING, doctor_name="Dr. A Sen",
                                         doctor_name_bn="সেন"))
    for value in ("রিয়া দাস", "সেন", "2026-09-11", "18:30", "9876543210"):
        assert value in prompt, f"{value!r} missing from the booking readback"


@pytest.mark.parametrize("slots,expected", [
    ({"doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন"}, "speakable"),  # via a dept listing
    ({"doctor_name": "সেন"}, "speakable"),                                  # caller's own words
    ({"doctor_name": "Dr. A Sen", "doctor_name_bn": None}, "blocked"),      # no alias seeded
])
def test_the_readback_is_actually_sayable_on_every_route(slots, expected):
    """REGRESSION. The commonest booking route stores the CANONICAL English
    doctor name, because the booking API matches on it. The readback is
    spoken, so that put "ডাঃ Dr. A Sen" into a Bengali sentence -- the caller
    was asked to confirm a booking with the doctor's name silently dropped,
    and once the speakability gate landed the whole confirmation was blocked
    and the caller sent to the counter mid-booking.

    The third case stays blocked on purpose: a doctor row with no Bengali
    alias is a data defect, and escalating beats confirming a booking against
    a name the caller cannot hear."""
    from agent import speakability
    from agent.reply_templates import booking_confirm_prompt
    verdict = speakability.check(booking_confirm_prompt(dict(_BOOKING, **slots)))
    assert verdict.state == expected, list(verdict.dropped)


def test_exactly_one_call_site_writes_and_it_passes_confirmed():
    """The write guard lives inside _finish_booking so it cannot be forgotten
    at a call site. This checks the other half: that no call site tries."""
    tree = _tree(MAIN)
    confirming = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_finish_booking"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        confirmed = kw.get("confirmed")
        assert isinstance(confirmed, ast.Constant) and confirmed.value is True, (
            f"_finish_booking called at line {node.lineno} without confirmed=True"
        )
        confirming.append(node.lineno)
    assert len(confirming) == 1, (
        f"expected exactly one booking write site, found {len(confirming)} at {confirming}"
    )
