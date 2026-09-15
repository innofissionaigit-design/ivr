"""Pure comparison-computation logic for "Caller asks the agent to compare
two options" story.

ADDED BY SOURAV -- new file for this story, mirroring agent/report_flow.py's
own established convention (see that module's docstring for the full
reasoning): the actual DECISIONS -- here, the actual ARITHMETIC -- live in
one plain-function module with no dependency on CallSession, WebSocket,
tools_client, or asyncio, so main.py and main_pcm.py both call the exact
same code and can never drift the way main_pcm.py's own module docstring
says "test_sample" once did by being hand-duplicated in two places.

WHY DECIMAL, NEVER FLOAT (Acceptance Criterion 1: "computed strictly in
Python code from live database values ... stated clearly"):
This codebase has an existing, hard-won rule that a money figure is never
mutated by so much as one digit end to end -- see NUMBER_FIDELITY_
IMPLEMENTATION.md and agent/tools_client.py's `_parse_exact()` docstring,
which exists specifically because Python's `float(100.00) == float(100.0)`
silently eats a trailing zero that matters for money. Until this story,
nothing in the pipeline ever did ARITHMETIC on a price at all -- every
number just passed through a template unchanged. Computing a PRICE
DIFFERENCE is the first place this codebase does real arithmetic on a
live money value, so it is done with `decimal.Decimal` against the exact
source string/int clinic-api returned (never `float`), for the same
reason `_parse_exact()` avoids float: base-10 money arithmetic in binary
floating point can produce an inexact result (the classic 0.1 + 0.2
problem) that would then get spoken as a wrong number with total
confidence -- exactly the class of bug this whole codebase's docstring
history has been careful never to reintroduce.

WHAT THIS MODULE DELIBERATELY DOES NOT DO (Acceptance Criteria 2 & 3):
This module's output (`build_comparison()`'s return dict) contains ONLY
factual fields: found-ness, price delta, which side is cheaper, test-count
delta, and which specific tests differ. There is no field for "which one
is better", no adjective, no ranking beyond the two purely arithmetic
"cheaper"/"more_tests_side" facts (a price is either lower or it isn't --
that is arithmetic, not a value judgement). Nothing here ever weighs a
CLINICAL or suitability axis, because neither the LabTest nor the
HealthPackage catalogue exposes one -- there is structurally nothing for
this function, or agent/llm.py's classifier upstream, to base a medical
recommendation on even if asked to. agent/reply_templates.py's
compare_options_reply() renders this dict into a sentence and adds no
new facts of its own.
"""
from __future__ import annotations

from decimal import Decimal


def _to_decimal(raw) -> Decimal | None:
    """Exact base-10 parse of a clinic-api money value. `raw` is either a
    plain `int` (whole-rupee amounts, e.g. `650`) or a `str` preserving
    the exact source digits (e.g. `"199.55"`, `"250.0"`) -- see
    agent/tools_client.py's `_parse_exact()` docstring for why the shape
    varies. `Decimal(str(raw))` is exact for both: unlike `float()`, it
    never introduces a binary-fraction rounding error, because it parses
    the base-10 text directly rather than approximating it in base 2.
    Returns None for a missing/unparseable value rather than raising --
    an entity that was found but happens to carry no price is a fact
    build_comparison() below handles honestly, not a crash.
    """
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except Exception:  # noqa: BLE001 - defensive: never let a bad DB value 500
        return None


def _price_of(entity: dict) -> Decimal | None:
    """entity is one side's ALREADY-RESOLVED lookup (see main.py's
    `_resolve_comparable_entity()`), tagged with `"kind"`: "test" reads
    the same `rate_inr` field test_rate_reply() already speaks; "package"
    reads the same `price_inr` field health_package_reply() already
    speaks. Neither field is renamed or reshaped here -- this only picks
    which one to read for whichever kind this side turned out to be."""
    kind = entity.get("kind")
    if kind == "test":
        return _to_decimal(entity.get("rate_inr"))
    if kind == "package":
        return _to_decimal(entity.get("price_inr"))
    return None


def build_comparison(entity_a: dict, entity_b: dict) -> dict:
    """entity_a / entity_b: each a dict tagged `"kind"` ("test", "package",
    or "not_found") plus whichever fields that lookup returned -- this
    function performs NO tool calls or I/O of its own (main.py owns both
    lookups, mirroring agent/report_flow.py's interpret_*_result()
    functions, which likewise only ever receive an already-fetched
    result).

    Returns a plain dict of FACTS ONLY (see module docstring for why),
    for agent/reply_templates.py's compare_options_reply() to render into
    one sentence:
      {
        "a_found": bool, "b_found": bool,
        "a_kind": "test" | "package" | "not_found", "b_kind": same,
        "price_delta": str | None,       # digit-exact absolute difference
                                          # (Decimal, stringified) -- only
                                          # when BOTH sides have a price.
        "cheaper": "a" | "b" | None,     # None when EQUAL, or when either
                                          # side has no price to compare.
        "both_packages": bool,
        "identical_tests": bool | None,  # only meaningful when both_packages
        "test_count_delta": int | None,  # only when both_packages
        "more_tests_side": "a" | "b" | None,  # None when equal COUNT, even
                                               # if the actual tests differ
                                               # (see "identical_tests").
        "extra_tests_a": list[dict],     # tests package A has that B does
                                          # not: [{"name": en, "name_bn": bn
                                          # or None}, ...] -- only when
                                          # both_packages.
        "extra_tests_b": list[dict],     # symmetric, B's-not-A's.
      }
    """
    result = {
        "a_found": entity_a.get("kind") != "not_found",
        "b_found": entity_b.get("kind") != "not_found",
        "a_kind": entity_a.get("kind"),
        "b_kind": entity_b.get("kind"),
        "price_delta": None, "cheaper": None,
        "both_packages": False, "identical_tests": None,
        "test_count_delta": None, "more_tests_side": None,
        "extra_tests_a": [], "extra_tests_b": [],
    }
    if not result["a_found"] or not result["b_found"]:
        return result

    price_a = _price_of(entity_a)
    price_b = _price_of(entity_b)
    if price_a is not None and price_b is not None:
        delta = price_a - price_b
        if delta > 0:
            result["cheaper"] = "b"
        elif delta < 0:
            result["cheaper"] = "a"
        # delta == 0 -> "cheaper" stays None: a real, worth-stating fact
        # (they cost the same), not a missing comparison.
        result["price_delta"] = str(abs(delta))

    result["both_packages"] = entity_a.get("kind") == "package" and entity_b.get("kind") == "package"
    if result["both_packages"]:
        names_a = entity_a.get("tests") or []
        names_bn_a = entity_a.get("tests_bn") or []
        names_b = entity_b.get("tests") or []
        names_bn_b = entity_b.get("tests_bn") or []
        # Positional pairing (English name at index i <-> its own Bengali
        # alias at the same index i), same convention
        # reply_templates._spoken_package_tests() already relies on --
        # kept as (name, name_bn) pairs here rather than two parallel
        # lists so a set-difference on the English name (the identity)
        # carries its own alias along, with no re-lookup needed downstream.
        pairs_a = {en: bn for en, bn in zip(names_a, names_bn_a)} if names_bn_a else {en: None for en in names_a}
        pairs_b = {en: bn for en, bn in zip(names_b, names_bn_b)} if names_bn_b else {en: None for en in names_b}
        set_a, set_b = set(pairs_a), set(pairs_b)

        extra_a = sorted(set_a - set_b)
        extra_b = sorted(set_b - set_a)
        result["extra_tests_a"] = [{"name": n, "name_bn": pairs_a.get(n)} for n in extra_a]
        result["extra_tests_b"] = [{"name": n, "name_bn": pairs_b.get(n)} for n in extra_b]
        result["identical_tests"] = not extra_a and not extra_b

        count_delta = len(set_a) - len(set_b)
        result["test_count_delta"] = abs(count_delta)
        if count_delta > 0:
            result["more_tests_side"] = "a"
        elif count_delta < 0:
            result["more_tests_side"] = "b"
        # count_delta == 0 -> "more_tests_side" stays None, even if
        # identical_tests is False (same COUNT, different actual tests --
        # see compare_options_reply()'s own handling of that case).

    return result
