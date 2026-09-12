"""A figure is grouped so a caller can write it down.

story title: Figures are spoken at a pace a caller can write down
user story: As a patient noting a price or a reference, I want it grouped and
    slower, so that I do not have to ask twice.
acceptance criteria: Prices, phone numbers and reference identifiers are
    spoken with grouping and a reduced rate through the per-request speed
    parameter. A listening test confirms callers transcribe correctly on
    first hearing.

WHAT IS IMPLEMENTED AND WHAT IS NOT
------------------------------------
GROUPING is implemented and is what this file tests.

THE REDUCED RATE IS NOT. Slowing only the figure needs the synthesiser to
render a sentence in pieces at different rates, and the existing per-chunk
split is on PUNCTUATION, so a figure sits inside a chunk and setting a rate
per chunk would slow the whole carrier sentence with it. That is an
architectural change to tts_server plus a new request contract, and it was
deliberately not started -- see the workbook row and agent/bn_normalize.py's
policy comment.

So this file tests a real half of the story and does not pretend to test the
other. There is no test here asserting the rate works, because it does not.

THE GROUPING NUMBERS ARE POLICY, NOT REQUIREMENTS. The story says "grouping".
It does not say 5+5, or fours, or that identifiers keep their own hyphens.
The tests below pin the policy as it stands so a change to it is deliberate
and visible -- NOT because the policy is correct. Only the listening study
can say that.

Everything here is pure -- no GPU, no model, no network.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import speakability  # noqa: E402
from agent.bn_normalize import (  # noqa: E402
    GROUP_SEPARATOR, GROUPING_POLICY_VERSION, MAX_GROUP_DIGITS, _group_sizes,
    digits_one_by_one, grouped_digits, number_to_bn_words, spell_out, verbalize,
)

EVALUATIONS = ROOT / "docs" / "evaluations"

# Required keys in a listening-study result. A study that does not record the
# policy and rate it heard cannot be acted on afterwards, which is the whole
# reason this is validated rather than trusted.
EVALUATION_SCHEMA = {
    "date": str,
    "participants": int,
    "device": str,
    "grouping_policy_version": str,
    "figure_length_scale": (float, int, type(None)),
    "scoring": dict,          # per-type rule, as in docs/listening-test.md
    "results": dict,          # per-type accuracy
    "stimulus_manifest": str,
}


def _digits(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


# ------------------------------------------------------- the group sizes

@pytest.mark.parametrize("n,expected", [
    (1, (1,)), (3, (3,)), (4, (4,)),
    (5, (3, 2)), (6, (3, 3)), (8, (4, 4)), (9, (3, 3, 3)),
    (10, (5, 5)),            # the Indian mobile convention, explicitly
    (11, (4, 4, 3)), (12, (4, 4, 4)),
])
def test_the_grouping_policy_is_what_it_says_it_is(n, expected):
    """Pins the POLICY, not its correctness. If the listening study moves
    these, this test is the thing that has to be edited alongside -- which is
    the point: the numbers cannot drift without somebody noticing."""
    assert _group_sizes(n) == expected


def test_no_group_is_longer_than_the_maximum_except_by_convention():
    for n in range(1, 40):
        sizes = _group_sizes(n)
        assert sum(sizes) == n, f"{n}: groups do not reconstruct the number"
        if n not in (10,):
            assert max(sizes) <= MAX_GROUP_DIGITS, n


def test_no_group_is_a_single_digit_unless_the_number_is():
    """A group of one is not a group -- it reads as a stumble, not a break."""
    for n in range(2, 40):
        if n in (10,):
            continue
        assert min(_group_sizes(n)) > 1, n


def test_the_policy_carries_a_version():
    """A result that cannot say which policy it heard is a result nobody can
    act on six months later."""
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}\.[a-z]+", GROUPING_POLICY_VERSION)


# ----------------------------------------------------------- phone numbers

@pytest.mark.parametrize("phone", [
    "9876543210", "9331045678", "8420076543", "6291100045",
])
def test_a_phone_number_is_grouped_five_and_five(phone):
    spoken = grouped_digits(phone)
    groups = [g.strip() for g in spoken.split(GROUP_SEPARATOR)]
    assert len(groups) == 2
    assert [len(g.split()) for g in groups] == [5, 5]


@pytest.mark.parametrize("phone", ["9876543210", "9000000009", "1234509876"])
def test_grouping_never_loses_or_reorders_a_digit(phone):
    """The number-fidelity guarantee, restated against the grouped form. A
    group boundary is a pause, never a licence to drop or move anything."""
    spoken = verbalize(f"ফোন {phone}")
    from agent.bn_normalize import _ONES_TO_99
    cursor = 0
    for digit in phone:
        word = _ONES_TO_99[int(digit)]
        found = spoken.find(word, cursor)
        assert found != -1, f"{phone}: {word} missing or out of order in {spoken!r}"
        cursor = found + len(word)


def test_a_grouped_number_is_still_sayable():
    for phone in ("9876543210", "6291100045"):
        reply = f"আপনার দেওয়া ফোন নম্বর {phone}।"
        verdict = speakability.check(reply)
        assert verdict.state == speakability.SPEAKABLE
        assert ":" not in verdict.spoken


# ------------------------------------------------------------ identifiers

REAL_IDS = [
    "KCD-20260914-4A2F",
    "KCD-20260914-0031",
    "KCD-20261231-FFFF",
    "KCD-20260101-B7C0",
]


@pytest.mark.parametrize("conf_id", REAL_IDS)
def test_an_identifier_keeps_the_structure_it_already_had(conf_id):
    """The whole win on identifiers. The hyphens in KCD-20260914-4A2F were
    thrown away before speech, so fifteen tokens arrived in one breath with
    no boundary where a pen would pause. The structure was in the string and
    was discarded on the way to the synthesiser."""
    groups = [g.strip() for g in spell_out(conf_id).split(GROUP_SEPARATOR)]
    # KCD | 2026 | 0914 | 4A2F -- the middle run is over the maximum and is
    # cut again, which is why this is four and not three.
    assert len(groups) == 4
    assert groups[0] == "কে সি ডি"


@pytest.mark.parametrize("conf_id", REAL_IDS)
def test_an_identifier_loses_no_character(conf_id):
    spoken_words = spell_out(conf_id).replace(GROUP_SEPARATOR, " ").split()
    alnum = [c for c in conf_id if c.isalnum()]
    assert len(spoken_words) == len(alnum), (
        f"{conf_id}: {len(alnum)} characters became {len(spoken_words)} words")


def test_an_identifier_with_no_separators_is_still_grouped():
    """Nothing in the format is guaranteed. A flat identifier must not fall
    back to one unbroken run."""
    groups = [g for g in spell_out("AB1234567890").split(GROUP_SEPARATOR) if g.strip()]
    assert len(groups) > 1


# ------------------------------------------------------------- what is NOT

@pytest.mark.parametrize("rate", [250, 650, 1250, 12500, 7])
def test_a_price_is_not_digit_grouped(rate):
    """A price is spoken as a QUANTITY -- "এক হাজার দুইশো পঞ্চাশ" -- and
    chopping it into digit groups would make it worse, not better. This is a
    scoping decision, recorded so it is visible: the story says "grouping"
    and this is a reading of it, not a quotation from it."""
    spoken = verbalize(f"রেট {rate} টাকা।")
    assert GROUP_SEPARATOR not in spoken
    assert number_to_bn_words(rate) in spoken


def test_the_decimal_path_is_byte_identical_to_before():
    """digits_one_by_one is the ungrouped reader and is shared with the
    fractional half of a decimal, where a group separator would be wrong --
    "দশমিক পাঁচ, শূন্য" is not a thing anyone says. Grouping is opt-in
    through grouped_digits precisely so this path did not change."""
    assert GROUP_SEPARATOR not in digits_one_by_one("50")
    assert GROUP_SEPARATOR not in verbalize("রেট 100.50 টাকা।")


def test_the_carrier_sentence_is_unchanged_apart_from_the_figure():
    """Grouping must not rewrite the words around the number."""
    spoken = verbalize("আপনার দেওয়া ফোন নম্বর 9876543210।")
    assert spoken.startswith("আপনার দেওয়া ফোন নম্বর ")


# ------------------------------------------- the listening study, as data

def test_the_evaluations_directory_exists():
    """The study has somewhere to land. Its absence is not a failure --
    absence of a result is absence of a result -- but a missing directory
    means somebody would have to invent a location, and then the schema check
    below would never run."""
    assert EVALUATIONS.is_dir()


def _evaluation_files():
    return sorted(EVALUATIONS.glob("figure-listening-*.json"))


@pytest.mark.parametrize("path", _evaluation_files() or [None])
def test_any_listening_result_matches_the_schema(path):
    """Validates the SHAPE of a result, not that one exists.

    Deliberately not a test that fails until a study is run. That would test
    project bookkeeping rather than product behaviour, and it would be
    satisfiable by editing a constant -- which is exactly the kind of green
    tick that makes a suite worth less than it looks.

    What is worth checking is that a result, once written, is complete enough
    to act on: which policy was heard, at what rate, by how many people, on
    what device. A half-filled artifact is worse than none, because it looks
    like evidence.
    """
    if path is None:
        pytest.skip("no listening study has been run yet")

    data = json.loads(path.read_text(encoding="utf-8"))
    for field, kind in EVALUATION_SCHEMA.items():
        assert field in data, f"{path.name}: missing {field!r}"
        assert isinstance(data[field], kind), (
            f"{path.name}: {field!r} should be {kind}, got {type(data[field])}")

    assert data["participants"] > 0
    for kind_name in ("price", "phone", "reference"):
        assert kind_name in data["results"], (
            f"{path.name}: no result for {kind_name} -- the three types are "
            f"scored separately because their failure modes are not comparable")


def test_the_protocol_is_written_down():
    """A stimulus generator with no protocol beside it is a pile of clips."""
    doc = ROOT / "docs" / "listening-test.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    # The two things most likely to be lost if somebody rewrites this: that
    # first hearing is the whole point, and that this is not E12-S11.
    assert "once" in text.lower()
    assert "E12-S11" in text


def test_the_stimulus_generator_runs():
    """Imported rather than shelled out: the point is that it composes real
    carrier sentences from the real verbaliser, not that argparse works."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "listening_stimuli", ROOT / "tools" / "listening_stimuli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    stimuli = module._stimuli()
    assert len(stimuli) >= 12
    assert {s["kind"] for s in stimuli} == {"price", "phone", "reference"}

    for s in stimuli:
        assert s["expected"] in s["text"], "the figure is not in its carrier"
        assert s["text"] != s["spoken"], "the stimulus was never verbalised"
        if s["kind"] in ("phone", "reference"):
            assert GROUP_SEPARATOR in s["spoken"], (
                f"{s['id']}: the stimulus is not grouped, so the study would "
                f"be measuring the old behaviour")
