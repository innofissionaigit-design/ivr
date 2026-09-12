"""Several rows match, so the caller is asked which -- not guessed at.

story title: Near matches are offered rather than guessed or refused
user story: As a caller naming something loosely, I want the close matches
    offered, so that I am not told my test does not exist when it does.
acceptance criteria: When several catalogue rows fall within the match band
    the agent offers up to three by name and asks which. Candidates are
    generated across every supported language and romanised spelling. The
    did-you-mean path covers the ambiguous case and not only total failure.

THE FOUR CASES AT THE TOP ARE REPRODUCTIONS, not invented fixtures. Every one
of them was run against the real seeded catalogue before this story and
resolved to whichever row the database listed first, with no scoring and no
record that a choice had been made:

    "Blood Sugar" / "sugar" / "সুগার"  ->  Blood Sugar Fasting | Blood Sugar PP
    "Vitamin" / "ভিটামিন"               ->  Vitamin D (25-OH)   | Vitamin B12

They are parametrised against seed.py's actual rows rather than a hand-written
list, so a catalogue edit that makes two rows indistinguishable shows up here.

WHAT THIS FILE DOES NOT TEST
----------------------------
The romanised-spelling half of the criterion. Departments carry romanised
aliases already ("cardio", "heart"); lab tests and doctors do not, and adding
them means editing seed.py, whose seed() is destructive. That was deliberately
left out of this change, so the gap is asserted AS a gap below
(test_the_romanised_gap_is_real) rather than quietly skipped -- a test that
records what is missing is worth more than a story that claims it is not.

Everything here is pure -- no GPU, no model, no network, no database.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import pathlib
import re
import sys
import unicodedata

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from agent import answer_ledger, fast_path, speakability, tool_contract, tool_outcome  # noqa: E402
from agent.bn_normalize import verbalize  # noqa: E402
from agent.reply_templates import NEAR_MATCH_UNCLEAR_BN, near_match_prompt  # noqa: E402

MAIN = ROOT / "main.py"
CLINIC = ROOT / "clinic-api"


def _load(name: str):
    """clinic-api is not an importable package -- the hyphen makes it not an
    identifier -- so its pure modules are loaded by path. Same reason
    test_speakability.py reads seed.py as text instead of importing it."""
    spec = importlib.util.spec_from_file_location(name, CLINIC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


match_band = _load("match_band")


def _seed_literal(name: str):
    """Lift one top-level literal out of seed.py without importing it (it
    pulls in SQLAlchemy and a live database at import time)."""
    source = (CLINIC / "seed.py").read_text(encoding="utf-8")
    match = re.search(name + r"\s*=\s*[\[{].*?\n[\]}]", source, re.S)
    assert match, f"{name} is gone from seed.py"
    namespace: dict = {}
    exec(match.group(0), namespace)  # noqa: S102 - our own source, read by path
    return namespace[name]


LAB_TESTS = _seed_literal("LAB_TESTS")
SURNAME_BN = _seed_literal("SURNAME_BN")
DEPARTMENTS = _seed_literal("DEPARTMENTS")
DEPARTMENT_ALIASES = _seed_literal("DEPARTMENT_ALIASES")

TEST_ROWS = [(name, [name, *aliases]) for name, aliases, *_ in LAB_TESTS]
DOCTOR_NAMES = [d for docs in DEPARTMENTS.values() for d, _quals in docs]
DOCTOR_ROWS = [(n, [n, n.split()[-1], *SURNAME_BN.get(n.split()[-1], [])])
               for n in DOCTOR_NAMES]
DEPT_ROWS = [(d, [d, *DEPARTMENT_ALIASES.get(d, [])]) for d in DEPARTMENTS]


def _verdict(query, rows):
    verdict, candidates = match_band.decide(match_band.rank(query, rows))
    return verdict, [c.key for c in candidates]


def _says(phrase: str, sentence: str) -> bool:
    return unicodedata.normalize("NFC", phrase) in unicodedata.normalize("NFC", sentence)


# ------------------------------------------- the reproductions, as fixtures

@pytest.mark.parametrize("query,expected", [
    ("Blood Sugar", {"Blood Sugar Fasting", "Blood Sugar PP"}),
    ("sugar", {"Blood Sugar Fasting", "Blood Sugar PP"}),
    ("সুগার", {"Blood Sugar Fasting", "Blood Sugar PP"}),
    ("Vitamin", {"Vitamin D (25-OH)", "Vitamin B12"}),
    ("ভিটামিন", {"Vitamin D (25-OH)", "Vitamin B12"}),
])
def test_a_query_matching_two_rows_is_offered_not_guessed(query, expected):
    verdict, rows = _verdict(query, TEST_ROWS)
    assert verdict == match_band.AMBIGUOUS, (
        f"{query!r} resolved to {rows} instead of asking")
    assert expected <= set(rows)


def test_a_clear_match_still_commits():
    """The guard must be invisible on every unambiguous lookup. This is the
    regression half: a story that makes the agent ask more often is only
    correct if it does not ask when it already knew."""
    for query, expected in (("সিবিসি", "Complete Blood Count (CBC)"),
                            ("Uric Acid", "Uric Acid"),
                            ("থাইরয়েড", "Thyroid Profile (T3 T4 TSH)"),
                            ("কোলেস্টেরল", "Lipid Profile")):
        verdict, rows = _verdict(query, TEST_ROWS)
        assert (verdict, rows) == (match_band.COMMIT, [expected]), query


def test_an_exact_alias_beats_a_containment():
    """সেন EQUALS Dr Sen's alias and is CONTAINED IN Dr Sengupta's. Flat
    containment scoring would ask the caller to choose between a doctor they
    named exactly and one they did not -- correct-looking, and worse than the
    bug it replaced."""
    assert _verdict("সেন", DOCTOR_ROWS) == (match_band.COMMIT, ["Dr. A. Sen"])
    assert _verdict("Sen", DOCTOR_ROWS) == (match_band.COMMIT, ["Dr. A. Sen"])
    assert _verdict("সেনগুপ্ত", DOCTOR_ROWS) == (match_band.COMMIT, ["Dr. P. Sengupta"])


def test_the_doctor_nobody_incident_now_asks():
    """The bug that set FUZZY_SURNAME_FLOOR to 0.60: "Doctor Nobody" fuzzy-
    matched a real doctor at 0.522. The floor made it a refusal. It is now a
    question -- which the caller can answer with "no", and which is a better
    outcome than either the original wrong commit or the refusal that
    replaced it."""
    verdict, rows = _verdict("Doctor Nobody", DOCTOR_ROWS)
    assert verdict == match_band.AMBIGUOUS
    assert rows == ["Dr. N. Roy"]


def test_nothing_close_is_still_nothing():
    assert _verdict("xyzzy", TEST_ROWS) == (match_band.NONE, [])
    assert _verdict("", TEST_ROWS) == (match_band.NONE, [])


# --------------------------------------------------------- the band itself

def test_the_offer_is_capped_at_three():
    ranked = [match_band.Candidate(f"row{n}", 0.9, f"form{n}") for n in range(9)]
    verdict, offered = match_band.decide(ranked)
    assert verdict == match_band.AMBIGUOUS
    assert len(offered) == match_band.MAX_OFFERED == 3


def test_the_offer_is_ranked_best_first():
    ranked = match_band.rank("সুগার ফাস্টিং", TEST_ROWS)
    _verdict_, offered = match_band.decide(ranked)
    assert [c.score for c in offered] == sorted((c.score for c in offered), reverse=True)


def test_a_row_does_not_outrank_another_by_having_more_aliases():
    """One entry per row, keeping its best form. Counting forms instead would
    let a test with four aliases beat the row the caller actually named."""
    rows = [("many", ["সুগার", "সুগার", "সুগার", "সুগার"]), ("one", ["সুগার"])]
    ranked = match_band.rank("সুগার", rows)
    assert len(ranked) == 2
    assert {c.key for c in ranked} == {"many", "one"}


def test_a_thin_margin_is_ambiguous_and_a_wide_one_is_not():
    thin = [match_band.Candidate("a", 0.80, "a"), match_band.Candidate("b", 0.77, "b")]
    wide = [match_band.Candidate("a", 0.95, "a"), match_band.Candidate("b", 0.40, "b")]
    assert match_band.decide(thin)[0] == match_band.AMBIGUOUS
    assert match_band.decide(wide) == (match_band.COMMIT, [wide[0]])


def test_a_single_candidate_below_the_commit_floor_is_a_question():
    """The old did-you-mean, now reached by the same route as a two-way tie.
    That merge IS the criterion's third clause."""
    lone = [match_band.Candidate("a", 0.69, "a")]
    assert match_band.decide(lone) == (match_band.AMBIGUOUS, lone)


# ------------------------------------------------------------ the sentence

SUGAR = [{"name": "Blood Sugar Fasting", "name_bn": "সুগার ফাস্টিং"},
         {"name": "Blood Sugar PP", "name_bn": "সুগার পিপি"}]


def test_the_offer_names_the_candidates_and_asks():
    prompt = near_match_prompt(SUGAR)
    assert _says("সুগার ফাস্টিং", prompt) and _says("সুগার পিপি", prompt)
    assert prompt.endswith("?")


def test_the_offer_is_sayable():
    for candidates in (SUGAR, SUGAR[:1], SUGAR + [{"name": "H", "name_bn": "এইচবিএ১সি"}]):
        prompt = near_match_prompt(candidates)
        assert speakability.check(prompt).state == speakability.SPEAKABLE
        assert ":" not in verbalize(prompt)


def test_a_candidate_with_no_spoken_name_is_never_offered_alone():
    """Dropping the unsayable one would turn "which of these two" back into
    "did you mean this one" -- a guess wearing a question mark."""
    assert near_match_prompt([{"name": "X", "name_bn": None}]) == NEAR_MATCH_UNCLEAR_BN
    assert near_match_prompt([]) == NEAR_MATCH_UNCLEAR_BN


def test_the_reask_does_not_claim_the_thing_is_missing():
    """It is a re-ask, not a not-found. The row exists; the agent could not
    tell which one."""
    assert speakability.check(NEAR_MATCH_UNCLEAR_BN).state == speakability.SPEAKABLE
    assert not _says("তালিকায় নেই", NEAR_MATCH_UNCLEAR_BN)


# -------------------------------------------------- the outcome and gates

AMBIGUOUS_RESPONSE = {"found": False, "ambiguous": True, "query": "সুগার",
                      "candidates": SUGAR}


def test_ambiguous_is_not_counted_as_not_found():
    """An ambiguity is the opposite of a thing not existing -- it is the
    thing existing twice. Counting it as not-found would let a catalogue
    growing harder to disambiguate read as a catalogue emptying out."""
    assert tool_outcome.classify("test_rate", AMBIGUOUS_RESPONSE) == tool_outcome.AMBIGUOUS
    assert tool_outcome.classify(
        "book_appointment",
        {"success": False, "reason": "doctor_ambiguous"}) == tool_outcome.AMBIGUOUS


def test_the_ambiguous_rate_has_its_own_number_and_leaves_not_found_alone():
    counter = tool_outcome.OutcomeCounter()
    counter.record("test_rate", tool_outcome.ANSWERED)
    counter.record("test_rate", tool_outcome.NOT_FOUND)
    counter.record("test_rate", tool_outcome.AMBIGUOUS)
    counter.record("test_rate", tool_outcome.AMBIGUOUS)
    snap = counter.snapshot()["test_rate"]
    assert snap[tool_outcome.AMBIGUOUS] == 2
    assert snap["not_found_rate"] == 0.25      # 1 of 4 answers, not 1 of 2
    assert snap["ambiguous_rate"] == 0.5


def test_the_contract_requires_candidates_on_an_ambiguous_response():
    tool_contract.validate("test_rate", AMBIGUOUS_RESPONSE)
    with pytest.raises(tool_contract.ToolContractError):
        tool_contract.validate("test_rate", {"found": False, "ambiguous": True,
                                             "query": "সুগার"})


def test_an_empty_candidate_list_is_a_valid_ambiguous_response():
    """It means several rows matched and one of them cannot be said aloud.
    Requiring content would report a data defect as the clinic being down."""
    tool_contract.validate("test_rate", {"found": False, "ambiguous": True,
                                         "query": "সুগার", "candidates": []})


def test_an_ambiguous_response_is_never_ledgered_as_an_answer():
    """Recording it would give the entry empty facts, and the disambiguated
    answer one turn later would then be announced as a change that never
    happened."""
    ledger = answer_ledger.AnswerLedger()
    verdict, previous = ledger.check("test_rate", {"test_name": "সুগার"},
                                     AMBIGUOUS_RESPONSE)
    assert (verdict, previous) == (answer_ledger.FIRST, None)
    assert ledger.snapshot()["entries"] == 0


# ----------------------------------------------------- the fast path, too

class _Catalogue(fast_path.Catalogue):
    def __init__(self):
        super().__init__({
            "tests": [{"name": name, "aliases_bn": list(aliases)}
                      for name, aliases, *_ in LAB_TESTS],
            "doctors": [{"name": n, "surname": n.split()[-1],
                         "aliases_bn": SURNAME_BN.get(n.split()[-1], [])}
                        for n in DOCTOR_NAMES],
        })


def test_the_fast_path_reports_the_runner_up():
    name, _form, best, runner_up = _Catalogue().match(
        "ভিটামিন টেস্টের রেট কত", "test")
    assert name is not None
    assert runner_up > 0.0, "a second row scored and was not reported"
    # The measurement COMMIT_MARGIN was set from: Vitamin D and Vitamin B12
    # separated by 0.087, against a worst case of 0.158 for a caller who
    # names a test in full.
    assert round(best - runner_up, 3) == 0.087
    assert (best - runner_up) < fast_path.COMMIT_MARGIN


def test_the_runner_up_is_a_different_row_not_another_alias():
    """A test with four aliases must not look ambiguous with itself."""
    rows = _Catalogue()
    rows.tests = [("Only", ["সুগার ফাস্টিং", "সুগার", "খালি পেটে সুগার"])]
    _name, _form, best, runner_up = rows.match("সুগার", "test")
    assert best > 0.0 and runner_up == 0.0


def test_the_fast_path_abstains_on_a_tie_instead_of_committing():
    """The abstention has to happen HERE or nowhere: a fast-path commit
    writes a CANONICAL name into the slot, which reaches clinic-api as an
    exact string and resolves to one row. The ambiguity never gets there."""
    fp = fast_path.FastPath(_Catalogue())
    assert fp.resolve("ভিটামিন টেস্টের রেট কত") is None
    assert fp.snapshot()["abstained"] >= 1


def test_the_fast_path_still_serves_an_unambiguous_turn():
    fp = fast_path.FastPath(_Catalogue())
    result = fp.resolve("ইউরিক অ্যাসিড টেস্টের রেট কত")
    assert result is not None and result.intent == "test_rate"


# ------------------------------------------------------- the wired turn

class _Session:
    call_id = "nm01"
    utt_seq = 1

    def __init__(self):
        self.answer_ledger = answer_ledger.AnswerLedger()
        self.pending = None


@pytest.fixture
def wired(monkeypatch):
    said: list[str] = []

    async def _say(session, text, fallback_reason=None):
        said.append(text)

    monkeypatch.setattr(main, "_speak", _say)
    monkeypatch.setattr(main, "_consistency",
                        {"repeats": 0, answer_ledger.SAME: 0,
                         answer_ledger.CHANGED: 0})
    return _Session(), said


def test_an_ambiguous_lookup_offers_and_opens_the_choice_state(wired):
    session, said = wired
    handled = asyncio.run(main._speak_fact(
        session, "test_rate", {"test_name": "সুগার"}, AMBIGUOUS_RESPONSE,
        "THIS REPLY MUST NOT BE SPOKEN"))

    assert handled is True, "the caller must end the turn on an offer"
    assert said == [near_match_prompt(SUGAR)]
    assert session.pending["awaiting"] == "entity_choice"
    assert session.pending["intent"] == "test_rate"
    assert session.pending["candidates"] == SUGAR


def test_the_date_is_carried_into_the_choice_state(wired):
    """Otherwise "is Dr Sen in on Tuesday", answered with a choice and then
    resolved, silently becomes a question about today."""
    session, _said = wired
    asyncio.run(main._speak_fact(
        session, "doctor_availability", {"doctor_name": "সেন"},
        {"found": False, "ambiguous": True, "query": "সেন", "candidates": SUGAR},
        "unused", offered_date="2026-09-15"))
    assert session.pending["offered_date"] == "2026-09-15"


def test_an_unsayable_ambiguity_asks_again_without_opening_a_choice(wired):
    session, said = wired
    asyncio.run(main._speak_fact(
        session, "test_rate", {"test_name": "সুগার"},
        {"found": False, "ambiguous": True, "query": "সুগার", "candidates": []},
        "unused"))
    assert said == [NEAR_MATCH_UNCLEAR_BN]
    assert session.pending is None, "nothing to choose from, so no choice state"


def test_a_normal_answer_is_untouched(wired):
    session, said = wired
    handled = asyncio.run(main._speak_fact(
        session, "test_rate", {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "test_name_bn": "ইউরিক অ্যাসিড",
         "rate_inr": "250", "sample_type": "রক্ত", "report_time_hours": "12"},
        "THE NORMAL REPLY"))
    assert handled is False
    assert said == ["THE NORMAL REPLY"]
    assert session.pending is None


# -------------------------------------------------- answering the offer

def test_the_choice_is_matched_against_what_was_offered():
    assert main._match_offered("সুগার ফাস্টিং", SUGAR)["name"] == "Blood Sugar Fasting"
    assert main._match_offered("সুগার পিপি", SUGAR)["name"] == "Blood Sugar PP"


def test_an_answer_that_fits_both_offers_equally_is_refused():
    """The turn after an offer is the one place guessing would be least
    forgivable: the whole point of the previous turn was that the agent had
    stopped guessing."""
    assert main._match_offered("সুগার", SUGAR) is None


def test_an_unrelated_answer_is_refused():
    assert main._match_offered("থাইরয়েড", SUGAR) is None
    assert main._match_offered("", SUGAR) is None


def test_a_second_failed_choice_drops_the_state_rather_than_asking_again(wired):
    session, said = wired
    session.pending = {"awaiting": "entity_choice", "intent": "test_rate",
                       "slots": {}, "candidates": SUGAR, "offered_date": None,
                       "retries": 0}

    assert asyncio.run(main._continue_pending(session, "কিছু একটা")) is True
    assert said == [NEAR_MATCH_UNCLEAR_BN]
    assert asyncio.run(main._continue_pending(session, "আবার কিছু")) is True
    # Third failure: hand the turn back for a fresh classification instead of
    # asking a fourth time.
    assert asyncio.run(main._continue_pending(session, "আরো কিছু")) is False
    assert session.pending is None


def test_choosing_re_runs_the_lookup_against_the_canonical_name(wired, monkeypatch):
    """Against the CANONICAL name, not the caller's words -- so the second
    attempt cannot be ambiguous for the same reason the first was."""
    session, said = wired
    asked_for: list[str] = []
    answer = {"found": True, "test_name": "Blood Sugar PP",
              "test_name_bn": "সুগার পিপি", "rate_inr": "120",
              "sample_type": "রক্ত", "report_time_hours": "4"}

    class _Tools:
        async def get_test_rate(self, name):
            asked_for.append(name)
            return answer

    monkeypatch.setattr(main, "_tools", _Tools())
    session.pending = {"awaiting": "entity_choice", "intent": "test_rate",
                       "slots": {}, "candidates": SUGAR, "offered_date": None,
                       "retries": 0}

    assert asyncio.run(main._continue_pending(session, "সুগার পিপি")) is True
    assert asked_for == ["Blood Sugar PP"]
    assert _says("১২০", verbalize(said[-1])) or "120" in said[-1]
    assert session.pending is None


# ------------------------------------------------- survives every refactor

def _tree() -> ast.Module:
    return ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))


def test_every_speak_fact_call_ends_the_turn_when_it_returns_true():
    """_speak_fact returns True when it spoke an offer instead of the reply.
    A call site that ignores that carries on and speaks a second sentence --
    an answer about a thing the caller was just asked to choose between.

    Enforced statically because there are eleven call sites and the failure
    is silent: the offer still goes out, so a manual test looks fine right
    up to the second sentence.
    """
    tree = _tree()

    every = {node.lineno for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "_speak_fact"}
    assert len(every) >= 8, f"expected the factual routes, found {len(every)}"

    # An AST walk cannot see a node's parent, so collect the GUARDED shape --
    # `if await _speak_fact(...): return` -- and require the two sets to be
    # the same. Anything in one and not the other is a bare call.
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Await):
            continue
        call = node.test.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "_speak_fact"
                and len(node.body) == 1 and isinstance(node.body[0], ast.Return)):
            guarded.add(call.lineno)

    unguarded = sorted(every - guarded)
    assert not unguarded, (
        f"_speak_fact at line(s) {unguarded} ignores its return value -- an "
        f"ambiguity offer would be followed by an answer about the very thing "
        f"the caller was just asked to choose between"
    )


def test_no_clinic_lookup_keeps_its_own_best():
    """The rule that stops a fourth lookup being added next month with the
    old pattern: resolving an entity goes through match_band.decide, or it
    does not happen."""
    source = (CLINIC / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CLINIC / "main.py"))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = {t.id for n in ast.walk(node) if isinstance(n, ast.Assign)
                 for t in n.targets if isinstance(t, ast.Name)}
        suspect = {n for n in names if n.startswith("best")}
        assert not suspect, (
            f"{node.name}() keeps its own {sorted(suspect)} -- entity "
            f"resolution belongs in match_band.decide()"
        )


def test_the_three_lookups_all_go_through_the_band():
    source = (CLINIC / "main.py").read_text(encoding="utf-8")
    for fn in ("search_test", "_resolve_doctor", "_resolve_department"):
        assert re.search(fn + r"\b", source), f"{fn} is gone"
    assert source.count("match_band.decide(") == 3, (
        "expected exactly three entity lookups through the band")


# ----------------------------------------------------- the documented gap

def test_the_romanised_gap_is_real():
    """AC2 asks for candidates across every supported language and romanised
    spelling. Departments have romanised aliases; tests and doctors do not,
    and closing that means reseeding a destructive seed() -- an operational
    decision, deliberately not taken in this change.

    This test asserts the CURRENT state, so it fails the day someone adds the
    aliases, which is the day the gap closes and this should be rewritten as
    a coverage assertion rather than a gap one.
    """
    assert any(re.search(r"[a-z]", a) for a in DEPARTMENT_ALIASES["Cardiology"]), (
        "departments lost their romanised aliases")

    romanised_tests = [name for name, aliases, *_ in LAB_TESTS
                       if any(re.search(r"[a-z]", a) for a in aliases)]
    assert romanised_tests == [], (
        f"lab tests gained romanised aliases {romanised_tests} -- the AC2 gap "
        f"is closing; rewrite this test as a coverage assertion")
