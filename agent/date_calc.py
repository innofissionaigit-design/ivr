"""The deterministic date calculator. Code owns the calendar; the model does not.

story title: The model never originates a fact
user story: As a clinical lead, I want every price, date and identifier to come
    from a verified system response, so that a wrong answer is a data bug rather
    than a model bug.
acceptance criteria: Every factual sentence is a template substitution from a
    validated tool response and the model is never shown a figure it could
    restate. An automated assertion on every commit proves no model-composed
    span reaches synthesis on a factual intent.

THE DIVISION OF AUTHORITY
-------------------------
    LLM              understand that "আগামী সপ্তাহে" means NEXT WEEK
    this module      turn NEXT WEEK into 2026-09-14 .. 2026-09-20
    caller           confirm, when the interpretation covers more than one day
    clinic-api       establish what is actually true on those dates
    templates        say the result

The model may name the meaning. It may not name the dates. Those are two
different jobs and they were previously done by the same component: llm.py's
prompt carried today's date and asked the model to "resolve relative Bengali
time words ... to an ISO yyyy-mm-dd", so a 7B model was doing calendar
arithmetic and the answer went straight into a tool call and out of the
agent's mouth.

WHY AN EXPRESSION VOCABULARY AND NOT A DATE
-------------------------------------------
Three things get fixed by moving the model one step back:

1. Arithmetic stops being guesswork. "আগামী শনিবার" is a lookup here and a
   plausible-token-sequence there.

2. THE CACHE STOPS GOING STALE. agent/semantic_cache.py stores the extraction
   keyed by the caller's words, with a 6-hour TTL. An ISO date cached at 20:00
   for "কাল" is served at 01:00 to a caller for whom "কাল" now means a
   different day -- a wrong date, from a correct cache, with nothing anywhere
   logging a problem. An EXPRESSION never goes stale: "tomorrow" is still
   "tomorrow" six hours later, and it is resolved fresh on every turn.

3. Today's date leaves the prompt entirely. It was the one figure the model
   was ever shown, and "the model is never shown a figure it could restate"
   becomes literally true rather than nearly true.

WHY A RANGE IS DIFFERENT FROM A DAY
-----------------------------------
"আগামীকাল" is one day: there is nothing for the caller to disambiguate and
asking would be noise. "আগামী সপ্তাহে" is seven, and the agent cannot answer
"is Dr Sen in?" about seven days at once -- so it says which seven it means
and lets the caller agree or narrow.

The confirmation sentence may safely contain the dates BECAUSE THEY WERE
COMPUTED HERE. That is the whole point of the split: a date the model invented
could not be read out, even inside a question, without asserting something
nobody verified.
"""
from __future__ import annotations

import calendar
import dataclasses
import datetime
import logging

logger = logging.getLogger("date_calc")

SOURCE_PARSED = "parsed"            # read from the caller's own words, in code
SOURCE_INTERPRETED = "interpreted"  # model named the expression, code did the maths
SOURCE_UNMAPPED = "unmapped"        # caller named a day; nothing could express it
SOURCE_ABSENT = "absent"            # no date anywhere

# The escape hatch in the vocabulary. Without it "the model returned no
# expression" would mean two different things -- the caller mentioned no day,
# and the caller mentioned one nothing could express -- and those need opposite
# handling: default to today, versus ask which day. Collapsing them is how the
# old code came to answer about today when the caller had clearly said
# otherwise.
EXPRESSION_UNMAPPED = "other"

# Weeks start on MONDAY. Today is Thursday 10 Sep 2026 -> "next week" is
# 14 Sep (Mon) to 20 Sep (Sun), which is what a Kolkata caller means by
# "আগামী সপ্তাহে". Named rather than inlined so the convention is arguable
# in one place instead of being buried in three date offsets.
WEEK_STARTS_ON = 0  # Monday, matching datetime.date.weekday()

_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# The CLOSED vocabulary llm.py may emit. Anything outside it is treated as no
# answer at all -- see resolve_expression(). A model that invents
# "the_week_after_next" gets a re-prompt, not a guess: an unrecognised
# expression is exactly the case where quietly picking something is how a
# wrong date reaches a caller.
EXPRESSIONS = frozenset(
    {"today", "tomorrow", "day_after_tomorrow",
     "this_week", "next_week", "this_weekend", "next_weekend",
     "this_month", "next_month"}
    | set(_WEEKDAY_INDEX)
    | {f"next_{d}" for d in _WEEKDAY_INDEX}
)

# What the model may emit: everything resolvable, plus the escape hatch.
# llm.py imports THIS and renders it into its prompt rather than restating the
# list, so the vocabulary the model is told about and the one this module can
# resolve cannot drift apart -- the standing failure of every "keep these two
# lists in sync" comment ever written.
VOCABULARY = EXPRESSIONS | {EXPRESSION_UNMAPPED}


@dataclasses.dataclass(frozen=True)
class DateSpan:
    """One resolved date, or one resolved range. Frozen: a computed date is a
    finding, not a working value."""

    start: str | None       # ISO
    end: str | None         # ISO; equal to start for a single day
    expression: str         # what was interpreted, for the log and the tests
    source: str             # parsed | interpreted | absent

    @property
    def is_range(self) -> bool:
        return self.start is not None and self.start != self.end

    @property
    def needs_confirmation(self) -> bool:
        """Policy, stated once: a range is confirmed, a single day is not.

        A single day resolved from the caller's own words has nothing to
        disambiguate, and asking about it every time would train callers to
        say হ্যাঁ without listening -- which is how a confirmation step stops
        being one.
        """
        return self.is_range


def _week_bounds(day: datetime.date) -> tuple[datetime.date, datetime.date]:
    start = day - datetime.timedelta(days=(day.weekday() - WEEK_STARTS_ON) % 7)
    return start, start + datetime.timedelta(days=6)


def _month_bounds(day: datetime.date) -> tuple[datetime.date, datetime.date]:
    last = calendar.monthrange(day.year, day.month)[1]
    return day.replace(day=1), day.replace(day=last)


def _next_weekday(today: datetime.date, weekday: int, *, skip_a_week: bool) -> datetime.date:
    ahead = (weekday - today.weekday()) % 7
    ahead = ahead or 7  # naming today's weekday means the NEXT one
    if skip_a_week:
        ahead += 7
    return today + datetime.timedelta(days=ahead)


def resolve_expression(expression: str | None,
                       today: datetime.date | None = None) -> tuple[str, str] | None:
    """-> (start_iso, end_iso) for one expression, or None if unrecognised.

    None is deliberate and load-bearing. The alternative -- falling back to
    today, or to a best guess -- would turn a model that emitted nonsense into
    an agent that answered about the wrong day, which is the exact failure this
    whole story exists to remove.
    """
    if not expression or expression not in EXPRESSIONS:
        if expression:
            logger.warning("unrecognised date expression %r -- treating as no date",
                           expression)
        return None

    today = today or datetime.date.today()

    if expression == "today":
        return today.isoformat(), today.isoformat()
    if expression == "tomorrow":
        d = today + datetime.timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if expression == "day_after_tomorrow":
        d = today + datetime.timedelta(days=2)
        return d.isoformat(), d.isoformat()

    if expression in ("this_week", "next_week"):
        start, end = _week_bounds(today)
        if expression == "next_week":
            start, end = start + datetime.timedelta(days=7), end + datetime.timedelta(days=7)
        elif start < today:
            # "this week" from mid-week means the REST of this week. Offering a
            # caller Monday when it is already Thursday is not an answer.
            start = today
        return start.isoformat(), end.isoformat()

    if expression in ("this_weekend", "next_weekend"):
        week_start, _ = _week_bounds(today)
        if expression == "next_weekend":
            week_start += datetime.timedelta(days=7)
        saturday = week_start + datetime.timedelta(days=5)
        if expression == "this_weekend" and saturday < today:
            saturday += datetime.timedelta(days=7)
        return saturday.isoformat(), (saturday + datetime.timedelta(days=1)).isoformat()

    if expression in ("this_month", "next_month"):
        anchor = today
        if expression == "next_month":
            anchor = (today.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
        start, end = _month_bounds(anchor)
        if expression == "this_month" and start < today:
            start = today
        return start.isoformat(), end.isoformat()

    skip = expression.startswith("next_")
    name = expression[5:] if skip else expression
    d = _next_weekday(today, _WEEKDAY_INDEX[name], skip_a_week=skip)
    return d.isoformat(), d.isoformat()


def resolve(transcript: str, date_expr: str | None = None,
            today: datetime.date | None = None) -> DateSpan:
    """-> DateSpan. The one entry point main.py uses.

    ORDER OF AUTHORITY, highest first:

      1. slot_parse.parse_date() on the caller's literal words. Fully
         deterministic and needs nobody's interpretation, so it wins outright
         and is never confirmed.
      2. The model's expression, run through this module's calendar. The model
         supplied the MEANING; every digit came from code here.
      3. Nothing.

    Imported inside the function rather than at module scope: slot_parse
    imports nothing from here today, and keeping the dependency one-way and
    lazy means neither module can become the other's prerequisite later.
    """
    from agent.slot_parse import parse_date

    literal = parse_date(transcript, today=today)
    if literal is not None:
        return DateSpan(literal, literal, "literal", SOURCE_PARSED)

    if date_expr == EXPRESSION_UNMAPPED:
        # The caller named a day and the model could not express it. Not the
        # same as no date: answering about today here would be exactly the
        # silent wrong answer this story removes.
        return DateSpan(None, None, EXPRESSION_UNMAPPED, SOURCE_UNMAPPED)

    span = resolve_expression(date_expr, today=today)
    if span is not None:
        return DateSpan(span[0], span[1], date_expr, SOURCE_INTERPRETED)

    return DateSpan(None, None, "none", SOURCE_ABSENT)
