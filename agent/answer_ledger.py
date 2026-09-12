"""What this caller has already been told, so a repeat cannot contradict it.

story title: The same question gets the same answer within one call
user story: As a caller who asks twice, I want the same answer, so that I
    know which one to believe.
acceptance criteria: Repeating a question in one call produces an identical
    factual answer unless the underlying data changed, in which case the
    change is stated. A test asserts consistency across three repeats with
    an unchanged backend.

WHAT THIS IS NOT: A REPLY CACHE
-------------------------------
The Architecture Plan's failure-mode table already rules on the caching half
of this question. Its row for "a patient asks the same question twice in one
call" reads: the second answer is instant because the cache holds the INTENT
and never the answer, so a price that changed between the two turns still
reaches the caller live -- and its prescribed mitigation is, in full, "None.
Do not optimise by caching replies."

Nothing here caches a reply or serves a remembered one. Every repeat still
runs its own clinic lookup and renders its own sentence from that response.
This module only WATCHES: it records what the clinic said, and when the same
question comes round again it compares. The answer the caller hears is always
the fresh one. The only thing the record can add is a sentence saying the
figure moved.

That is the half the doc never addressed. Its row establishes that a repeat is
never stale; it says nothing about the caller who wrote down 450 and is now
told 500 with no acknowledgement that anything happened. Silence there is the
worse failure of the two, because from the caller's seat a live update and a
system contradicting itself sound exactly alike -- which is the whole of "so
that I know which one to believe".

WHY THE CLINIC CANNOT SIMPLY BE ASKED
-------------------------------------
E3-S5 (versioned knowledge with effective dates) is ABSENT: clinic-api stores
one current rate per test with no valid_from/valid_to, so a price change
overwrites history and there is no field anywhere that reports one. Observing
across turns is not a shortcut here, it is the only mechanism available.

FACTS, NOT SENTENCES
--------------------
The obvious reading of "an identical factual answer" is to compare the two
spoken strings. That is wrong, and it fails on correct behaviour.

reply_templates._spoken_test_name falls back to ECHOING THE CALLER'S OWN WORDS
when a test carries no seeded Bengali alias. Two repeats of one question are
two different recordings, so two ASR strings, so possibly two different
slots["test_name"] -- two different sentences carrying one identical fact. A
string comparison would announce a data change that never happened, on a path
where the clinic never moved.

So the ledger stores the values that CAME FROM THE CLINIC and compares those.
The sentence is free to vary; the fact is not.

WHAT MAKES TWO QUESTIONS THE SAME QUESTION
-------------------------------------------
The RESOLVED catalogue identity, never the caller's words. If clinic-api
resolved two differently-worded utterances to the same row, it is the same
question and consistency is owed regardless of how it was asked. If it
resolved them to different rows, they are different questions and consistency
was never owed -- no false contradiction.

The caller's words are kept as a SECOND key into the same entry, which is what
lets a not-found followed by a found be seen as one question. It has a second
effect worth stating out loud rather than discovering later: when the same
words resolve to a DIFFERENT row on the second ask, that is reported as a
change. It is not a data change -- it is the fuzzy match or the semantic cache
landing somewhere else -- but from the caller's seat it is indistinguishable
from one, and E9-S12 is explicit that a wrong hit is a wrong answer and
belongs with the safety metrics. Announcing it is right; the log line says
which case it was.

BOOKING IS EXCLUDED, DELIBERATELY
----------------------------------
A booking is a write, not a question, and asking twice is a second booking
rather than a repeat. The Gap Analysis puts that squarely in E4: "generate an
idempotency key per booking intent... return the original result on a repeat."
Folding it in here would build half of that story badly.

It also keeps every caller-specific field out of this object. A ledger holds
catalogue identities and clinic facts -- no patient name, no phone -- so
turn_log.py's PHI rule holds here without needing an exception.
"""
from __future__ import annotations

import re

FIRST = "first"
SAME = "same"
CHANGED = "changed"

# Booking is absent on purpose -- see the module docstring.
LEDGERED_INTENTS = ("test_rate", "doctor_availability", "doctors_by_department")

# Which response field carries the catalogue's own name for the thing. The
# caller's name for the same thing arrives in the slot of the same name, which
# is why one map serves both -- they are read separately below because only
# one of the two is authoritative.
_CANONICAL_FIELD = {
    "test_rate": "test_name",
    "doctor_availability": "doctor_name",
    "doctors_by_department": "department",
}

# A call has tens of turns, not thousands, so this is a bound against a
# pathological session rather than a working eviction policy.
MAX_ENTRIES = 64

_RE_WS = re.compile(r"\s+")
_RE_STRIP = re.compile("[।?!,.'\"‌‍]+")


def _fold(value) -> str:
    """Loose normalisation, for key comparison only and never for anything
    spoken.

    Deliberately a local copy of the same three lines in
    semantic_cache.normalize_text rather than an import: that one is tuned for
    embedding lookup and is free to change for reasons that have nothing to do
    with this, and sharing it would couple a cache-tuning decision to whether
    two answers are judged to contradict each other.
    """
    return _RE_WS.sub(" ", _RE_STRIP.sub(" ", str(value).strip().lower())).strip()


def _text(value):
    """Compare a clinic value as the STRING it arrived as.

    Never float(). tools_client._parse_exact already parses with
    parse_float=str precisely so a rate keeps the digits the clinic sent, and
    float("450.00") == float("450.0") would erase a real change at exactly the
    boundary the number-fidelity story exists to protect.

    bool passes through rather than being stringified, so `available` stays a
    two-valued thing instead of becoming "True"/"False" text.
    """
    if value is None or isinstance(value, bool):
        return value
    return str(value)


def identity_keys(intent: str, slots: dict, result: dict) -> list[tuple]:
    """Every key under which this answer should be findable, best first.

    The "id" key is the catalogue's identity and is authoritative. The "said"
    key is the caller's own words, and exists so that a not-found answer --
    which carries no catalogue identity at all, only the query echoed back --
    can still be recognised when the same thing is asked again.

    The date is part of the key for the two dated intents: "is Dr Sen in
    today" asked at 23:59 and again at 00:01 are different questions, and a
    date-less ledger would report a contradiction between two perfectly
    correct answers.

    One known limit, stated rather than papered over: a not-found availability
    response carries no date, so it cannot sit on the same axis as a dated
    answer, and a doctor who appears in the catalogue mid-call is not compared
    against the earlier "no such doctor". test_rate carries no date and does
    not have this gap.
    """
    field = _CANONICAL_FIELD.get(intent)
    if field is None:
        return []

    date = _text(result.get("date")) if intent != "test_rate" else None

    keys: list[tuple] = []
    canonical = result.get(field)
    if canonical:
        keys.append((intent, "id", _fold(canonical), date))
    for spoken in (slots.get(field), result.get("query")):
        if spoken:
            key = (intent, "said", _fold(spoken), date)
            if key not in keys:
                keys.append(key)
    return keys


def facts(intent: str, result: dict) -> tuple:
    """The values this intent's template actually speaks, and nothing else.

    A field the caller never hears must not be able to trigger a change
    announcement, so each tuple mirrors what its template in
    reply_templates.py puts into the sentence.

    The resolved identity leads every tuple. It is redundant whenever the
    entry was found by its "id" key, and load-bearing whenever it was found by
    the caller's words: the same words resolving to a different row is a
    changed answer even in the rare case where both rows quote the same price.
    """
    field = _CANONICAL_FIELD.get(intent)
    if field is None:
        raise ValueError(f"no facts defined for intent {intent!r}")

    canonical = result.get(field)
    ident = _fold(canonical) if canonical else None
    found = bool(result.get("found"))

    if intent == "test_rate":
        if not found:
            # The suggestions are spoken, so a change in them changes the
            # answer. Sorted, because the clinic promises no order and a
            # reshuffle is not something to announce.
            return (ident, False, tuple(sorted(result.get("did_you_mean_bn") or ())))
        return (ident, True, _text(result.get("rate_inr")),
                _text(result.get("sample_type")),
                _text(result.get("report_time_hours")))

    if intent == "doctor_availability":
        if not found:
            return (ident, False)
        return (ident, True, bool(result.get("available")),
                _text(result.get("chamber_hours")),
                _text(result.get("next_available_date")))

    if not found:
        return (ident, False)
    # The name as SPOKEN, in doctors_by_department_reply's own order of
    # preference, so a doctor gaining a Bengali alias mid-call does not read
    # as the roster having changed.
    names = tuple(sorted(
        str(doc.get("doctor_name_bn") or doc.get("name") or "")
        for doc in (result.get("doctors") or [])
    ))
    return (ident, True, names)


class AnswerLedger:
    """One per call. Dies with the call.

    Not thread-safe, and not by oversight: every touch happens on the asyncio
    event loop inside CallSession.dispatch_lock, and a lock here would buy
    nothing but the impression that this is shared with something.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES):
        self._max = max_entries
        self._entries: dict[int, dict] = {}     # slot id -> {facts, asked}
        self._index: dict[tuple, int] = {}      # any identity key -> slot id
        self._next_id = 0
        self.stats = {"repeats": 0, SAME: 0, CHANGED: 0}

    def check(self, intent: str, slots: dict, result: dict) -> tuple[str, tuple | None]:
        """-> (verdict, the facts said last time), and record this answer.

        Recording is a side effect rather than a second call on purpose: a
        check that CAN be performed without recording is a check somebody
        eventually performs without recording, and the entry is then missing
        for the turn after.
        """
        if intent not in LEDGERED_INTENTS or not isinstance(result, dict):
            return FIRST, None

        # story title: Near matches are offered rather than guessed or refused
        # An ambiguous response is a QUESTION, not an answer, and recording it
        # would be worse than not recording it: the entry's facts would be
        # (None, False), and the disambiguated answer one turn later would
        # then be reported as a change and prefixed with "that has changed
        # since I told you" -- announcing a move that never happened, on the
        # turn where the caller is least sure of themselves.
        if result.get("ambiguous"):
            return FIRST, None

        keys = identity_keys(intent, slots, result)
        if not keys:
            # Nothing nameable was resolved, so there is nothing this answer
            # could be consistent WITH. Silence beats a guessed identity.
            return FIRST, None

        current = facts(intent, result)
        slot_id = next((self._index[key] for key in keys if key in self._index), None)

        if slot_id is None:
            slot_id = self._next_id
            self._next_id += 1
            self._entries[slot_id] = {"facts": current, "asked": 1}
            verdict, previous = FIRST, None
        else:
            entry = self._entries[slot_id]
            previous = entry["facts"]
            verdict = SAME if previous == current else CHANGED
            # Overwritten, so a third ask after a change says SAME rather than
            # announcing the same move twice. The caller is told once.
            entry["facts"] = current
            entry["asked"] += 1
            self.stats["repeats"] += 1
            self.stats[verdict] += 1

        for key in keys:
            self._index[key] = slot_id
        self._evict()
        return verdict, previous

    def _evict(self) -> None:
        while len(self._entries) > self._max:
            oldest = min(self._entries)
            del self._entries[oldest]
            self._index = {k: v for k, v in self._index.items() if v != oldest}

    def snapshot(self) -> dict:
        """Same shape as the other snapshot() methods in this package.

        `changed` is the alertable number. On a catalogue nobody is editing it
        should be zero; a rising rate is either real churn in the clinic's
        data or entity resolution landing on a different row for the same
        words, and both are things somebody should be told about rather than
        things a caller should discover.
        """
        return {**self.stats, "entries": len(self._entries)}
