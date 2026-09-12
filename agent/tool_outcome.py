"""Which of three things happened when the clinic was asked a question.

story title: A thing not existing is never confused with a system being down
user story: As a caller, I want to know whether my test does not exist or the
    system cannot be reached, so that I know whether to call back.
acceptance criteria: The two produce different spoken sentences and different
    metrics, and the distinction survives every refactor. This behaviour exists
    today and gains a permanent regression case.

    ANSWERED     the clinic responded, whatever it said
    NOT_FOUND    the clinic responded "that thing does not exist"
    UNREACHABLE  the clinic did not respond

NOT_FOUND is a SUBSET of ANSWERED, counted separately rather than instead,
because the interesting number is the ratio and not either half.

WHY THIS LIVES BESIDE tools_client AND NOT IN main.py
------------------------------------------------------
main.py has seven `except ToolCallError` handlers and five not-found branches,
spread across _dispatch_turn and _continue_pending. Counting at each of the
twelve is twelve places to remember and one to eventually forget -- and the
acceptance criterion is specifically that the distinction "survives every
refactor", which a twelve-site convention does not.

agent/tools_client.py is the single choke point: every clinic call returns
through it and every failure is raised from it, so it can classify without
anyone remembering to tell it. Same argument that put the speakability counter
inside TTSClient.synthesize rather than in _speak.

THE ONE RULE THAT NEEDS CARE
----------------------------
book_appointment fails for four different reasons and only ONE of them is a
thing not existing:

    doctor_not_found              -> NOT_FOUND
    slot_taken                    -> ANSWERED
    missing_field                 -> ANSWERED
    doctor_not_available_that_day -> ANSWERED

The last three are the clinic ANSWERING -- the doctor exists, the appointment
just cannot be made as asked. Counting them as not-found would make the metric
meaningless inside a week, and would do it quietly: the number would still go
up and down, it would simply stop meaning what its name says.

The same care applies to doctors_by_department, where `found: true` with an
empty `doctors` list means the department exists and nobody sits that day.
That is an answer, not an absence.
"""
from __future__ import annotations

ANSWERED = "answered"
NOT_FOUND = "not_found"
UNREACHABLE = "unreachable"

# story title: The agent says it cannot confirm rather than guessing
# user story: As a caller, I want to be told plainly when the system
#   cannot verify something, so that I am not given a confident guess.
# acceptance criteria: The insufficient-verified-information outcome has
#   its own template per language, its own metric and its own escalation
#   path, distinct from not-found and from an infrastructure apology. Its
#   rate is reported per intent because a rise means a data or
#   integration problem.
#
# THE FOURTH OUTCOME. The clinic replied, the write happened, and the reply is
# unusable -- see agent/outcomes.py for the scenario. It is counted here rather
# than in a parallel counter of its own so that one object, one endpoint and
# one snapshot shape carry all four; a second counter would mean a second thing
# to remember to look at, which is how half a metric gets built.
#
# It is NOT a subset of anything. answered/not_found describe what the clinic
# SAID; this describes a reply that cannot be read aloud at all, so it is
# excluded from not_found_rate's denominator for the same reason unreachable
# is: folding it in would let a data-integrity incident look like the catalogue
# improving.
INSUFFICIENT = "insufficient"

# story title: Near matches are offered rather than guessed or refused
# user story: As a caller naming something loosely, I want the close matches
#   offered, so that I am not told my test does not exist when it does.
# acceptance criteria: When several catalogue rows fall within the match band
#   the agent offers up to three by name and asks which. Candidates are
#   generated across every supported language and romanised spelling. The
#   did-you-mean path covers the ambiguous case and not only total failure.
#
# THE FIFTH OUTCOME, and unlike INSUFFICIENT it IS a subset of ANSWERED --
# the same relationship NOT_FOUND has, and counted the same way. The clinic
# replied, and what it said was "several rows match, I will not choose for
# you". That is an answer.
#
# It needs its own bucket rather than folding into ANSWERED because the rate
# is a catalogue-quality signal: a rising ambiguous_rate means rows the
# catalogue cannot tell apart by name, which is a data problem with a data
# fix. And it must NOT be counted as NOT_FOUND, which is the trap the
# clinic-api response shape is built to avoid -- an ambiguity is the opposite
# of a thing not existing, it is the thing existing twice.
AMBIGUOUS = "ambiguous"

TOOLS = ("test_rate", "doctor_availability", "doctors_by_department", "book_appointment")

# book_appointment reasons that mean the named thing does not exist, as opposed
# to existing and being unavailable. Explicit allow-list, not a "contains
# not_found" string test: a future reason like "slot_not_found" would mean a
# slot is taken, not that anything is missing.
_NOT_FOUND_BOOKING_REASONS = frozenset({"doctor_not_found"})


def classify(tool: str, payload: dict) -> str:
    """-> ANSWERED | NOT_FOUND, for a response the clinic actually returned.

    UNREACHABLE is never returned here: it is not a property of a payload,
    because there is no payload. tools_client records it on the exception
    path instead.
    """
    # story title: Near matches are offered rather than guessed or refused
    # Checked BEFORE `found`, because an ambiguous response carries
    # found=false and would otherwise be tallied as a thing that does not
    # exist -- the precise inversion this outcome exists to prevent.
    if payload.get("ambiguous") or payload.get("reason") == "doctor_ambiguous":
        return AMBIGUOUS

    if tool == "book_appointment":
        if payload.get("success"):
            return ANSWERED
        return (NOT_FOUND if payload.get("reason") in _NOT_FOUND_BOOKING_REASONS
                else ANSWERED)
    return ANSWERED if payload.get("found") else NOT_FOUND


class OutcomeCounter:
    """Per-tool tallies, read by /api/stats.

    Not thread-safe on purpose: every increment happens on the asyncio event
    loop, and a lock here would buy nothing but a false impression that this
    is shared with something.
    """

    def __init__(self):
        self._counts = {tool: {ANSWERED: 0, NOT_FOUND: 0, UNREACHABLE: 0,
                               INSUFFICIENT: 0, AMBIGUOUS: 0}
                        for tool in TOOLS}

    def record(self, tool: str, outcome: str) -> None:
        bucket = self._counts.get(tool)
        if bucket is not None and outcome in bucket:
            bucket[outcome] += 1

    def snapshot(self) -> dict:
        """Same shape the other snapshot() methods in this package return.

        not_found_rate is derived here rather than by whoever reads it, because
        it is the number an alert would watch and it should mean one thing.
        It is alertable in BOTH directions:

          unreachable rising      -> the clinic API, or the network to it.
          not_found_rate rising   -> either callers asking for things the
                                     clinic does not stock, which is a business
                                     signal, or the catalogue has emptied
                                     itself, which is an incident. Today a
                                     wiped database tells every caller their
                                     test does not exist, in a perfectly
                                     well-formed sentence, and nothing anywhere
                                     notices. This is the number that catches
                                     it.

        None rather than 0.0 when nothing has been asked yet: a rate over zero
        calls is not zero, it is unknown, and an alert on "rate == 0" firing at
        process start is how a metric earns itself an exclusion rule.
        """
        out = {}
        for tool, counts in self._counts.items():
            # story title: Near matches are offered rather than guessed or
            #   refused
            # AMBIGUOUS joins the denominator because it IS the clinic
            # answering -- leaving it out would make not_found_rate drift
            # upward every time the catalogue got harder to disambiguate,
            # which is the opposite of what that number means.
            answered = counts[ANSWERED] + counts[NOT_FOUND] + counts[AMBIGUOUS]
            out[tool] = {
                **counts,
                "not_found_rate": (round(counts[NOT_FOUND] / answered, 3)
                                   if answered else None),
                # A catalogue-quality signal, not a traffic one: it rises when
                # rows stop being tellable apart by name, and the fix is in
                # the data (an alias that distinguishes them), not the code.
                "ambiguous_rate": (round(counts[AMBIGUOUS] / answered, 3)
                                   if answered else None),
            }
        return out
