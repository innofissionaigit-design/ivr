"""Cross-turn dialogue state for "Caller asks a follow-up that depends on
the previous answer" story.

ADDED BY SOURAV -- new file for this story. Evidence backing the backlog
item ("Each turn is independent") was literally true before this change:
agent/llm.py's extract_intent() classifies ONE utterance with zero memory
of anything said earlier in the call (see that module's own docstring),
and the only thing that ever survived between turns was main.py's
session.pending -- a narrow, single-purpose mechanism for "this ONE flow
is mid-way through asking for ONE specific missing field" (a booking, an
insurance lookup, this codebase's other two-required-slot intents). It
was never meant to, and does not, answer a different question: "what did
we just finish TALKING ABOUT, so a bare pronoun in the NEXT, otherwise
unrelated question can be resolved without making the caller repeat
themselves."

This module answers exactly that second question, and only that one --
mirroring agent/report_flow.py's and agent/compare_flow.py's own
established convention of keeping DECISION logic (here: what does "eta"/
"that one" refer to) in pure functions with no dependency on CallSession,
WebSocket, tools_client, or asyncio. main.py owns the one piece of actual
state (a DialogueState instance, held on CallSession -- see that class's
own __init__) and calls the pure functions here to read and update it.

WHAT COUNTS AS "THE PRIMARY ENTITY" (Acceptance Criterion 1):
Only three entity KINDS are tracked -- "test" (test_name), "doctor"
(doctor_name), "package" (package_name) -- and only for the single-
required-slot intents that ask about exactly one of them (see
_INTENT_PRIMARY_SLOT below). Intents that need TWO entities at once
(insurance_coverage, compare_options) are deliberately excluded: a
follow-up naming only one more piece of information for one of those is
already handled correctly by the EXISTING pending-slot mechanism
(insurance_coverage_slot / compare_options_slot in main.py), which is a
different, already-solved problem ("continue an interrupted question"),
not this one ("start a brand-new question that refers back to an old
answer"). book_appointment/report_status/report_send/billing_balance/
doctors_by_department/clinic_info/smalltalk/unclear/out_of_scope are
excluded for the same reason, or because they have no single entity slot
this module's notion of "primary entity" applies to at all.

WHAT COUNTS AS AMBIGUOUS (Acceptance Criterion 2):
A tracked slot becomes ambiguous the moment a single turn is discovered
to have discussed TWO OR MORE DIFFERENT entities of the SAME kind (see
mark_ambiguous()'s own docstring for the call sites: compare_options
naming two tests, or two packages, in one turn; a multi-intent turn
asking about two DIFFERENT tests in the same breath). A bare pronoun the
very next turn cannot honestly pick one over the other, so
resolve_follow_up() below refuses to guess and instead tells main.py to
ask which one the caller meant -- never silently picks the first, last,
or most recent.

An EMPTY/never-set slot is NOT the same thing as ambiguous, and this
module deliberately does not treat it as one: resolve_follow_up() simply
returns "nothing to backfill" for a blank slot, which falls through to
main.py's EXISTING missing_slot_prompt() re-ask -- that re-ask already IS
"the agent asks for clarification rather than assuming" (the other half
of Acceptance Criterion 2), so there was no need to invent a second,
different way of asking for the same thing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Which entity KIND (used to name the DialogueState attribute
# "active_<kind>") each trackable slot key belongs to. Deliberately only
# the three kinds Acceptance Criterion 1 names/implies ("last discussed
# test_name, doctor_name" -- package_name added by the same reasoning,
# since health_package is otherwise a same-shape single-entity intent).
SLOT_TO_KIND = {
    "test_name": "test",
    "doctor_name": "doctor",
    "package_name": "package",
}

# Which slot key is the "primary entity" a follow-up for this intent
# should resolve against. Every intent NOT listed here is one this
# module deliberately stays out of -- see module docstring's "WHAT COUNTS
# AS THE PRIMARY ENTITY" section for exactly why each excluded intent is
# excluded (insurance_coverage/compare_options: two-slot intents already
# served by their own pending mechanism; book_appointment/report_status/
# report_send/billing_balance/doctors_by_department/clinic_info/
# smalltalk/unclear/out_of_scope: no single anaphora-eligible entity slot
# at all).
_INTENT_PRIMARY_SLOT = {
    "test_rate": "test_name",
    "test_sample": "test_name",
    "test_duration": "test_name",
    "test_preparation": "test_name",
    "walkin_eligibility": "test_name",
    "prescription_requirements": "test_name",
    "doctor_availability": "doctor_name",
    "doctor_schedule": "doctor_name",
    "health_package": "package_name",
}


def primary_slot_for_intent(intent: str) -> str | None:
    """-> the one slot key a follow-up for `intent` resolves against, or
    None for every intent this module does not apply to at all (see
    _INTENT_PRIMARY_SLOT's own comment)."""
    return _INTENT_PRIMARY_SLOT.get(intent)


def kind_for_slot(slot_key: str) -> str | None:
    """-> the DialogueState attribute suffix ("test"/"doctor"/"package")
    tracking `slot_key`, or None if this module does not track it."""
    return SLOT_TO_KIND.get(slot_key)


@dataclass
class EntitySlot:
    """One tracked entity kind's current state for one call.

    `names` holds exactly ONE name when a single, definite entity of this
    kind was last discussed (the ordinary case); TWO OR MORE when the
    last turn to touch this kind mentioned more than one DIFFERENT entity
    of it in the same breath (see mark_ambiguous()) -- deliberately never
    collapsed back down to "just pick one," since doing so would silently
    guess exactly what Acceptance Criterion 2 forbids. Empty (`()`) means
    nothing of this kind has been discussed yet, or nothing SUCCESSFULLY
    resolved (see DialogueState.mark()'s own docstring on why a not-found
    lookup never reaches here at all).
    """
    names: tuple[str, ...] = ()

    @property
    def is_set(self) -> bool:
        return len(self.names) > 0

    @property
    def is_ambiguous(self) -> bool:
        return len(self.names) > 1

    @property
    def primary(self) -> str | None:
        """The single name to backfill with -- None whenever there isn't
        exactly one (blank OR ambiguous), so a caller of this property
        can never accidentally treat "the first of several" as "the
        only one" by forgetting to check is_ambiguous first."""
        return self.names[0] if len(self.names) == 1 else None


@dataclass
class DialogueState:
    """Held on CallSession for the lifetime of one call (see main.py's
    CallSession.__init__) -- entirely independent of session.pending,
    which tracks a DIFFERENT, already-in-progress flow waiting on one
    specific missing field. This tracks what has already been fully
    asked and answered, for turns that start a BRAND NEW question but
    refer back to an earlier answer with a pronoun instead of a name.
    """
    active_test: EntitySlot = field(default_factory=EntitySlot)
    active_doctor: EntitySlot = field(default_factory=EntitySlot)
    active_package: EntitySlot = field(default_factory=EntitySlot)

    def slot_for(self, kind: str) -> EntitySlot:
        return getattr(self, f"active_{kind}")

    def mark(self, kind: str, name: str) -> None:
        """Records a single, definite entity of `kind` as the one most
        recently discussed, REPLACING whatever was tracked before it --
        including collapsing a previously-ambiguous state back down to
        one name, since a caller who just got asked "which one did you
        mean?" and answered has, by definition, resolved the ambiguity
        for anything that follows.

        Callers (main.py) are expected to invoke this ONLY with a name a
        real lookup actually confirmed exists (`result.get("found")` was
        true) -- a not-found lookup means nothing NEW was confirmed to
        exist this turn, so main.py deliberately does not call this for
        one, leaving whatever was already tracked (if anything) as the
        still-valid thing to backfill against next turn. Never call this
        twice in the same turn for two DIFFERENT names of the same kind
        -- the second call would silently stomp the first, hiding a real
        ambiguity; use mark_ambiguous() for that case instead.
        """
        setattr(self, f"active_{kind}", EntitySlot(names=(name,)))

    def mark_ambiguous(self, kind: str, names: list[str]) -> None:
        """Records that TWO OR MORE DIFFERENT entities of `kind` were
        just discussed together in one turn (compare_options naming two
        tests or two packages; a multi-intent turn asking about two
        different tests in one breath) -- deduplicated, order preserved.
        A bare pronoun in the caller's NEXT turn cannot honestly resolve
        to either one, so this deliberately leaves the slot unable to
        auto-backfill (see EntitySlot.primary) rather than picking
        arbitrarily; resolve_follow_up() below turns this into a
        clarifying question instead. A single-name list collapses to the
        same thing mark() would produce -- callers may use either when
        there is exactly one name, but should prefer mark() there for
        clarity of intent.
        """
        seen: list[str] = []
        for n in names:
            if n and n not in seen:
                seen.append(n)
        setattr(self, f"active_{kind}", EntitySlot(names=tuple(seen)))


def resolve_follow_up(state: DialogueState, intent: str, slots: dict) -> tuple[dict, str | None]:
    """Called by main.py's dispatch immediately after intent extraction,
    before any of the existing per-intent slot-checking logic runs.

    Returns (possibly-backfilled slots, ambiguous_kind):
      - `ambiguous_kind` is one of "test"/"doctor"/"package" when this
        intent's primary slot was left null by the extractor AND the
        tracked state for that kind is currently ambiguous (2+
        candidates) -- main.py must NOT run the intent at all in this
        case; it should speak a clarifying question instead (see
        module docstring's "WHAT COUNTS AS AMBIGUOUS").
      - `ambiguous_kind` is None in every other case, INCLUDING "nothing
        tracked yet" -- see module docstring for why a blank slot is
        deliberately NOT treated the same as an ambiguous one.

    Never mutates `slots` in place -- always returns either the exact
    same dict reference (nothing to do) or a new one with the single
    backfilled key added, the same immutable-input discipline every
    other pure function in this codebase (report_flow.py, compare_flow.py)
    already follows.
    """
    slot_key = _INTENT_PRIMARY_SLOT.get(intent)
    if slot_key is None:
        return slots, None
    if slots.get(slot_key):
        return slots, None  # caller named it fresh this turn -- never override.

    kind = SLOT_TO_KIND[slot_key]
    tracked = state.slot_for(kind)
    if tracked.is_ambiguous:
        return slots, kind
    if not tracked.is_set:
        return slots, None  # blank memory -- fall through to the ordinary ask.

    backfilled = dict(slots)
    backfilled[slot_key] = tracked.primary
    return backfilled, None
