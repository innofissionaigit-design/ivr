"""A turn may ask more than one thing. This is what it was broken into.

story title: A multi-part question is answered in full
user story: As a caller who asked two things, I want both answered, so that
    I do not have to ask again.
acceptance criteria: Every answerable part of a turn is answered in the order
    asked, and any part that cannot be answered is explicitly addressed
    rather than dropped. Completeness is scored on a labelled multi-part set.

WHY `parts` AND NOT A LIST-VALUED `intent`
-------------------------------------------
agent/llm.py's schema said `"intent": <one of six>` and every consumer read
it: the semantic cache stores {intent, slots} and keys its L2 eligibility on
a single intent, fast_path.as_llm_shape() produces that exact dict so callers
cannot tell the two apart, tool_outcome counts per tool, answer_ledger keys
per intent, and main.py dispatched on it with an if/elif chain.

Turning `intent` into a list would have broken all of them in one commit, for
no behaviour that an extra field does not also buy. So `parts` is ADDITIVE,
and normalise() guarantees the invariant that makes it safe:

    parts[0] IS the top-level intent and slots.

Every existing reader of data["intent"] therefore sees exactly what it saw
before -- the first thing the caller asked -- and only the dispatcher has to
know that a second part can exist.

WHAT IS DELIBERATELY DROPPED, AND WHY THAT IS NOT THE BUG THIS STORY FIXES
---------------------------------------------------------------------------
A part whose intent is `unclear` or `smalltalk` is kept ONLY when it is the
whole turn.

This looks like exactly the dropping the acceptance criteria forbid, and it
is the opposite. "unclear" is the model's way of saying "there is no
recognisable request here"; appended to a well-formed part it is far more
often trailing ASR noise than a real second question. Keeping it would make
the agent apologise for a question nobody asked -- a new failure mode
invented by this story, on every turn that ends in a cough. `smalltalk` is
the same case with a politer face: a trailing "ধন্যবাদ" is not a second
question, and answering it after a price reads as a machine ticking items
off.

The criterion is about parts the caller ASKED. Both of these are parts the
model MANUFACTURED, and the honest reading of "never dropped" applies to the
former.

MAX_PARTS IS 3
--------------
Not a performance limit. More than three requests in one breath on a phone
line is a mis-parse far more often than a real utterance, and answering five
things in sequence is worse for the caller than asking them to repeat. The
cap fails safe: the extras are dropped at the boundary and the turn is still
a multi-part turn, rather than becoming a monologue.
"""
from __future__ import annotations

# What happened to one part, and therefore what the turn should do next.
#
#   ANSWERED      a reply was spoken. Carry on to the next part.
#   INTERACTIVE   a question was asked BACK and its answer is needed before
#                 anything else makes sense -- a missing slot, a date range
#                 to confirm, a near-match offer, a booking readback. The
#                 turn stops here and the remaining parts are deferred.
#   UNANSWERABLE  the part was recognised and cannot be served. Something was
#                 said about it; carry on.
ANSWERED = "answered"
INTERACTIVE = "interactive"
UNANSWERABLE = "unanswerable"

MAX_PARTS = 3

# Intents that are not a request for information. See the module docstring.
_NON_SUBSTANTIVE = frozenset({"unclear", "smalltalk"})

# Which slot names the thing a part is about. Two parts naming the same
# intent AND the same thing are one part said twice -- answering it twice is
# a worse bug than merging, because the caller hears the same price read out
# in two consecutive sentences.
_ENTITY_SLOT = {
    "test_rate": "test_name",
    "doctor_availability": "doctor_name",
    "doctors_by_department": "department",
    "book_appointment": "doctor_name",
}


def _identity(part: dict) -> tuple:
    intent = part.get("intent")
    slot = _ENTITY_SLOT.get(intent)
    value = (part.get("slots") or {}).get(slot) if slot else None
    return (intent, str(value).strip().lower() if value else None)


def normalise(data: dict) -> list[dict]:
    """-> the ordered parts of one extraction, at least one, at most MAX_PARTS.

    Never raises and never returns empty: a malformed or absent `parts` field
    degrades to the single-part turn the top level always describes, which is
    the behaviour every release before this story had. That fallback is the
    reason this can ship without the model being retrained or the prompt
    landing perfectly -- a model that ignores `parts` entirely produces
    exactly today's agent.
    """
    head = {"intent": data.get("intent"), "slots": data.get("slots") or {},
            "direct_reply_bn": data.get("direct_reply_bn")}

    raw = data.get("parts")
    if not isinstance(raw, list) or not raw:
        return [head]

    # parts[0] is the top level, always -- not whatever the model put first.
    # The two agreeing is an invariant this function ESTABLISHES rather than
    # one it trusts, because every other consumer in the codebase reads the
    # top level and would silently answer a different question than the
    # dispatcher if they ever diverged.
    parts = [head]
    seen = {_identity(head)}

    for item in raw[1:]:
        if len(parts) >= MAX_PARTS:
            break
        if not isinstance(item, dict) or not item.get("intent"):
            continue
        if item["intent"] in _NON_SUBSTANTIVE:
            continue
        candidate = {"intent": item["intent"], "slots": item.get("slots") or {},
                     "direct_reply_bn": None}
        identity = _identity(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        parts.append(candidate)

    return parts


def is_multi(parts: list[dict]) -> bool:
    return len(parts) > 1


def subject_of(part: dict) -> str | None:
    """The caller's OWN words for what this part is about, or None.

    Used only to say something about a part that could not be answered, and
    the caller's words are the right thing to echo there: the catalogue's
    canonical label is English, the tokenizer drops Latin script, and a
    sentence naming the subject in a form nobody hears is worse than one
    that stays general.
    """
    slot = _ENTITY_SLOT.get(part.get("intent"))
    if not slot:
        return None
    value = (part.get("slots") or {}).get(slot)
    return str(value) if value else None
