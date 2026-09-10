"""Composes the spoken reply from TOOL DATA, never from the LLM's own
words, for any intent where a fact (a price, a date, a confirmation ID) is
at stake.

This is the same discipline voicerx/gate.py already applies to drug names
("the SLM proposes, the gazetteer decides") ported to this domain: the LLM
may decide WHAT the caller wants and WHICH slots it heard, but the actual
number in the caller's ear always comes from the clinic-api response,
substituted into a fixed template. The model never gets a chance to
misremember or round a price it was merely shown a moment ago.

Only "smalltalk" skips this file entirely and uses the LLM's own
direct_reply_bn -- there is no fact to get wrong in "নমস্কার" or "ধন্যবাদ".

WHERE THE WORDS WENT
--------------------
The sentences themselves now live in agent/i18n.py, one entry per
language. What stayed here is every DECISION: which branch of a result the
caller is in, whether a written confirmation may be promised, which of
three honest "sorry" answers applies. That logic is identical in Bengali,
Hindi and English, and interleaving three translations into it would hide
the decisions inside the prose.

EVERY PUBLIC FUNCTION TAKES `lang` LAST, DEFAULTED TO None
---------------------------------------------------------
None resolves to the pod's default language, which is Bengali. So every
existing call site -- main.py, main_pcm.py, the test suite -- keeps
working unchanged and keeps producing byte-identical Bengali. Multilingual
support is opt-in per call, not a rewrite of the callers.

NO FLOW MAY DEAD-END ON A SMARTPHONE
------------------------------------
Two of the functions below exist solely to serve that story:
payment_reply() and report_collection_reply(). Both answer with a path a
caller holding a feature phone can finish -- the counter, named first
rather than offered as a consolation. tests/test_no_smartphone.py asserts
that nothing any function here can emit contains a link, a QR instruction
or an app name.
"""
from __future__ import annotations

from agent import language as lang_mod
from agent.i18n import t, available_languages_phrase


def _lang(lang: str | None) -> str:
    """One place that turns "whatever the caller passed" into a language
    this pod can actually serve. None -> the pod default (Bengali)."""
    return lang_mod.resolve(lang)


def _spoken_test_name(slots: dict, result: dict, lang: str) -> str:
    """What the caller HEARS as the test's name.

    Order matters. The API's `test_name` is the catalogue's English label
    ("Uric Acid") and the Bengali TTS tokenizer drops Latin script
    outright, so putting it in a spoken sentence removes the name from the
    reply entirely -- the caller hears a price attached to nothing. Prefer
    the seeded Bengali alias; failing that, echo the caller's own words
    back, which is what a person at the counter would do anyway.

    In English the ordering flips: the catalogue label IS the right spoken
    form and the Bengali alias would be unreadable to an English TTS voice.
    """
    if lang == lang_mod.EN:
        return (result.get("test_name")
                or slots.get("test_name")
                or result.get("test_name_bn")
                or t(lang, "word.test"))
    return (result.get("test_name_bn")
            or slots.get("test_name")
            or result.get("test_name")
            or t(lang, "word.test"))


def _spoken_doctor_name(slots: dict, result: dict, lang: str) -> str:
    """Same problem, same order. Aliases are seeded as surnames ("সেন"),
    so this adds the honorific the English label already carried."""
    if lang == lang_mod.EN:
        name = result.get("doctor_name") or slots.get("doctor_name")
        if name:
            return name
        alias = result.get("doctor_name_bn")
        return t(lang, "honorific.doctor", name=alias) if alias else t(lang, "word.doctor")

    alias = result.get("doctor_name_bn")
    if alias:
        return t(lang, "honorific.doctor", name=alias)
    return slots.get("doctor_name") or result.get("doctor_name") or t(lang, "word.doctor")


def missing_slot_prompt(intent: str, missing: str, lang: str | None = None) -> str:
    code = _lang(lang)
    key = f"ask.{intent}.{missing}"
    prompt = t(code, key)
    # t() returns the key itself when nothing is registered for it, which
    # is the signal that this (intent, slot) pair has no prompt -- fall
    # back to the generic clarification rather than speaking a key.
    return t(code, "ask.fallback") if prompt == key else prompt


def test_rate_reply(slots: dict, result: dict, lang: str | None = None) -> str:
    code = _lang(lang)
    if not result.get("found"):
        suggestions = result.get("did_you_mean") or []
        if suggestions:
            return t(code, "test.not_found_suggest",
                     query=slots.get("test_name"), suggestions=", ".join(suggestions))
        return t(code, "test.not_found", query=slots.get("test_name"))

    rate = result["rate_inr"]
    name = _spoken_test_name(slots, result, code)
    sample = result.get("sample_type")
    hours = result.get("report_time_hours")

    # Avoid saying "test" twice when the spoken name already carries it.
    word_test = t(code, "word.test")
    key = "test.rate_bare" if word_test in name else "test.rate"
    reply = t(code, key, name=name, rate=rate)

    if sample:
        reply += t(code, "test.sample", sample=sample)
    if hours:
        reply += t(code, "test.report_hours", hours=hours)
    return reply


def doctor_availability_reply(slots: dict, result: dict, lang: str | None = None) -> str:
    code = _lang(lang)
    if not result.get("found"):
        return t(code, "doctor.not_found", query=slots.get("doctor_name"))

    name = _spoken_doctor_name(slots, result, code)
    if result.get("available"):
        hours = result.get("chamber_hours", "")
        date_txt = (t(code, "doctor.date_on", date=result.get("date"))
                    if result.get("date") else t(code, "doctor.date_today"))
        # Confirms the doctor is in, then keeps the caller moving straight
        # into booking instead of stopping here -- main.py stays listening
        # for the answer to this exact question (see its "date" pending
        # state), so "আজকেই" / "অন্য দিন" both continue the flow.
        return t(code, "doctor.available", date_txt=date_txt, name=name, hours=hours)

    next_date = result.get("next_available_date")
    if next_date:
        return t(code, "doctor.next_date", name=name, next_date=next_date)
    return t(code, "doctor.no_days", name=name)


def _written_confirmation_clause(result: dict, lang: str) -> str:
    """The one sentence that tells a caller a written copy is coming.

    ONLY SPOKEN WHEN A MESSAGE IS ACTUALLY ON ITS WAY. clinic-api reports
    the ledger row's status on every appointment response; "queued" means
    the row is committed and a send has been scheduled, and that is the
    only value that earns this promise.

    "skipped" and "failed" mean nothing will arrive. Promising an SMS then
    is worse than saying nothing: the caller stops noting the number down,
    hangs up satisfied, and finds out at the reception desk.

    THE STORY'S RULE APPLIES HERE TOO. When no message is coming, the
    caller is NOT simply left with a number to memorise -- they are told
    they can be found at reception by name and phone number. That is what
    makes this flow complete without a smartphone AND without a good
    memory, which is the same requirement wearing a different hat.
    """
    status = (result.get("notification") or {}).get("status")
    if status == "queued":
        return t(lang, "booking.written_clause")
    return t(lang, "booking.written_fallback")


def booking_reply(slots: dict, result: dict, lang: str | None = None) -> str:
    code = _lang(lang)
    if result.get("success"):
        return (t(code, "booking.success",
                  doctor=_spoken_doctor_name(slots, result, code),
                  date=result["date"], time=result["time_slot"],
                  cid=result["confirmation_id"])
                # The number is still read out even when a message is
                # going. It costs one sentence of TTS and it is the only
                # thing the caller has if the SMS is delayed or the handset
                # is off.
                + _written_confirmation_clause(result, code))

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return t(code, "booking.slot_taken_alts", alts=", ".join(alts))
        return t(code, "booking.slot_taken_none")
    if reason == "doctor_not_found":
        return t(code, "doctor.not_found", query=slots.get("doctor_name"))
    return t(code, "booking.failed")


def reschedule_reply(slots: dict, result: dict, lang: str | None = None) -> str:
    """Spoken confirmation for a moved appointment.

    States explicitly that the reference number has NOT changed. A caller
    who is still holding the first message needs to hear that the paper in
    their hand is still valid, otherwise the natural assumption is that it
    is not.
    """
    code = _lang(lang)
    if result.get("success"):
        return (t(code, "reschedule.success",
                  doctor=_spoken_doctor_name(slots, result, code),
                  date=result["date"], time=result["time_slot"],
                  cid=result["confirmation_id"])
                + _written_confirmation_clause(result, code))

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return t(code, "reschedule.slot_taken_alts", alts=", ".join(alts))
        return t(code, "reschedule.slot_taken_none")
    if reason == "appointment_not_found":
        return t(code, "reschedule.not_found")
    if reason == "appointment_cancelled":
        return t(code, "reschedule.cancelled")
    if reason == "doctor_not_available_that_day":
        return t(code, "reschedule.no_day")
    return t(code, "reschedule.failed")


def cancel_reply(slots: dict, result: dict, lang: str | None = None) -> str:
    """Spoken confirmation for a cancellation.

    `already_cancelled` is reported as the plain fact rather than as an
    error, because from the caller's side it is the outcome they asked for
    -- the appointment is not going to happen. clinic-api sends no second
    message in that case, so no message is promised here either.
    """
    code = _lang(lang)
    if result.get("success"):
        if result.get("already_cancelled"):
            return t(code, "cancel.already", date=result.get("date", "")).strip()
        return (t(code, "cancel.success",
                  date=result["date"], time=result["time_slot"],
                  cid=result["confirmation_id"])
                + _written_confirmation_clause(result, code))

    if result.get("reason") == "appointment_not_found":
        return t(code, "reschedule.not_found")
    return t(code, "cancel.failed")


def doctors_by_department_reply(slots: dict, result: dict, lang: str | None = None) -> str:
    code = _lang(lang)
    if not result.get("found"):
        return t(code, "department.not_found", query=slots.get("department"))

    department = result.get("department", slots.get("department"))
    doctors = result.get("doctors", [])
    # Present only when the caller (or main.py, defaulting to today) named
    # a date and clinic-api filtered the list to doctors who actually sit
    # that day -- see clinic-api/main.py's doctors_by_department(). Absent
    # means an unfiltered "who's in this department" listing.
    filtered_by_date = bool(result.get("date"))

    if not doctors:
        if filtered_by_date:
            return t(code, "department.none_today", department=department)
        return t(code, "department.none", department=department)

    doctor_names = []
    for doc in doctors:
        name_bn = doc.get("doctor_name_bn")
        if code == lang_mod.EN:
            doctor_names.append(doc.get("name")
                                or (t(code, "honorific.doctor", name=name_bn) if name_bn
                                    else t(code, "word.doctor")))
        elif name_bn:
            doctor_names.append(t(code, "honorific.doctor", name=name_bn))
        else:
            doctor_names.append(doc.get("name", t(code, "word.doctor")))

    conj = t(code, "conj.and")
    if len(doctor_names) == 1:
        names = doctor_names[0]
    elif len(doctor_names) == 2:
        names = doctor_names[0] + conj + doctor_names[1]
    else:
        names = ", ".join(doctor_names[:-1]) + conj + doctor_names[-1]

    # Keeps the caller moving straight into booking instead of stopping
    # after the list -- main.py stays listening for a bare doctor name
    # next and matches it against exactly this list (see its
    # "doctor_choice" pending state), so the caller never has to repeat
    # "আমি অ্যাপয়েন্টমেন্ট করতে চাই" to be understood.
    return (t(code, "department.listing", department=department, names=names)
            + t(code, "department.ask_which"))


# ===========================================================================
# The two flows that exist because of "every flow completes without a
# smartphone".
# ===========================================================================
def payment_reply(slots: dict, result: dict, lang: str | None = None) -> str:
    """How the caller pays -- and it is never through a link.

    Before this existed, a caller who asked "কত টাকা লাগবে, কীভাবে দেব?"
    got a price and nothing about HOW. That is a flow with no completion
    path: the caller knows the number and still does not know what to do
    next. The obvious modern answer -- text them a payment link -- is
    precisely what this story forbids, and it would exclude every caller on
    a feature phone, which on this line is a large share of them.

    So the answer is the counter, stated first and stated as normal.
    `result` is optional context from a test-rate lookup; the answer is
    complete without it, because a caller who has not named a test still
    deserves to know how payment works.
    """
    code = _lang(lang)
    reply = t(code, "payment.how")

    result = result or {}
    if result.get("found") and result.get("rate_inr"):
        reply += t(code, "payment.amount",
                   name=_spoken_test_name(slots, result, code), rate=result["rate_inr"])

    reply += t(code, "payment.no_advance")
    reply += t(code, "payment.counter_only")
    return reply


def report_collection_reply(slots: dict, result: dict, lang: str | None = None) -> str:
    """When the report is ready and how to get it, with no portal involved.

    Three completion paths, in the order a caller can actually use them:

      1. collect a printed copy at the counter, identified by name and
         phone -- deliberately NOT by the reference number, so a caller who
         lost it is not turned away;
      2. ring in and have it read out, for someone who cannot travel;
      3. send somebody else, who needs only the patient's name and number.

    None of the three needs a smartphone, an app, or a link. The report
    hours come from the catalogue when the caller named a test, and the
    answer stays useful when they did not.
    """
    code = _lang(lang)
    result = result or {}

    hours = result.get("report_time_hours")
    reply = (t(code, "report.when", hours=hours) if hours
             else t(code, "report.when_unknown"))

    reply += t(code, "report.collect")
    reply += t(code, "report.phone_readout")
    reply += t(code, "report.someone_else")
    return reply


def counter_fallback(lang: str | None = None, hours: str | None = None) -> str:
    """The universal completion path, for any turn that cannot finish on
    the phone.

    Exists so no branch anywhere ends with the caller holding nothing. A
    "sorry, I can't do that" with no next step is exactly the dead end the
    story names, even when no smartphone was ever mentioned.
    """
    code = _lang(lang)
    reply = t(code, "counter.walk_in")
    if hours:
        reply = t(code, "counter.hours", hours=hours) + " " + reply
    return reply


# ===========================================================================
# Language handling
# ===========================================================================
def language_switch_reply(lang: str | None = None) -> str:
    """Spoken IN THE NEW LANGUAGE, which is the point -- it is the caller's
    proof that the switch actually took."""
    return t(_lang(lang), "language.switched")


def language_unavailable_reply(current_lang: str | None = None) -> str:
    """The caller asked for a language this pod cannot serve.

    Answered in the language they are currently being understood in, and it
    names what IS available. Silently ignoring the request reads as the
    system not having heard them, and they ask again -- burning a turn and
    their patience on a line that will never say yes.
    """
    code = _lang(current_lang)
    return t(code, "language.unavailable", available=available_languages_phrase(code))
