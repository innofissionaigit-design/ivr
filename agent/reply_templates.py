"""Composes the spoken Bengali reply from TOOL DATA, never from the LLM's
own words, for any intent where a fact (a price, a date, a confirmation
ID) is at stake.

This is the same discipline voicerx/gate.py already applies to drug names
("the SLM proposes, the gazetteer decides") ported to this domain: the LLM
may decide WHAT the caller wants and WHICH slots it heard, but the actual
number in the caller's ear always comes from the Spring Boot response,
substituted into a fixed template. The model never gets a chance to
misremember or round a price it was merely shown a moment ago.

Only "smalltalk" skips this file entirely and uses the LLM's own
direct_reply_bn -- there is no fact to get wrong in "নমস্কার" or "ধন্যবাদ".
"""
from __future__ import annotations


def _spoken_test_name(slots: dict, result: dict) -> str:
    """What the caller HEARS as the test's name.

    Order matters. The API's `test_name` is the catalogue's English label
    ("Uric Acid") and the Bengali TTS tokenizer drops Latin script
    outright, so putting it in a spoken sentence removes the name from the
    reply entirely -- the caller hears a price attached to nothing. Prefer
    the seeded Bengali alias; failing that, echo the caller's own words
    back, which is what a person at the counter would do anyway.
    """
    return (result.get("test_name_bn")
            or slots.get("test_name")
            or result.get("test_name")
            or "টেস্ট")


def _spoken_doctor_name(slots: dict, result: dict) -> str:
    """Same problem, same order. Aliases are seeded as surnames ("সেন"),
    so this adds the honorific the English label already carried."""
    alias = result.get("doctor_name_bn")
    if alias:
        return f"ডাঃ {alias}"
    return slots.get("doctor_name") or result.get("doctor_name") or "ডাক্তার"


def missing_slot_prompt(intent: str, missing: str) -> str:
    prompts = {
        ("test_rate", "test_name"): "কোন টেস্টের রেট জানতে চান, একটু বলবেন?",
        ("doctor_availability", "doctor_name"): "কোন ডাক্তারের কথা জিজ্ঞেস করছেন?",
        ("doctors_by_department", "department"): "কোন বিভাগের ডাক্তার খুঁজছেন?",
        ("doctors_by_department", "date"): "কোন দিনের জন্য জানতে চান, একটু বলবেন?",
        ("book_appointment", "doctor_name"): "কোন ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট করতে চান?",
        ("book_appointment", "date"): "আজকের জন্য চান, নাকি অন্য কোনো দিনের জন্য অ্যাপয়েন্টমেন্ট চাই?",
        ("book_appointment", "time_slot"): "কোন সময়ে অ্যাপয়েন্টমেন্ট চাই, একটু বলবেন?",
        ("book_appointment", "patient_name"): "রোগীর নামটা বলবেন?",
        ("book_appointment", "phone"): "একটা ফোন নম্বর দেবেন, যাতে কনফার্মেশন পাঠাতে পারি?",
    }
    return prompts.get((intent, missing), "দুঃখিত, একটু স্পষ্ট করে বলবেন?")


def test_rate_reply(slots: dict, result: dict) -> str:
    if not result.get("found"):
        suggestions = result.get("did_you_mean") or []
        if suggestions:
            return (f"'{slots.get('test_name')}' নামে টেস্ট খুঁজে পাইনি। "
                     f"আপনি কি বলতে চাইছেন: {', '.join(suggestions)}?")
        return f"দুঃখিত, '{slots.get('test_name')}' নামে কোনো টেস্ট আমাদের তালিকায় নেই।"

    rate = result["rate_inr"]
    name = _spoken_test_name(slots, result)
    sample = result.get("sample_type")
    hours = result.get("report_time_hours")
    
    # Check if name already contains "টেস্ট" to avoid duplication
    if "টেস্ট" in name:
        reply = f"{name} রেট {rate} টাকা।"
    else:
        reply = f"{name} টেস্টের রেট {rate} টাকা।"
    
    if sample:
        reply += f" স্যাম্পল: {sample}।"
    if hours:
        reply += f" রিপোর্ট {hours} ঘণ্টার মধ্যে পাবেন।"
    return reply


def doctor_availability_reply(slots: dict, result: dict) -> str:
    if not result.get("found"):
        return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার আমাদের এখানে নেই।"

    name = _spoken_doctor_name(slots, result)
    if result.get("available"):
        hours = result.get("chamber_hours", "")
        date_txt = f" {result.get('date')} তারিখে" if result.get("date") else " আজ"
        # Confirms the doctor is in, then keeps the caller moving straight
        # into booking instead of stopping here -- main.py stays listening
        # for the answer to this exact question (see its "date" pending
        # state), so "আজকেই" / "অন্য দিন" both continue the flow.
        return (f"হ্যাঁ,{date_txt} {name} চেম্বারে থাকবেন। সময়: {hours}। "
                f"আজকের জন্যই অ্যাপয়েন্টমেন্ট করবেন, নাকি অন্য কোনো দিনের জন্য?")

    next_date = result.get("next_available_date")
    if next_date:
        return (f"{name} ওই দিন বসবেন না। পরবর্তী উপলব্ধ দিন: {next_date}। "
                f"ওই দিনের জন্য অ্যাপয়েন্টমেন্ট করতে চান?")
    return f"{name} এখন কোনো নির্দিষ্ট দিন বসছেন না। আমাদের কাউন্টারে খোঁজ নিতে পারেন।"


def _written_confirmation_clause(result: dict) -> str:
    """The one sentence that tells a caller a written copy is coming.

    ONLY SPOKEN WHEN A MESSAGE IS ACTUALLY ON ITS WAY. clinic-api reports
    the ledger row's status on every appointment response; "queued" means
    the row is committed and a send has been scheduled, and that is the
    only value that earns this promise.

    "skipped" (no gateway configured on this pod) and "failed" (rendering
    or registration is broken) both mean nothing will arrive. Promising an
    SMS in those cases is worse than saying nothing at all: the caller
    stops writing the number down, hangs up satisfied, and finds out at the
    reception desk. Silence leaves them with the spoken number, which is
    exactly where they were before this feature existed -- a smaller
    failure, and an honest one.
    """
    status = (result.get("notification") or {}).get("status")
    if status == "queued":
        return " কনফার্মেশনের একটা মেসেজ আপনার ফোনে পাঠানো হচ্ছে, রিসেপশনে ওটা দেখালেই হবে।"
    return ""


def booking_reply(slots: dict, result: dict) -> str:
    if result.get("success"):
        return (f"আপনার অ্যাপয়েন্টমেন্ট কনফার্ম হয়েছে। "
                f"{_spoken_doctor_name(slots, result)}, {result['date']}, সময় {result['time_slot']}। "
                f"কনফার্মেশন নম্বর: {result['confirmation_id']}।"
                # The number is still read out even when a message is
                # going. It costs one sentence of TTS and it is the only
                # thing the caller has if the SMS is delayed or the handset
                # is off -- see _written_confirmation_clause() on why the
                # promise, not the number, is the conditional part.
                f"{_written_confirmation_clause(result)}")

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return f"ওই সময়টা বুক হয়ে গেছে। এই সময়গুলো ফাঁকা আছে: {', '.join(alts)}। কোনটা চান?"
        return "ওই সময়টা বুক হয়ে গেছে, এবং কাছাকাছি কোনো সময় ফাঁকা নেই।"
    if reason == "doctor_not_found":
        return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার খুঁজে পেলাম না।"
    return "দুঃখিত, অ্যাপয়েন্টমেন্ট বুক করা গেল না। একটু পরে আবার চেষ্টা করুন, অথবা কাউন্টারে যোগাযোগ করুন।"


def reschedule_reply(slots: dict, result: dict) -> str:
    """Spoken confirmation for a moved appointment.

    States explicitly that the reference number has NOT changed. A caller
    who is still holding the first message needs to hear that the paper in
    their hand is still valid, otherwise the natural assumption is that it
    is not.
    """
    if result.get("success"):
        return (f"আপনার অ্যাপয়েন্টমেন্ট বদলে দেওয়া হয়েছে। "
                f"{_spoken_doctor_name(slots, result)}, {result['date']}, সময় {result['time_slot']}। "
                f"কনফার্মেশন নম্বর একই থাকছে: {result['confirmation_id']}।"
                f"{_written_confirmation_clause(result)}")

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return f"ওই সময়টা ফাঁকা নেই। এই সময়গুলো আছে: {', '.join(alts)}। কোনটা চান?"
        return "ওই সময়টা ফাঁকা নেই, এবং কাছাকাছি কোনো সময়ও নেই।"
    if reason == "appointment_not_found":
        return "ওই কনফার্মেশন নম্বরে কোনো অ্যাপয়েন্টমেন্ট খুঁজে পেলাম না। নম্বরটা আরেকবার বলবেন?"
    if reason == "appointment_cancelled":
        return "ওই অ্যাপয়েন্টমেন্টটা আগেই বাতিল হয়ে গেছে। নতুন করে বুক করে দেব?"
    if reason == "doctor_not_available_that_day":
        return "ডাক্তার ওই দিন বসছেন না। অন্য কোনো দিন দেখব?"
    return "দুঃখিত, অ্যাপয়েন্টমেন্টটা বদলানো গেল না। কাউন্টারে যোগাযোগ করুন, দয়া করে।"


def cancel_reply(slots: dict, result: dict) -> str:
    """Spoken confirmation for a cancellation.

    `already_cancelled` is reported as the plain fact rather than as an
    error, because from the caller's side it is the outcome they asked for
    -- the appointment is not going to happen. clinic-api sends no second
    message in that case, so no message is promised here either.
    """
    if result.get("success"):
        if result.get("already_cancelled"):
            return (f"ওই অ্যাপয়েন্টমেন্টটা আগেই বাতিল করা হয়েছে। "
                    f"{result.get('date', '')} তারিখের কিছু আর বুক করা নেই।").strip()
        return (f"আপনার অ্যাপয়েন্টমেন্ট বাতিল করা হয়েছে। "
                f"{result['date']}, সময় {result['time_slot']}। "
                f"কনফার্মেশন নম্বর: {result['confirmation_id']}।"
                f"{_written_confirmation_clause(result)}")

    if result.get("reason") == "appointment_not_found":
        return "ওই কনফার্মেশন নম্বরে কোনো অ্যাপয়েন্টমেন্ট খুঁজে পেলাম না। নম্বরটা আরেকবার বলবেন?"
    return "দুঃখিত, অ্যাপয়েন্টমেন্টটা বাতিল করা গেল না। কাউন্টারে যোগাযোগ করুন, দয়া করে।"


def doctors_by_department_reply(slots: dict, result: dict) -> str:
    if not result.get("found"):
        return f"দুঃখিত, '{slots.get('department')}' নামে কোনো বিভাগ আমাদের এখানে নেই।"

    department = result.get("department", slots.get("department"))
    doctors = result.get("doctors", [])
    # Present only when the caller (or main.py, defaulting to today) named
    # a date and clinic-api filtered the list to doctors who actually sit
    # that day -- see clinic-api/main.py's doctors_by_department(). Absent
    # means an unfiltered "who's in this department" listing.
    filtered_by_date = bool(result.get("date"))

    if not doctors:
        if filtered_by_date:
            return (f"দুঃখিত, {department} বিভাগে আজ কোনো ডাক্তার নেই। "
                     f"অন্য কোনো দিনের কথা জিজ্ঞেস করতে পারেন।")
        return f"{department} বিভাগে কোনো ডাক্তার নেই।"

    doctor_names = []
    for doc in doctors:
        name_bn = doc.get("doctor_name_bn")
        if name_bn:
            doctor_names.append(f"ডাঃ {name_bn}")
        else:
            doctor_names.append(doc.get("name", "ডাক্তার"))

    if len(doctor_names) == 1:
        listing = f"{department} বিভাগে {doctor_names[0]} আছেন।"
    elif len(doctor_names) == 2:
        listing = f"{department} বিভাগে {doctor_names[0]} এবং {doctor_names[1]} আছেন।"
    else:
        all_names = ", ".join(doctor_names[:-1]) + " এবং " + doctor_names[-1]
        listing = f"{department} বিভাগে {all_names} আছেন।"

    # Keeps the caller moving straight into booking instead of stopping
    # after the list -- main.py stays listening for a bare doctor name
    # next and matches it against exactly this list (see its
    # "doctor_choice" pending state), so the caller never has to repeat
    # "আমি অ্যাপয়েন্টমেন্ট করতে চাই" to be understood.
    return listing + " অ্যাপয়েন্টমেন্টের জন্য কোন ডাক্তারের নাম বলবেন?"
