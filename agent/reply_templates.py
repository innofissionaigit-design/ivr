"""Composes the spoken reply from TOOL DATA, never from the LLM's
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

LANGUAGE SUPPORT:
- Primary: Bengali (বাংলা) - default for TTS synthesis
- Secondary: English - for English-speaking callers
- Tertiary: Hinglish (Hindi-English mix) - for mixed-language contexts
- Also supported by the booking-confirmation prompts (below): Banglish
  (Bengali-English mix) -- Kolkata callers code-switch Bengali with
  English at least as often as Hindi with English, and "hinglish" here
  is Hindi vocabulary, not Bengali, so a Bengali-English code-switcher
  needs its own branch rather than being folded into "hinglish".
All templates preserve exact values regardless of language.
"""
from __future__ import annotations

import re

from agent.bn_normalize import detect_language

# Fallback word for "the doctor" when no name is available at all, per
# language -- see _spoken_doctor_name() below.
_DOCTOR_FALLBACK = {
    "bengali": "ডাক্তার",
    "english": "the doctor",
    "hinglish": "doctor",
    "banglish": "doctor",
}


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


def _spoken_doctor_name(slots: dict, result: dict, language: str = "bengali") -> str:
    """Same problem, same order. Aliases are seeded as surnames ("সেন"),
    so this adds the honorific the English label already carried.

    `language` was added for the booking-confirmation readback
    (booking_confirmation_prompt, below): previously this function always
    returned the Bengali "ডাঃ" honorific regardless of which language the
    surrounding sentence was in, because its only two callers
    (doctor_availability_reply, booking_reply) never passed a language
    through even though they already accept one. That glued a Bengali
    prefix into the middle of an English or Hinglish sentence whenever
    those branches were used. Passing `language="bengali"` (the default,
    and the only value either of those two callers has ever actually
    used in production) reproduces the exact previous output -- this is
    additive, not a behaviour change for existing Bengali call sites.
    """
    alias = result.get("doctor_name_bn") or slots.get("doctor_name_bn")
    if language == "bengali":
        if alias:
            return f"ডাঃ {alias}"
        return slots.get("doctor_name") or result.get("doctor_name") or _DOCTOR_FALLBACK["bengali"]

    # english / hinglish / banglish: "ডাঃ" does not belong in a sentence
    # that is otherwise English or transliterated -- "Dr." is the
    # honorific a bilingual caller actually expects here. The Bengali
    # alias (pure Bengali script, e.g. "সেন") is deliberately NOT used in
    # this branch even when available -- it would be exactly as
    # unreadable/unspeakable in an English or transliterated sentence as
    # the "ডাঃ" prefix would be, just in the other direction.
    name = slots.get("doctor_name") or result.get("doctor_name")
    if not name:
        return _DOCTOR_FALLBACK.get(language, _DOCTOR_FALLBACK["english"])
    # clinic-api/seed.py seeds every doctor's canonical name WITH its own
    # "Dr." already ("Dr. A. Sen") -- prepending another one here would
    # speak "Dr. Dr. A. Sen". Only add the honorific when the name
    # doesn't already carry one.
    if re.match(r"^dr\.?\s", name.strip(), flags=re.IGNORECASE):
        return name
    return f"Dr. {name}"


def missing_slot_prompt(intent: str, missing: str, language: str = "bengali") -> str:
    """Generate a prompt for missing slot information in the specified language."""
    if language == "english":
        prompts = {
            ("test_rate", "test_name"): "Which test rate would you like to know?",
            ("doctor_availability", "doctor_name"): "Which doctor are you asking about?",
            ("doctors_by_department", "department"): "Which department are you looking for?",
            ("doctors_by_department", "date"): "Which date would you like to know about?",
            ("book_appointment", "doctor_name"): "Which doctor would you like to book with?",
            ("book_appointment", "date"): "Would you like today or another day?",
            ("book_appointment", "time_slot"): "What time would you prefer?",
            ("book_appointment", "patient_name"): "Could you tell me the patient's name?",
            ("book_appointment", "phone"): "Could you provide a phone number for confirmation?",
        }
        return prompts.get((intent, missing), "Sorry, could you please clarify?")
    elif language == "hinglish":
        prompts = {
            ("test_rate", "test_name"): "Kaunse test ka rate jaanna chahte ho?",
            ("doctor_availability", "doctor_name"): "Kaunse doctor ke baare mein pooch rahe ho?",
            ("doctors_by_department", "department"): "Kaunse department mein doctor dhundh rahe ho?",
            ("doctors_by_department", "date"): "Kis din ke liye jaanna chahte ho?",
            ("book_appointment", "doctor_name"): "Kaunse doctor ke saath appointment karna chahte ho?",
            ("book_appointment", "date"): "Aaj ke liye chaahiye ya kisi aur din ke liye?",
            ("book_appointment", "time_slot"): "Kya time prefer karte ho?",
            ("book_appointment", "patient_name"): "Patient ka naam bata sakte ho?",
            ("book_appointment", "phone"): "Confirmation ke liye phone number de sakte ho?",
        }
        return prompts.get((intent, missing), "Sorry, thoda clear kar sakte ho?")
    else:  # bengali (default)
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


def test_rate_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    if not result.get("found"):
        suggestions = result.get("did_you_mean") or []
        if language == "english":
            if suggestions:
                return (f"I couldn't find a test named '{slots.get('test_name')}'. "
                         f"Did you mean: {', '.join(suggestions)}?")
            return f"Sorry, we don't have a test named '{slots.get('test_name')}'."
        elif language == "hinglish":
            if suggestions:
                return (f"'{slots.get('test_name')}' naam ka test nahi mila. "
                         f"Kya aap kehna chahte the: {', '.join(suggestions)}?")
            return f"Sorry, '{slots.get('test_name')}' naam ka test hamari list mein nahi hai."
        else:  # bengali
            if suggestions:
                return (f"'{slots.get('test_name')}' নামে টেস্ট খুঁজে পাইনি। "
                         f"আপনি কি বলতে চাইছেন: {', '.join(suggestions)}?")
            return f"দুঃখিত, '{slots.get('test_name')}' নামে কোনো টেস্ট আমাদের তালিকায় নেই।"

    rate = result["rate_inr"]
    name = _spoken_test_name(slots, result)
    sample = result.get("sample_type")
    hours = result.get("report_time_hours")
    
    # Preserve exact rate value in all languages
    if language == "english":
        if "test" in name.lower():
            reply = f"{name} rate is {rate} rupees."
        else:
            reply = f"{name} test rate is {rate} rupees."
        if sample:
            reply += f" Sample: {sample}."
        if hours:
            reply += f" Report available in {hours} hours."
    elif language == "hinglish":
        if "test" in name.lower():
            reply = f"{name} ka rate {rate} rupaye hai."
        else:
            reply = f"{name} test ka rate {rate} rupaye hai."
        if sample:
            reply += f" Sample: {sample}."
        if hours:
            reply += f" Report {hours} ghante mein milega."
    else:  # bengali
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


def doctor_availability_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    if not result.get("found"):
        if language == "english":
            return f"Sorry, we don't have a doctor named '{slots.get('doctor_name')}'."
        elif language == "hinglish":
            return f"Sorry, '{slots.get('doctor_name')}' naam ka doctor yahan nahi hai."
        else:  # bengali
            return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার আমাদের এখানে নেই।"

    name = _spoken_doctor_name(slots, result, language=language)
    if result.get("available"):
        hours = result.get("chamber_hours", "")
        date_txt = f" {result.get('date')}" if result.get("date") else " today"
        
        if language == "english":
            return (f"Yes,{date_txt} {name} will be in chamber. Time: {hours}. "
                    f"Would you like to book for today or another day?")
        elif language == "hinglish":
            return (f"Haan,{date_txt} {name} chamber mein honge. Time: {hours}. "
                    f"Aaj ke liye appointment karna chahte ho ya kisi aur din ke liye?")
        else:  # bengali
            date_txt_bn = f" {result.get('date')} তারিখে" if result.get("date") else " আজ"
            return (f"হ্যাঁ,{date_txt_bn} {name} চেম্বারে থাকবেন। সময়: {hours}। "
                    f"আজকের জন্যই অ্যাপয়েন্টমেন্ট করবেন, নাকি অন্য কোনো দিনের জন্য?")

    next_date = result.get("next_available_date")
    if next_date:
        if language == "english":
            return (f"{name} won't be available that day. Next available date: {next_date}. "
                    f"Would you like to book for that day?")
        elif language == "hinglish":
            return (f"{name} us din nahi honge. Agla available date: {next_date}. "
                    f"Us din ke liye appointment karna chahte ho?")
        else:  # bengali
            return (f"{name} ওই দিন বসবেন না। পরবর্তী উপলব্ধ দিন: {next_date}। "
                    f"ওই দিনের জন্য অ্যাপয়েন্টমেন্ট করতে চান?")
    
    if language == "english":
        return f"{name} doesn't have a fixed schedule right now. Please check at our counter."
    elif language == "hinglish":
        return f"{name} abhi koi fixed date nahi hai. Hamare counter mein check kar sakte ho."
    else:  # bengali
        return f"{name} এখন কোনো নির্দিষ্ট দিন বসছেন না। আমাদের কাউন্টারে খোঁজ নিতে পারেন।"


def booking_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    if result.get("success"):
        # Preserve exact values in all languages
        doctor = _spoken_doctor_name(slots, result, language=language)
        date = result['date']
        time_slot = result['time_slot']
        confirmation_id = result['confirmation_id']
        
        if language == "english":
            return (f"Your appointment is confirmed. "
                    f"{doctor}, {date}, time {time_slot}. "
                    f"Confirmation number: {confirmation_id}.")
        elif language == "hinglish":
            return (f"Aapka appointment confirm ho gaya. "
                    f"{doctor}, {date}, time {time_slot}. "
                    f"Confirmation number: {confirmation_id}.")
        else:  # bengali
            return (f"আপনার অ্যাপয়েন্টমেন্ট কনফার্ম হয়েছে। "
                    f"{doctor}, {date}, সময় {time_slot}। "
                    f"কনফার্মেশন নম্বর: {confirmation_id}।")

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            if language == "english":
                return f"That time is already booked. These times are available: {', '.join(alts)}. Which would you prefer?"
            elif language == "hinglish":
                return f"Wo time already book ho gaya. Ye times available hain: {', '.join(alts)}. Kaunsa prefer karte ho?"
            else:  # bengali
                return f"ওই সময়টা বুক হয়ে গেছে। এই সময়গুলো ফাঁকা আছে: {', '.join(alts)}। কোনটা চান?"
        if language == "english":
            return "That time is already booked, and there are no nearby available times."
        elif language == "hinglish":
            return "Wo time already book ho gaya, aur paas mein koi available time nahi hai."
        else:  # bengali
            return "ওই সময়টা বুক হয়ে গেছে, এবং কাছাকাছি কোনো সময় ফাঁকা নেই।"
    if reason == "doctor_not_found":
        if language == "english":
            return f"Sorry, I couldn't find a doctor named '{slots.get('doctor_name')}'."
        elif language == "hinglish":
            return f"Sorry, '{slots.get('doctor_name')}' naam ka doctor nahi mila."
        else:  # bengali
            return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার খুঁজে পেলাম না।"
    
    if language == "english":
        return "Sorry, couldn't book the appointment. Please try again later, or contact our counter."
    elif language == "hinglish":
        return "Sorry, appointment book nahi ho paya. Thodi der baad phir try karein, ya hamare counter se contact karein."
    else:  # bengali
        return "দুঃখিত, অ্যাপয়েন্টমেন্ট বুক করা গেল না। একটু পরে আবার চেষ্টা করুন, অথবা কাউন্টারে যোগাযোগ করুন।"


def booking_confirmation_prompt(slots: dict, language: str = "bengali") -> str:
    """Every critical value is read back before it is used (Answer Quality
    and Grounding): this is spoken once all five booking fields are known,
    BEFORE main.py ever calls book_appointment(). A misheard phone digit
    or date gets caught here, not after the write.

    Same passthrough discipline test_reply_templates_fidelity.py already
    locks in for booking_reply(): values are dropped into the sentence
    exactly as slot_parse.py produced them (ISO date, 24h "HH:MM", raw
    phone digits) with no reformatting here. agent/tts.py's synthesize()
    runs bn_normalize.verbalize() on this text before it reaches the
    caller's ear, which is what turns the ISO date/time and the phone
    digits into spoken words digit-faithfully -- nothing in this function
    needs to do that itself.

    `language` covers all four this system speaks: "bengali" (default),
    "english", "hinglish" (Hindi-English), and "banglish" (Bengali-English
    -- Kolkata's own code-switch, distinct from Hindi-English and not the
    same as "hinglish"). The doctor's name goes through
    _spoken_doctor_name() rather than being read straight off `slots`, so
    the honorific matches the sentence's language ("ডাঃ" only in Bengali,
    "Dr." otherwise) instead of a Bengali prefix landing in the middle of
    an English or transliterated sentence.

    KNOWN LIMITATION, inherited from booking_reply()/doctor_availability_
    reply() and not introduced here: `slots` never carries a
    `doctor_name_bn` alias today -- main.py/main_pcm.py only ever store
    the catalogue's plain `doctor_name` in pending["slots"], even on the
    turns where a Bengali alias WAS available (see _continue_pending's
    "doctor_choice" state, which has the alias in `candidates` and drops
    it once matched). _spoken_doctor_name() checks for the alias first
    and uses it the moment some future call site starts threading it
    through, but until that data-flow gap is closed, the Bengali branch
    below may still speak the doctor's Latin-script catalogue name inside
    an otherwise-Bengali sentence. That gap is pre-existing and shared
    with booking_reply(); closing it needs pending["slots"] to start
    carrying the alias, which is a data-flow change beyond this story.
    """
    doctor = _spoken_doctor_name(slots, {}, language=language)
    date = slots.get("date") or ""
    time_slot = slots.get("time_slot") or ""
    patient_name = slots.get("patient_name") or ""
    phone = slots.get("phone") or ""

    if language == "english":
        return (f"Let me confirm before I book this. "
                f"{doctor}, {date}, time {time_slot}, patient {patient_name}, "
                f"phone number {phone}. Is that all correct?")
    elif language == "hinglish":
        return (f"Book karne se pehle confirm kar lete hain. "
                f"{doctor}, {date}, time {time_slot}, patient {patient_name}, "
                f"phone number {phone}. Sab sahi hai?")
    elif language == "banglish":
        return (f"Book korar age ekbar confirm kore nin. "
                f"{doctor}, {date}, time {time_slot}, patient-er naam {patient_name}, "
                f"phone number {phone}. Sob thik ache to?")
    else:  # bengali
        return (f"বুক করার আগে একবার শুনে নিন। "
                f"{doctor}, {date}, সময় {time_slot}, রোগীর নাম {patient_name}, "
                f"ফোন নম্বর {phone}। সব ঠিক আছে তো?")


def booking_correction_prompt(language: str = "bengali") -> str:
    """Asked when the caller rejects booking_confirmation_prompt() above.
    Acceptance criterion: "a rejection opens a correction path rather than
    repeating the prompt" -- this is a DIFFERENT question (which field is
    wrong?), never a re-read of the same five values, and it hands the
    caller a specific menu instead of restarting the whole booking flow.

    Same four languages as booking_confirmation_prompt() above.
    """
    if language == "english":
        return "No problem -- which one should I fix: doctor, date, time, name, or phone number?"
    elif language == "hinglish":
        return "Koi baat nahi -- kya theek karna hai: doctor, date, time, naam, ya phone number?"
    elif language == "banglish":
        return "Kono problem nei -- ki thik korte hobe: doctor, date, time, naam, na ki phone number?"
    else:  # bengali
        return "ঠিক আছে, কোনটা ঠিক করে দেব - ডাক্তার, তারিখ, সময়, নাম, নাকি ফোন নম্বর?"


def doctors_by_department_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    if not result.get("found"):
        if language == "english":
            return f"Sorry, we don't have a department named '{slots.get('department')}'."
        elif language == "hinglish":
            return f"Sorry, '{slots.get('department')}' naam ka department yahan nahi hai."
        else:  # bengali
            return f"দুঃখিত, '{slots.get('department')}' নামে কোনো বিভাগ আমাদের এখানে নেই।"

    department = result.get("department", slots.get("department"))
    doctors = result.get("doctors", [])
    filtered_by_date = bool(result.get("date"))

    if not doctors:
        if filtered_by_date:
            if language == "english":
                return (f"Sorry, there are no doctors in {department} today. "
                         f"You can ask about another day.")
            elif language == "hinglish":
                return (f"Sorry, {department} department mein aaj koi doctor nahi hai. "
                         f"Aur din ke baare mein pooch sakte ho.")
            else:  # bengali
                return (f"দুঃখিত, {department} বিভাগে আজ কোনো ডাক্তার নেই। "
                         f"অন্য কোনো দিনের কথা জিজ্ঞেস করতে পারেন।")
        if language == "english":
            return f"There are no doctors in {department}."
        elif language == "hinglish":
            return f"{department} department mein koi doctor nahi hai."
        else:  # bengali
            return f"{department} বিভাগে কোনো ডাক্তার নেই।"

    doctor_names = []
    for doc in doctors:
        name_bn = doc.get("doctor_name_bn")
        if language == "english":
            doctor_names.append(doc.get("name", "Doctor"))
        elif language == "hinglish":
            doctor_names.append(f"Dr. {name_bn}" if name_bn else doc.get("name", "Doctor"))
        else:  # bengali
            if name_bn:
                doctor_names.append(f"ডাঃ {name_bn}")
            else:
                doctor_names.append(doc.get("name", "ডাক্তার"))

    if language == "english":
        if len(doctor_names) == 1:
            listing = f"{department} has {doctor_names[0]}."
        elif len(doctor_names) == 2:
            listing = f"{department} has {doctor_names[0]} and {doctor_names[1]}."
        else:
            all_names = ", ".join(doctor_names[:-1]) + ", and " + doctor_names[-1]
            listing = f"{department} has {all_names}."
        return listing + " Which doctor would you like to book with?"
    elif language == "hinglish":
        if len(doctor_names) == 1:
            listing = f"{department} mein {doctor_names[0]} hain."
        elif len(doctor_names) == 2:
            listing = f"{department} mein {doctor_names[0]} aur {doctor_names[1]} hain."
        else:
            all_names = ", ".join(doctor_names[:-1]) + ", aur " + doctor_names[-1]
            listing = f"{department} mein {all_names} hain."
        return listing + " Appointment ke liye kaunse doctor ka naam batayenge?"
    else:  # bengali
        if len(doctor_names) == 1:
            listing = f"{department} বিভাগে {doctor_names[0]} আছেন।"
        elif len(doctor_names) == 2:
            listing = f"{department} বিভাগে {doctor_names[0]} এবং {doctor_names[1]} আছেন।"
        else:
            all_names = ", ".join(doctor_names[:-1]) + " এবং " + doctor_names[-1]
            listing = f"{department} বিভাগে {all_names} আছেন।"
        return listing + " অ্যাপয়েন্টমেন্টের জন্য কোন ডাক্তারের নাম বলবেন?"
