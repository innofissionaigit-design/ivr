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

SPOKEN PUNCTUATION ("Answers sound like a person, not a database row"):
No template here ever emits a field label followed by a colon (e.g.
"Sample: X", "Confirmation number: Y") -- a spoken colon or bracket reads
to a caller as a database field, not a sentence, which is exactly what
this story bans. Every value is folded into a natural clause instead
("You'll need to give a X sample.", "Your confirmation number is Y.").
This is unrelated to the colons inside raw values like a 24h time
("14:30") -- those are consumed by agent/bn_normalize.py's verbalize()
at synthesis time (see tts.py) before anything is actually spoken, so
they were never the problem; only a literal label-colon that survives
verbalize() unchanged is. test_reply_templates_fidelity.py enforces this
across every function and language with an automated check that runs
output through verbalize() and fails the build if a colon or bracket
survives.
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

# Same idea for a test's name -- see _spoken_test_name() below.
_TEST_FALLBACK = {
    "bengali": "টেস্ট",
    "english": "the test",
    "hinglish": "test",
    "banglish": "test",
}


def _spoken_test_name(slots: dict, result: dict, language: str = "bengali") -> str:
    """What the caller HEARS as the test's name.

    `language` was added for "Caller asks the price of a test" (Epic:
    Conversation -- Information and Enquiry), fixing a real, measured bug
    found while verifying that story: this function used to have NO
    language parameter at all and unconditionally preferred the seeded
    Bengali alias, so an English/Hinglish/Banglish caller asking about any
    test that has a Bengali alias (most of them) heard raw Bengali script
    glued into an otherwise-English sentence -- e.g. "সিবিসি test rate is
    400 rupees." This is the exact same bug class _spoken_doctor_name()
    above was already fixed for; this function just never got the same
    treatment. Confirmed by actually running test_rate_reply() against a
    real seeded test with a real alias, not assumed.

    Order matters, per language:
    - bengali (default): the Bengali TTS tokenizer drops Latin script
      outright, so putting the English name in a Bengali sentence removes
      it from the reply entirely -- prefer the seeded Bengali alias;
      failing that, echo the caller's own words back.
    - english / hinglish / banglish: the Bengali alias is exactly as
      unreadable in one of these sentences as "ডাঃ" would be in
      _spoken_doctor_name()'s non-Bengali branch -- never use it here,
      even when available. Use the caller's own words, or the catalogue's
      English name, instead.
    """
    if language == "bengali":
        return (result.get("test_name_bn")
                or slots.get("test_name")
                or result.get("test_name")
                or _TEST_FALLBACK["bengali"])
    name = slots.get("test_name") or result.get("test_name")
    return name or _TEST_FALLBACK.get(language, _TEST_FALLBACK["english"])


def _name_already_says_test(name: str) -> bool:
    """"Caller asks what sample is needed" AC (word "test"/"টেস্ট" must
    never come up twice). _spoken_test_name() can fall through to the
    catalogue's plain English test_name (e.g. "Widal Test") when no
    Bengali alias is available for that row -- checking only "টেস্ট" in a
    supposedly-Bengali sentence misses that case, since the name itself
    would still be Latin script. Checking both scripts, regardless of
    which language branch is calling, closes that gap in every language,
    not just English."""
    return "test" in name.lower() or "টেস্ট" in name


_VOWEL_SOUNDS = ("a", "e", "i", "o", "u")


def _a_or_an(word: str) -> str:
    """Indefinite article for a catalogue value we don't control the
    spelling of ("Imaging" needs "an", "Blood" needs "a")."""
    return "an" if word[:1].lower() in _VOWEL_SOUNDS else "a"


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
            ("test_sample", "test_name"): "Which test's sample were you asking about?",
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
            ("test_sample", "test_name"): "Kaunse test ka sample jaanna chahte ho?",
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
            ("test_sample", "test_name"): "কোন টেস্টের স্যাম্পলের কথা জিজ্ঞেস করছেন?",
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


def _test_not_found_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    """Shared "no such test in the catalogue" reply, used by both
    test_rate_reply() and sample_type_reply() -- extracted here (Caller
    asks what sample is needed, Conversation: Information and Enquiry) so
    the not-found wording lives in exactly one place instead of being
    duplicated a second time for the new sample-only question."""
    suggestions = result.get("did_you_mean") or []
    if language == "english":
        if suggestions:
            return (f"I couldn't find a test named '{slots.get('test_name')}'. "
                     f"Did you mean {', '.join(suggestions)}?")
        return f"Sorry, we don't have a test named '{slots.get('test_name')}'."
    elif language == "hinglish":
        if suggestions:
            return (f"'{slots.get('test_name')}' naam ka test nahi mila. "
                     f"Kya aap kehna chahte the {', '.join(suggestions)}?")
        return f"Sorry, '{slots.get('test_name')}' naam ka test hamari list mein nahi hai."
    elif language == "banglish":
        if suggestions:
            return (f"'{slots.get('test_name')}' name-r test khunje pelam na. "
                     f"Apni ki bolte chaichen {', '.join(suggestions)}?")
        return f"Dukkhito, '{slots.get('test_name')}' name-r kono test amader list-e nei."
    else:  # bengali
        if suggestions:
            return (f"'{slots.get('test_name')}' নামে টেস্ট খুঁজে পাইনি। "
                     f"আপনি কি বলতে চাইছেন {', '.join(suggestions)}?")
        return f"দুঃখিত, '{slots.get('test_name')}' নামে কোনো টেস্ট আমাদের তালিকায় নেই।"


# "Caller asks what sample is needed" (Conversation: Information and
# Enquiry). AC: "The English clinical term is preserved if the caller
# used it. Multiple samples for one test are all stated." clinic-api's
# seed.py has exactly one flat sample_type string per test today (Blood,
# Urine, Imaging, Cardiac, or "Sample (Cervical)") -- no test needs more
# than one. This still handles a "|"-joined value correctly (the same
# multi-value convention Doctor.aliases_bn / LabTest.aliases_bn already
# use elsewhere in this codebase) so a future catalogue row listing more
# than one sample is spoken naturally and pluralised, without fabricating
# multiple samples for the tests that only ever need one.
_SAMPLE_RENAMES = {
    # "Sample (Cervical)" already contains the word "sample" -- left as
    # the raw catalogue value, every sentence that names it would say
    # "sample" twice ("a Sample (Cervical) sample"). Renamed to just the
    # clinical term; the parenthetical was also a spoken-punctuation risk
    # Story 2's automated check never covered (it only bans ':[]{}', not
    # '()').
    "Sample (Cervical)": "Cervical",
}

_SAMPLE_JOIN_WORD = {"english": "and", "bengali": "এবং", "hinglish": "aur", "banglish": "ar"}


def _spoken_sample_types(sample: str, language: str = "bengali") -> tuple[str, bool]:
    """-> (naturally-joined sample description, is_plural).

    Splits on "|" (today's data never contains it, but the split is a
    no-op on a single value, so this is free correctness for whenever a
    row does), applies _SAMPLE_RENAMES to each part, and joins 2+ values
    with the language's own word for "and" rather than a raw comma or
    pipe character.
    """
    parts = [p.strip() for p in sample.split("|") if p.strip()]
    cleaned = [_SAMPLE_RENAMES.get(p, p) for p in parts] or [sample]
    join_word = _SAMPLE_JOIN_WORD.get(language, _SAMPLE_JOIN_WORD["bengali"])
    if len(cleaned) == 1:
        return cleaned[0], False
    return f"{', '.join(cleaned[:-1])} {join_word} {cleaned[-1]}", True


def test_rate_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    """"Caller asks the price of a test" (Epic: Conversation -- Information
    and Enquiry). AC: "The price is read from the live catalogue and
    spoken as a natural sentence with the sample type and reporting time.
    The figure is a template substitution and is never composed by the
    model. An unknown test produces the not-found path with near matches
    offered."

    SCOPE, per explicit instruction: "only price will be told with a
    normalize[d] tone for the tests, not anything else." This function
    used to bundle rate + sample + duration into one sentence (the shape
    the AC above literally describes); it now speaks ONLY the price, in
    the same plain, consistent sentence structure across all four
    languages -- no sample clause, no duration clause. This mirrors the
    same one-question-one-answer discipline "Caller asks what sample is
    needed" (sample_type_reply(), below) already established for the
    sample-only question.

    FLAGGED, not silently absorbed: this removes the only caller-visible
    path that ever spoke bn_normalize.hours_to_duration_phrase()'s output
    ("Caller asks how long results take", the previous story) -- that
    function is untouched and still directly unit-tested, but nothing
    dispatches to it anymore. A dedicated "how long does it take" intent
    would need to be built for that story's answer to reach a caller
    again; that is not part of this change. sample_type_reply()'s own
    intent (test_sample) is unaffected -- it never called this function.

    Also fixes a real, measured bug found while verifying this story:
    _spoken_test_name() is now called with `language`, so an English/
    Hinglish/Banglish caller no longer hears a raw Bengali-script alias
    glued into their sentence (see that function's docstring for the
    full story). rate_inr keeps its pre-existing exact-passthrough
    discipline unchanged (see tests/test_number_fidelity*.py).
    """
    if not result.get("found"):
        return _test_not_found_reply(slots, result, language)

    rate = result["rate_inr"]
    name = _spoken_test_name(slots, result, language)
    name_has_test = _name_already_says_test(name)

    # Preserve exact rate value in all languages -- price only, nothing else.
    if language == "english":
        reply = f"{name} rate is {rate} rupees." if name_has_test else f"{name} test rate is {rate} rupees."
    elif language == "hinglish":
        reply = f"{name} ka rate {rate} rupaye hai." if name_has_test else f"{name} test ka rate {rate} rupaye hai."
    elif language == "banglish":
        reply = f"{name} rate {rate} taka." if name_has_test else f"{name} test-er rate {rate} taka."
    else:  # bengali
        # name_has_test checks both scripts -- see _name_already_says_test()
        reply = f"{name} রেট {rate} টাকা।" if name_has_test else f"{name} টেস্টের রেট {rate} টাকা।"
    return reply


def sample_type_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    """"Caller asks what sample is needed" (Epic: Conversation --
    Information and Enquiry). AC: "The sample type is spoken as a natural
    clause rather than a field and a colon. The English clinical term is
    preserved if the caller used it. Multiple samples for one test are
    all stated."

    Answers ONLY the sample-type question, for the caller who asked
    nothing but "what sample do I need for X" -- test_rate_reply() (above)
    now answers ONLY the price question, per the same one-question-one-
    answer discipline (see that function's docstring). Reuses the exact
    same clinic-api lookup test_rate_reply() does (the API already
    returns sample_type on every test-info call; nothing new was added to
    clinic-api for this) -- only what gets SPOKEN differs.

    `_spoken_test_name(..., language)` -- fixes the same mixed-script bug
    test_rate_reply() was fixed for ("Caller asks the price of a test"):
    this call used to omit `language` entirely, so an English/Hinglish/
    Banglish caller could hear a raw Bengali-script alias in this
    sentence too (e.g. "For সিবিসি test, you'll need to give..."). See
    _spoken_test_name()'s own docstring for the full story.

    "The English clinical term is preserved" is genuinely satisfied for
    english/hinglish/banglish here -- sample_type flows through
    unaltered, same as rate_inr's digit-fidelity discipline. It is NOT
    literally possible for the bengali branch: the Bengali TTS model
    cannot pronounce untranslated Latin script at all (see bn_normalize.
    py's module docstring and unspeakable_spans() -- this was measured,
    not assumed), which is exactly why _LATIN_SPOKEN_BN exists. Before
    this story, "Imaging", "Cardiac" and "Sample (Cervical)" had no entry
    in that table, so a Bengali caller asking about any of the 7 tests
    using one of those 3 sample types heard nothing at all for the
    sample -- not a wrong word, silence. This story extends
    _LATIN_SPOKEN_BN with the 3 missing entries, the same treatment
    "blood"/"urine" already got, so the Bengali branch speaks a real word
    instead of dropping the caller's answer entirely.

    Never fabricates a second sample for a test that only needs one --
    see _spoken_sample_types()'s module comment for why LabTest.
    sample_type has no way to represent more than one today.
    """
    if not result.get("found"):
        return _test_not_found_reply(slots, result, language)

    name = _spoken_test_name(slots, result, language)
    sample = result.get("sample_type")
    name_has_test = _name_already_says_test(name)

    if not sample:
        # Honest fallback for a malformed/incomplete catalogue row -- see
        # the "Truth Validator" epic's sibling stories for schema-drift
        # handling in general; this is not that, just a guard against
        # ever inventing a sample that was never returned.
        if language == "english":
            return f"Sorry, I don't have the sample details for {name} right now."
        elif language == "hinglish":
            return f"Sorry, {name} ke liye sample ki jaankari abhi available nahi hai."
        elif language == "banglish":
            return f"Dukkhito, {name}-er sample-er kotha ekhon bolte parchi na."
        else:  # bengali
            return f"দুঃখিত, {name}-এর স্যাম্পল সম্পর্কে এখন বলতে পারছি না।"

    sample_str, sample_plural = _spoken_sample_types(sample, language)

    if language == "english":
        noun = f"{sample_str} samples" if sample_plural else f"{_a_or_an(sample_str)} {sample_str} sample"
        suffix = "" if name_has_test else " test"
        return f"For {name}{suffix}, you'll need to give {noun}."
    elif language == "hinglish":
        noun = f"{sample_str} samples" if sample_plural else f"{sample_str} sample"
        suffix = "" if name_has_test else " test"
        return f"{name}{suffix} ke liye {noun} dena hoga."
    elif language == "banglish":
        noun = f"{sample_str} samples" if sample_plural else f"{sample_str} sample ta"
        suffix = "" if name_has_test else " test"
        return f"{name}{suffix} er jonno {noun} lagbe."
    else:  # bengali
        noun = f"{sample_str} স্যাম্পলগুলো" if sample_plural else f"{sample_str} স্যাম্পলটা"
        if name_has_test:
            return f"{name}-এর জন্য {noun} লাগবে।"
        return f"{name} টেস্টের জন্য {noun} লাগবে।"


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
            return (f"Yes,{date_txt} {name} will be in chamber. The chamber hours are {hours}. "
                    f"Would you like to book for today or another day?")
        elif language == "hinglish":
            return (f"Haan,{date_txt} {name} chamber mein honge. Chamber ka time hai {hours}. "
                    f"Aaj ke liye appointment karna chahte ho ya kisi aur din ke liye?")
        else:  # bengali
            date_txt_bn = f" {result.get('date')} তারিখে" if result.get("date") else " আজ"
            return (f"হ্যাঁ,{date_txt_bn} {name} চেম্বারে থাকবেন। চেম্বারের সময় {hours}। "
                    f"আজকের জন্যই অ্যাপয়েন্টমেন্ট করবেন, নাকি অন্য কোনো দিনের জন্য?")

    next_date = result.get("next_available_date")
    if next_date:
        if language == "english":
            return (f"{name} won't be available that day. The next available date is {next_date}. "
                    f"Would you like to book for that day?")
        elif language == "hinglish":
            return (f"{name} us din nahi honge. Agla available date hai {next_date}. "
                    f"Us din ke liye appointment karna chahte ho?")
        else:  # bengali
            return (f"{name} ওই দিন বসবেন না। পরবর্তী উপলব্ধ দিনটা হলো {next_date}। "
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
                    f"Your confirmation number is {confirmation_id}.")
        elif language == "hinglish":
            return (f"Aapka appointment confirm ho gaya. "
                    f"{doctor}, {date}, time {time_slot}. "
                    f"Aapka confirmation number hai {confirmation_id}.")
        else:  # bengali
            return (f"আপনার অ্যাপয়েন্টমেন্ট কনফার্ম হয়েছে। "
                    f"{doctor}, {date}, সময় {time_slot}। "
                    f"আপনার কনফার্মেশন নম্বর হলো {confirmation_id}।")

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            if language == "english":
                return f"That time is already booked, but {', '.join(alts)} are available. Which would you prefer?"
            elif language == "hinglish":
                return f"Wo time already book ho gaya, lekin {', '.join(alts)} available hain. Kaunsa prefer karte ho?"
            else:  # bengali
                return f"ওই সময়টা বুক হয়ে গেছে, তবে {', '.join(alts)} সময়গুলো ফাঁকা আছে। কোনটা চান?"
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
        return "No problem -- which one should I fix, the doctor, date, time, name, or phone number?"
    elif language == "hinglish":
        return "Koi baat nahi -- kya theek karna hai, doctor, date, time, naam, ya phone number?"
    elif language == "banglish":
        return "Kono problem nei -- ki thik korte hobe, doctor, date, time, naam, na ki phone number?"
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
