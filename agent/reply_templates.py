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
            # ADDED BY SOURAV -- "Lab Report Status & Secure Delivery".
            # Identity for a report lookup is the caller's REGISTERED
            # phone (Rule 14/15), not their name -- so unlike every other
            # intent above, "phone" here is the identity check itself,
            # not a delivery-confirmation courtesy.
            ("report_status", "phone"): "Could you tell me your registered phone number?",
            ("report_send", "phone"): "Could you tell me your registered phone number?",
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
            ("report_status", "phone"): "Apna registered phone number bata sakte ho?",
            ("report_send", "phone"): "Apna registered phone number bata sakte ho?",
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
            ("report_status", "phone"): "আপনার নিবন্ধিত ফোন নম্বরটা বলবেন?",
            ("report_send", "phone"): "আপনার নিবন্ধিত ফোন নম্বরটা বলবেন?",
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


def _digit_faithful_rate(raw_rate) -> str:
    """UPDATED BY SOURAV -- real production bug found while re-verifying
    this combined story: clinic-api's `LabTest.rate_inr` column is a
    SQLAlchemy Float, so the live catalogue always serializes it as a
    JSON float (e.g. 250.0), even for a rate that was seeded as a plain
    integer. agent/tools_client.py's `_parse_exact()` deliberately keeps
    that as the STRING "250.0" (digit-fidelity design, see
    tests/test_number_fidelity*.py) -- so every real caller was hearing
    a literal trailing ".0" in the price ("... রেট 250.0 টাকা।"), which
    verbalize() then spoke aloud as "point zero". That is wrong for a
    whole-rupee amount and was never caught before because no earlier
    test asserted digit-faithfulness against a REAL DB-sourced rate
    through this exact function (tests/test_live_test_price_lookup.py's
    test_real_rate_is_digit_faithful_through_the_full_live_pipeline is
    the first one that does).

    This strips ONLY an exact whole-number trailing ".0" -- it does not
    round or otherwise touch a genuinely fractional value (e.g.
    "199.55" is returned completely unchanged), preserving the same
    never-mutate-a-digit discipline `_parse_exact()` was built for.
    Accepts a str, int, or float, since call sites differ (real
    production hands this a string via `_parse_exact`; some tests hand
    it a native float straight from `r.json()` or a DB row).
    """
    text = str(raw_rate)
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


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

    # UPDATED BY SOURAV -- was `rate = result["rate_inr"]` (raw
    # passthrough). See _digit_faithful_rate()'s docstring above for the
    # real bug this fixes (a literal trailing ".0"/"point zero" spoken
    # for every test's price) and why this is still digit-fidelity safe.
    rate = _digit_faithful_rate(result["rate_inr"])
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


# =============================================================================
# ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story
# (previously two separate stories: "is my report ready" and "send my
# report"). Every function below composes the reply exactly the same way
# every function above it does -- a template substitution over a real
# clinic-api response, never a fact the model states on its own -- see
# this file's own module docstring. FLOWS INTO: agent/report_flow.py's
# interpret_*() functions decide WHICH of these to call and what the next
# pending state should be; main_pcm.py / main.py call report_flow.py and
# then _speak() whatever text comes back.
#
# RULE 9 (never reveal the OTP) as a structural property of this file:
# search this whole block -- no function below ever reads an `otp_code`
# key out of any `result` dict, or receives one as a parameter. There is
# no code path here that COULD speak the correct OTP even by accident.
#
# RULE 10 (never expose the full registered phone number): every function
# that mentions the caller's phone uses `_last4()` below, never the full
# value clinic-api's `masked_phone` field already is (clinic-api masks
# to 4 digits itself -- `_last4` here exists for the one caller-supplied
# phone this file ever touches directly: the raw digits main_pcm.py/
# main.py parsed with agent.slot_parse.parse_phone(), before any tool
# call has happened yet, e.g. while composing the "is my report ready"
# offer question. Once a tool response comes back, its OWN
# `masked_phone` field is used instead, kept in the same masked shape.
# =============================================================================

def _last4(phone: str | None) -> str:
    """Rule 10. A local copy of clinic-api/main.py's `_mask_phone_last4`
    -- duplicated rather than imported because this file (the voice
    agent) and clinic-api are two separate deployables that do not share
    a Python import path; see agent/tools_client.py's module docstring."""
    if not phone:
        return "----"
    digits = "".join(c for c in phone if c.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits


def patient_not_found_reply(language: str = "bengali") -> str:
    """The phone number the caller gave does not match any registered
    patient. RULE 1's "never invent" extends to identity too -- this
    never guesses which patient they might mean."""
    if language == "english":
        return "I couldn't find a patient registered with that phone number. Could you double-check it?"
    elif language == "hinglish":
        return "Is phone number se koi patient registered nahi mila. Number ek baar check kar lenge?"
    elif language == "banglish":
        return "Ei number diye kono patient registered pelam na. Number ta ektu check korben?"
    else:  # bengali
        return "এই ফোন নম্বর দিয়ে কোনো রোগী নিবন্ধিত পাইনি। নম্বরটা একটু দেখে বলবেন?"


def report_not_found_reply(language: str = "bengali") -> str:
    """RULE 1: an honest NOT_FOUND, whether it's "no reports at all"
    (Patient G) or "no report matching that test name". Never claims a
    report exists, is ready, or offers delivery/OTP for it."""
    if language == "english":
        return "I couldn't find a matching report for you. Please check with the counter."
    elif language == "hinglish":
        return "Aapka koi matching report nahi mila. Counter mein ek baar check kar lijiye."
    elif language == "banglish":
        return "Apnar matching kono report khunje pelam na. Counter e ektu check kore nin."
    else:  # bengali
        return "আপনার সাথে মেলে এমন কোনো রিপোর্ট খুঁজে পাইনি। দয়া করে কাউন্টারে খোঁজ নিন।"


def report_ambiguous_reply(result: dict, language: str = "bengali") -> str:
    """RULE 13: multiple reports match -- ask, using SAFE identifying
    information (test name only, per the plan's own suggestion), never
    guess or pick the first one. Candidates come straight from
    clinic-api's `candidates` list (see main.py's report_status())."""
    names = [c["test_name"] for c in (result.get("candidates") or [])]
    if language == "english":
        listing = ", ".join(names[:-1]) + f", and {names[-1]}" if len(names) > 1 else (names[0] if names else "")
        return f"You have more than one report on file -- {listing}. Which one do you mean?"
    elif language == "hinglish":
        listing = ", ".join(names[:-1]) + f", aur {names[-1]}" if len(names) > 1 else (names[0] if names else "")
        return f"Aapke naam pe ek se zyada report hai -- {listing}. Kaunsi wali chahiye?"
    elif language == "banglish":
        listing = ", ".join(names[:-1]) + f", ar {names[-1]}" if len(names) > 1 else (names[0] if names else "")
        return f"Apnar naame ekadhik report ache -- {listing}. Konta bolchen?"
    else:  # bengali
        listing = ", ".join(names[:-1]) + f" এবং {names[-1]}" if len(names) > 1 else (names[0] if names else "")
        return f"আপনার নামে একাধিক রিপোর্ট আছে -- {listing}। কোনটার কথা বলছেন?"


_STATUS_WORDS = {
    "NOT_READY": {"english": "not ready yet", "hinglish": "abhi ready nahi hai",
                  "banglish": "ekhono ready hoyni", "bengali": "এখনো তৈরি হয়নি"},
    "PROCESSING": {"english": "still being processed", "hinglish": "process ho raha hai",
                   "banglish": "processing chalche", "bengali": "এখনো প্রসেস হচ্ছে"},
    "CANCELLED": {"english": "cancelled", "hinglish": "cancel ho gaya hai",
                  "banglish": "cancel hoye geche", "bengali": "বাতিল হয়ে গেছে"},
}


def report_status_reply(result: dict, language: str = "bengali") -> str:
    """RULE 2 (NOT_READY means no delivery), RULE 3 (READY required
    before delivery), RULE 16 (delivery_enabled gate). This function is
    also where RULE 2's "no clinical value is read aloud" holds
    structurally: `result` (clinic-api's report_status() response) never
    contains a clinical value at all -- no LabReport column stores one,
    per models.py's own docstring ("NO clinical value is read aloud
    under this story") -- there is nothing here that COULD leak one.

    Only offers delivery (appends the offer question) when the report is
    READY *and* delivery_enabled -- Patient I (READY, delivery_enabled
    False) hears the true status but is never asked if they want it sent.
    """
    test_name = result.get("test_name", "")
    status = result.get("status")

    if status == "READY":
        if result.get("delivery_enabled"):
            if language == "english":
                return (f"Good news -- your {test_name} report is ready. "
                        f"Would you like me to send it to your registered phone?")
            elif language == "hinglish":
                return (f"Achi khabar -- aapka {test_name} report ready hai. "
                        f"Kya aapke registered phone pe bhej doon?")
            elif language == "banglish":
                return (f"Bhalo khobor -- apnar {test_name} report ready hoye geche. "
                        f"Apnar registered phone e pathiye debo?")
            else:  # bengali
                return (f"সুখবর -- আপনার {test_name} রিপোর্ট তৈরি হয়ে গেছে। "
                        f"আপনার নিবন্ধিত ফোনে পাঠিয়ে দেব?")
        # READY but delivery_enabled is False (Patient I) -- true status,
        # no offer, and no explanation of WHY (that is an internal flag,
        # not something a caller-facing reply should describe -- see
        # RULE 12/ATTACK 13's "don't expose internal implementation").
        if language == "english":
            return f"Your {test_name} report is ready. Please collect it in person from the clinic."
        elif language == "hinglish":
            return f"Aapka {test_name} report ready hai. Please clinic se khud collect kar lein."
        elif language == "banglish":
            return f"Apnar {test_name} report ready. Please clinic theke nijei collect korben."
        else:  # bengali
            return f"আপনার {test_name} রিপোর্ট তৈরি হয়ে গেছে। দয়া করে ক্লিনিক থেকে সশরীরে সংগ্রহ করুন।"

    status_words = _STATUS_WORDS.get(status, _STATUS_WORDS["PROCESSING"])
    words = status_words.get(language, status_words["bengali"])
    if language == "english":
        return f"Your {test_name} report is {words}. You don't need to travel yet -- please check back later."
    elif language == "hinglish":
        return f"Aapka {test_name} report {words}. Abhi aane ki zaroorat nahi -- thodi der baad check kariye."
    elif language == "banglish":
        return f"Apnar {test_name} report {words}. Ekhon asar dorkar nei -- pore abar check korben."
    else:  # bengali
        return f"আপনার {test_name} রিপোর্ট {words}। এখনই আসার দরকার নেই -- একটু পরে আবার খোঁজ নেবেন।"


def delivery_blocked_reply(reason: str, language: str = "bengali") -> str:
    """Used by the "report_send" flow (caller opened directly with "send
    my report") when the report cannot enter delivery at all -- RULE 3 /
    RULE 16, and TEST 12-15 in the plan's final matrix
    (DELIVERY_BLOCKED / DELIVERY_DISABLED). Deliberately never says WHY
    beyond the plain status word for NOT_READY/PROCESSING/CANCELLED, and
    for DELIVERY_DISABLED never mentions the internal flag name at all
    (ATTACK 13)."""
    if reason == "DELIVERY_DISABLED":
        if language == "english":
            return "This report isn't available for phone delivery. Please collect it in person from the clinic."
        elif language == "hinglish":
            return "Yeh report phone pe deliver nahi ho sakti. Please clinic se khud collect kar lein."
        elif language == "banglish":
            return "Ei report phone e deliver kora jabe na. Please clinic theke nijei collect korben."
        else:  # bengali
            return "এই রিপোর্টটা ফোনে পাঠানো যাচ্ছে না। দয়া করে ক্লিনিক থেকে সশরীরে সংগ্রহ করুন।"

    status_words = _STATUS_WORDS.get(reason, {})
    words = status_words.get(language, status_words.get("bengali", ""))
    if language == "english":
        return f"I can't send that report yet -- it's {words}. Please check back later, or visit the clinic."
    elif language == "hinglish":
        return f"Abhi woh report bhej nahi sakte -- {words}. Baad mein check kariye, ya clinic aa jaiye."
    elif language == "banglish":
        return f"Ekhon oi report pathano jabe na -- {words}. Pore check korben, na hoy clinic e asben."
    else:  # bengali
        return f"এখনই ওই রিপোর্ট পাঠানো যাচ্ছে না -- এটা {words}। পরে খোঁজ নেবেন, বা ক্লিনিকে আসতে পারেন।"


def delivery_declined_reply(language: str = "bengali") -> str:
    """Caller was offered delivery (report_status_reply's READY+enabled
    branch) and said no."""
    if language == "english":
        return "Alright, no problem. Is there anything else I can help with?"
    elif language == "hinglish":
        return "Theek hai, koi baat nahi. Aur kuch madad chahiye?"
    elif language == "banglish":
        return "Thik ache, kono problem nei. Aro kichu jante chan?"
    else:  # bengali
        return "ঠিক আছে, কোনো সমস্যা নেই। আর কিছু জানতে চান?"


def otp_requested_reply(result: dict, language: str = "bengali") -> str:
    """RULE 4 (OTP required), RULE 9 (never reveal it), RULE 10 (masked
    phone only). `result` is clinic-api's request_report_delivery()
    success response -- its `masked_phone` field is already
    last-4-digits (see clinic-api/main.py's `_mask_phone_last4`); this
    function only ever speaks that field, never a raw phone slot."""
    masked = result.get("masked_phone", "----")
    if language == "english":
        return f"I've sent an OTP to your registered number ending in {masked}. Please tell me the OTP."
    elif language == "hinglish":
        return f"Aapke registered number, jo {masked} pe khatam hota hai, us par OTP bhej diya hai. OTP bataiye."
    elif language == "banglish":
        return f"Apnar registered number, ja {masked} diye shesh, e OTP pathiye diyechi. OTP ta bolun."
    else:  # bengali
        return f"আপনার নিবন্ধিত নম্বরে, যেটা {masked} দিয়ে শেষ, একটা ওটিপি পাঠিয়েছি। ওটিপিটা বলুন।"


def otp_disclosure_refusal_reply(language: str = "bengali") -> str:
    """ATTACK 8: "tell me the OTP you sent." RULE 9 in its most direct
    form -- refuses outright, and redirects to the only acceptable
    source (the caller's own phone), rather than a generic "didn't
    understand, try again" that could read as evasive rather than a
    deliberate refusal."""
    if language == "english":
        return "I'm not able to tell you the OTP -- please read it from the message on your phone and tell me."
    elif language == "hinglish":
        return "Main OTP nahi bata sakta -- please apne phone par aaye message se OTP padh kar bataiye."
    elif language == "banglish":
        return "Ami OTP ta bolte parbo na -- please apnar phone e asha message theke OTP ta bolun."
    else:  # bengali
        return "আমি ওটিপিটা বলতে পারব না -- দয়া করে আপনার ফোনে আসা মেসেজ থেকে ওটিপিটা পড়ে বলুন।"


def otp_verify_reply(result: dict, language: str = "bengali") -> str:
    """The entire OTP/delivery outcome surface (plan Section 6 and 9) in
    one function -- mirrors clinic-api/main.py's verify_report_otp()
    reason enum exactly, one branch per reason, so a new reason added
    there is a loud KeyError-shaped gap here rather than a silently
    generic reply. RULE 9 holds throughout: none of these branches ever
    receives or reads the actual OTP value."""
    reason = result.get("reason")

    if reason == "DELIVERY_SENT":
        minutes = result.get("signed_link_expires_minutes", 15)
        if language == "english":
            return f"Your report has been securely sent. The link will expire in {minutes} minutes."
        elif language == "hinglish":
            return f"Aapka report securely bhej diya gaya hai. Link {minutes} minute mein expire ho jayega."
        elif language == "banglish":
            return f"Apnar report securely pathiye deoya hoyeche. Link {minutes} minute e expire hoye jabe."
        else:  # bengali
            return f"আপনার রিপোর্ট নিরাপদে পাঠানো হয়েছে। লিংকটা {minutes} মিনিটের মধ্যে মেয়াদ শেষ হয়ে যাবে।"

    if reason == "OTP_INVALID":
        if language == "english":
            return "That OTP doesn't match. Please check your phone and tell me the OTP again."
        elif language == "hinglish":
            return "Yeh OTP match nahi kar raha. Phone check karke dobara OTP bataiye."
        elif language == "banglish":
            return "Ei OTP ta mile na. Phone check kore abar OTP ta bolun."
        else:  # bengali
            return "এই ওটিপিটা মিলছে না। ফোন দেখে আবার ওটিপিটা বলুন।"

    if reason == "OTP_EXPIRED":
        if language == "english":
            return "That OTP has expired. Let me know if you'd still like the report sent, and I'll send a new one."
        elif language == "hinglish":
            return "Yeh OTP expire ho gaya hai. Agar abhi bhi report chahiye toh bataiye, naya OTP bhej dunga."
        elif language == "banglish":
            return "Ei OTP ta expire hoye geche. Ekhono report chan ki na bolun, notun OTP pathiye debo."
        else:  # bengali
            return "এই ওটিপিটার মেয়াদ শেষ হয়ে গেছে। এখনো রিপোর্ট চান কি না বলুন, নতুন ওটিপি পাঠিয়ে দেব।"

    if reason == "OTP_ALREADY_USED":
        if language == "english":
            return "That OTP has already been used. Please ask me to send the report again if you need a new one."
        elif language == "hinglish":
            return "Yeh OTP pehle hi use ho chuka hai. Naya chahiye toh dobara report bhejne ko boliye."
        elif language == "banglish":
            return "Ei OTP ta age e use hoye geche. Notun lagle abar report pathate bolun."
        else:  # bengali
            return "এই ওটিপিটা আগেই ব্যবহার হয়ে গেছে। নতুন লাগলে আবার রিপোর্ট পাঠাতে বলুন।"

    if reason == "OTP_MAX_ATTEMPTS":
        # RULE 8: locked out. Never reveals the correct value, and
        # explicitly points to a fresh flow rather than repeating the
        # same OTP prompt (which would be pointless -- the row is dead).
        if language == "english":
            return ("You've entered the wrong OTP too many times, so I can't verify it right now. "
                     "Please ask me to send the report again to get a new OTP.")
        elif language == "hinglish":
            return ("Bahut baar galat OTP diya gaya hai, isliye abhi verify nahi kar sakte. "
                     "Naya OTP ke liye dobara report bhejne ko boliye.")
        elif language == "banglish":
            return ("Onek bar bhul OTP deoya hoyeche, tai ekhon verify kora jabe na. "
                     "Notun OTP er jonno abar report pathate bolun.")
        else:  # bengali
            return ("অনেকবার ভুল ওটিপি দেওয়া হয়েছে, তাই এখন যাচাই করা যাচ্ছে না। "
                     "নতুন ওটিপির জন্য আবার রিপোর্ট পাঠাতে বলুন।")

    if reason == "OTP_NOT_REQUESTED":
        if language == "english":
            return "I haven't sent an OTP for this report yet. Would you like me to send one?"
        elif language == "hinglish":
            return "Iss report ke liye abhi OTP bheja hi nahi hai. Bhej doon?"
        elif language == "banglish":
            return "Ei report er jonno ekhono OTP pathano hoyni. Pathiye debo?"
        else:  # bengali
            return "এই রিপোর্টের জন্য এখনো ওটিপি পাঠানো হয়নি। পাঠিয়ে দেব?"

    if reason == "DELIVERY_FAILED":
        # RULE 17: OTP succeeded, but delivery itself failed -- never
        # claim success. Offers the honest fallback (collection in
        # person / try again), matching RULE 17's exact wording.
        if language == "english":
            return "We couldn't send the report right now. Please try again later or collect it in person from the clinic."
        elif language == "hinglish":
            return "Abhi report bhej nahi paye. Please thodi der baad try kariye ya clinic se khud collect kar lein."
        elif language == "banglish":
            return "Ekhon report pathate parlam na. Please pore abar try korben ba clinic theke nijei collect korben."
        else:  # bengali
            return "এই মুহূর্তে রিপোর্টটা পাঠাতে পারলাম না। দয়া করে পরে আবার চেষ্টা করুন, বা ক্লিনিক থেকে সশরীরে সংগ্রহ করুন।"

    # Defensive re-checks: the report's state changed between the
    # delivery offer and the OTP being verified (RULE 3/RULE 16 checked
    # again server-side -- see clinic-api's verify_report_otp()).
    if reason in ("NOT_READY", "PROCESSING", "CANCELLED"):
        return delivery_blocked_reply(reason, language)
    if reason == "DELIVERY_DISABLED":
        return delivery_blocked_reply(reason, language)
    if reason in ("PATIENT_NOT_FOUND",):
        return patient_not_found_reply(language)
    # NOT_FOUND -- the report vanished/mismatched between calls.
    return report_not_found_reply(language)
