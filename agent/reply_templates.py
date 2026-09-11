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
# ADDED BY SOURAV -- "Caller asks when a doctor sits" story:
# doctor_schedule_reply() below speaks a doctor's weekly sitting days by
# name, in whichever of the 4 reply languages it was asked for. See that
# function's docstring for how this differs from doctor_availability_reply
# just above it.
from agent.bn_normalize import weekday_to_words
# ADDED BY SOURAV -- real production bug fix: a caller asking "how long
# does it take to get the urine test report" was being answered with the
# test's PRICE instead ("Urine test rate is 200 taka."). Root cause:
# test_rate_reply() above was narrowed by an earlier story ("Caller asks
# the price of a test") to speak ONLY the price, and that story's own
# docstring explicitly flagged that this orphaned bn_normalize.
# hours_to_duration_phrase() -- built and unit-tested by an even earlier
# story ("Caller asks how long results take") -- with no caller-visible
# path left to reach it. test_duration_reply() below is that missing
# path: a new, dedicated intent that reuses this exact same primitive,
# never rebuilds it. See test_duration_reply()'s own docstring for the
# full bug writeup.
from agent.bn_normalize import hours_to_duration_phrase

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
            # ADDED BY SOURAV -- real production bug fix (see
            # test_duration_reply()'s docstring). Same wording pattern as
            # test_sample just above -- "which test" is identical
            # regardless of whether the caller then wants the sample or
            # the report turnaround time.
            ("test_duration", "test_name"): "Which test's report time were you asking about?",
            # ADDED BY SOURAV -- "Caller asks how to prepare for a test" story.
            ("test_preparation", "test_name"): "Which test's preparation instructions were you asking about?",
            ("doctor_availability", "doctor_name"): "Which doctor are you asking about?",
            # ADDED BY SOURAV -- "Caller asks when a doctor sits" story.
            # Same question wording as doctor_availability just above --
            # "which doctor" is identical regardless of whether the
            # caller then wants a specific day's availability or the
            # general weekly schedule.
            ("doctor_schedule", "doctor_name"): "Which doctor are you asking about?",
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
            ("test_duration", "test_name"): "Kaunse test ki report ka time jaanna chahte ho?",
            ("test_preparation", "test_name"): "Kaunse test ki taiyari ke baare mein pooch rahe ho?",
            ("doctor_availability", "doctor_name"): "Kaunse doctor ke baare mein pooch rahe ho?",
            ("doctor_schedule", "doctor_name"): "Kaunse doctor ke baare mein pooch rahe ho?",
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
            ("test_duration", "test_name"): "কোন টেস্টের রিপোর্টের সময়ের কথা জিজ্ঞেস করছেন?",
            ("test_preparation", "test_name"): "কোন টেস্টের প্রস্তুতির কথা জিজ্ঞেস করছেন?",
            ("doctor_availability", "doctor_name"): "কোন ডাক্তারের কথা জিজ্ঞেস করছেন?",
            ("doctor_schedule", "doctor_name"): "কোন ডাক্তারের কথা জিজ্ঞেস করছেন?",
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


def test_duration_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    """ADDED BY SOURAV -- fixes a real production bug, reported directly
    from a live call transcript:

        [User] How long does it take to get the urine test report?
        [AI]   Urine test rate is 200 taka.
        [User] How long will it take to get the urine test report?
        [AI]   Urine test rate is 200 taka.

    A caller asking about REPORT TURNAROUND TIME was being misclassified
    as "test_rate" and answered with the test's PRICE instead -- twice in
    a row, since nothing about the exchange gave the classifier a reason
    to reconsider.

    ROOT CAUSE: "Caller asks the price of a test" (see test_rate_reply()'s
    own docstring, above) deliberately narrowed test_rate_reply() to speak
    ONLY the price, and its docstring explicitly flagged this as removing
    the only caller-visible path to bn_normalize.hours_to_duration_phrase()
    -- built and unit-tested by an even earlier story ("Caller asks how
    long results take") but never wired to anything a caller could
    actually trigger. agent/llm.py's intent prompt was never updated to
    match: it kept describing test_rate as covering "how long results
    take," so the classifier kept routing duration questions to test_rate,
    which (correctly, per its own later-narrowed scope) speaks only the
    price. Two separate, true things -- test_rate_reply()'s narrowed scope,
    and llm.py's stale intent description -- combined into exactly the bug
    reported above. Fixed in agent/llm.py by adding a dedicated
    "test_duration" intent and correcting both intents' descriptions.

    A SECOND bug was found and fixed alongside this one, in
    agent/fast_path.py: the pre-LLM local matcher's _RATE_CUES set
    included ambiguous verbs ("কত লাগবে", "কত পড়বে", "কত নেবে") that mean
    either "how much will it COST" or "how much/long will it TAKE" in
    colloquial Bengali. A bare "রিপোর্ট পেতে কত লাগবে" (no explicit time
    word) was being fast-pathed straight to test_rate, bypassing the LLM's
    new, correct distinction entirely -- reproducing this exact bug at a
    second layer. See agent/fast_path.py's _AMBIGUOUS_RATE_CUES /
    _DURATION_SIGNAL_CUES comments for that fix.

    This function itself does no new lookup -- it reuses the SAME
    clinic-api response test_rate_reply()/sample_type_reply() already get
    from get_test_rate() (report_time_hours has always been present in
    that payload; nothing new was added to clinic-api for this), and
    reuses hours_to_duration_phrase() exactly as it already exists and is
    already unit-tested -- never rebuilds it. One-question-one-answer
    discipline applies here too: this speaks ONLY the duration, never the
    price or the sample, mirroring test_rate_reply()/sample_type_reply().
    """
    if not result.get("found"):
        return _test_not_found_reply(slots, result, language)

    name = _spoken_test_name(slots, result, language)
    hours = result.get("report_time_hours")
    name_has_test = _name_already_says_test(name)

    if hours is None:
        # Honest fallback for a malformed/incomplete catalogue row -- same
        # discipline as sample_type_reply()'s missing-sample fallback just
        # above: never fabricate a duration that was never returned.
        if language == "english":
            return f"Sorry, I don't have the report time for {name} right now."
        elif language == "hinglish":
            return f"Sorry, {name} ke report time ki jaankari abhi available nahi hai."
        elif language == "banglish":
            return f"Dukkhito, {name}-er report time ekhon bolte parchi na."
        else:  # bengali
            return f"দুঃখিত, {name}-এর রিপোর্টের সময় সম্পর্কে এখন বলতে পারছি না।"

    duration = hours_to_duration_phrase(hours, language)

    if language == "english":
        suffix = "" if name_has_test else " test"
        return f"The {name}{suffix} report will be ready {duration}."
    elif language == "hinglish":
        suffix = "" if name_has_test else " test"
        return f"{name}{suffix} ki report {duration} ready ho jaayegi."
    elif language == "banglish":
        suffix = "" if name_has_test else " test"
        return f"{name}{suffix}-er report {duration} ready hoye jabe."
    else:  # bengali
        if name_has_test:
            return f"{name}-এর রিপোর্ট {duration} রেডি হয়ে যাবে।"
        return f"{name} টেস্টের রিপোর্ট {duration} রেডি হয়ে যাবে।"


def _test_preparation_unavailable_reply(name: str, language: str = "bengali") -> str:
    """ADDED BY SOURAV -- "Caller asks how to prepare for a test" story.
    Shared honest fallback for BOTH real cases where no preparation
    script can be spoken: `advisory_available: False` (this test exists
    but the business has never supplied preparation content for it --
    see clinic-api/models.py's own comment on why LabTest's advisory
    columns are nullable with no default), and the defensive case of a
    malformed catalogue row that claims `advisory_available: True` but is
    actually missing the script for THIS language. Never says "no special
    preparation needed" -- that is a specific, possibly-wrong medical
    claim this codebase has no basis to make up; it says plainly that the
    information isn't available and points the caller at a human instead,
    same discipline as agent/outcomes.py's insufficient-verified-
    information outcome."""
    if language == "english":
        return (f"I don't have preparation instructions for {name} yet. "
                 f"Please check with the counter or your doctor.")
    elif language == "hinglish":
        return (f"{name} ke liye abhi preparation ki jaankari mere paas nahi hai. "
                 f"Counter ya apne doctor se check kar lijiye.")
    elif language == "banglish":
        return (f"{name}-er jonno ekhon preparation-er information amar kache nei. "
                 f"Counter othoba apnar doctor-ke jiggesh korben.")
    else:  # bengali
        return (f"{name}-এর জন্য এখন প্রস্তুতির তথ্য আমার কাছে নেই। "
                 f"দয়া করে কাউন্টারে বা আপনার ডাক্তারকে জিজ্ঞেস করুন।")


_ADVISORY_SCRIPT_FIELD_FOR_LANGUAGE = {
    "english": "advisory_script_en",
    "hinglish": "advisory_script_hinglish",
    "banglish": "advisory_script_banglish",
    "bengali": "advisory_script_bn",
}


def test_preparation_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    """"Caller asks how to prepare for a test" (Epic: Conversation --
    Information and Enquiry). Speaks the business's own supplied
    preparation script (clinic-api/seed.py's LAB_TEST_ADVISORIES,
    sourced verbatim from the lab_tests_with_fallback_config sample
    file) for the caller's language, with the literal "{test_name}"
    placeholder substituted using the exact same Bengali-alias-vs-
    caller's-own-words convention _spoken_test_name() already applies
    everywhere else in this file.

    THREE distinct outcomes, matching clinic-api/main.py's
    _test_preparation_reply_dict() docstring exactly:

      1. found=False -> the shared not-found/did-you-mean reply (same
         helper test_rate_reply()/sample_type_reply()/test_duration_
         reply() already use).
      2. found=True, advisory_available=False -> the honest "we don't
         have this yet" fallback above. This is the entire reason
         LabTest's advisory columns are nullable with no default: a test
         nobody has actually reviewed must never be told "no special
         preparation needed" -- see clinic-api/models.py's own comment.
      3. found=True, advisory_available=True -> speaks the pre-written,
         business-approved advisory_script_* for this language VERBATIM
         (only the {test_name} placeholder is substituted) -- never
         recomposed from the structured fasting_required/fasting_hours/
         water_allowance/medication_hold/timing_rule fields, for the same
         reason clinic-api/seed.py stores these scripts verbatim instead
         of generating sentences from those fields: they are the exact
         wording the business already reviewed and approved, and this
         codebase has no business rephrasing a medical instruction.
    """
    if not result.get("found"):
        return _test_not_found_reply(slots, result, language)

    name = _spoken_test_name(slots, result, language)

    if not result.get("advisory_available"):
        return _test_preparation_unavailable_reply(name, language)

    field = _ADVISORY_SCRIPT_FIELD_FOR_LANGUAGE.get(language, "advisory_script_bn")
    script = result.get(field)
    if not script:
        # Defensive only -- clinic-api/seed.py always populates all 4
        # languages together for every advisory-covered test (see this
        # story's TEST_REPORT), so a real response never actually hits
        # this branch. Kept anyway rather than raising or speaking a
        # blank reply, same "never trust a response shape blindly"
        # discipline as sample_type_reply()'s missing-sample fallback.
        return _test_preparation_unavailable_reply(name, language)
    return script.replace("{test_name}", name)


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


# ADDED BY SOURAV -- "Caller asks when a doctor sits" story. Groups the
# doctor's schedule rows by their (start_time, end_time) pair so several
# weekdays sharing identical chamber hours (the common case -- see
# clinic-api/seed.py's SHIFT_TEMPLATES, which gives every seeded doctor
# ONE set of hours across all their sitting days) are spoken as a single
# natural clause ("Monday, Wednesday and Friday, 10 to 12") instead of
# three separate, repetitive sentences. Still handles the general case
# where a real future doctor's hours genuinely differ by day -- that just
# produces more than one group, each spoken as its own clause. Ordered by
# each group's EARLIEST weekday so a multi-group reply is still spoken
# Monday-first, not in whatever order the hours happened to be seeded.
def _group_schedule_by_hours(schedule: list[dict]) -> list[tuple[str, list[int]]]:
    groups: dict[str, list[int]] = {}
    for entry in schedule:
        hours = f"{entry['start_time']}-{entry['end_time']}"
        groups.setdefault(hours, []).append(entry["weekday"])
    ordered = sorted(groups.items(), key=lambda kv: min(kv[1]))
    return [(hours, sorted(days)) for hours, days in ordered]


# ADDED BY SOURAV -- joins 1+ weekday names naturally ("Monday", "Monday
# and Wednesday", "Monday, Wednesday and Friday"), the same shape
# _spoken_sample_types() above already established for joining 2+ sample
# values. Reuses that function's _SAMPLE_JOIN_WORD map directly rather
# than duplicating a second copy of the same four "and" words -- despite
# its sample-specific name, it is just a language -> "and" lookup, and
# duplicating it here would be the one thing certain to drift the two out
# of sync the next time either needed a fifth language.
def _spoken_weekday_list(weekdays: list[int], language: str = "bengali") -> str:
    names = [weekday_to_words(w, language) for w in weekdays]
    if len(names) == 1:
        return names[0]
    join_word = _SAMPLE_JOIN_WORD.get(language, _SAMPLE_JOIN_WORD["bengali"])
    return f"{', '.join(names[:-1])} {join_word} {names[-1]}"


def doctor_schedule_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    """"Caller asks when a doctor sits" (Epic: Conversation -- Information
    and Enquiry). A caller asking, in general, which days a named doctor
    sits -- with no date mentioned at all -- gets that doctor's full
    recurring weekly schedule read back naturally, in whichever of the 4
    languages this was asked to answer in.

    Deliberately a SEPARATE function from doctor_availability_reply()
    above rather than a new branch inside it: that function's shape is
    built entirely around resolving to ONE day (available today / not
    available, next available date) and asks a follow-up booking
    question tied to that one day ("book today or another day?") -- none
    of which makes sense for a caller who did not name a day at all and
    is not (yet) trying to book anything. Sharing the DoctorSchedule data
    is handled at the clinic-api layer (both endpoints query the same
    table); the two reply shapes stay genuinely distinct here.

    NOT built (flagged, not silently absorbed): unlike
    doctor_availability_reply(), this does not end with a "would you
    like to book an appointment?" follow-up question, and main.py does
    not open a pending state after it -- this intent is purely
    informational. Adding a booking hand-off here would be a reasonable
    follow-up story, not assumed as part of this one.
    """
    if not result.get("found"):
        if language == "english":
            return f"Sorry, we don't have a doctor named '{slots.get('doctor_name')}'."
        elif language == "hinglish":
            return f"Sorry, '{slots.get('doctor_name')}' naam ka doctor yahan nahi hai."
        elif language == "banglish":
            return f"Dukkhito, '{slots.get('doctor_name')}' naam-e kono doctor amader ekhane nei."
        else:  # bengali
            return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার আমাদের এখানে নেই।"

    name = _spoken_doctor_name(slots, result, language=language)
    schedule = result.get("schedule") or []

    if not schedule:
        # Honest, real edge case -- see clinic-api/main.py::doctor_schedule()'s
        # docstring: a doctor can exist with zero DoctorSchedule rows (e.g.
        # on indefinite leave). Never fabricate a sitting day here.
        if language == "english":
            return f"{name} doesn't have a fixed schedule right now. Please check at our counter."
        elif language == "hinglish":
            return f"{name} abhi koi fixed din nahi baithte. Hamare counter mein check kar sakte ho."
        elif language == "banglish":
            return f"{name} ekhon kono nirdishto din boshchen na. Amader counter-e khoj nite paren."
        else:  # bengali
            return f"{name} এখন কোনো নির্দিষ্ট দিন বসছেন না। আমাদের কাউন্টারে খোঁজ নিতে পারেন।"

    # Multiple hour-groups (a doctor whose hours genuinely differ by day --
    # not exercised by today's seed data, see SHIFT_TEMPLATES, but the
    # schema allows it) are joined with a plain ", and "/"আর"-style
    # conjunction, deliberately never a semicolon or any other punctuation
    # a caller would hear as a pause artefact rather than a spoken word.
    groups = _group_schedule_by_hours(schedule)

    if language == "english":
        clauses = [f"on {_spoken_weekday_list(days, language)}, chamber hours {hours}"
                   for hours, days in groups]
        return f"{name} sits {', and '.join(clauses)}."
    elif language == "hinglish":
        clauses = [f"{_spoken_weekday_list(days, language)} ko {hours} baithte hain"
                   for hours, days in groups]
        return f"{name} {', aur '.join(clauses)}."
    elif language == "banglish":
        # "shomoy-e" (not "{hours}-e") deliberately keeps the Bengali
        # locative "-e" suffix attached to a WORD ("shomoy" = "time"),
        # never hyphenated directly onto the raw "HH:MM-HH:MM" digits --
        # verbalize() rewrites that span before synthesis, and a suffix
        # glued straight onto digits it is about to rewrite is exactly
        # the kind of artefact "Answers sound like a person" story 2 was
        # about eliminating.
        clauses = [f"{_spoken_weekday_list(days, language)} {hours} shomoy-e boshen"
                   for hours, days in groups]
        return f"{name} {', ar '.join(clauses)}."
    else:  # bengali
        clauses = [f"{_spoken_weekday_list(days, language)} {hours} সময়ে বসেন"
                   for hours, days in groups]
        return f"{name} {', আর '.join(clauses)}।"


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


# =============================================================================
# ADDED BY SOURAV -- "Caller asks about a health package" combined with
# "Caller asks opening hours, address or directions" (Epic: Conversation --
# Information and Enquiry). Same discipline as every function above: every
# fact spoken here (a price, an address, a set of hours) comes straight
# from clinic-api's response, never invented or guessed by this file or
# the LLM. See agent/llm.py's own comment on VALID_INTENTS for why
# "health_package" carries an OPTIONAL package_name (a caller asking "what
# packages do you have" is a complete, valid question, not an incomplete
# one waiting on a missing_slot_prompt) and why "clinic_info" bundles
# hours/address/directions as one intent narrowed by "info_topic".
# =============================================================================

_PACKAGE_FALLBACK = {
    "bengali": "প্যাকেজ",
    "english": "the package",
    "hinglish": "package",
    "banglish": "package",
}


def _join_natural(items: list[str], language: str) -> str:
    """"a, b and c" -- shared list-joining helper for the health-package
    replies below. Reuses _SAMPLE_JOIN_WORD's per-language "and" word
    (already used by _spoken_sample_types() above for exactly this
    purpose) rather than inventing a second word list."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    join_word = _SAMPLE_JOIN_WORD.get(language, _SAMPLE_JOIN_WORD["bengali"])
    return f"{', '.join(items[:-1])} {join_word} {items[-1]}"


def _spoken_package_name(slots: dict, result: dict, language: str = "bengali") -> str:
    """Mirrors _spoken_test_name()'s own bengali-vs-other split exactly,
    for the same reason: the Bengali TTS tokenizer drops Latin script
    outright (see that function's docstring), so the bengali branch
    prefers the seeded Bengali alias (clinic-api's package_name_bn,
    computed the same way _first_alias_bn() already picks one for a
    LabTest/Doctor); every other language always uses the English
    catalogue name, never the Bengali alias."""
    if language == "bengali":
        return (result.get("package_name_bn")
                 or slots.get("package_name")
                 or result.get("package_name")
                 or _PACKAGE_FALLBACK["bengali"])
    name = slots.get("package_name") or result.get("package_name")
    return name or _PACKAGE_FALLBACK.get(language, _PACKAGE_FALLBACK["english"])


def _spoken_package_tests(result: dict, language: str = "bengali") -> list[str]:
    """Same bengali-vs-other split as _spoken_package_name() above,
    applied per included test: bengali prefers each test's own Bengali
    alias (clinic-api's tests_bn, positionally paired with tests); every
    other language speaks the English catalogue name. Falls back to the
    English name for any position where no Bengali alias was returned,
    rather than silently dropping that test from the spoken list."""
    names_en = result.get("tests") or []
    if language != "bengali":
        return list(names_en)
    names_bn = result.get("tests_bn") or []
    return [(bn or en) for bn, en in zip(names_bn, names_en)] or list(names_en)


def _package_not_found_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    """Mirrors _test_not_found_reply() above exactly, for health packages
    instead of lab tests -- same found=false/did_you_mean shape, same
    "did you mean" phrasing pattern, just a different catalogue."""
    suggestions = result.get("did_you_mean") or []
    query = slots.get("package_name") or result.get("query") or ""
    if language == "english":
        if suggestions:
            return (f"I couldn't find a health package named '{query}'. "
                     f"Did you mean {_join_natural(suggestions, language)}?")
        return f"Sorry, we don't have a health package named '{query}'."
    elif language == "hinglish":
        if suggestions:
            return (f"'{query}' naam ka health package nahi mila. "
                     f"Kya aap kehna chahte the {_join_natural(suggestions, language)}?")
        return f"Sorry, '{query}' naam ka koi health package hamari list mein nahi hai."
    elif language == "banglish":
        if suggestions:
            return (f"'{query}' name-r health package khunje pelam na. "
                     f"Apni ki bolte chaichen {_join_natural(suggestions, language)}?")
        return f"Dukkhito, '{query}' name-r kono health package amader list-e nei."
    else:  # bengali
        if suggestions:
            return (f"'{query}' নামে হেলথ প্যাকেজ খুঁজে পাইনি। "
                     f"আপনি কি বলতে চাইছেন {_join_natural(suggestions, language)}?")
        return f"দুঃখিত, '{query}' নামে কোনো হেলথ প্যাকেজ আমাদের তালিকায় নেই।"


def health_package_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    """"Caller asks about a health package" -- ONE named package's full
    details, from clinic-api's GET /api/v1/health-packages/search. The
    caller already named (or fast_path/the LLM already matched) a
    specific package -- see health_packages_list_reply() just below for
    the companion "what packages do you have" case with none named.

    Price uses the exact same digit-fidelity helper (_digit_faithful_rate)
    test_rate_reply() already established for LabTest.rate_inr -- clinic-
    api's HealthPackage.price_inr is the identical SQLAlchemy Float
    column shape, so it carries the identical trailing-".0" risk, fixed
    the identical way rather than re-solving it.

    `description` (clinic-api's HealthPackage.description) is English-
    only in the database -- there is no Bengali/Hinglish/Banglish
    translation of it anywhere in the seed data. Spoken ONLY in the
    english branch for exactly that reason: injecting untranslated
    English prose into a Bengali/Hinglish/Banglish sentence is the same
    mixed-script risk _spoken_test_name()'s own docstring already
    describes, and for Bengali specifically the TTS tokenizer would
    silently drop most of it rather than mispronounce it (see
    bn_normalize.py's module docstring). Flagged, not silently worked
    around -- a real per-language description would need real translated
    content in the database, the same gap class this codebase's other
    stories have flagged before rather than fabricating a translation.
    """
    if not result.get("found"):
        return _package_not_found_reply(slots, result, language)

    name = _spoken_package_name(slots, result, language)
    price = _digit_faithful_rate(result.get("price_inr"))
    tests = _join_natural(_spoken_package_tests(result, language), language)
    description = (result.get("description") or "").strip()

    if language == "english":
        reply = f"{name} costs {price} rupees."
        if description:
            reply += f" {description}"
        if tests:
            reply += f" It includes {tests}."
        return reply
    elif language == "hinglish":
        reply = f"{name} ka price {price} rupaye hai."
        if tests:
            reply += f" Isme {tests} shamil hain."
        return reply
    elif language == "banglish":
        reply = f"{name}-er price {price} taka."
        if tests:
            reply += f" Ete {tests} include kora ache."
        return reply
    else:  # bengali
        reply = f"{name}-এর মূল্য {price} টাকা।"
        if tests:
            reply += f" এর মধ্যে {tests} আছে।"
        return reply


def health_packages_list_reply(result: dict, language: str = "bengali") -> str:
    """Companion to health_package_reply() above, for the OTHER real
    caller phrasing clinic-api/models.py's own HealthPackage docstring
    gives as the FIRST example: "What health packages do you have?" -- a
    caller who did not name any specific package at all. main.py's
    dispatch calls this (via clinic-api's GET /api/v1/health-packages, no
    name filter) whenever "package_name" was left null, instead of
    re-prompting for a package name the caller never intended to give --
    see agent/llm.py's own comment on VALID_INTENTS for why this is a
    deliberate design choice, not the missing-slot re-prompt every other
    single-entity intent in this file uses.
    """
    packages = result.get("packages") or []
    if not packages:
        # Honest edge case -- every active package was withdrawn, or the
        # catalogue is empty. Never fabricates a package that doesn't exist.
        if language == "english":
            return "Sorry, we don't have any health packages available right now."
        elif language == "hinglish":
            return "Sorry, abhi koi health package available nahi hai."
        elif language == "banglish":
            return "Dukkhito, ekhon kono health package available nei."
        else:  # bengali
            return "দুঃখিত, এই মুহূর্তে কোনো হেলথ প্যাকেজ নেই।"

    entries = []
    for pkg in packages:
        name = pkg.get("package_name_bn") if language == "bengali" else pkg.get("package_name")
        name = name or pkg.get("package_name") or _PACKAGE_FALLBACK.get(language, _PACKAGE_FALLBACK["english"])
        price = _digit_faithful_rate(pkg.get("price_inr"))
        if language == "english":
            entries.append(f"{name} at {price} rupees")
        elif language == "hinglish":
            entries.append(f"{name}, {price} rupaye")
        elif language == "banglish":
            entries.append(f"{name}, {price} taka")
        else:  # bengali
            entries.append(f"{name}, {price} টাকা")

    listing = _join_natural(entries, language)
    if language == "english":
        return f"We have {listing}. Which one would you like to know more about?"
    elif language == "hinglish":
        return f"Hamare paas {listing} hain. Kis package ke baare mein aur jaanna chahenge?"
    elif language == "banglish":
        return f"Amader kache {listing} ache. Kon package ta niye aro janben?"
    else:  # bengali
        return f"আমাদের কাছে {listing} আছে। কোন প্যাকেজ সম্পর্কে আরও জানতে চান?"


# Ordered Monday-first (0=Monday..6=Sunday), matching main.py's own
# `datetime.date.today().weekday()` convention (the same one
# DoctorSchedule/doctor_schedule_reply() already use) and clinic-api/
# main.py's own `_CLINIC_WEEKDAYS` tuple -- duplicated here rather than
# imported, same as _last4()/_mask_phone_last4() above: this file and
# clinic-api are two separate deployables (see agent/tools_client.py's
# module docstring).
_CLINIC_WEEKDAY_KEYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)


def clinic_info_reply(slots: dict, result: dict, language: str = "bengali") -> str:
    """"Caller asks opening hours, address or directions" -- one intent
    backing all three of that story's own bundled phrasings (models.py's
    own ClinicInfo docstring: "When do you open?" / "Where is the
    clinic?" / "Give me directions."), matching the single ClinicInfo
    table backing all three. `slots["info_topic"]` (agent/llm.py's
    extraction -- "hours"/"address"/"directions"/None) narrows which part
    gets spoken; None (caller asked generally, e.g. "tell me about your
    clinic", or asked more than one of the three at once) speaks all
    three together rather than guessing which one to leave out.

    `slots["today_weekday"]` is main.py's own resolved
    `datetime.date.today().weekday()` -- resolving "which day" is main.py's
    job everywhere else in this file (see doctor_availability's own
    date_iso default), so this function only renders whatever day it's
    given rather than importing datetime itself.

    KNOWN, FLAGGED GAP: clinic-api's `address`/`directions` fields are
    English-only free text in the database -- there is no Bengali/
    Hinglish/Banglish translation of either anywhere in the seed data.
    They are still spoken as-is in every language (an address is
    information a caller needs regardless of language, so staying silent
    is worse than speaking it in English), but for the bengali branch
    specifically, the actual Bengali TTS synthesis step will silently
    DROP the Latin-script portions of that text (street names, area
    names -- confirmed directly: `bn_normalize.unspeakable_spans()` flags
    exactly this against the real seeded address/directions strings).
    Fixing this for real would need genuinely translated Bengali-script
    address/directions text seeded in the database -- fabricating one
    here would violate this whole file's "never invent a fact" discipline
    (see module docstring), so it is flagged, not silently worked around.
    """
    if not result.get("found"):
        if language == "english":
            return "Sorry, I don't have our clinic's information available right now. Please contact our counter."
        elif language == "hinglish":
            return "Sorry, abhi clinic ki jaankari available nahi hai. Please hamare counter se contact kariye."
        elif language == "banglish":
            return "Dukkhito, ekhon clinic-er tottho available nei. Please amader counter-e jogajog korun."
        else:  # bengali
            return "দুঃখিত, এই মুহূর্তে ক্লিনিকের তথ্য পাওয়া যাচ্ছে না। দয়া করে কাউন্টারে যোগাযোগ করুন।"

    topic = slots.get("info_topic")
    weekday_idx = slots.get("today_weekday")
    hours = result.get("hours") or {}
    today_key = (
        _CLINIC_WEEKDAY_KEYS[weekday_idx]
        if weekday_idx is not None and 0 <= weekday_idx <= 6
        else None
    )
    today_hours = hours.get(today_key) if today_key else None

    def hours_sentence() -> str:
        if not today_hours:
            # today_weekday wasn't resolved (or is out of range) -- an
            # honest fallback, never guesses a day's hours.
            if language == "english":
                return "Sorry, I don't have today's hours right now."
            elif language == "hinglish":
                return "Sorry, aaj ke hours abhi available nahi hain."
            elif language == "banglish":
                return "Dukkhito, ajker hours ekhon bolte parchi na."
            else:  # bengali
                return "দুঃখিত, আজকের সময়সূচি এখন বলতে পারছি না।"
        if today_hours.get("closed"):
            if language == "english":
                return "We are closed today."
            elif language == "hinglish":
                return "Aaj hum band hain."
            elif language == "banglish":
                return "Aj amra bondho achi."
            else:  # bengali
                return "আজ আমরা বন্ধ আছি।"
        open_, close_ = today_hours.get("open"), today_hours.get("close")
        if language == "english":
            return f"We're open today from {open_} to {close_}."
        elif language == "hinglish":
            return f"Aaj hum {open_} se {close_} tak khule hain."
        elif language == "banglish":
            return f"Aj amra {open_} theke {close_} porjonto khola achi."
        else:  # bengali
            return f"আজ আমরা {open_} থেকে {close_} পর্যন্ত খোলা আছি।"

    def address_sentence() -> str:
        address = result.get("address") or ""
        if language == "english":
            return f"Our address is {address}."
        elif language == "hinglish":
            return f"Hamara address hai {address}."
        elif language == "banglish":
            return f"Amader address {address}."
        else:  # bengali
            return f"আমাদের ঠিকানা হলো {address}।"

    def directions_sentence() -> str:
        directions = result.get("directions") or ""
        if language == "english":
            return f"Here's how to find us. {directions}"
        elif language == "hinglish":
            return f"Humein aise dhundh sakte hain. {directions}"
        elif language == "banglish":
            return f"Amader emon vabe khuje paben. {directions}"
        else:  # bengali
            return f"আমাদের এভাবে খুঁজে পাবেন। {directions}"

    if topic == "hours":
        return hours_sentence()
    if topic == "address":
        return address_sentence()
    if topic == "directions":
        return directions_sentence()

    # No specific topic -- speak all three together, since the caller may
    # have asked more than one of these in the same breath (e.g. "when do
    # you open and what's the address"), and guessing which one to leave
    # out would silently drop half the answer.
    return " ".join([hours_sentence(), address_sentence(), directions_sentence()])


# =============================================================================
# ADDED BY SOURAV -- "Caller asks how to prepare for a test" story's bundled
# human_fallback config (lab_tests_with_fallback_config sample file's
# voice_agent_config.human_fallback block).
#
# The config's own trigger_condition is "query_unresolved_or_low_confidence"
# -- this codebase's one real, already-existing signal for exactly that is
# agent/llm.py's "unclear" intent (see its own docstring for exactly when
# the classifier returns it; there is no separate numeric confidence score
# threaded through to main.py's dispatch to check against, so this does not
# invent one). The config's action is "transfer_to_human_agent" -- but there
# is no telephony transfer capability anywhere in this codebase (no SIP/PSTN
# library, no call-control API of any kind), so an ACTUAL call transfer is
# not something this file can honestly build. See agent/outcomes.py's
# record_human_handoff() for the other, buildable half of that action (an
# escalation-ledger entry a human follow-up process can act on, the same
# pattern already established there for the insufficient-verified-
# information outcome) -- this function is ONLY the spoken half.
# =============================================================================

def human_fallback_reply(language: str = "bengali") -> str:
    """The business's own "connecting you to an expert" script, stored
    verbatim (same "genuine, business-reviewed translation -- store as
    given, don't recompose" discipline as clinic-api/seed.py's
    LAB_TEST_ADVISORIES scripts). No dynamic value of any kind -- always
    exactly one of these four fixed sentences."""
    if language == "english":
        return "I understand. Let me connect you with one of our experts right away."
    elif language == "hinglish":
        return "Acha samjh gaya! Mai aapko humare expert ke saath connect kar deta hoon."
    elif language == "banglish":
        return "Acha, bujhte perechi! Ami apnake amader ekjon expert-er sathe connect kore dichi."
    else:  # bengali
        return "আচ্ছা, বুঝতে পেরেছি! আমি আপনাকে আমাদের একজন এক্সপার্টের সাথে কানেক্ট করে দিচ্ছি।"
