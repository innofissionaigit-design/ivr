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

# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
# Spoken instead of a reply that contains a span the synthesizer would drop
# (agent/speakability.py). Three deliberate choices in one sentence:
#
#  * It admits the agent's own limit rather than blaming the line or the
#    caller. The failure is a missing spoken form on this side.
#  * It offers the counter, matching the repair ladder main.py already uses
#    after repeated low-agreement turns. It does NOT say "স্টাফের কাছে দিচ্ছি"
#    the way the tool_failure clip does -- that claims a live transfer this
#    system does not implement, and a promise the agent cannot keep is a
#    worse outcome than the hole it replaced.
#  * It is pure Bengali, so it can never itself trip the gate it serves.
#    main.py asserts exactly that at startup; see _assert_canned_lines_speakable.
UNSPEAKABLE_ESCALATION = (
    "দুঃখিত, এই তথ্যটা আমি ঠিকভাবে বলে উঠতে পারছি না। "
    "কাউন্টারে একবার কথা বলে নিলে ভালো হয়।"
)


# story title: Answers sound like a person, not a database row
# user story: As a patient, I want to hear a sentence, so that the agent
#   sounds like someone at the counter.
# acceptance criteria: No field label, colon or bracket is ever spoken and
#   every structured value renders as a natural clause in the reply
#   language. An automated check fails a build containing a spoken
#   punctuation artefact.
#
# A list a person would say out loud: "A, B নাকি C", not "A, B, C". The last
# item gets the disjunction because every place this is used is offering the
# caller a choice, and a bare comma before the final option reads as a
# database row being recited rather than a question being asked.
# story title: The agent says it cannot confirm rather than guessing
# user story: As a caller, I want to be told plainly when the system
#   cannot verify something, so that I am not given a confident guess.
# acceptance criteria: The insufficient-verified-information outcome has
#   its own template per language, its own metric and its own escalation
#   path, distinct from not-found and from an infrastructure apology. Its
#   rate is reported per intent because a rise means a data or
#   integration problem.
#
# ITS OWN TEMPLATE, and it lives HERE rather than in agent/outcomes.py as it
# does on dev_sourav. Every spoken sentence in this branch is in this file, and
# two gates scan it -- tests/test_spoken_punctuation.py and
# tests/test_fact_provenance.py. A sentence defined anywhere else is a sentence
# neither gate can see, which is how a spoken colon or a fabricated fact gets
# back in.
#
# It must differ from BOTH of the other two outcomes, and the difference is
# what the caller is told to DO:
#   not-found     -> the thing does not exist. Stop asking.
#   unreachable   -> could not check. Call back.
#   this          -> it IS done, the agent just cannot read the details back.
#                    Do NOT rebook. Someone will follow up.
# Saying "call back" here would invite a duplicate booking of an appointment
# that already exists, which is the specific harm this outcome prevents.
INSUFFICIENT_VERIFIED_INFORMATION_BN = (
    "আপনার অ্যাপয়েন্টমেন্টটা হয়ে গেছে, কিন্তু বিস্তারিত তথ্যগুলো এই মুহূর্তে মিলিয়ে দেখতে পারছি না। আমাদের থেকে আপনাকে জানানো হবে। আবার বুক করার দরকার নেই।"
)


# story title: The same question gets the same answer within one call
# user story: As a caller who asks twice, I want the same answer, so that I
#   know which one to believe.
# acceptance criteria: Repeating a question in one call produces an identical
#   factual answer unless the underlying data changed, in which case the
#   change is stated. A test asserts consistency across three repeats with an
#   unchanged backend.
#
# THE CHANGE IS STATED -- and it deliberately does not restate the old value.
#
# The tempting version names both figures: more informative on paper, and
# worse on a phone line. A caller who catches only half of it has then been
# read a number that is no longer true, by the one sentence whose entire job
# is to stop them believing a stale figure. A change statement should contain
# exactly one number -- the current one -- and the freshly rendered reply that
# follows carries it.
#
# It therefore states no fact of its own and needs no verified response to
# render from, which is why it is a bare constant rather than a function
# taking the previous answer.
ANSWER_CHANGED_BN = (
    "একটু আগে আমি অন্য তথ্য বলেছিলাম, এইমাত্র দেখে নিলাম সেটা বদলে গেছে।"
)


# story title: Near matches are offered rather than guessed or refused
# user story: As a caller naming something loosely, I want the close matches
#   offered, so that I am not told my test does not exist when it does.
# acceptance criteria: When several catalogue rows fall within the match band
#   the agent offers up to three by name and asks which. Candidates are
#   generated across every supported language and romanised spelling. The
#   did-you-mean path covers the ambiguous case and not only total failure.
#
# Said when the caller named something that matches several catalogue rows,
# or one row not clearly enough to act on. Used by all three read intents, so
# an ambiguous test, doctor and department are asked about in the same voice.
#
# The fallback is a RE-ASK, not an apology and not a not-found. An empty
# candidate list means the clinic found several rows and at least one of them
# has no Bengali alias -- clinic-api refuses to offer a partial list, because
# dropping a candidate turns "which of these two" back into "did you mean
# this one", which is a guess wearing a question mark. Telling the caller it
# does not exist would be false; asking them to say it again is true.
NEAR_MATCH_UNCLEAR_BN = (
    "দুঃখিত, ঠিক কোনটার কথা বলছেন বুঝতে পারিনি। একটু পরিষ্কার করে নামটা বলবেন?"
)


def near_match_prompt(candidates) -> str:
    """Offer up to three named candidates and ask which one.

    Reads the SPOKEN name, never the catalogue's English label: the Bengali
    tokenizer drops Latin script, and a question offering two silences is
    worse than no question. clinic-api caps the list at three; this does not
    re-cap it, so a cap change lives in one place.
    """
    spoken = [c.get("name_bn") for c in (candidates or []) if c.get("name_bn")]
    if not spoken:
        return NEAR_MATCH_UNCLEAR_BN
    return f"আপনি কি {_spoken_list(spoken)} বলতে চাইছেন?"


# story title: A multi-part question is answered in full
# user story: As a caller who asked two things, I want both answered, so that
#   I do not have to ask again.
# acceptance criteria: Every answerable part of a turn is answered in the
#   order asked, and any part that cannot be answered is explicitly addressed
#   rather than dropped. Completeness is scored on a labelled multi-part set.
#
# THE "EXPLICITLY ADDRESSED RATHER THAN DROPPED" HALF, in three sentences.
#
# None of them counts an ordinal. "আপনার দ্বিতীয় প্রশ্ন" is wrong the moment
# the parse is off by one -- and being off by one is exactly the state the
# agent is in when it is apologising for not understanding something. Where
# the part named a thing, the thing is said; where it did not, the sentence
# stays general rather than claiming a position in a list it may have
# miscounted.
#
# The boundary with E12-S5 (dead ends always offer a next step, ABSENT) is
# worth keeping visible: these say "I heard it and could not serve it". What
# to do instead is that story's job, and folding it in here would build half
# of it badly.
UNANSWERED_PART_BN = "আপনার আরেকটা প্রশ্ন ছিল, সেটা ঠিক বুঝতে পারিনি। একটু বলবেন?"

DEFERRED_PART_BN = "আপনার আরেকটা প্রশ্নও ছিল, সেটা এক্ষুনি দেখে বলছি।"

RESUMING_PART_BN = "এবার আপনার অন্য প্রশ্নটা।"


def unanswered_part_prompt(subject: str | None = None) -> str:
    """Said about a part that was heard and cannot be served.

    `subject` is the caller's OWN words for the thing, when the part named
    one -- echoed rather than translated, the same choice _spoken_test_name
    makes, and for the same reason: the catalogue's English label would be
    dropped by the tokenizer and the caller would hear a sentence with a hole
    where the subject belongs.
    """
    if not subject:
        return UNANSWERED_PART_BN
    return f"'{subject}' নিয়ে আপনার প্রশ্নটার উত্তর এই মুহূর্তে দিতে পারছি না।"


def with_change_notice(reply: str) -> str:
    """Prefix a freshly rendered reply with the change statement.

    A function rather than an f-string at the call site, so the two can never
    be joined without the space, and so the punctuation and speakability gates
    see the joined sentence exactly as the caller will hear it.
    """
    return f"{ANSWER_CHANGED_BN} {reply}"


def _spoken_list(items) -> str:
    items = [str(i) for i in items if i]
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])} নাকি {items[-1]}"


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


# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
def _spoken_department(slots: dict, result: dict) -> str:
    """Same problem as the two above, and it was the worst of the three.

    clinic-api returns `department` as the English column ("Cardiology"), and
    the listing template puts it at the head of the sentence -- so the Bengali
    tokenizer dropped the SUBJECT and the caller heard "...বিভাগে ডাঃ সেন
    আছেন", a sentence with a hole where the thing they asked about should be.
    That was true of all eight seeded departments, not an edge case.

    Prefer the seeded Bengali alias; failing that, echo the caller's own words,
    which is what a person at the counter would do.
    """
    return (result.get("department_bn")
            or slots.get("department")
            or result.get("department")
            or "এই বিভাগে")


def missing_slot_prompt(intent: str, missing: str) -> str:
    prompts = {
        ("test_rate", "test_name"): "কোন টেস্টের রেট জানতে চান, একটু বলবেন?",
        ("doctor_availability", "doctor_name"): "কোন ডাক্তারের কথা জিজ্ঞেস করছেন?",
        ("doctors_by_department", "department"): "কোন বিভাগের ডাক্তার খুঁজছেন?",
        ("doctors_by_department", "date"): "কোন দিনের জন্য জানতে চান, একটু বলবেন?",
        # story title: The model never originates a fact
        # user story: As a clinical lead, I want every price, date and identifier
        #   to come from a verified system response, so that a wrong answer is a
        #   data bug rather than a model bug.
        # acceptance criteria: Every factual sentence is a template substitution
        #   from a validated tool response and the model is never shown a figure
        #   it could restate. An automated assertion on every commit proves no
        #   model-composed span reaches synthesis on a factual intent.
        #
        # Asked when the caller named a day the local parser could not
        # resolve. Before this story that turn silently used the model's
        # guess; now it asks, and this is the sentence it asks with. It
        # existed nowhere, so the turn fell through to the generic
        # "একটু স্পষ্ট করে বলবেন?" which does not name what is unclear.
        ("doctor_availability", "date"): "কোন দিনের কথা বলছেন, একটু বলবেন?",
        ("book_appointment", "doctor_name"): "কোন ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট করতে চান?",
        ("book_appointment", "date"): "আজকের জন্য চান, নাকি অন্য কোনো দিনের জন্য অ্যাপয়েন্টমেন্ট চাই?",
        ("book_appointment", "time_slot"): "কোন সময়ে অ্যাপয়েন্টমেন্ট চাই, একটু বলবেন?",
        ("book_appointment", "patient_name"): "রোগীর নামটা বলবেন?",
        ("book_appointment", "phone"): "একটা ফোন নম্বর দেবেন, যাতে কনফার্মেশন পাঠাতে পারি?",
    }
    return prompts.get((intent, missing), "দুঃখিত, একটু স্পষ্ট করে বলবেন?")


def test_rate_reply(slots: dict, result: dict) -> str:
    if not result.get("found"):
        # STORY [Answer Quality and Grounding]
        # As a patient, I want to hear the whole sentence, so that I am
        # not left guessing what the agent tried to say.
        # The SPOKEN suggestions, not the catalogue's English labels. Reading
        # `did_you_mean` aloud produced "আপনি কি বলতে চাইছেন:" followed by
        # silence -- the whole point of the list gone, on the one path where
        # the caller has already failed once and most needs the words.
        #
        # An empty spoken list falls through to the plain not-found line rather
        # than to the English one: a suggestion nobody can hear is not a
        # suggestion, and offering it would be the same hole with extra steps.
        suggestions = result.get("did_you_mean_bn") or []
        if suggestions:
            # story title: Answers sound like a person, not a database row
            # user story: As a patient, I want to hear a sentence, so that the
            #   agent sounds like someone at the counter.
            # acceptance criteria: No field label, colon or bracket is ever
            #   spoken and every structured value renders as a natural clause
            #   in the reply language. An automated check fails a build
            #   containing a spoken punctuation artefact.
            #
            # Was "আপনি কি বলতে চাইছেন: X, Y?" -- a colon read aloud, and a
            # comma-separated list where a person would say "or". The options
            # now sit inside the question rather than being announced by it.
            return (f"'{slots.get('test_name')}' নামে টেস্ট খুঁজে পাইনি। "
                     f"আপনি কি {_spoken_list(suggestions)} বলতে চাইছেন?")
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
        reply += f" {sample} স্যাম্পল দিতে হবে।"
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
        return (f"হ্যাঁ,{date_txt} {name} {hours} চেম্বারে থাকবেন। "
                f"আজকের জন্যই অ্যাপয়েন্টমেন্ট করবেন, নাকি অন্য কোনো দিনের জন্য?")

    next_date = result.get("next_available_date")
    if next_date:
        return (f"{name} ওই দিন বসবেন না। পরবর্তী উপলব্ধ দিন হলো {next_date}। "
                f"ওই দিনের জন্য অ্যাপয়েন্টমেন্ট করতে চান?")
    return f"{name} এখন কোনো নির্দিষ্ট দিন বসছেন না। আমাদের কাউন্টারে খোঁজ নিতে পারেন।"


def heard_confirm_prompt(transcript: str) -> str:
    """Echo back what the recogniser produced, when the two decoders did not
    agree enough to act on it.

    The caller's OWN words are read back verbatim -- not a paraphrase and not
    a cleaned-up version. If the transcript is wrong, hearing it wrong is
    exactly what lets the caller say না; smoothing it over would hide the
    error this prompt exists to surface.
    """
    return f"আমি শুনলাম — {transcript}। ঠিক বলেছি?"


def date_range_confirm_prompt(start_iso: str, end_iso: str) -> str:
    """Read a CALCULATED date range back before answering about it.

    story title: The model never originates a fact
    user story: As a clinical lead, I want every price, date and identifier to
        come from a verified system response, so that a wrong answer is a data
        bug rather than a model bug.
    acceptance criteria: Every factual sentence is a template substitution from
        a validated tool response and the model is never shown a figure it
        could restate. An automated assertion on every commit proves no
        model-composed span reaches synthesis on a factual intent.

    Spoken when the caller said something like "আগামী সপ্তাহে" -- the model
    said which expression that was, agent/date_calc.py worked out which seven
    days it covers, and the agent cannot answer "is Dr Sen in?" about seven
    days at once. So it says which seven it means.

    THE DATES IN THIS SENTENCE ARE SAFE TO SPEAK, and the reason is the whole
    architecture in one line: they were computed by date_calc, not produced by
    the model. A date the model invented could not be read out even inside a
    question -- asking "did you mean the 14th to the 20th?" asserts that those
    are the dates of next week, which is a fact, and facts do not come from the
    model. Here they came from the calendar.

    bn_normalize.verbalize() spells both dates into Bengali words before
    synthesis, so the caller hears "সেপ্টেম্বর মাসের চোদ্দো তারিখ" rather than
    a string of Latin digits the tokenizer would silently drop.
    """
    return (f"আপনি কি {start_iso} থেকে {end_iso} — এই সময়ের মধ্যে "
            f"জানতে চাইছেন?")


def booking_confirm_prompt(slots: dict) -> str:
    """Read the whole booking back before it is committed.

    This is the last point at which a mishearing is still free to correct. Every
    value in it came from the caller, so nothing here is a fact the agent is
    originating -- it is the same template-substitution discipline the rest of
    this module applies to API responses, pointed at the caller's own words.

    Deliberately NOT handed to the LLM to phrase more warmly: that would put a
    date and a phone number back into generated text, which is precisely what
    this file exists to prevent. bn_normalize.verbalize() already reads the
    phone number digit-by-digit and spells the date into Bengali words, so the
    caller hears it the way a person would say it back.
    """
    # story title: The model never originates a fact
    # user story: As a clinical lead, I want every price, date and identifier
    #   to come from a verified system response, so that a wrong answer is a
    #   data bug rather than a model bug.
    # acceptance criteria: Every factual sentence is a template substitution
    #   from a validated tool response and the model is never shown a figure
    #   it could restate. An automated assertion on every commit proves no
    #   model-composed span reaches synthesis on a factual intent.
    #
    # doctor_name_bn first, and this is a bug fix, not a preference. On the
    # commonest booking route -- list a department's doctors, caller picks
    # one, book -- main.py stores the CANONICAL English name, because that is
    # what the booking API matches on. This sentence is SPOKEN, so reading
    # that field put "ডাঃ Dr. A Sen" into a Bengali utterance: the Bengali
    # tokenizer drops Latin script, so the caller was asked to confirm a
    # booking with the doctor's name missing from it, and after the
    # speakability gate landed the whole confirmation was blocked and the
    # caller sent to the counter mid-booking.
    #
    # Both forms are now carried through the flow: the English one goes to
    # the API, the Bengali one is said out loud. Where a doctor row has no
    # Bengali alias this still falls back to the English label and the
    # speakability gate refuses the reply -- which is correct. A missing
    # alias is a data defect, and confirming a booking against a name the
    # caller cannot hear is worse than escalating.
    doctor = slots.get("doctor_name_bn") or slots.get("doctor_name")
    return (f"একটু মিলিয়ে নিই। "
            f"রোগী {slots['patient_name']}, ডাঃ {doctor}, "
            f"{slots['date']} তারিখে, সময় {slots['time_slot']}, "
            f"ফোন {slots['phone']}। সব ঠিক আছে?")


# story title: Every critical value is read back before it is used
# user story: As a patient giving a phone number, I want it read back, so that
#   a misheard digit does not send my report to a stranger.
# acceptance criteria: Phone numbers, dates, times and names are confirmed
#   aloud before any write, and a rejection opens a correction path rather than
#   repeating the prompt. Readback is mandatory regardless of confidence for
#   values that affect a write.
def booking_correction_prompt() -> str:
    """Asked when the caller rejects the readback.

    This is the clause the previous implementation failed outright. It read
    all five values back and required a হ্যাঁ, which satisfied the first half
    of the criterion -- but a rejection re-asked "শুধু বলুন — হ্যাঁ, নাকি না?"
    twice and then abandoned the booking. That is literally "repeating the
    prompt", which the criterion names as the thing not to do, and it is a
    poor experience besides: the caller has just told the agent something is
    wrong and the agent's answer is to ask the same question again, then hang
    up on the booking.

    So this is a DIFFERENT question, not a re-read: which one value is wrong.
    Naming the five options aloud matters on a phone line -- an open "কী ভুল?"
    invites a sentence the parser will not resolve, while a menu gets a
    one-word answer that agent/slot_parse.parse_correction_field() can map.

    Ported from dev_sourav, Bengali only; that branch carries the same prompt
    in four languages, which belongs to a multilingual story this one is not.
    """
    return "ঠিক আছে, কোনটা ঠিক করে দেব — ডাক্তার, তারিখ, সময়, নাম, নাকি ফোন নম্বর?"


def booking_reply(slots: dict, result: dict) -> str:
    if result.get("success"):
        return (f"আপনার অ্যাপয়েন্টমেন্ট কনফার্ম হয়েছে। "
                f"{_spoken_doctor_name(slots, result)}, {result['date']}, সময় {result['time_slot']}। "
                f"আপনার কনফার্মেশন নম্বর হলো {result['confirmation_id']}।")

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
        # story title: Answers sound like a person, not a database row
        # user story: As a patient, I want to hear a sentence, so that the
        #   agent sounds like someone at the counter.
        # acceptance criteria: No field label, colon or bracket is ever
        #   spoken and every structured value renders as a natural clause
        #   in the reply language. An automated check fails a build
        #   containing a spoken punctuation artefact.
            return (f"ওই সময়টা বুক হয়ে গেছে, তবে {_spoken_list(alts)} "
                    f"সময়গুলো ফাঁকা আছে। কোনটা চান?")
        return "ওই সময়টা বুক হয়ে গেছে, এবং কাছাকাছি কোনো সময় ফাঁকা নেই।"
    if reason == "doctor_not_found":
        return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার খুঁজে পেলাম না।"
    return "দুঃখিত, অ্যাপয়েন্টমেন্ট বুক করা গেল না। একটু পরে আবার চেষ্টা করুন, অথবা কাউন্টারে যোগাযোগ করুন।"


def doctors_by_department_reply(slots: dict, result: dict) -> str:
    if not result.get("found"):
        return f"দুঃখিত, '{slots.get('department')}' নামে কোনো বিভাগ আমাদের এখানে নেই।"

    department = _spoken_department(slots, result)
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
            # STORY [Answer Quality and Grounding]
            # As a patient, I want to hear the whole sentence, so that I am
            # not left guessing what the agent tried to say.
            # Deliberately left as the English label rather than dropped.
            # A doctor seeded without a Bengali alias is a DATA defect, and
            # the two honest responses to it are to fix the row or to refuse
            # the reply -- silently omitting the doctor from a list the caller
            # is choosing from would be neither. The speakability gate turns
            # this into an escalation, which is loud and countable; skipping
            # the name would be quiet and permanent.
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
