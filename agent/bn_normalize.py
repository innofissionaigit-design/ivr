"""Verbalize a reply string into something the Bengali TTS can actually SAY.

This is not a cosmetic prettifier -- it fixes a proven, silent data-loss
bug. AI4Bharat's Bengali FastPitch tokenizer drops Latin digits entirely.
Measured on the live pod (22050Hz mono 16-bit, so bytes/2/22050 = seconds):

    "রেট টাকা।"            -> 102476 bytes   (no number at all)
    "রেট 250 টাকা।"        -> 100940 bytes   <- identical...
    "রেট 987654 টাকা।"     -> 100940 bytes   <- ...to a 6-digit number
    "রেট দুইশো পঞ্চাশ টাকা।" -> 143948 bytes   (spelled out: actually spoken)

Two different numbers producing the same audio, shorter than the sentence
with the number removed, is conclusive: every price, report time, chamber
hour and confirmation number this agent has ever "spoken" was silence.
Callers heard "রেট ___ টাকা" and reported it as the agent skipping words.

So every number reaching TTS gets spelled into Bengali words HERE, in
code, before synthesis. Deliberately not asked of the LLM: the model is
never allowed to restate a figure (see llm.py's module docstring), and
that rule does not get to be quietly relaxed just because the figure
needs reformatting.

LANGUAGE SUPPORT:
- Primary: Bengali (বাংলা) - for TTS synthesis
- Secondary: Hinglish support - for mixed-language contexts
- The system handles Bengali, English, and Hinglish (Hindi-English mix)
while maintaining digit-faithful number verbalization.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- numbers

_ONES_TO_99 = [
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়",
    "দশ", "এগারো", "বারো", "তেরো", "চোদ্দো", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ",
    "কুড়ি", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আটাশ", "ঊনত্রিশ",
    "ত্রিশ", "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ", "ঊনচল্লিশ",
    "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ", "ঊনপঞ্চাশ",
    "পঞ্চাশ", "একান্ন", "বাহান্ন", "তিপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "ঊনষাট",
    "ষাট", "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি", "সাতষট্টি", "আটষট্টি", "ঊনসত্তর",
    "সত্তর", "একাত্তর", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "ঊনআশি",
    "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "অষ্টাশি", "ঊননব্বই",
    "নব্বই", "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
]

_HUNDREDS = [
    "", "একশো", "দুইশো", "তিনশো", "চারশো", "পাঁচশো", "ছয়শো", "সাতশো", "আটশো", "নয়শো",
]

# Bengali digit glyphs -> ASCII, so ২৫০ and 250 take the same path.
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# ---------------------------------------------------------------- Hinglish support

# English number words for Hinglish context (digit-faithful conversion)
_ENGLISH_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven", "twenty-eight", "twenty-nine",
    "thirty", "thirty-one", "thirty-two", "thirty-three", "thirty-four", "thirty-five", "thirty-six", "thirty-seven", "thirty-eight", "thirty-nine",
    "forty", "forty-one", "forty-two", "forty-three", "forty-four", "forty-five", "forty-six", "forty-seven", "forty-eight", "forty-nine",
    "fifty", "fifty-one", "fifty-two", "fifty-three", "fifty-four", "fifty-five", "fifty-six", "fifty-seven", "fifty-eight", "fifty-nine",
    "sixty", "sixty-one", "sixty-two", "sixty-three", "sixty-four", "sixty-five", "sixty-six", "sixty-seven", "sixty-eight", "sixty-nine",
    "seventy", "seventy-one", "seventy-two", "seventy-three", "seventy-four", "seventy-five", "seventy-six", "seventy-seven", "seventy-eight", "seventy-nine",
    "eighty", "eighty-one", "eighty-two", "eighty-three", "eighty-four", "eighty-five", "eighty-six", "eighty-seven", "eighty-eight", "eighty-nine",
    "ninety", "ninety-one", "ninety-two", "ninety-three", "ninety-four", "ninety-five", "ninety-six", "ninety-seven", "ninety-eight", "ninety-nine",
]

_ENGLISH_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
]

_ENGLISH_HUNDREDS = [
    "", "one hundred", "two hundred", "three hundred", "four hundred", "five hundred",
    "six hundred", "seven hundred", "eight hundred", "nine hundred",
]


def number_to_bn_words(n: int) -> str:
    """Indian numbering system (হাজার / লাখ / কোটি), not the short scale."""
    if n < 0:
        return "মাইনাস " + number_to_bn_words(-n)
    if n < 100:
        return _ONES_TO_99[n]
    if n < 1000:
        head, rest = divmod(n, 100)
        out = _HUNDREDS[head]
        return out if rest == 0 else f"{out} {number_to_bn_words(rest)}"
    for divisor, word in ((10_000_000, "কোটি"), (100_000, "লাখ"), (1_000, "হাজার")):
        if n >= divisor:
            head, rest = divmod(n, divisor)
            out = f"{number_to_bn_words(head)} {word}"
            return out if rest == 0 else f"{out} {number_to_bn_words(rest)}"
    return str(n)  # unreachable


def number_to_english_words(n: int) -> str:
    """Convert number to English words for Hinglish support (digit-faithful)."""
    if n < 0:
        return "minus " + number_to_english_words(-n)
    if n < 100:
        return _ENGLISH_ONES[n]
    if n < 1000:
        head, rest = divmod(n, 100)
        out = _ENGLISH_HUNDREDS[head]
        return out if rest == 0 else f"{out} and {number_to_english_words(rest)}"
    # Use Indian numbering system for consistency with Bengali
    for divisor, word in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        if n >= divisor:
            head, rest = divmod(n, divisor)
            out = f"{number_to_english_words(head)} {word}"
            return out if rest == 0 else f"{out} {number_to_english_words(rest)}"
    return str(n)  # unreachable


def number_to_hinglish_words(n: int) -> str:
    """Convert number to Hinglish (Hindi-English mix) words.
    
    Hinglish commonly uses Hindi number words for small numbers (1-10)
    and English for larger numbers, or mixes them naturally.
    This provides a digit-faithful conversion that sounds natural in
    Hinglish contexts while preserving exact values.
    """
    # Common Hindi numbers used in Hinglish
    _HINDI_SMALL = {
        0: "zero", 1: "ek", 2: "do", 3: "teen", 4: "chaar",
        5: "paanch", 6: "chhe", 7: "saat", 8: "aath", 9: "nau",
        10: "das", 20: "bees", 30: "tees", 40: "chaalis", 50: "pachaas",
        60: "saath", 70: "sattar", 80: "assi", 90: "nabbe", 100: "ek sau"
    }
    
    if n < 0:
        return "minus " + number_to_hinglish_words(-n)
    
    # Use Hindi words for small numbers (common in Hinglish)
    if n in _HINDI_SMALL:
        return _HINDI_SMALL[n]
    
    # For larger numbers, use English with Indian system
    if n < 1000:
        return number_to_english_words(n)
    
    # Use Indian numbering system with English words
    for divisor, word in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        if n >= divisor:
            head, rest = divmod(n, divisor)
            out = f"{number_to_hinglish_words(head)} {word}"
            return out if rest == 0 else f"{out} {number_to_hinglish_words(rest)}"
    
    return str(n)


# Latin letters read aloud in Bengali. Confirmation IDs ("KCD-4471") are
# the only place these reach TTS, and without this the letters vanish the
# same way the digits did -- the caller hears four digits and no prefix.
_LETTER_BN = {
    "a": "এ", "b": "বি", "c": "সি", "d": "ডি", "e": "ই", "f": "এফ", "g": "জি",
    "h": "এইচ", "i": "আই", "j": "জে", "k": "কে", "l": "এল", "m": "এম",
    "n": "এন", "o": "ও", "p": "পি", "q": "কিউ", "r": "আর", "s": "এস",
    "t": "টি", "u": "ইউ", "v": "ভি", "w": "ডব্লিউ", "x": "এক্স", "y": "ওয়াই", "z": "জেড",
}


def spell_out(s: str) -> str:
    """Character-by-character, the way an ID is read over a phone."""
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(_ONES_TO_99[int(ch)])
        elif ch.isalpha() and ch.lower() in _LETTER_BN:
            out.append(_LETTER_BN[ch.lower()])
    return " ".join(out)


def digits_one_by_one(s: str) -> str:
    """Phone numbers and confirmation IDs are read digit-by-digit, the way
    a person reads them aloud -- "নয় আট সাত" not "নয়শো সাতাশি"."""
    return " ".join(_ONES_TO_99[int(c)] if c.isdigit() else c for c in s if not c.isspace())


# ------------------------------------------------------------------ time

_HOUR_WORD = {
    1: "একটা", 2: "দুটো", 3: "তিনটে", 4: "চারটে", 5: "পাঁচটা", 6: "ছটা",
    7: "সাতটা", 8: "আটটা", 9: "নটা", 10: "দশটা", 11: "এগারোটা", 12: "বারোটা",
}


def _day_part(hour24: int) -> str:
    if 4 <= hour24 < 12:
        return "সকাল"
    if 12 <= hour24 < 16:
        return "দুপুর"
    if 16 <= hour24 < 18:
        return "বিকেল"
    if 18 <= hour24 < 20:
        return "সন্ধ্যা"
    return "রাত"


def time_to_bn_words(hh: int, mm: int) -> str:
    """Bengali speakers say সাড়ে/সোয়া/পৌনে for :30/:15/:45 -- reading
    "দশটা ত্রিশ মিনিট" instead is understandable but immediately marks the
    voice as a machine."""
    part = _day_part(hh)
    h12 = hh % 12 or 12
    if mm == 0:
        return f"{part} {_HOUR_WORD[h12]}"
    if mm == 30:
        return f"{part} সাড়ে {_HOUR_WORD[h12]}"
    if mm == 15:
        return f"{part} সোয়া {_HOUR_WORD[h12]}"
    if mm == 45:
        nxt = (h12 % 12) + 1
        return f"{_day_part((hh + 1) % 24)} পৌনে {_HOUR_WORD[nxt]}"
    return f"{part} {_HOUR_WORD[h12]} বেজে {number_to_bn_words(mm)} মিনিট"


def time_to_english_words(hh: int, mm: int) -> str:
    """Convert time to English words for Hinglish support."""
    h12 = hh % 12 or 12
    period = "AM" if hh < 12 else "PM"
    
    if mm == 0:
        return f"{_ENGLISH_ONES[h12]} o'clock {period}"
    if mm == 30:
        return f"half past {_ENGLISH_ONES[h12]} {period}"
    if mm == 15:
        return f"quarter past {_ENGLISH_ONES[h12]} {period}"
    if mm == 45:
        nxt = (h12 % 12) + 1
        return f"quarter to {_ENGLISH_ONES[nxt]} {period}"
    return f"{_ENGLISH_ONES[h12]} {number_to_english_words(mm)} {period}"


def time_to_hinglish_words(hh: int, mm: int) -> str:
    """Convert time to Hinglish (Hindi-English mix) words.
    
    Hinglish speakers often use Hindi time words like "baje" for hours
    and English for minutes, or mix them naturally.
    """
    h12 = hh % 12 or 12
    # Hindi number words for hours (common in Hinglish)
    _HINDI_HOURS = {
        1: "ek", 2: "do", 3: "teen", 4: "chaar", 5: "paanch",
        6: "chhe", 7: "saat", 8: "aath", 9: "nau", 10: "das", 11: "gyaarah", 12: "baarah"
    }
    
    period = "morning" if 4 <= hh < 12 else ("afternoon" if 12 <= hh < 16 else ("evening" if 16 <= hh < 20 else "night"))
    
    if mm == 0:
        return f"{_HINDI_HOURS[h12]} baje {period}"
    if mm == 30:
        return f"{_HINDI_HOURS[h12]} baje {number_to_hinglish_words(mm)} {period}"
    if mm == 15:
        return f"{_HINDI_HOURS[h12]} baje {number_to_hinglish_words(mm)} {period}"
    if mm == 45:
        nxt = (h12 % 12) + 1
        return f"{_HINDI_HOURS[nxt]} baje {number_to_hinglish_words(60 - mm)} {period}"
    return f"{_HINDI_HOURS[h12]} baje {number_to_hinglish_words(mm)} {period}"


# ------------------------------------------------------------------ date

_MONTHS_BN = [
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
]


def date_to_bn_words(y: int, m: int, d: int) -> str:
    if not 1 <= m <= 12:
        return f"{number_to_bn_words(d)} তারিখ"
    return f"{_MONTHS_BN[m - 1]} মাসের {number_to_bn_words(d)} তারিখ"


_MONTHS_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def date_to_english_words(y: int, m: int, d: int) -> str:
    """Convert date to English words for Hinglish support."""
    if not 1 <= m <= 12:
        return f"{number_to_english_words(d)} date"
    return f"{_MONTHS_EN[m - 1]} {number_to_english_words(d)}, {number_to_english_words(y)}"


_MONTHS_HI = [
    "Janvari", "Febrari", "March", "April", "May", "June",
    "Julai", "August", "September", "October", "November", "December",
]


def date_to_hinglish_words(y: int, m: int, d: int) -> str:
    """Convert date to Hinglish (Hindi-English mix) words.
    
    Hinglish speakers often use English month names with Hindi grammar
    or mixed date formats.
    """
    if not 1 <= m <= 12:
        return f"{number_to_hinglish_words(d)} tarikh"
    return f"{_MONTHS_HI[m - 1]} {number_to_hinglish_words(d)}, {number_to_hinglish_words(y)}"


# ------------------------------------------------------------- the pass

_RE_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b(\s*তারিখে?)?")
_RE_TIME_RANGE = re.compile(r"\b(\d{1,2}):(\d{2})\s*[-–—to]{1,2}\s*(\d{1,2}):(\d{2})\b")
_RE_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_RE_PHONE = re.compile(r"\b(\d{10,})\b")
_RE_CONF_ID = re.compile(r"\b([A-Z]{2,}-?\d{3,}(?:-\d+)*)\b")
_RE_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_RE_INT = re.compile(r"\d+")

# Latin fragments that survive in clinic data (test names like "Uric Acid",
# "CBC", sample types like "Blood"). The Bengali tokenizer drops these the
# same way it drops digits, so anything still in Latin script after
# verbalization is a word the caller will never hear. The lookup service
# owns the Bengali aliases (clinic-api seeds aliases_bn); this table is the
# last-resort spoken form for the handful of fields the API returns in
# English regardless of how the caller phrased the question.
_LATIN_SPOKEN_BN = {
    "blood": "রক্ত",
    "urine": "মূত্র",
    "stool": "মল",
    "serum": "সিরাম",
    "saliva": "লালা",
    "swab": "সোয়াব",
    "plasma": "প্লাজমা",
}


def _sub_int(match: re.Match) -> str:
    return number_to_bn_words(int(match.group(0)))


def verbalize(text: str, language: str = "bengali") -> str:
    """Rewrite `text` so every token in it is actually pronounceable by the
    Bengali FastPitch model. Order matters: the most specific patterns
    (dates, time ranges, long digit runs) must run before the bare-integer
    sweep, or "2026-08-25" gets read as three unrelated numbers.
    
    Args:
        text: The text to verbalize
        language: Target language - "bengali" (default), "english", or "hinglish"
    
    Returns:
        Verbalized text with numbers converted to words in the target language
    """
    if not text:
        return text

    # Select appropriate functions based on language
    if language == "english":
        number_func = number_to_english_words
        date_func = date_to_english_words
        time_func = time_to_english_words
        currency_word = " rupees "
        percent_word = " percent "
    elif language == "hinglish":
        number_func = number_to_hinglish_words
        date_func = date_to_hinglish_words
        time_func = time_to_hinglish_words
        currency_word = " rupaye "
        percent_word = " percent "
    else:  # bengali (default)
        number_func = number_to_bn_words
        date_func = date_to_bn_words
        time_func = time_to_bn_words
        currency_word = " টাকা "
        percent_word = " শতাংশ "

    text = text.translate(_BN_DIGITS)
    text = text.replace("₹", currency_word).replace("%", percent_word)

    # Date conversion - language-specific
    if language == "bengali":
        # group(4) is a "তারিখ"/"তারিখে" the template already supplied. Absorb it:
        # date_to_bn_words ends in "তারিখ", so leaving it produces "... তারিখ তারিখে".
        text = _RE_DATE.sub(
            lambda m: date_func(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            + ("ে" if (m.group(4) or "").strip().endswith("ে") else ""), text,
        )
    else:
        text = _RE_DATE.sub(
            lambda m: date_func(int(m.group(1)), int(m.group(2)), int(m.group(3))), text,
        )

    # Time range conversion
    if language == "bengali":
        text = _RE_TIME_RANGE.sub(
            lambda m: (f"{time_func(int(m.group(1)), int(m.group(2)))} থেকে "
                       f"{time_func(int(m.group(3)), int(m.group(4)))} পর্যন্ত"), text,
        )
    elif language == "hinglish":
        text = _RE_TIME_RANGE.sub(
            lambda m: (f"{time_func(int(m.group(1)), int(m.group(2)))} se "
                       f"{time_func(int(m.group(3)), int(m.group(4)))} tak"), text,
        )
    else:  # english
        text = _RE_TIME_RANGE.sub(
            lambda m: (f"{time_func(int(m.group(1)), int(m.group(2)))} to "
                       f"{time_func(int(m.group(3)), int(m.group(4)))}"), text,
        )

    # Time conversion
    text = _RE_TIME.sub(lambda m: time_func(int(m.group(1)), int(m.group(2))), text)
    
    # Confirmation ID and phone conversion
    text = _RE_CONF_ID.sub(lambda m: spell_out(m.group(1)), text)
    text = _RE_PHONE.sub(lambda m: digits_one_by_one(m.group(1)), text)
    
    # Decimal conversion
    if language == "bengali":
        text = _RE_DECIMAL.sub(
            lambda m: f"{number_func(int(m.group(1)))} দশমিক {digits_one_by_one(m.group(2))}", text,
        )
    elif language == "hinglish":
        text = _RE_DECIMAL.sub(
            lambda m: f"{number_func(int(m.group(1)))} decimal {digits_one_by_one(m.group(2))}", text,
        )
    else:  # english
        text = _RE_DECIMAL.sub(
            lambda m: f"{number_func(int(m.group(1)))} point {digits_one_by_one(m.group(2))}", text,
        )

    # Integer conversion
    text = _RE_INT.sub(lambda m: number_func(int(m.group(0))), text)

    # Whole-word, case-insensitive: only rewrites a Latin word we have a
    # spoken Bengali form for. Anything else Latin is left alone and
    # reported by `unspeakable_spans()` rather than silently mangled.
    if language == "bengali":
        for latin, bn in _LATIN_SPOKEN_BN.items():
            text = re.sub(rf"\b{latin}\b", bn, text, flags=re.IGNORECASE)

    return re.sub(r"\s{2,}", " ", text).strip()


def detect_language(text: str) -> str:
    """Detect the primary language of text for automatic verbalization.
    
    Returns:
        "bengali", "english", or "hinglish" based on character analysis
    """
    if not text:
        return "bengali"
    
    # Count Bengali characters
    bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    total_chars = len(text.replace(" ", "").replace("\n", ""))
    
    if total_chars == 0:
        return "bengali"
    
    bengali_ratio = bengali_chars / total_chars
    
    # If mostly Bengali characters, use Bengali
    if bengali_ratio > 0.7:
        return "bengali"
    # If some Bengali but mostly English, likely Hinglish
    elif bengali_ratio > 0.2:
        return "hinglish"
    # Otherwise English
    else:
        return "english"


_RE_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z .'-]*")


def unspeakable_spans(text: str) -> list[str]:
    """Latin-script runs left after verbalize() -- these WILL be dropped
    silently by the tokenizer, exactly like the digits were. Callers log
    this so the gap shows up in the logs instead of only in a caller's ear.
    """
    return [m.group(0).strip() for m in _RE_LATIN_RUN.finditer(text) if len(m.group(0).strip()) > 1]
