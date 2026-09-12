"""Can this reply actually be SAID? The verdict, and nothing else.

WHAT THE SIGNAL IS
------------------
agent/bn_normalize.py has known for months which spans the Bengali FastPitch
tokenizer will silently discard -- `unspeakable_spans()` finds them, and
`verbalize()` rewrites the ones it has a spoken form for. agent/tts.py called
both and then logged a warning and synthesized the sentence anyway, hole and
all. The gap analysis put it plainly: the detection is already built and
correct; only the enforcement is missing.

This module is the enforcement's decision half. It decides; main.py acts.
That split is the same one agent/confidence.py makes, for the same reason: a
pure function is testable against the real catalogue without a GPU, a model,
or a websocket.

WHY THERE IS NO THRESHOLD HERE
------------------------------
confidence.py has two floors, both reasoned rather than measured, and a long
note about not tuning them to make the agent ask fewer questions. This module
has none, and that is not an omission. A span is either in the tokenizer's
vocabulary or it is not; there is no confidence to trade off, no band where
the right answer is "probably fine". That is the whole reason this story is
small, and it is why nothing here should ever grow a knob.

WHY THERE IS NO RUNTIME REWRITE
-------------------------------
The acceptance criterion says the reply is "rewritten or escalates", and the
rewrite is real -- it is `verbalize()`'s spoken-form table, which has already
run by the time this function looks at the text. What this module deliberately
does NOT do is attempt a cleverer second rewrite when that table misses:

  * An LLM paraphrase would put a price, a date or a confirmation number back
    into generated text. agent/reply_templates.py exists specifically to stop
    that, and "the sentence was unspeakable" is not a good enough reason to
    relax the rule that the model never restates a fact.
  * Automatic transliteration is worse than the bug. A hole is detectable and
    a caller knows something is missing; a confidently mispronounced word is
    neither. The table is hand-written per entry for exactly this reason.

So: rewrite in the table, block everything the table missed, and let the
blocked count tell you which entries the table still needs.

THE MATRIX-LANGUAGE COUPLING, WRITTEN DOWN BEFORE IT BITES
----------------------------------------------------------
`unspeakable_spans()` treats Latin script as unspeakable. That is correct only
because this agent is monolingual Bengali today -- agent/asr.py hardcodes
language_id="bn" and nothing sets CallState.language.

Under E2-S4E the reply mirrors the caller's own mixture, and a caller who says
a test name in English is answered in English. At that point a Latin span is
the CORRECT output, and a gate that blocks Latin would refuse every
English-matrix reply on the line. So `check()` takes the language now, while
it is still None everywhere, and returns UNCHECKED rather than SPEAKABLE for
anything it does not model.

UNCHECKED is not a pass. It is the same discipline as ASRResult's
decoder_agreement=None: when nothing was actually examined, say so, rather
than reporting the reassuring value and letting a consumer mistake it for a
measurement.
"""
# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
from __future__ import annotations

import dataclasses

from agent.bn_normalize import unspeakable_spans, verbalize

SPEAKABLE = "speakable"
BLOCKED = "blocked"
UNCHECKED = "unchecked"

# The one matrix language this detector models. See the module docstring.
MODELLED_LANGUAGE = "bn"


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Frozen: a verdict is a finding, not a working value. Nothing
    downstream gets to soften one by assigning to it."""

    state: str                      # speakable | blocked | unchecked
    spoken: str                     # the verbalized text the check ran against
    dropped: tuple[str, ...] = ()   # spans the tokenizer would discard

    @property
    def is_blocked(self) -> bool:
        return self.state == BLOCKED


def check(text_bn: str, language: str | None = None) -> Verdict:
    """-> Verdict for one composed reply, before it reaches the synthesizer.

    `spoken` is returned rather than recomputed by the caller so the gate and
    the synthesizer cannot end up disagreeing about what was examined. It is
    the same pure string pass either way; agent/tts.py still calls verbalize()
    itself, because it is the last thing in front of the tokenizer and must
    hold on its own even if a future caller forgets this function exists.
    """
    # verbalize() runs either way: spelling numbers into words is what stops
    # every price being silence, and that is needed whatever the verdict.
    # (When a second matrix language does land, verbalize() will need the
    # language too -- it spells numbers into BENGALI words. One more reason
    # this parameter exists before there is anything to pass to it.)
    spoken = verbalize(text_bn)

    if language is not None and not language.lower().startswith(MODELLED_LANGUAGE):
        # Not Bengali. This detector's rule -- "Latin script is unspeakable" --
        # is simply false here, so it reports that it did not look rather than
        # returning a pass it did not earn.
        return Verdict(UNCHECKED, spoken, ())

    dropped = tuple(unspeakable_spans(spoken))
    return Verdict(BLOCKED if dropped else SPEAKABLE, spoken, dropped)
