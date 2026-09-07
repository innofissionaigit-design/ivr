"""AI4Bharat Bengali TTS -- FastAPI wrapper around coqui-tts's Synthesizer.

Matches the contract agent/tts.py's TTSClient expects: POST /synthesize
with {"text": ..., "lang": "bn"} -> raw WAV bytes.

The Synthesizer is loaded ONCE at module import (not per-request) --
model load takes several seconds and holds real GPU memory, so this must
be a long-lived process, not spawned per call.

WHY THIS DOES MORE THAN CALL synthesizer.tts()
----------------------------------------------
A single FastPitch pass over a whole multi-clause reply is what makes this
voice sound like a machine, and it is fixable without changing models:

* No breaths. FastPitch renders one flat prosodic contour across an entire
  utterance. Real speakers stop between clauses. Splitting on Bengali
  sentence and clause boundaries and inserting real silence is the single
  largest naturalness gain available here.
* Ragged padding. Each pass emits its own leading/trailing near-silence of
  arbitrary length, so naive concatenation produces gaps that are too long
  in some places and absent in others. Each chunk is trimmed, then padded
  by an amount chosen for the punctuation that ended it.
* Rushed delivery. length_scale 1.0 is noticeably fast for a service line
  a caller is trying to write a price down from. Slightly above 1 reads as
  measured rather than sluggish.
* Inconsistent level. Peak varies per utterance, which on a phone sounds
  like the speaker keeps moving. Normalizing to a fixed peak fixes it.

Every one of these is tunable per-request (see SynthesizeRequest) so the
settings can be A/B'd against a real handset without a redeploy.
"""
import io
import os
import re
import threading

# The AI4Bharat checkpoint's speaker manager resolves a RELATIVE path
# ("models/v1/bn/fastpitch/speakers.pth") baked in at save time, against
# whatever the process's CWD happens to be -- not the checkpoint's own
# location. Pin CWD explicitly so this doesn't depend on how/where this
# script gets launched from.
os.chdir("/workspace/tts_checkpoints")

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from TTS.utils.synthesizer import Synthesizer

app = FastAPI()

CKPT = "/workspace/tts_checkpoints/bn"

synthesizer = Synthesizer(
    tts_checkpoint=f"{CKPT}/fastpitch/best_model.pth",
    tts_config_path=f"{CKPT}/fastpitch/config.json",
    tts_speakers_file=f"{CKPT}/fastpitch/speakers.pth",
    vocoder_checkpoint=f"{CKPT}/hifigan/best_model.pth",
    vocoder_config=f"{CKPT}/hifigan/config.json",
    use_cuda=True,
)

# One GPU model instance, shared by every request. FastAPI runs a sync
# `def` endpoint (see /synthesize below) in a thread-pool, so overlapping
# calls -- a genuinely concurrent second caller, OR agent/tts.py's own
# retry firing a second request while the first is still running after a
# client-side timeout -- would otherwise run two `.tts()` passes on the
# SAME Synthesizer/CUDA context at once, and both mutate its shared
# `length_scale` attribute right before calling it (see _render() below).
# That is a real mechanism for exactly "voice sometimes jamming/cracking
# in long conversations": longer calls mean more requests and more chances
# for the GPU to be briefly busy enough to trigger a client retry while the
# original synthesis is still in flight, so the corrupt-concurrent-call
# case gets MORE likely as a conversation runs longer, not less. A single
# lock around inference serializes those calls -- request queueing costs a
# little latency under real overlap, which is strictly better than
# occasionally handing back garbled or partially-overwritten audio.
_synth_lock = threading.Lock()

SAMPLE_RATE = synthesizer.output_sample_rate or 22050
DEFAULT_SPEAKER = os.environ.get("TTS_SPEAKER", "female")

# >1 slows delivery. 1.08 was the original measurement, but real callers
# on the opening greeting -- a caller's very first impression of the whole
# system -- still reported it reading like a rushed, robotic list of words
# rather than someone actually saying hello. Raised to 1.18: still a
# measured, unscientific bump rather than a re-measurement against real
# handset audio (see the module docstring's caveat on `speed` being
# per-request precisely so this can be A/B'd properly later), but a bigger
# step than 1.08 turned out to be, in the direction the actual complaint
# points. Tune via /synthesize's `speed` override before changing this
# default further.
DEFAULT_LENGTH_SCALE = float(os.environ.get("TTS_LENGTH_SCALE", "1.18"))

# Silence inserted AFTER a chunk, by the punctuation that ended it. Nudged
# up alongside the length_scale change above -- a slower voice with the
# same short gaps between clauses/sentences still runs its breaths
# together and can end up sounding just as machine-like, only slower.
PAUSE_S = {"sentence": 0.34, "clause": 0.20, "none": 0.08}

TARGET_PEAK = 0.89          # ~-1 dBFS; loud without clipping on phone speakers
TRIM_THRESHOLD = 0.012      # below this is padding, not speech

# Bengali sentence enders (danda + Latin punctuation, since clinic data
# mixes both) and clause separators.
_RE_SENTENCE = re.compile(r"([^।?!\n]+[।?!\n]?)")
_RE_CLAUSE = re.compile(r"([^,;:]+[,;:]?)")

MAX_CHUNK_CHARS = 90        # long single clauses still get a breath


def _split_for_prosody(text: str) -> list[tuple[str, str]]:
    """-> [(chunk_text, pause_kind)]. Sentences first, then clauses inside
    any sentence that either runs long OR already carries clause
    punctuation.

    That second condition is new, and it is what actually fixes the
    greeting specifically: "নমস্কার, কলকাতা কেয়ার ডায়াগনস্টিকসে স্বাগতম।" is
    well under MAX_CHUNK_CHARS, so the old rule fed the WHOLE sentence
    through FastPitch as one uninterrupted pass -- comma and all. A real
    person says "Hello," with a small breath before continuing, and a
    single flat pass with no internal pause reads as one run-on machine
    sentence, no matter how the length_scale is tuned. Every reply
    template in reply_templates.py has this same shape (short sentences
    with a comma in them), so this is not a greeting-only fix.
    """
    out: list[tuple[str, str]] = []
    for raw_sentence in _RE_SENTENCE.findall(text):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        has_clause_punct = any(p in sentence for p in ",;:")
        if len(sentence) <= MAX_CHUNK_CHARS and not has_clause_punct:
            out.append((sentence, "sentence"))
            continue
        clauses = [c.strip() for c in _RE_CLAUSE.findall(sentence) if c.strip()]
        for i, clause in enumerate(clauses):
            out.append((clause, "sentence" if i == len(clauses) - 1 else "clause"))
    if not out:
        out = [(text.strip(), "sentence")]
    return out


def _trim_silence(wav: np.ndarray) -> np.ndarray:
    loud = np.where(np.abs(wav) > TRIM_THRESHOLD)[0]
    if loud.size == 0:
        return wav[:0]
    return wav[loud[0]:loud[-1] + 1]


def _normalize_peak(wav: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav * (TARGET_PEAK / peak) if peak > 1e-6 else wav


def _render(text: str, speaker: str, speed: float, pauses: bool) -> np.ndarray:
    if hasattr(synthesizer.tts_model, "length_scale"):
        synthesizer.tts_model.length_scale = speed

    chunks = _split_for_prosody(text) if pauses else [(text, "sentence")]
    pieces: list[np.ndarray] = []

    for chunk_text, pause_kind in chunks:
        # split_sentences: this module already decided the chunking, and
        # letting Coqui re-split would reintroduce the ragged joins -- but
        # the vendored AI4Bharat/gokulkarthik TTS fork we're pinned to
        # (mainline coqui-tts is Python <3.11 only; see deploy notes)
        # predates that kwarg entirely and its Synthesizer.tts() doesn't
        # accept it. It also never re-splits internally, so simply not
        # passing it is equivalent, not a behavior change.
        raw = synthesizer.tts(chunk_text, speaker_name=speaker)
        wav = _trim_silence(np.asarray(raw, dtype=np.float32))
        if wav.size == 0:
            continue
        pieces.append(wav)
        if pauses:
            pieces.append(np.zeros(int(PAUSE_S[pause_kind] * SAMPLE_RATE), dtype=np.float32))

    if not pieces:
        return np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)

    joined = np.concatenate(pieces)
    # A short lead-in stops the very first phoneme being clipped by
    # playback devices that ramp up on stream start.
    return _normalize_peak(
        np.concatenate([np.zeros(int(0.04 * SAMPLE_RATE), dtype=np.float32), joined]),
    )


class SynthesizeRequest(BaseModel):
    text: str
    lang: str = "bn"
    speaker: str | None = None
    speed: float | None = None      # length_scale; >1 slower
    pauses: bool = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "speaker": DEFAULT_SPEAKER,
        "sample_rate": SAMPLE_RATE,
        "length_scale": DEFAULT_LENGTH_SCALE,
    }


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    # See _synth_lock's comment above -- this is a real GPU model shared
    # across every request FastAPI's thread-pool might run concurrently.
    with _synth_lock:
        wav = _render(
            req.text,
            req.speaker or DEFAULT_SPEAKER,
            req.speed or DEFAULT_LENGTH_SCALE,
            req.pauses,
        )
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
