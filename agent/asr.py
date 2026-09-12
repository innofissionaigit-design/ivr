"""ASR for one caller turn -- self-contained IndicConformer integration.

This project runs on its OWN pod, independent of voice-to-rx-repo (that
system lives on a different pod entirely -- see its own HANDOFF.md). An
earlier version of this file imported voicerx.asr.ASRNode directly to
reuse that project's already-debugged NeMo integration, but that created
a live dependency on an unrelated codebase (drug gazetteers, prescription
extraction -- none of it relevant here) that isn't even deployed on this
pod. Fixed by inlining just the actual proven ASR logic instead.

The logic below -- NOT the code coupling -- is what's actually worth
reusing, because it encodes real, previously-debugged failures:

  - Mainline NeMo cannot load this checkpoint at all: IndicConformer uses
    a multilingual AGGREGATE tokenizer, and mainline's
    _setup_monolingual_tokenizer raises KeyError: 'dir'. Only AI4Bharat's
    NeMo fork (github.com/AI4Bharat/NeMo, branch nemo-v2) handles it.

  - RNNT's default "greedy_batch" decoding strategy silently returns
    empty text on real audio; "greedy" does not. Confirmed on this exact
    pod: verified 2026-08-24 that "greedy_batch" is the library default
    and must be overridden explicitly via change_decoding_strategy().

  - .transcribe() has been observed to return either a flat list of
    strings or a list of single-item lists depending on internal batching
    state -- unwrap defensively.
"""
from __future__ import annotations

import asyncio
import dataclasses
import glob
import logging
import os

import torch
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

logger = logging.getLogger("asr")

MODEL_ID = "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large"


def _resolve_nemo_file() -> str:
    """Locate the .nemo checkpoint without hardcoding a machine-specific
    HF cache path. Resolution order: explicit env var override, then a
    glob of the HF cache this project's own download step populates."""
    explicit = os.environ.get("VOICE_AGENT_NEMO_FILE")
    if explicit and os.path.exists(explicit):
        return explicit

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hits = glob.glob(os.path.join(hf_home, "hub", "**", "*indicconformer*", "**", "*.nemo"),
                      recursive=True)
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Could not locate the IndicConformer .nemo checkpoint under {hf_home}. "
        f"Download it first: huggingface_hub.snapshot_download('{MODEL_ID}', "
        f"token=<HF_TOKEN>) -- the model is gated, accept its licence on "
        f"huggingface.co first."
    )


def _first_text(texts) -> str:
    if not texts:
        return ""
    item = texts[0]
    while isinstance(item, (list, tuple)):
        if not item:
            return ""
        item = item[0]
    return (item or "").strip()


def _word_agreement(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@dataclasses.dataclass
class ASRResult:
    """One transcribed turn, plus the confidence signal that came free with it.

    decoder_agreement is None -- NOT 0.0 and NOT 1.0 -- whenever the two
    decoders were not actually compared. It previously carried a sentinel on
    those paths, which was wrong in two directions at once: ctc_fallback
    reported 0.0 ("total disagreement") when the truth was "RNNT returned
    nothing, so there was nothing to compare", and the empty result reported
    1.0 ("perfect agreement") for a turn where nothing was decoded at all.

    That mattered beyond tidiness. This field is exported per turn and is meant
    to be correlated against human labels; a study that ingests those sentinels
    as measurements would be correlating against fabricated values, and the
    1.0 case would pull the "high agreement means correct" relationship in
    exactly the wrong direction. None is unambiguous and forces the consumer to
    decide what to do about it.

    ctc_words / rnnt_words are the sizes of the two word sets the agreement was
    computed over. Jaccard on a one-word utterance can only ever be 0.0 or 1.0,
    so the score means something quite different at n=1 than at n=8; without
    these counts that is invisible in the exported data.
    """
    text: str
    decoder_used: str
    decoder_agreement: float | None = None
    ctc_words: int = 0
    rnnt_words: int = 0


class TurnASR:
    """One instance shared across all calls -- the NeMo model is the
    expensive singleton, loaded once at process startup."""

    def __init__(self, nemo_file: str | None = None, language_id: str = "bn",
                 device: str | None = None):
        self.language_id = language_id
        nemo_file = nemo_file or _resolve_nemo_file()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = nemo_asr.models.ASRModel.restore_from(restore_path=nemo_file)
        self.model = self.model.to(self.device)
        self.model.freeze()

        # The proven fix -- see module docstring for why the library
        # default ("greedy_batch") is not used here.
        rnnt_cfg = OmegaConf.structured(RNNTDecodingConfig(strategy="greedy"))
        self.model.change_decoding_strategy(rnnt_cfg, decoder_type="rnnt")

    def _transcribe_clip(self, clip_path: str) -> tuple[str, str]:
        self.model.cur_decoder = "ctc"
        ctc_texts = self.model.transcribe(
            [clip_path], batch_size=1, logprobs=False, language_id=self.language_id,
        )
        ctc_text = _first_text(ctc_texts)

        self.model.cur_decoder = "rnnt"
        # Add a small retry for RNNT to handle transient failures
        try:
            rnnt_texts = self.model.transcribe([clip_path], batch_size=1, language_id=self.language_id)
            rnnt_text = _first_text(rnnt_texts)
        except Exception as e:
            logger.warning("RNNT transcription failed, falling back to CTC: %s", e)
            rnnt_text = ""

        return ctc_text, rnnt_text

    def transcribe_utterance_sync(self, wav_path: str) -> ASRResult:
        """Blocking. Callers MUST run this via asyncio.to_thread -- it
        holds the GIL through GPU inference and would otherwise stall
        every other WebSocket connection's audio handling on this process."""
        ctc_text, rnnt_text = self._transcribe_clip(wav_path)

        n_ctc, n_rnnt = len(set(ctc_text.split())), len(set(rnnt_text.split()))

        if rnnt_text:
            # The only path where the two decoders were actually compared, and
            # therefore the only path that carries a real agreement score.
            return ASRResult(text=rnnt_text, decoder_used="rnnt",
                             decoder_agreement=round(_word_agreement(ctc_text, rnnt_text), 2),
                             ctc_words=n_ctc, rnnt_words=n_rnnt)
        if ctc_text:
            # RNNT returned nothing, so there was no second opinion to compare
            # against. That is NOT disagreement -- see ASRResult's docstring.
            return ASRResult(text=ctc_text, decoder_used="ctc_fallback",
                             decoder_agreement=None,
                             ctc_words=n_ctc, rnnt_words=0)
        return ASRResult(text="", decoder_used="none", decoder_agreement=None)

    async def transcribe_utterance(self, wav_path: str) -> ASRResult:
        return await asyncio.to_thread(self.transcribe_utterance_sync, wav_path)
