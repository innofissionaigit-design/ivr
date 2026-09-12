"""Stubs for the heavy ASR stack, so orchestration logic can be tested.

main.py imports torchaudio at module level and agent/asr.py imports NeMo and
omegaconf, which means importing main.py to test a twenty-line branch drags in
a multi-gigabyte model runtime that only exists on the GPU pod. That is why
main.py had no tests at all.

These stubs are registered in sys.modules BEFORE the first import, so the
import succeeds and nothing in them is ever called: every test that uses them
exercises pure orchestration -- which branch runs, what gets spoken, what gets
logged -- and stubs the TTS client itself. If a test ever needs one of these to
actually DO something, that is the signal it belongs on the pod instead.

Deliberately narrow. Only the names the import chain touches are stubbed, so a
new real dependency fails loudly here rather than silently resolving to a mock.
"""
# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
from __future__ import annotations

import sys
import types


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_stubs() -> None:
    try:
        import nemo.collections.asr  # noqa: F401
    except Exception:
        nemo = _module("nemo")
        collections = _module("nemo.collections")
        asr = _module("nemo.collections.asr")
        asr.models = types.SimpleNamespace(ASRModel=object)
        nemo.collections = collections
        collections.asr = asr
        for part in ("nemo.collections.asr.parts",
                     "nemo.collections.asr.parts.submodules"):
            _module(part)
        decoding = _module("nemo.collections.asr.parts.submodules.rnnt_decoding")
        decoding.RNNTDecodingConfig = object

    try:
        import omegaconf  # noqa: F401
    except Exception:
        oc = _module("omegaconf")
        oc.OmegaConf = types.SimpleNamespace(structured=lambda *a, **k: None)

    try:
        import torchaudio  # noqa: F401
    except Exception:
        ta = _module("torchaudio")
        ta.load = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("torchaudio is stubbed; this test should not decode audio"))
        ta.save = ta.load


_install_stubs()
