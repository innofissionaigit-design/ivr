"""SUPERSEDED.

This file held an earlier, separate attempt at the "Caller asks two
questions in one breath" story, written directly against the device and
never reconciled with the codebase's own established patterns for this
story (its own dispatch functions -- _dispatch_multi_intents(),
_combine_responses(), _get_fallback_response(), _dispatch_single_intent_
internal() -- and its _validate() change, which made the LEGACY
single-intent shape fail validation outright: `test_schema_validation_
single_intent` in the old version of this file asserted exactly that,
which would have broken most of the pre-existing test suite that still
hands _validate()/_resolve_intent() plain single-intent dicts).

The actual implementation of this story lives in:
  - agent/llm.py (_validate(), _apply_backward_compat_mirror(), the new
    "intents" array in SYSTEM_PROMPT_TEMPLATE) -- backward-compatible
    with the legacy single-intent shape, on purpose.
  - agent/reply_templates.py (multi_intent_missing_info_reply(),
    multi_intent_out_of_scope_reply(), multi_intent_needs_separate_flow_reply()).
  - main.py / main_pcm.py (_dispatch_multi_intent_turn(),
    _resolve_combinable_intent_fragment()).
  - agent/semantic_cache.py (_is_l2_eligible()'s multi-intent guard).

Test coverage for all of the above lives in
tests/test_multi_intent_dispatch.py. This file is kept (rather than left
missing) only so nothing on disk still references it; it intentionally
defines no tests of its own.
"""
