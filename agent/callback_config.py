"""ADDED BY SOURAV -- "Caller asks to be called back" story, Acceptance
Criterion 3's "disabled in config" half.

Deliberately its OWN tiny module rather than a new constant bolted onto
agent/callback_flow.py: that module's whole point (see its own docstring)
is to stay a pure decision function with no os/env access of its own, the
same discipline agent/report_flow.py and agent/compare_flow.py already
hold themselves to. Reading the environment at import time is exactly the
kind of thing clinic-api/db.py and clinic-api/otp_messaging_config.py
already do for their own settings -- this module follows that same,
already-established convention, just for a plain on/off switch instead of
a URL.

DATABASE_URL still overrides everything for the database itself; this
flag is unrelated to that one and does not touch persistence at all -- it
only decides whether main.py is willing to START collecting a callback
request in the first place (see agent/callback_flow.py's
check_callback_availability(), the ONE caller of this constant).
"""
from __future__ import annotations

import os

# Accepts the same handful of "off" spellings a human is likely to type in
# an env file; anything else (including the variable being unset, hence
# the "true" default) means the feature stays on.
CALLBACKS_ENABLED = os.environ.get("CALLBACKS_ENABLED", "true").strip().lower() not in (
    "false", "0", "no", "off", "disabled",
)
