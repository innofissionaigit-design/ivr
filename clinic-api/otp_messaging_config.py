"""ADDED BY SOURAV -- OTP / Authentication Messaging Provider: URL
placeholder. The user's own explicit instruction: OTP must not be
hardcoded, and there should be ONE separate, clearly-marked file where a
deploying company drops in their own URL, wired straight into the
authentication (OTP) and messaging (actually sending it) parts of this
system -- nothing else in the codebase should need to change.

======================================================================
WHAT TO DO
======================================================================
Set OTP_MESSAGING_WEBHOOK_URL below to your own provider's endpoint --
whatever your team's SMS/WhatsApp/e-mail gateway or existing automation
flow expects to receive a "send this code to this number" request.

Preferred: don't edit this file at all. Instead set the
OTP_MESSAGING_WEBHOOK_URL environment variable before starting clinic-api
(see deploy/env.sh, the same place CLINIC_API_BASE and TTS_URL already
live) -- the environment variable always wins over the literal below, so
this file then never needs to change between environments (dev/staging/
prod each just export a different value).

That is the ONLY change needed. clinic-api/main.py already calls
send_otp_via_provider() -- defined below -- every single time it mints a
fresh OTP, regardless of whether a real URL is set.

======================================================================
WHAT GETS SENT
======================================================================
A plain JSON POST to OTP_MESSAGING_WEBHOOK_URL:

    {"phone": "<patient's phone number, digits only>",
     "otp_code": "<the real 6-digit code, generated fresh each time --
                   see models.py's generate_otp_code()>"}

Shape it into whatever your own provider/automation flow expects on your
side of that URL (a templated SMS, a WhatsApp session message, a ticket
in an existing workflow tool, etc.) -- that translation is entirely your
side's responsibility, not this file's.

======================================================================
IF NO URL IS SET (the default, out of the box)
======================================================================
send_otp_via_provider() is a safe no-op: it logs (without ever logging
the OTP code itself -- see RULE 9 below) that no provider is configured,
and returns False. The OTP is still generated and stored exactly as
before, so this system stays fully runnable and testable with no real
provider connected -- whoever needs the code for a test call just reads
it from the database instead of it arriving by SMS.

======================================================================
WHY A PROVIDER FAILURE NEVER BLOCKS THE OTP FLOW
======================================================================
The OTP row is already committed to the database BEFORE this is ever
called (see clinic-api/main.py's request_report_delivery()). A network
error, timeout, or non-2xx response from your provider only means this
one push notification attempt failed -- it must never be the reason a
patient can't proceed, any more than clinic-api/main.py's own existing
DELIVERY_PROVIDER_FAILURE simulation (RULE 17) treats a delivery failure
as anything other than its own distinct, honestly-reported outcome.
send_otp_via_provider() therefore always degrades to returning False on
any error instead of raising.

======================================================================
RULE 9 -- never let the OTP code itself leak into a log line
======================================================================
This is a security-relevant credential (it gates report delivery, same
posture as agent/tools_client.py's own verify_report_otp() comment on
this). Nothing in this file ever logs otp_code -- only the last 4 digits
of the phone number, purely for a human operator to correlate a log line
with a support ticket.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("clinic_api.otp_messaging")

# <<< A DEPLOYING COMPANY PUTS THEIR OTP / MESSAGING PROVIDER'S URL HERE >>>
# Leave this exactly as "" to keep running without a real provider
# connected (see "IF NO URL IS SET" above). Prefer setting the
# OTP_MESSAGING_WEBHOOK_URL environment variable instead of editing this
# literal -- it always takes priority.
OTP_MESSAGING_WEBHOOK_URL = os.environ.get("OTP_MESSAGING_WEBHOOK_URL", "")

OTP_MESSAGING_TIMEOUT_S = 5.0


def _masked_phone_last4(phone: str) -> str:
    digits = "".join(c for c in phone if c.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits


def send_otp_via_provider(phone: str, otp_code: str) -> bool:
    """POSTs {"phone": phone, "otp_code": otp_code} to
    OTP_MESSAGING_WEBHOOK_URL. Returns True only on a real 2xx response
    from that URL; False whenever no URL is configured or the call fails
    for any reason. Never raises -- see this file's own module docstring
    ("WHY A PROVIDER FAILURE NEVER BLOCKS THE OTP FLOW")."""
    if not OTP_MESSAGING_WEBHOOK_URL:
        logger.info(
            "OTP_MESSAGING_WEBHOOK_URL is not set -- no real provider connected; "
            "skipping send for phone ending in %s (see clinic-api/otp_messaging_config.py).",
            _masked_phone_last4(phone),
        )
        return False
    try:
        r = httpx.post(
            OTP_MESSAGING_WEBHOOK_URL,
            json={"phone": phone, "otp_code": otp_code},
            timeout=OTP_MESSAGING_TIMEOUT_S,
        )
        r.raise_for_status()
        return True
    except httpx.HTTPError as e:
        # RULE 9: otp_code is deliberately never interpolated into this
        # (or any) log line -- same discipline as agent/tools_client.py's
        # own verify_report_otp().
        logger.error(
            "OTP messaging provider call failed for phone ending in %s: %s",
            _masked_phone_last4(phone), e,
        )
        return False
