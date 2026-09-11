"""ADDED BY SOURAV -- single index of every external URL a deploying
company can plug into this stack.

WHY THIS FILE EXISTS, GIVEN db.py AND otp_messaging_config.py ALREADY
DO THIS THEMSELVES
======================================================================
clinic-api/db.py already reads DATABASE_URL from the environment.
clinic-api/otp_messaging_config.py already reads
OTP_MESSAGING_WEBHOOK_URL the same way. Both already work correctly on
their own -- this file does NOT replace either, and does NOT duplicate
their logic (importing DATABASE_URL from here instead of db.py would
create two sources of truth for the same setting). What it adds is a
single place to LOOK: one file listing every company-specific URL this
stack knows about, existing ones included, so a deploying company (or a
future engineer) does not have to go hunting through db.py, otp_
messaging_config.py, and deploy/env.sh separately to find out what's
configurable.

It also carries placeholders for the Phase 1 (Walk-in Eligibility /
Prescription Requirements / Insurance Coverage Policy / Outstanding
Balance) stories' MOST LIKELY future external integrations --
NOT YET CALLED ANYWHERE. Phase 1 reads walk-in/prescription/insurance/
billing data from this same local clinic-api database only (see
clinic-api/models.py's InsuranceProvider/InsurancePolicy/PatientBilling
and LabTest's new columns) -- a deliberate, explicit scoping decision for
this first pass, not an oversight. A real deployment that wants a live
insurer eligibility check or a real billing/ERP system wired in instead
of (or in addition to) this local table would set the corresponding
variable below AND add the actual HTTP call in clinic-api/main.py's
insurance_coverage() / patient_billing() endpoints -- setting the
variable alone does nothing by itself, unlike OTP_MESSAGING_WEBHOOK_URL
which send_otp_via_provider() already calls unconditionally.

======================================================================
WHAT TO DO
======================================================================
Preferred: don't edit this file. Set the environment variables below
before starting clinic-api (see deploy/env.sh, which documents all of
them next to CLINIC_API_BASE/TTS_URL) -- the environment variable always
wins over the literal default here, so this file never needs to change
between environments (dev/staging/prod each just export a different
value). This is the exact same convention otp_messaging_config.py
already established; this file does not introduce a new one.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------
# ALREADY WIRED IN AND CALLED -- read directly by their own modules
# (listed here for visibility only; import from THOSE modules, not from
# here, if real code needs the value).
# ---------------------------------------------------------------------

# Already read directly by clinic-api/db.py. See that file's own
# docstring for why the default is a local SQLite path, not empty.
DATABASE_URL = os.environ.get("DATABASE_URL", "<put your db url here>")

# Already read directly by clinic-api/otp_messaging_config.py, and
# already called unconditionally by send_otp_via_provider() every time a
# fresh OTP is minted.
OTP_MESSAGING_WEBHOOK_URL = os.environ.get(
    "OTP_MESSAGING_WEBHOOK_URL", "<put your messaging url here>",
)

# ---------------------------------------------------------------------
# NOT YET CALLED ANYWHERE -- Phase 1 reads from the local clinic-api DB
# only (see module docstring above). Set these, and add the actual call
# in clinic-api/main.py, only if/when you want a real external system
# wired in instead.
# ---------------------------------------------------------------------

# A real-time insurance eligibility/coverage-check API, if you have one,
# instead of (or in addition to) clinic-api's own InsurancePolicy table.
INSURANCE_PROVIDER_API_URL = os.environ.get(
    "INSURANCE_PROVIDER_API_URL", "<put your insurance provider api url here>",
)

# A real billing/ERP/accounting system's API, if you have one, instead of
# (or in addition to) clinic-api's own PatientBilling table.
BILLING_SYSTEM_API_URL = os.environ.get(
    "BILLING_SYSTEM_API_URL", "<put your billing system api url here>",
)
