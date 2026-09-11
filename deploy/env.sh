# Environment for every service in the stack. Source before launching.
#
# Everything referenced here MUST live under /workspace. RunPod wipes the
# container's overlay filesystem ("/" and "/root") on every restart and
# keeps only the network volume, so anything installed to /usr/local/bin
# or stored in /var/lib is gone the next time the pod boots. That is not a
# hypothetical: the ollama binary and the entire Postgres installation
# have each been lost to it three times.

export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export HF_HUB_ENABLE_HF_TRANSFER=0

export OLLAMA_MODELS=/workspace/.ollama/models
# -1 keeps models resident indefinitely. Without it Ollama unloads after
# five idle minutes and the next caller pays a 47s cold start mid-call.
export OLLAMA_KEEP_ALIVE=-1

export PYTHONPATH=/workspace/AI4Bharat_NeMo:/workspace/kolkata-care-voice-agent:

# DATABASE_URL is deliberately NOT set: clinic-api/db.py then defaults to
# sqlite:////workspace/clinic.db, which persists across restarts. Postgres
# physically cannot run on this volume -- see that file's docstring. Set
# this only when pointing at a real external Postgres.
export CLINIC_API_BASE=http://localhost:8080
export TTS_URL=http://localhost:8002/synthesize
export SILERO_VAD_REPO=/workspace/silero-vad

# ADDED BY SOURAV -- "otp will not be hardcoded". OTP_MESSAGING_WEBHOOK_URL
# is deliberately NOT set here: clinic-api/otp_messaging_config.py then
# defaults it to "" and runs with no real provider connected (every OTP
# is still generated for real -- see models.generate_otp_code() -- it
# just isn't pushed anywhere; read it from the database instead). Set
# this to your own SMS/WhatsApp/e-mail provider's endpoint to actually
# deliver OTPs -- see that file's own module docstring for exactly what
# gets POSTed to it. This is the ONLY environment variable a deploying
# company needs to add to connect their own provider; nothing else in
# this stack needs to change.
# export OTP_MESSAGING_WEBHOOK_URL=https://your-provider.example.com/send-otp

# ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables. Both
# deliberately NOT set here, same "safe default, opt in" pattern as
# OTP_MESSAGING_WEBHOOK_URL just above -- unlike that one, though, setting
# either of these alone does NOT connect anything yet: Phase 1's walk-in/
# prescription/insurance/billing stories read from clinic-api's own local
# database only (see clinic-api/company_config.py's own module docstring).
# These exist so a real insurer or billing system's URL has one obvious
# place to go WHEN clinic-api/main.py is updated to actually call it.
# export INSURANCE_PROVIDER_API_URL=https://your-insurer.example.com/eligibility
# export BILLING_SYSTEM_API_URL=https://your-billing-system.example.com/api

# /workspace/bin first: that is where the persistent ollama binary lives.
export PATH=/workspace/bin:/workspace/venv/bin:${PATH:-}
