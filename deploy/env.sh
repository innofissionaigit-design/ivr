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

# /workspace/bin first: that is where the persistent ollama binary lives.
export PATH=/workspace/bin:/workspace/venv/bin:${PATH:-}

# ---------------------------------------------------------------------------
# LANGUAGES -- Bengali, Hindi, English
# ---------------------------------------------------------------------------
# UNSET MEANS BENGALI ONLY, and that is the correct default today. The ASR
# checkpoint on this pod is indicconformer_stt_bn_* -- a BENGALI-ONLY model.
# It cannot transcribe Hindi or English, and no configuration changes that.
#
# agent/language.py's enabled() will NOT advertise a language whose ASR
# checkpoint variable is unset, precisely so this file cannot claim a
# capability the pod does not have. Listing "hi" below without setting
# VOICE_AGENT_NEMO_FILE_HI does nothing at all -- deliberately.
export VOICE_AGENT_LANGUAGES=${VOICE_AGENT_LANGUAGES:-bn}
export VOICE_AGENT_DEFAULT_LANG=${VOICE_AGENT_DEFAULT_LANG:-bn}

# fixed | script | parallel -- see agent/language.py.
#   fixed    every caller gets the default language. What the system has
#            always done, and the only honest setting while ASR is one
#            Bengali checkpoint.
#   script   read the transcript's script. Only meaningful once ASR can
#            EMIT more than one script.
#   parallel decode turn one with every enabled checkpoint and keep the
#            best. The only true audio-based detection, and the most
#            expensive: N decodes on turn one, on a shared GPU.
export VOICE_AGENT_LANG_STRATEGY=${VOICE_AGENT_LANG_STRATEGY:-fixed}

# Per-language ASR checkpoints. Setting one is what actually turns a
# language on. Loaded LAZILY, on the first caller who speaks it, because
# each IndicConformer checkpoint is VRAM on a card already holding
# Qwen2.5 and the TTS voices.
export VOICE_AGENT_NEMO_FILE_HI=${VOICE_AGENT_NEMO_FILE_HI:-}
export VOICE_AGENT_NEMO_FILE_EN=${VOICE_AGENT_NEMO_FILE_EN:-}

# Per-language TTS. tts_server.py looks for <TTS_CKPT_ROOT>/<lang> when the
# explicit variable is empty, and falls back to the Bengali voice when a
# language has no checkpoint -- reporting which language it ACTUALLY spoke
# in the X-TTS-Lang response header, so the agent is never lied to.
export TTS_CKPT_ROOT=${TTS_CKPT_ROOT:-/workspace/tts_checkpoints}
export TTS_CKPT_HI=${TTS_CKPT_HI:-}
export TTS_CKPT_EN=${TTS_CKPT_EN:-}
export TTS_SPEAKER_HI=${TTS_SPEAKER_HI:-}
export TTS_SPEAKER_EN=${TTS_SPEAKER_EN:-}

# ---------------------------------------------------------------------------
# HOSPITAL SMS GATEWAY -- the patient's written confirmation
# ---------------------------------------------------------------------------
# THE SECRET IS NOT IN THIS FILE, AND MUST NOT BE ADDED TO IT.
# HOSPITAL_GATEWAY_API_KEY and HOSPITAL_GATEWAY_DLR_TOKEN are the only two
# credentials this repository has ever needed, and this file is committed,
# read by every service, and routinely pasted into bug reports. Export them
# from the shell, a systemd drop-in, or an uncommitted deploy/env.secret.sh
# sourced after this one.
#
# LEAVING HOSPITAL_GATEWAY_URL UNSET IS A SUPPORTED STATE, not a broken one.
# clinic-api then records every message as `skipped` with the reason
# attached, and reply_templates.py stops promising callers an SMS -- see
# notify_service.queue_message() and _written_confirmation_clause(). A bench
# pod behaves correctly and messages nobody.
export HOSPITAL_GATEWAY_URL=${HOSPITAL_GATEWAY_URL:-}
export HOSPITAL_GATEWAY_AUTH_HEADER=${HOSPITAL_GATEWAY_AUTH_HEADER:-Authorization}

# DLT registration, from the hospital's account on the operator portal.
# SENDER_ID is the registered 6-character header; ENTITY_ID is the Principal
# Entity ID. Both are submitted with every message and both are matched by
# the operator -- a mismatch is a silent non-delivery, not an error we see.
export HOSPITAL_GATEWAY_SENDER_ID=${HOSPITAL_GATEWAY_SENDER_ID:-}
export HOSPITAL_GATEWAY_ENTITY_ID=${HOSPITAL_GATEWAY_ENTITY_ID:-}

# Registered content template IDs, one per event. clinic-api refuses to send
# a template whose ID is empty rather than paying for an operator rejection
# -- see notifications.preflight(). The bodies these IDs correspond to are
# in clinic-api/message_templates.py and must match the portal exactly.
export HOSPITAL_GATEWAY_TEMPLATE_BOOKED=${HOSPITAL_GATEWAY_TEMPLATE_BOOKED:-}
export HOSPITAL_GATEWAY_TEMPLATE_RESCHEDULED=${HOSPITAL_GATEWAY_TEMPLATE_RESCHEDULED:-}
export HOSPITAL_GATEWAY_TEMPLATE_CANCELLED=${HOSPITAL_GATEWAY_TEMPLATE_CANCELLED:-}

# agent/slot_parse.parse_phone() stores the last 10 digits; the gateway wants
# E.164. This is the prefix put back on.
export HOSPITAL_GATEWAY_COUNTRY_CODE=${HOSPITAL_GATEWAY_COUNTRY_CODE:-91}

# 8s, comfortably longer than the 4s agent/tools_client.py allows this
# service -- and safe precisely BECAUSE the send is a background task that
# runs after the response has already gone back to the caller.
export HOSPITAL_GATEWAY_TIMEOUT_S=${HOSPITAL_GATEWAY_TIMEOUT_S:-8}

# How long a `sent` message may go without a delivery receipt before the
# staff queue calls it a failure. See notifications.DEFAULT_STALE_MINUTES.
export HOSPITAL_GATEWAY_STALE_MINUTES=${HOSPITAL_GATEWAY_STALE_MINUTES:-15}
