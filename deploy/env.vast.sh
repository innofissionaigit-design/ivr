# Environment for every service in the stack, on VAST.AI.
#
# WHY A SEPARATE FILE FROM env.sh
# -------------------------------
# env.sh encodes a RunPod-specific fact: RunPod wipes the container overlay
# ("/" and "/root") on every restart and keeps only the network volume, so
# everything MUST live under /workspace or it is lost on reboot.
#
# Vast.ai does not work that way. An instance's container filesystem IS the
# rented disk and it persists across stop/start, so the "/workspace or it
# vanishes" rule does not apply. /workspace is still used below purely as a
# convention, so paths match env.sh and nothing else in the repo has to know
# which provider it is running on.
#
# The real vast.ai differences are further down: PORTS and the DB backend.

export VOICE_AGENT_PROVIDER=vast

export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export HF_HUB_ENABLE_HF_TRANSFER=0

export OLLAMA_MODELS=/workspace/.ollama/models
# -1 keeps models resident indefinitely. Without it Ollama unloads after
# five idle minutes and the next caller pays a 47s cold start mid-call
# (measured -- see agent/llm.py's _call_ollama docstring).
export OLLAMA_KEEP_ALIVE=-1

# Bound Ollama's own internal queue. Unset, concurrent turns queue INSIDE
# ollama where this codebase cannot see them, and a caller waiting in that
# queue is indistinguishable from a cold start -- so llm.py's 90s timeout
# never fires and the socket dies first. See the concurrency audit, P1.
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_QUEUE=32

export PYTHONPATH=/workspace/AI4Bharat_NeMo:/workspace/kolkata-care-voice-agent:

# SQLite on the instance disk. Same reasoning as env.sh: a 74-row
# read-mostly catalogue with one writer process does not need Postgres, and
# clinic-api/db.py already defaults here. Set DATABASE_URL to override.
export CLINIC_DB_PATH=/workspace/clinic.db

# CLINIC-API IS ON 8081, NOT 8080 -- deliberate, and a real collision, not taste.
# The vastai/base-image template this instance uses already binds 8080 for
# Jupyter, and advertises it in PORTAL_CONFIG as both "Jupyter" and
# "Jupyter Terminal" (localhost:8080:8080:/terminals/1). Leaving clinic-api on
# 8080 means whichever process starts second fails to bind -- and start_all.sh
# would report the port as already healthy, because its `start()` probe treats
# ANY 200 on /api/health OR /health as "already up on :8080". Jupyter answering
# that probe would make a dead clinic-api look successfully started.
export CLINIC_API_BASE=http://localhost:8081
export TTS_URL=http://localhost:8002/synthesize
export SILERO_VAD_REPO=/workspace/silero-vad

export PATH=/workspace/bin:/workspace/venv/bin:${PATH:-}

# ---------------------------------------------------------------------------
# PORTS -- the one thing that genuinely differs from RunPod
# ---------------------------------------------------------------------------
# RunPod fronts services with an HTTPS proxy that upgrades WebSockets on the
# same origin as the page. Vast.ai does not: it maps each container port to
# an arbitrary EXTERNAL port on the host's public IP, reachable as
# http://<PUBLIC_IP>:<EXTERNAL_PORT>.
#
# This costs no code change, and that is worth stating explicitly rather than
# discovering later: static/index.html builds its socket URL as
#
#     `${wsProtocol}//${window.location.host}/ws/audio`
#
# -- relative to whatever host:port served the page. main.py serves its own
# static files (app.mount("/")), so the browser hits the same mapped port it
# loaded the page from and the URL resolves correctly on its own.
#
# TWO CONSEQUENCES that DO matter:
#
# 1. Ports must be requested AT RENT TIME in the instance's Docker options:
#        -p 8100:8100 -p 8101:8101 -p 8080:8080 -p 8002:8002
#    They cannot be added to a running instance. 8100 is the only one that
#    strictly must be public; 8080/8002 are internal and are exposed here
#    only for debugging.
#
# 2. Vast.ai maps plain HTTP, not HTTPS. Browsers refuse getUserMedia() on
#    an insecure origin -- EXCEPT for localhost. So a caller on a remote
#    machine cannot grant mic access over http://<IP>:8100 and the Start
#    Call button will fail with a permission error that looks like a mic
#    problem but is not one. Reach it through an SSH tunnel instead:
#
#        ssh -p <SSH_PORT> -L 8100:localhost:8100 root@<PUBLIC_IP>
#
#    then open http://localhost:8100 -- a secure origin, mic works.
# PORTS ARE CHOSEN TO MATCH WHAT THE STOCK TEMPLATE ALREADY MAPS.
# vastai/base-image:cuda-12.8.1-auto ships with exactly these published:
#     1111 (Instance Portal)  6006 (Tensorboard)  8080 (Jupyter + Jupyter
#     Terminal)  8384 (Syncthing)  10100  10200  72299
# A vast.ai instance's published ports are fixed AT CREATION and cannot be
# added later, and adding 8100/8101 meant saving a customised template -- which
# vast.ai creates as a PUBLIC template under the account. Reusing the spare
# already-mapped ports avoids publishing anything and needs no template at all.
#
#   voice agent PCM (main_pcm) -> 10100   (external; this is the caller's URL)
#   voice agent WebM (main.py) -> 10200   (external; legacy transport, kept
#                                          reachable for A/B and rollback)
#   clinic-api                 -> 8081    (internal only, never published)
#   TTS                        -> 8002    (internal only, never published)
#
# clinic-api and TTS are only ever reached over localhost by main.py, so they
# do not need to be published at all -- see CLINIC_API_BASE/TTS_URL above.
#
# WHY main_pcm IS ON THE CALLER-FACING PORT, NOT main.py
# ------------------------------------------------------
# These two values were swapped deliberately. main.py's transport is WebM via
# MediaRecorder, and MediaRecorder writes the container header ONLY into the
# first chunk of a session, so a still-growing buffer can only be read by
# re-decoding it from byte 0. main.py's _decode_to_wav() therefore spawns an
# ffmpeg process every POLL_INTERVAL_S (0.5s) and re-decodes the WHOLE call
# each time -- O(T) per poll, O(T^2) per call. agent/pcm_buffer.py's docstring
# does the arithmetic: a 90-second call decodes ~8,100 seconds of audio, a 90x
# amplification, plus ~2 process spawns per second PER CALL. That saturates CPU
# long before the GPU is busy and is the hard ceiling on concurrent callers.
#
# main_pcm.py sends raw pcm_s16le instead: no container, so no header to be
# missing, so nothing to re-decode. Appending is O(chunk) and reading the
# unprocessed tail is O(tail). No ffmpeg, no subprocess, no temp files.
#
# Both apps are self-contained -- main.py serves static/, main_pcm.py serves
# static/pcm/ (its own AudioWorklet client), and each client builds its socket
# URL from window.location.host -- so whichever app owns a port serves a
# matching client. Swapping the values below is all it takes to move callers
# between transports; nothing else in the stack needs to know.
#
# NOTE the variable names describe WHICH APP, not which role: start_all.sh
# binds main:app to VOICE_AGENT_PORT and main_pcm:app to VOICE_AGENT_PCM_PORT.
# So VOICE_AGENT_PCM_PORT=10100 is what puts the PCM app in front of callers.
# To roll back to the WebM transport, swap these two values back.
export VOICE_AGENT_PORT=10200
export VOICE_AGENT_PCM_PORT=10100
export CLINIC_API_PORT=8081
export TTS_PORT=8002

# Reaching it: vast.ai publishes plain HTTP on <PUBLIC_IP>:<mapped 10100>, and
# browsers refuse getUserMedia() on an insecure origin, so the mic will fail
# there. Use the Instance Portal / Jupyter Terminal for shell access, and for
# actually placing a call open the page over a tunnel so the origin is secure.
