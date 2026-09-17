# CLAUDE.md -- Kolkata Care Diagnostics voice agent

A Bengali/Hindi/English phone line for a diagnostic clinic: ASR (`agent/asr.py`),
intent extraction (`agent/llm.py`, fast path `agent/fast_path.py`), clinic API
calls (`agent/tools_client.py` -> `clinic-api/`), templated replies
(`agent/reply_templates.py`, `agent/i18n.py`), TTS. `main.py` is the WebM
transport; `main_pcm.py` is GENERATED from it by `python tools/make_pcm_variant.py`
-- edit `main.py`, then regenerate. See `README.md` for the architecture.

This is a healthcare system. Patient safety and patient privacy outrank speed.

Claude does not commit or push unless the user explicitly asks.

## Project safety rules

- PHI -- phone numbers, names, dates of birth, PINs, transcripts, medical history,
  tokens -- never reaches logs, exception messages or audit records beyond the existing
  design (`agent/call_audit.py` redacts verification answers, tokens and history).
- The LLM never states a price, schedule, confirmation number or medical fact. Replies
  are templated from backend data; `direct_reply_bn` exists only for smalltalk.
- Patient history is disclosed only after verification and only on a private audio path
  (`agent/privacy.py`); an unclassified path counts as unsafe.
- Every flow completes without a smartphone: no links, QR codes, apps or online payment.
  The clinic counter is the fallback for everything.
- A backend failure is never worded as "not found", and "not found" is never worded as
  a failure.

## Commands

```bash
python tools/make_pcm_variant.py                  # after any edit to main.py
```
