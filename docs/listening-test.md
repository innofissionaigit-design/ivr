# Figure listening test — protocol

> story title: Figures are spoken at a pace a caller can write down
> user story: As a patient noting a price or a reference, I want it grouped and
> slower, so that I do not have to ask twice.
> acceptance criteria: Prices, phone numbers and reference identifiers are
> spoken with grouping and a reduced rate through the per-request speed
> parameter. A listening test confirms callers transcribe correctly on first
> hearing.

This document is the third clause of that criterion. **No amount of code
closes it.** The question is whether a human being, hearing a number once,
over a phone, writes down the right one — and only a human being can answer
that.

## What this study is NOT

It is **not** E12-S11. That story is a different study with a different
population and a different output: *at least thirty patients* rating
**naturalness, warmth and clarity**, recruited across Bengali-matrix,
Hindi-matrix and English-matrix speech, segmented by age and by mixture, with
findings driving template revision before the traffic ramp.

This one is narrow and mechanical: **can you write the number down.** It needs
far fewer participants and answers nothing about tone. Running this one does
not discharge E12-S11, and a reviewer who sees "listening test: done" against
both is being misled.

## Stimuli

Generate with:

```bash
python tools/listening_stimuli.py --out stimuli/
```

That emits the stimulus **text** and an answer key. Rendering to audio happens
on the pod, through the real TTS, because the whole point is to hear what the
caller hears:

```bash
curl -s -X POST localhost:5002/synthesize \
  -H 'content-type: application/json' \
  -d '{"text": "<stimulus>", "lang": "bn"}' -o clip.wav
```

Every stimulus is a figure **inside an ordinary carrier sentence**, never a
bare number. A caller never hears a number in isolation, and testing one in
isolation measures something the service does not do.

## Method

- Playback on a **handset earpiece**, not laptop speakers or headphones.
- Each clip played **once**. No repeats, no pausing. "First hearing" is the
  criterion; a participant who can get it on the second hearing has told us
  nothing about the story.
- Participant writes what they heard. Transcription, not recognition — no
  multiple choice.
- Randomise stimulus order per participant so fatigue does not land on the
  same figure type every time.

## Scoring

Scored **per figure type**, because the failure modes are not comparable:

| Type | Rule | Rationale |
|---|---|---|
| Phone number | **Exact match.** Any digit wrong, missing, extra or transposed is a failure. | A number that is one digit off sends a report to a stranger. There is no partial credit for "nearly". |
| Reference ID | **Exact match**, letters and digits, in order. Case and separators ignored. | Same. The counter either finds the appointment or does not. |
| Price | **Value match.** Formatting, currency word and grouping ignored. | ₹1250 written as "1,250" or "1250 taka" is correct. ₹1250 written as ₹12500 is not, and that is the failure that costs a caller money. |

Report accuracy per type, and report **which** digits went wrong when they
did — a systematic failure at a group boundary means the grouping is wrong,
while scattered failures mean the rate is wrong. Those have different fixes.

**No threshold is fixed here.** Setting one before a pilot would be inventing
a number, which is exactly what the rest of this story is trying to stop.
Run a pilot, look at where the distribution actually sits, then set the bar in
the artifact and justify it there.

## Result artifact

Write the result to `docs/evaluations/figure-listening-<yyyy-mm>.json`.
`tests/test_figure_pacing.py` validates any file in that directory against the
required schema — so a malformed or half-filled result fails the build, while
the absence of a result is simply the absence of a result.

Required fields are listed in `EVALUATION_SCHEMA` in that test. The two that
matter most and are easiest to forget:

- `grouping_policy_version` — from `agent/bn_normalize.py`. A result that does
  not say which policy it heard cannot be acted on later.
- `figure_length_scale` — the rate the clips were rendered at, or `null` if
  the per-figure rate was not yet implemented when the study ran.

That second field is why this document exists before the feature is finished:
the grouping half shipped, the per-figure rate did not, and a study run today
would be measuring grouping alone. That is a useful thing to measure — it just
has to say so.
