"""Clinic data service -- implements the exact 3-endpoint contract
agent/tools_client.py in the voice agent already expects. Backed by
PostgreSQL, seeded with dummy departments/doctors/schedules/tests via
seed.py.

Entity matching lives in match_band.py, which scores every spelling the
catalogue holds and returns one of three verdicts -- commit, ambiguous,
none. It is still simple (difflib plus a containment tier) rather than the
phonetic-fold gazetteer that production callers slurring "লিপিড প্রোফাইল"
through a phone mic deserve. What changed is that it no longer resolves an
ambiguity by picking a row: several plausible rows come back as several, and
the agent asks.
"""
from __future__ import annotations

import logging

import datetime
import uuid

from fastapi import FastAPI, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

import match_band
from db import get_db, SessionLocal
from models import Department, Doctor, DoctorSchedule, LabTest, Appointment

app = FastAPI(title="Kolkata Care Diagnostics -- Clinic Data API (dummy)")

SLOT_STEP_MIN = 15


@app.on_event("startup")
def _ensure_seeded():
    """Create the schema, and seed it only if it is EMPTY.

    The catalogue is derived data -- 8 departments, 32 doctors, 34 tests,
    all defined in seed.py -- so regenerating it costs nothing and removes
    the manual reseed step that every pod restart used to require.

    Guarded on emptiness because seed() itself is destructive (drop_all
    then create_all). Running it unconditionally at startup would wipe
    every appointment booked since the last boot, turning a convenience
    into data loss.
    """
    from db import engine
    from models import Base, LabTest
    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        if db.query(LabTest).count() == 0:
            logging.getLogger("clinic-api").info("empty database -- seeding catalogue")
            from seed import seed
            seed()
        else:
            logging.getLogger("clinic-api").info("catalogue already present, not reseeding")
    finally:
        db.close()


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    return {
        "status": "ok",
        "departments": db.query(Department).count(),
        "doctors": db.query(Doctor).count(),
        "lab_tests": db.query(LabTest).count(),
    }


# =============================================================================
# Tool 1: GET /api/v1/tests/search?name=...
# =============================================================================
def _first_alias_bn(aliases_bn: str) -> str | None:
    """The spoken form. The reply the caller HEARS is synthesized by a
    Bengali-only tokenizer that silently drops Latin script, so returning
    only `t.name` ("Uric Acid") means the caller is read a price with the
    test name missing from the sentence. Every row is seeded with at least
    one Bengali alias for exactly this reason -- see seed.py."""
    for alias in (aliases_bn or "").split("|"):
        if alias.strip():
            return alias.strip()
    return None


def _test_reply_dict(t: LabTest) -> dict:
    return {
        "found": True, "test_name": t.name, "test_name_bn": _first_alias_bn(t.aliases_bn),
        "rate_inr": t.rate_inr,
        "sample_type": t.sample_type, "report_time_hours": t.report_time_hours,
    }


# story title: Near matches are offered rather than guessed or refused
# user story: As a caller naming something loosely, I want the close matches
#   offered, so that I am not told my test does not exist when it does.
# acceptance criteria: When several catalogue rows fall within the match band
#   the agent offers up to three by name and asks which. Candidates are
#   generated across every supported language and romanised spelling. The
#   did-you-mean path covers the ambiguous case and not only total failure.
#
# EVERY FORM THE CATALOGUE HOLDS, in every script, scored through one
# function. The English canonical name and the Bengali aliases go into the
# same pool because a caller may say either and the ASR may land on either --
# the same argument that put aliases_bn into the old fuzzy fallback, applied
# now to the whole lookup rather than only to its last resort.
def _forms(row, *, extra=()) -> list[str]:
    aliases = [a.strip() for a in (row.aliases_bn or "").split("|") if a.strip()]
    return [row.name, *extra, *aliases]


def _candidate_dicts(rows) -> list[dict]:
    """-> [{"name", "name_bn"}, ...], or [] if ANY row cannot be said aloud.

    All or nothing, deliberately. Dropping the one candidate that has no
    Bengali alias would turn "which of these two did you mean" back into
    "did you mean this one" -- a guess wearing a question mark, which is the
    exact failure this story exists to remove. An empty list tells the agent
    to ask the caller to name it again instead, which is honest.

    Every seeded row has an alias today (seed.py guarantees it), so this is a
    trap being closed rather than a bug being fixed.
    """
    out = []
    for row in rows:
        spoken = _first_alias_bn(row.aliases_bn)
        if not spoken:
            return []
        out.append({"name": row.name, "name_bn": spoken})
    return out


def _ambiguous_reply(query: str, rows) -> dict:
    """The third response shape, alongside found and not-found.

    `ambiguous` is its own flag rather than an overloaded found=false,
    because agent/tool_outcome.py reads a falsy `found` as NOT_FOUND and an
    ambiguity counted as a thing-not-existing would poison not_found_rate --
    the one metric the preceding story built to tell an empty catalogue from
    a working one.
    """
    return {"found": False, "ambiguous": True, "query": query,
            "candidates": _candidate_dicts(rows)}


@app.get("/api/v1/catalogue")
def catalogue(db: Session = Depends(get_db)):
    """Every test and doctor with their Bengali aliases, in one call.

    Exists for the voice agent's deterministic fast path: matching a
    caller's words against a 74-row catalogue is a local string operation,
    but only if the caller HAS the catalogue. Fetching it once at startup
    turns "which test did they say" from a 7B-model inference into a
    microsecond comparison -- see agent/fast_path.py.
    """
    return {
        "tests": [
            {"name": t.name,
             "aliases_bn": [a for a in (t.aliases_bn or "").split("|") if a]}
            for t in db.query(LabTest).all()
        ],
        "doctors": [
            {"name": d.name,
             "surname": d.name.split()[-1],
             "aliases_bn": [a for a in (d.aliases_bn or "").split("|") if a]}
            for d in db.query(Doctor).all()
        ],
    }


@app.get("/api/v1/tests/search")
def search_test(name: str = Query(...), db: Session = Depends(get_db)):
    # story title: Near matches are offered rather than guessed or refused
    # user story: As a caller naming something loosely, I want the close matches
    #   offered, so that I am not told my test does not exist when it does.
    # acceptance criteria: When several catalogue rows fall within the match band
    #   the agent offers up to three by name and asks which. Candidates are
    #   generated across every supported language and romanised spelling. The
    #   did-you-mean path covers the ambiguous case and not only total failure.
    #
    # WHAT THIS REPLACED, because the shape of the bug is not the shape people
    # expect. There were three passes: an English substring query ending in
    # .first(), a Bengali alias loop ending in `return` on its first hit, and
    # only then a fuzzy fallback. The first two did no scoring AT ALL -- so
    # "Blood Sugar", which matches both "Blood Sugar Fasting" and "Blood Sugar
    # PP", was resolved by row order, silently, and so were "সুগার",
    # "ভিটামিন" and "Vitamin". A runner-up margin alone would not have touched
    # any of them; there was no runner-up to compare against, only a list and
    # an index.
    #
    # Now every form of every row is scored once, and the same question --
    # is second place close? -- is asked on every path. A substring hit is a
    # high score rather than an early return.
    all_tests = db.query(LabTest).all()
    verdict, candidates = match_band.decide(
        match_band.rank(name, [(t, _forms(t)) for t in all_tests]))

    if verdict == match_band.COMMIT:
        return _test_reply_dict(candidates[0].key)

    if verdict == match_band.AMBIGUOUS:
        return _ambiguous_reply(name, [c.key for c in candidates])

    # Nothing cleared match_band.BAND_FLOOR, which is the same 0.50 the old
    # difflib.get_close_matches(cutoff=0.5) used -- so this is exactly the
    # case that used to produce an EMPTY suggestion list, and the keys are
    # still emitted, still empty, so a client reading them sees no change.
    #
    # The non-empty case they used to carry is now the ambiguous branch
    # above. That is the story's third clause: did-you-mean stops being a
    # total-failure consolation and becomes the same mechanism that handles
    # two rows tying at 0.90.
    return {"found": False, "query": name, "did_you_mean": [], "did_you_mean_bn": []}


# =============================================================================
# Tool 2: GET /api/v1/doctors/availability?name=...&date=YYYY-MM-DD (optional)
# Tool 2b: GET /api/v1/doctors/by-department?department=...
# =============================================================================
# FUZZY_SURNAME_FLOOR -- LOW confidence, reasoned not measured (no real call
# audio to calibrate against yet, unlike voicerx/gate.py's SIMILARITY_FLOOR).
#
# This exists because of a bug caught in local testing: matching the raw
# query against the full formatted name ("Dr. A. Sen") let a query for
# "Doctor Nobody" fuzzy-match "Dr. N. Roy" at ratio 0.522 -- HIGHER than the
# ratio for a real garbled name against its own doctor ("sen" vs "Dr. A. Sen"
# scores only 0.462, because SequenceMatcher penalizes the length mismatch
# against the "Dr. X." prefix on both sides, so short queries and wrong
# queries land in the same range). That is this system's own small version
# of the "Naloxone" bug: confidently answering with the wrong doctor's real
# schedule instead of saying "not found".
#
# Fix: match against the SURNAME only, which cleanly separates the two
# cases in testing -- genuine garbles (e.g. "mukharji" vs "Mukherjee")
# scored 0.70-0.80; unrelated queries (e.g. "doctor nobody" vs "Roy")
# scored <=0.44. 0.60 sits in the gap. Recalibrate once real call audio
# exists, the same way gate.py's floors were tightened from real samples.
FUZZY_SURNAME_FLOOR = 0.60


# story title: Near matches are offered rather than guessed or refused
# user story: As a caller naming something loosely, I want the close matches
#   offered, so that I am not told my test does not exist when it does.
# acceptance criteria: When several catalogue rows fall within the match band
#   the agent offers up to three by name and asks which. Candidates are
#   generated across every supported language and romanised spelling. The
#   did-you-mean path covers the ambiguous case and not only total failure.
#
# The surname is passed as an extra form because it is what callers actually
# say -- "Sen", not "Dr. A. Sen" -- and because it is the form that makes the
# tiering matter: "সেন" EQUALS Dr Sen's alias (1.0) and is CONTAINED IN Dr
# Sengupta's (0.90), so it clears the margin and commits, where a flat
# containment score for both would have asked the caller to choose between a
# doctor they named exactly and one they did not.
#
# FUZZY_SURNAME_FLOOR's 0.60 is no longer a commit threshold -- match_band
# commits at 0.72 and OFFERS between 0.50 and 0.72. That band used to be a
# silent commit. The Doctor Nobody incident that set 0.60 in the first place
# (a wrong doctor matched at 0.522) lands in it: the caller is now asked
# "did you mean Dr Roy?" and can say no, instead of being answered about a
# doctor they never named.
def _resolve_doctor(db: Session, name: str) -> tuple[str, Doctor | None, list[Doctor]]:
    """-> (verdict, the one doctor if committing, the rows to offer)."""
    all_doctors = db.query(Doctor).all()
    rows = [(d, _forms(d, extra=[d.name.split()[-1]])) for d in all_doctors]
    verdict, candidates = match_band.decide(match_band.rank(name, rows))
    if verdict == match_band.COMMIT:
        return verdict, candidates[0].key, []
    return verdict, None, [c.key for c in candidates]


# story title: Near matches are offered rather than guessed or refused
# user story: As a caller naming something loosely, I want the close matches
#   offered, so that I am not told my test does not exist when it does.
# acceptance criteria: When several catalogue rows fall within the match band
#   the agent offers up to three by name and asks which. Candidates are
#   generated across every supported language and romanised spelling. The
#   did-you-mean path covers the ambiguous case and not only total failure.
#
# Departments are the one entity type whose aliases ALREADY span scripts --
# "কার্ডিওলজি", "হার্ট", "heart", "cardio". They go into the pool exactly as
# they are, which is why the criterion's romanised-spelling clause is met here
# and not for tests or doctors: it is a data question, and the seed rows for
# those two carry Bengali script plus the English name only. Noted as the
# known gap rather than closed, because seed() is destructive and reseeding is
# an operational decision, not a deploy.
def _resolve_department(db: Session, department_name: str) -> tuple[str, Department | None, list[Department]]:
    """-> (verdict, the one department if committing, the rows to offer)."""
    all_departments = db.query(Department).all()
    verdict, candidates = match_band.decide(
        match_band.rank(department_name, [(d, _forms(d)) for d in all_departments]))
    if verdict == match_band.COMMIT:
        return verdict, candidates[0].key, []
    return verdict, None, [c.key for c in candidates]


def _schedule_for_weekday(db: Session, doctor_id: int, weekday: int) -> DoctorSchedule | None:
    return db.query(DoctorSchedule).filter_by(doctor_id=doctor_id, weekday=weekday).first()


def _next_available_date(db: Session, doctor_id: int, from_date: datetime.date,
                          horizon_days: int = 14) -> str | None:
    for offset in range(horizon_days):
        d = from_date + datetime.timedelta(days=offset)
        if _schedule_for_weekday(db, doctor_id, d.weekday()):
            return d.isoformat()
    return None


@app.get("/api/v1/doctors/availability")
def doctor_availability(name: str = Query(...), date: str | None = Query(None),
                         db: Session = Depends(get_db)):
    verdict, doctor, offered = _resolve_doctor(db, name)
    if verdict == match_band.AMBIGUOUS:
        return _ambiguous_reply(name, offered)
    if not doctor:
        return {"found": False, "query": name}

    today = datetime.date.today()

    if date:
        try:
            target = datetime.date.fromisoformat(date)
        except ValueError:
            return {"found": False, "query": name}
        sched = _schedule_for_weekday(db, doctor.id, target.weekday())
        if sched:
            return {
                "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": target.isoformat(),
                "available": True, "chamber_hours": f"{sched.start_time}-{sched.end_time}",
                "next_available_date": None,
            }
        next_date = _next_available_date(db, doctor.id, target + datetime.timedelta(days=1))
        return {
            "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": target.isoformat(),
            "available": False, "chamber_hours": None, "next_available_date": next_date,
        }

    # No date given -> "when is this doctor next available"
    next_date = _next_available_date(db, doctor.id, today)
    if not next_date:
        return {
            "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": None,
            "available": False, "chamber_hours": None, "next_available_date": None,
        }
    sched = _schedule_for_weekday(db, doctor.id, datetime.date.fromisoformat(next_date).weekday())
    return {
        "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": next_date,
        "available": True, "chamber_hours": f"{sched.start_time}-{sched.end_time}",
        "next_available_date": None,
    }


@app.get("/api/v1/doctors/by-department")
def doctors_by_department(department: str = Query(...), date: str | None = Query(None),
                           db: Session = Depends(get_db)):
    """Get all doctors in a department, supports short forms like 'ortho' for
    Orthopaedics.

    `date` is OPTIONAL, same convention as /doctors/availability above.
    Omit it for a plain "who's in this department" listing (every doctor,
    unfiltered -- unchanged from before this parameter existed). Pass it
    for "which ortho doctor is available TODAY/that day": the list is
    filtered down to doctors who actually have a DoctorSchedule row for
    that date's weekday, and each gets its chamber_hours attached, mirroring
    what doctor_availability() already reports for a single named doctor.
    """
    verdict, dept, offered = _resolve_department(db, department)
    if verdict == match_band.AMBIGUOUS:
        return _ambiguous_reply(department, offered)
    if not dept:
        return {"found": False, "query": department}

    doctors = db.query(Doctor).filter_by(department_id=dept.id).all()

    target = None
    if date:
        try:
            target = datetime.date.fromisoformat(date)
        except ValueError:
            target = None  # malformed date -- fall back to the unfiltered listing

    out = []
    for d in doctors:
        entry = {
            "name": d.name,
            "doctor_name_bn": _first_alias_bn(d.aliases_bn),
            "qualifications": d.qualifications,
        }
        if target is not None:
            sched = _schedule_for_weekday(db, d.id, target.weekday())
            if not sched:
                continue  # doesn't sit that day -- excluded, not just flagged
            entry["chamber_hours"] = f"{sched.start_time}-{sched.end_time}"
        out.append(entry)

    return {
        "found": True,
        "department": dept.name,
        # STORY [Answer Quality and Grounding]
        # As a patient, I want to hear the whole sentence, so that I am
        # not left guessing what the agent tried to say.
        # The SPOKEN department name. Without it the agent's listing reply
        # reads "<English> বিভাগে ... আছেন" and the Bengali tokenizer drops the
        # Latin word, so the caller loses the SUBJECT of the sentence -- on
        # every one of the eight seeded departments, not an edge case. Same
        # helper, same reason, as test_name_bn and doctor_name_bn above; every
        # department is seeded with a Bengali alias first (see seed.py's
        # DEPARTMENT_ALIASES), so this needs no new data.
        "department_bn": _first_alias_bn(dept.aliases_bn),
        "date": target.isoformat() if target else None,
        "doctors": out,
    }


# =============================================================================
# Tool 3: POST /api/v1/appointments
# =============================================================================
class BookingRequest(BaseModel):
    doctor_name: str
    date: str
    time_slot: str
    patient_name: str
    phone: str


def _generate_slots(start: str, end: str, step_min: int = SLOT_STEP_MIN) -> list[str]:
    t = datetime.datetime.strptime(start, "%H:%M")
    end_t = datetime.datetime.strptime(end, "%H:%M")
    slots = []
    while t < end_t:
        slots.append(t.strftime("%H:%M"))
        t += datetime.timedelta(minutes=step_min)
    return slots


@app.post("/api/v1/appointments")
def book_appointment(req: BookingRequest, db: Session = Depends(get_db)):
    # story title: Near matches are offered rather than guessed or refused
    # user story: As a caller naming something loosely, I want the close matches
    #   offered, so that I am not told my test does not exist when it does.
    # acceptance criteria: When several catalogue rows fall within the match band
    #   the agent offers up to three by name and asks which. Candidates are
    #   generated across every supported language and romanised spelling. The
    #   did-you-mean path covers the ambiguous case and not only total failure.
    #
    # A WRITE NEVER PROCEEDS UNDER AMBIGUITY. Booking the higher-scoring of
    # two plausible doctors is the worst version of this bug: the caller
    # leaves believing they have an appointment, and they do -- with someone
    # else. The refusal carries the candidates so the agent can ask, rather
    # than reporting "no such doctor" for a doctor who exists twice over.
    verdict, doctor, offered = _resolve_doctor(db, req.doctor_name)
    if verdict == match_band.AMBIGUOUS:
        return {"success": False, "reason": "doctor_ambiguous",
                "candidates": _candidate_dicts(offered)}
    if not doctor:
        return {"success": False, "reason": "doctor_not_found"}

    try:
        target = datetime.date.fromisoformat(req.date)
    except ValueError:
        return {"success": False, "reason": "missing_field"}

    sched = _schedule_for_weekday(db, doctor.id, target.weekday())
    if not sched:
        # Doctor doesn't sit that day at all -- not in the caller-facing
        # reason enum reply_templates.booking_reply() specifically handles,
        # so it falls to that function's generic "couldn't book" message,
        # which remains true and safe rather than a false "slot taken".
        return {"success": False, "reason": "doctor_not_available_that_day"}

    valid_slots = _generate_slots(sched.start_time, sched.end_time)
    if req.time_slot not in valid_slots:
        return {"success": False, "reason": "slot_taken", "alternative_slots": valid_slots[:3]}

    taken = {
        a.time_slot for a in db.query(Appointment).filter_by(
            doctor_id=doctor.id, date=req.date,
        ).all()
    }
    if req.time_slot in taken:
        free = [s for s in valid_slots if s not in taken][:3]
        return {"success": False, "reason": "slot_taken", "alternative_slots": free}

    confirmation_id = f"KCD-{req.date.replace('-', '')}-{uuid.uuid4().hex[:4].upper()}"
    appt = Appointment(
        confirmation_id=confirmation_id, doctor_id=doctor.id, date=req.date,
        time_slot=req.time_slot, patient_name=req.patient_name, phone=req.phone,
        created_at=datetime.datetime.now(),
    )
    db.add(appt)
    db.commit()

    return {
        "success": True, "confirmation_id": confirmation_id,
        "doctor_name": doctor.name,
        "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": req.date, "time_slot": req.time_slot,
    }
