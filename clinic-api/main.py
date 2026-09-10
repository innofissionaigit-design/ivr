"""Clinic data service -- implements the exact 3-endpoint contract
agent/tools_client.py in the voice agent already expects. Backed by
PostgreSQL, seeded with dummy departments/doctors/schedules/tests via
seed.py.

Matching is deliberately simple (ILIKE + difflib) for this prototype --
production callers slurring "লিপিড প্রোফাইল" through a phone mic deserve
something closer to voicerx/glossary.py's phonetic-fold gazetteer, not a
plain substring match. Flagged here rather than silently left as if this
were already that robust.

======================================================================
UPDATED BY SOURAV -- "Lab Report Status & Secure Delivery" combined
story (previously two separate stories: "is my report ready" and "send
my report"), per the VOICE CARE AGENT FINAL TESTING / STORY / RULES /
EDGE-CASE ATTACK PLAN doc.

Added below (search "SOURAV" for every changed/new block):
  - GET  /api/v1/reports/status           (Rule 1-3, 13, 14; Section 4)
  - POST /api/v1/reports/delivery/request (Rule 3, 4, 15, 16)
  - POST /api/v1/reports/otp/verify       (Rule 4-9, 17; the whole OTP
                                            attack surface in Section 6)
  - GET  /api/v1/reports/link/{token}     (Rule 11, 12; Section 8 link
                                            attacks)

FAIL SAFE, NOT FAIL OPEN (plan Section 23) is the guiding principle for
every one of these: every endpoint below returns a NAMED reason instead
of a generic error/boolean whenever it refuses to act, precisely so the
voice agent (main.py / main_pcm.py) never has to guess why something
didn't happen, and never defaults to acting just because a check was
inconclusive.
======================================================================
"""
from __future__ import annotations

import logging

import datetime
import difflib
import secrets
import uuid

from fastapi import FastAPI, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from db import get_db, SessionLocal
from models import (
    Department, Doctor, DoctorSchedule, LabTest, Appointment,
    # SOURAV: needed for the report-status/delivery/OTP endpoints below.
    Patient, LabReport, ReportOTP, ReportDelivery,
)

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
    # English substring match -- covers callers who say the test name in
    # English/transliterated form.
    exact = db.query(LabTest).filter(func.lower(LabTest.name).contains(name.lower())).first()
    if exact:
        return _test_reply_dict(exact)

    # Bengali-script match -- covers the actual common case. A caller
    # saying "ইউরিক এসিড" was matched against nothing before this existed:
    # the DB only stored the English name "Uric Acid", and Bengali script
    # shares zero characters with Latin script, so substring AND fuzzy
    # matching against the English column alone can NEVER succeed on
    # Bengali input, regardless of how close the pronunciation is.
    all_tests = db.query(LabTest).all()
    for t in all_tests:
        aliases = [a for a in t.aliases_bn.split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            return _test_reply_dict(t)

    # Fuzzy fallback -- try both the English name and every Bengali alias,
    # so suggestions are useful regardless of which script the caller used.
    candidates = []
    for t in all_tests:
        candidates.append(t.name)
        candidates.extend(a for a in t.aliases_bn.split("|") if a)
    suggestions = difflib.get_close_matches(name, candidates, n=3, cutoff=0.5)
    # Map suggested aliases back to their canonical English name for display.
    alias_to_name = {a: t.name for t in all_tests for a in t.aliases_bn.split("|") if a}
    suggestions = list(dict.fromkeys(alias_to_name.get(s, s) for s in suggestions))
    return {"found": False, "query": name, "did_you_mean": suggestions}


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


def _find_doctor(db: Session, name: str) -> Doctor | None:
    # English substring match (e.g. "Sen", "Dr Sen").
    exact = db.query(Doctor).filter(func.lower(Doctor.name).contains(name.lower())).first()
    if exact:
        return exact

    all_doctors = db.query(Doctor).all()

    # Bengali-script exact match -- a real caller says "ডক্টর সেন", which
    # shares no characters with the Latin "Dr. A. Sen" stored as the
    # canonical name. Same root cause and same fix as search_test()'s
    # aliases_bn check.
    for d in all_doctors:
        aliases = [a for a in d.aliases_bn.split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            return d

    # Fuzzy fallback, against BOTH the English surname and the Bengali
    # alias(es) -- garbled ASR output can land on either script depending
    # on what the caller actually said and how the decoder heard it.
    best_doctor, best_ratio = None, 0.0
    for d in all_doctors:
        candidates = [d.name.split()[-1].lower()] + [a for a in d.aliases_bn.split("|") if a]
        for c in candidates:
            ratio = difflib.SequenceMatcher(None, name.lower(), c.lower()).ratio()
            if ratio > best_ratio:
                best_doctor, best_ratio = d, ratio

    return best_doctor if best_ratio >= FUZZY_SURNAME_FLOOR else None


def _find_department(db: Session, department_name: str) -> Department | None:
    """Find department by name or alias (e.g., 'ortho' for Orthopaedics)."""
    # Exact match first
    exact = db.query(Department).filter(func.lower(Department.name).contains(department_name.lower())).first()
    if exact:
        return exact

    # Match against aliases
    all_departments = db.query(Department).all()
    for dept in all_departments:
        aliases = [a for a in dept.aliases_bn.split("|") if a]
        if any(department_name.lower() in alias.lower() or alias.lower() in department_name.lower() for alias in aliases):
            return dept

    # Fuzzy match
    best_dept, best_ratio = None, 0.0
    for dept in all_departments:
        candidates = [dept.name.lower()] + [a.lower() for a in dept.aliases_bn.split("|") if a]
        for c in candidates:
            ratio = difflib.SequenceMatcher(None, department_name.lower(), c).ratio()
            if ratio > best_ratio:
                best_dept, best_ratio = dept, ratio

    return best_dept if best_ratio >= 0.6 else None


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
    doctor = _find_doctor(db, name)
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
    dept = _find_department(db, department)
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
    doctor = _find_doctor(db, req.doctor_name)
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


# =============================================================================
# Tool 5: GET /api/v1/reports/status?phone=...&test_name=... (optional)
# Tool 6: POST /api/v1/reports/delivery/request
# Tool 7: POST /api/v1/reports/otp/verify
# Tool 8: GET /api/v1/reports/link/{token}
#
# ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story.
# See this file's module docstring for the rule references. Every helper
# and endpoint below is new; nothing above this line was touched except
# the import block.
# =============================================================================

OTP_VALIDITY_MINUTES = 10
SIGNED_LINK_VALIDITY_MINUTES = 15
# Prototype-only fixed OTP for a report that has no pre-seeded ReportOTP
# row (or whose only rows are used/expired/maxed) when delivery is
# requested live. The user's own explicit decision for this story: "the
# otp [is] hardcoded, for now user will tell the otp and the matching
# will be done" -- there is no real SMS/e-mail provider behind this
# prototype, so whichever code is "sent" has to be knowable in advance
# for testing. Seeded patients that already carry their own ReportOTP
# row (Arjun/Sohini/Amit -- see seed.py SECTION 9) always take priority
# over this constant; it only fires for reports with no usable row yet
# (e.g. Mita's and the two Rahul Das reports), so every READY,
# delivery-enabled report in the seed data can be driven through a full
# live OTP flow, not just the three pre-scripted ones.
FRESH_OTP_CODE = "135790"


def _mask_phone_last4(phone: str) -> str:
    """Rule 10: never say the full registered number back to the caller.
    Used only in this file's own response payloads (masked_phone) --
    reply_templates.py has its own copy for anything it composes
    directly from a phone slot the caller spoke, since that module must
    not import from clinic-api (they are separate deployables)."""
    digits = "".join(c for c in phone if c.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits


def _find_patient_by_phone(db: Session, phone: str) -> Patient | None:
    return db.query(Patient).filter_by(phone=phone).first()


def _report_summary(report: LabReport, test_name: str) -> dict:
    return {
        "report_number": report.report_number,
        "test_name": test_name,
        "status": report.status,
        "delivery_enabled": report.delivery_enabled,
        "expected_ready_at": report.expected_ready_at.isoformat() if report.expected_ready_at else None,
        "ready_at": report.ready_at.isoformat() if report.ready_at else None,
    }


@app.get("/api/v1/reports/status")
def report_status(phone: str = Query(...), test_name: str | None = Query(None),
                   db: Session = Depends(get_db)):
    """RULE 1 (never invent a report), RULE 13 (multiple reports require
    clarification), RULE 14 (same-name patients must not be merged).

    Identity is resolved by PHONE, never by name -- this is what makes
    RULE 14 hold structurally rather than by convention: Patient.phone is
    a unique column (see models.py), so two "Rahul Das" rows can never
    collide here regardless of how the caller pronounces the name. A
    caller who states only a name and no phone is a slot the voice agent
    must ask for BEFORE calling this endpoint at all (see main_pcm.py's
    new "phone" pending state) -- this endpoint has no name-based lookup
    path to accidentally fall back to.
    """
    patient = _find_patient_by_phone(db, phone)
    if not patient:
        return {"patient_found": False}

    reports = db.query(LabReport).filter_by(patient_id=patient.id).all()
    if not reports:
        # RULE 1: an honest NOT_FOUND, never a fabricated report.
        return {"patient_found": True, "found": False, "reason": "NOT_FOUND"}

    lab_test_ids = {r.lab_test_id for r in reports}
    tests_by_id = {t.id: t for t in db.query(LabTest).filter(LabTest.id.in_(lab_test_ids)).all()}

    if test_name:
        matches = [r for r in reports if test_name.lower() in tests_by_id[r.lab_test_id].name.lower()]
    else:
        matches = reports

    if not matches:
        # A test_name was given but nothing this patient has matches it --
        # honest NOT_FOUND rather than silently falling back to "all
        # reports" (which would let a misheard test name return an
        # unrelated report).
        return {"patient_found": True, "found": False, "reason": "NOT_FOUND"}

    if len(matches) > 1:
        # RULE 13: multiple candidates -> ask, never guess.
        return {
            "patient_found": True, "found": False, "reason": "AMBIGUOUS",
            "candidates": [_report_summary(r, tests_by_id[r.lab_test_id].name) for r in matches],
        }

    report = matches[0]
    return {"patient_found": True, "found": True, **_report_summary(report, tests_by_id[report.lab_test_id].name)}


def _blocking_reason_for(report: LabReport) -> str | None:
    """RULE 3 / RULE 16: only a READY, delivery-enabled report may enter
    the delivery flow. Returns the specific reason delivery is blocked,
    or None if it is allowed to proceed -- used identically by both the
    delivery-request and otp-verify endpoints below, so a report that
    changes state between those two calls (e.g. cancelled in between) is
    re-checked, not trusted from the first call."""
    if report.status != "READY":
        return report.status  # NOT_READY / PROCESSING / CANCELLED
    if not report.delivery_enabled:
        return "DELIVERY_DISABLED"
    return None


def _resolve_patient_and_report(db: Session, phone: str, report_number: str):
    """Shared identity+report resolution for delivery/request and
    otp/verify. Returns (patient, report, error_reason). error_reason is
    None on success.

    RULE 15: the report must belong to the PHONE-resolved patient, not
    merely exist. A real report_number for a DIFFERENT patient (ATTACK
    15) and a nonexistent one both return the same "NOT_FOUND" -- never a
    distinct "wrong patient" reason, which would let an attacker probe
    which report numbers are real by watching the error change.
    """
    patient = _find_patient_by_phone(db, phone)
    if not patient:
        return None, None, "PATIENT_NOT_FOUND"

    report = db.query(LabReport).filter_by(report_number=report_number).first()
    if not report or report.patient_id != patient.id:
        return patient, None, "NOT_FOUND"

    return patient, report, None


class DeliveryRequest(BaseModel):
    phone: str
    report_number: str


@app.post("/api/v1/reports/delivery/request")
def request_report_delivery(req: DeliveryRequest, db: Session = Depends(get_db)):
    """RULE 4 (OTP required before delivery), RULE 6/7/8 (an exhausted,
    expired or already-used OTP is never silently reused -- a fresh
    request always gets a fresh, USABLE OTP row instead)."""
    patient, report, error = _resolve_patient_and_report(db, req.phone, req.report_number)
    if error:
        return {"success": False, "reason": error}

    blocked = _blocking_reason_for(report)
    if blocked:
        return {"success": False, "reason": blocked}

    now = datetime.datetime.now()

    # Reuse the most recent OTP row for this (report, patient) ONLY if it
    # is still usable. RULE 8's "require a new verification flow" is what
    # this is: a maxed-out, expired or already-used row is treated as
    # dead, and a brand-new row is minted instead of ever handing the
    # same exhausted OTP back out.
    active = (
        db.query(ReportOTP)
        .filter_by(report_id=report.id, patient_id=patient.id)
        # UPDATED BY SOURAV -- tie-break on id, not just created_at.
        # Two OTP rows for the same (report, patient) can share an
        # identical created_at timestamp (Python's datetime.now() and
        # SQLite's storage resolution both make this a real possibility,
        # not just a seed-data artifact -- clinic-api/seed.py's own
        # Patient E rows originally did exactly this before being fixed
        # to use distinct timestamps). Without a secondary sort key,
        # "most recent" was undefined on a tie -- caught by
        # tests/test_clinic_api_reports.py::TestOtpVerify::
        # test_max_attempts_already_reached returning OTP_EXPIRED instead
        # of OTP_MAX_ATTEMPTS. `id` is autoincrement, so it is a reliable
        # insertion-order tie-breaker regardless of what the clock reads.
        .order_by(ReportOTP.created_at.desc(), ReportOTP.id.desc())
        .first()
    )
    reusable = (
        active is not None
        and not active.used
        and active.expires_at > now
        and active.attempt_count < active.max_attempts
    )

    if not reusable:
        active = ReportOTP(
            report_id=report.id,
            patient_id=patient.id,
            phone=patient.phone,
            otp_code=FRESH_OTP_CODE,
            created_at=now,
            expires_at=now + datetime.timedelta(minutes=OTP_VALIDITY_MINUTES),
            used=False,
            attempt_count=0,
        )
        db.add(active)
        db.commit()

    return {
        "success": True, "reason": "OTP_REQUIRED",
        "masked_phone": _mask_phone_last4(patient.phone),
    }


class OtpVerifyRequest(BaseModel):
    phone: str
    report_number: str
    otp_code: str
    # SOURAV: test-only hook for CASE 2 ("Delivery failure tests" /
    # provider failure) in the attack plan's Section 9. The voice agent
    # itself never sets this -- there is no real SMS/e-mail provider in
    # this prototype to fail on its own, so this is how the automated
    # test suite deterministically exercises "OTP was right, but the
    # provider failed" (RULE 17) without needing real delivery infra.
    simulate_delivery_failure: bool = False


@app.post("/api/v1/reports/otp/verify")
def verify_report_otp(req: OtpVerifyRequest, db: Session = Depends(get_db)):
    """The entire OTP attack surface (plan Section 6) lives in this one
    function, in a fixed order, on purpose -- RULE 9 says never reveal
    the correct OTP, so the code is compared LAST, after every other
    reason to refuse has already been ruled out; an attacker never
    learns anything about the correct value from which check fired."""
    patient, report, error = _resolve_patient_and_report(db, req.phone, req.report_number)
    if error:
        return {"success": False, "reason": error}

    blocked = _blocking_reason_for(report)
    if blocked:
        return {"success": False, "reason": blocked}

    now = datetime.datetime.now()

    active = (
        db.query(ReportOTP)
        .filter_by(report_id=report.id, patient_id=patient.id)
        # UPDATED BY SOURAV -- tie-break on id, not just created_at.
        # Two OTP rows for the same (report, patient) can share an
        # identical created_at timestamp (Python's datetime.now() and
        # SQLite's storage resolution both make this a real possibility,
        # not just a seed-data artifact -- clinic-api/seed.py's own
        # Patient E rows originally did exactly this before being fixed
        # to use distinct timestamps). Without a secondary sort key,
        # "most recent" was undefined on a tie -- caught by
        # tests/test_clinic_api_reports.py::TestOtpVerify::
        # test_max_attempts_already_reached returning OTP_EXPIRED instead
        # of OTP_MAX_ATTEMPTS. `id` is autoincrement, so it is a reliable
        # insertion-order tie-breaker regardless of what the clock reads.
        .order_by(ReportOTP.created_at.desc(), ReportOTP.id.desc())
        .first()
    )
    if active is None:
        return {"success": False, "reason": "OTP_NOT_REQUESTED"}

    if active.used:
        # RULE 7: an OTP is single-use, full stop -- even if the caller
        # types the exact right value again.
        return {"success": False, "reason": "OTP_ALREADY_USED"}

    if active.expires_at <= now:
        # RULE 6.
        return {"success": False, "reason": "OTP_EXPIRED"}

    if active.attempt_count >= active.max_attempts:
        # RULE 8 -- checked BEFORE the code comparison, so a caller who
        # is already locked out never gets another "wrong"/"right"
        # signal about a code that no longer matters.
        return {"success": False, "reason": "OTP_MAX_ATTEMPTS"}

    if active.otp_code != req.otp_code:
        # RULE 9: the response says only that it was wrong, never what
        # the right one is.
        active.attempt_count += 1
        db.commit()
        if active.attempt_count >= active.max_attempts:
            return {"success": False, "reason": "OTP_MAX_ATTEMPTS"}
        return {"success": False, "reason": "OTP_INVALID"}

    # Correct code, and every gate above passed -- RULE 5 held throughout
    # because `active` was scoped to (report.id, patient.id) from the
    # very first query above: an OTP that is valid for a DIFFERENT report
    # or patient is simply never the row being compared against here, so
    # ATTACK 5/6 (cross-report / cross-patient OTP reuse) fail not
    # because of a special case, but because the right row was never
    # found in the first place.
    active.used = True
    active.verified_at = now
    db.commit()

    if req.simulate_delivery_failure:
        # RULE 17: OTP success must NOT be conflated with delivery
        # success -- the two are recorded and reported separately.
        delivery = ReportDelivery(
            report_id=report.id, patient_id=patient.id,
            recipient=f"registered contact for {patient.phone}",
            delivery_channel="EMAIL", verification_status="VERIFIED",
            delivery_status="FAILED", created_at=now, verified_at=now,
            failed_at=now, failure_reason="DELIVERY_PROVIDER_FAILURE",
            audit_note="OTP verified successfully; delivery provider failed on send.",
        )
        db.add(delivery)
        db.commit()
        return {"success": False, "reason": "DELIVERY_FAILED"}

    token = secrets.token_urlsafe(16)
    expires_at = now + datetime.timedelta(minutes=SIGNED_LINK_VALIDITY_MINUTES)
    delivery = ReportDelivery(
        report_id=report.id, patient_id=patient.id,
        recipient=f"registered contact for {patient.phone}",
        delivery_channel="EMAIL", verification_status="VERIFIED",
        delivery_status="SENT", signed_link_token=token,
        signed_link_expires_at=expires_at, created_at=now,
        verified_at=now, sent_at=now,
        audit_note="OTP verified; report delivered successfully.",
    )
    db.add(delivery)
    db.commit()

    return {
        "success": True, "reason": "DELIVERY_SENT",
        "masked_phone": _mask_phone_last4(patient.phone),
        "signed_link_expires_minutes": SIGNED_LINK_VALIDITY_MINUTES,
    }


@app.get("/api/v1/reports/link/{token}")
def validate_report_link(token: str, db: Session = Depends(get_db)):
    """RULE 11 (links must expire), RULE 12 (no direct access without
    authorization). This is an HTTP-level endpoint, not something the
    voice agent itself calls -- the caller never speaks a token back
    over the phone. It exists so the plan's Section 8 link/token attacks
    (23-28: expired, reused, modified, empty, random, cross-report) have
    something real to attack, matching a real "click the link we sent
    you" step in the actual delivery channel."""
    if not token:
        return {"valid": False, "reason": "LINK_INVALID"}

    delivery = db.query(ReportDelivery).filter_by(signed_link_token=token).first()
    if not delivery:
        # Covers a genuinely random token, an empty one, AND a modified
        # one (flipping one character of a real token overwhelmingly
        # lands on a value nothing in signed_link_token equals) --
        # ATTACK 25/26/27 all resolve to this same branch, which is the
        # point: a corrupted token carries no partial credit.
        return {"valid": False, "reason": "LINK_INVALID"}

    if not delivery.signed_link_expires_at or delivery.signed_link_expires_at <= datetime.datetime.now():
        return {"valid": False, "reason": "LINK_EXPIRED"}

    linked_report = db.query(LabReport).filter_by(id=delivery.report_id).first()
    return {
        "valid": True,
        "report_number": linked_report.report_number if linked_report else None,
    }
