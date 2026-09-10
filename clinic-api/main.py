"""Clinic data service -- implements the exact 3-endpoint contract
agent/tools_client.py in the voice agent already expects. Backed by
PostgreSQL, seeded with dummy departments/doctors/schedules/tests via
seed.py.

Matching is deliberately simple (ILIKE + difflib) for this prototype --
production callers slurring "লিপিড প্রোফাইল" through a phone mic deserve
something closer to voicerx/glossary.py's phonetic-fold gazetteer, not a
plain substring match. Flagged here rather than silently left as if this
were already that robust.
"""
from __future__ import annotations

import logging

import datetime
import difflib
import uuid

from fastapi import BackgroundTasks, FastAPI, Depends, Header, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import history_service
import message_templates as mt
import notifications as notify
import notify_service
import verification as verify
from db import get_db, SessionLocal
from models import (
    APPT_BOOKED, APPT_CANCELLED, APPT_RESCHEDULED, SLOT_LOCK_ACTIVE,
    Department, Doctor, DoctorSchedule, LabTest, Appointment, NotificationAttempt,
    Patient, TestRecord, DisclosureAudit,
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
    _check_appointment_schema()
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


# Set by _check_appointment_schema() and reported by /api/health. Not a
# bool: the health endpoint says WHICH columns are missing, because the
# person reading it is usually the person about to run the migration.
_SCHEMA_WARNING: str | None = None


def _check_appointment_schema() -> None:
    """Detect an `appointments` table predating the notification work.

    create_all() creates tables that do not exist; it does NOT alter ones
    that do. A database that already held appointments before this change
    therefore comes up with the OLD three-column unique constraint and
    without `status`/`slot_lock`/`updated_at` -- and SQLite cannot add or
    drop a table-level UNIQUE constraint with ALTER at all, so this is a
    table rebuild, not a column addition.

    That rebuild is NOT run automatically. This service's own startup
    docstring already explains why the seed is guarded on emptiness: an
    unconditional destructive step at boot "would wipe every appointment
    booked since the last boot, turning a convenience into data loss". A
    silent table rebuild is that same hazard with a wider blast radius. So
    the mismatch is detected, logged loudly, and surfaced on /api/health,
    and `python migrate_notifications.py` performs the rebuild when someone
    decides to run it.
    """
    global _SCHEMA_WARNING
    from sqlalchemy import inspect
    from db import engine

    try:
        insp = inspect(engine)
        if "appointments" not in insp.get_table_names():
            _SCHEMA_WARNING = None
            return
        columns = {c["name"] for c in insp.get_columns("appointments")}
    except Exception as e:                                # noqa: BLE001
        logging.getLogger("clinic-api").warning("could not inspect schema: %s", e)
        _SCHEMA_WARNING = None
        return

    missing = sorted({"status", "slot_lock", "updated_at"} - columns)
    if not missing:
        _SCHEMA_WARNING = None
        return

    _SCHEMA_WARNING = (
        f"appointments table is missing {', '.join(missing)} -- run "
        f"`python migrate_notifications.py` in clinic-api/. Until then, "
        f"bookings will fail and cancelled slots cannot be rebooked."
    )
    logging.getLogger("clinic-api").error(_SCHEMA_WARNING)


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    """Now also reports the delivery ledger.

    deploy/status.sh and anything else that polls this endpoint gets the
    notification backlog for free, which is the difference between a
    failing gateway being noticed in minutes and being noticed by a patient
    at a counter. `notifications.open_failures` is the number staff are
    expected to act on -- see notify_service.open_failures().
    """
    config = notify.load_config()
    return {
        "status": "ok",
        "departments": db.query(Department).count(),
        "doctors": db.query(Doctor).count(),
        "lab_tests": db.query(LabTest).count(),
        "gateway_configured": config.configured,
        "notifications": notify_service.summary(db),
        # A burst of failed verifications is somebody guessing. Surfaced on
        # the endpoint that is already polled rather than on one nobody
        # remembers to look at -- see history_service.summary().
        "history_disclosure": history_service.summary(db),
        # None on a healthy schema. Non-null means the appointments table
        # predates this change and migrate_notifications.py has not been
        # run -- see _check_appointment_schema().
        "schema_warning": _SCHEMA_WARNING,
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


def _live_slots(db: Session, doctor_id: int, date: str) -> set[str]:
    """Which of a doctor's slots are actually held on a date.

    `slot_lock == SLOT_LOCK_ACTIVE` is the filter that makes cancellation
    mean something: a cancelled row is kept for the audit trail and for the
    notification ledger that points at it, but it must stop occupying the
    slot it released. See the Appointment docstring in models.py.

    Every "is this taken" question in this file goes through here rather
    than repeating the filter, because forgetting it in one place is a bug
    that presents as "the slot I cancelled can never be rebooked" -- and
    only for the callers unlucky enough to want that slot.
    """
    return {
        a.time_slot for a in db.query(Appointment).filter_by(
            doctor_id=doctor_id, date=date, slot_lock=SLOT_LOCK_ACTIVE,
        ).all()
    }


def _prepare_notification(db: Session, appt: Appointment, event: str,
                          doctor_name: str) -> NotificationAttempt | None:
    """Stage the patient's written confirmation IN THE CALLER'S TRANSACTION.

    MUST be called BEFORE the commit that persists the appointment change,
    and never after it. That ordering is the whole reason "silently
    dropped" is not a state this system can reach:

      * one commit persists the appointment change and the ledger row
        together, so there is no window in which a booking exists and no
        record says a message was due;
      * if the commit fails or the booking loses a slot race, the rollback
        discards BOTH -- no orphan ledger row promising a message for an
        appointment that was never made.

    This works because `confirmation_id` is generated in Python before the
    insert, not by the database, so the message can be composed against an
    appointment row that has not been written yet.

    Returns None if the row could not even be staged. Nothing in here may
    raise into the caller: a booking that succeeded must never be reported
    as failed because an SMS could not be composed.
    """
    try:
        return notify_service.queue_message(db, appt, event, doctor_name)
    except Exception:                                     # noqa: BLE001
        logging.getLogger("clinic-api").exception(
            "could not stage a %s notification for %s (patient %s on %s) -- "
            "the appointment itself stands",
            event, appt.confirmation_id, appt.patient_name, appt.phone,
        )
        return None


def _schedule_delivery(background: BackgroundTasks,
                       attempt: NotificationAttempt | None, event: str) -> dict:
    """Hand a committed ledger row to the background sender, and describe it
    for the response.

    Called AFTER the commit, so the row the background task will re-read
    definitely exists -- a task scheduled against an uncommitted row would
    find nothing when it opened its own session.
    """
    if attempt is None:
        return {"status": "not_recorded", "event": event}

    if attempt.status == notify.STATUS_QUEUED:
        background.add_task(notify_service.deliver_now, attempt.id)

    return {
        "id": attempt.id,
        "event": event,
        "status": attempt.status,
        "error_code": attempt.error_code,
    }


@app.post("/api/v1/appointments")
def book_appointment(req: BookingRequest, background: BackgroundTasks,
                     db: Session = Depends(get_db)):
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

    def _free_slots() -> list[str]:
        taken = _live_slots(db, doctor.id, req.date)
        return [s for s in valid_slots if s not in taken][:3]

    if req.time_slot in _live_slots(db, doctor.id, req.date):
        return {"success": False, "reason": "slot_taken", "alternative_slots": _free_slots()}

    confirmation_id = f"KCD-{req.date.replace('-', '')}-{uuid.uuid4().hex[:4].upper()}"
    appt = Appointment(
        confirmation_id=confirmation_id, doctor_id=doctor.id, date=req.date,
        time_slot=req.time_slot, patient_name=req.patient_name, phone=req.phone,
        created_at=datetime.datetime.now(),
        status=APPT_BOOKED, slot_lock=SLOT_LOCK_ACTIVE,
    )

    # Staged BEFORE the commit below, deliberately, so one transaction
    # carries the booking and the record that a message is owed for it. If
    # the commit loses the slot race, the rollback discards both.
    attempt = _prepare_notification(db, appt, mt.EVENT_BOOKED, doctor.name)

    # THE CHECK ABOVE IS NOT ATOMIC WITH THIS INSERT.
    #
    # Two callers being told the same slot is free, then both booking it, is
    # a genuine interleaving -- and it is MOST likely at peak, when the same
    # popular slots are being offered to several people at once.
    #
    # models.py's UniqueConstraint("doctor_id", "date", "time_slot") already
    # makes that safe for the DATA: the second insert cannot succeed. What it
    # did not do was make it safe for the CALLER. The IntegrityError was
    # uncaught, so it surfaced as a 500, which tools_client.py turns into a
    # ToolCallError, which main.py speaks as "এই মুহূর্তে দেখতে পারছি না" --
    # "I can't check right now". That is misleading: the system checked
    # perfectly well and knows exactly what happened.
    #
    # Catching it here turns the race into the honest answer the caller
    # deserves -- "that slot just went, here are three others" -- reusing the
    # same slot_taken shape reply_templates.booking_reply() already handles.
    try:
        db.add(appt)
        db.commit()
    except IntegrityError:
        db.rollback()
        if req.time_slot in _live_slots(db, doctor.id, req.date):
            logging.getLogger("clinic-api").info(
                "slot %s on %s for doctor %s lost a booking race -- offering alternatives",
                req.time_slot, req.date, doctor.name,
            )
            return {"success": False, "reason": "slot_taken",
                    "alternative_slots": _free_slots()}

        # The slot is still free, so the collision was on the only other
        # unique column, confirmation_id -- a 16^4 UUID-suffix clash. Do not
        # dress this up as slot_taken; it is not, and telling the caller to
        # pick another time would be a lie. booking_reply() renders an
        # unrecognised reason as its generic "couldn't book" message.
        logging.getLogger("clinic-api").warning(
            "unexpected IntegrityError booking %s on %s (slot still free) -- "
            "likely confirmation_id collision", req.time_slot, req.date,
        )
        return {"success": False, "reason": "booking_failed"}

    # The booking AND its ledger row are now committed together.
    notification = _schedule_delivery(background, attempt, mt.EVENT_BOOKED)

    return {
        "success": True, "confirmation_id": confirmation_id,
        "doctor_name": doctor.name,
        "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), "date": req.date, "time_slot": req.time_slot,
        # What the patient is owed in writing, so reply_templates.py can
        # decide whether to promise them a message. `queued` at this point
        # is the normal case -- the send has not been attempted yet.
        "notification": notification,
    }


# =============================================================================
# Tool 5: POST /api/v1/appointments/{confirmation_id}/reschedule
# =============================================================================
# The story's acceptance criterion covers booking, reschedule AND
# cancellation, and only the first of the three existed. These two
# endpoints are what make the other two events real events rather than
# hypothetical ones -- there was previously no way for an appointment to
# move or end at all, so there was nothing to message about.
class RescheduleRequest(BaseModel):
    date: str
    time_slot: str


class CancelRequest(BaseModel):
    # Optional, purely for the audit log. Nothing branches on it, and it is
    # deliberately NOT put into the patient's message: the cancellation
    # template is registered with four variables and adding a free-text
    # fifth is exactly the kind of change an operator rejects.
    reason: str | None = None


def _find_live_appointment(db: Session, confirmation_id: str) -> Appointment | None:
    return (db.query(Appointment)
              .filter_by(confirmation_id=confirmation_id.strip().upper())
              .first())


@app.post("/api/v1/appointments/{confirmation_id}/reschedule")
def reschedule_appointment(confirmation_id: str, req: RescheduleRequest,
                           background: BackgroundTasks, db: Session = Depends(get_db)):
    """Move an existing appointment, KEEPING its confirmation_id.

    The id is deliberately not reissued. The patient's whole complaint in
    the story is having to track a reference number; handing them a second
    one every time a clinic moves their slot would make that worse, and the
    reschedule template says "রেফারেন্স {#var#} একই থাকছে" -- the reference
    stays the same -- precisely so the earlier message they may still be
    holding does not become misleading.
    """
    appt = _find_live_appointment(db, confirmation_id)
    if appt is None:
        return {"success": False, "reason": "appointment_not_found"}
    if appt.status == APPT_CANCELLED:
        return {"success": False, "reason": "appointment_cancelled"}

    try:
        target = datetime.date.fromisoformat(req.date)
    except ValueError:
        return {"success": False, "reason": "missing_field"}

    doctor = db.get(Doctor, appt.doctor_id)
    if doctor is None:
        return {"success": False, "reason": "doctor_not_found"}

    sched = _schedule_for_weekday(db, doctor.id, target.weekday())
    if not sched:
        return {"success": False, "reason": "doctor_not_available_that_day"}

    valid_slots = _generate_slots(sched.start_time, sched.end_time)
    taken = _live_slots(db, doctor.id, req.date) - {
        # The appointment's own current slot is not an obstacle to itself.
        # Without this, "move it fifteen minutes later, same day" works but
        # "confirm the slot it already has" reports slot_taken, which reads
        # to a caller as the system having lost their booking.
        appt.time_slot if appt.date == req.date else ""
    }
    if req.time_slot not in valid_slots:
        free = [s for s in valid_slots if s not in taken][:3]
        return {"success": False, "reason": "slot_taken", "alternative_slots": free}
    if req.time_slot in taken:
        free = [s for s in valid_slots if s not in taken][:3]
        return {"success": False, "reason": "slot_taken", "alternative_slots": free}

    previous = {"date": appt.date, "time_slot": appt.time_slot}
    appt.date = req.date
    appt.time_slot = req.time_slot
    appt.status = APPT_RESCHEDULED
    appt.updated_at = datetime.datetime.now()

    # Staged after the new date/slot are set, so the message carries the
    # values the patient is being moved to -- and before the commit, so
    # losing the race below discards the message with the move.
    attempt = _prepare_notification(db, appt, mt.EVENT_RESCHEDULED, doctor.name)

    try:
        db.commit()
    except IntegrityError:
        # Same race as book_appointment()'s, same honest answer: somebody
        # took the target slot between the check above and this commit.
        db.rollback()
        logging.getLogger("clinic-api").info(
            "reschedule of %s to %s %s lost a race", confirmation_id, req.date, req.time_slot,
        )
        return {"success": False, "reason": "slot_taken",
                "alternative_slots": [s for s in valid_slots
                                      if s not in _live_slots(db, doctor.id, req.date)][:3]}

    notification = _schedule_delivery(background, attempt, mt.EVENT_RESCHEDULED)

    return {
        "success": True, "confirmation_id": appt.confirmation_id,
        "doctor_name": doctor.name, "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
        "date": appt.date, "time_slot": appt.time_slot,
        "previous_date": previous["date"], "previous_time_slot": previous["time_slot"],
        "notification": notification,
    }


@app.post("/api/v1/appointments/{confirmation_id}/cancel")
def cancel_appointment(confirmation_id: str, req: CancelRequest,
                       background: BackgroundTasks, db: Session = Depends(get_db)):
    """Cancel an appointment, release its slot, and tell the patient.

    The row is kept, not deleted -- see the Appointment docstring for why,
    and for what slot_lock does to keep the unique constraint honest once a
    cancelled row stops owning its slot.

    Cancelling an already-cancelled appointment is reported as success with
    `already_cancelled` set, and sends NO second message. A gateway retry,
    a double-tap in a staff UI and a patient calling twice all reach this
    endpoint, and none of them is a reason to message somebody about a
    cancellation they were already told about.
    """
    appt = _find_live_appointment(db, confirmation_id)
    if appt is None:
        return {"success": False, "reason": "appointment_not_found"}

    doctor = db.get(Doctor, appt.doctor_id)
    doctor_name = doctor.name if doctor else ""

    if appt.status == APPT_CANCELLED:
        return {
            "success": True, "already_cancelled": True,
            "confirmation_id": appt.confirmation_id,
            "date": appt.date, "time_slot": appt.time_slot,
            "notification": {"status": "not_resent", "event": mt.EVENT_CANCELLED},
        }

    appt.status = APPT_CANCELLED
    # Releases the slot. The confirmation_id is unique, so this row can
    # never collide with the live booking that replaces it, nor with any
    # other cancelled row on the same slot.
    appt.slot_lock = appt.confirmation_id
    appt.updated_at = datetime.datetime.now()

    # Staged before the commit, same as the other two events: the
    # cancellation and the record that the patient must be told about it
    # land together or not at all.
    attempt = _prepare_notification(db, appt, mt.EVENT_CANCELLED, doctor_name)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        logging.getLogger("clinic-api").exception(
            "cancelling %s hit an integrity error", confirmation_id)
        return {"success": False, "reason": "cancel_failed"}

    if req.reason:
        logging.getLogger("clinic-api").info(
            "appointment %s cancelled, reason given: %s", confirmation_id, req.reason[:200])

    notification = _schedule_delivery(background, attempt, mt.EVENT_CANCELLED)

    return {
        "success": True, "already_cancelled": False,
        "confirmation_id": appt.confirmation_id,
        "doctor_name": doctor_name,
        "doctor_name_bn": _first_alias_bn(doctor.aliases_bn) if doctor else None,
        "date": appt.date, "time_slot": appt.time_slot,
        "notification": notification,
    }


# =============================================================================
# Delivery receipts and the staff failure queue
# =============================================================================
# "Delivery receipts are recorded and failures surfaced to staff rather
# than silently dropped" is half of the story's acceptance criterion, and
# it is the half that needs endpoints rather than just a send call.
class DeliveryReceipt(BaseModel):
    """What the gateway POSTs back once it knows what happened to a message.

    Field names follow the same tolerance as notifications._provider_id():
    gateways disagree, and either identifier is enough to find the row.
    """
    status: str
    provider_message_id: str | None = None
    message_id: str | None = None
    client_ref: str | None = None
    error_code: str | None = None
    error_detail: str | None = None


@app.post("/api/v1/notifications/receipt")
def delivery_receipt(receipt: DeliveryReceipt, db: Session = Depends(get_db),
                     x_gateway_token: str | None = Header(default=None)):
    """The gateway's delivery-receipt callback.

    AUTHENTICATION IS CONDITIONAL, and the condition is deliberate: a
    deployment with a configured gateway MUST set HOSPITAL_GATEWAY_DLR_TOKEN
    and matching receipts are rejected without it, because anything on the
    network could otherwise mark a patient's message delivered. A bench pod
    with no gateway at all has no token to check and accepts receipts so
    the path can be tested.

    ALWAYS ANSWERS 200 for an unmatched receipt rather than an error.
    Gateways retry a receipt that gets a non-2xx, often indefinitely; a
    receipt we cannot match is our problem to see in the log, not a reason
    to make theirs a loop.
    """
    config = notify.load_config()
    if config.configured:
        if not config.dlr_token:
            logging.getLogger("clinic-api").error(
                "receipt rejected: gateway is configured but HOSPITAL_GATEWAY_DLR_TOKEN is not")
            return {"accepted": False, "reason": "receipt_auth_not_configured"}
        if x_gateway_token != config.dlr_token:
            logging.getLogger("clinic-api").warning("receipt rejected: bad gateway token")
            return {"accepted": False, "reason": "unauthorized"}

    attempt = notify_service.apply_receipt(
        db,
        raw_status=receipt.status,
        provider_message_id=receipt.provider_message_id or receipt.message_id,
        client_ref=receipt.client_ref,
        error_code=receipt.error_code,
        error_detail=receipt.error_detail,
    )
    if attempt is None:
        logging.getLogger("clinic-api").warning(
            "delivery receipt matched no ledger row (provider id %r, client_ref %r)",
            receipt.provider_message_id or receipt.message_id, receipt.client_ref,
        )
        return {"accepted": False, "reason": "unknown_message"}

    db.commit()
    return {"accepted": True, "id": attempt.id, "status": attempt.status}


@app.get("/api/v1/notifications/failures")
def notification_failures(stale_minutes: int = Query(notify.DEFAULT_STALE_MINUTES, ge=1),
                          include_acknowledged: bool = Query(False),
                          include_skipped: bool = Query(False),
                          db: Session = Depends(get_db)):
    """The staff queue: every patient still owed a written confirmation.

    This is what "surfaced to staff rather than silently dropped" means in
    practice -- a list somebody at the clinic can work through, including
    the messages that failed in the quietest way possible, by being
    accepted and then never delivered. See notify_service.open_failures()
    for the three conditions that put a row here.
    """
    rows = notify_service.open_failures(
        db, stale_minutes=stale_minutes,
        include_acknowledged=include_acknowledged,
        include_skipped=include_skipped,
    )
    return {"count": len(rows), "stale_minutes": stale_minutes,
            "failures": [notify_service.as_dict(r) for r in rows]}


@app.get("/api/v1/appointments/{confirmation_id}/notifications")
def appointment_notifications(confirmation_id: str, db: Session = Depends(get_db)):
    """Every message ever owed for one appointment, newest first.

    The reception-desk question: a patient is standing there saying they
    were told to expect a message. This answers whether one was sent, what
    it said, and whether the operator ever confirmed delivery.
    """
    rows = (db.query(NotificationAttempt)
              .filter_by(confirmation_id=confirmation_id.strip().upper())
              .order_by(NotificationAttempt.created_at.desc()).all())
    return {"confirmation_id": confirmation_id.strip().upper(),
            "count": len(rows),
            "notifications": [notify_service.as_dict(r) for r in rows]}


class AcknowledgeRequest(BaseModel):
    staff: str


@app.post("/api/v1/notifications/{attempt_id}/acknowledge")
def acknowledge_failure(attempt_id: int, req: AcknowledgeRequest,
                        db: Session = Depends(get_db)):
    """A member of staff takes a failure on -- they have phoned the patient,
    or handed them a printed slip. The row leaves the queue; its status
    stays whatever it actually was."""
    attempt = notify_service.acknowledge(db, attempt_id, req.staff)
    if attempt is None:
        return {"success": False, "reason": "not_found"}
    db.commit()
    return {"success": True, "notification": notify_service.as_dict(attempt)}


@app.post("/api/v1/notifications/{attempt_id}/retry")
def retry_notification(attempt_id: int, background: BackgroundTasks,
                       db: Session = Depends(get_db)):
    """Send a failed message again, from the stored body.

    Re-sends what was composed at the time, NOT a freshly rendered message
    -- see the NotificationAttempt docstring. If the template has changed
    since, a re-render would put text in front of the patient that nobody
    ever approved for this appointment.
    """
    attempt = notify_service.requeue(db, attempt_id)
    if attempt is None:
        return {"success": False, "reason": "not_found_or_already_queued"}
    db.commit()
    background.add_task(notify_service.deliver_now, attempt.id)
    return {"success": True, "notification": notify_service.as_dict(attempt)}


# =============================================================================
# PATIENT HISTORY -- disclosed only after verification
# =============================================================================
# Author: Chakravardhan
#
# Story: "As a patient, I want my history spoken only to me, so that whoever
# else uses this handset cannot hear what tests I have had."
#
# THE PHONE NUMBER IS NOT A CREDENTIAL. It says which record to LOOK AT, and
# nothing at all about who is holding the phone. Every endpoint below is
# built on that distinction; see clinic-api/verification.py for why an SMS
# OTP is not the answer here.
#
# EVERY ONE OF THESE IS A POST, INCLUDING THE READ. A token in a query string
# lands in the access log, the proxy log and anything scraping either -- and
# a token is the one value in this system that grants access on its own. Bodies
# are not logged; URLs are.
class VerifyBeginRequest(BaseModel):
    phone: str
    call_id: str | None = None


class VerifyAnswerRequest(BaseModel):
    phone: str
    factor: str
    answer: str
    call_id: str | None = None


class HistoryRequest(BaseModel):
    token: str
    call_id: str | None = None


class RefusalRequest(BaseModel):
    phone: str
    reason: str
    call_id: str | None = None


@app.post("/api/v1/history/verify/begin")
def history_verify_begin(req: VerifyBeginRequest, db: Session = Depends(get_db)):
    """Which proof to ask this caller for.

    ANSWERS IDENTICALLY FOR A NUMBER WE HAVE NEVER SEEN. An unknown caller is
    asked for a date of birth exactly as a known one is. Replying "no patient
    with that number" would turn this line into a lookup for whether a named
    person attends this clinic -- which is itself information about them, and
    is the enumeration hole that most verification systems leak through.
    """
    return history_service.begin(db, req.phone, call_id=req.call_id)


@app.post("/api/v1/history/verify")
def history_verify(req: VerifyAnswerRequest, db: Session = Depends(get_db)):
    """One attempt. -> {"reply": ..., "token": ... | null}

    `reply` is the ONLY thing the caller may learn, and it is deliberately
    coarser than what the audit row records: a wrong PIN, an unknown number
    and a patient with no factor at all all come back as "failed". The
    reasons differ; what the caller can distinguish must not.
    """
    decision, token = history_service.attempt(
        db, phone=req.phone, factor=req.factor, answer=req.answer, call_id=req.call_id,
    )
    return {
        "reply": decision.reply,
        "verified": decision.verified,
        # Returned so the agent can ask the same question again. Never a hint
        # about the ANSWER -- only about which kind of proof is wanted.
        "factor": decision.factor,
        "token": token,
    }


@app.post("/api/v1/history/read")
def history_read(req: HistoryRequest, db: Session = Depends(get_db)):
    """The history itself. Opens only with a token from a verified attempt.

    A missing or expired token is answered as `not_verified` rather than as
    an error -- from the caller's side it is the same situation as never
    having verified, and the agent handles both by asking again.
    """
    data = history_service.history(db, req.token, call_id=req.call_id)
    if data is None:
        return {"found": False, "reason": "not_verified"}
    return {"found": True, **data}


@app.post("/api/v1/history/refusal")
def history_refusal(req: RefusalRequest, db: Session = Depends(get_db)):
    """Record a disclosure the AGENT refused -- a speakerphone, an
    unclassified audio path.

    Audited here rather than only in the agent's log because without a row,
    "the system would not tell me my history" has no explanation on the
    clinic's side, and the likeliest support response would be to switch the
    check off.
    """
    history_service.record_refusal(db, phone=req.phone, reason=req.reason,
                                   call_id=req.call_id)
    return {"recorded": True}


class SetPinRequest(BaseModel):
    phone: str
    pin: str
    staff: str


@app.post("/api/v1/patients/pin")
def set_patient_pin(req: SetPinRequest, db: Session = Depends(get_db)):
    """FOR COUNTER STAFF, IN PERSON. Never reachable from the voice line.

    A PIN that can be set by whoever is holding the handset is not a second
    factor -- it is a button labelled "make me verified", and it would undo
    the whole story. The voice agent has no client method for this endpoint,
    deliberately (see agent/tools_client.py).
    """
    ok = history_service.set_pin(db, phone=req.phone, pin=req.pin)
    if ok:
        logging.getLogger("clinic-api").info(
            "PIN set for %s by staff %s", req.phone, req.staff[:60])
    return {"success": ok}


@app.get("/api/v1/history/audit")
def history_audit(limit: int = Query(50, ge=1, le=500),
                  outcome: str | None = Query(None),
                  db: Session = Depends(get_db)):
    """The audit trail, for staff.

    The failures are the interesting part: a run of them against one number
    is somebody guessing, and that is invisible without this. No secret is
    stored in these rows -- only which KIND of proof was attempted.
    """
    q = db.query(DisclosureAudit)
    if outcome:
        q = q.filter_by(outcome=outcome)
    rows = q.order_by(DisclosureAudit.created_at.desc()).limit(limit).all()
    return {"count": len(rows), "audit": [
        {"id": r.id, "phone": r.phone, "patient_id": r.patient_id,
         "factor": r.factor, "outcome": r.outcome, "detail": r.detail,
         "call_id": r.call_id,
         "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in rows
    ]}
