"""SQLAlchemy models for the clinic's dummy PostgreSQL data.

Prototype-grade on purpose: this exists so agent/tools_client.py has a
real backend to call while the voice agent is being bench-tested, not as
a production clinic management system. Swap in the real hospital DB later
without touching main.py's turn loop -- only this service's queries
change, because it speaks the exact contract voice-agent already expects.
"""
from __future__ import annotations

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, ForeignKey, DateTime, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    # Bengali and English aliases for department names to handle short forms like "ortho"
    aliases_bn = Column(String, nullable=False, default="")

    doctors = relationship("Doctor", back_populates="department")


class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)              # "Dr. S. Mukherjee"
    qualifications = Column(String, nullable=False)     # "MBBS, MD (Gen. Med.)"
    # Bengali-script spellings of the surname, "|"-joined. Real callers say
    # "ডক্টর সেন", not "Dr. Sen" -- matching only the Latin name against
    # Bengali ASR output silently fails 100% of the time, not just on
    # near-misses, since the two scripts share no characters at all.
    aliases_bn = Column(String, nullable=False, default="")
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)

    department = relationship("Department", back_populates="doctors")
    schedule = relationship("DoctorSchedule", back_populates="doctor")


class DoctorSchedule(Base):
    """One row per weekday a doctor sits. A doctor with 3 chamber days has
    3 rows here, not a serialized list -- keeps the "is Dr. X in on
    Wednesday" query a plain WHERE clause."""
    __tablename__ = "doctor_schedule"
    id = Column(Integer, primary_key=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    weekday = Column(Integer, nullable=False)   # 0=Monday ... 6=Sunday (Python's convention)
    start_time = Column(String, nullable=False)  # "18:00"
    end_time = Column(String, nullable=False)    # "20:00"

    doctor = relationship("Doctor", back_populates="schedule")

    __table_args__ = (UniqueConstraint("doctor_id", "weekday", name="uq_doctor_weekday"),)


class LabTest(Base):
    __tablename__ = "lab_tests"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    # Bengali-script names/synonyms a real caller would actually say,
    # "|"-joined -- see Doctor.aliases_bn for why this exists at all.
    aliases_bn = Column(String, nullable=False, default="")
    rate_inr = Column(Integer, nullable=False)
    sample_type = Column(String, nullable=False)         # "Blood" / "Urine" / "Imaging" / "Cardiac"
    report_time_hours = Column(Integer, nullable=False)


# Appointment.status values. Persisted, so they are a data format.
APPT_BOOKED = "booked"
APPT_RESCHEDULED = "rescheduled"   # still live; moved at least once
APPT_CANCELLED = "cancelled"

# Appointment.slot_lock -- see the Appointment docstring. A live row holds
# this constant; a cancelled row holds its own confirmation_id instead.
SLOT_LOCK_ACTIVE = "ACTIVE"


class Appointment(Base):
    """One appointment, plus the two columns cancellation forced us to add.

    WHY slot_lock EXISTS
    --------------------
    The unique constraint on (doctor_id, date, time_slot) is what makes
    double-booking impossible even when two callers race -- see
    main.book_appointment()'s IntegrityError handler, which turns that race
    into "that slot just went, here are three others".

    Cancellation breaks that arrangement. A cancelled row has to be KEPT:
    it is the audit trail, and every NotificationAttempt for the
    cancellation message points at it. But a kept row goes on occupying its
    slot under that constraint, so nobody could ever book a slot somebody
    else had released -- the cancellation would free the patient and not
    the appointment.

    Deleting the row instead would solve the constraint and lose the
    history, in the one domain where "we have no record of that
    appointment" is the worst possible answer to give at a counter.

    So the constraint gains a fourth column. A live row sets
    slot_lock=SLOT_LOCK_ACTIVE, so at most one live row can hold a given
    doctor/date/slot -- exactly the old guarantee. A cancelled row sets
    slot_lock to its own confirmation_id, which is unique by its own
    constraint, so any number of cancelled rows can pile up on the same
    slot without ever colliding with each other or with the live one.

    Every "is this slot taken" query must therefore filter on
    slot_lock == SLOT_LOCK_ACTIVE. There are three in book_appointment()
    and one in reschedule_appointment().

    RESCHEDULING NEEDS NO TOMBSTONE: the row itself moves to the new
    date/time_slot and stays ACTIVE, which frees the old slot as a
    side effect of the UPDATE.
    """
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True)
    confirmation_id = Column(String, nullable=False, unique=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    date = Column(String, nullable=False)        # ISO yyyy-mm-dd
    time_slot = Column(String, nullable=False)   # "18:15"
    patient_name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)

    # APPT_BOOKED / APPT_RESCHEDULED / APPT_CANCELLED.
    status = Column(String, nullable=False, default=APPT_BOOKED)
    # SLOT_LOCK_ACTIVE while the slot is held, the confirmation_id once
    # released. Read the class docstring before changing either value.
    slot_lock = Column(String, nullable=False, default=SLOT_LOCK_ACTIVE)
    # Null until the first reschedule or cancellation. created_at stays the
    # moment of the ORIGINAL booking, because that is what a patient means
    # when they say "I booked this last week".
    updated_at = Column(DateTime, nullable=True)

    doctor = relationship("Doctor")

    __table_args__ = (
        UniqueConstraint("doctor_id", "date", "time_slot", "slot_lock", name="uq_doctor_slot"),
    )


class NotificationAttempt(Base):
    """The delivery ledger: one row per message this service owes a patient.

    THE ROW IS WRITTEN BEFORE ANYTHING IS SENT, in the same transaction as
    the booking it belongs to. That ordering is the whole design. It means
    there is no window in which we have taken a booking and have no record
    that a message was due -- if the process dies before the background
    task runs, the row is still there, still `queued`, and the staff queue
    reports it as stale. "Silently dropped" is not a state this table can
    represent.

    The rendered `body` is stored rather than re-derived on demand. Two
    reasons, both operational: reception needs to see the exact text the
    patient was or was not sent when they turn up disputing it, and
    templates change -- re-rendering a two-week-old failure through today's
    template would show staff a message that was never composed.

    THIS TABLE HOLDS PII (patient name and phone, in a column and again
    inside `body`). So does `appointments`. Retention is not implemented
    here and is flagged in the implementation notes as outstanding work,
    rather than left to look like an oversight.
    """
    __tablename__ = "notification_attempts"
    id = Column(Integer, primary_key=True)

    # Not a ForeignKey on purpose. The ledger has to outlive the row it
    # describes -- if an appointment is ever purged for retention, the
    # evidence that we did or did not message that patient must not be
    # cascaded away with it.
    confirmation_id = Column(String, nullable=False, index=True)

    event = Column(String, nullable=False)      # message_templates.EVENT_*
    channel = Column(String, nullable=False, default="sms")
    phone = Column(String, nullable=False)      # E.164 without '+', as sent
    template_id = Column(String, nullable=False, default="")   # DLT content ID
    body = Column(String, nullable=False)       # exactly what was submitted

    status = Column(String, nullable=False, index=True)   # notifications.STATUS_*
    provider_message_id = Column(String, nullable=True, index=True)
    attempts = Column(Integer, nullable=False, default=0)
    error_code = Column(String, nullable=True)
    error_detail = Column(String, nullable=True)

    created_at = Column(DateTime, nullable=False)
    # Moves on every state change. The staff queue's staleness rule is
    # measured from here, not from created_at, so a row that was retried
    # ten minutes ago is not immediately stale again.
    updated_at = Column(DateTime, nullable=False)
    delivered_at = Column(DateTime, nullable=True)

    # Set when a member of staff has seen the failure and taken it on.
    # An acknowledged row leaves the queue without its status being
    # rewritten -- the failure stays a failure in the record.
    acknowledged_by = Column(String, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
