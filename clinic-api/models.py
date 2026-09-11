"""
SQLAlchemy models for the clinic's dummy PostgreSQL data.

Prototype-grade on purpose: this exists so the voice agent can call
a realistic backend while being bench-tested.

All data is fictional and intended only for testing.

======================================================================
CHATGPT ADDITION NOTE
======================================================================


1. Add clinic opening and closing time.
2. Add clinic address and directions.
3. Add health packages.
4. Add patient details such as name and phone number.
5. Add patient-specific laboratory reports.
6. Support testing whether a patient's report is ready or not ready.
7. Support report-not-found cases.
8. Add OTP verification for sending reports.
9. Add report delivery tracking and audit information.
10. Add expiring signed-link information instead of permanent attachments.
11. Support English, Hinglish/Banglish and Bengali-script caller inputs.
12. Add enough structure to test edge cases for:
       - report ready
       - report not ready
       - report not found
       - wrong patient
       - wrong phone
       - wrong OTP
       - expired OTP
       - expired report link
       - failed delivery
       - successful delivery
       - repeated delivery attempts
13. Keep the existing doctor, department, schedule, laboratory and
    appointment functionality intact.

Created for Sourav.
======================================================================
"""

from __future__ import annotations

import secrets

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
    DateTime,
    UniqueConstraint,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


# ============================================================================
# CLINIC INFORMATION
# ============================================================================
#
# Used for queries such as:
#
# English:
#   "When do you open?"
#   "What time does the clinic close?"
#   "Where is the clinic?"
#   "Give me directions."
#
# Hinglish / Banglish:
#   "Clinic kab khulta hai?"
#   "Clinic koto khon porjonto open thake?"
#   "Address ta ki?"
#   "Kivabe jabo?"
#
# Bengali script:
#   "ক্লিনিক কখন খোলে?"
#   "ক্লিনিক কখন বন্ধ হয়?"
#   "ঠিকানাটা কী?"
#
# CHATGPT ADDITION - CREATED BY SOURAV:
# This section was added to make clinic-information intents testable
# without hardcoding the information inside the voice agent.
# ============================================================================

class ClinicInfo(Base):
    __tablename__ = "clinic_info"

    id = Column(Integer, primary_key=True)

    clinic_name = Column(String, nullable=False)

    # Main contact number of the clinic.
    phone = Column(String, nullable=False)

    # Full physical address.
    address = Column(Text, nullable=False)

    # Simple landmark / directions text.
    directions = Column(Text, nullable=False)

    # Weekly opening and closing times.
    #
    # Example:
    # Monday = 09:00 - 21:00
    #
    # Stored as strings intentionally for prototype simplicity.
    monday_open = Column(String, nullable=False, default="09:00")
    monday_close = Column(String, nullable=False, default="21:00")

    tuesday_open = Column(String, nullable=False, default="09:00")
    tuesday_close = Column(String, nullable=False, default="21:00")

    wednesday_open = Column(String, nullable=False, default="09:00")
    wednesday_close = Column(String, nullable=False, default="21:00")

    thursday_open = Column(String, nullable=False, default="09:00")
    thursday_close = Column(String, nullable=False, default="21:00")

    friday_open = Column(String, nullable=False, default="09:00")
    friday_close = Column(String, nullable=False, default="21:00")

    saturday_open = Column(String, nullable=False, default="09:00")
    saturday_close = Column(String, nullable=False, default="21:00")

    # Nullable because Sunday can be closed.
    sunday_open = Column(String, nullable=True)
    sunday_close = Column(String, nullable=True)

    sunday_closed = Column(Boolean, nullable=False, default=True)


# ============================================================================
# HEALTH PACKAGES
# ============================================================================
#
# Used for:
#
#   "What health packages do you have?"
#   "Diabetes package ache?"
#   "Heart checkup package koto?"
#   "General health package e ki ki test ache?"
#
# CHATGPT ADDITION - CREATED BY SOURAV:
# Added so package-related calls can be tested against database data
# instead of hardcoded responses.
# ============================================================================

class HealthPackage(Base):
    __tablename__ = "health_packages"

    id = Column(Integer, primary_key=True)

    name = Column(String, nullable=False, unique=True)

    # Bengali / Banglish / English aliases.
    #
    # Example:
    # "diabetes checkup|diabetes package|ডায়াবেটিস প্যাকেজ"
    aliases = Column(String, nullable=False, default="")

    description = Column(Text, nullable=False)

    price_inr = Column(Float, nullable=False)

    active = Column(Boolean, nullable=False, default=True)

    tests = relationship(
        "HealthPackageTest",
        back_populates="package",
        cascade="all, delete-orphan",
    )


class HealthPackageTest(Base):
    __tablename__ = "health_package_tests"

    id = Column(Integer, primary_key=True)

    package_id = Column(
        Integer,
        ForeignKey("health_packages.id"),
        nullable=False,
    )

    lab_test_id = Column(
        Integer,
        ForeignKey("lab_tests.id"),
        nullable=False,
    )

    package = relationship(
        "HealthPackage",
        back_populates="tests",
    )

    lab_test = relationship("LabTest")

    __table_args__ = (
        UniqueConstraint(
            "package_id",
            "lab_test_id",
            name="uq_package_test",
        ),
    )


# ============================================================================
# DEPARTMENT
# ============================================================================

class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True)

    name = Column(String, nullable=False, unique=True)

    # English + Bengali-script + Banglish aliases.
    #
    # Example:
    # "ortho|orthopedics|অর্থোপেডিক্স|অর্থো|bone"
    aliases_bn = Column(
        String,
        nullable=False,
        default="",
    )

    doctors = relationship(
        "Doctor",
        back_populates="department",
    )


# ============================================================================
# DOCTOR
# ============================================================================

class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True)

    name = Column(
        String,
        nullable=False,
    )

    qualifications = Column(
        String,
        nullable=False,
    )

    # Bengali-script and spoken aliases.
    #
    # Example:
    # "সেন|ডক্টর সেন|Dr Sen|doctor sen"
    aliases_bn = Column(
        String,
        nullable=False,
        default="",
    )

    department_id = Column(
        Integer,
        ForeignKey("departments.id"),
        nullable=False,
    )

    department = relationship(
        "Department",
        back_populates="doctors",
    )

    schedule = relationship(
        "DoctorSchedule",
        back_populates="doctor",
        cascade="all, delete-orphan",
    )


# ============================================================================
# DOCTOR SCHEDULE
# ============================================================================

class DoctorSchedule(Base):
    """
    One row per weekday a doctor sits.

    weekday:
        0 = Monday
        1 = Tuesday
        ...
        6 = Sunday
    """

    __tablename__ = "doctor_schedule"

    id = Column(Integer, primary_key=True)

    doctor_id = Column(
        Integer,
        ForeignKey("doctors.id"),
        nullable=False,
    )

    weekday = Column(
        Integer,
        nullable=False,
    )

    start_time = Column(
        String,
        nullable=False,
    )

    end_time = Column(
        String,
        nullable=False,
    )

    doctor = relationship(
        "Doctor",
        back_populates="schedule",
    )

    __table_args__ = (
        UniqueConstraint(
            "doctor_id",
            "weekday",
            name="uq_doctor_weekday",
        ),
    )


# ============================================================================
# LAB TEST
# ============================================================================

class LabTest(Base):
    __tablename__ = "lab_tests"

    id = Column(Integer, primary_key=True)

    name = Column(
        String,
        nullable=False,
        unique=True,
    )

    # English + Bengali-script + Banglish aliases.
    #
    # Example:
    # "cbc|সি বি সি|সিবিসি|complete blood count"
    aliases_bn = Column(
        String,
        nullable=False,
        default="",
    )

    rate_inr = Column(
        Float,
        nullable=False,
    )

    sample_type = Column(
        String,
        nullable=False,
    )

    report_time_hours = Column(
        Integer,
        nullable=False,
    )

    # ========================================================================
    # ADDED BY SOURAV -- "Caller asks how to prepare for a test" story.
    #
    # Every column below is nullable with NO default, on purpose: only the
    # tests the business has actually supplied real preparation content
    # for (see clinic-api/seed.py's LAB_TEST_ADVISORIES) ever get these
    # filled in. Every other LabTest row leaves them all as None.
    #
    # This is a deliberate safety choice, not an oversight. Defaulting
    # fasting_required to False for a test nobody has actually reviewed
    # would be FABRICATING a medical instruction -- "no fasting needed"
    # is not a safe guess, it is a specific claim that could be wrong.
    # Same discipline as RULE 1 elsewhere in this file ("never invent a
    # report"), applied here to something with real physical stakes if
    # gotten wrong: the API layer and the voice agent must both treat
    # "no advisory row" as "we don't know yet, don't answer", never as
    # "assume no restrictions".
    # ========================================================================

    # Whether the caller must fast before this test. None = not seeded
    # yet (see above) -- never treat as False.
    fasting_required = Column(Boolean, nullable=True)

    # Free text, not an int: real fasting windows are given as ranges
    # ("8-12 hours"), not single numbers.
    fasting_hours = Column(String, nullable=True)

    # What the caller may drink while fasting/preparing (e.g. "Only plain
    # water permitted during fasting period").
    water_allowance = Column(String, nullable=True)

    # Which medications (if any) to hold and until when (e.g. "Hold
    # morning anti-diabetic medication until after blood collection").
    medication_hold = Column(Text, nullable=True)

    # Any other timing constraint on the test itself (e.g. "Exactly 2
    # hours post-meal", "Morning sample collection preferred").
    timing_rule = Column(String, nullable=True)

    # Ready-to-speak, business-authored advisory sentences, one per
    # supported language, each containing a literal "{test_name}"
    # placeholder the reply layer fills in at speak-time (see
    # agent/reply_templates.py's test_preparation_reply() -- same
    # canonical-name-vs-Bengali-alias selection every other reply in this
    # codebase already uses). Stored as real per-language text rather
    # than composed from the structured fields above, because the
    # business supplied genuine, reviewed Bengali/Hinglish/Banglish
    # phrasing for these -- unlike ClinicInfo.address/directions or
    # HealthPackage.description, this is NOT the English-only gap
    # flagged elsewhere in this codebase (see those models' own
    # comments); recomposing it programmatically from the structured
    # fields would only risk mangling wording the business already
    # approved.
    advisory_script_en = Column(Text, nullable=True)
    advisory_script_hinglish = Column(Text, nullable=True)
    advisory_script_banglish = Column(Text, nullable=True)
    advisory_script_bn = Column(Text, nullable=True)

    # ========================================================================
    # ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables. Walk-in
    # Eligibility + Prescription Requirements stories.
    #
    # Same discipline as the advisory columns just above, extended to two
    # more regulatory/operational facts: every column below is nullable
    # with NO default. None means "nobody has reviewed this test for
    # walk-in/prescription policy yet" -- never treat that as "eligible"
    # or "not required". Defaulting walkin_eligible to True, or
    # prescription_required to False, for a test nobody actually
    # confirmed would be fabricating an operational/regulatory fact
    # exactly like guessing a fasting rule would be a medical one -- see
    # this class's own comment above for the identical reasoning.
    # ========================================================================

    # Whether a caller can walk in for this test without a prior
    # appointment. None = not reviewed yet, never treat as False.
    walkin_eligible = Column(Boolean, nullable=True)

    # Free text, not structured hours: real walk-in windows are given as
    # day-specific ranges ("Mon-Sat 7am-11am, no walk-ins Sunday"), not a
    # single machine-parseable value -- same reasoning as fasting_hours
    # above being a string, not an int.
    walkin_hours = Column(String, nullable=True)

    # Whether this test requires a doctor's prescription before it can be
    # performed. None = not reviewed yet, never treat as False.
    prescription_required = Column(Boolean, nullable=True)

    # Pipe-delimited list of accepted submission channels for the
    # prescription (e.g. "whatsapp_photo|email|counter_in_person") --
    # same delimited-list convention as aliases_bn above and
    # HealthPackage.aliases below, not a JSON column (nothing else in
    # this file uses one). Split on "|" at the API layer, same as
    # aliases_bn is split on "|" wherever it's read.
    prescription_channels = Column(String, nullable=True)


# ============================================================================
# INSURANCE PROVIDER / INSURANCE POLICY
# ============================================================================
#
# ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables. Insurance
# Coverage Policy story.
#
# A caller asks "does my Star Health cover this test" by naming their
# insurer OUT LOUD, the same way they name a test or a doctor -- so
# InsuranceProvider gets the same name+aliases voice-matching shape as
# LabTest.aliases_bn / Doctor.aliases_bn / HealthPackage.aliases, instead
# of being a bare fixed string/enum on the policy row. InsurancePolicy
# itself is a (test, provider) pair, per the plan: one row is "does
# provider X cover test Y", never a whole-provider blanket answer --
# a real insurer's coverage differs test-by-test.
# ============================================================================

class InsuranceProvider(Base):
    __tablename__ = "insurance_providers"

    id = Column(Integer, primary_key=True)

    name = Column(String, nullable=False, unique=True)

    # English + Bengali-script + Hinglish/Banglish aliases, same "|"-joined
    # convention as LabTest.aliases_bn.
    #
    # Example:
    # "star health|স্টার হেলথ|star"
    aliases = Column(String, nullable=False, default="")

    active = Column(Boolean, nullable=False, default=True)

    policies = relationship(
        "InsurancePolicy",
        back_populates="provider",
        cascade="all, delete-orphan",
    )


class InsurancePolicy(Base):
    __tablename__ = "insurance_policies"

    id = Column(Integer, primary_key=True)

    test_id = Column(
        Integer,
        ForeignKey("lab_tests.id"),
        nullable=False,
    )

    provider_id = Column(
        Integer,
        ForeignKey("insurance_providers.id"),
        nullable=False,
    )

    # COVERED / NOT_COVERED / PARTIAL. Nullable with no default, same
    # reasoning as LabTest's advisory/walk-in/prescription columns: no
    # row for a (test, provider) pair means "we don't know", never a
    # guessed COVERED or NOT_COVERED.
    coverage_status = Column(String, nullable=True)

    pre_auth_required = Column(Boolean, nullable=True)

    lab_test = relationship("LabTest")

    provider = relationship(
        "InsuranceProvider",
        back_populates="policies",
    )

    __table_args__ = (
        UniqueConstraint(
            "test_id",
            "provider_id",
            name="uq_test_provider_policy",
        ),
    )


# ============================================================================
# PATIENT
# ============================================================================
#
# This is one of the most important additions.
#
# The old database had patient_name and phone directly inside Appointment.
#
# That is not enough for report-related testing because the system needs
# to identify the SAME patient across multiple reports and conversations.
#
# CHATGPT ADDITION - CREATED BY SOURAV:
# Added patient identity, phone, language preference and spoken-name aliases
# so the agent can test patient-specific report queries.
# ============================================================================

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)

    # Patient's canonical name.
    name = Column(
        String,
        nullable=False,
    )

    # Alternate names / spellings.
    #
    # Example:
    # "Sourav Upadhyay|Sourav|সৌরভ|সৌরভ উপাধ্যায়"
    #
    # This helps test ASR variations.
    name_aliases = Column(
        String,
        nullable=False,
        default="",
    )

    # Phone number used for patient verification and report delivery.
    phone = Column(
        String,
        nullable=False,
        unique=True,
    )

    # Optional secondary phone.
    alternate_phone = Column(
        String,
        nullable=True,
    )

    # Preferred caller language.
    #
    # Supported test values:
    #   english
    #   hinglish
    #   banglish
    #   bengali
    language = Column(
        String,
        nullable=False,
        default="english",
    )

    date_of_birth = Column(
        String,
        nullable=True,
    )

    gender = Column(
        String,
        nullable=True,
    )

    active = Column(
        Boolean,
        nullable=False,
        default=True,
    )

    reports = relationship(
        "LabReport",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    appointments = relationship(
        "Appointment",
        back_populates="patient",
    )


# ============================================================================
# PATIENT BILLING
# ============================================================================
#
# ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables. Outstanding
# Balance / Billing story.
#
# One row per patient (per the plan's schema), not itemized per invoice --
# a running "what do they currently owe" total, mirroring how a caller
# actually asks ("do I have any dues pending"), not an itemized statement.
# NO ROW for a patient is a real, distinct outcome from a row with
# outstanding_amount=0.0: the former means "we have no billing record for
# this patient at all" (honest not-found, same as LabTest's advisory
# columns being None), the latter is a real, reviewed "confirmed zero
# balance". Never create a row just to fill it with a guessed 0.
#
# Identity is resolved by PHONE (patient_id, via Patient.phone), same
# RULE 14/15 discipline as reports -- never by name. Per an explicit
# scoping decision for this first pass: a balance is spoken after a plain
# phone-based lookup, the same friction level as report_status, NOT
# gated behind OTP verification the way report delivery is. Revisit this
# if it should be tightened later -- it was a deliberate choice, not an
# oversight.
# ============================================================================

class PatientBilling(Base):
    __tablename__ = "patient_billing"

    id = Column(Integer, primary_key=True)

    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
        nullable=False,
        unique=True,
    )

    # None = no billing record for this patient (see class docstring
    # above) -- never treat as 0.
    outstanding_amount = Column(Float, nullable=True)

    due_date = Column(DateTime, nullable=True)

    # When this row was last reviewed/updated -- lets a future story
    # answer "how current is this" without guessing.
    updated_at = Column(DateTime, nullable=True)

    patient = relationship("Patient")


# ============================================================================
# LAB REPORT
# ============================================================================
#
# This directly supports:
#
#   "Is my report ready?"
#   "Amar report ready?"
#   "Amar report ta ready hoyeche?"
#   "Report ready na?"
#
# Important statuses:
#
#   READY
#   NOT_READY
#   PROCESSING
#   CANCELLED
#
# A separate NOT_FOUND case happens when no report exists for the
# requested patient/report/test.
#
# CHATGPT ADDITION - CREATED BY SOURAV:
# Added patient-specific report status so the voice agent can test
# ready vs not-ready vs missing-report outcomes.
# ============================================================================

class LabReport(Base):
    __tablename__ = "lab_reports"

    id = Column(Integer, primary_key=True)

    # Human-readable report identifier.
    #
    # Example:
    # LAB-2026-0001
    report_number = Column(
        String,
        nullable=False,
        unique=True,
    )

    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
        nullable=False,
    )

    lab_test_id = Column(
        Integer,
        ForeignKey("lab_tests.id"),
        nullable=False,
    )

    # Sample collection date.
    collected_at = Column(
        DateTime,
        nullable=False,
    )

    # Expected completion time.
    expected_ready_at = Column(
        DateTime,
        nullable=False,
    )

    # Actual completion time.
    ready_at = Column(
        DateTime,
        nullable=True,
    )

    # READY / NOT_READY / PROCESSING / CANCELLED
    status = Column(
        String,
        nullable=False,
        default="PROCESSING",
    )

    # Optional reason when report is delayed.
    status_reason = Column(
        String,
        nullable=True,
    )

    # Whether the report can currently be delivered.
    delivery_enabled = Column(
        Boolean,
        nullable=False,
        default=False,
    )

    # Version helps test situations where a report is regenerated.
    report_version = Column(
        Integer,
        nullable=False,
        default=1,
    )

    patient = relationship(
        "Patient",
        back_populates="reports",
    )

    lab_test = relationship(
        "LabTest",
    )

    deliveries = relationship(
        "ReportDelivery",
        back_populates="report",
        cascade="all, delete-orphan",
    )

    otp_verifications = relationship(
        "ReportOTP",
        back_populates="report",
        cascade="all, delete-orphan",
    )


# ============================================================================
# REPORT OTP
# ============================================================================
#
# Used for:
#
#   "Send my report"
#   "Report pathanor age OTP lagbe?"
#   "OTP is 123456"
#
# Testable states:
#
#   valid OTP
#   wrong OTP
#   expired OTP
#   already-used OTP
#   too many attempts
#
# NOTE:
# In a real production system the OTP should NOT be stored in plaintext.
# This prototype intentionally stores a dummy value so testing is easy.
#
# CHATGPT ADDITION - CREATED BY SOURAV.
# ============================================================================

# ADDED BY SOURAV -- the user's own explicit instruction: "the otp and
# other things will not be hardcoded". This replaces two previously
# hardcoded, guessable literals:
#   - clinic-api/main.py's own FRESH_OTP_CODE constant ("135790"), minted
#     every time request_report_delivery() has no reusable OTP row to
#     hand back out.
#   - clinic-api/seed.py's OTP_DATA list, which used to give every seeded
#     ReportOTP row ("482913", "615204", "903217", "731846") a fixed
#     literal too.
# Both call sites now call THIS function instead, so there is exactly one
# place in the whole codebase that decides what an OTP code looks like.
# `secrets.randbelow` (not `random`) because this is a real, security-
# relevant credential (RULE 9: it gates report delivery) even though this
# is a prototype -- see this class's own "Production should store a hash
# instead" comment just below for the next hardening step past this one.
def generate_otp_code() -> str:
    """A real random 6-digit OTP, zero-padded (e.g. "042913") -- never a
    fixed, predictable value. How this code actually reaches the patient
    is a separate, deliberately pluggable concern -- see
    clinic-api/otp_messaging_config.py, the one file a deploying company
    edits to connect this to their own SMS/WhatsApp/e-mail provider."""
    return f"{secrets.randbelow(1_000_000):06d}"


class ReportOTP(Base):
    __tablename__ = "report_otps"

    id = Column(Integer, primary_key=True)

    report_id = Column(
        Integer,
        ForeignKey("lab_reports.id"),
        nullable=False,
    )

    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
        nullable=False,
    )

    # Phone number to which OTP was sent.
    phone = Column(
        String,
        nullable=False,
    )

    # Prototype-only OTP.
    #
    # Production should store a hash instead.
    otp_code = Column(
        String,
        nullable=False,
    )

    created_at = Column(
        DateTime,
        nullable=False,
    )

    expires_at = Column(
        DateTime,
        nullable=False,
    )

    verified_at = Column(
        DateTime,
        nullable=True,
    )

    used = Column(
        Boolean,
        nullable=False,
        default=False,
    )

    attempt_count = Column(
        Integer,
        nullable=False,
        default=0,
    )

    max_attempts = Column(
        Integer,
        nullable=False,
        default=3,
    )

    report = relationship(
        "LabReport",
        back_populates="otp_verifications",
    )

    patient = relationship(
        "Patient",
    )


# ============================================================================
# REPORT DELIVERY
# ============================================================================
#
# This handles the second story:
#
#   "Send my report."
#
# The acceptance criteria require:
#
#   - OTP verification
#   - expiring signed link
#   - audit trail
#   - recipient
#   - verification path
#   - failed verification should offer collection in person
#
# CHATGPT ADDITION - CREATED BY SOURAV:
# Added explicit delivery state and signed-link expiry so the test system
# can intentionally create success and failure scenarios.
# ============================================================================

class ReportDelivery(Base):
    __tablename__ = "report_deliveries"

    id = Column(Integer, primary_key=True)

    report_id = Column(
        Integer,
        ForeignKey("lab_reports.id"),
        nullable=False,
    )

    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
        nullable=False,
    )

    # Phone/email destination.
    recipient = Column(
        String,
        nullable=False,
    )

    # PHONE / EMAIL / WHATSAPP etc.
    delivery_channel = Column(
        String,
        nullable=False,
        default="SMS",
    )

    # OTP_REQUIRED / VERIFIED / FAILED / EXPIRED
    verification_status = Column(
        String,
        nullable=False,
        default="OTP_REQUIRED",
    )

    # PENDING / SENT / FAILED / EXPIRED
    delivery_status = Column(
        String,
        nullable=False,
        default="PENDING",
    )

    # Unique signed URL identifier.
    #
    # Do NOT use a real permanent URL in this prototype.
    signed_link_token = Column(
        String,
        nullable=True,
        unique=True,
    )

    # When the signed link becomes invalid.
    signed_link_expires_at = Column(
        DateTime,
        nullable=True,
    )

    # Number of times delivery was attempted.
    attempt_count = Column(
        Integer,
        nullable=False,
        default=0,
    )

    created_at = Column(
        DateTime,
        nullable=False,
    )

    verified_at = Column(
        DateTime,
        nullable=True,
    )

    sent_at = Column(
        DateTime,
        nullable=True,
    )

    failed_at = Column(
        DateTime,
        nullable=True,
    )

    # Human-readable failure reason.
    #
    # Examples:
    # "WRONG_OTP"
    # "OTP_EXPIRED"
    # "LINK_EXPIRED"
    # "DELIVERY_PROVIDER_FAILURE"
    # "PHONE_MISMATCH"
    failure_reason = Column(
        String,
        nullable=True,
    )

    # Audit information.
    #
    # Example:
    # "OTP verified on registered phone ending 1234"
    audit_note = Column(
        Text,
        nullable=True,
    )

    report = relationship(
        "LabReport",
        back_populates="deliveries",
    )

    patient = relationship(
        "Patient",
    )


# ============================================================================
# APPOINTMENT
# ============================================================================

class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True)

    confirmation_id = Column(
        String,
        nullable=False,
        unique=True,
    )

    doctor_id = Column(
        Integer,
        ForeignKey("doctors.id"),
        nullable=False,
    )

    date = Column(
        String,
        nullable=False,
    )

    time_slot = Column(
        String,
        nullable=False,
    )

    # Keep these fields for backward compatibility with the existing
    # appointment seed/tooling.
    patient_name = Column(
        String,
        nullable=False,
    )

    phone = Column(
        String,
        nullable=False,
    )

    # Optional connection to the new Patient table.
    #
    # Existing appointments can still work even if patient_id is NULL.
    patient_id = Column(
        Integer,
        ForeignKey("patients.id"),
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
    )

    doctor = relationship(
        "Doctor",
    )

    patient = relationship(
        "Patient",
        back_populates="appointments",
    )

    __table_args__ = (
        UniqueConstraint(
            "doctor_id",
            "date",
            "time_slot",
            name="uq_doctor_slot",
        ),
    )