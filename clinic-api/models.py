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