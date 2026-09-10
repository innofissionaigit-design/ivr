"""
Populates dummy clinic data for the voice-care-agent prototype.

EXISTING DATA:
- 8 departments
- 32 doctors (4 each)
- Weekly chamber schedule per doctor
- 34 lab tests

ADDED TESTING DATA:
- Clinic information
- Opening / closing time
- Address
- Directions
- Health packages
- Patients
- Patient phone numbers
- Patient emails
- Patient verification status
- Lab reports
- READY / NOT_READY / PROCESSING states
- Multiple reports per patient
- OTP verification data
- Valid / expired / used OTP cases
- Report delivery records
- Signed-link expiry
- Delivery audit trail

All fictional -- names, qualifications, prices, patient data,
clinic details and test data are representative prototype data
and are NOT sourced from real patient or clinic records.

Run:
    python3 seed.py

The script is idempotent:
it wipes the existing database and reloads all seed data.
"""

from __future__ import annotations

# ============================================================
# ADDED BY CHATGPT FOR SOURAV
#
# Prompt used:
# "add patient details with phone number, OTP etc."
# "add opening and closing time, address, directions,
# health packages, report ready/not ready and delivery testing."
#
# Created by ChatGPT for Sourav.
#
# datetime/timedelta are required for:
# - report generation time
# - report ready time
# - OTP expiry
# - signed-link expiry
# ============================================================

from datetime import datetime, timedelta


from db import engine, SessionLocal


# ============================================================
# ADDED BY CHATGPT FOR SOURAV
#
# Prompt used:
# "give me the full updated seed.py with all data required
# for testing report readiness and report delivery."
#
# Created by ChatGPT for Sourav.
#
# These additional models are used for:
# - clinic information
# - health packages
# - patients
# - reports
# - OTP verification
# - report delivery
# - delivery audit
# ============================================================

from models import (
    Base,
    Department,
    Doctor,
    DoctorSchedule,
    LabTest,

    # Added models
    ClinicInfo,
    HealthPackage,
    Patient,
    LabReport,
    OTPVerification,
    ReportDelivery,
    ReportDeliveryAudit,
)


# ---------------------------------------------------------------------------
# 8 departments, 4 doctors each = 32 doctors.
#
# Schedule pattern rotates across 4 shift templates so the 32 doctors don't
# all sit at the same time -- a caller asking "is anyone free this evening"
# gets a realistic mixed answer.
# ---------------------------------------------------------------------------

# Bengali-script spelling(s) for each surname, so a caller saying
# "ডক্টর সেন" actually matches "Dr. A. Sen".
#
# Keyed on the surname as it appears as the last word of the English name.

SURNAME_BN = {
    "Mukherjee": ["মুখার্জী", "মুখোপাধ্যায়"],
    "Sen": ["সেন"],
    "Ghosh": ["ঘোষ"],
    "Chowdhury": ["চৌধুরী"],
    "Bhattacharya": ["ভট্টাচার্য"],
    "Roy": ["রায়"],
    "Banerjee": ["ব্যানার্জী", "বন্দ্যোপাধ্যায়"],
    "Dutta": ["দত্ত"],
    "Chatterjee": ["চ্যাটার্জী", "চট্টোপাধ্যায়"],
    "Basu": ["বসু"],
    "Mitra": ["মিত্র"],
    "Sengupta": ["সেনগুপ্ত"],
    "Das": ["দাস"],
    "Bose": ["বসু"],
    "Kar": ["কর"],
    "Nandi": ["নন্দী"],
    "Pal": ["পাল"],
    "Halder": ["হালদার"],
    "Guha": ["গুহ"],
    "Chanda": ["চন্দ", "চাঁদা"],
    "Saha": ["সাহা"],
    "Dey": ["দে"],
    "Adhikari": ["অধিকারী"],
    "Bagchi": ["বাগচী"],
    "Biswas": ["বিশ্বাস"],
    "Majumder": ["মজুমদার"],
    "Mondal": ["মন্ডল"],
    "Ganguly": ["গাঙ্গুলী", "গঙ্গোপাধ্যায়"],
    "Sinha": ["সিনহা"],
    "Ray": ["রায়"],
    "Sarkar": ["সরকার"],
    "Chakraborty": ["চক্রবর্তী"],
}


SHIFT_TEMPLATES = [
    {
        "days": [0, 2, 4],
        "start": "10:00",
        "end": "12:00",
    },

    {
        "days": [1, 3, 5],
        "start": "10:00",
        "end": "12:00",
    },

    {
        "days": [0, 2, 4],
        "start": "18:00",
        "end": "20:00",
    },

    {
        "days": [1, 3, 5],
        "start": "17:30",
        "end": "19:30",
    },
]


DEPARTMENTS = {
    "General Medicine": [
        ("Dr. S. Mukherjee", "MBBS, MD (Gen. Med.)"),
        ("Dr. A. Sen", "MBBS, MD (Gen. Med.)"),
        ("Dr. P. Ghosh", "MBBS, DNB (Gen. Med.)"),
        ("Dr. R. Chowdhury", "MBBS, MD"),
    ],

    "Cardiology": [
        ("Dr. K. Bhattacharya", "MBBS, MD, DM (Cardiology)"),
        ("Dr. N. Roy", "MBBS, DM (Cardiology)"),
        ("Dr. S. Banerjee", "MBBS, MD, DM (Cardiology)"),
        ("Dr. M. Dutta", "MBBS, DM (Cardiology)"),
    ],

    "Gynaecology & Obstetrics": [
        ("Dr. S. Chatterjee", "MBBS, MS (Obs & Gynae)"),
        ("Dr. A. Basu", "MBBS, DGO"),
        ("Dr. R. Mitra", "MBBS, MS (Obs & Gynae)"),
        ("Dr. P. Sengupta", "MBBS, DNB (Obs & Gynae)"),
    ],

    "Orthopaedics": [
        ("Dr. D. Das", "MBBS, MS (Ortho)"),
        ("Dr. T. Bose", "MBBS, D.Ortho"),
        ("Dr. A. Kar", "MBBS, MS (Ortho)"),
        ("Dr. S. Nandi", "MBBS, DNB (Ortho)"),
    ],

    "ENT": [
        ("Dr. R. Pal", "MBBS, MS (ENT)"),
        ("Dr. K. Halder", "MBBS, DLO"),
        ("Dr. S. Guha", "MBBS, MS (ENT)"),
        ("Dr. B. Chanda", "MBBS, DLO"),
    ],

    "Dermatology": [
        ("Dr. M. Saha", "MBBS, MD (Dermatology)"),
        ("Dr. A. Dey", "MBBS, DVD"),
        ("Dr. P. Adhikari", "MBBS, MD (Dermatology)"),
        ("Dr. S. Bagchi", "MBBS, DDV"),
    ],

    "Paediatrics": [
        ("Dr. N. Biswas", "MBBS, MD (Paediatrics)"),
        ("Dr. R. Majumder", "MBBS, DCH"),
        ("Dr. A. Mondal", "MBBS, MD (Paediatrics)"),
        ("Dr. S. Ganguly", "MBBS, DCH"),
    ],

    "Diabetology & Endocrinology": [
        ("Dr. K. Sinha", "MBBS, MD, DM (Endocrinology)"),
        ("Dr. P. Ray", "MBBS, MD (Diabetology)"),
        ("Dr. A. Sarkar", "MBBS, DM (Endocrinology)"),
        ("Dr. S. Chakraborty", "MBBS, MD (Diabetology)"),
    ],
}


# Bengali and English aliases for departments to handle short forms like
# "ortho".

DEPARTMENT_ALIASES = {
    "General Medicine": [
        "জেনারেল মেডিসিন",
        "জেনারেল",
        "medicine",
        "general",
    ],

    "Cardiology": [
        "কার্ডিওলজি",
        "হার্ট",
        "heart",
        "cardio",
        "cardiology",
    ],

    "Gynaecology & Obstetrics": [
        "গাইনি",
        "gyne",
        "gynaecology",
        "obstetrics",
        "women",
    ],

    "Orthopaedics": [
        "অর্থোপেডিক্স",
        "অর্থো",
        "ortho",
        "orthopedics",
        "bone",
        "bones",
    ],

    "ENT": [
        "ইএনটি",
        "ent",
        "ear",
        "nose",
        "throat",
    ],

    "Dermatology": [
        "ডার্মাটোলজি",
        "স্কিন",
        "skin",
        "derma",
        "dermatology",
    ],

    "Paediatrics": [
        "পিডিয়াট্রিক্স",
        "শিশু",
        "child",
        "children",
        "paedia",
        "pediatrics",
    ],

    "Diabetology & Endocrinology": [
        "ডায়াবেটোলজি",
        "সুগার",
        "diabetes",
        "sugar",
        "endo",
        "endocrinology",
    ],
}


# ---------------------------------------------------------------------------
# Lab test data
#
# (name, [Bengali aliases], rate_inr, sample_type, report_time_hours)
# ---------------------------------------------------------------------------

LAB_TESTS = [
    (
        "Complete Blood Count (CBC)",
        ["সিবিসি", "সি বি সি"],
        400,
        "Blood",
        6,
    ),

    (
        "ESR",
        ["ইএসআর", "ই এস আর"],
        150,
        "Blood",
        6,
    ),

    (
        "Blood Sugar Fasting",
        ["ব্লাড সুগার ফাস্টিং", "সুগার ফাস্টিং", "খালি পেটে সুগার"],
        120,
        "Blood",
        4,
    ),

    (
        "Blood Sugar PP",
        ["সুগার পিপি", "পিপি সুগার", "খাওয়ার পরে সুগার"],
        120,
        "Blood",
        4,
    ),

    (
        "HbA1c",
        ["এইচবিএ১সি", "হিমোগ্লোবিন এ১সি"],
        650,
        "Blood",
        24,
    ),

    (
        "Lipid Profile",
        ["লিপিড প্রোফাইল", "কোলেস্টেরল টেস্ট"],
        670.17,
        "Blood",
        24,
    ),

    (
        "Liver Function Test (LFT)",
        ["লিভার ফাংশন টেস্ট", "এলএফটি", "লিভার টেস্ট"],
        800,
        "Blood",
        24,
    ),

    (
        "Kidney Function Test (KFT)",
        ["কিডনি ফাংশন টেস্ট", "কেএফটি", "কিডনি টেস্ট"],
        750,
        "Blood",
        24,
    ),

    (
        "Thyroid Profile (T3 T4 TSH)",
        ["থাইরয়েড প্রোফাইল", "থাইরয়েড টেস্ট"],
        700,
        "Blood",
        24,
    ),

    (
        "TSH",
        ["টিএসএইচ"],
        350,
        "Blood",
        24,
    ),

    (
        "Urine Routine Examination",
        ["ইউরিন টেস্ট", "প্রস্রাব পরীক্ষা", "ইউরিন রুটিন"],
        200,
        "Urine",
        6,
    ),

    (
        "Widal Test",
        ["ওয়াইডাল টেস্ট", "উইডাল টেস্ট", "টাইফয়েড টেস্ট"],
        250,
        "Blood",
        12,
    ),

    (
        "Dengue NS1 Antigen",
        ["ডেঙ্গু এনএস১", "ডেঙ্গু টেস্ট"],
        900,
        "Blood",
        6,
    ),

    (
        "Dengue IgG/IgM",
        ["ডেঙ্গু আইজিজি", "ডেঙ্গু আইজিএম"],
        900,
        "Blood",
        6,
    ),

    (
        "Malaria Antigen",
        ["ম্যালেরিয়া টেস্ট", "ম্যালেরিয়া এন্টিজেন"],
        400,
        "Blood",
        4,
    ),

    (
        "CRP (C-Reactive Protein)",
        ["সিআরপি"],
        500,
        "Blood",
        12,
    ),

    (
        "Vitamin D (25-OH)",
        ["ভিটামিন ডি"],
        1800,
        "Blood",
        72,
    ),

    (
        "Vitamin B12",
        ["ভিটামিন বি১২", "বি১২"],
        1249.00,
        "Blood",
        48,
    ),

    (
        "Serum Creatinine",
        ["ক্রিয়াটিনিন", "সিরাম ক্রিয়াটিনিন"],
        250,
        "Blood",
        12,
    ),

    (
        "Serum Electrolytes",
        ["ইলেক্ট্রোলাইটস", "ইলেকট্রোলাইট টেস্ট"],
        450,
        "Blood",
        12,
    ),

    (
        "Blood Grouping & Rh Typing",
        ["ব্লাড গ্রুপ", "রক্তের গ্রুপ"],
        200,
        "Blood",
        4,
    ),

    (
        "HIV Test (ELISA)",
        ["এইচআইভি টেস্ট", "এইডস টেস্ট"],
        500,
        "Blood",
        24,
    ),

    (
        "HBsAg",
        ["এইচবিএসএজি", "হেপাটাইটিস বি"],
        400,
        "Blood",
        24,
    ),

    (
        "HCV",
        ["এইচসিভি", "হেপাটাইটিস সি"],
        600,
        "Blood",
        24,
    ),

    (
        "ECG",
        ["ইসিজি", "ইলেক্ট্রোকার্ডিওগ্রাম"],
        355.50,
        "Cardiac",
        1,
    ),

    (
        "Chest X-Ray (PA view)",
        ["বুকের এক্স-রে", "চেস্ট এক্সরে"],
        400,
        "Imaging",
        4,
    ),

    (
        "USG Whole Abdomen",
        ["পেটের আলট্রাসাউন্ড", "হোল অ্যাবডোমেন ইউএসজি", "পেটের ইউএসজি"],
        1500,
        "Imaging",
        4,
    ),

    (
        "USG Pregnancy Profile",
        ["প্রেগন্যান্সি আলট্রাসাউন্ড", "প্রেগনেন্সি ইউএসজি"],
        1600,
        "Imaging",
        4,
    ),

    (
        "2D Echocardiography",
        ["ইকো টেস্ট", "একোকার্ডিওগ্রাফি", "ইকোকার্ডিওগ্রাম"],
        2000,
        "Cardiac",
        4,
    ),

    (
        "TMT (Treadmill Test)",
        ["টিএমটি", "ট্রেডমিল টেস্ট"],
        2200,
        "Cardiac",
        4,
    ),

    (
        "Pap Smear",
        ["প্যাপ স্মিয়ার"],
        989.10,
        "Sample (Cervical)",
        72,
    ),

    (
        "PSA (Prostate Specific Antigen)",
        ["পিএসএ"],
        900,
        "Blood",
        48,
    ),

    (
        "Uric Acid",
        ["ইউরিক অ্যাসিড", "ইউরিক এসিড", "ইউরিক এসিদ"],
        250,
        "Blood",
        12,
    ),

    (
        "Calcium (Serum)",
        ["ক্যালসিয়াম", "সিরাম ক্যালসিয়াম"],
        250,
        "Blood",
        12,
    ),
]


def seed():

    # ============================================================
    # EXISTING BEHAVIOUR
    #
    # The database is wiped and recreated every time.
    #
    # This is useful during testing because you always start with
    # exactly the same test data.
    # ============================================================

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    db = SessionLocal()

    try:

        # ========================================================
        # SECTION 1
        # EXISTING: DEPARTMENTS
        #
        # Used for questions like:
        #
        # "Do you have a cardiology department?"
        # "I need an ortho doctor."
        # "Which department handles skin problems?"
        # ========================================================

        doctor_index = 0

        for dept_name, doctors in DEPARTMENTS.items():

            aliases = "|".join(
                DEPARTMENT_ALIASES.get(dept_name, [])
            )

            dept = Department(
                name=dept_name,
                aliases_bn=aliases,
            )

            db.add(dept)

            # Get department ID before creating doctors.
            db.flush()

            # ====================================================
            # SECTION 2
            # EXISTING: DOCTORS
            #
            # Used for:
            #
            # "Do you have Dr. Sen?"
            # "Which cardiologists are available?"
            # "What qualification does Dr. Roy have?"
            # ====================================================

            for doc_name, quals in doctors:

                surname = doc_name.split()[-1]

                aliases = "|".join(
                    SURNAME_BN.get(surname, [])
                )

                doc = Doctor(
                    name=doc_name,
                    qualifications=quals,
                    aliases_bn=aliases,
                    department_id=dept.id,
                )

                db.add(doc)

                db.flush()

                # =================================================
                # SECTION 3
                # EXISTING: DOCTOR SCHEDULE
                #
                # Used for:
                #
                # "When is Dr. Sen available?"
                # "Is any doctor available this evening?"
                # =================================================

                template = SHIFT_TEMPLATES[
                    doctor_index % len(SHIFT_TEMPLATES)
                ]

                for weekday in template["days"]:

                    db.add(
                        DoctorSchedule(
                            doctor_id=doc.id,
                            weekday=weekday,
                            start_time=template["start"],
                            end_time=template["end"],
                        )
                    )

                doctor_index += 1

        # ========================================================
        # SECTION 4
        # EXISTING: LAB TESTS
        #
        # Used for:
        #
        # "How much is CBC?"
        # "What sample is needed for CBC?"
        # "How long does a Vitamin D report take?"
        #
        # sample_type and report_time_hours are especially useful
        # for the voice-agent test cases.
        # ========================================================

        for name, aliases_bn, rate, sample, hours in LAB_TESTS:

            db.add(
                LabTest(
                    name=name,
                    aliases_bn="|".join(aliases_bn),
                    rate_inr=rate,
                    sample_type=sample,
                    report_time_hours=hours,
                )
            )

        db.flush()

        # ========================================================
        # ADDED BY CHATGPT FOR SOURAV
        #
        # Prompt:
        # "add opening and closing time, address and directions."
        #
        # Created by ChatGPT for Sourav.
        #
        # WHY:
        # This supports clinic-information questions.
        #
        # Example:
        # "What time do you open?"
        # "When do you close?"
        # "Where are you located?"
        # "How do I reach the clinic?"
        #
        # All values are fictional prototype data.
        # ========================================================

        clinic = ClinicInfo(
            clinic_name="Kolkata Care Polyclinic",

            address=(
                "42 Lake View Road, Kolkata, "
                "West Bengal 700029"
            ),

            directions=(
                "Near Lake View Crossing. "
                "Approximately 5 minutes from the nearest "
                "metro station. The clinic is on the ground floor."
            ),

            opening_time="08:00",
            closing_time="20:00",

            phone="03340001234",
        )

        db.add(clinic)

        db.flush()

        # ========================================================
        # ADDED BY CHATGPT FOR SOURAV
        #
        # Prompt:
        # "add health packages."
        #
        # Created by ChatGPT for Sourav.
        #
        # WHY:
        # Used to test:
        #
        # "What health packages do you have?"
        # "How much is the diabetes package?"
        # "What tests are included?"
        #
        # All package information is fictional.
        # ========================================================

        HEALTH_PACKAGES = [

            {
                "name": "Basic Health Checkup",

                "description": (
                    "Routine screening package "
                    "for general health."
                ),

                "price": 999,

                "tests": (
                    "Complete Blood Count (CBC), "
                    "Blood Sugar Fasting, "
                    "Lipid Profile"
                ),
            },

            {
                "name": "Diabetes Screening Package",

                "description": (
                    "Basic diabetes-focused "
                    "screening package."
                ),

                "price": 1299,

                "tests": (
                    "Blood Sugar Fasting, "
                    "Blood Sugar PP, "
                    "HbA1c"
                ),
            },

            {
                "name": "Full Body Health Package",

                "description": (
                    "Comprehensive routine "
                    "health screening."
                ),

                "price": 2499,

                "tests": (
                    "Complete Blood Count (CBC), "
                    "Blood Sugar Fasting, "
                    "HbA1c, "
                    "Lipid Profile, "
                    "Liver Function Test (LFT), "
                    "Kidney Function Test (KFT), "
                    "Thyroid Profile (T3 T4 TSH)"
                ),
            },

            {
                "name": "Women's Wellness Package",

                "description": (
                    "Routine women's health "
                    "screening package."
                ),

                "price": 2199,

                "tests": (
                    "Complete Blood Count (CBC), "
                    "Thyroid Profile (T3 T4 TSH), "
                    "Vitamin D (25-OH), "
                    "Pap Smear"
                ),
            },
        ]

        for package in HEALTH_PACKAGES:

            db.add(
                HealthPackage(
                    name=package["name"],
                    description=package["description"],
                    price_inr=package["price"],
                    included_tests=package["tests"],
                )
            )

        db.flush()

        # ========================================================
        # ADDED BY CHATGPT FOR SOURAV
        #
        # Prompt:
        # "user details with test report ready or not is missing.
        # add some patient details with phone number."
        #
        # Created by ChatGPT for Sourav.
        #
        # WHY:
        # The voice agent needs actual patients in the database
        # to test report-related conversations.
        #
        # Each patient has:
        # - name
        # - phone
        # - email
        # - phone verification state
        #
        # All patient data is fictional.
        # ========================================================

        PATIENTS = [

            {
                "name": "Arjun Sen",
                "phone": "9000000001",
                "email": "arjun.test@example.com",
                "verified": True,
            },

            {
                "name": "Riya Das",
                "phone": "9000000002",
                "email": "riya.test@example.com",
                "verified": True,
            },

            {
                "name": "Rahul Ghosh",
                "phone": "9000000003",
                "email": "rahul.test@example.com",
                "verified": True,
            },

            {
                "name": "Mita Roy",
                "phone": "9000000004",
                "email": "mita.test@example.com",
                "verified": False,
            },

            {
                "name": "Amit Banerjee",
                "phone": "9000000005",
                "email": "amit.test@example.com",
                "verified": True,
            },

            {
                "name": "Sohini Mukherjee",
                "phone": "9000000006",
                "email": "sohini.test@example.com",
                "verified": True,
            },
        ]

        patient_objects = {}

        for data in PATIENTS:

            patient = Patient(
                name=data["name"],
                phone=data["phone"],
                email=data["email"],
                phone_verified=data["verified"],
            )

            db.add(patient)

            db.flush()

            patient_objects[data["name"]] = patient

        # ========================================================
        # ADDED BY CHATGPT FOR SOURAV
        #
        # Prompt:
        # "add report ready or not and edge-case report data."
        #
        # Created by ChatGPT for Sourav.
        #
        # WHY:
        # This is the main dataset for the story:
        #
        # "Caller asks whether their report is ready."
        #
        # We deliberately create:
        #
        # READY
        # NOT_READY
        # PROCESSING
        # Multiple reports for one patient
        #
        # IMPORTANT:
        # There is intentionally NO clinical result/value here.
        #
        # This allows the agent to answer:
        # "Your report is ready."
        #
        # without accidentally reading a medical result.
        # ========================================================

        now = datetime.now()

        # Build a lookup dictionary so we can find LabTest objects.
        lab_tests = {
            test.name: test
            for test in db.query(LabTest).all()
        }

        REPORT_DATA = [

            # ----------------------------------------------------
            # HAPPY PATH
            #
            # Arjun has a READY CBC report.
            #
            # Test:
            # "Is my CBC report ready?"
            # ----------------------------------------------------

            {
                "patient": "Arjun Sen",
                "test": "Complete Blood Count (CBC)",
                "report_id": "RPT-10001",
                "status": "READY",
                "hours_ago": 8,
            },

            # ----------------------------------------------------
            # NOT READY
            #
            # Test:
            # "Is my lipid report ready?"
            #
            # Expected:
            # NOT READY
            # ----------------------------------------------------

            {
                "patient": "Riya Das",
                "test": "Lipid Profile",
                "report_id": "RPT-10002",
                "status": "NOT_READY",
                "hours_ago": 4,
            },

            # ----------------------------------------------------
            # PROCESSING
            #
            # Tests whether the agent incorrectly converts
            # PROCESSING into READY or NOT_READY.
            # ----------------------------------------------------

            {
                "patient": "Rahul Ghosh",
                "test": "Vitamin D (25-OH)",
                "report_id": "RPT-10003",
                "status": "PROCESSING",
                "hours_ago": 2,
            },

            # ----------------------------------------------------
            # MULTIPLE REPORTS
            #
            # Same patient has:
            #
            # CBC     -> READY
            # Thyroid -> NOT_READY
            #
            # Test:
            # "Is my report ready?"
            #
            # Agent should clarify which report if required.
            # ----------------------------------------------------

            {
                "patient": "Mita Roy",
                "test": "Complete Blood Count (CBC)",
                "report_id": "RPT-10004",
                "status": "READY",
                "hours_ago": 10,
            },

            {
                "patient": "Mita Roy",
                "test": "Thyroid Profile (T3 T4 TSH)",
                "report_id": "RPT-10005",
                "status": "NOT_READY",
                "hours_ago": 3,
            },

            # ----------------------------------------------------
            # MULTIPLE REPORTS - SECOND PATIENT
            #
            # Sugar Fasting -> READY
            # HbA1c         -> NOT_READY
            # ----------------------------------------------------

            {
                "patient": "Amit Banerjee",
                "test": "Blood Sugar Fasting",
                "report_id": "RPT-10006",
                "status": "READY",
                "hours_ago": 12,
            },

            {
                "patient": "Amit Banerjee",
                "test": "HbA1c",
                "report_id": "RPT-10007",
                "status": "NOT_READY",
                "hours_ago": 1,
            },

            # ----------------------------------------------------
            # READY REPORT FOR DELIVERY TESTING
            # ----------------------------------------------------

            {
                "patient": "Sohini Mukherjee",
                "test": "Liver Function Test (LFT)",
                "report_id": "RPT-10008",
                "status": "READY",
                "hours_ago": 24,
            },
        ]

        report_objects = {}

        for data in REPORT_DATA:

            generated = (
                now
                - timedelta(hours=data["hours_ago"])
            )

            # Only READY reports receive a ready_at time.
            ready_at = (
                generated + timedelta(hours=1)
                if data["status"] == "READY"
                else None
            )

            report = LabReport(
                report_id=data["report_id"],

                patient_id=patient_objects[
                    data["patient"]
                ].id,

                lab_test_id=lab_tests[
                    data["test"]
                ].id,

                status=data["status"],

                generated_at=generated,

                ready_at=ready_at,
            )

            db.add(report)

            db.flush()

            report_objects[
                data["report_id"]
            ] = report

        # ========================================================
        # ADDED BY CHATGPT FOR SOURAV
        #
        # Prompt:
        # "add OTP etc. for report delivery testing."
        #
        # Created by ChatGPT for Sourav.
        #
        # WHY:
        # Story 2 requires:
        #
        # Report
        #    ↓
        # OTP
        #    ↓
        # Verification
        #    ↓
        # Delivery
        #
        # We deliberately create:
        #
        # VALID OTP
        # EXPIRED OTP
        # USED OTP
        # Multiple failed attempts
        #
        # These are test values only.
        # ========================================================

        OTP_DATA = [

            # ----------------------------------------------------
            # VALID OTP
            #
            # Happy-path delivery test.
            # ----------------------------------------------------

            {
                "patient": "Arjun Sen",
                "otp": "482913",
                "status": "VALID",
                "expires_minutes": 10,
                "attempts": 0,
            },

            # ----------------------------------------------------
            # EXPIRED OTP
            #
            # Test:
            # Enter OTP after expiration.
            # ----------------------------------------------------

            {
                "patient": "Sohini Mukherjee",
                "otp": "615204",
                "status": "EXPIRED",
                "expires_minutes": -10,
                "attempts": 0,
            },

            # ----------------------------------------------------
            # USED OTP
            #
            # Test:
            # Try reusing an OTP that has already succeeded.
            # ----------------------------------------------------

            {
                "patient": "Amit Banerjee",
                "otp": "731846",
                "status": "USED",
                "expires_minutes": 10,
                "attempts": 1,
            },

            # ----------------------------------------------------
            # MULTIPLE FAILED ATTEMPTS
            #
            # Useful for testing rate limiting / retry handling.
            # ----------------------------------------------------

            {
                "patient": "Riya Das",
                "otp": "294817",
                "status": "VALID",
                "expires_minutes": 10,
                "attempts": 3,
            },
        ]

        for data in OTP_DATA:

            otp = OTPVerification(

                patient_id=patient_objects[
                    data["patient"]
                ].id,

                otp_code=data["otp"],

                status=data["status"],

                expires_at=(
                    now
                    + timedelta(
                        minutes=data["expires_minutes"]
                    )
                ),

                attempts=data["attempts"],
            )

            db.add(otp)

        db.flush()

        # ========================================================
        # ADDED BY CHATGPT FOR SOURAV
        #
        # Prompt:
        # "add report delivery records with recipient,
        # verification path, signed link expiry and
        # success/failure states."
        #
        # Created by ChatGPT for Sourav.
        #
        # WHY:
        # This supports the second story:
        #
        # "Send my report to me."
        #
        # The data represents:
        #
        # SUCCESS
        # PENDING
        # FAILED
        #
        # It also provides signed-token and expiry data
        # for testing secure report links.
        #
        # These tokens are fictional test tokens.
        # ========================================================

        DELIVERY_DATA = [

            # ----------------------------------------------------
            # SUCCESSFUL DELIVERY
            # ----------------------------------------------------

            {
                "patient": "Arjun Sen",
                "report": "RPT-10001",

                "recipient": (
                    "arjun.test@example.com"
                ),

                "verification": "OTP",

                "token": (
                    "SIGNED-ARJUN-10001"
                ),

                "expires_minutes": 15,

                "status": "SUCCESS",
            },

            # ----------------------------------------------------
            # PENDING DELIVERY
            # ----------------------------------------------------

            {
                "patient": "Sohini Mukherjee",
                "report": "RPT-10008",

                "recipient": (
                    "sohini.test@example.com"
                ),

                "verification": "OTP",

                "token": (
                    "SIGNED-SOHINI-10008"
                ),

                "expires_minutes": 15,

                "status": "PENDING",
            },

            # ----------------------------------------------------
            # FAILED DELIVERY
            #
            # Link has already expired.
            # ----------------------------------------------------

            {
                "patient": "Amit Banerjee",
                "report": "RPT-10006",

                "recipient": (
                    "amit.test@example.com"
                ),

                "verification": "OTP",

                "token": (
                    "SIGNED-AMIT-10006"
                ),

                "expires_minutes": -10,

                "status": "FAILED",
            },
        ]

        for data in DELIVERY_DATA:

            delivery = ReportDelivery(

                report_id=report_objects[
                    data["report"]
                ].id,

                patient_id=patient_objects[
                    data["patient"]
                ].id,

                recipient=data["recipient"],

                verification_method=(
                    data["verification"]
                ),

                signed_token=data["token"],

                link_expires_at=(
                    now
                    + timedelta(
                        minutes=data["expires_minutes"]
                    )
                ),

                delivery_status=data["status"],
            )

            db.add(delivery)

        db.flush()

        # ========================================================
        # ADDED BY CHATGPT FOR SOURAV
        #
        # Prompt:
        # "add audit trail data because every delivery must be
        # written to the audit trail with recipient and
        # verification path."
        #
        # Created by ChatGPT for Sourav.
        #
        # WHY:
        # Your acceptance criteria explicitly requires:
        #
        # Every delivery -> audit record
        #
        # The audit should tell us:
        #
        # Who requested it
        # Which report
        # Who received it
        # How verification happened
        # Whether delivery succeeded
        # Why it failed
        # ========================================================

        AUDIT_DATA = [

            # ----------------------------------------------------
            # SUCCESSFUL DELIVERY
            # ----------------------------------------------------

            {
                "patient": "Arjun Sen",
                "report": "RPT-10001",

                "recipient": (
                    "arjun.test@example.com"
                ),

                "verification": "OTP",

                "status": "SUCCESS",

                "reason": None,
            },

            # ----------------------------------------------------
            # FAILED DELIVERY
            #
            # Useful for testing expired signed links.
            # ----------------------------------------------------

            {
                "patient": "Amit Banerjee",
                "report": "RPT-10006",

                "recipient": (
                    "amit.test@example.com"
                ),

                "verification": "OTP",

                "status": "FAILED",

                "reason": "Signed link expired",
            },

            # ----------------------------------------------------
            # FAILED BECAUSE REPORT IS NOT READY
            # ----------------------------------------------------

            {
                "patient": "Riya Das",
                "report": "RPT-10002",

                "recipient": (
                    "riya.test@example.com"
                ),

                "verification": "OTP",

                "status": "FAILED",

                "reason": "Report not ready",
            },
        ]

        for data in AUDIT_DATA:

            audit = ReportDeliveryAudit(

                patient_id=patient_objects[
                    data["patient"]
                ].id,

                report_id=report_objects[
                    data["report"]
                ].id,

                recipient=data["recipient"],

                verification_path=(
                    data["verification"]
                ),

                delivery_status=data["status"],

                failure_reason=data["reason"],

                created_at=now,
            )

            db.add(audit)

        # ========================================================
        # COMMIT EVERYTHING
        # ========================================================

        db.commit()

        print(
            f"Seeded {len(DEPARTMENTS)} departments, "
            f"{doctor_index} doctors, "
            f"{len(LAB_TESTS)} lab tests, "
            f"{len(PATIENTS)} patients, "
            f"{len(REPORT_DATA)} reports, "
            f"{len(OTP_DATA)} OTP records, "
            f"{len(DELIVERY_DATA)} delivery records, "
            f"{len(AUDIT_DATA)} audit records, "
            f"{len(HEALTH_PACKAGES)} health packages."
        )

    finally:
        db.close()


if __name__ == "__main__":
    seed()