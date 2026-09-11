"""
Populates dummy clinic data for the voice-care-agent prototype.

SUPPORTED LANGUAGES:
- English
- Hinglish
- Bengalish / Banglish
- Bengali script

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
- OTP verification
- VALID / EXPIRED / USED OTP cases
- Report delivery
- Signed-link expiry
- Delivery audit trail

MULTILINGUAL TESTING:
The aliases intentionally contain:
- English
- Hinglish
- Bengalish/Banglish
- Bengali script

This allows the voice agent to resolve common ways callers
may refer to departments, doctors and laboratory tests.

All data is fictional prototype data.

Run:
    python3 seed.py

The script is idempotent:
it wipes the existing database and reloads all seed data.
"""

from __future__ import annotations

from datetime import datetime, timedelta


# ============================================================
# EXISTING DATABASE IMPORT
# ============================================================

from db import engine, SessionLocal


# ============================================================
# ADDED BY CHATGPT FOR SOURAV
#
# Prompt used:
#
# "our system contains both hinglish bengalish, english so
# modify the seed.py if needed and give the full updated code"
#
# Created by ChatGPT for Sourav.
#
# These additional models support:
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
    HealthPackageTest,
    Patient,
    LabReport,
    ReportOTP,
    ReportDelivery,
    # ADDED BY SOURAV -- "otp will not be hardcoded": every seeded
    # ReportOTP row below now gets a real random code from this shared
    # generator instead of a fixed literal -- see its own docstring in
    # models.py for why it lives there rather than duplicated per file.
    generate_otp_code,
)

# NOTE (reconciliation with the real models.py, done after Sourav's
# ChatGPT-assisted edits to models.py and seed.py drifted apart):
#
# - `OTPVerification` was renamed to `ReportOTP` to match models.py.
# - `ReportDeliveryAudit` does not exist as a separate table in
#   models.py. Rather than invent a new model unilaterally, its audit
#   content (recipient, verification path, status, failure reason) is
#   folded into ReportDelivery's own `verification_status`,
#   `failure_reason` and `audit_note` fields below -- including the
#   "delivery requested before the report was ready" case, which now
#   gets its own ReportDelivery row instead of an audit-only entry
#   with no delivery record at all.


# ============================================================
# DOCTOR SURNAME ALIASES
#
# These are used for multilingual voice matching.
#
# Example:
#
# "Dr Sen"
# "Doctor Sen"
# "Dr. Sen"
# "doctor সেন"
# "ডক্টর সেন"
# "ডাক্তার সেন"
#
# Bengalish examples are also included:
#
# "doctor sen ke"
# "sen er doctor"
# ============================================================

SURNAME_BN = {
    "Mukherjee": [
        "মুখার্জী",
        "মুখোপাধ্যায়",
        "মুখার্জি",
        "mukherjee",
        "mukharjee",
    ],

    "Sen": [
        "সেন",
        "sen",
    ],

    "Ghosh": [
        "ঘোষ",
        "ghosh",
    ],

    "Chowdhury": [
        "চৌধুরী",
        "চৌধুরি",
        "chowdhury",
    ],

    "Bhattacharya": [
        "ভট্টাচার্য",
        "ভট্টাচার্য্য",
        "bhattacharya",
    ],

    "Roy": [
        "রায়",
        "রয়",
        "roy",
    ],

    "Banerjee": [
        "ব্যানার্জী",
        "বন্দ্যোপাধ্যায়",
        "ব্যানার্জি",
        "banerjee",
    ],

    "Dutta": [
        "দত্ত",
        "dutta",
    ],

    "Chatterjee": [
        "চ্যাটার্জী",
        "চট্টোপাধ্যায়",
        "চ্যাটার্জি",
        "chatterjee",
    ],

    "Basu": [
        "বসু",
        "basu",
    ],

    "Mitra": [
        "মিত্র",
        "mitra",
    ],

    "Sengupta": [
        "সেনগুপ্ত",
        "sengupta",
    ],

    "Das": [
        "দাস",
        "das",
    ],

    "Bose": [
        "বসু",
        "বোose",
        "bose",
    ],

    "Kar": [
        "কর",
        "kar",
    ],

    "Nandi": [
        "নন্দী",
        "nandi",
    ],

    "Pal": [
        "পাল",
        "pal",
    ],

    "Halder": [
        "হালদার",
        "halder",
    ],

    "Guha": [
        "গুহ",
        "guha",
    ],

    "Chanda": [
        "চন্দ",
        "চাঁদা",
        "chanda",
    ],

    "Saha": [
        "সাহা",
        "saha",
    ],

    "Dey": [
        "দে",
        "dey",
    ],

    "Adhikari": [
        "অধিকারী",
        "adhikari",
    ],

    "Bagchi": [
        "বাগচী",
        "bagchi",
    ],

    "Biswas": [
        "বিশ্বাস",
        "biswas",
    ],

    "Majumder": [
        "মজুমদার",
        "majumdar",
    ],

    "Mondal": [
        "মন্ডল",
        "mondal",
    ],

    "Ganguly": [
        "গাঙ্গুলী",
        "গঙ্গোপাধ্যায়",
        "ganguly",
    ],

    "Sinha": [
        "সিনহা",
        "sinha",
    ],

    "Ray": [
        "রায়",
        "রয়",
        "ray",
    ],

    "Sarkar": [
        "সরকার",
        "sarkar",
    ],

    "Chakraborty": [
        "চক্রবর্তী",
        "chakraborty",
    ],
}


# ============================================================
# DOCTOR SCHEDULE TEMPLATES
# ============================================================

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


# ============================================================
# DEPARTMENTS
# ============================================================

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


# ============================================================
# MULTILINGUAL DEPARTMENT ALIASES
#
# ADDED BY CHATGPT FOR SOURAV
#
# Prompt:
# "our system contains both hinglish bengalish, english"
#
# Created by ChatGPT for Sourav.
#
# WHY:
# A caller may not use the exact department name.
#
# English:
#   cardiology
#
# Hinglish:
#   heart ka doctor
#
# Bengalish:
#   heart er doctor
#
# Bengali:
#   হার্টের ডাক্তার
# ============================================================

DEPARTMENT_ALIASES = {

    "General Medicine": [
        # English
        "general medicine",
        "general",
        "medicine",
        "physician",
        "general doctor",

        # Hinglish
        "general medicine doctor",
        "medicine ka doctor",
        "general doctor chahiye",

        # Bengalish
        "general medicine doctor chai",
        "medicine er doctor",
        "general doctor chai",

        # Bengali
        "জেনারেল মেডিসিন",
        "জেনারেল",
        "মেডিসিন",
        "মেডিসিন ডাক্তার",
    ],

    "Cardiology": [
        # English
        "cardiology",
        "cardiologist",
        "cardio",
        "heart",
        "heart doctor",

        # Hinglish
        "heart ka doctor",
        "dil ka doctor",
        "heart specialist",
        "cardio doctor",

        # Bengalish
        "heart er doctor",
        "hridoy er doctor",
        "heart specialist chai",
        "cardio doctor chai",

        # Bengali
        "কার্ডিওলজি",
        "হার্ট",
        "হার্টের ডাক্তার",
        "হৃদরোগের ডাক্তার",
    ],

    "Gynaecology & Obstetrics": [
        # English
        "gynaecology",
        "gynecology",
        "gynae",
        "gyne",
        "obstetrics",
        "women doctor",
        "women's doctor",

        # Hinglish
        "gynae doctor",
        "ladies doctor",
        "women ka doctor",
        "pregnancy doctor",

        # Bengalish
        "gynae doctor chai",
        "ladies doctor chai",
        "meyeder doctor",
        "pregnancy doctor chai",

        # Bengali
        "গাইনি",
        "গাইনোকলজি",
        "মহিলাদের ডাক্তার",
        "প্রেগন্যান্সি ডাক্তার",
    ],

    "Orthopaedics": [
        # English
        "orthopaedics",
        "orthopedics",
        "ortho",
        "bone doctor",
        "orthopedic doctor",

        # Hinglish
        "ortho doctor",
        "haddi ka doctor",
        "bone ka doctor",
        "haddi wala doctor",

        # Bengalish
        "ortho doctor chai",
        "haddi er doctor",
        "bone er doctor",
        "haddi doctor",

        # Bengali
        "অর্থোপেডিক্স",
        "অর্থো",
        "হাড়ের ডাক্তার",
        "হাড়ের ডাক্তার চাই",
    ],

    "ENT": [
        # English
        "ent",
        "ear",
        "nose",
        "throat",
        "ent doctor",
        "ear nose throat",

        # Hinglish
        "kaan ka doctor",
        "naak ka doctor",
        "gale ka doctor",
        "ent specialist",

        # Bengalish
        "kan er doctor",
        "nak er doctor",
        "golar doctor",
        "ent doctor chai",

        # Bengali
        "ইএনটি",
        "কানের ডাক্তার",
        "নাকের ডাক্তার",
        "গলার ডাক্তার",
    ],

    "Dermatology": [
        # English
        "dermatology",
        "dermatologist",
        "derma",
        "skin",
        "skin doctor",

        # Hinglish
        "skin ka doctor",
        "skin specialist",
        "derma doctor",

        # Bengalish
        "skin er doctor",
        "skin specialist chai",
        "chormo rog er doctor",

        # Bengali
        "ডার্মাটোলজি",
        "স্কিন",
        "ত্বকের ডাক্তার",
        "চর্মরোগের ডাক্তার",
    ],

    "Paediatrics": [
        # English
        "paediatrics",
        "pediatrics",
        "paediatrician",
        "pediatrician",
        "child doctor",
        "children doctor",

        # Hinglish
        "bachche ka doctor",
        "bache ka doctor",
        "child specialist",
        "kids doctor",

        # Bengalish
        "bacchar doctor",
        "bachader doctor",
        "shishu doctor",
        "bacchar doctor chai",

        # Bengali
        "পিডিয়াট্রিক্স",
        "শিশুর ডাক্তার",
        "বাচ্চাদের ডাক্তার",
        "শিশু বিশেষজ্ঞ",
    ],

    "Diabetology & Endocrinology": [
        # English
        "diabetology",
        "diabetes",
        "diabetic doctor",
        "endocrinology",
        "endocrinologist",
        "endo",

        # Hinglish
        "sugar ka doctor",
        "diabetes ka doctor",
        "sugar specialist",
        "hormone doctor",

        # Bengalish
        "sugar er doctor",
        "diabetes er doctor",
        "sugar specialist chai",
        "hormone er doctor",

        # Bengali
        "ডায়াবেটোলজি",
        "সুগারের ডাক্তার",
        "ডায়াবেটিসের ডাক্তার",
        "এন্ডোক্রিনোলজি",
    ],
}


# ============================================================
# LAB TESTS
#
# Each test now has multilingual aliases.
#
# English + Hinglish + Bengalish + Bengali.
#
# This is particularly important for voice testing because
# users will rarely speak the exact database test name.
# ============================================================

LAB_TESTS = [

    (
        "Complete Blood Count (CBC)",
        [
            # English
            "cbc",
            "complete blood count",
            "blood count",

            # Hinglish
            "cbc test",
            "blood count test",
            "mera cbc",

            # Bengalish
            "cbc test chai",
            "amar cbc",
            "cbc report",

            # Bengali
            "সিবিসি",
            "সি বি সি",
            "কমপ্লিট ব্লাড কাউন্ট",

            # ADDED BY SOURAV -- "Caller asks how to prepare for a test"
            # story: merged in from the business's own lab_tests_with_
            # fallback_config sample file (see clinic-api/models.py's own
            # comment on LAB_TEST_ADVISORIES for why). Only genuinely new
            # entries not already covered above.
            "সিবিসি টেস্ট",
        ],
        400,
        "Blood",
        6,
    ),

    (
        "ESR",
        [
            "esr",
            "esr test",
            "e s r",
            "ইএসআর",
            "ই এস আর",
            "esr test chai",
            "esr korate chai",
            "esr korbo",
        ],
        150,
        "Blood",
        6,
    ),

    (
        "Blood Sugar Fasting",
        [
            "blood sugar fasting",
            "fasting sugar",
            "fasting blood sugar",
            "sugar fasting",
            "fasting sugar test",
            "khali pete sugar",
            "khali pet er sugar",
            "khali pete sugar test",
            "খালি পেটে সুগার",
            "ব্লাড সুগার ফাস্টিং",

            # ADDED BY SOURAV -- merged from lab_tests_with_fallback_config
            # (see the CBC entry above for why).
            "fbs",
            "khali pet sugar",
            "সুগার ফাস্টিং",
        ],
        120,
        "Blood",
        4,
    ),

    (
        "Blood Sugar PP",
        [
            "blood sugar pp",
            "pp sugar",
            "post meal sugar",
            "after meal sugar",
            "sugar pp",
            "khabar er pore sugar",
            "khabar por sugar",
            "খাওয়ার পরে সুগার",
            "সুগার পিপি",

            # ADDED BY SOURAV -- merged from lab_tests_with_fallback_config.
            "post prandial blood sugar",
            "khane ke baad sugar",
            "পিপি সুগার",
        ],
        120,
        "Blood",
        4,
    ),

    (
        "HbA1c",
        [
            "hba1c",
            "hb a1c",
            "a1c",
            "hba one c",
            "diabetes average sugar test",
            "sugar er three month test",
            "three month sugar test",
            "tin masher sugar test",
            "তিন মাসের সুগার টেস্ট",
            "এইচবিএ১সি",
            "হিমোগ্লোবিন এ১সি",
        ],
        650,
        "Blood",
        24,
    ),

    (
        "Lipid Profile",
        [
            "lipid profile",
            "lipid test",
            "cholesterol test",
            "cholesterol",
            "lipid",
            "cholesterol ka test",
            "cholesterol er test",
            "cholesterol test chai",
            "লিপিড প্রোফাইল",
            "কোলেস্টেরল টেস্ট",

            # ADDED BY SOURAV -- merged from lab_tests_with_fallback_config.
            "cholesterol profile",
        ],
        # UPDATED BY SOURAV -- "if prices are not same keep the price from
        # seed.py as it is" (the user's own explicit instruction): the
        # sample file's price_inr for this test was 650, disagreeing with
        # this already-seeded 670.17. This value is left exactly as it
        # was -- only the alias above and this test's new advisory data
        # (see LAB_TEST_ADVISORIES below) came from that file.
        670.17,
        "Blood",
        24,
    ),

    (
        "Liver Function Test (LFT)",
        [
            "lft",
            "liver function test",
            "liver test",
            "liver function",
            "lft test",
            "liver ka test",
            "liver er test",
            "liver test chai",
            "লিভার ফাংশন টেস্ট",
            "এলএফটি",
            "লিভার টেস্ট",
        ],
        800,
        "Blood",
        24,
    ),

    (
        "Kidney Function Test (KFT)",
        [
            "kft",
            "kidney function test",
            "kidney test",
            "kidney function",
            "kft test",
            "kidney ka test",
            "kidney er test",
            "kidney test chai",
            "কিডনি ফাংশন টেস্ট",
            "কেএফটি",
            "কিডনি টেস্ট",
        ],
        750,
        "Blood",
        24,
    ),

    (
        "Thyroid Profile (T3 T4 TSH)",
        [
            "thyroid profile",
            "thyroid test",
            "thyroid",
            "t3 t4 tsh",
            "thyroid profile test",
            "thyroid ka test",
            "thyroid er test",
            "thyroid test chai",
            "থাইরয়েড প্রোফাইল",
            "থাইরয়েড টেস্ট",

            # ADDED BY SOURAV -- merged from lab_tests_with_fallback_config.
            "t3 t4 tsh test",
        ],
        700,
        "Blood",
        24,
    ),

    (
        "TSH",
        [
            "tsh",
            "tsh test",
            "thyroid tsh",
            "tsh ka test",
            "tsh er test",
            "tsh test chai",
            "টিএসএইচ",
        ],
        350,
        "Blood",
        24,
    ),

    (
        "Urine Routine Examination",
        [
            "urine test",
            "urine routine",
            "urine routine examination",
            "urine examination",
            "urine test chai",
            "peshab test",
            "peshab ka test",
            "prosab test",
            "prosab er test",
            "ইউরিন টেস্ট",
            "ইউরিন রুটিন",
            "প্রস্রাব পরীক্ষা",

            # ADDED BY SOURAV -- merged from lab_tests_with_fallback_config.
            "urine re",
        ],
        200,
        "Urine",
        6,
    ),

    (
        "Widal Test",
        [
            "widal",
            "widal test",
            "typhoid test",
            "typhoid",
            "widal test chai",
            "typhoid ka test",
            "typhoid er test",
            "টাইফয়েড টেস্ট",
            "ওয়াইডাল টেস্ট",
            "উইডাল টেস্ট",
        ],
        250,
        "Blood",
        12,
    ),

    (
        "Dengue NS1 Antigen",
        [
            "dengue ns1",
            "ns1",
            "dengue test",
            "dengue antigen",
            "ns1 test",
            "dengue ka test",
            "dengue er test",
            "dengue test chai",
            "ডেঙ্গু এনএস১",
            "ডেঙ্গু টেস্ট",
        ],
        900,
        "Blood",
        6,
    ),

    (
        "Dengue IgG/IgM",
        [
            "dengue igg igm",
            "dengue igg",
            "dengue igm",
            "dengue antibody test",
            "dengue test",
            "dengue antibody",
            "ডেঙ্গু আইজিজি",
            "ডেঙ্গু আইজিএম",
        ],
        900,
        "Blood",
        6,
    ),

    (
        "Malaria Antigen",
        [
            "malaria",
            "malaria test",
            "malaria antigen",
            "malaria test chai",
            "malaria ka test",
            "malaria er test",
            "ম্যালেরিয়া টেস্ট",
            "ম্যালেরিয়া এন্টিজেন",
        ],
        400,
        "Blood",
        4,
    ),

    (
        "CRP (C-Reactive Protein)",
        [
            "crp",
            "crp test",
            "c reactive protein",
            "crp ka test",
            "crp er test",
            "crp test chai",
            "সিআরপি",
        ],
        500,
        "Blood",
        12,
    ),

    (
        "Vitamin D (25-OH)",
        [
            "vitamin d",
            "vitamin d test",
            "vit d",
            "25 oh vitamin d",
            "vitamin d ka test",
            "vitamin d er test",
            "vitamin d test chai",
            "ভিটামিন ডি",
            "ভিটামিন ডি টেস্ট",
        ],
        1800,
        "Blood",
        72,
    ),

    (
        "Vitamin B12",
        [
            "vitamin b12",
            "b12",
            "vitamin b twelve",
            "b12 test",
            "vitamin b12 ka test",
            "b12 er test",
            "ভিটামিন বি১২",
            "বি১২",
        ],
        1249.00,
        "Blood",
        48,
    ),

    (
        "Serum Creatinine",
        [
            "creatinine",
            "serum creatinine",
            "creatinine test",
            "creatinine ka test",
            "creatinine er test",
            "ক্রিয়াটিনিন",
            "সিরাম ক্রিয়াটিনিন",
        ],
        250,
        "Blood",
        12,
    ),

    (
        "Serum Electrolytes",
        [
            "electrolytes",
            "serum electrolytes",
            "electrolyte test",
            "electrolytes test",
            "electrolyte ka test",
            "electrolyte er test",
            "ইলেক্ট্রোলাইটস",
            "ইলেকট্রোলাইট টেস্ট",
        ],
        450,
        "Blood",
        12,
    ),

    (
        "Blood Grouping & Rh Typing",
        [
            "blood group",
            "blood grouping",
            "rh typing",
            "blood group test",
            "blood group ka test",
            "blood group er test",
            "amar blood group",
            "ব্লাড গ্রুপ",
            "রক্তের গ্রুপ",
        ],
        200,
        "Blood",
        4,
    ),

    (
        "HIV Test (ELISA)",
        [
            "hiv",
            "hiv test",
            "hiv elisa",
            "hiv screening",
            "hiv ka test",
            "hiv er test",
            "এইচআইভি টেস্ট",
            "এইডস টেস্ট",
        ],
        500,
        "Blood",
        24,
    ),

    (
        "HBsAg",
        [
            "hbsag",
            "hepatitis b",
            "hepatitis b test",
            "hbsag test",
            "hepatitis b ka test",
            "hepatitis b er test",
            "এইচবিএসএজি",
            "হেপাটাইটিস বি",
        ],
        400,
        "Blood",
        24,
    ),

    (
        "HCV",
        [
            "hcv",
            "hcv test",
            "hepatitis c",
            "hepatitis c test",
            "hcv ka test",
            "hcv er test",
            "এইচসিভি",
            "হেপাটাইটিস সি",
        ],
        600,
        "Blood",
        24,
    ),

    (
        "ECG",
        [
            "ecg",
            "ecg test",
            "electrocardiogram",
            "heart test",
            "ecg ka test",
            "ecg er test",
            "heart er test",
            "ইসিজি",
            "ইলেক্ট্রোকার্ডিওগ্রাম",
        ],
        355.50,
        "Cardiac",
        1,
    ),

    (
        "Chest X-Ray (PA view)",
        [
            "chest xray",
            "chest x ray",
            "xray chest",
            "chest x-ray",
            "xray",
            "chest ka xray",
            "buker xray",
            "buk er xray",
            "বুকের এক্স-রে",
            "চেস্ট এক্সরে",

            # ADDED BY SOURAV -- merged from lab_tests_with_fallback_config.
            "chest xray pa",
            "bina fasting xray",
        ],
        400,
        "Imaging",
        4,
    ),

    (
        "USG Whole Abdomen",
        [
            "usg whole abdomen",
            "whole abdomen usg",
            "abdomen ultrasound",
            "stomach ultrasound",
            "whole abdomen scan",
            "pet er ultrasound",
            "pet er usg",
            "pet er scan",
            "পেটের আলট্রাসাউন্ড",
            "পেটের ইউএসজি",
            "হোল অ্যাবডোমেন ইউএসজি",

            # ADDED BY SOURAV -- merged from lab_tests_with_fallback_config.
            "usg abdomen",
            "ultrasound whole abdomen",
            "pet ka usg",
            "ultrasound abdomen",
        ],
        1500,
        "Imaging",
        4,
    ),

    (
        "USG Pregnancy Profile",
        [
            "pregnancy usg",
            "pregnancy ultrasound",
            "pregnancy scan",
            "pregnancy test scan",
            "pregnancy ka ultrasound",
            "pregnancy er usg",
            "pregnancy scan chai",
            "প্রেগন্যান্সি আলট্রাসাউন্ড",
            "প্রেগনেন্সি ইউএসজি",
        ],
        1600,
        "Imaging",
        4,
    ),

    (
        "2D Echocardiography",
        [
            "2d echo",
            "echo",
            "echocardiography",
            "echo test",
            "heart echo",
            "echo ka test",
            "heart er echo",
            "echo test chai",
            "ইকো টেস্ট",
            "একোকার্ডিওগ্রাফি",
            "ইকোকার্ডিওগ্রাম",
        ],
        2000,
        "Cardiac",
        4,
    ),

    (
        "TMT (Treadmill Test)",
        [
            "tmt",
            "tmt test",
            "treadmill test",
            "treadmill",
            "tmt ka test",
            "tmt er test",
            "টিএমটি",
            "ট্রেডমিল টেস্ট",
        ],
        2200,
        "Cardiac",
        4,
    ),

    (
        "Pap Smear",
        [
            "pap smear",
            "pap test",
            "pap smear test",
            "pap test chai",
            "pap smear korate chai",
            "প্যাপ স্মিয়ার",
        ],
        989.10,
        "Sample (Cervical)",
        72,
    ),

    (
        "PSA (Prostate Specific Antigen)",
        [
            "psa",
            "psa test",
            "prostate test",
            "prostate specific antigen",
            "psa ka test",
            "psa er test",
            "পিএসএ",
        ],
        900,
        "Blood",
        48,
    ),

    (
        "Uric Acid",
        [
            "uric acid",
            "uric acid test",
            "uric",
            "uric acid ka test",
            "uric acid er test",
            "uric test chai",
            "ইউরিক অ্যাসিড",
            "ইউরিক এসিড",
            "ইউরিক এসিদ",
        ],
        250,
        "Blood",
        12,
    ),

    (
        "Calcium (Serum)",
        [
            "calcium",
            "serum calcium",
            "calcium test",
            "calcium ka test",
            "calcium er test",
            "calcium test chai",
            "ক্যালসিয়াম",
            "সিরাম ক্যালসিয়াম",
        ],
        250,
        "Blood",
        12,
    ),
]


# ============================================================
# LAB TEST PREPARATION ADVISORIES
#
# ADDED BY SOURAV -- "Caller asks how to prepare for a test" story.
# Sourced verbatim (structured fields AND the four ready-to-speak
# per-language scripts) from the business's own lab_tests_with_
# fallback_config sample file -- nothing here is invented; every
# sentence below is exactly what that file supplied, keyed here by
# LabTest.name so it can be applied after LAB_TESTS creates the rows.
#
# Deliberately keyed by name, NOT folded into the LAB_TESTS tuples
# above: only these 8 of the ~27 tests in LAB_TESTS have real advisory
# content today. Every test not listed here gets no advisory row at
# all -- see models.py's own comment on LabTest's advisory columns for
# why that must stay an honest "we don't know", never a guessed
# default. price_inr is deliberately NOT part of this dict: the user's
# own explicit instruction was to keep LAB_TESTS' existing seeded
# prices untouched even where the sample file's own price_inr disagreed
# (see the Lipid Profile entry above, the one real mismatch found).
#
# advisory_script_* fields all contain a literal "{test_name}"
# placeholder -- agent/reply_templates.py fills that in at speak time
# with whichever name the language should use (the Bengali alias for
# the bengali branch, the English catalogue name everywhere else, the
# same convention every other reply in this codebase already follows).
# ============================================================

LAB_TEST_ADVISORIES = {
    "Complete Blood Count (CBC)": {
        "fasting_required": False,
        "fasting_hours": "0 hours",
        "water_allowance": "Normal water intake allowed",
        "medication_hold": "No medication hold required unless specified by doctor",
        "timing_rule": "Can be taken at any time of the day",
        "advisory_script_en": (
            "For {test_name}, no fasting is required. You can have your "
            "regular food and water. Avoid heavy exercise right before "
            "the test."
        ),
        "advisory_script_hinglish": (
            "{test_name} ke liye fasting ki zaroorat nahi hai. Aap normal "
            "khana aur paani le sakte hain, bas test se theek pehle heavy "
            "workout mat kijiye."
        ),
        "advisory_script_banglish": (
            "{test_name}-er jonno kono fasting-er dorkar nei. Apni normal "
            "kheye-deye aste paren, tobe test-er agey khub heavy workout "
            "korben nah."
        ),
        "advisory_script_bn": (
            "{test_name}-এর জন্য কোনো ফাস্টিং-এর দরকার নেই। আপনি নরমাল "
            "খেয়ে-দেয়ে আসতে পারেন, তবে টেস্ট-এর আগে খুব হেভি ওয়ার্কআউট "
            "করবেন না।"
        ),
    },
    "Blood Sugar Fasting": {
        "fasting_required": True,
        "fasting_hours": "8-12 hours",
        "water_allowance": "Only plain water permitted during fasting period",
        "medication_hold": "Hold morning anti-diabetic medication until after blood collection",
        "timing_rule": "Morning sample collection preferred",
        "advisory_script_en": (
            "For {test_name}, you need to fast for 8 to 12 hours "
            "overnight. Only plain water is allowed. Do not take morning "
            "diabetes medicine, tea, coffee, or cigarettes before the "
            "test."
        ),
        "advisory_script_hinglish": (
            "{test_name} ke liye aapko 8 se 12 ghante khali pet rehna "
            "hoga. Sirf saada paani pee sakte hain. Test se pehle chai, "
            "coffee, cigarette ya morning sugar ki dawai na lein."
        ),
        "advisory_script_banglish": (
            "{test_name}-er jonno apnake 8 theke 12 ghanta khali pete "
            "thakte hobe. Sakal-ebela jol chara r kichu khaben nah — cha, "
            "coffee, cigarette ba sugar-er osudh bondho rakhben."
        ),
        "advisory_script_bn": (
            "{test_name}-এর জন্য আপনাকে ৮ থেকে ১২ ঘণ্টা খালি পেটে থাকতে "
            "হবে। সকাল-বেলা জল ছাড়া আর কিছু খাবেন না — চা, কফি, সিগারেট বা "
            "সুগারের ওষুধ বন্ধ রাখবেন।"
        ),
    },
    "Blood Sugar PP": {
        "fasting_required": False,
        "fasting_hours": "0 hours",
        "water_allowance": "Plain water allowed",
        "medication_hold": "Take prescribed post-meal medications as advised by your doctor",
        "timing_rule": "Exactly 2 hours post-meal",
        "advisory_script_en": (
            "For {test_name}, the sample must be given exactly 2 hours "
            "after your main meal. Do not eat any additional snacks or "
            "drink sugary beverages during these 2 hours."
        ),
        "advisory_script_hinglish": (
            "{test_name} ke liye khana khane ke theek 2 ghante baad "
            "sample dena hoga. Is 2 ghante ke dauran koi extra snacks ya "
            "sugary drinks na lein."
        ),
        "advisory_script_banglish": (
            "{test_name}-er jonno apnar khabar sesh hobar thik 2 ghantar "
            "mathay sample dite hobe. Ei 2 ghantar modhe extra kichu "
            "khaben nah."
        ),
        "advisory_script_bn": (
            "{test_name}-এর জন্য আপনার খাবার শেষ হবার ঠিক ২ ঘণ্টার মাথায় "
            "স্যাম্পল দিতে হবে। এই ২ ঘণ্টার মধ্যে এক্সট্রা কিছু খাবেন না।"
        ),
    },
    "Lipid Profile": {
        "fasting_required": True,
        "fasting_hours": "10-12 hours",
        "water_allowance": "Only plain water permitted",
        "medication_hold": "Do not take lipid-lowering medication on the morning of test without doctor approval",
        "timing_rule": "Morning sample collection preferred",
        "advisory_script_en": (
            "For {test_name}, a mandatory 10 to 12 hour overnight fast is "
            "required. You can drink plain water. Avoid alcohol and heavy "
            "fatty meals 24 hours prior."
        ),
        "advisory_script_hinglish": (
            "{test_name} ke liye 10 se 12 ghante ka fasting zaroori hai. "
            "Aap sirf saada paani pee sakte hain. 24 ghante pehle heavy "
            "fatty khana aur alcohol se bachein."
        ),
        "advisory_script_banglish": (
            "{test_name}-er jonno apnake 10 theke 12 ghanta khali pete "
            "thakte hobe. Sudhu jol khete paren. Test-er 24 ghanta agey "
            "heavy oily khabar ba alcohol khaben nah."
        ),
        "advisory_script_bn": (
            "{test_name}-এর জন্য আপনাকে ১০ থেকে ১২ ঘণ্টা খালি পেটে থাকতে "
            "হবে। শুধু জল খেতে পারেন। টেস্ট-এর ২৪ ঘণ্টা আগে হেভি অয়েলি "
            "খাবার বা অ্যালকোহল খাবেন না।"
        ),
    },
    "Thyroid Profile (T3 T4 TSH)": {
        "fasting_required": False,
        "fasting_hours": "0 hours",
        "water_allowance": "Normal water intake allowed",
        "medication_hold": "Hold morning thyroid medication until after blood collection",
        "timing_rule": "Early morning sample preferred",
        "advisory_script_en": (
            "For {test_name}, fasting is not mandatory, but you must "
            "hold your morning thyroid medicine until after the blood "
            "sample is collected."
        ),
        "advisory_script_hinglish": (
            "{test_name} ke liye fasting zaroori nahi hai, lekin apni "
            "subah ki thyroid ki tablet blood sample dene ke baad hi "
            "lein."
        ),
        "advisory_script_banglish": (
            "{test_name}-er jonno fasting lagbe nah, tobe shokaler "
            "thyroid-er bori-ta sample deowar por khaben."
        ),
        "advisory_script_bn": (
            "{test_name}-এর জন্য ফাস্টিং লাগবে না, তবে সকালের থাইরয়েডের "
            "বড়ি-টা স্যাম্পল দেওয়ার পর খাবেন।"
        ),
    },
    "Urine Routine Examination": {
        "fasting_required": False,
        "fasting_hours": "0 hours",
        "water_allowance": "Normal water intake allowed",
        "medication_hold": "Inform lab about ongoing antibiotic medication",
        "timing_rule": "First morning mid-stream specimen preferred",
        "advisory_script_en": (
            "For {test_name}, collect the mid-stream portion of your "
            "first morning urine in a sterile container without touching "
            "the inside of the container."
        ),
        "advisory_script_hinglish": (
            "{test_name} ke liye subah ka pehla urine ka mid-stream "
            "sample sterile container mein collect karein. Container ke "
            "andar touch na karein."
        ),
        "advisory_script_banglish": (
            "{test_name}-er jonno shokal-er prothom urine-er majher "
            "onsho-ta (mid-stream) sterile container-e collect korben. "
            "Container-er bhetor touch korben nah."
        ),
        "advisory_script_bn": (
            "{test_name}-এর জন্য সকাল-এর প্রথম ইউরিনের মাঝের অংশ-টা "
            "(mid-stream) স্টেরাইল কন্টেইনারে কালেক্ট করবেন। কন্টেইনারের "
            "ভেতর টাচ করবেন না।"
        ),
    },
    "Chest X-Ray (PA view)": {
        "fasting_required": False,
        "fasting_hours": "0 hours",
        "water_allowance": "Normal water intake allowed",
        "medication_hold": "No medication hold required",
        "timing_rule": "Can be conducted any time during operational hours",
        "advisory_script_en": (
            "For {test_name}, no fasting is needed. Please wear loose "
            "clothing and remove metal items, necklaces, or innerwear "
            "with metallic hooks before the scan."
        ),
        "advisory_script_hinglish": (
            "{test_name} ke liye kisi fasting ki zaroorat nahi hai. "
            "Dheele kapde pehniye aur koi metal item, chain ya metal hook "
            "waale innerwear utar dein."
        ),
        "advisory_script_banglish": (
            "{test_name}-er jonno kono fasting lagbe nah. Halka "
            "jama-kapor pore ashben abong kono metal item ba chain khule "
            "rakhben."
        ),
        "advisory_script_bn": (
            "{test_name}-এর জন্য কোনো ফাস্টিং লাগবে না। হালকা জামা-কাপড় "
            "পরে আসবেন এবং কোনো মেটাল আইটেম বা চেইন খুলে রাখবেন।"
        ),
    },
    "USG Whole Abdomen": {
        "fasting_required": True,
        "fasting_hours": "6-8 hours",
        "water_allowance": "Drink 1-1.5 liters of plain water 1 hour prior to scan for a full bladder",
        "medication_hold": "Take regular vital medications with small sips of water",
        "timing_rule": "Fasting + Full bladder state required",
        "advisory_script_en": (
            "For {test_name}, fast for 6 to 8 hours. You need a full "
            "bladder, so drink 1 to 1.5 liters of water 1 hour before the "
            "scan and do not urinate until the scan is done."
        ),
        "advisory_script_hinglish": (
            "{test_name} ke liye 6 se 8 ghante fasting rakhein. Bladder "
            "full hona zaroori hai, isliye scan se 1 ghanta pehle 1-1.5 "
            "liter paani piyein aur peshab na rokein mat."
        ),
        "advisory_script_banglish": (
            "{test_name}-er jonno 6 theke 8 ghanta fasting lagbe. Apnar "
            "bladder full thakte hobe, tai test-er 1 ghanta agey 1-1.5 "
            "litre jol kheye urine chepe rakhben."
        ),
        "advisory_script_bn": (
            "{test_name}-এর জন্য ৬ থেকে ৮ ঘণ্টা ফাস্টিং লাগবে। আপনার "
            "ব্লাডার ফুল থাকতে হবে, তাই টেস্ট-এর ১ ঘণ্টা আগে ১-১.৫ লিটার "
            "জল খেয়ে ইউরিন চেপে রাখবেন।"
        ),
    },
}


# ============================================================
# SEED FUNCTION
# ============================================================

def seed():

    # ========================================================
    # EXISTING DATABASE RESET
    #
    # WHY:
    #
    # Every test run starts from the same known state.
    #
    # This is useful when you deliberately try to break the
    # voice agent and then want to reset everything.
    # ========================================================

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    db = SessionLocal()

    try:

        # ====================================================
        # SECTION 1
        # DEPARTMENTS
        #
        # Supports:
        #
        # English:
        # "Do you have cardiology?"
        #
        # Hinglish:
        # "Heart ka doctor hai?"
        #
        # Bengalish:
        # "Heart er doctor ache?"
        #
        # Bengali:
        # "হার্টের ডাক্তার আছে?"
        # ====================================================

        doctor_index = 0

        for dept_name, doctors in DEPARTMENTS.items():

            aliases = "|".join(
                DEPARTMENT_ALIASES.get(
                    dept_name,
                    []
                )
            )

            dept = Department(
                name=dept_name,
                aliases_bn=aliases,
            )

            db.add(dept)

            db.flush()

            # =================================================
            # SECTION 2
            # DOCTORS
            #
            # Supports doctor lookup in multiple languages.
            # =================================================

            for doc_name, qualifications in doctors:

                surname = doc_name.split()[-1]

                aliases = "|".join(
                    SURNAME_BN.get(
                        surname,
                        []
                    )
                )

                doctor = Doctor(
                    name=doc_name,
                    qualifications=qualifications,
                    aliases_bn=aliases,
                    department_id=dept.id,
                )

                db.add(doctor)

                db.flush()

                # =============================================
                # SECTION 3
                # DOCTOR SCHEDULE
                # =============================================

                template = SHIFT_TEMPLATES[
                    doctor_index % len(SHIFT_TEMPLATES)
                ]

                for weekday in template["days"]:

                    db.add(
                        DoctorSchedule(
                            doctor_id=doctor.id,
                            weekday=weekday,
                            start_time=template["start"],
                            end_time=template["end"],
                        )
                    )

                doctor_index += 1

        # ====================================================
        # SECTION 4
        # LAB TESTS
        #
        # Multilingual aliases allow the voice agent to match
        # different ways of saying the same test.
        #
        # Also stores:
        # - price
        # - sample type
        # - expected report time
        # ====================================================

        for (
            name,
            aliases,
            rate,
            sample,
            hours
        ) in LAB_TESTS:

            db.add(
                LabTest(
                    name=name,
                    aliases_bn="|".join(aliases),
                    rate_inr=rate,
                    sample_type=sample,
                    report_time_hours=hours,
                )
            )

        db.flush()

        # Built here (not just inside SECTION 8) so SECTION 6 can also
        # link health packages to real LabTest rows instead of the
        # non-existent `included_tests` field seed.py used to pass.
        lab_tests = {
            test.name: test
            for test in db.query(LabTest).all()
        }

        # ADDED BY SOURAV -- "Caller asks how to prepare for a test"
        # story. Applied as a second pass over the already-created rows
        # (rather than folded into the LAB_TESTS tuples above) because
        # only these 8 of the ~27 seeded tests have real advisory content
        # -- see LAB_TEST_ADVISORIES's own comment for the full reasoning.
        for test_name, advisory in LAB_TEST_ADVISORIES.items():
            test_row = lab_tests[test_name]
            for column, value in advisory.items():
                setattr(test_row, column, value)
        db.flush()

        # ====================================================
        # SECTION 5
        # CLINIC INFORMATION
        #
        # ADDED BY CHATGPT FOR SOURAV
        #
        # Prompt:
        # "add opening and closing time, address, directions"
        #
        # Created by ChatGPT for Sourav.
        #
        # Supports:
        #
        # English:
        # "What time do you open?"
        #
        # Hinglish:
        # "Clinic kab khulta hai?"
        #
        # Bengalish:
        # "Clinic koto tay khole?"
        #
        # Bengali:
        # "ক্লিনিক কখন খোলে?"
        #
        # IMPORTANT:
        # The actual voice response language should ideally be
        # handled by the agent/template layer.
        #
        # The DB stores the factual information.
        # ====================================================

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

            # models.py stores per-weekday hours rather than a single
            # opening_time/closing_time pair (which don't exist as
            # columns) -- Monday-Saturday 08:00-20:00, Sunday closed.
            monday_open="08:00", monday_close="20:00",
            tuesday_open="08:00", tuesday_close="20:00",
            wednesday_open="08:00", wednesday_close="20:00",
            thursday_open="08:00", thursday_close="20:00",
            friday_open="08:00", friday_close="20:00",
            saturday_open="08:00", saturday_close="20:00",
            sunday_open=None,
            sunday_close=None,
            sunday_closed=True,

            phone="03340001234",
        )

        db.add(clinic)

        db.flush()

        # ====================================================
        # SECTION 6
        # HEALTH PACKAGES
        #
        # ADDED BY CHATGPT FOR SOURAV
        #
        # Prompt:
        # "add health packages"
        #
        # Created by ChatGPT for Sourav.
        #
        # Supports package questions in English/Hinglish/
        # Bengalish/Bengali.
        # ====================================================

        # ADDED BY SOURAV -- "Caller asks about a health package" story.
        # Every other catalogue table in this file (LabTest, Doctor,
        # Department, a few lines below/above) already carries a
        # multilingual "|"-joined alias list so a spoken caller utterance
        # (English/Hinglish/Banglish/Bengali-script) can be matched against
        # it -- see e.g. LabTest's own "Multilingual aliases allow the
        # voice agent to match different ways of saying the same test"
        # comment further down this file. HealthPackage.aliases (models.py)
        # was left "" for every row here since this section was first
        # written -- a real, silent gap: a caller who didn't say a
        # package's exact English name (e.g. "ডায়াবেটিস প্যাকেজ", the
        # model's own docstring example) could never be matched. Fixed by
        # giving each package the same kind of alias list the other
        # catalogues already have, reusing this file's own established
        # per-package terms rather than inventing new ones.
        HEALTH_PACKAGES = [

            {
                "name": "Basic Health Checkup",

                "description": (
                    "Routine screening package "
                    "for general health."
                ),

                "price": 999,

                "aliases": (
                    "basic checkup|basic health package|basic package|"
                    "normal checkup|shadharon checkup|shadharon health package|"
                    "সাধারণ স্বাস্থ্য পরীক্ষা|বেসিক চেকআপ প্যাকেজ"
                ),

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

                "aliases": (
                    "diabetes checkup|diabetes package|diabetes screening|"
                    "sugar checkup|sugar package|diabetes er package|"
                    "ডায়াবেটিস প্যাকেজ|সুগার চেকআপ প্যাকেজ"
                ),

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

                "aliases": (
                    "full body checkup|full body package|complete health checkup|"
                    "full body checkup package|puro sharirer checkup|"
                    "ফুল বডি চেকআপ|সম্পূর্ণ স্বাস্থ্য পরীক্ষা"
                ),

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

                "aliases": (
                    "women's health package|women's wellness checkup|"
                    "ladies health package|ladies checkup package|"
                    "mohilader health package|women der package|"
                    "মহিলাদের স্বাস্থ্য প্যাকেজ|নারী স্বাস্থ্য পরীক্ষা"
                ),

                "tests": (
                    "Complete Blood Count (CBC), "
                    "Thyroid Profile (T3 T4 TSH), "
                    "Vitamin D (25-OH), "
                    "Pap Smear"
                ),
            },
        ]

        for package in HEALTH_PACKAGES:

            # `included_tests` is not a real column on HealthPackage --
            # the model links tests through a real HealthPackageTest
            # join table instead. `package["tests"]` is kept as the
            # human-readable comma-separated description string (still
            # useful to speak aloud) and additionally parsed below to
            # populate the real join rows, so packages are genuinely
            # linked to LabTest rows rather than orphaned.
            pkg = HealthPackage(
                name=package["name"],
                aliases=package["aliases"],
                description=package["description"],
                price_inr=package["price"],
            )
            db.add(pkg)
            db.flush()

            for test_name in package["tests"].split(","):
                test_name = test_name.strip()
                test = lab_tests.get(test_name)
                if test is None:
                    raise ValueError(
                        f"Health package {package['name']!r} references "
                        f"unknown lab test {test_name!r}"
                    )
                db.add(
                    HealthPackageTest(
                        package_id=pkg.id,
                        lab_test_id=test.id,
                    )
                )

        db.flush()

        # ====================================================
        # SECTION 7
        # PATIENTS
        #
        # UPDATED BY SOURAV -- "Lab Report Status & Secure Delivery"
        # combined story (previously two separate stories: "report
        # ready?" and "send my report"), per the attack-plan doc's
        # Section 5 "TEST DATA SCENARIOS" (Patients A-J).
        #
        # WHY THIS CHANGED FROM THE EARLIER VERSION OF THIS FILE:
        # The earlier 6-patient seed only covered the happy path plus
        # basic ready/not-ready/processing cases. This combined story
        # is explicitly about proving the agent FAILS SAFE on edge
        # cases (Section 23 of the plan), so the patient roster below
        # is deliberately built to cover every lettered scenario:
        #   Patient A - Arjun Sen        : happy path (READY, OTP, SENT)
        #   Patient B - Riya Das         : NOT_READY (no OTP ever issued)
        #   Patient C - Rahul Ghosh      : PROCESSING
        #   Patient D - Debashish Roy    : CANCELLED
        #   Patient E - Sohini Mukherjee : READY, but OTP expired/maxed
        #   Patient F - Mita Roy         : multiple reports (READY +
        #                                  NOT_READY + READY), forces
        #                                  clarification (Rule 13)
        #   Patient G - Priyanka Sengupta: no report at all -> NOT_FOUND
        #   Patient H - Rahul Das (x2)   : same-name collision, must
        #                                  not be merged (Rule 14)
        #   Patient I - Indrani Ghosh    : READY but delivery_enabled=False
        #   Patient J - Amit Banerjee    : previously-generated link now
        #                                  expired (old link must not be
        #                                  reusable)
        #
        # FLOWS INTO: clinic-api's forthcoming report-status/delivery
        # endpoints (Phase 2), which query Patient by phone/name to
        # resolve identity before ever touching a LabReport row -- see
        # Rule 14 (same-name patients must not be merged) and Rule 15
        # (wrong phone must not bypass identity).
        # ====================================================

        PATIENTS = [

            # Patient A -- happy path.
            {
                "name": "Arjun Sen",
                "phone": "9000000001",
                "email": "arjun.test@example.com",
                "verified": True,
            },

            # Patient B -- report not ready.
            {
                "name": "Riya Das",
                "phone": "9000000002",
                "email": "riya.test@example.com",
                "verified": True,
            },

            # Patient C -- report processing.
            {
                "name": "Rahul Ghosh",
                "phone": "9000000003",
                "email": "rahul.test@example.com",
                "verified": True,
            },

            # Patient D -- report cancelled. NEW for the combined story.
            {
                "name": "Debashish Roy",
                "phone": "9000000007",
                "email": "debashish.test@example.com",
                "verified": True,
            },

            # Patient E -- OTP attack surface (expired + max-attempts),
            # both against a genuinely READY report.
            {
                "name": "Sohini Mukherjee",
                "phone": "9000000006",
                "email": "sohini.test@example.com",
                "verified": True,
            },

            # Patient F -- multiple reports, forces clarification.
            {
                "name": "Mita Roy",
                "phone": "9000000004",
                "email": "mita.test@example.com",
                "verified": False,
            },

            # Patient G -- exists, but has zero reports. NEW.
            {
                "name": "Priyanka Sengupta",
                "phone": "9000000008",
                "email": "priyanka.test@example.com",
                "verified": True,
            },

            # Patient H -- same-name collision pair. NEW. Two distinct
            # patients, deliberately given the exact same name (per the
            # plan's own "Rahul Das / Rahul Das" example) but different
            # phones and different reports, so identity resolution MUST
            # key off something other than name alone (Rule 14).
            {
                "name": "Rahul Das",
                "phone": "9000000009",
                "email": "rahul.das.1.test@example.com",
                "verified": True,
            },
            {
                "name": "Rahul Das",
                "phone": "9000000010",
                "email": "rahul.das.2.test@example.com",
                "verified": True,
            },

            # Patient I -- READY report, delivery intentionally disabled
            # at the report level. NEW.
            {
                "name": "Indrani Ghosh",
                "phone": "9000000011",
                "email": "indrani.test@example.com",
                "verified": True,
            },

            # Patient J -- previously-generated signed link, now expired.
            {
                "name": "Amit Banerjee",
                "phone": "9000000005",
                "email": "amit.test@example.com",
                "verified": True,
            },
        ]

        # `email`/`phone_verified` are not real columns on Patient (see
        # the reconciliation note near the top of this file) -- kept
        # here, unused, as human-readable scenario documentation only.
        patient_objects = {}
        # Patient H has two rows sharing the same name, so `name` alone
        # cannot key patient_objects for them -- keyed by phone instead,
        # with a name-keyed convenience alias for the single-patient
        # letters (A-G, I, J) where the name is already unique.
        patient_objects_by_phone = {}

        for data in PATIENTS:

            patient = Patient(

                name=data["name"],

                phone=data["phone"],
            )

            db.add(patient)

            db.flush()

            patient_objects_by_phone[data["phone"]] = patient

            if data["name"] not in patient_objects:
                patient_objects[data["name"]] = patient
            # else: a same-name collision (Patient H) -- deliberately
            # left unresolved here to the *first* row under
            # patient_objects[name]; every reference below that needs
            # the second Rahul Das row uses patient_objects_by_phone
            # instead, exactly the disambiguation Rule 14 requires from
            # the live agent too.

        # ====================================================
        # SECTION 8
        # PATIENT REPORTS
        #
        # UPDATED BY SOURAV -- see SECTION 7's header for the full
        # Patient A-J mapping this supports. Statuses now include
        # CANCELLED (Patient D), which the earlier version of this file
        # never seeded, and every report explicitly sets
        # `delivery_enabled` (previously left at the model's default of
        # False for every row, which would have made even Patient A's
        # "happy path" delivery scenario fail DELIVERY_DISABLED).
        #
        # FLOWS INTO: RULE 3 (READY required before delivery), RULE 16
        # (delivery_enabled gate), and the REPORT_STATUS_MODEL outcomes
        # in Section 4 of the plan (READY / NOT_READY / PROCESSING /
        # CANCELLED / NOT_FOUND -- NOT_FOUND has no row at all, which is
        # why Patient G is seeded with none).
        # ====================================================

        now = datetime.now()

        # (lab_tests was already built above, right after SECTION 4,
        # so SECTION 6 could also use it -- no need to query again.)

        REPORT_DATA = [

            # Patient A -- happy path: CBC, READY, delivery enabled.
            {
                "patient_phone": "9000000001",
                "test": "Complete Blood Count (CBC)",
                "report_id": "RPT-10001",
                "status": "READY",
                "hours_ago": 8,
                "delivery_enabled": True,
            },

            # Patient B -- NOT_READY. Test changed to Vitamin D to match
            # the plan's Patient B exactly (was Lipid Profile before).
            {
                "patient_phone": "9000000002",
                "test": "Vitamin D (25-OH)",
                "report_id": "RPT-10002",
                "status": "NOT_READY",
                "hours_ago": 1,
                "delivery_enabled": False,
            },

            # Patient C -- PROCESSING. Test changed to the standalone
            # "TSH" lab test (distinct from the "Thyroid Profile (T3 T4
            # TSH)" panel) to match the plan's Patient C literally.
            {
                "patient_phone": "9000000003",
                "test": "TSH",
                "report_id": "RPT-10003",
                "status": "PROCESSING",
                "hours_ago": 2,
                "delivery_enabled": False,
            },

            # Patient D -- CANCELLED. NEW status value for this file.
            {
                "patient_phone": "9000000007",
                "test": "Uric Acid",
                "report_id": "RPT-10009",
                "status": "CANCELLED",
                "hours_ago": 20,
                "delivery_enabled": False,
            },

            # Patient F -- three reports on one patient, forcing
            # clarification (Rule 13) rather than a random pick.
            {
                "patient_phone": "9000000004",
                "test": "Complete Blood Count (CBC)",
                "report_id": "RPT-10004",
                "status": "READY",
                "hours_ago": 10,
                "delivery_enabled": True,
            },
            {
                "patient_phone": "9000000004",
                "test": "Vitamin D (25-OH)",
                "report_id": "RPT-10005",
                "status": "NOT_READY",
                "hours_ago": 3,
                "delivery_enabled": False,
            },
            {
                "patient_phone": "9000000004",
                "test": "TSH",
                "report_id": "RPT-10010",
                "status": "READY",
                "hours_ago": 15,
                "delivery_enabled": True,
            },

            # Patient H -- same-name pair, each with their OWN report,
            # to prove the wrong one is never picked (Rule 14).
            {
                "patient_phone": "9000000009",
                "test": "Complete Blood Count (CBC)",
                "report_id": "RPT-10011",
                "status": "READY",
                "hours_ago": 6,
                "delivery_enabled": True,
            },
            {
                "patient_phone": "9000000010",
                "test": "Lipid Profile",
                "report_id": "RPT-10012",
                "status": "READY",
                "hours_ago": 6,
                "delivery_enabled": True,
            },

            # Patient I -- READY, but delivery is explicitly disabled at
            # the report level (Rule 16 / DELIVERY_DISABLED).
            {
                "patient_phone": "9000000011",
                "test": "Kidney Function Test (KFT)",
                "report_id": "RPT-10013",
                "status": "READY",
                "hours_ago": 5,
                "delivery_enabled": False,
            },

            # Patient J -- READY, delivery-enabled, but its signed link
            # (seeded in SECTION 10 below) is already expired.
            {
                "patient_phone": "9000000005",
                "test": "Blood Sugar Fasting",
                "report_id": "RPT-10006",
                "status": "READY",
                "hours_ago": 12,
                "delivery_enabled": True,
            },
            {
                "patient_phone": "9000000005",
                "test": "HbA1c",
                "report_id": "RPT-10007",
                "status": "NOT_READY",
                "hours_ago": 1,
                "delivery_enabled": False,
            },

            # Patient E -- READY, but every OTP seeded against it is bad
            # (expired, or already at the attempt limit).
            {
                "patient_phone": "9000000006",
                "test": "Liver Function Test (LFT)",
                "report_id": "RPT-10008",
                "status": "READY",
                "hours_ago": 24,
                "delivery_enabled": True,
            },

            # Patient G (Priyanka Sengupta) deliberately gets NO report
            # row at all -- the live REPORT_NOT_FOUND path (Rule 1) has
            # to come from an honest "no matching row", not a
            # fabricated one.
        ]

        report_objects = {}

        for data in REPORT_DATA:

            generated = (
                now
                - timedelta(
                    hours=data["hours_ago"]
                )
            )

            ready_at = (
                generated + timedelta(hours=1)
                if data["status"] == "READY"
                else None
            )

            test_row = lab_tests[data["test"]]
            patient = patient_objects_by_phone[data["patient_phone"]]

            report = LabReport(

                report_number=data["report_id"],

                patient_id=patient.id,

                lab_test_id=test_row.id,

                collected_at=generated,

                expected_ready_at=(
                    generated
                    + timedelta(hours=test_row.report_time_hours)
                ),

                status=data["status"],

                ready_at=ready_at,

                delivery_enabled=data["delivery_enabled"],
            )

            db.add(report)

            db.flush()

            report_objects[
                data["report_id"]
            ] = report

        # ====================================================
        # SECTION 9
        # OTP VERIFICATION
        #
        # UPDATED BY SOURAV -- Riya Das's (Patient B) old OTP row was
        # removed entirely: her report is NOT_READY, and RULE 2 / RULE
        # 4 say a report that never became READY must never have had an
        # OTP issued for it in the first place -- seeding one would
        # have modeled a state the business rules forbid. Sohini
        # Mukherjee (Patient E) now carries TWO OTP rows against the
        # SAME report, so both "expired" and "max attempts reached"
        # (RULE 6, RULE 8) are exercised without inventing a second
        # patient for it.
        #
        # FLOWS INTO: the forthcoming OTP-verification endpoint (Phase
        # 2) and RULE 5 (an OTP must belong to the correct
        # patient/report/phone) -- every row below is tied to exactly
        # one report_id + patient_id + phone, so ATTACK 5/6 (cross-
        # report / cross-patient OTP reuse) has real seeded data to
        # attack.
        # ====================================================

        # UPDATED BY SOURAV -- "otp will not be hardcoded". Every row
        # below used to carry its own fixed literal "otp" value
        # ("482913" / "615204" / "903217" / "731846"). None of these
        # values matter to what each row is FOR (a valid/expired/maxed/
        # used OTP) -- only its `status` does -- so each dict's actual
        # code is now generated fresh at seed time by
        # models.generate_otp_code() (see the loop below), and the old
        # per-row comments that used to cite a specific sibling row's
        # code by number now refer to it descriptively instead. A test
        # (or a human) that needs a seeded row's real code reads it back
        # from the database, the same way tests/test_clinic_api_reports.py
        # already does for every OTHER code path in this file (freshly
        # minted rows included) -- see that file's own real_clinic_api
        # fixture.
        OTP_DATA = [

            # Patient A -- valid, happy path.
            {
                "patient": "Arjun Sen",
                "report": "RPT-10001",
                "status": "VALID",
                "expires_minutes": 10,
                "attempts": 0,
            },

            # Patient E -- expired OTP. A REAL narrative, not just two
            # independent rows: this one was issued first and expired
            # unused; the caller then requested again, got the row just
            # below, and burned through its attempts.
            #
            # UPDATED BY SOURAV -- added "created_minutes_ago" (below,
            # both rows). Both rows used to share the exact same
            # `created_at` (the loop's one shared `now`), which made
            # clinic-api/main.py's "most recent OTP row for this
            # (report, patient)" query (`order_by(created_at.desc())`,
            # used by both request_report_delivery and verify_report_otp)
            # non-deterministic on a tie -- verifying against the MAXED
            # row below was observed to resolve this STALE row instead
            # and return OTP_EXPIRED rather than OTP_MAX_ATTEMPTS,
            # flakily, depending on SQLite's undefined tie-break order.
            # Caught by tests/test_clinic_api_reports.py::TestOtpVerify::
            # test_max_attempts_already_reached. Giving the two rows
            # distinct, ordered timestamps (this one older) makes the row
            # below unambiguously "the current outstanding OTP" --
            # matching the real narrative above, and matching how
            # request_report_delivery's own "most recent" reuse logic is
            # meant to work. main.py's queries also now sort by `id` as a
            # tie-breaker (defense in depth for any other same-timestamp
            # case).
            {
                "patient": "Sohini Mukherjee",
                "report": "RPT-10008",
                "status": "EXPIRED",
                "expires_minutes": -10,
                "attempts": 0,
                "created_minutes_ago": 30,
            },

            # Patient E -- SECOND row, same report: already at the
            # attempt limit (max_attempts defaults to 3 on the model).
            # The row verify_report_otp actually checks (most recently
            # created) -- see the comment on the row above.
            {
                "patient": "Sohini Mukherjee",
                "report": "RPT-10008",
                "status": "MAXED",
                "expires_minutes": 10,
                "attempts": 3,
                "created_minutes_ago": 0,
            },

            # Patient J -- used OTP (should never be accepted again).
            {
                "patient": "Amit Banerjee",
                "report": "RPT-10006",
                "status": "USED",
                "expires_minutes": 10,
                "attempts": 1,
            },
        ]

        # models.py's ReportOTP tracks OTP state with `used` +
        # `verified_at` + `attempt_count`/`max_attempts`, not a single
        # `status`/`attempts` string pair (and it also requires
        # `report_id`, `phone` and `created_at`, none of which the
        # original ChatGPT-authored seed.py ever set -- see the
        # reconciliation note near the top of this file). Mapped as:
        #   VALID -> used=False, verified_at=None
        #   EXPIRED -> used=False, verified_at=None (expires_at already
        #              in the past via a negative expires_minutes)
        #   MAXED -> used=False, verified_at=None, attempt_count==max_attempts
        #   USED -> used=True, verified_at=now
        _OTP_USED = {"VALID": False, "EXPIRED": False, "MAXED": False, "USED": True}

        for data in OTP_DATA:

            report = report_objects[data["report"]]
            patient = patient_objects[data["patient"]]
            used = _OTP_USED[data["status"]]

            otp = ReportOTP(

                report_id=report.id,

                patient_id=patient.id,

                phone=patient.phone,

                # UPDATED BY SOURAV -- "otp will not be hardcoded": a
                # real random code per row, not a fixed literal (see
                # OTP_DATA's own comment above, and generate_otp_code()'s
                # docstring in models.py).
                otp_code=generate_otp_code(),

                # UPDATED BY SOURAV -- "created_minutes_ago" (default 0,
                # only Patient E's two rows set it non-zero) keeps rows on
                # the same report from sharing one identical timestamp.
                # See the OTP_DATA comment on RPT-10008's rows above for
                # why that mattered.
                created_at=now - timedelta(minutes=data.get("created_minutes_ago", 0)),

                expires_at=(
                    now
                    + timedelta(
                        minutes=data["expires_minutes"]
                    )
                ),

                verified_at=now if used else None,

                used=used,

                attempt_count=data["attempts"],
            )

            db.add(otp)

        db.flush()

        # ====================================================
        # SECTION 10
        # REPORT DELIVERY
        #
        # UPDATED BY SOURAV -- field names reconciled against the real
        # ReportDelivery model (see the reconciliation note near the
        # top of this file: `verification_method` -> `verification_status`,
        # `signed_token` -> `signed_link_token`, `link_expires_at` ->
        # `signed_link_expires_at`, plus `created_at` which the
        # original ChatGPT-authored seed.py never set). There is no
        # separate ReportDeliveryAudit table in models.py, so the old
        # SECTION 11 "delivery audit trail" rows are folded into each
        # ReportDelivery row's own `verification_status`,
        # `failure_reason` and `audit_note` fields -- including Riya
        # Das's (Patient B) and Debashish Roy's (Patient D) REJECTED
        # requests, which now get their own ReportDelivery row instead
        # of an audit-only entry with no delivery record at all.
        #
        # FLOWS INTO: RULE 11 (links must expire), RULE 17 (delivery
        # failure must be truthful), and ATTACK 23-28 (link/token
        # attacks) -- Patient J's row below is the "previously
        # generated, now expired" link those attacks target.
        # ====================================================

        _DELIVERY_STATE = {
            "SUCCESS": ("VERIFIED", "SENT"),
            "PENDING": ("OTP_REQUIRED", "PENDING"),
            "FAILED": ("EXPIRED", "FAILED"),
        }

        DELIVERY_DATA = [

            # Patient A -- successful delivery.
            {
                "patient": "Arjun Sen",
                "report": "RPT-10001",
                "recipient": "arjun.test@example.com",
                "verification": "OTP",
                "token": "SIGNED-ARJUN-10001",
                "expires_minutes": 15,
                "status": "SUCCESS",
            },

            # Patient E -- still pending: both seeded OTPs are bad, so
            # this request never resolves to VERIFIED.
            {
                "patient": "Sohini Mukherjee",
                "report": "RPT-10008",
                "recipient": "sohini.test@example.com",
                "verification": "OTP",
                "token": "SIGNED-SOHINI-10008",
                "expires_minutes": 15,
                "status": "PENDING",
            },

            # Patient J -- link already expired before delivery
            # completed. This is the "previously generated, now expired
            # link" ATTACK 23/24 targets.
            {
                "patient": "Amit Banerjee",
                "report": "RPT-10006",
                "recipient": "amit.test@example.com",
                "verification": "OTP",
                "token": "SIGNED-AMIT-10006",
                "expires_minutes": -10,
                "status": "FAILED",
            },
        ]

        _AUDIT_NOTE_BY_REPORT = {
            "RPT-10001": "OTP verified; report delivered successfully.",
            "RPT-10006": "Signed link expired before delivery could complete.",
        }

        for data in DELIVERY_DATA:

            report = report_objects[data["report"]]
            verification_status, delivery_status = _DELIVERY_STATE[data["status"]]
            sent_time = now if data["status"] == "SUCCESS" else None
            failed_time = now if data["status"] == "FAILED" else None

            delivery = ReportDelivery(

                report_id=report.id,

                patient_id=patient_objects[data["patient"]].id,

                recipient=data["recipient"],

                delivery_channel="EMAIL",

                verification_status=verification_status,

                delivery_status=delivery_status,

                signed_link_token=data["token"],

                signed_link_expires_at=(
                    now
                    + timedelta(
                        minutes=data["expires_minutes"]
                    )
                ),

                created_at=now,

                verified_at=sent_time,

                sent_at=sent_time,

                failed_at=failed_time,

                failure_reason=(
                    "LINK_EXPIRED" if data["status"] == "FAILED" else None
                ),

                audit_note=_AUDIT_NOTE_BY_REPORT.get(
                    data["report"],
                    f"Verification path: {data['verification']}.",
                ),
            )

            db.add(delivery)

        db.flush()

        # Patient B -- delivery was requested while the report was
        # still NOT_READY, so it was rejected before any OTP was ever
        # sent (RULE 2: NOT_READY means no delivery). No separate audit
        # table exists in models.py (see SECTION 10's header), so this
        # rejected request is its own ReportDelivery row instead of
        # being silently dropped.
        rejected_report = report_objects["RPT-10002"]
        db.add(
            ReportDelivery(
                report_id=rejected_report.id,
                patient_id=patient_objects["Riya Das"].id,
                recipient="riya.test@example.com",
                delivery_channel="EMAIL",
                verification_status="FAILED",
                delivery_status="FAILED",
                created_at=now,
                failed_at=now,
                failure_reason="REPORT_NOT_READY",
                audit_note=(
                    "Delivery requested for RPT-10002 while status was "
                    "NOT_READY; request rejected before any OTP was sent."
                ),
            )
        )

        # Patient D -- delivery was requested against a CANCELLED
        # report. NEW failure_reason value (REPORT_CANCELLED) added by
        # Sourav for this combined story, alongside the model's own
        # documented examples (WRONG_OTP / OTP_EXPIRED / LINK_EXPIRED /
        # DELIVERY_PROVIDER_FAILURE / PHONE_MISMATCH).
        cancelled_report = report_objects["RPT-10009"]
        db.add(
            ReportDelivery(
                report_id=cancelled_report.id,
                patient_id=patient_objects["Debashish Roy"].id,
                recipient="debashish.test@example.com",
                delivery_channel="EMAIL",
                verification_status="FAILED",
                delivery_status="FAILED",
                created_at=now,
                failed_at=now,
                failure_reason="REPORT_CANCELLED",
                audit_note=(
                    "Delivery requested for RPT-10009 while status was "
                    "CANCELLED; request rejected before any OTP was sent."
                ),
            )
        )

        _EXTRA_DELIVERY_ROWS = 2  # Riya (Patient B) + Debashish (Patient D)

        # ====================================================
        # FINAL COMMIT
        # ====================================================

        db.commit()

        print(
            "\n"
            "============================================\n"
            " DATABASE SEEDED SUCCESSFULLY\n"
            "============================================\n"
            f"Departments       : {len(DEPARTMENTS)}\n"
            f"Doctors           : {doctor_index}\n"
            f"Lab Tests         : {len(LAB_TESTS)}\n"
            f"Health Packages   : {len(HEALTH_PACKAGES)}\n"
            f"Patients          : {len(PATIENTS)}\n"
            f"Reports           : {len(REPORT_DATA)}\n"
            f"OTP Records       : {len(OTP_DATA)}\n"
            f"Delivery Records  : {len(DELIVERY_DATA) + _EXTRA_DELIVERY_ROWS}\n"
            "  (the extra rows are Riya Das's and Debashish Roy's\n"
            "  rejected pre-delivery requests, folded into\n"
            "  ReportDelivery -- no separate audit table exists in\n"
            "  models.py; see SECTION 10's header)\n"
            "Languages         : English / Hinglish / "
            "Bengalish / Bengali\n"
            "============================================\n"
        )

    finally:

        db.close()


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    seed()