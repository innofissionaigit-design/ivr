"""Test suite for multilingual reply templates.

This test verifies that reply templates preserve exact values across
Bengali, English, and Hinglish languages while maintaining the
digit-faithful principle.
"""
import pytest
from agent.reply_templates import (
    test_rate_reply as rate_reply, doctor_availability_reply, booking_reply,
    doctors_by_department_reply, missing_slot_prompt,
)


class TestMultilingualRatePreservation:
    """Test that rate reply preserves exact values across languages."""

    def test_bengali_rate_preserves_value(self):
        """Bengali rate reply should preserve exact rate."""
        slots = {"test_name": "Uric Acid"}
        result = {
            "found": True,
            "test_name": "Uric Acid",
            "test_name_bn": "ইউরিক এসিড",
            "rate_inr": 650,
            "sample_type": "Blood",
            "report_time_hours": 24,
        }
        
        reply = rate_reply(slots, result, language="bengali")
        assert "650" in reply
        assert "টাকা" in reply

    def test_english_rate_preserves_value(self):
        """English rate reply should preserve exact rate."""
        slots = {"test_name": "Uric Acid"}
        result = {
            "found": True,
            "test_name": "Uric Acid",
            "test_name_bn": "ইউরিক এসিড",
            "rate_inr": 650,
            "sample_type": "Blood",
            "report_time_hours": 24,
        }
        
        reply = rate_reply(slots, result, language="english")
        assert "650" in reply
        assert "rupees" in reply

    def test_hinglish_rate_preserves_value(self):
        """Hinglish rate reply should preserve exact rate."""
        slots = {"test_name": "Uric Acid"}
        result = {
            "found": True,
            "test_name": "Uric Acid",
            "test_name_bn": "ইউরিক এসিড",
            "rate_inr": 650,
            "sample_type": "Blood",
            "report_time_hours": 24,
        }
        
        reply = rate_reply(slots, result, language="hinglish")
        assert "650" in reply
        assert "rupaye" in reply

    def test_all_languages_same_rate_value(self):
        """All languages should preserve the same exact rate value."""
        slots = {"test_name": "CBC"}
        result = {
            "found": True,
            "test_name": "CBC",
            "test_name_bn": "সিবিসি",
            "rate_inr": 850,
            "sample_type": "Blood",
            "report_time_hours": 4,
        }
        
        reply_bn = rate_reply(slots, result, language="bengali")
        reply_en = rate_reply(slots, result, language="english")
        reply_hi = rate_reply(slots, result, language="hinglish")
        
        # All should contain the exact rate value
        assert "850" in reply_bn
        assert "850" in reply_en
        assert "850" in reply_hi


class TestMultilingualBookingReply:
    """Test that booking reply preserves exact values across languages."""

    def test_bengali_booking_preserves_values(self):
        """Bengali booking reply should preserve exact values."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply = booking_reply(slots, result, language="bengali")
        assert "KCD-20260824-0031" in reply
        assert "2026-08-24" in reply
        assert "09:30" in reply

    def test_english_booking_preserves_values(self):
        """English booking reply should preserve exact values."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply = booking_reply(slots, result, language="english")
        assert "KCD-20260824-0031" in reply
        assert "2026-08-24" in reply
        assert "09:30" in reply

    def test_hinglish_booking_preserves_values(self):
        """Hinglish booking reply should preserve exact values."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply = booking_reply(slots, result, language="hinglish")
        assert "KCD-20260824-0031" in reply
        assert "2026-08-24" in reply
        assert "09:30" in reply


class TestMultilingualDoctorAvailability:
    """Test that doctor availability reply preserves exact values across languages."""

    def test_bengali_availability_preserves_values(self):
        """Bengali availability reply should preserve exact values."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "found": True,
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "available": True,
            "chamber_hours": "18:00-20:00",
            "next_available_date": None,
        }
        
        reply = doctor_availability_reply(slots, result, language="bengali")
        assert "2026-08-24" in reply
        assert "18:00-20:00" in reply

    def test_english_availability_preserves_values(self):
        """English availability reply should preserve exact values."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "found": True,
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "available": True,
            "chamber_hours": "18:00-20:00",
            "next_available_date": None,
        }
        
        reply = doctor_availability_reply(slots, result, language="english")
        assert "2026-08-24" in reply
        assert "18:00-20:00" in reply

    def test_hinglish_availability_preserves_values(self):
        """Hinglish availability reply should preserve exact values."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "found": True,
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "available": True,
            "chamber_hours": "18:00-20:00",
            "next_available_date": None,
        }
        
        reply = doctor_availability_reply(slots, result, language="hinglish")
        assert "2026-08-24" in reply
        assert "18:00-20:00" in reply


class TestMultilingualMissingSlotPrompts:
    """Test that missing slot prompts work across languages."""

    def test_bengali_missing_slot_prompt(self):
        """Bengali missing slot prompt should be in Bengali."""
        prompt = missing_slot_prompt("test_rate", "test_name", language="bengali")
        assert "টেস্ট" in prompt or "বলবেন" in prompt

    def test_english_missing_slot_prompt(self):
        """English missing slot prompt should be in English."""
        prompt = missing_slot_prompt("test_rate", "test_name", language="english")
        assert "test" in prompt.lower() or "which" in prompt.lower()

    def test_hinglish_missing_slot_prompt(self):
        """Hinglish missing slot prompt should be in Hinglish."""
        prompt = missing_slot_prompt("test_rate", "test_name", language="hinglish")
        assert "test" in prompt.lower() or "kaunsa" in prompt.lower()


class TestMultilingualDepartmentReply:
    """Test that department reply preserves exact values across languages."""

    def test_bengali_department_preserves_values(self):
        """Bengali department reply should preserve exact department name."""
        slots = {"department": "Orthopaedics"}
        result = {
            "found": True,
            "department": "Orthopaedics",
            "date": "2026-08-24",
            "doctors": [
                {"name": "Dr. A. Sen", "doctor_name_bn": "সেন"},
            ],
        }
        
        reply = doctors_by_department_reply(slots, result, language="bengali")
        assert "Orthopaedics" in reply or "অর্থোপেডিক্স" in reply

    def test_english_department_preserves_values(self):
        """English department reply should preserve exact department name."""
        slots = {"department": "Orthopaedics"}
        result = {
            "found": True,
            "department": "Orthopaedics",
            "date": "2026-08-24",
            "doctors": [
                {"name": "Dr. A. Sen", "doctor_name_bn": "সেন"},
            ],
        }
        
        reply = doctors_by_department_reply(slots, result, language="english")
        assert "Orthopaedics" in reply

    def test_hinglish_department_preserves_values(self):
        """Hinglish department reply should preserve exact department name."""
        slots = {"department": "Orthopaedics"}
        result = {
            "found": True,
            "department": "Orthopaedics",
            "date": "2026-08-24",
            "doctors": [
                {"name": "Dr. A. Sen", "doctor_name_bn": "সেন"},
            ],
        }
        
        reply = doctors_by_department_reply(slots, result, language="hinglish")
        assert "Orthopaedics" in reply


class TestMultilingualValueConsistency:
    """Test that values remain consistent across all languages."""

    def test_confirmation_id_consistency(self):
        """Confirmation ID should be identical across all languages."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply_bn = booking_reply(slots, result, language="bengali")
        reply_en = booking_reply(slots, result, language="english")
        reply_hi = booking_reply(slots, result, language="hinglish")
        
        # All should have the exact same confirmation ID
        assert "KCD-20260824-0031" in reply_bn
        assert "KCD-20260824-0031" in reply_en
        assert "KCD-20260824-0031" in reply_hi

    def test_date_consistency(self):
        """Date should be identical across all languages."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply_bn = booking_reply(slots, result, language="bengali")
        reply_en = booking_reply(slots, result, language="english")
        reply_hi = booking_reply(slots, result, language="hinglish")
        
        # All should have the exact same date
        assert "2026-08-24" in reply_bn
        assert "2026-08-24" in reply_en
        assert "2026-08-24" in reply_hi

    def test_time_slot_consistency(self):
        """Time slot should be identical across all languages."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply_bn = booking_reply(slots, result, language="bengali")
        reply_en = booking_reply(slots, result, language="english")
        reply_hi = booking_reply(slots, result, language="hinglish")
        
        # All should have the exact same time slot
        assert "09:30" in reply_bn
        assert "09:30" in reply_en
        assert "09:30" in reply_hi


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
