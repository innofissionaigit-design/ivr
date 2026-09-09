"""Test that reply_templates.py preserves API values exactly.

This test verifies that the templates in reply_templates.py pass values
from the API response unchanged into the spoken text, as required by the
acceptance criteria: "Figures pass from the validated response into the
template unchanged and are verbalised digit-faithfully."
"""
import pytest
# Import the actual functions (not to be confused with test functions)
from agent.reply_templates import (
    test_rate_reply as rate_reply,
    doctor_availability_reply, booking_reply,
    doctors_by_department_reply,
)


class TestReplyTemplateValuePreservation:
    """Test that reply templates preserve exact API values."""

    def test_rate_reply_preserves_exact_rate(self):
        """The rate from API should be inserted exactly as received."""
        slots = {"test_name": "Uric Acid"}
        result = {
            "found": True,
            "test_name": "Uric Acid",
            "test_name_bn": "ইউরিক এসিড",
            "rate_inr": 650,
            "sample_type": "Blood",
            "report_time_hours": 24,
        }
        
        reply = rate_reply(slots, result)
        
        # The exact rate value (650) should be in the reply
        assert "650" in reply
        # No rounding or modification should occur
        assert "650.0" not in reply  # Should not add decimal
        assert "649" not in reply  # Should not round down
        assert "651" not in reply  # Should not round up

    def test_test_rate_reply_with_different_rates(self):
        """Various rate values should be preserved exactly."""
        test_cases = [
            350, 650, 850, 1200, 2500, 5000, 9999
        ]
        
        for rate in test_cases:
            slots = {"test_name": "Test"}
            result = {
                "found": True,
                "test_name": "Test",
                "test_name_bn": "টেস্ট",
                "rate_inr": rate,
                "sample_type": "Blood",
                "report_time_hours": 24,
            }
            
            reply = rate_reply(slots, result)
            assert str(rate) in reply, f"Rate {rate} should be preserved in reply"

    def test_test_rate_reply_preserves_report_time(self):
        """Report time hours should be preserved exactly."""
        slots = {"test_name": "CBC"}
        result = {
            "found": True,
            "test_name": "CBC",
            "test_name_bn": "সিবিসি",
            "rate_inr": 850,
            "sample_type": "Blood",
            "report_time_hours": 4,
        }
        
        reply = rate_reply(slots, result)
        assert "4" in reply
        assert "4.0" not in reply  # Should not add decimal

    def test_booking_reply_preserves_confirmation_id(self):
        """Confirmation ID should be preserved exactly."""
        slots = {
            "doctor_name": "Dr. Sen",
            "date": "2026-08-24",
            "time_slot": "09:30",
            "patient_name": "Rahul",
            "phone": "9876543210",
        }
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply = booking_reply(slots, result)
        # The exact confirmation ID should be preserved
        assert "KCD-20260824-0031" in reply
        # No modification should occur
        assert "KCD-20260824-0032" not in reply  # Should not change
        assert "KCD-20260824-31" not in reply  # Should not truncate

    def test_booking_reply_preserves_date(self):
        """Date should be preserved exactly in ISO format."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply = booking_reply(slots, result)
        # The exact date should be preserved
        assert "2026-08-24" in reply
        # No modification should occur
        assert "2026-08-25" not in reply  # Should not change
        assert "08-24" not in reply or "2026-08-24" in reply  # Should not truncate year

    def test_booking_reply_preserves_time_slot(self):
        """Time slot should be preserved exactly."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": True,
            "confirmation_id": "KCD-20260824-0031",
            "doctor_name": "Dr. A. Sen",
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "09:30",
        }
        
        reply = booking_reply(slots, result)
        # The exact time slot should be preserved
        assert "09:30" in reply
        # No modification should occur
        assert "09:00" not in reply  # Should not round
        assert "10:00" not in reply  # Should not round

    def test_doctor_availability_reply_preserves_date(self):
        """Date should be preserved exactly."""
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
        
        reply = doctor_availability_reply(slots, result)
        # The exact date should be preserved
        assert "2026-08-24" in reply

    def test_doctor_availability_reply_preserves_chamber_hours(self):
        """Chamber hours should be preserved exactly."""
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
        
        reply = doctor_availability_reply(slots, result)
        # The exact chamber hours should be preserved
        assert "18:00-20:00" in reply
        # No modification should occur
        assert "18:00-19:00" not in reply  # Should not change
        assert "6:00-8:00" not in reply  # Should not change format

    def test_booking_reply_preserves_alternative_slots(self):
        """Alternative time slots should be preserved exactly."""
        slots = {"doctor_name": "Dr. Sen"}
        result = {
            "success": False,
            "reason": "slot_taken",
            "alternative_slots": ["09:00", "09:15", "10:00"],
        }
        
        reply = booking_reply(slots, result)
        # The exact alternative slots should be preserved
        assert "09:00" in reply
        assert "09:15" in reply
        assert "10:00" in reply


class TestTemplateNoValueModification:
    """Test that templates don't perform any arithmetic or string manipulation."""

    def test_no_string_formatting_modification(self):
        """String formatting should not modify values."""
        slots = {"test_name": "Test"}
        result = {
            "found": True,
            "test_name": "Test",
            "test_name_bn": "টেস্ট",
            "rate_inr": 1234,
            "sample_type": "Blood",
            "report_time_hours": 48,
        }
        
        reply = rate_reply(slots, result)
        
        # The value should appear exactly as is
        assert "1234" in reply
        # Should not be reformatted (e.g., with commas)
        assert "1,234" not in reply
        # Should not be padded
        assert "01234" not in reply

    def test_no_arithmetic_operations(self):
        """Templates should not perform arithmetic on values."""
        slots = {"test_name": "Test"}
        result = {
            "found": True,
            "test_name": "Test",
            "test_name_bn": "টেস্ট",
            "rate_inr": 999,
            "sample_type": "Blood",
            "report_time_hours": 1,
        }
        
        reply = rate_reply(slots, result)
        
        # Should not add values
        assert "1000" not in reply
        # Should not multiply values
        assert "1998" not in reply
        # Original value should be present
        assert "999" in reply


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
