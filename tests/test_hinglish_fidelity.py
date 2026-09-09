"""Test suite for Hinglish (Hindi-English mix) number verbalization.

This test verifies that Hinglish number verbalization is digit-faithful
and preserves exact values while sounding natural in Hinglish contexts.
"""
import pytest
from agent.bn_normalize import (
    number_to_english_words, number_to_hinglish_words,
    time_to_english_words, time_to_hinglish_words,
    date_to_english_words, date_to_hinglish_words,
    verbalize, detect_language,
)


class TestHinglishNumberVerbalization:
    """Test that Hinglish number verbalization preserves exact values."""

    def test_small_numbers_hindi_words(self):
        """Small numbers (1-10) should use Hindi words common in Hinglish."""
        assert number_to_hinglish_words(0) == "zero"
        assert number_to_hinglish_words(1) == "ek"
        assert number_to_hinglish_words(2) == "do"
        assert number_to_hinglish_words(3) == "teen"
        assert number_to_hinglish_words(4) == "chaar"
        assert number_to_hinglish_words(5) == "paanch"
        assert number_to_hinglish_words(6) == "chhe"
        assert number_to_hinglish_words(7) == "saat"
        assert number_to_hinglish_words(8) == "aath"
        assert number_to_hinglish_words(9) == "nau"
        assert number_to_hinglish_words(10) == "das"

    def test_tens_hindi_words(self):
        """Tens should use Hindi words common in Hinglish."""
        assert number_to_hinglish_words(20) == "bees"
        assert number_to_hinglish_words(30) == "tees"
        assert number_to_hinglish_words(40) == "chaalis"
        assert number_to_hinglish_words(50) == "pachaas"
        assert number_to_hinglish_words(60) == "saath"
        assert number_to_hinglish_words(70) == "sattar"
        assert number_to_hinglish_words(80) == "assi"
        assert number_to_hinglish_words(90) == "nabbe"
        assert number_to_hinglish_words(100) == "ek sau"

    def test_large_numbers_english_words(self):
        """Large numbers should use English words with Indian system."""
        assert "thousand" in number_to_hinglish_words(1000)  # Uses Hindi for small numbers
        assert "thousand" in number_to_hinglish_words(1500)
        assert "lakh" in number_to_hinglish_words(100000)  # May use Hindi for "one"
        assert "lakh" in number_to_hinglish_words(150000)
        assert "crore" in number_to_hinglish_words(10000000)

    def test_negative_numbers(self):
        """Negative numbers should preserve sign."""
        assert number_to_hinglish_words(-5) == "minus paanch"
        assert number_to_hinglish_words(-100) == "minus ek sau"

    def test_realistic_prices_hinglish(self):
        """Realistic prices should be preserved exactly in Hinglish."""
        assert "six hundred" in number_to_hinglish_words(650)
        assert "fifty" in number_to_hinglish_words(650)
        assert "thousand" in number_to_hinglish_words(1200)
        assert "two hundred" in number_to_hinglish_words(1200)
        assert "thousand" in number_to_hinglish_words(2500)


class TestEnglishNumberVerbalization:
    """Test that English number verbalization preserves exact values."""

    def test_single_digits(self):
        """Single digits should be exact English words."""
        assert number_to_english_words(0) == "zero"
        assert number_to_english_words(1) == "one"
        assert number_to_english_words(5) == "five"
        assert number_to_english_words(9) == "nine"

    def test_tens(self):
        """Tens should be exact English words."""
        assert number_to_english_words(10) == "ten"
        assert number_to_english_words(15) == "fifteen"
        assert number_to_english_words(20) == "twenty"
        assert number_to_english_words(99) == "ninety-nine"

    def test_hundreds(self):
        """Hundreds should preserve exact value."""
        assert number_to_english_words(100) == "one hundred"
        assert number_to_english_words(250) == "two hundred and fifty"
        assert number_to_english_words(999) == "nine hundred and ninety-nine"

    def test_indian_system_english(self):
        """Large numbers should use Indian numbering system."""
        assert number_to_english_words(1000) == "one thousand"
        assert number_to_english_words(100000) == "one lakh"
        assert number_to_english_words(10000000) == "one crore"


class TestHinglishTimeVerbalization:
    """Test that Hinglish time verbalization is natural and accurate."""

    def test_on_the_hour_hinglish(self):
        """Exact hours should use Hindi "baje"."""
        result = time_to_hinglish_words(9, 0)
        assert "baje" in result
        assert "morning" in result
        result = time_to_hinglish_words(14, 0)
        assert "baje" in result
        assert "afternoon" in result

    def test_with_minutes_hinglish(self):
        """Times with minutes should mix Hindi and English naturally."""
        assert "baje" in time_to_hinglish_words(9, 15)
        assert "baje" in time_to_hinglish_words(9, 30)
        assert "baje" in time_to_hinglish_words(9, 45)

    def test_time_periods_hinglish(self):
        """Time periods should be in English."""
        assert "morning" in time_to_hinglish_words(9, 0)
        assert "afternoon" in time_to_hinglish_words(14, 0)
        assert "evening" in time_to_hinglish_words(18, 0)
        assert "night" in time_to_hinglish_words(22, 0)


class TestEnglishTimeVerbalization:
    """Test that English time verbalization is accurate."""

    def test_on_the_hour_english(self):
        """Exact hours should use "o'clock"."""
        assert time_to_english_words(9, 0) == "nine o'clock AM"
        assert time_to_english_words(14, 0) == "two o'clock PM"

    def test_quarter_hour_english(self):
        """Quarter hours should use natural English."""
        assert time_to_english_words(9, 15) == "quarter past nine AM"
        assert time_to_english_words(9, 30) == "half past nine AM"
        assert time_to_english_words(9, 45) == "quarter to ten AM"

    def test_with_minutes_english(self):
        """Other minutes should be exact."""
        assert time_to_english_words(9, 10) == "nine ten AM"
        assert time_to_english_words(14, 25) == "two twenty-five PM"


class TestHinglishDateVerbalization:
    """Test that Hinglish date verbalization is natural and accurate."""

    def test_date_hinglish(self):
        """Dates should use transliterated Hindi month names."""
        result = date_to_hinglish_words(2026, 8, 24)
        assert "August" in result or "tarikh" in result
        # Year is converted to words in Hinglish
        assert "twenty-six" in result or "do" in result

    def test_invalid_month_hinglish(self):
        """Invalid months should fall back gracefully."""
        result = date_to_hinglish_words(2026, 13, 5)
        assert "tarikh" in result


class TestEnglishDateVerbalization:
    """Test that English date verbalization is accurate."""

    def test_date_english(self):
        """Dates should use standard English format."""
        assert date_to_english_words(2026, 8, 24) == "August twenty-four, two thousand twenty-six"
        assert date_to_english_words(2026, 9, 1) == "September one, two thousand twenty-six"

    def test_invalid_month_english(self):
        """Invalid months should fall back gracefully."""
        assert date_to_english_words(2026, 13, 5) == "five date"


class TestLanguageDetection:
    """Test automatic language detection for verbalization."""

    def test_detect_bengali(self):
        """Predominantly Bengali text should be detected as Bengali."""
        bengali_text = "এটি বাংলা টেক্সট"
        assert detect_language(bengali_text) == "bengali"

    def test_detect_english(self):
        """Predominantly English text should be detected as English."""
        english_text = "This is English text"
        assert detect_language(english_text) == "english"

    def test_detect_hinglish(self):
        """Mixed text should be detected as Hinglish."""
        hinglish_text = "This is mixed বাংলা text"
        assert detect_language(hinglish_text) == "hinglish"

    def test_detect_empty(self):
        """Empty text should default to Bengali."""
        assert detect_language("") == "bengali"


class TestVerbalizeWithLanguages:
    """Test the verbalize function with different language options."""

    def test_verbalize_bengali_default(self):
        """Default should be Bengali."""
        text = "Rate 650 rupees"
        result = verbalize(text)
        # Should use Bengali number words
        assert "ছয়শো" in result or "six" in result

    def test_verbalize_english(self):
        """Explicit English verbalization."""
        text = "Rate 650 rupees"
        result = verbalize(text, language="english")
        # Should use English number words (with "and" for hundreds)
        assert "six hundred" in result
        assert "fifty" in result

    def test_verbalize_hinglish(self):
        """Explicit Hinglish verbalization."""
        text = "Rate 650 rupees"
        result = verbalize(text, language="hinglish")
        # Should use Hinglish number words
        assert "six hundred" in result
        assert "fifty" in result

    def test_verbalize_date_english(self):
        """Date verbalization in English."""
        text = "Date 2026-08-24"
        result = verbalize(text, language="english")
        assert "August" in result
        assert "2026" in result or "two thousand" in result

    def test_verbalize_date_hinglish(self):
        """Date verbalization in Hinglish."""
        text = "Date 2026-08-24"
        result = verbalize(text, language="hinglish")
        assert "August" in result or "tarikh" in result

    def test_verbalize_time_english(self):
        """Time verbalization in English."""
        text = "Time 09:30"
        result = verbalize(text, language="english")
        assert "nine" in result or "thirty" in result

    def test_verbalize_time_hinglish(self):
        """Time verbalization in Hinglish."""
        text = "Time 09:30"
        result = verbalize(text, language="hinglish")
        assert "baje" in result or "nine" in result

    def test_verbalize_confirmation_id(self):
        """Confirmation ID should be spelled out regardless of language."""
        text = "Confirmation ID: KCD-4471"
        result_bengali = verbalize(text, language="bengali")
        result_english = verbalize(text, language="english")
        result_hinglish = verbalize(text, language="hinglish")
        
        # All should spell out the ID
        assert "KCD" not in result_bengali or "কে" in result_bengali
        assert "4471" not in result_english or "four" in result_english


class TestHinglishValuePreservation:
    """Test that Hinglish verbalization preserves exact values."""

    def test_amount_semantic_preservation_hinglish(self):
        """The spoken amount should represent the exact same value."""
        test_cases = [
            (650, "six hundred"),
            (650, "fifty"),
            (1200, "thousand"),
            (1200, "two hundred"),
            (850, "eight hundred"),
            (850, "fifty"),
        ]
        
        for numeric_value, expected_component in test_cases:
            result = number_to_hinglish_words(numeric_value)
            assert expected_component in result, (
                f"Numeric value {numeric_value} should contain {expected_component}, "
                f"got {result} instead"
            )

    def test_no_rounding_hinglish(self):
        """Hinglish verbalization should not round values."""
        result = number_to_hinglish_words(999)
        assert "nine hundred" in result
        assert "ninety-nine" in result or "ninety nine" in result
        assert "one thousand" not in result
        result = number_to_hinglish_words(1001)
        assert "thousand" in result
        assert "one" in result or "ek" in result  # Hinglish may use Hindi for small numbers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
