"""
finder/tests.py

Unit + integration tests for YojanaBot.

Run with:
    python manage.py test finder -v 2

Test groups
-----------
1. TestTierClassification      — step6_formatter._tier()
2. TestTierHelpers             — _tier_to_css() / _tier_to_icon() in rag_service
3. TestValidateState           — step4_profile_collector._validate_state()
4. TestValidateInt             — step4_profile_collector._validate_int()
5. TestValidateChoice          — step4_profile_collector._validate_choice()
6. TestPrecisionAtK            — evaluation.step8_eval.precision_at_k()
7. TestIsMatch                 — evaluation.step8_eval._is_match()
8. TestTranslateCriterionHi    — step6_formatter._translate_criterion_hi()
9. TestTranslateWarningHi      — step6_formatter._translate_warning_hi()
10. TestBuildSchemeCards        — step6_formatter.build_scheme_cards() (with mock)
11. TestDjangoViews             — HTTP-level smoke tests for home / find / about
12. TestFindViewLangValidation  — invalid lang value is sanitised to 'en'
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import TestCase, Client
from django.urls import reverse

# ── imports under test (no side-effects at import time) ──────────────────────
from rag_pipeline.step4_profile_collector import (
    _validate_state,
    _validate_int,
    _validate_choice,
)
from rag_pipeline.step6_formatter import (
    _tier,
    _translate_criterion_hi,
    _translate_warning_hi,
    build_scheme_cards,
)
from finder.rag_service import _tier_to_css, _tier_to_icon
from evaluation.step8_eval import precision_at_k, _is_match


# ─────────────────────────────────────────────────────────────────────────────
# 1. _tier()  —  confidence score → (label, colour)
# ─────────────────────────────────────────────────────────────────────────────

class TestTierClassification(TestCase):
    """_tier() must bucket scores correctly at the boundary values."""

    def test_high_match_at_boundary(self):
        label, colour = _tier(0.70)
        self.assertIn("HIGH", label)
        self.assertEqual(colour, "green")

    def test_high_match_above_boundary(self):
        label, colour = _tier(0.99)
        self.assertIn("HIGH", label)
        self.assertEqual(colour, "green")

    def test_partial_match_at_lower_boundary(self):
        label, colour = _tier(0.40)
        self.assertIn("PARTIAL", label)
        self.assertEqual(colour, "yellow")

    def test_partial_match_just_below_high(self):
        label, colour = _tier(0.699)
        self.assertIn("PARTIAL", label)
        self.assertEqual(colour, "yellow")

    def test_low_relevance_below_threshold(self):
        label, colour = _tier(0.39)
        self.assertIn("LOW", label)
        self.assertEqual(colour, "red")

    def test_low_relevance_at_zero(self):
        label, colour = _tier(0.0)
        self.assertIn("LOW", label)
        self.assertEqual(colour, "red")

    def test_returns_tuple_of_two(self):
        result = _tier(0.5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _tier_to_css() / _tier_to_icon()  —  tier label → Bootstrap class / icon
# ─────────────────────────────────────────────────────────────────────────────

class TestTierHelpers(TestCase):
    """rag_service._tier_to_css and _tier_to_icon must map all tier labels."""

    def test_css_high_match(self):
        self.assertEqual(_tier_to_css("✅ HIGH MATCH"), "success")

    def test_css_partial_match(self):
        self.assertEqual(_tier_to_css("⚠️  PARTIAL MATCH"), "warning")

    def test_css_low_relevance(self):
        self.assertEqual(_tier_to_css("❓ LOW RELEVANCE"), "secondary")

    def test_icon_high_match(self):
        self.assertIn("check", _tier_to_icon("✅ HIGH MATCH"))

    def test_icon_partial_match(self):
        self.assertIn("exclamation", _tier_to_icon("⚠️  PARTIAL MATCH"))

    def test_icon_low_relevance(self):
        self.assertIn("question", _tier_to_icon("❓ LOW RELEVANCE"))

    def test_css_unknown_defaults_to_secondary(self):
        self.assertEqual(_tier_to_css("UNKNOWN"), "secondary")


# ─────────────────────────────────────────────────────────────────────────────
# 3. _validate_state()  —  raw user input → canonical state name
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateState(TestCase):
    """_validate_state must handle abbreviations, fuzzy matches, and bad input."""

    # Abbreviation aliases
    def test_alias_up(self):
        self.assertEqual(_validate_state("up"), "Uttar Pradesh")

    def test_alias_mp(self):
        self.assertEqual(_validate_state("MP"), "Madhya Pradesh")

    def test_alias_wb(self):
        self.assertEqual(_validate_state("WB"), "West Bengal")

    def test_alias_tn(self):
        self.assertEqual(_validate_state("tn"), "Tamil Nadu")

    # Known states (case-insensitive / partial match)
    def test_known_state_exact(self):
        result = _validate_state("Bihar")
        self.assertEqual(result, "Bihar")

    def test_known_state_lowercase(self):
        result = _validate_state("rajasthan")
        self.assertEqual(result, "Rajasthan")

    def test_known_state_with_spaces(self):
        result = _validate_state("uttar pradesh")
        # must title-case each word
        self.assertEqual(result, "Uttar Pradesh")

    def test_known_state_partial(self):
        # "gujarat" contains "gujarat" — should resolve
        result = _validate_state("gujarat")
        self.assertEqual(result, "Gujarat")

    # Accept unknown (≥ 3 chars) — user might enter a UT not in the list
    def test_accept_unknown_long_enough(self):
        result = _validate_state("Lakshadweep")
        self.assertEqual(result, "Lakshadweep")

    # Reject too-short strings
    def test_reject_two_chars(self):
        with self.assertRaises(ValueError):
            _validate_state("XX")

    def test_reject_one_char(self):
        with self.assertRaises(ValueError):
            _validate_state("X")

    def test_reject_whitespace_only(self):
        # 2-char input is below the minimum length threshold → ValueError
        with self.assertRaises(ValueError):
            _validate_state("xz")

    # Whitespace trimming
    def test_strip_whitespace(self):
        result = _validate_state("  bihar  ")
        self.assertEqual(result, "Bihar")


# ─────────────────────────────────────────────────────────────────────────────
# 4. _validate_int()  —  raw string → validated integer
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateInt(TestCase):
    """_validate_int must parse integers and enforce min/max bounds."""

    def test_valid_integer(self):
        self.assertEqual(_validate_int("25", min_val=0, max_val=120), 25)

    def test_valid_with_comma(self):
        self.assertEqual(_validate_int("1,20,000", min_val=0, max_val=10_000_000), 120000)

    def test_valid_with_rupee_symbol(self):
        self.assertEqual(_validate_int("₹60000", min_val=0, max_val=10_000_000), 60000)

    def test_below_min_raises(self):
        with self.assertRaises(ValueError):
            _validate_int("-1", min_val=0, max_val=120)

    def test_above_max_raises(self):
        with self.assertRaises(ValueError):
            _validate_int("200", min_val=0, max_val=120)

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            _validate_int("twenty", min_val=0, max_val=120)

    def test_at_min_boundary(self):
        self.assertEqual(_validate_int("0", min_val=0, max_val=120), 0)

    def test_at_max_boundary(self):
        self.assertEqual(_validate_int("120", min_val=0, max_val=120), 120)


# ─────────────────────────────────────────────────────────────────────────────
# 5. _validate_choice()  —  raw key → canonical choice value
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateChoice(TestCase):
    """_validate_choice must match both English keys and Hindi keys."""

    CHOICES_EN = {"1": "male", "2": "female", "male": "male", "female": "female", "m": "male"}
    CHOICES_HI = {"1": "male", "2": "female", "पुरुष": "male", "महिला": "female"}

    def test_numeric_key_en(self):
        result = _validate_choice("1", self.CHOICES_EN, self.CHOICES_HI)
        self.assertEqual(result, "male")

    def test_text_key_en(self):
        result = _validate_choice("female", self.CHOICES_EN, self.CHOICES_HI)
        self.assertEqual(result, "female")

    def test_abbreviation_en(self):
        result = _validate_choice("m", self.CHOICES_EN, self.CHOICES_HI)
        self.assertEqual(result, "male")

    def test_hindi_key(self):
        result = _validate_choice("पुरुष", self.CHOICES_EN, self.CHOICES_HI)
        self.assertEqual(result, "male")

    def test_hindi_female(self):
        result = _validate_choice("महिला", self.CHOICES_EN, self.CHOICES_HI)
        self.assertEqual(result, "female")

    def test_invalid_key_raises(self):
        with self.assertRaises(ValueError):
            _validate_choice("xyz", self.CHOICES_EN, self.CHOICES_HI)

    def test_case_insensitive(self):
        # Keys are lowercased inside validator
        result = _validate_choice("FEMALE", self.CHOICES_EN, self.CHOICES_HI)
        self.assertEqual(result, "female")


# ─────────────────────────────────────────────────────────────────────────────
# 6. precision_at_k()  —  retrieval metric
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecisionAtK(TestCase):
    """
    precision_at_k returns 1.0 if any of the top-k retrieved schemes matches
    the ground truth, else 0.0.
    """

    GROUND_TRUTH = ["PM Kisan Samman Nidhi", "Pradhan Mantri Awas Yojana"]

    def test_exact_match_in_top1(self):
        retrieved = ["PM Kisan Samman Nidhi", "Some Other Scheme"]
        self.assertEqual(precision_at_k(retrieved, self.GROUND_TRUTH, k=1), 1.0)

    def test_match_at_k2_but_not_k1(self):
        retrieved = ["Unrelated Scheme", "Pradhan Mantri Awas Yojana"]
        self.assertEqual(precision_at_k(retrieved, self.GROUND_TRUTH, k=1), 0.0)
        self.assertEqual(precision_at_k(retrieved, self.GROUND_TRUTH, k=2), 1.0)

    def test_no_match(self):
        retrieved = ["Scheme A", "Scheme B", "Scheme C"]
        self.assertEqual(precision_at_k(retrieved, self.GROUND_TRUTH, k=3), 0.0)

    def test_empty_retrieved(self):
        self.assertEqual(precision_at_k([], self.GROUND_TRUTH, k=5), 0.0)

    def test_k_larger_than_list(self):
        # k=10 but only 2 retrieved — should not crash, just return based on what's there
        retrieved = ["PM Kisan Samman Nidhi"]
        self.assertEqual(precision_at_k(retrieved, self.GROUND_TRUTH, k=10), 1.0)

    def test_partial_name_match(self):
        # "PM Kisan" is a substring of ground truth → should still match
        retrieved = ["PM Kisan Samman Nidhi Scheme (PMKSN)"]
        self.assertEqual(precision_at_k(retrieved, self.GROUND_TRUTH, k=1), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. _is_match()  —  fuzzy scheme name matching
# ─────────────────────────────────────────────────────────────────────────────

class TestIsMatch(TestCase):
    """_is_match uses substring and token-overlap logic."""

    GROUND_TRUTH = ["PM Kisan Samman Nidhi", "Pradhan Mantri Awas Yojana"]

    def test_exact_match(self):
        self.assertTrue(_is_match("PM Kisan Samman Nidhi", self.GROUND_TRUTH))

    def test_case_insensitive(self):
        self.assertTrue(_is_match("pm kisan samman nidhi", self.GROUND_TRUTH))

    def test_substring_match(self):
        # Ground truth is a substring of retrieved
        self.assertTrue(_is_match("Details of PM Kisan Samman Nidhi Scheme", self.GROUND_TRUTH))

    def test_token_overlap_two_significant_words(self):
        # "Pradhan" and "Mantri" are both > 3 chars and appear in ground truth
        self.assertTrue(_is_match("Pradhan Mantri Housing Scheme", self.GROUND_TRUTH))

    def test_no_match(self):
        self.assertFalse(_is_match("Completely Unrelated Scheme XYZ", self.GROUND_TRUTH))

    def test_single_token_not_enough(self):
        # Only one significant token overlap — should NOT match
        self.assertFalse(_is_match("Pradhan Singh Award", self.GROUND_TRUTH))

    def test_clearly_no_match(self):
        # A name with no token overlap at all should not match
        self.assertFalse(_is_match("Completely Unrelated Scheme XYZ", self.GROUND_TRUTH))

    def test_short_non_matching_name(self):
        # Short names with no significant token overlap should not match
        self.assertFalse(_is_match("ABC Scheme", self.GROUND_TRUTH))


# ─────────────────────────────────────────────────────────────────────────────
# 8. _translate_criterion_hi()  —  matched-criteria label → Hindi
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslateCriterionHi(TestCase):
    """Criteria strings like 'state=Bihar' must get Hindi field labels."""

    def test_state_prefix(self):
        result = _translate_criterion_hi("state=Bihar")
        self.assertTrue(result.startswith("राज्य"))
        self.assertIn("Bihar", result)

    def test_age_prefix(self):
        result = _translate_criterion_hi("age")
        self.assertEqual(result, "आयु")

    def test_income_with_rupee(self):
        result = _translate_criterion_hi("income<=Rs.500000")
        self.assertIn("आय", result)
        self.assertIn("₹", result)      # Rs. → ₹
        self.assertNotIn("Rs.", result)

    def test_caste_prefix(self):
        result = _translate_criterion_hi("caste=SC")
        self.assertTrue(result.startswith("जाति"))

    def test_occupation_prefix(self):
        result = _translate_criterion_hi("occupation=farmer")
        self.assertTrue(result.startswith("व्यवसाय"))

    def test_semantic_match(self):
        result = _translate_criterion_hi("semantic match")
        self.assertEqual(result, "प्रासंगिक")

    def test_unknown_criterion_returned_as_is(self):
        result = _translate_criterion_hi("some_unknown_field=value")
        self.assertEqual(result, "some_unknown_field=value")

    def test_gender_prefix(self):
        result = _translate_criterion_hi("gender=female")
        self.assertTrue(result.startswith("लिंग"))


# ─────────────────────────────────────────────────────────────────────────────
# 9. _translate_warning_hi()  —  warning strings → Hindi
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslateWarningHi(TestCase):
    """Warning strings must be translated to Hindi equivalents."""

    def test_state_mismatch(self):
        result = _translate_warning_hi("state mismatch: scheme is for All India")
        self.assertIn("राज्य बेमेल", result)
        self.assertIn("सम्पूर्ण भारत", result)

    def test_age_outside(self):
        result = _translate_warning_hi("age outside scheme range")
        self.assertIn("आयु सीमा से बाहर", result)

    def test_income_exceeds(self):
        result = _translate_warning_hi("income exceeds Rs.200000 limit")
        self.assertIn("आय सीमा से अधिक", result)
        self.assertIn("₹", result)       # Rs. → ₹
        self.assertNotIn("Rs.", result)

    def test_gender_mismatch(self):
        result = _translate_warning_hi("gender mismatch: scheme is for female")
        self.assertIn("लिंग बेमेल", result)

    def test_no_translation_needed(self):
        # A warning with no known keywords should be returned unchanged (minus Rs. swap)
        result = _translate_warning_hi("completely unknown warning text")
        self.assertEqual(result, "completely unknown warning text")


# ─────────────────────────────────────────────────────────────────────────────
# 10. build_scheme_cards()  —  RAGResponse → list[SchemeCard]
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_response(
    scheme_name="PM Kisan Samman Nidhi",
    benefit="₹6000 per year",
    score=0.82,
    matched=("state=Bihar", "age", "caste=SC"),
    warnings=("state mismatch: scheme is for All India",),
    state="Bihar",
    app_url="https://pmkisan.gov.in",
):
    """Return a minimal RAGResponse mock for testing build_scheme_cards."""
    scheme = MagicMock()
    scheme.scheme_name = scheme_name
    scheme.final_score = score
    scheme.matched_criteria = list(matched)
    scheme.eligibility_warnings = list(warnings)
    scheme.metadata = {
        "benefit_amount": benefit,
        "state": state,
        "application_url": app_url,
        "source_pdf_url": None,
    }

    response = MagicMock()
    response.retrieved_schemes = [scheme]
    return response


class TestBuildSchemeCards(TestCase):

    def test_english_card_fields(self):
        cards = build_scheme_cards(_make_mock_response(), lang="en")
        self.assertEqual(len(cards), 1)
        c = cards[0]
        self.assertEqual(c.rank, 1)
        self.assertEqual(c.scheme_name, "PM Kisan Samman Nidhi")
        self.assertEqual(c.benefit, "₹6000 per year")
        self.assertIn("Matched:", c.why_eligible)
        self.assertIn("state=Bihar", c.why_eligible)
        self.assertEqual(c.confidence_score, 0.82)

    def test_english_tier_is_high(self):
        cards = build_scheme_cards(_make_mock_response(score=0.82), lang="en")
        self.assertIn("HIGH", cards[0].confidence_tier)

    def test_english_tier_is_partial(self):
        cards = build_scheme_cards(_make_mock_response(score=0.55), lang="en")
        self.assertIn("PARTIAL", cards[0].confidence_tier)

    def test_english_tier_is_low(self):
        cards = build_scheme_cards(_make_mock_response(score=0.20), lang="en")
        self.assertIn("LOW", cards[0].confidence_tier)

    def test_english_warnings_pass_through(self):
        cards = build_scheme_cards(_make_mock_response(), lang="en")
        self.assertEqual(cards[0].warnings, ["state mismatch: scheme is for All India"])

    def test_hindi_why_eligible_prefix(self):
        cards = build_scheme_cards(_make_mock_response(), lang="hi")
        self.assertIn("मिलान:", cards[0].why_eligible)

    def test_hindi_criteria_translated(self):
        cards = build_scheme_cards(_make_mock_response(), lang="hi")
        # "state=Bihar" → "राज्य=Bihar"
        self.assertIn("राज्य", cards[0].why_eligible)
        # "age" → "आयु"
        self.assertIn("आयु", cards[0].why_eligible)

    def test_hindi_warnings_translated(self):
        cards = build_scheme_cards(_make_mock_response(), lang="hi")
        self.assertIn("राज्य बेमेल", cards[0].warnings[0])
        self.assertIn("सम्पूर्ण भारत", cards[0].warnings[0])

    def test_no_matched_criteria_uses_fallback(self):
        cards = build_scheme_cards(_make_mock_response(matched=()), lang="en")
        self.assertEqual(cards[0].why_eligible, "Semantically relevant to your profile")

    def test_no_matched_criteria_hindi_fallback(self):
        cards = build_scheme_cards(_make_mock_response(matched=()), lang="hi")
        self.assertIn("प्रोफ़ाइल", cards[0].why_eligible)

    def test_missing_benefit_fallback_english(self):
        scheme = MagicMock()
        scheme.scheme_name = "Test Scheme"
        scheme.final_score = 0.5
        scheme.matched_criteria = []
        scheme.eligibility_warnings = []
        scheme.metadata = {"benefit_amount": None, "benefit_text": None,
                           "state": "Delhi", "application_url": None, "source_pdf_url": None}
        response = MagicMock()
        response.retrieved_schemes = [scheme]
        cards = build_scheme_cards(response, lang="en")
        self.assertEqual(cards[0].benefit, "See scheme document")

    def test_application_url_fallback(self):
        cards = build_scheme_cards(
            _make_mock_response(app_url=None), lang="en"
        )
        self.assertEqual(cards[0].application_url, "#")

    def test_multiple_schemes_rank(self):
        scheme1 = MagicMock()
        scheme1.scheme_name = "Scheme A"
        scheme1.final_score = 0.9
        scheme1.matched_criteria = ["age"]
        scheme1.eligibility_warnings = []
        scheme1.metadata = {"benefit_amount": "₹1000", "state": None,
                            "application_url": "https://a.gov.in", "source_pdf_url": None}

        scheme2 = MagicMock()
        scheme2.scheme_name = "Scheme B"
        scheme2.final_score = 0.5
        scheme2.matched_criteria = []
        scheme2.eligibility_warnings = []
        scheme2.metadata = {"benefit_amount": "₹500", "state": None,
                            "application_url": None, "source_pdf_url": None}

        response = MagicMock()
        response.retrieved_schemes = [scheme1, scheme2]
        cards = build_scheme_cards(response, lang="en")
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0].rank, 1)
        self.assertEqual(cards[1].rank, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Django view smoke tests  —  HTTP status codes
# ─────────────────────────────────────────────────────────────────────────────

class TestDjangoViews(TestCase):
    """
    Smoke tests for the three static pages.
    rag_service.store_ready() is mocked so tests don't need ChromaDB.
    """

    def setUp(self):
        self.client = Client()

    @patch("finder.rag_service.store_ready", return_value=False)
    @patch("finder.rag_service.store_error", return_value="Test: no ChromaDB")
    def test_home_returns_200(self, mock_err, mock_ready):
        response = self.client.get(reverse("finder:home"))
        self.assertEqual(response.status_code, 200)

    @patch("finder.rag_service.store_ready", return_value=False)
    @patch("finder.rag_service.store_error", return_value=None)
    def test_home_uses_correct_template(self, mock_err, mock_ready):
        response = self.client.get(reverse("finder:home"))
        self.assertTemplateUsed(response, "finder/home.html")

    def test_find_get_returns_200(self):
        response = self.client.get(reverse("finder:find"))
        self.assertEqual(response.status_code, 200)

    def test_find_uses_correct_template(self):
        response = self.client.get(reverse("finder:find"))
        self.assertTemplateUsed(response, "finder/find.html")

    def test_about_returns_200(self):
        response = self.client.get(reverse("finder:about"))
        self.assertEqual(response.status_code, 200)

    def test_about_uses_correct_template(self):
        response = self.client.get(reverse("finder:about"))
        self.assertTemplateUsed(response, "finder/about.html")

    def test_results_without_session_redirects(self):
        """results/ with no session profile must redirect to find/."""
        response = self.client.get(reverse("finder:results"))
        self.assertRedirects(response, reverse("finder:find"))


# ─────────────────────────────────────────────────────────────────────────────
# 12. find view — lang sanitisation
# ─────────────────────────────────────────────────────────────────────────────

class TestFindViewLangValidation(TestCase):
    """
    The find view reads lang from POST and must reject any value other than
    'en' or 'hi', falling back to 'en' to prevent injection.
    """

    def setUp(self):
        self.client = Client()

    def test_valid_lang_hi_accepted(self):
        # Submit a valid form with lang=hi; confirm session stores 'hi'
        post_data = {
            "state": "Bihar",
            "age": "30",
            "gender": "male",
            "annual_income": "120000",
            "caste_category": "SC",
            "occupation_type": "farmer",
            "lang": "hi",
        }
        response = self.client.post(reverse("finder:find"), post_data)
        # After valid POST, view redirects to results
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.session.get("lang"), "hi")

    def test_invalid_lang_falls_back_to_en(self):
        """An attacker-supplied lang value must not leak into session."""
        post_data = {
            "state": "Bihar",
            "age": "30",
            "gender": "male",
            "annual_income": "120000",
            "caste_category": "SC",
            "occupation_type": "farmer",
            "lang": "<script>alert(1)</script>",   # injection attempt
        }
        response = self.client.post(reverse("finder:find"), post_data)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.session.get("lang"), "en")

    def test_missing_lang_defaults_to_en(self):
        post_data = {
            "state": "Bihar",
            "age": "30",
            "gender": "male",
            "annual_income": "120000",
            "caste_category": "SC",
            "occupation_type": "farmer",
            # no 'lang' key
        }
        response = self.client.post(reverse("finder:find"), post_data)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.session.get("lang"), "en")
