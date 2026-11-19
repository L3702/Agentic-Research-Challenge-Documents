"""Tests for E9's offline parser, schema checks, and diagnostics."""

from __future__ import annotations

import json
import unittest

from e9_core import DIMS, PARSER_RULES, error_status, language_check, parse_output, validate_object


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.score = {"sentiment_score": 0.4, "stance": 1}
        self.text = json.dumps(self.score)

    def test_plain_json_matches_all_rules(self):
        for rule in PARSER_RULES:
            with self.subTest(rule=rule):
                self.assertEqual(parse_output(self.text, rule), (self.score, None, False))

    def test_strict_rejects_fences_lenient_recovers(self):
        fenced = "```json\n" + self.text + "\n```"
        self.assertIsNone(parse_output(fenced, "strict")[0])
        self.assertEqual(parse_output(fenced, "e6_lenient"), (self.score, None, False))

    def test_lenient_commentary_recovery(self):
        text = "Result follows:\n" + self.text + "\nEnd."
        self.assertIsNone(parse_output(text, "strict")[0])
        self.assertEqual(parse_output(text, "e6_lenient"), (self.score, None, False))

    def test_deepseek_double_braces_recovered_only_by_explicit_rule(self):
        doubled = "{{\n\"sentiment_score\":0.40,\n\"stance\":1\n}}"
        for rule in ("strict", "e6_lenient"):
            self.assertIsNone(parse_output(doubled, rule)[0])
        self.assertEqual(parse_output(doubled, "brace_normalized"), (self.score, None, True))

    def test_normalization_preserves_nested_dimensions(self):
        score = dict(self.score, dimensions=dict.fromkeys(DIMS, 0.1))
        parsed, error, repaired = parse_output("{" + json.dumps(score) + "}", "brace_normalized")
        self.assertEqual(parsed, score)
        self.assertIsNone(error)
        self.assertTrue(repaired)

    def test_does_not_repair_commentary_or_partial_braces(self):
        candidates = ["comment {" + self.text + "}", "{" + self.text, self.text + "}"]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                parsed, _, repaired = parse_output(candidate, "brace_normalized")
                self.assertIsNone(parsed)
                self.assertFalse(repaired)

    def test_strict_rejects_duplicate_keys(self):
        text = '{"sentiment_score":0.4,"sentiment_score":0.9,"stance":1}'
        parsed, error, repaired = parse_output(text, "strict")
        self.assertIsNone(parsed)
        self.assertIn("duplicate", error)
        self.assertFalse(repaired)
        self.assertEqual(parse_output(text, "e6_lenient")[0]["sentiment_score"], 0.9)

    def test_strict_rejects_nonfinite_constants(self):
        for constant in ("NaN", "Infinity", "-Infinity", "1e400", "-1e400"):
            with self.subTest(constant=constant):
                text = '{"sentiment_score":' + constant + ',"stance":1}'
                self.assertIsNone(parse_output(text, "strict")[0])
                self.assertFalse(validate_object(parse_output(text, "e6_lenient")[0], False)["schema_valid"])

    def test_nonobject_json_and_unknown_rules(self):
        for text in ("[]", "null", "4", '"text"'):
            self.assertIsNone(parse_output(text, "strict")[0])
        with self.assertRaises(ValueError):
            parse_output(self.text, "unknown")


class SchemaTests(unittest.TestCase):
    def test_complete_dimensions(self):
        score = {"sentiment_score": -0.5, "stance": -1, "dimensions": dict.fromkeys(DIMS, 0)}
        result = validate_object(score, True)
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["score"], -0.5)
        self.assertEqual(result["stance"], -1.0)

    def test_bool_numeric_strings_and_nonfinite_are_rejected(self):
        for field in ("sentiment_score", "stance"):
            for invalid in (True, False, "0.5", float("nan"), float("inf"), None):
                with self.subTest(field=field, invalid=invalid):
                    score = {"sentiment_score": 0.5, "stance": 1}
                    score[field] = invalid
                    self.assertFalse(validate_object(score, False)["schema_valid"])

    def test_fractional_and_out_of_set_stance(self):
        for stance in (-3, -1.5, 0.4, 3):
            result = validate_object({"sentiment_score": 0.1, "stance": stance}, False)
            self.assertFalse(result["schema_valid"])
            self.assertTrue(result["stance_out_of_bounds"])

    def test_score_and_dimension_bounds(self):
        dimensions = dict.fromkeys(DIMS, 0.2)
        dimensions["risk_tone"] = -1.01
        result = validate_object({"sentiment_score": 1.01, "stance": 1, "dimensions": dimensions}, True)
        self.assertFalse(result["schema_valid"])
        self.assertTrue(result["score_out_of_bounds"])
        self.assertTrue(result["dimension_out_of_bounds"])

    def test_missing_dimensions_only_allowed_for_minimal_schema(self):
        score = {"sentiment_score": 0.1, "stance": 0}
        self.assertTrue(validate_object(score, False)["schema_valid"])
        self.assertFalse(validate_object(score, True)["schema_valid"])
        score["dimensions"] = dict.fromkeys(DIMS[:-1], 0)
        self.assertFalse(validate_object(score, True)["schema_valid"])

    def test_invalid_top_level_and_dimensions_types(self):
        for invalid in ([], None, "text", 2):
            self.assertFalse(validate_object(invalid, False)["schema_valid"])
        for invalid in ([], None, 0, "text"):
            self.assertFalse(validate_object({"sentiment_score": 0, "stance": 0, "dimensions": invalid}, False)["schema_valid"])


class DiagnosticTests(unittest.TestCase):
    def test_minimal_schema_language_is_not_applicable(self):
        result = language_check({"sentiment_score": 0.4, "stance": 1, "dimensions": dict.fromkeys(DIMS, 0)})
        self.assertFalse(result["language_applicable"])
        self.assertFalse(result["language_drift"])
        self.assertEqual(result["latin_chars"], 0)

    def test_english_narrative_drifts(self):
        result = language_check({"reasoning": "The report describes an expansionary monetary policy."})
        self.assertTrue(result["language_applicable"])
        self.assertTrue(result["language_drift"])

    def test_chinese_narrative_with_acronyms_does_not_drift(self):
        result = language_check({"reasoning": "GDP M2 CPI", "key_evidence": ["报告说明货币政策支持实体经济，流动性保持合理充裕。"]})
        self.assertTrue(result["language_applicable"])
        self.assertFalse(result["language_drift"])
        self.assertGreater(result["chinese_chars"], result["latin_chars"])

    def test_http_429_and_api_errors_do_not_become_json_errors(self):
        self.assertEqual(error_status({"parse_error": "call failed after 3 retries: HTTP 429"}), "http_429")
        self.assertEqual(error_status({"parse_error": "call failed after 3 retries: timeout"}), "api_error")
        self.assertEqual(error_status({"parse_error": "json decode error: unexpected end"}), "")
        self.assertEqual(error_status({"raw_output_text": "", "usage": {}}), "")


if __name__ == "__main__":
    unittest.main()
