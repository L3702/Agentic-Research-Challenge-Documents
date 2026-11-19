"""Offline parsing and validation for E9 cached model responses."""

from __future__ import annotations

import json
import math
import re
from numbers import Real
from typing import Any


PARSER_RULES = ("strict", "e6_lenient", "brace_normalized")
DIMS = (
    "policy_stance",
    "economic_judgment",
    "risk_tone",
    "liquidity_credit",
    "policy_outlook",
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def _strict_loads(candidate: str) -> Any:
    return json.loads(
        candidate,
        object_pairs_hook=_unique_pairs,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )


def _strip_fences(raw_text: str) -> str:
    candidate = raw_text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
    return re.sub(r"\s*```$", "", candidate).strip()


def parse_output(
    raw_text: str, rule: str
) -> tuple[dict[str, Any] | None, str | None, bool]:
    if rule not in PARSER_RULES:
        raise ValueError(f"unknown parser rule: {rule}")
    if not isinstance(raw_text, str):
        return None, "response text is not a string", False

    if rule == "strict":
        try:
            result = _strict_loads(raw_text.strip())
        except (json.JSONDecodeError, ValueError) as error:
            return None, str(error), False
        if not isinstance(result, dict):
            return None, "parsed JSON is not an object", False
        return result, None, False

    candidate = _strip_fences(raw_text)
    last_error = "parsed JSON is not an object"
    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result, None, False
        return None, last_error, False
    except json.JSONDecodeError as error:
        last_error = str(error)

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result, None, False
        except json.JSONDecodeError:
            pass

    if rule == "brace_normalized" and candidate.startswith("{{") and candidate.endswith("}}"):
        normalized = candidate[1:-1]
        try:
            result = _strict_loads(normalized)
        except (json.JSONDecodeError, ValueError):
            pass
        else:
            if isinstance(result, dict):
                return result, None, True
    return None, f"json decode error: {last_error}", False


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_object(obj: Any, require_dimensions: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_valid": False,
        "schema_error": "",
        "score": None,
        "stance": None,
        "score_out_of_bounds": False,
        "stance_out_of_bounds": False,
        "dimension_out_of_bounds": False,
    }
    if not isinstance(obj, dict):
        result["schema_error"] = "JSON object required"
        return result

    errors: list[str] = []
    score = _finite_number(obj.get("sentiment_score"))
    stance = _finite_number(obj.get("stance"))
    result["score"] = score
    result["stance"] = stance
    if score is None:
        errors.append("sentiment_score must be a finite JSON number")
    elif not -1.0 <= score <= 1.0:
        result["score_out_of_bounds"] = True
        errors.append("sentiment_score outside [-1, 1]")
    if stance is None:
        errors.append("stance must be a finite JSON number")
    elif stance not in (-2.0, -1.0, 0.0, 1.0, 2.0):
        result["stance_out_of_bounds"] = True
        errors.append("stance must be one of -2, -1, 0, 1, 2")

    dimensions = obj.get("dimensions")
    if require_dimensions or "dimensions" in obj:
        if not isinstance(dimensions, dict):
            errors.append("dimensions must be a JSON object")
        else:
            for dimension in DIMS:
                if dimension not in dimensions and not require_dimensions:
                    continue
                dimension_value = _finite_number(dimensions.get(dimension))
                if dimension_value is None:
                    errors.append(f"dimensions.{dimension} must be a finite JSON number")
                elif not -1.0 <= dimension_value <= 1.0:
                    result["dimension_out_of_bounds"] = True
                    errors.append(f"dimensions.{dimension} outside [-1, 1]")

    result["schema_valid"] = not errors
    result["schema_error"] = "; ".join(errors)
    return result


def language_check(obj: Any) -> dict[str, Any]:
    text_fields: list[str] = []
    if isinstance(obj, dict):
        reasoning = obj.get("reasoning")
        evidence = obj.get("key_evidence")
        if isinstance(reasoning, str):
            text_fields.append(reasoning)
        if isinstance(evidence, str):
            text_fields.append(evidence)
        elif isinstance(evidence, list):
            text_fields.extend(value for value in evidence if isinstance(value, str))
    text = " ".join(text_fields)
    chinese_chars = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    english_words = re.findall(r"\b[A-Za-z]{2,}\b", text)
    applicable = chinese_chars + latin_chars > 0
    return {
        "language_applicable": applicable,
        "language_drift": applicable and latin_chars > chinese_chars and len(english_words) >= 3,
        "chinese_chars": chinese_chars,
        "latin_chars": latin_chars,
    }


def error_status(record: dict[str, Any]) -> str:
    error = str(record.get("parse_error") or "")
    if "429" in error:
        return "http_429"
    if "call failed" in error.lower():
        return "api_error"
    return ""
