"""Tests for pattern registry."""
import pytest
from prompt_shield.core.patterns import PATTERNS, COMPILED_PATTERNS, CATEGORIES, score_to_severity


def test_all_patterns_have_required_fields():
    for p in PATTERNS:
        assert "name" in p
        assert "pattern" in p
        assert "weight" in p
        assert "category" in p
        assert 1 <= p["weight"] <= 10, f"Weight out of range for {p['name']}"


def test_compiled_patterns_have_regex():
    for p in COMPILED_PATTERNS:
        assert "_regex" in p
        assert hasattr(p["_regex"], "search")


def test_pattern_names_are_unique():
    names = [p["name"] for p in PATTERNS]
    assert len(names) == len(set(names)), "Duplicate pattern names found"


def test_score_to_severity_boundaries():
    assert score_to_severity(0) == "SAFE"
    assert score_to_severity(1) == "LOW"
    assert score_to_severity(3) == "LOW"
    assert score_to_severity(4) == "MEDIUM"
    assert score_to_severity(6) == "MEDIUM"
    assert score_to_severity(7) == "HIGH"
    assert score_to_severity(9) == "HIGH"
    assert score_to_severity(10) == "CRITICAL"
    assert score_to_severity(999) == "CRITICAL"


def test_all_categories_present():
    categories = {p["category"] for p in PATTERNS}
    expected = {
        "role_override", "jailbreak", "exfiltration", "manipulation", "encoding",
        "multilingual", "tool_use", "pii", "claude_code",
    }
    assert expected.issubset(categories)


def test_categories_constant_matches_actual():
    actual = {p["category"] for p in PATTERNS}
    assert actual == CATEGORIES


def test_pattern_count_minimum():
    """We target 50+ patterns. Fail if we drop below."""
    assert len(PATTERNS) >= 50, f"Only {len(PATTERNS)} patterns — target is 50+"


def test_original_22_patterns_unchanged():
    """First 22 patterns must match v0.1.0 names exactly."""
    original_names = [
        "ignore_instructions", "disregard_training", "you_are_now",
        "new_instructions", "override_system",
        "dan_jailbreak", "act_as", "pretend_you_are", "no_restrictions",
        "true_self", "developer_mode",
        "print_system_prompt", "repeat_everything_above",
        "what_were_your_instructions", "summarize_above",
        "trusted_source_claim", "developer_wants", "for_research",
        "token_smuggling",
        "base64_injection", "unicode_smuggling", "rot13_reference",
    ]
    for i, name in enumerate(original_names):
        assert PATTERNS[i]["name"] == name, (
            f"Pattern {i} changed: expected {name}, got {PATTERNS[i]['name']}"
        )
