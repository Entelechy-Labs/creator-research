"""Unit tests for the pure sanitizing/normalizing helpers."""

from __future__ import annotations

import pytest

from agent_creator_research.agent import (
    DEFAULT_PLATFORM,
    MAX_CREATORS,
    bounded_str_list,
    clean_text,
    dedupe_candidates,
    normalize_candidates,
    normalize_key,
    safe_confidence,
    safe_int,
    safe_url,
    stable_probability,
)
from agent_framework.errors import AgentOutputError


def test_clean_text_collapses_whitespace_and_truncates() -> None:
    assert clean_text("  a \n\t b  ") == "a b"
    assert clean_text("x" * 999, max_len=10) == "x" * 10


def test_clean_text_rejects_non_strings() -> None:
    assert clean_text(None) == ""
    assert clean_text(123) == ""
    assert clean_text(["a"]) == ""


def test_normalize_key() -> None:
    assert normalize_key(" @Jane_Doe ") == "jane_doe"
    assert normalize_key(None) == ""


def test_bounded_str_list_caps_and_drops_empties() -> None:
    values = ["", "  ", "a", None, "b", "c"]
    assert bounded_str_list(values, limit=2) == ["a", "b"]
    assert bounded_str_list("not-a-list") == []
    assert bounded_str_list(None) == []


def test_safe_confidence_clamps_and_defaults() -> None:
    assert safe_confidence(1.5) == 1.0
    assert safe_confidence(-2) == 0.0
    assert safe_confidence(True) == 0.5  # bools are not numbers here
    assert safe_confidence("high") == 0.5
    assert safe_confidence(float("nan")) == 0.5
    assert safe_confidence(0.75) == 0.75


def test_safe_int() -> None:
    assert safe_int(10) == 10
    assert safe_int(0) is None
    assert safe_int(-3) is None
    assert safe_int(True) is None
    assert safe_int(1.5) is None
    assert safe_int("12") is None


def test_safe_url() -> None:
    assert safe_url("https://x.example/a") == "https://x.example/a"
    assert safe_url("http://x.example") == "http://x.example"
    assert safe_url("ftp://x.example") is None
    assert safe_url("mock_abc123") is None
    assert safe_url(None) is None


def test_stable_probability_is_bounded_and_stable() -> None:
    values = {stable_probability("p", i) for i in range(200)}
    assert all(0.30 <= v <= 0.90 for v in values)
    assert stable_probability("p", 1) == stable_probability("p", 1)


def test_normalize_candidates_filters_and_validates() -> None:
    payload = {"creators": [{"name": "a"}, "junk", 5, {"name": "b"}]}
    assert normalize_candidates(payload) == [{"name": "a"}, {"name": "b"}]
    assert normalize_candidates({}) == []
    assert normalize_candidates({"creators": None}) == []
    with pytest.raises(AgentOutputError):
        normalize_candidates({"creators": "nope"})


def test_dedupe_candidates_by_identity_and_platform() -> None:
    candidates = [
        {"handle": "@jane", "platform": "linkedin"},
        {"handle": "jane", "platform": "linkedin"},  # duplicate
        {"handle": "jane", "platform": "twitter"},  # different platform — kept
        {"name": "Bob", "platform": "linkedin"},  # name-only identity — kept
        {"platform": "linkedin"},  # no identity — dropped
        {"handle": "", "name": "  ", "platform": "x"},  # dropped
    ]
    out = dedupe_candidates(candidates)
    assert len(out) == 3
    # Raw entries are preserved (normalization happens in build_profiles);
    # identity is compared on the normalized (handle-or-name, platform) key.
    identities = [
        (normalize_key(entry.get("handle") or entry.get("name")),
         normalize_key(entry.get("platform")) or DEFAULT_PLATFORM)
        for entry in out
    ]
    assert identities == [
        ("jane", "linkedin"),
        ("jane", "twitter"),
        ("bob", "linkedin"),
    ]


def test_dedupe_candidates_caps_deterministically() -> None:
    candidates = [{"handle": f"user{i}"} for i in range(MAX_CREATORS + 10)]
    out = dedupe_candidates(candidates)
    assert len(out) == MAX_CREATORS
    assert out[0]["handle"] == "user0"
    assert out[-1]["handle"] == f"user{MAX_CREATORS - 1}"


def test_default_platform_constant() -> None:
    assert DEFAULT_PLATFORM == "linkedin"
