"""Unit tests for creator-research normalization (agent-specific logic).

Generic sanitizer behavior is covered by agent-framework's test_sanitize.py;
here we test how this agent applies it to creator candidates.
"""

from __future__ import annotations

import pytest

from agent_creator_research.agent import (
    DEFAULT_PLATFORM,
    MAX_CREATORS,
    dedupe_candidates,
    normalize_candidates,
)
from agent_creator_research.agent import (
    normalize_key,  # re-exported from agent_framework.sanitize
)
from agent_framework.errors import AgentOutputError


def test_normalize_key() -> None:
    assert normalize_key(" @Jane_Doe ") == "jane_doe"


def test_default_platform_constant() -> None:
    assert DEFAULT_PLATFORM == "linkedin"


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
