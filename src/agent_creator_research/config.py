"""Typed definition of the creator-research agent.

The ``AgentConfig`` is the definition-time contract of this agent: mission,
model role, allowed/forbidden tools, output contracts, and the JSON schema
its LLM calls must satisfy. The router validates model output against this
schema (fail closed) before the agent ever sees it.
"""

from __future__ import annotations

from typing import Any

from agent_framework.config import AgentConfig
from linkgen_contracts.llm.models import LLMRole, ResponseFormat

CREATOR_RESEARCH_SLUG = "creator-research"

# JSON schema for the model's structured output. Restricted to the router's
# documented schema subset (type/properties/items) so it fails closed rather
# than being silently ignored. Ids and timestamps are NEVER requested from the
# model — deterministic Python derives them (see agent.py).
CREATOR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "creators": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "handle": {"type": "string"},
                    "platform": {"type": "string"},
                    "url": {"type": "string"},
                    "description": {"type": "string"},
                    "follower_count": {"type": "integer"},
                    "verified": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "content_style": {"type": "string"},
                    "posting_frequency": {"type": "string"},
                    "engagement_notes": {"type": "array", "items": {"type": "string"}},
                    "top_content": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "assertion": {"type": "string"},
                    "confidence": {"type": "number"},
                    "caveats": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

_SYSTEM_PROMPT = """You are the creator-research agent of the LinkGen content system.
Mission: find and analyze relevant individual creators for the requested focus.
Prioritize relevance, consistency, engagement quality, discussion, useful
content, authority, and recognizable voice rather than follower count alone.
Inspect actual creator posts rather than relying only on search snippets, and
analyze topics, audience problems, hooks, opening lines, structures,
storytelling, frameworks, contrarian opinions, mistakes, case studies,
personal experiences, lessons, observations, data, lists, short/long form,
comments/discussion, CTAs, writing style, and formatting.
Never copy creator wording or distinctive material; produce original analysis.
Respond only with JSON matching the requested schema."""


def build_creator_research_config() -> AgentConfig:
    """Build the definition-time configuration of the creator-research agent."""
    return AgentConfig(
        slug=CREATOR_RESEARCH_SLUG,
        model_role=LLMRole.REASONING,
        output_contracts=[
            "ResearchReport",
            "CreatorProfile",
            "CreatorContentAnalysis",
        ],
        system_prompt=_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=2048,
        max_steps=8,
        tools=["web-search", "browser"],
        forbidden_tools=["publish", "schedule", "linkedin-post"],
        response_format=ResponseFormat.JSON,
        json_schema=CREATOR_OUTPUT_SCHEMA,
        description="Find and analyze relevant creators (research phase).",
        version="1.0.0",
    )
