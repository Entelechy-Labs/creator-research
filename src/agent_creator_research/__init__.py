"""creator-research — finds and analyzes relevant creators (Phase 08).

Research-phase agent of the LinkGen content-intelligence system. Consumes a
``ResearchRequest`` and produces contract-conforming ``ResearchReport``,
``CreatorProfile[]``, and ``CreatorContentAnalysis[]`` artifacts through the
shared agent framework and ModelRouter (mock mode by default).
"""

from __future__ import annotations

from agent_creator_research.config import (
    CREATOR_OUTPUT_SCHEMA,
    CREATOR_RESEARCH_SLUG,
    build_creator_research_config,
)
from agent_creator_research.agent import CreatorResearchAgent
from agent_creator_research.version import __version__

__all__ = [
    "__version__",
    "CREATOR_OUTPUT_SCHEMA",
    "CREATOR_RESEARCH_SLUG",
    "CreatorResearchAgent",
    "build_creator_research_config",
]
