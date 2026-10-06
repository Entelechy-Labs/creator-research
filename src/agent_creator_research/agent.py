"""Creator-research agent — Phase 08 research system.

Finds and analyzes relevant individual creators for a research request and
produces contract-conforming artifacts:

- ``ResearchReport`` (findings + sources)
- ``CreatorProfile[]`` (one per distinct creator)
- ``CreatorContentAnalysis[]`` (one per creator)

Design rules:
- The LLM is used only for content judgment; ids, timestamps, dedupe, capping,
  and confidence bounds are deterministic Python.
- Model output is untrusted: it arrives router-validated as JSON, and every
  field is sanitized before entering a contract model.
- Ids are derived deterministically (uuid5) so identical inputs produce
  identical artifacts (idempotent replay, golden tests, checkpoint recovery).
- No network, clock, or randomness: time comes from the injected context
  clock; failures come from the router as typed, duck-typed-normalized errors.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

from linkgen_contracts.common.ids import (
    CreatorId,
    FindingId,
    ResultId,
    RunId,
    SourceId,
    TaskId,
)
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.research.models import (
    CreatorContentAnalysis,
    CreatorProfile,
    ResearchFinding,
    ResearchReport,
    ResearchRequest,
    ResearchSource,
    SourceKind,
)
from pydantic import ValidationError

from agent_creator_research.config import (
    CREATOR_OUTPUT_SCHEMA,
    build_creator_research_config,
)
from agent_framework.base import AgentBase
from agent_framework.config import AgentConfig
from agent_framework.context import AgentContext
from agent_framework.errors import (
    AgentError,
    AgentInputError,
    AgentOutputError,
)
from agent_framework.ids import deterministic_id
from agent_framework.structured import extract_json_dict

# Bounded resource usage (AGENTS rule 11): the source spec targets ~10
# creators; the caps below are generous hard limits, never unbounded growth.
MAX_CREATORS = 25
MAX_FINDINGS = 50
MAX_TAGS = 10
MAX_NOTES = 10
MAX_TEXT = 500
MAX_SNIPPET = 200
DEFAULT_PLATFORM = "linkedin"

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Pure sanitizing helpers (unit-testable without a router).
# ---------------------------------------------------------------------------


def clean_text(value: Any, *, max_len: int = MAX_TEXT) -> str:
    """Collapse whitespace and strip; '' for anything that is not a string."""
    if not isinstance(value, str):
        return ""
    return _WS.sub(" ", value).strip()[:max_len]


def normalize_key(value: Any) -> str:
    """Normalize an identity key: trimmed, lowercased, '@' stripped."""
    return clean_text(value).lstrip("@").lower()


def bounded_str_list(value: Any, *, limit: int = MAX_NOTES, max_len: int = MAX_TEXT) -> list[str]:
    """Sanitize an untrusted list into at most ``limit`` non-empty strings."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = clean_text(item, max_len=max_len)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def safe_confidence(value: Any, default: float = 0.5) -> float:
    """Clamp an untrusted confidence to [0, 1]; non-numbers get the default."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if number != number:  # NaN
        return default
    return min(1.0, max(0.0, number))


def safe_int(value: Any) -> int | None:
    """Return a positive int or None (bools, floats, zero, negatives rejected)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def safe_url(value: Any) -> str | None:
    """Return the value only when it is an http(s) URL string, else None."""
    text = clean_text(value)
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return None


def stable_probability(*parts: Any) -> float:
    """Deterministic pseudo-probability in [0.30, 0.90] derived from parts."""
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()
    return 0.30 + (int(digest[:8], 16) % 61) / 100.0


def normalize_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter the untrusted ``creators`` array down to dict entries."""
    raw = payload.get("creators")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AgentOutputError("model output field 'creators' must be an array")
    return [entry for entry in raw if isinstance(entry, dict)]


def dedupe_candidates(
    candidates: list[dict[str, Any]], *, cap: int = MAX_CREATORS
) -> list[dict[str, Any]]:
    """Dedupe by (identity, platform), drop unidentifiable entries, cap.

    Identity is the handle (or name when the handle is missing). Duplicates
    keep the first occurrence; the cap keeps the first ``cap`` entries, so the
    result is a deterministic function of the input order.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for entry in candidates:
        handle = normalize_key(entry.get("handle"))
        name = clean_text(entry.get("name"))
        platform = normalize_key(entry.get("platform")) or DEFAULT_PLATFORM
        identity = handle or normalize_key(name)
        if not identity:
            continue  # cannot identify the creator — skip deterministically
        key = (identity, platform)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
        if len(out) >= cap:
            break
    return out


def validate_request_input(input_data: Any) -> ResearchRequest:
    """Validate the agent input against the ``ResearchRequest`` contract."""
    if not isinstance(input_data, dict):
        raise AgentInputError(
            "creator-research input must be a dict with a 'request' key"
        )
    raw_request = input_data.get("request")
    if raw_request is None:
        raise AgentInputError("creator-research input is missing the 'request' object")
    try:
        return ResearchRequest.model_validate(raw_request)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        raise AgentInputError(
            f"research request failed contract validation at '{location}': "
            f"{first.get('msg', 'invalid')}"
        ) from exc


class _RouterFailure(AgentError):
    """Agent-level wrapper for a router failure (category/code preserved)."""


def normalize_router_failure(exc: Exception) -> AgentError:
    """Normalize a router failure without importing provider internals.

    Router errors carry ``category``/``code``/``retryable``; they are preserved
    by duck-typing so ``TaskResult`` mapping keeps retry semantics honest.
    """
    failure = _RouterFailure(f"router call failed: {getattr(exc, 'message', None) or exc}")
    raw_category = getattr(exc, "category", ErrorCategory.INTERNAL)
    failure.category = raw_category if isinstance(raw_category, ErrorCategory) else ErrorCategory.INTERNAL
    failure.code = str(getattr(exc, "code", "router_failure"))
    failure.retryable = bool(getattr(exc, "retryable", False))
    return failure


def build_creator_research_prompt(request: ResearchRequest) -> str:
    """Deterministic user prompt for the creator-research LLM call."""
    lines = [
        "Research creators for a LinkedIn content system.",
        f"Focus: {request.focus}",
    ]
    if request.objectives:
        lines.append("Objectives: " + "; ".join(request.objectives))
    if request.creators:
        lines.append("Named creators to inspect: " + ", ".join(request.creators))
    if request.domains:
        lines.append("Domains: " + ", ".join(request.domains))
    lines.append(
        "Prioritize relevance, consistency, engagement quality, discussion, "
        "useful content, authority, and recognizable voice over follower "
        "count. Inspect actual creator posts, not only search snippets. "
        "Do not copy creator wording or distinctive material."
    )
    lines.append("Return JSON with 'creators' and 'findings' arrays.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Artifact builders (pure functions of their inputs).
# ---------------------------------------------------------------------------


def build_profiles(
    *,
    run_id: RunId,
    candidates: list[dict[str, Any]],
    captured_at: datetime,
) -> tuple[list[CreatorProfile], list[ResearchSource]]:
    """Build one ``CreatorProfile`` + one platform-profile source per candidate.

    ``candidates`` must already be deduped (``dedupe_candidates``); profiles
    and sources are index-aligned with it.
    """
    profiles: list[CreatorProfile] = []
    sources: list[ResearchSource] = []
    for entry in candidates:
        handle = normalize_key(entry.get("handle")) or normalize_key(entry.get("name"))
        name = clean_text(entry.get("name")) or handle
        platform = normalize_key(entry.get("platform")) or DEFAULT_PLATFORM
        description = clean_text(entry.get("description")) or None
        url = safe_url(entry.get("url"))
        profile = CreatorProfile(
            creator_id=deterministic_id(CreatorId, str(run_id), handle, platform),
            name=name,
            handle=handle,
            platform=platform,
            url=url,
            description=description,
            follower_count=safe_int(entry.get("follower_count")),
            verified=bool(entry.get("verified", False)),
            captured_at=captured_at,
            tags=bounded_str_list(entry.get("tags"), limit=MAX_TAGS),
        )
        source = ResearchSource(
            source_id=deterministic_id(SourceId, str(run_id), "creator", handle, platform),
            kind=SourceKind.PLATFORM_PROFILE,
            title=f"{name} ({handle}) on {platform}",
            url=url,
            domain=platform,
            retrieved_at=captured_at,
            snippet=description[:MAX_SNIPPET] if description else None,
            confidence=stable_probability("source", str(run_id), handle, platform),
        )
        profiles.append(profile)
        sources.append(source)
    return profiles, sources


def build_analyses(
    *,
    run_id: RunId,
    candidates: list[dict[str, Any]],
    profiles: list[CreatorProfile],
    generated_at: datetime,
) -> list[CreatorContentAnalysis]:
    """Build one ``CreatorContentAnalysis`` per creator (index-aligned)."""
    analyses: list[CreatorContentAnalysis] = []
    for entry, profile in zip(candidates, profiles):
        analyses.append(
            CreatorContentAnalysis(
                analysis_id=deterministic_id(
                    ResultId, str(run_id), "analysis", str(profile.creator_id)
                ),
                creator_id=profile.creator_id,
                run_id=run_id,
                generated_at=generated_at,
                content_style=clean_text(entry.get("content_style")) or "unspecified",
                posting_frequency=clean_text(entry.get("posting_frequency")) or "unspecified",
                engagement_notes=bounded_str_list(entry.get("engagement_notes")),
                top_content=bounded_str_list(entry.get("top_content")),
            )
        )
    return analyses


def build_findings(
    *,
    run_id: RunId,
    request: ResearchRequest,
    payload: dict[str, Any],
    sources: list[ResearchSource],
    profiles: list[CreatorProfile],
) -> list[ResearchFinding]:
    """Map untrusted model findings into contract-valid findings.

    A deterministic coverage finding is always inserted first when creators
    were identified, so the research gate always has evidence to inspect.
    """
    known_source_ids = {str(source.source_id) for source in sources}
    findings: list[ResearchFinding] = []
    seen: set[tuple[str, str]] = set()

    raw_findings = payload.get("findings")
    for entry in raw_findings if isinstance(raw_findings, list) else []:
        if not isinstance(entry, dict) or len(findings) >= MAX_FINDINGS:
            continue
        title = clean_text(entry.get("title"))
        assertion = clean_text(entry.get("assertion"))
        if not title or not assertion:
            continue
        key = (title, assertion)
        if key in seen:
            continue
        seen.add(key)
        # Only cite sources this run actually captured; model-invented ids are
        # untrusted and dropped.
        supporting = [
            SourceId(sid)
            for sid in bounded_str_list(entry.get("supporting_source_ids"))
            if sid in known_source_ids
        ]
        findings.append(
            ResearchFinding(
                finding_id=deterministic_id(FindingId, str(run_id), title, assertion),
                title=title,
                assertion=assertion,
                supporting_source_ids=supporting,
                confidence=safe_confidence(entry.get("confidence")),
                caveats=bounded_str_list(entry.get("caveats")),
            )
        )

    if profiles:
        coverage_title = "Creator coverage"
        coverage_assertion = (
            f"Identified {len(profiles)} distinct creators for focus '{request.focus}'."
        )
        if (coverage_title, coverage_assertion) not in seen:
            findings.insert(
                0,
                ResearchFinding(
                    finding_id=deterministic_id(
                        FindingId, str(run_id), coverage_title, coverage_assertion
                    ),
                    title=coverage_title,
                    assertion=coverage_assertion,
                    supporting_source_ids=[source.source_id for source in sources],
                    confidence=stable_probability(
                        "coverage", str(run_id), str(request.request_id)
                    ),
                    caveats=[] if sources else ["no creator sources captured"],
                ),
            )
    return findings


def build_report(
    *,
    run_id: RunId,
    request: ResearchRequest,
    profiles: list[CreatorProfile],
    findings: list[ResearchFinding],
    sources: list[ResearchSource],
    generated_at: datetime,
) -> ResearchReport:
    """Build the run's ``ResearchReport``."""
    summary = (
        f"Creator research for '{request.focus}': "
        f"{len(profiles)} creators, {len(findings)} findings, "
        f"{len(sources)} sources."
    )
    return ResearchReport(
        report_id=deterministic_id(
            ResultId, str(run_id), "research-report", str(request.request_id)
        ),
        run_id=run_id,
        generated_at=generated_at,
        summary=summary,
        findings=findings,
        sources=sources,
        confidence=stable_probability("report", str(run_id), str(request.request_id)),
    )


# ---------------------------------------------------------------------------
# The agent.
# ---------------------------------------------------------------------------


class CreatorResearchAgent(AgentBase):
    """Find and analyze relevant creators (research phase, Phase 08).

    Input (validated against the ``ResearchRequest`` contract):
        ``{"request": {request_id, run_id, focus, objectives?, creators?, domains?}}``

    Output:
        ``{"report": ResearchReport, "creators": [CreatorProfile],
          "analyses": [CreatorContentAnalysis]}`` — all JSON-serialized and
        contract-validated by construction.
    """

    def __init__(
        self, context: AgentContext, config: AgentConfig | None = None
    ) -> None:
        super().__init__(
            config if config is not None else build_creator_research_config(), context
        )

    def process(
        self,
        *,
        run_id: RunId,
        task_id: TaskId,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        request = validate_request_input(input_data)
        payload = self._gather(run_id=run_id, task_id=task_id, request=request)

        candidates = dedupe_candidates(normalize_candidates(payload))
        now = self.context.now()
        profiles, sources = build_profiles(
            run_id=run_id, candidates=candidates, captured_at=now
        )
        analyses = build_analyses(
            run_id=run_id, candidates=candidates, profiles=profiles, generated_at=now
        )
        findings = build_findings(
            run_id=run_id,
            request=request,
            payload=payload,
            sources=sources,
            profiles=profiles,
        )
        report = build_report(
            run_id=run_id,
            request=request,
            profiles=profiles,
            findings=findings,
            sources=sources,
            generated_at=now,
        )
        return {
            "report": report.model_dump(mode="json"),
            "creators": [profile.model_dump(mode="json") for profile in profiles],
            "analyses": [analysis.model_dump(mode="json") for analysis in analyses],
        }

    def _gather(
        self, *, run_id: RunId, task_id: TaskId, request: ResearchRequest
    ) -> dict[str, Any]:
        """Call the router for structured creator candidates (single LLM hop).

        Fault markers in the prompt (mock mode) or a failing adapter surface
        as normalized agent errors; they never escape as raw provider errors.
        """
        prompt = build_creator_research_prompt(request)
        llm_request = self.build_llm_request(
            run_id=run_id,
            task_id=task_id,
            prompt=prompt,
            json_schema=CREATOR_OUTPUT_SCHEMA,
        )
        try:
            result = self.context.router.complete_json(llm_request, CREATOR_OUTPUT_SCHEMA)
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalized below, never swallowed
            raise normalize_router_failure(exc) from exc
        return extract_json_dict(result)
