"""Agent-level tests: normal cases, edge cases, and malformed inputs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_creator_research.agent import (
    CreatorResearchAgent,
    build_creator_research_prompt,
    build_findings,
    build_profiles,
    build_report,
    validate_request_input,
)
from agent_creator_research.config import CREATOR_RESEARCH_SLUG
from agent_framework.errors import AgentInputError
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.research.models import (
    CreatorContentAnalysis,
    CreatorProfile,
    ResearchReport,
    ResearchRequest,
)
from linkgen_contracts.workflow.models import TaskStatus

from conftest import RUN_ID, TASK_ID, make_request

FIXED_CAPTURE = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)


class TestInputValidation:
    def test_valid_request_round_trips(self) -> None:
        request = validate_request_input({"request": make_request()})
        assert isinstance(request, ResearchRequest)
        assert request.focus.startswith("B2B SaaS")

    def test_non_dict_input_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_request_input("not a dict")

    def test_missing_request_key_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_request_input({"other": {}})

    def test_empty_focus_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_request_input({"request": make_request(focus="   ")})

    def test_wrong_types_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_request_input({"request": make_request(focus=123)})
        with pytest.raises(AgentInputError):
            validate_request_input({"request": make_request(request_id="bogus")})


class TestNormalRun:
    def test_process_produces_contract_valid_output(
        self, agent: CreatorResearchAgent, run_id, task_id
    ) -> None:
        output = agent.process(
            run_id=run_id, task_id=task_id, input_data={"request": make_request()}
        )
        assert set(output) == {"report", "creators", "analyses"}
        report = ResearchReport.model_validate(output["report"])
        profiles = [CreatorProfile.model_validate(item) for item in output["creators"]]
        analyses = [
            CreatorContentAnalysis.model_validate(item) for item in output["analyses"]
        ]
        assert report.run_id == run_id
        assert len(profiles) == len(analyses)
        assert report.summary
        # Every profile has an analysis and a captured platform-profile source.
        source_ids = {str(s.source_id) for s in report.sources}
        for profile in profiles:
            assert str(profile.creator_id).startswith("creator_")
            assert profile.handle
            assert profile.platform
            assert any(a.creator_id == profile.creator_id for a in analyses)

    def test_counts_match_summary(
        self, agent: CreatorResearchAgent, run_id, task_id
    ) -> None:
        output = agent.process(
            run_id=run_id, task_id=task_id, input_data={"request": make_request()}
        )
        report = ResearchReport.model_validate(output["report"])
        assert f"{len(output['creators'])} creators" in report.summary
        assert f"{len(output['analyses'])}" in report.summary

    def test_run_reports_success(
        self, agent: CreatorResearchAgent, run_id, task_id
    ) -> None:
        result = agent.run(
            run_id=run_id, task_id=task_id, input_data={"request": make_request()}
        )
        assert result.status is TaskStatus.SUCCEEDED
        assert result.errors == []

    def test_minimal_request_accepted(
        self, agent: CreatorResearchAgent, run_id, task_id
    ) -> None:
        minimal = make_request()
        minimal.pop("objectives")
        minimal.pop("creators")
        minimal.pop("domains")
        output = agent.process(
            run_id=run_id, task_id=task_id, input_data={"request": minimal}
        )
        assert ResearchReport.model_validate(output["report"]).summary

    def test_config_declares_expected_boundaries(
        self, agent: CreatorResearchAgent
    ) -> None:
        assert agent.config.slug == CREATOR_RESEARCH_SLUG
        assert "publish" in agent.config.forbidden_tools
        assert agent.config.response_format.value == "json"


class TestRunFailures:
    def test_malformed_input_maps_to_failed_result(
        self, agent: CreatorResearchAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=task_id, input_data="junk")
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.VALIDATION

    def test_missing_request_maps_to_failed_result(
        self, agent: CreatorResearchAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=task_id, input_data={})
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].code == "agent_input_invalid"


class TestArtifactBuilders:
    CANDIDATES = [
        {
            "name": "Jane Doe",
            "handle": "@jane",
            "platform": "linkedin",
            "url": "https://linkedin.com/in/jane",
            "description": "Writes about PLG",
            "follower_count": 12000,
            "verified": True,
            "tags": ["plg", "saas"],
            "content_style": "case-study led",
            "posting_frequency": "3x/week",
            "engagement_notes": ["high comment quality"],
            "top_content": ["PLG teardown series"],
        },
        {
            "name": "John Roe",
            "handle": "john",
            "platform": "twitter",
            "description": "Pricing threads",
        },
    ]

    def test_build_profiles_alignment_and_ids(self, run_id) -> None:
        profiles, sources = build_profiles(
            run_id=run_id, candidates=self.CANDIDATES, captured_at=FIXED_CAPTURE
        )
        assert len(profiles) == len(sources) == 2
        assert profiles[0].follower_count == 12000
        assert profiles[0].verified is True
        assert profiles[1].follower_count is None  # absent → None, not fabricated
        assert sources[0].kind.value == "platform_profile"
        assert sources[0].domain == "linkedin"
        assert sources[0].snippet == "Writes about PLG"

    def test_findings_drop_model_invented_source_ids(self, run_id) -> None:
        request = ResearchRequest.model_validate(make_request())
        profiles, sources = build_profiles(
            run_id=run_id,
            candidates=self.CANDIDATES,
            captured_at=FIXED_CAPTURE,
        )
        payload = {
            "findings": [
                {
                    "title": "Hooks win",
                    "assertion": "Question hooks outperform statements",
                    "confidence": 0.8,
                    "supporting_source_ids": ["source_deadbeefdeadbeefdeadbeefdeadbeef"],
                },
                {"title": "", "assertion": "dropped — empty title"},
                {"title": "t", "assertion": ""},  # dropped — empty assertion
            ]
        }
        findings = build_findings(
            run_id=run_id,
            request=request,
            payload=payload,
            sources=sources,
            profiles=profiles,
        )
        assert [f.title for f in findings] == ["Creator coverage", "Hooks win"]
        assert findings[0].supporting_source_ids == [
            s.source_id for s in sources
        ]
        assert findings[1].supporting_source_ids == []  # invented id dropped
        assert findings[1].confidence == 0.8

    def test_report_summary_counts(self, run_id) -> None:
        request = ResearchRequest.model_validate(make_request())
        profiles, sources = build_profiles(
            run_id=run_id,
            candidates=self.CANDIDATES,
            captured_at=FIXED_CAPTURE,
        )
        report = build_report(
            run_id=run_id,
            request=request,
            profiles=profiles,
            findings=[],
            sources=sources,
            generated_at=FIXED_CAPTURE,
        )
        assert "2 creators" in report.summary
        assert report.confidence == report.confidence  # finite float
        assert 0.0 <= report.confidence <= 1.0

    def test_prompt_contains_focus_and_ground_rules(self) -> None:
        request = ResearchRequest.model_validate(make_request())
        prompt = build_creator_research_prompt(request)
        assert request.focus in prompt
        assert "Do not copy creator wording" in prompt
