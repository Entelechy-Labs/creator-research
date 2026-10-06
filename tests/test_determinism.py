"""Determinism and id-stability regression tests (golden path).

The same input, clock, and router must produce identical artifacts; different
run ids must produce different derived ids (no cross-run bleed).
"""

from __future__ import annotations

from agent_creator_research.agent import CreatorResearchAgent
from agent_framework.ids import deterministic_id
from linkgen_contracts.common.ids import CreatorId, RunId, TaskId

from conftest import RUN_ID, TASK_ID, make_context, make_request


def _run_agent(request: dict, *, run_id: str = RUN_ID) -> dict:
    agent = CreatorResearchAgent(context=make_context())
    return agent.process(
        run_id=RunId(run_id), task_id=TaskId(TASK_ID), input_data={"request": request}
    )


def test_same_input_same_output() -> None:
    first = _run_agent(make_request())
    second = _run_agent(make_request())
    assert first == second


def test_creator_ids_are_derived_deterministically() -> None:
    output = _run_agent(make_request())
    for profile in output["creators"]:
        expected = deterministic_id(
            CreatorId, RUN_ID, profile["handle"], profile["platform"]
        )
        assert profile["creator_id"] == expected


def test_different_run_id_yields_different_ids() -> None:
    other_run = "run_" + "e4" * 16
    base = _run_agent(make_request())
    other = _run_agent(make_request(run_id=other_run), run_id=other_run)
    base_ids = {p["creator_id"] for p in base["creators"]}
    other_ids = {p["creator_id"] for p in other["creators"]}
    if base_ids and other_ids:
        assert base_ids != other_ids


def test_report_id_depends_only_on_run_and_request() -> None:
    # report_id is derived from (run_id, request_id) — not from the focus text
    # — so a re-run under the same run/request is idempotent at the artifact
    # level, while the summary reflects the actual focus.
    first = _run_agent(make_request())
    second = _run_agent(make_request(focus="indie hackers sharing revenue"))
    assert first["report"]["report_id"] == second["report"]["report_id"]
    assert first["report"]["summary"] != second["report"]["summary"]
