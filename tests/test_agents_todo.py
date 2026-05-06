"""Tests for agent implementations (replaces skeleton TODO tests)."""

from unittest.mock import MagicMock, patch

import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


def _make_state(**kwargs: object) -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"), **kwargs)


def _mock_llm_response(content: str) -> MagicMock:
    resp = MagicMock(spec=LLMResponse)
    resp.content = content
    resp.input_tokens = 10
    resp.output_tokens = 20
    resp.cost_usd = 0.0001
    return resp


@patch("multi_agent_research_lab.agents.supervisor.LLMClient")
def test_supervisor_routes_to_researcher_when_no_notes(mock_llm_cls: MagicMock) -> None:
    mock_llm = mock_llm_cls.return_value
    mock_llm.complete.return_value = _mock_llm_response('{"route": "researcher", "reason": "no notes yet"}')

    state = _make_state()
    result = SupervisorAgent().run(state)

    assert result.route_history[-1] == "researcher"
    assert result.iteration == 1


@patch("multi_agent_research_lab.agents.supervisor.LLMClient")
def test_supervisor_routes_to_done_when_answer_ready(mock_llm_cls: MagicMock) -> None:
    mock_llm = mock_llm_cls.return_value
    mock_llm.complete.return_value = _mock_llm_response('{"route": "done", "reason": "answer complete"}')

    state = _make_state()
    state.research_notes = "some notes"
    state.analysis_notes = "some analysis"
    state.final_answer = "final answer here"
    result = SupervisorAgent().run(state)

    assert result.route_history[-1] == "done"


@patch("multi_agent_research_lab.agents.supervisor.LLMClient")
def test_supervisor_fallback_on_invalid_json(mock_llm_cls: MagicMock) -> None:
    mock_llm = mock_llm_cls.return_value
    mock_llm.complete.return_value = _mock_llm_response("not valid json at all")

    state = _make_state()
    result = SupervisorAgent().run(state)

    assert result.route_history[-1] in {"researcher", "analyst", "writer", "done"}


@patch("multi_agent_research_lab.agents.supervisor.LLMClient")
def test_supervisor_enforces_max_iterations(mock_llm_cls: MagicMock) -> None:
    from multi_agent_research_lab.core.config import get_settings

    state = _make_state()
    state.iteration = get_settings().max_iterations

    result = SupervisorAgent().run(state)

    assert result.route_history[-1] in {"writer", "done"}
