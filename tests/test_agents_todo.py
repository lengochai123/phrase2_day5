"""Tests for implemented agents -- replaces the original skeleton TODO test."""

import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_empty() -> None:
    """Supervisor should route to researcher when no notes exist yet."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "researcher"
    assert result.iteration == 1


def test_supervisor_routes_to_analyst_after_research() -> None:
    """Supervisor should route to analyst once research_notes is populated."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Some research notes."
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_after_analysis() -> None:
    """Supervisor should route to writer once both notes are populated."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "Some research notes."
    state.analysis_notes = "Some analysis notes."
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_routes_done_when_answer_exists() -> None:
    """Supervisor should stop once final_answer is set."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "The final answer."
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_enforces_max_iterations() -> None:
    """Supervisor must stop and record error when max_iterations is reached."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    # Simulate already at max
    for _ in range(6):
        state.record_route("researcher")
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"
    assert any("max_iterations" in e for e in result.errors)
