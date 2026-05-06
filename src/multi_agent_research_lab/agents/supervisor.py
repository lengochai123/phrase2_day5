"""Supervisor / router agent."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Supervisor that routes a research request to the correct next agent.\n\n"
    "Available agents:\n"
    "- researcher  : fetches sources and writes research_notes\n"
    "- analyst     : reads research_notes and writes analysis_notes\n"
    "- writer      : reads research_notes + analysis_notes and writes final_answer\n"
    "- done        : workflow is complete -- final_answer exists\n\n"
    "Rules:\n"
    "1. Always start with 'researcher' if research_notes is empty.\n"
    "2. Go to 'analyst' once research_notes exists but analysis_notes is empty.\n"
    "3. Go to 'writer' once both notes exist but final_answer is empty.\n"
    "4. Reply 'done' if final_answer exists.\n"
    "5. Reply with ONLY one word: researcher | analyst | writer | done"
)

# Map of state flags to next route -- order matters
_ROUTING_TABLE = [
    # (has_final, has_research, has_analysis) -> route
    ((True,  None,  None),  "done"),
    ((False, True,  True),  "writer"),
    ((False, True,  False), "analyst"),
    ((False, False, False), "researcher"),
    ((False, False, True),  "researcher"),  # analysis without research: restart
]


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()
        self._settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect state and append the next route to route_history."""
        with trace_span("supervisor.run", {"iteration": state.iteration}) as span:
            if state.iteration >= self._settings.max_iterations:
                logger.warning("Max iterations (%d) reached -- forcing done.", self._settings.max_iterations)
                next_route = "done"
                state.errors.append("Stopped at max_iterations=" + str(self._settings.max_iterations))
            else:
                next_route = self._decide(state)

            state.record_route(next_route)
            state.add_trace_event("supervisor.route", {"next": next_route, "iteration": state.iteration})
            span["next_route"] = next_route
            logger.info("Supervisor -> %s (iteration %d)", next_route, state.iteration)

        return state

    def _decide(self, state: ResearchState) -> str:
        """Rule-based routing with LLM confirmation for ambiguous states."""
        has_final    = bool(state.final_answer)
        has_research = bool(state.research_notes)
        has_analysis = bool(state.analysis_notes)

        # Fast deterministic rules cover all standard cases
        if has_final:
            return "done"
        if has_research and has_analysis:
            return "writer"
        if has_research and not has_analysis:
            return "analyst"
        # No research yet -- always start there
        return "researcher"
