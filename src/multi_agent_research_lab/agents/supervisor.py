"""Supervisor / router agent - decides which worker runs next."""

import json
import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_VALID_ROUTES = {"researcher", "analyst", "writer", "done"}

_SYSTEM_PROMPT = """You are a research supervisor that routes tasks between specialized agents.
Given the current state of a research task, decide which agent should run next.

Available agents:
- researcher: Gathers sources and creates research notes (run when no research_notes)
- analyst: Analyzes research notes and extracts insights (run when no analysis_notes)
- writer: Synthesizes everything into a final answer (run when no final_answer)
- done: The task is complete (final_answer exists)

Routing rules:
1. Always route to researcher first if research_notes is missing
2. Route to analyst after research_notes exist but analysis_notes is missing
3. Route to writer after analysis_notes exist but final_answer is missing
4. Route to done when final_answer exists and is high quality
5. If max iterations is nearly reached, skip to writer or done immediately

Respond with ONLY a JSON object (no markdown, no explanation):
{"route": "<agent_name>", "reason": "<one line reason>"}"""


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        if state.iteration >= self._settings.max_iterations:
            route = "writer" if state.final_answer is None else "done"
            reason = "max iterations reached — forcing completion"
            state.record_route(route)
            state.add_trace_event("supervisor", {"route": route, "reason": reason})
            logger.warning("Max iterations reached; routing to %s", route)
            return state

        user_prompt = (
            f"Research query: {state.request.query}\n"
            f"Has research_notes: {state.research_notes is not None}\n"
            f"Has analysis_notes: {state.analysis_notes is not None}\n"
            f"Has final_answer: {state.final_answer is not None}\n"
            f"Iteration: {state.iteration}\n"
            f"Route history: {state.route_history}\n\n"
            "Which agent should run next?"
        )

        try:
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            data = json.loads(response.content.strip())
            route = str(data.get("route", "")).strip()
            reason = str(data.get("reason", ""))
        except Exception as exc:
            logger.warning("Supervisor LLM routing failed (%s); using fallback", exc)
            route = ""
            reason = "fallback routing"

        if route not in _VALID_ROUTES:
            route = self._fallback_route(state)
            reason = f"invalid route '{route}'; fallback applied"

        state.record_route(route)
        state.add_trace_event("supervisor", {"route": route, "reason": reason})
        logger.info("Supervisor → %s (%s)", route, reason)
        return state

    def _fallback_route(self, state: ResearchState) -> str:
        if state.research_notes is None:
            return "researcher"
        if state.analysis_notes is None:
            return "analyst"
        if state.final_answer is None:
            return "writer"
        return "done"
