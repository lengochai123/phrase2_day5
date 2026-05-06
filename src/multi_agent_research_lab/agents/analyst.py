"""Analyst agent -- synthesises research notes into structured insights."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Research Analyst. You receive research notes and produce structured analysis.\n\n"
    "Your analysis MUST include:\n"
    "- ## Analysis\n"
    "- **Strengths identified:** (what the evidence supports well)\n"
    "- **Conflicting viewpoints:** (where sources disagree)\n"
    "- **Weak evidence / gaps:** (claims that lack strong backing)\n"
    "- **Verdict:** one sentence summarising the overall reliability of the evidence\n\n"
    "Be critical, concise, and grounded in the provided notes. "
    "Do not introduce facts not in the notes."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured analytical insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.analysis_notes from state.research_notes."""

        if not state.research_notes:
            raise AgentExecutionError("AnalystAgent requires research_notes to be populated first.")

        with trace_span("analyst.run") as span:
            try:
                user_prompt = (
                    "Query: " + state.request.query + "\n"
                    "Audience: " + state.request.audience + "\n\n"
                    "Research notes to analyse:\n" + state.research_notes
                )
                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                state.analysis_notes = response.content

                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.ANALYST,
                        content=response.content,
                        metadata={
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event("analyst.done", {"output_length": len(response.content)})
                span["output_length"] = len(response.content)
                logger.info("AnalystAgent wrote %d chars of analysis.", len(response.content))

            except AgentExecutionError:
                raise
            except Exception as exc:
                logger.error("AnalystAgent failed: %s", exc)
                state.errors.append("AnalystAgent error: " + str(exc))
                raise AgentExecutionError("AnalystAgent failed: " + str(exc)) from exc

        return state
