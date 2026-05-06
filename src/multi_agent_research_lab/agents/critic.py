"""Critic agent -- fact-checks the final answer against research notes."""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Fact-Checker and Quality Reviewer. Given a final answer and the "
    "research notes it was based on, you must:\n\n"
    "1. Check whether each major claim in the answer is traceable to the research notes.\n"
    "2. Flag any statements that appear hallucinated (not grounded in notes).\n"
    "3. Check citation coverage -- are all sources referenced in the answer?\n"
    "4. Assign a quality score from 0-10 (10 = perfect, 0 = completely unreliable).\n\n"
    "Format your review as:\n"
    "## Critic Review\n"
    "- **Hallucination check:** PASS / FAIL (list issues if FAIL)\n"
    "- **Citation coverage:** X/Y sources referenced\n"
    "- **Quality score:** N/10\n"
    "- **Verdict:** PASS | NEEDS_REVISION\n"
    "- **Suggestions:** (optional -- only if NEEDS_REVISION)"
)


class CriticAgent(BaseAgent):
    """Optional fact-checking and quality-review agent."""

    name = "critic"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final_answer and append critic findings to agent_results."""
        if not state.final_answer:
            raise AgentExecutionError("CriticAgent requires final_answer to be populated.")
        if not state.research_notes:
            raise AgentExecutionError("CriticAgent requires research_notes.")

        with trace_span("critic.run") as span:
            try:
                user_prompt = (
                    "Query: " + state.request.query + "\n\n"
                    "## Research Notes\n" + state.research_notes + "\n\n"
                    "## Final Answer to review\n" + state.final_answer + "\n\n"
                    "Number of sources available: " + str(len(state.sources)) + "\n"
                    "Review the final answer now."
                )
                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                quality_score = self._parse_quality_score(response.content)

                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.CRITIC,
                        content=response.content,
                        metadata={
                            "quality_score": quality_score,
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event("critic.done", {"quality_score": quality_score})
                span["quality_score"] = quality_score
                logger.info("CriticAgent quality score: %s/10", quality_score)

            except AgentExecutionError:
                raise
            except Exception as exc:
                logger.error("CriticAgent failed: %s", exc)
                state.errors.append("CriticAgent error: " + str(exc))
                raise AgentExecutionError("CriticAgent failed: " + str(exc)) from exc

        return state

    def _parse_quality_score(self, content: str) -> float | None:
        matches = re.findall(r"quality score[:\s]+(\d+(?:\.\d+)?)\s*/\s*10", content, re.IGNORECASE)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        return None
