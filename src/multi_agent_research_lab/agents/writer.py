"""Writer agent -- synthesises research + analysis into a polished final answer."""

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
    "You are an expert technical writer. You receive research notes and an analyst's "
    "structured insights, then produce a clear, well-cited final answer.\n\n"
    "Requirements:\n"
    "- Write in clear prose (not bullet points unless naturally appropriate).\n"
    "- Aim for approximately 400-600 words unless the query asks otherwise.\n"
    "- Where evidence is strong, state it confidently; where weak, hedge appropriately.\n"
    "- End with a '## References' section listing all source titles/URLs.\n"
    "- Do NOT introduce facts not present in the notes or analysis."
)


class WriterAgent(BaseAgent):
    """Produces a polished final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.final_answer from research_notes + analysis_notes."""
        if not state.research_notes:
            raise AgentExecutionError("WriterAgent requires research_notes.")
        if not state.analysis_notes:
            raise AgentExecutionError("WriterAgent requires analysis_notes.")

        with trace_span("writer.run") as span:
            try:
                source_lines = "\n".join(
                    "- [" + s.title + "](" + s.url + ")" if s.url else "- " + s.title
                    for s in state.sources
                )
                user_prompt = (
                    "Query: " + state.request.query + "\n"
                    "Audience: " + state.request.audience + "\n\n"
                    "## Research Notes\n" + state.research_notes + "\n\n"
                    "## Analyst Notes\n" + state.analysis_notes + "\n\n"
                    "## Available Sources\n" + source_lines + "\n\n"
                    "Write the final answer now."
                )
                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                state.final_answer = response.content

                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.WRITER,
                        content=response.content,
                        metadata={
                            "word_count": len(response.content.split()),
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event("writer.done", {"word_count": len(response.content.split())})
                span["word_count"] = len(response.content.split())
                logger.info("WriterAgent wrote %d words.", len(response.content.split()))

            except AgentExecutionError:
                raise
            except Exception as exc:
                logger.error("WriterAgent failed: %s", exc)
                state.errors.append("WriterAgent error: " + str(exc))
                raise AgentExecutionError("WriterAgent failed: " + str(exc)) from exc

        return state
