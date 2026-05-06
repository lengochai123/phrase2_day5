"""Researcher agent -- fetches sources and writes research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Research Specialist. Your job is to distil raw search results into "
    "concise, structured research notes that an analyst can build on.\n\n"
    "Format your notes in Markdown with:\n"
    "- ## Research Notes\n"
    "- **Key Findings:** (bullet list)\n"
    "- **Conflicting claims:** (if any)\n"
    "- **Knowledge gaps:** (what is still unknown)\n"
    "- **Sources consulted:** (count and brief topic description)\n\n"
    "Be factual and concise. Do not hallucinate. Cite sources by number when possible."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and writes structured research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        self._llm = llm or LLMClient()
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate state.sources and state.research_notes."""
        with trace_span("researcher.run", {"query": state.request.query}) as span:
            try:
                sources = self._search.search(
                    query=state.request.query,
                    max_results=state.request.max_sources,
                )
                state.sources = sources
                logger.info("Researcher fetched %d sources.", len(sources))

                context = self._format_sources(sources)
                user_prompt = (
                    "Research query: " + state.request.query + "\n"
                    "Target audience: " + state.request.audience + "\n\n"
                    "Search results:\n" + context + "\n\n"
                    "Write concise research notes based on these results."
                )
                response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
                state.research_notes = response.content

                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.RESEARCHER,
                        content=response.content,
                        metadata={
                            "num_sources": len(sources),
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                state.add_trace_event("researcher.done", {"num_sources": len(sources)})
                span["num_sources"] = len(sources)

            except Exception as exc:
                logger.error("ResearcherAgent failed: %s", exc)
                state.errors.append("ResearcherAgent error: " + str(exc))
                raise AgentExecutionError("ResearcherAgent failed: " + str(exc)) from exc

        return state

    def _format_sources(self, sources: list) -> str:
        lines: list[str] = []
        for i, src in enumerate(sources, start=1):
            url_part = " (" + src.url + ")" if src.url else ""
            lines.append("[" + str(i) + "] " + src.title + url_part + "\n    " + src.snippet)
        return "\n\n".join(lines)
