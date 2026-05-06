"""Researcher agent - gathers sources and writes research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a research specialist. Given a query and search results,
create structured research notes covering:
1. Key facts and findings (with source citations [1], [2], etc.)
2. Important statistics and data points
3. Main approaches or methodologies mentioned
4. Consensus views and notable disagreements
5. Relevant limitations or caveats noted in sources

Be precise and cite sources. Format as clear bullet points under each section."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        sources = self._search.search(state.request.query, state.request.max_sources)
        state.sources.extend(sources)
        logger.info("Researcher gathered %d sources", len(sources))

        source_text = "\n\n".join(
            f"[{i + 1}] **{s.title}**\nURL: {s.url or 'N/A'}\n{s.snippet}"
            for i, s in enumerate(sources)
        )
        user_prompt = (
            f"Research query: {state.request.query}\n"
            f"Target audience: {state.request.audience}\n\n"
            f"Search results:\n{source_text}\n\n"
            "Create detailed research notes from these sources."
        )

        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "sources_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher",
            {
                "sources_gathered": len(sources),
                "notes_chars": len(response.content),
                "cost_usd": response.cost_usd,
            },
        )
        return state
