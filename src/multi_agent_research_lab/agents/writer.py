"""Writer agent - synthesizes research and analysis into a final answer."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a technical writer synthesizing research into a clear, authoritative answer.

Structure your response as:
## Introduction
Brief context and what this answer covers.

## Key Findings
3-5 bullet points with the most important information. Cite sources as [1], [2], etc.

## In-Depth Analysis
2-3 paragraphs covering nuances, comparisons, and evidence quality.

## Limitations & Caveats
What readers should be aware of: gaps, contradictions, or weak evidence.

## Conclusion
1-2 sentence summary of the bottom line.

Match tone and depth to the stated audience. Cite sources wherever possible."""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        source_refs = "\n".join(
            f"[{i + 1}] {s.title} — {s.url or 'N/A'}"
            for i, s in enumerate(state.sources)
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes or 'N/A'}\n\n"
            f"Analysis:\n{state.analysis_notes or 'N/A'}\n\n"
            f"Sources:\n{source_refs or 'No sources available.'}\n\n"
            "Write the final comprehensive answer."
        )

        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "writer",
            {
                "answer_chars": len(response.content),
                "cost_usd": response.cost_usd,
            },
        )
        logger.info("Writer produced final answer (%d chars)", len(response.content))
        return state
