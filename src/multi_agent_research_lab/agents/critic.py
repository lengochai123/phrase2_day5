"""Critic agent - fact-checks the final answer and scores quality."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a rigorous fact-checker and quality reviewer.

Given a final answer and its supporting sources, evaluate and respond in this exact format:

QUALITY_SCORE: <integer 0-10>
CITATION_COVERAGE: <fraction e.g. 4/6 claims have citations>
HALLUCINATION_RISK: <low|medium|high>
ISSUES:
- <issue 1>
- <issue 2>
SUGGESTIONS:
- <suggestion 1>
- <suggestion 2>

Scoring guide:
- 9-10: Accurate, well-cited, clear, comprehensive
- 7-8: Minor gaps or weak citations
- 5-6: Some unsupported claims or unclear sections
- 3-4: Significant inaccuracies or missing citations
- 0-2: Mostly hallucinated or off-topic"""


class CriticAgent(BaseAgent):
    """Optional fact-checking and quality-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and record quality findings."""
        source_snippets = "\n".join(
            f"[{i + 1}] {s.snippet[:300]}"
            for i, s in enumerate(state.sources)
        )
        user_prompt = (
            f"Final answer to review:\n{state.final_answer or 'No final answer yet.'}\n\n"
            f"Supporting source snippets:\n{source_snippets or 'No sources.'}\n\n"
            "Provide your quality review."
        )

        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        quality_score = self._extract_score(response.content)

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
        state.add_trace_event(
            "critic",
            {
                "quality_score": quality_score,
                "cost_usd": response.cost_usd,
            },
        )
        logger.info("Critic quality score: %s", quality_score)
        return state

    def _extract_score(self, content: str) -> float | None:
        for line in content.splitlines():
            if line.startswith("QUALITY_SCORE:"):
                try:
                    return float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return None
