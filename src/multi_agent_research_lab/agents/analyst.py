"""Analyst agent - turns research notes into structured insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an analytical specialist. Given research notes on a topic, produce:

1. **Key Claims** — the 3-5 most important findings (label each: strong/moderate/weak evidence)
2. **Viewpoint Comparison** — where sources agree vs. disagree
3. **Evidence Quality** — assess overall strength: what's well-supported vs. speculative
4. **Knowledge Gaps** — what is unknown or insufficiently studied
5. **Implications** — what these findings mean for the target audience

Be critical and evidence-based. Flag weak or uncited claims explicitly."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        user_prompt = (
            f"Original query: {state.request.query}\n"
            f"Target audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes or 'No research notes available.'}\n\n"
            "Produce structured analysis."
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
        state.add_trace_event(
            "analyst",
            {
                "analysis_chars": len(response.content),
                "cost_usd": response.cost_usd,
            },
        )
        logger.info("Analyst produced %d chars of analysis", len(response.content))
        return state
