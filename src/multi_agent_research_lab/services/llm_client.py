"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

_COST_INPUT_PER_1K = 0.000150
_COST_OUTPUT_PER_1K = 0.000600


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client -- OpenAI by default, mock fallback when no key."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = self._build_client()

    def _build_client(self) -> object | None:
        if not self._settings.openai_api_key:
            logger.warning("OPENAI_API_KEY not set -- using mock LLM client.")
            return None
        try:
            from openai import OpenAI
            return OpenAI(api_key=self._settings.openai_api_key)
        except ImportError:
            logger.warning("openai package not available -- using mock LLM client.")
            return None

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000) * _COST_INPUT_PER_1K + (output_tokens / 1000) * _COST_OUTPUT_PER_1K

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry, timeout, and token logging."""
        if self._client is None:
            return self._mock_complete(system_prompt, user_prompt)
        try:
            response = self._client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=self._settings.timeout_seconds,
            )
            content = response.choices[0].message.content or ""
            in_tok = response.usage.prompt_tokens if response.usage else None
            out_tok = response.usage.completion_tokens if response.usage else None
            cost = self._estimate_cost(in_tok or 0, out_tok or 0)
            logger.info("LLM call: in=%s out=%s cost=$%.5f", in_tok, out_tok, cost)
            return LLMResponse(content=content, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        prompt_lower = (system_prompt + user_prompt).lower()
        content = self._generate_mock_content(prompt_lower, user_prompt)
        in_tok = (len(system_prompt) + len(user_prompt)) // 4
        out_tok = len(content) // 4
        time.sleep(0.05)
        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=self._estimate_cost(in_tok, out_tok),
        )

    def _generate_mock_content(self, prompt_lower: str, user_prompt: str) -> str:
        if "route" in prompt_lower or "next agent" in prompt_lower or "decide" in prompt_lower:
            if "research_notes" in prompt_lower and "analysis_notes" in prompt_lower:
                return "writer"
            if "research_notes" in prompt_lower:
                return "analyst"
            return "researcher"

        if "research" in prompt_lower and "notes" in prompt_lower:
            return (
                "## Research Notes\n\n"
                "**Key Findings:**\n"
                "- Multi-agent systems divide complex tasks among specialised agents, improving quality and parallelism.\n"
                "- GraphRAG combines knowledge graphs with retrieval-augmented generation for structured reasoning.\n"
                "- Recent benchmarks show multi-agent pipelines outperform single-agent by 15-30%% on complex tasks.\n\n"
                "**Sources consulted:** 3 documents covering architecture patterns, benchmark studies, and implementation guides.\n"
            )

        if "analys" in prompt_lower:
            return (
                "## Analysis\n\n"
                "**Strengths identified:**\n"
                "1. Clear separation of concerns (research -> analyse -> write) reduces error propagation.\n"
                "2. Shared state (ResearchState) enables full traceability of agent decisions.\n\n"
                "**Weak evidence / gaps:**\n"
                "- Benchmark comparisons rely on small sample sizes; results may not generalise.\n"
                "- Latency overhead of orchestration (~20%%) is a trade-off worth noting.\n\n"
                "**Verdict:** Evidence is moderately strong; quality gain justifies orchestration cost.\n"
            )

        if "write" in prompt_lower or "summary" in prompt_lower or "answer" in prompt_lower:
            topic = user_prompt[:120]
            return (
                "## Summary\n\n"
                "Based on the research and analysis conducted, here is a comprehensive answer.\n\n"
                "Multi-agent research systems represent a significant advancement in AI-powered knowledge work. "
                "By distributing responsibilities across specialised agents -- Researcher, Analyst, and Writer -- "
                "the system achieves higher accuracy, better source attribution, and more structured output than "
                "a single monolithic agent.\n\n"
                "**Key takeaways:**\n"
                "- The Supervisor routes tasks dynamically based on what information is still missing.\n"
                "- Research notes capture raw evidence; analysis notes synthesise insights.\n"
                "- The Writer produces a polished, cited answer from both.\n\n"
                "## References\n"
                "- Building Effective Agents (Anthropic)\n"
                "- GraphRAG: Structured Knowledge for LLM Reasoning\n"
                "- LangGraph: Stateful Multi-Actor Applications\n"
            )

        if "critic" in prompt_lower or "fact" in prompt_lower:
            return (
                "## Critic Review\n\n"
                "- No obvious hallucinations detected in the final answer.\n"
                "- All key claims can be traced back to research notes.\n"
                "- Citation coverage: 3/3 sources referenced.\n"
                "- **Quality score:** 8/10\n"
                "- **Verdict:** PASS -- answer is suitable for delivery.\n"
            )

        return (
            "I have processed the request and produced a thoughtful response. "
            "(Mock mode -- set OPENAI_API_KEY in .env for real completions.)"
        )
